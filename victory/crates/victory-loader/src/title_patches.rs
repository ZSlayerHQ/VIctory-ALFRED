//! Load-time code patches read from a private folder outside the repository.
//!
//! One TOML file per patch. A file names the exact build it was made for and
//! each site carries the BLAKE3 of the bytes it replaces, so a patch can
//! never silently hit another build. Every site of a file is checked before
//! any of them is written.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// One patch file as written on disk.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchFile {
    name: String,
    build: String,
    #[serde(default = "enabled_by_default")]
    enabled: bool,
    #[serde(default, rename = "why")]
    _why: String,
    #[serde(default)]
    site: Vec<SiteFile>,
}

fn enabled_by_default() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteFile {
    rva: u32,
    expect: String,
    bytes: String,
}

/// A validated patch file with its hashes and bytes decoded.
#[derive(Debug)]
struct Patch {
    name: String,
    build: blake3::Hash,
    enabled: bool,
    sites: Vec<Site>,
}

#[derive(Debug)]
struct Site {
    rva: u32,
    expect: blake3::Hash,
    bytes: Vec<u8>,
}

/// What [`apply`] did, for tests and the one summary log line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Names of the patch files that were applied, in file-name order.
    pub applied: Vec<String>,
    /// Files skipped: disabled, another build, or a failed site check.
    pub skipped: usize,
}

/// Applies every `*.toml` in `dir` to `image`.
///
/// `None` (the feature is not configured) does nothing and logs nothing. A
/// configured but missing folder is an empty patch set. Files are read in
/// file-name order; a malformed file rejects the folder with an error, while
/// a build mismatch or a failed site check skips just that file.
pub fn apply(dir: Option<&Path>, build: &blake3::Hash, image: &mut impl Image) -> Result<Summary> {
    let Some(dir) = dir else {
        return Ok(Summary::default());
    };
    if !dir.is_dir() {
        let summary = Summary::default();
        log_summary(&summary);
        return Ok(summary);
    }

    let patches = read_folder(dir)?;
    let mut summary = Summary::default();
    for patch in &patches {
        if !patch.enabled {
            summary.skipped += 1;
            continue;
        }
        if patch.build != *build {
            tracing::info!(patch = %patch.name, "title patch skipped: different build");
            summary.skipped += 1;
            continue;
        }
        match check(patch, image) {
            Ok(()) => {
                for site in &patch.sites {
                    image
                        .write(site.rva as usize, &site.bytes)
                        .with_context(|| format!("writing {} site {:#x}", patch.name, site.rva))?;
                }
                summary.applied.push(patch.name.clone());
            }
            Err(SiteFailure::OutOfRange { rva }) => {
                tracing::warn!(
                    patch = %patch.name,
                    rva = format_args!("{rva:#x}"),
                    "title patch skipped: site outside the image"
                );
                summary.skipped += 1;
            }
            Err(SiteFailure::Mismatch { rva, found }) => {
                tracing::warn!(
                    patch = %patch.name,
                    rva = format_args!("{rva:#x}"),
                    found = %found.to_hex(),
                    "title patch skipped: expected bytes not found"
                );
                summary.skipped += 1;
            }
        }
    }
    log_summary(&summary);
    Ok(summary)
}

fn log_summary(summary: &Summary) {
    tracing::info!(
        "title patches: {} applied ({}), {} skipped",
        summary.applied.len(),
        summary.applied.join(", "),
        summary.skipped
    );
}

/// Every patch in `dir`, sorted by file name.
fn read_folder(dir: &Path) -> Result<Vec<Patch>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    paths.iter().map(|path| read_file(path)).collect()
}

fn read_file(path: &Path) -> Result<Patch> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text).with_context(|| format!("parsing {}", path.display()))
}

fn parse(text: &str) -> Result<Patch> {
    // Where it is malformed, and nothing of what is there: toml's own message
    // quotes the line, which may be a site's replacement bytes (the
    // merged-work review, item 3).
    let file: PatchFile = toml::from_str(text).map_err(|e| match e.span() {
        Some(span) => {
            let (line, column) = line_and_column(text, span.start);
            anyhow::anyhow!("malformed patch file at line {line}, column {column}")
        }
        None => anyhow::anyhow!("malformed patch file"),
    })?;
    let build = blake3::Hash::from_hex(&file.build).context("bad build hash")?;
    let mut sites = Vec::with_capacity(file.site.len());
    for site in file.site {
        let bytes = decode_hex(&site.bytes)
            .with_context(|| format!("site {:#x}: bad hex bytes", site.rva))?;
        ensure!(!bytes.is_empty(), "site {:#x}: empty bytes", site.rva);
        let expect = blake3::Hash::from_hex(&site.expect)
            .with_context(|| format!("site {:#x}: bad expect hash", site.rva))?;
        sites.push(Site {
            rva: site.rva,
            expect,
            bytes,
        });
    }
    reject_overlaps(&sites)?;
    Ok(Patch {
        name: file.name,
        build,
        enabled: file.enabled,
        sites,
    })
}

/// The 1-based line and column (in characters) of byte `at` of `text`.
fn line_and_column(text: &str, at: usize) -> (usize, usize) {
    let before = &text[..text.floor_char_boundary(at.min(text.len()))];
    let line = before.matches('\n').count() + 1;
    let column = before.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
    (line, column)
}

fn decode_hex(text: &str) -> Result<Vec<u8>> {
    let text = text.trim();
    let text = text.strip_prefix("0x").unwrap_or(text);
    ensure!(text.len().is_multiple_of(2), "odd number of hex digits");
    let mut out = Vec::with_capacity(text.len() / 2);
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_digit(pair[0]).with_context(|| format!("bad hex digit at byte {i}"))?;
        let lo = hex_digit(pair[1]).with_context(|| format!("bad hex digit at byte {i}"))?;
        out.push(hi << 4 | lo);
    }
    Ok(out)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn reject_overlaps(sites: &[Site]) -> Result<()> {
    let mut ranges = Vec::with_capacity(sites.len());
    for site in sites {
        let start = u64::from(site.rva);
        let end = start
            .checked_add(site.bytes.len() as u64)
            .context("site range overflows")?;
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            bail!("sites at {:#x} and {:#x} overlap", pair[0].0, pair[1].0);
        }
    }
    Ok(())
}

enum SiteFailure {
    OutOfRange { rva: u32 },
    Mismatch { rva: u32, found: blake3::Hash },
}

/// Checks every site of one file without writing anything.
fn check(patch: &Patch, image: &impl Image) -> Result<(), SiteFailure> {
    for site in &patch.sites {
        let Some(current) = image.read(site.rva as usize, site.bytes.len()) else {
            return Err(SiteFailure::OutOfRange { rva: site.rva });
        };
        let found = blake3::hash(current);
        if found != site.expect {
            return Err(SiteFailure::Mismatch {
                rva: site.rva,
                found,
            });
        }
    }
    Ok(())
}

/// The image bytes patches read and write. In the loader this is the mapped
/// image; tests use a `Vec<u8>`.
pub trait Image {
    /// `len` bytes at `offset`, or `None` if the range is outside the image.
    fn read(&self, offset: usize, len: usize) -> Option<&[u8]>;

    /// Writes `bytes` at `offset`, making that range writable first.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()>;
}

impl Image for Vec<u8> {
    fn read(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        self.get(offset..end)
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .context("site range overflows")?;
        let target = self
            .get_mut(offset..end)
            .context("site outside the image")?;
        target.copy_from_slice(bytes);
        Ok(())
    }
}

/// A mapped PE image. Writes change the page protection, restore it and
/// flush the instruction cache.
#[cfg(windows)]
#[derive(Debug)]
pub struct MappedImage {
    base: usize,
    size: usize,
}

#[cfg(windows)]
impl MappedImage {
    /// # Safety
    /// `base` must point at `size` readable bytes of a mapped image, and no
    /// code in it may run until [`apply`] returns.
    pub unsafe fn new(base: usize, size: usize) -> Self {
        Self { base, size }
    }
}

#[cfg(windows)]
impl Image for MappedImage {
    fn read(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.size {
            return None;
        }
        // SAFETY: `new` guarantees the whole image is readable, and `end` is
        // inside it.
        Some(unsafe { std::slice::from_raw_parts((self.base + offset) as *const u8, len) })
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        use std::ffi::c_void;

        use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
        use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let end = offset
            .checked_add(bytes.len())
            .context("site range overflows")?;
        ensure!(end <= self.size, "site {offset:#x} outside the image");
        let address = (self.base + offset) as *const c_void;
        let mut old = 0;
        // SAFETY: the range is inside the mapped image and no title code runs.
        if unsafe { VirtualProtect(address, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            bail!(
                "VirtualProtect({offset:#x}) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: the range is now writable and was checked above.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
        // SAFETY: restoring the protection we changed, then flushing the
        // instruction cache for the bytes just written.
        unsafe {
            VirtualProtect(address, bytes.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), address, bytes.len());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn file_text(name: &str, build: &blake3::Hash, enabled: bool, sites: &str) -> String {
        format!(
            "name = \"{name}\"\nbuild = \"{}\"\nenabled = {enabled}\nwhy = \"test\"\n\n{sites}",
            build.to_hex()
        )
    }

    fn site_text(rva: u32, expect: &blake3::Hash, bytes: &[u8]) -> String {
        format!(
            "[[site]]\nrva = {rva}\nexpect = \"{}\"\nbytes = \"{}\"\n",
            expect.to_hex(),
            hex(bytes)
        )
    }

    fn write_file(dir: &Path, name: &str, text: &str) {
        std::fs::write(dir.join(name), text).unwrap();
    }

    #[test]
    fn one_site_applies_and_changes_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x11u8; 64];
        let original = image.clone();
        let replacement = [0x90u8, 0x90, 0xc3];
        let expect = blake3::hash(&image[0x20..0x23]);
        let sites = site_text(0x20, &expect, &replacement);
        write_file(
            dir.path(),
            "a.toml",
            &file_text("one", &build, true, &sites),
        );

        let summary = apply(Some(dir.path()), &build, &mut image).unwrap();
        assert_eq!(summary.applied, vec!["one".to_owned()]);
        assert_eq!(summary.skipped, 0);
        assert_eq!(&image[0x20..0x23], &replacement);
        assert_eq!(&image[..0x20], &original[..0x20]);
        assert_eq!(&image[0x23..], &original[0x23..]);
    }

    #[test]
    fn one_failed_site_applies_nothing_in_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x22u8; 64];
        let original = image.clone();
        let first = [0xaa_u8, 0xbb];
        let first_expect = blake3::hash(&image[0x10..0x12]);
        let second = [0xcc_u8; 3];
        let second_expect = blake3::hash(b"not the current bytes");
        let sites = format!(
            "{}{}",
            site_text(0x10, &first_expect, &first),
            site_text(0x30, &second_expect, &second)
        );
        write_file(
            dir.path(),
            "a.toml",
            &file_text("two-sites", &build, true, &sites),
        );

        let summary = apply(Some(dir.path()), &build, &mut image).unwrap();
        assert!(summary.applied.is_empty());
        assert_eq!(summary.skipped, 1);
        assert_eq!(image, original, "nothing may be written when a site fails");
    }

    #[test]
    fn another_build_applies_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let patch_build = blake3::hash(b"build the patch was made for");
        let loaded_build = blake3::hash(b"another build");
        let mut image = vec![0x33u8; 32];
        let original = image.clone();
        let expect = blake3::hash(&image[0x08..0x09]);
        let sites = site_text(0x08, &expect, &[0xff]);
        write_file(
            dir.path(),
            "a.toml",
            &file_text("other-build", &patch_build, true, &sites),
        );

        let summary = apply(Some(dir.path()), &loaded_build, &mut image).unwrap();
        assert!(summary.applied.is_empty());
        assert_eq!(summary.skipped, 1);
        assert_eq!(image, original);
    }

    #[test]
    fn disabled_file_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x44u8; 32];
        let original = image.clone();
        let expect = blake3::hash(&image[0x04..0x05]);
        let sites = site_text(0x04, &expect, &[0x00]);
        write_file(
            dir.path(),
            "a.toml",
            &file_text("disabled", &build, false, &sites),
        );

        let summary = apply(Some(dir.path()), &build, &mut image).unwrap();
        assert!(summary.applied.is_empty());
        assert_eq!(summary.skipped, 1);
        assert_eq!(image, original);
    }

    #[test]
    fn bad_hex_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x55u8; 16];
        let original = image.clone();
        let expect = blake3::hash(&image[0x00..0x01]);
        let sites = format!(
            "[[site]]\nrva = 0\nexpect = \"{}\"\nbytes = \"zz\"\n",
            expect.to_hex()
        );
        write_file(
            dir.path(),
            "a.toml",
            &file_text("bad-hex", &build, true, &sites),
        );

        assert!(apply(Some(dir.path()), &build, &mut image).is_err());
        assert_eq!(image, original);
    }

    /// A malformed patch file is named with the line and column where it
    /// breaks, and nothing of its text: a broken `bytes` line's content
    /// never reaches the error (or the log it goes to).
    #[test]
    fn a_malformed_bytes_line_is_not_quoted() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x77u8; 16];
        let secret = "c0ffee5ec2e7b17e5ca1ab1e";
        let expect = blake3::hash(&image[0x00..0x01]);
        // The string is never closed.
        let sites = format!("[[site]]\nrva = 0\nexpect = \"{}\"\nbytes = \"{secret}\n", expect.to_hex());
        write_file(dir.path(), "a.toml", &file_text("broken", &build, true, &sites));
        let err = apply(Some(dir.path()), &build, &mut image).expect_err("a malformed file is refused");
        let text = format!("{err:#} {err:?}");
        assert!(!text.contains(secret), "the error quotes the bytes line: {text}");
        assert!(text.contains("a.toml") && text.contains("line ") && text.contains("column "), "{text}");
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x66u8; 16];
        let text = format!(
            "name = \"unknown\"\nbuild = \"{}\"\nsurprise = 1\n",
            build.to_hex()
        );
        write_file(dir.path(), "a.toml", &text);

        assert!(apply(Some(dir.path()), &build, &mut image).is_err());
    }

    #[test]
    fn overlapping_sites_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x77u8; 32];
        let expect1 = blake3::hash(&image[0x10..0x14]);
        let expect2 = blake3::hash(&image[0x12..0x16]);
        let sites = format!(
            "{}{}",
            site_text(0x10, &expect1, &[1, 2, 3, 4]),
            site_text(0x12, &expect2, &[5, 6, 7, 8])
        );
        write_file(
            dir.path(),
            "a.toml",
            &file_text("overlap", &build, true, &sites),
        );

        assert!(apply(Some(dir.path()), &build, &mut image).is_err());
    }

    #[test]
    fn out_of_range_site_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x88u8; 16];
        let original = image.clone();
        let expect = blake3::hash(&image[0x0e..0x10]);
        let sites = site_text(0x0e, &expect, &[1, 2, 3, 4]);
        write_file(
            dir.path(),
            "a.toml",
            &file_text("past-the-end", &build, true, &sites),
        );

        let summary = apply(Some(dir.path()), &build, &mut image).unwrap();
        assert!(summary.applied.is_empty());
        assert_eq!(summary.skipped, 1);
        assert_eq!(image, original);
    }

    #[test]
    fn no_folder_is_a_no_op() {
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0x99u8; 8];
        let original = image.clone();

        let summary = apply(None, &build, &mut image).unwrap();
        assert_eq!(summary, Summary::default());
        assert_eq!(image, original);

        let missing = std::path::Path::new("this-folder-does-not-exist");
        let summary = apply(Some(missing), &build, &mut image).unwrap();
        assert_eq!(summary, Summary::default());
        assert_eq!(image, original);
    }

    #[test]
    fn files_are_applied_in_file_name_order() {
        let dir = tempfile::tempdir().unwrap();
        let build = blake3::hash(b"made-up build one");
        let mut image = vec![0u8; 32];
        let expect_a = blake3::hash(&image[0x00..0x01]);
        let expect_b = blake3::hash(&image[0x10..0x11]);
        write_file(
            dir.path(),
            "b.toml",
            &file_text("bee", &build, true, &site_text(0x10, &expect_b, &[0xb0])),
        );
        write_file(
            dir.path(),
            "a.toml",
            &file_text("ay", &build, true, &site_text(0x00, &expect_a, &[0xa0])),
        );

        let summary = apply(Some(dir.path()), &build, &mut image).unwrap();
        assert_eq!(summary.applied, vec!["ay".to_owned(), "bee".to_owned()]);
    }
}
