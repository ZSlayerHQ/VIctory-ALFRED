//! Diagnostics for the BCPack unpack paths: what `VICTORY_BCPACK_CHECK`
//! compares on every request, and what its replay test reads back.
//!
//! Three results are compared: the serial decode, [`super::unpack`] with the
//! lanes it chooses (a round at once where the processor allows), and
//! [`super::unpack`] forced to one lane at a time. A difference is reported
//! by its first byte, its byte count and the block that byte falls in.

use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::{Component, Path};

/// Whether a `VICTORY_BCPACK_*` switch value means "on": exactly `1`.
pub fn enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Whether requests may be kept in `dir`: an absolute folder outside every
/// repository (no ancestor holds a `.git` entry), with no `..` in it. The
/// kept streams are the title's data, so they must never sit in a checkout.
/// The walk is made twice: on the path as written, and on its deepest
/// existing ancestor resolved, so a symlink or junction into a checkout is
/// inside it. A path whose existing part cannot be resolved is refused.
pub fn keep_allowed(dir: &Path) -> bool {
    let in_repository = |path: &Path| path.ancestors().any(|ancestor| ancestor.join(".git").exists());
    if !dir.is_absolute() || dir.components().any(|c| matches!(c, Component::ParentDir)) || in_repository(dir) {
        return false;
    }
    let Some(existing) = dir.ancestors().find(|ancestor| ancestor.exists()) else {
        return false;
    };
    std::fs::canonicalize(existing).is_ok_and(|resolved| !in_repository(&resolved))
}

const LANES_1: [u8; 12] = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7];
const LANES_3: [u8; 12] = [3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7];
const LANES_4: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const LANES_5: [u8; 12] = [3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];
const LANES_6: [u8; 16] = [3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4];
const LANES_7: [u8; 16] = [3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4];

/// A format's block bytes, lane widths and symbols per block, as the
/// unpacker lays a stream out.
pub fn layout(bc: u8) -> Option<(usize, &'static [u8], usize)> {
    Some(match bc {
        1 => (8, &LANES_1, 1),
        3 => (16, &LANES_3, 2),
        4 => (8, &LANES_4, 1),
        5 => (16, &LANES_5, 2),
        6 => (16, &LANES_6, 2),
        7 => (16, &LANES_7, 2),
        _ => return None,
    })
}

/// The first place one path's bytes differ from the serial decode's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diff {
    /// The first byte that differs.
    pub first: usize,
    /// How many bytes differ.
    pub count: usize,
    /// The block `first` falls in.
    pub block: usize,
}

/// How the two faster paths compare with the serial decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Comparison {
    pub serial_error: Option<&'static str>,
    pub wide: Option<Diff>,
    pub one_lane: Option<Diff>,
    pub wide_error: Option<&'static str>,
    pub one_lane_error: Option<&'static str>,
}

impl Comparison {
    /// Whether any path differs from the serial result (or failed where it
    /// gave bytes).
    pub fn differs(&self) -> bool {
        self.serial_error.is_some()
            || self.wide.is_some()
            || self.one_lane.is_some()
            || self.wide_error.is_some()
            || self.one_lane_error.is_some()
    }
}

/// Compares the wide-lane and one-lane results with the serial one.
pub fn compare(serial: &[u8], wide: &[u8], one_lane: &[u8], block: usize) -> Comparison {
    Comparison {
        wide: diff(serial, wide, block),
        one_lane: diff(serial, one_lane, block),
        ..Comparison::default()
    }
}

fn diff(serial: &[u8], other: &[u8], block: usize) -> Option<Diff> {
    let mut first = None;
    let mut count = 0usize;
    for (at, (&a, &b)) in serial.iter().zip(other).enumerate() {
        if a != b {
            first.get_or_insert(at);
            count += 1;
        }
    }
    if serial.len() != other.len() {
        first.get_or_insert(serial.len().min(other.len()));
        count += serial.len().abs_diff(other.len());
    }
    first.map(|first| Diff {
        first,
        count,
        block: first / block.max(1),
    })
}

/// The mode of the first packed block, for the formats that carry one.
pub fn mode0(bc: u8, dst: &[u8]) -> Option<u8> {
    match bc {
        // The control word's lowest set bit names the mode.
        7 => dst
            .first()
            .map(|byte| byte.trailing_zeros() as u8)
            .filter(|&mode| mode < 8),
        // The mode code sits in the block's low bits (see `bc6`).
        6 if dst.len() >= 2 => {
            let code = u16::from_le_bytes([dst[0], dst[1]]) & 31;
            Some(match code {
                0 => 1,
                1 => 2,
                2 => 3,
                3 => 11,
                6 => 4,
                7 => 12,
                10 => 5,
                11 => 13,
                14 => 6,
                15 => 14,
                18 => 7,
                22 => 8,
                26 => 9,
                30 => 10,
                _ => return None,
            })
        }
        _ => None,
    }
}

/// One checked request, formatted as the line the check logs and the replay
/// test prints.
pub struct Report<'a> {
    pub ordinal: u64,
    pub bc: u8,
    pub source_size: usize,
    pub destination_size: usize,
    /// BLAKE3 of the compressed input, hex (the request's name in the keep
    /// folder).
    pub hash: &'a str,
    pub mode0: Option<u8>,
    pub comparison: Comparison,
}

impl Report<'_> {
    /// One line: sizes, the stream's layout fields and mode, the input's
    /// hash, then each path's first difference. No bytes, file name or path.
    pub fn line(&self) -> String {
        let (block, widths, per) = layout(self.bc).unwrap_or((0, &[], 0));
        let mut line = format!(
            "bcpack check #{}: bc={} kind=BC{} block={block} source={} unpacked={} lanes={} per={per} widths=",
            self.ordinal,
            self.bc,
            self.bc,
            self.source_size,
            self.destination_size,
            widths.len(),
        );
        for (k, width) in widths.iter().enumerate() {
            if k > 0 {
                line.push(',');
            }
            let _ = write!(line, "{width}");
        }
        if let Some(mode) = self.mode0 {
            let _ = write!(line, " mode0={mode}");
        }
        let _ = write!(line, " blake3={}", self.hash);
        if let Some(error) = self.comparison.serial_error {
            let _ = write!(line, " serial=error={error}");
        } else {
            let _ = write!(
                line,
                " wide={} one-lane={}",
                side(self.comparison.wide, self.comparison.wide_error),
                side(self.comparison.one_lane, self.comparison.one_lane_error),
            );
        }
        line
    }
}

fn side(diff: Option<Diff>, error: Option<&str>) -> String {
    match (error, diff) {
        (Some(error), _) => format!("error={error}"),
        (None, Some(diff)) => format!(
            "first={} count={} block={}",
            diff.first, diff.count, diff.block
        ),
        (None, None) => "same".to_owned(),
    }
}

/// Writes a checked request's compressed input and parameters into the keep
/// folder: `<hash>.bin` beside `<hash>.txt`. Refuses any folder inside a
/// repository.
pub fn keep_request(
    dir: &Path,
    report: &Report<'_>,
    helpers: usize,
    source: &[u8],
) -> std::io::Result<()> {
    if !keep_allowed(dir) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the keep folder is inside a repository or is not absolute",
        ));
    }
    let Some((block, widths, per)) = layout(report.bc) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no layout for the BC format",
        ));
    };
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{}.bin", report.hash)), source)?;
    let mut text = String::new();
    let _ = writeln!(text, "blake3={}", report.hash);
    let _ = writeln!(text, "ordinal={}", report.ordinal);
    let _ = writeln!(text, "bc={}", report.bc);
    let _ = writeln!(text, "block={block}");
    let _ = writeln!(text, "source_size={}", report.source_size);
    let _ = writeln!(text, "destination_size={}", report.destination_size);
    let _ = writeln!(text, "lanes={}", widths.len());
    let _ = writeln!(text, "per={per}");
    text.push_str("widths=");
    for (k, width) in widths.iter().enumerate() {
        if k > 0 {
            text.push(',');
        }
        let _ = write!(text, "{width}");
    }
    let _ = writeln!(text);
    let _ = writeln!(text, "helpers={helpers}");
    let _ = writeln!(text, "wide=1");
    let _ = writeln!(text, "one_lane=1");
    std::fs::write(dir.join(format!("{}.txt", report.hash)), text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_finds_equal_buffers_and_a_known_offset() {
        let serial: Vec<u8> = (0..32).collect();
        let equal = compare(&serial, &serial, &serial, 8);
        assert!(!equal.differs());
        assert_eq!(equal, Comparison::default());

        let mut wide = serial.clone();
        wide[9] = 0xff;
        wide[17] = 0xee;
        let mut one = serial.clone();
        one[24] = 0xdd;
        let found = compare(&serial, &wide, &one, 8);
        assert_eq!(
            found.wide,
            Some(Diff {
                first: 9,
                count: 2,
                block: 1
            })
        );
        assert_eq!(
            found.one_lane,
            Some(Diff {
                first: 24,
                count: 1,
                block: 3
            })
        );
        assert!(found.differs());
    }

    #[test]
    fn the_check_is_off_by_default() {
        assert!(!enabled(None));
        assert!(!enabled(Some(OsStr::new("0"))));
        assert!(!enabled(Some(OsStr::new("on"))));
        assert!(enabled(Some(OsStr::new("1"))));
    }

    #[test]
    fn a_keep_folder_inside_a_repository_is_refused() {
        let base = std::env::temp_dir().join(format!("victory-bcpack-keep-{}", std::process::id()));
        let repo = base.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        assert!(!keep_allowed(&repo.join("run")));
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        assert!(keep_allowed(&outside));
        assert!(keep_allowed(&base.join("not-created-yet")));
        assert!(!keep_allowed(Path::new("relative/run")));
        assert!(!keep_allowed(&outside.join("..").join("other")));
        std::fs::remove_dir_all(&base).ok();
    }

    /// A directory link (a symlink, or on Windows a junction, which needs no privilege) at `link` to `target`;
    /// false where this platform or account cannot make either.
    fn dir_link(target: &Path, link: &Path) -> bool {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_dir(target, link).is_ok()
                || std::process::Command::new("cmd")
                    .args(["/C", "mklink", "/J"])
                    .arg(link)
                    .arg(target)
                    .output()
                    .is_ok_and(|out| out.status.success())
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (target, link);
            false
        }
    }

    /// A folder outside every repository by its spelling, reached through a link into one, is inside it: the
    /// check follows the link (Sol's and the laptop's finding on the lexical walk).
    #[test]
    fn a_keep_folder_reached_through_a_link_into_a_repository_is_refused() {
        /// Removes the link, then the folders, when the test ends or fails.
        struct Tidy(std::path::PathBuf);
        impl Drop for Tidy {
            fn drop(&mut self) {
                // The link goes first, so removing the base never walks into the repository through it.
                std::fs::remove_dir(self.0.join("link")).ok();
                std::fs::remove_dir_all(&self.0).ok();
            }
        }
        let base = std::env::temp_dir().join(format!("victory-bcpack-link-{}", std::process::id()));
        let _tidy = Tidy(base.clone());
        let repo = base.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let inside = repo.join("target").join("deep");
        std::fs::create_dir_all(inside.join("made")).unwrap();
        // A link to a folder inside the repository, not to its root: nothing on the link's own path holds a `.git`.
        let link = base.join("link");
        if !dir_link(&inside, &link) {
            eprintln!("no directory symlink or junction can be made here; skipped");
            return;
        }
        assert!(!keep_allowed(&link), "the link itself leads into the repository");
        assert!(!keep_allowed(&link.join("made")), "an existing folder through the link");
        assert!(!keep_allowed(&link.join("run")), "a folder not made yet, through the link");
    }

    #[test]
    fn the_layout_matches_the_synthetic_streams() {
        for bc in [1u8, 3, 4, 5, 6, 7] {
            let (block, lanes, per) = layout(bc).unwrap();
            let (synth_lanes, synth_per, synth_block) = super::super::synth::layout(bc);
            assert_eq!(lanes, synth_lanes, "bc {bc} lanes");
            assert_eq!(per, synth_per, "bc {bc} per");
            assert_eq!(block, synth_block, "bc {bc} block");
        }
    }
}
