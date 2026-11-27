//! The texture pack (off unless configured): pictures of the owner's own in
//! the place of the title's textures. `VICTORY_TEXTURE_OVERRIDE=<folder>`
//! names a folder outside every repository holding `overrides.toml` and the
//! pictures it names (PNG or JPEG files in the folder itself):
//!
//! ```toml
//! # How a picture of another shape fills a texture: "cover" (all of the
//! # texture, the picture's overflow cut off evenly on both sides; the
//! # default) or "letterbox" (all of the picture, centred, black bars beside
//! # it or above and below).
//! fit = "cover"
//!
//! [[texture]]
//! id = "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98-m11"
//! image = "replacement.jpg"
//! # fit = "letterbox"    (this texture's own, if not the file's)
//! ```
//!
//! A texture's identity (`id`) says what the title wrote into it and what it
//! is: a hash of the texels of every level, from level 0, as the title laid
//! them out (detiled, each row without the host's padding; the first 16 bytes
//! of their BLAKE3 hash, in hex), then its width, height, host format (the
//! DXGI number) and levels. A texture has one only while all of it is the
//! title's and staged: a placed texture staged from a smaller level, or one
//! with a level not held yet (levels on demand), has none and is left as the
//! title has it. Two textures of one identity hold the same texels.
//!
//! Whenever a texture of a mapped size, format and level count is detiled
//! (its first upload, and every refresh that finds its memory changed), every
//! level goes through a buffer of the pack's, which takes its identity. A
//! mapped one is prepared on a thread of the pack's, never on the submitting
//! one: the picture decoded (Windows' own PNG and JPEG decoders, WIC, through
//! the `windows` crate the workspace has), fitted to the texture, each level
//! filtered down from the one above (a tent filter as wide as the reduction),
//! and converted to its format, block-compressed formats encoded (BC1, BC3,
//! and BC7 in its mode 6). Until it is ready the title's texels are uploaded;
//! once it is, the texture is detiled anew at its next refresh and the picture
//! goes in its place, every level or none. Formats: 8-bit RGBA, BGRA and BGRX,
//! BC1, BC3 and BC7, their sRGB and typeless forms included (the picture's
//! bytes are taken as they are, no conversion of gamma). Other formats,
//! arrays, cubes and 3-D textures are left as the title has them.
//!
//! What it takes from the folder is bounded: `overrides.toml` and each picture
//! must be plain files in the folder itself (a bare file name; no link, no
//! folder, nothing a link leads elsewhere to, checked on the opened file), of
//! at most [`MOST_MAPPING_BYTES`] and [`MOST_PICTURE_BYTES`]; at most
//! [`MOST_ENTRIES`] entries; pictures and textures of at most [`MOST_SIDE`] a
//! side and [`MOST_PIXELS`] texels; one preparation at a time; prepared
//! pictures kept within [`MOST_RETAINED`] bytes (decoded pictures are let go).
//! Every failure is said in fixed words that carry no input (an entry is
//! named by its number in the file). No line names a folder.
//!
//! Finding the identities (diagnostics builds): with
//! `VICTORY_TEXTURE_OVERRIDE_LOG=1`, one line per distinct identity of a large
//! texture (both sides 512 or more) that a draw over most of the screen reads
//! (its first viewport covers three quarters of its first target, a target
//! of at least 1280x720): at the draw, or at the detiling that follows it
//! within a frame. E.g. `texture pack:
//! 5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98-m11 (1920x1080 BC7_UNORM,
//! 11 levels) drawn over most of the screen, frame 412`. A texture that shows
//! more than [`log::MOST_PICTURES`] pictures (a movie's surface) is said to,
//! once, and its next pictures are not logged. `=2` also logs every large
//! texture's identity as it is uploaded, drawn over the screen or not (for one
//! a shader picks by an index the draw does not name).
//!
//! shortcut: a view whose channels the title swizzles shows the picture's
//! channels swizzled too; a JPEG's orientation tag is not followed; a native
//! fault inside Windows' decoders is not contained (only its two built-in
//! decoders are used, on bounded input).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The file in the folder that maps identities to pictures.
const MAPPING: &str = "overrides.toml";
/// Its bytes at most.
pub(crate) const MOST_MAPPING_BYTES: u64 = 64 << 10;
/// Its entries at most.
pub(crate) const MOST_ENTRIES: usize = 256;
/// A picture file's bytes at most.
pub(crate) const MOST_PICTURE_BYTES: u64 = 64 << 20;
/// A picture's or a replaced texture's side, and texels, at most.
pub(crate) const MOST_SIDE: u32 = 8192;
pub(crate) const MOST_PIXELS: u64 = 16 << 20;
/// The bytes of prepared pictures kept, at most.
pub(crate) const MOST_RETAINED: usize = 256 << 20;

/// A texture as the pack sees it: a 2-D texture of one slice, the only kind
/// it replaces or logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Shape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The host texture's format (a DXGI number).
    pub(crate) format: u32,
    pub(crate) mips: u32,
}

/// Whether a picture or texture of `width` x `height` is one the pack takes.
fn within(width: u32, height: u32) -> bool {
    width >= 1 && height >= 1 && width <= MOST_SIDE && height <= MOST_SIDE && u64::from(width) * u64::from(height) <= MOST_PIXELS
}

/// What a texture is and what the title wrote into it (see the module
/// comment); printed and parsed as `hash-WIDTHxHEIGHT-fFORMAT-mLEVELS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Identity {
    hash: u128,
    shape: Shape,
}

impl Identity {
    /// Of a texture of `shape` whose levels, each from level 0 on, are
    /// `levels`: `(texels, pitch, rows, row_bytes)`, rows `pitch` apart.
    pub(crate) fn of<'a>(shape: &Shape, levels: impl IntoIterator<Item = (&'a [u8], usize, usize, usize)>) -> Identity {
        let mut hasher = blake3::Hasher::new();
        for (texels, pitch, rows, row_bytes) in levels {
            for row in 0..rows {
                hasher.update(&texels[row * pitch..][..row_bytes]);
            }
        }
        let mut first = [0u8; 16];
        first.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
        Identity { hash: u128::from_be_bytes(first), shape: *shape }
    }

    /// Its size, format and levels, in words.
    #[cfg(feature = "diagnostics")]
    fn described(&self) -> String {
        let s = self.shape;
        let kept = if Encoding::of(s.format).is_none() { ", no encoder here: not replaceable" } else { "" };
        format!("{}x{} {}, {} levels{kept}", s.width, s.height, format_name(s.format), s.mips)
    }
}

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.shape;
        write!(f, "{:032x}-{}x{}-f{}-m{}", self.hash, s.width, s.height, s.format, s.mips)
    }
}

/// Not an identity as the log prints them.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NotAnIdentity;

impl std::str::FromStr for Identity {
    type Err = NotAnIdentity;

    fn from_str(s: &str) -> Result<Identity, NotAnIdentity> {
        let mut parts = s.trim().split('-');
        let (Some(hash), Some(size), Some(format), Some(mips), None) =
            (parts.next(), parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(NotAnIdentity);
        };
        let (width, height) = size.split_once('x').ok_or(NotAnIdentity)?;
        let number = |s: Option<&str>| s.and_then(|s| s.parse::<u32>().ok()).ok_or(NotAnIdentity);
        if hash.len() != 32 {
            return Err(NotAnIdentity);
        }
        Ok(Identity {
            hash: u128::from_str_radix(hash, 16).map_err(|_| NotAnIdentity)?,
            shape: Shape {
                width: number(Some(width))?,
                height: number(Some(height))?,
                format: number(format.strip_prefix('f'))?,
                mips: number(mips.strip_prefix('m'))?,
            },
        })
    }
}

/// How a picture of another shape fills a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Fit {
    /// All of the texture; the picture's overflow cut off evenly.
    #[default]
    Cover,
    /// All of the picture, centred; black bars where it does not reach.
    Letterbox,
}

/// `overrides.toml`.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    fit: Fit,
    #[serde(default)]
    texture: Vec<FileEntry>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEntry {
    id: String,
    image: String,
    fit: Option<Fit>,
}

struct Entry {
    /// Its place in the file, from 1: how messages name it.
    number: usize,
    /// The picture's file name, in the folder.
    image: String,
    fit: Fit,
}

/// What became of a mapped identity's preparation.
enum State {
    Preparing,
    Ready(Arc<Prepared>),
    Refused,
}

/// What the pack has for a texture of an identity.
pub(crate) enum Replacement {
    /// The picture, prepared.
    Ready(Arc<Prepared>),
    /// The picture is being prepared: the title's texels for now.
    Pending,
    /// Nothing: the identity is not mapped, or its picture cannot be used.
    None,
}

/// Counts finished preparations: a texture that waits for one is detiled
/// anew once this moves (`crate::texture`, `Staging::pack_wait`).
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// How many preparations have finished.
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// The mapping and what was made from it.
pub(crate) struct Pack {
    /// The folder, its links resolved.
    folder: PathBuf,
    entries: HashMap<Identity, Entry>,
    /// The shapes mapped: only textures of one of them are hashed.
    shapes: HashSet<Shape>,
    state: Mutex<HashMap<Identity, State>>,
    /// Bytes of prepared pictures kept (and being prepared), and the most.
    retained: AtomicUsize,
    most_retained: usize,
}

impl Pack {
    /// The mapping in `folder`. The error is in fixed words.
    pub(crate) fn load(folder: &Path) -> Result<Pack, String> {
        let folder = std::fs::canonicalize(folder).map_err(|_| "the folder could not be opened".to_string())?;
        let bytes = read_in(&folder, MAPPING, MOST_MAPPING_BYTES).map_err(|why| format!("{MAPPING} {why}"))?;
        let text = String::from_utf8(bytes).map_err(|_| format!("{MAPPING} is not UTF-8 text"))?;
        Pack::parse(&text, &folder)
    }

    fn parse(text: &str, folder: &Path) -> Result<Pack, String> {
        let file: File = toml::from_str(text).map_err(|e| {
            let line = e.span().map_or(0, |span| text[..span.start.min(text.len())].matches('\n').count() + 1);
            format!("{MAPPING} is not a mapping the texture pack reads (line {line})")
        })?;
        if file.texture.len() > MOST_ENTRIES {
            return Err(format!("{MAPPING} has more than {MOST_ENTRIES} entries"));
        }
        let mut entries: HashMap<Identity, Entry> = HashMap::new();
        for (i, entry) in file.texture.into_iter().enumerate() {
            let number = i + 1;
            let id: Identity = entry
                .id
                .parse()
                .map_err(|_| format!("{MAPPING}: the id of entry {number} is not an identity as the log prints them"))?;
            if !plain_name(&entry.image) {
                return Err(format!("{MAPPING}: the image of entry {number} is not the name of a file in the folder"));
            }
            if let Some(before) = entries.get(&id) {
                return Err(format!("{MAPPING}: entry {number} maps the identity of entry {} again", before.number));
            }
            let fit = entry.fit.unwrap_or(file.fit);
            entries.insert(id, Entry { number, image: entry.image, fit });
        }
        Ok(Pack {
            folder: folder.to_path_buf(),
            shapes: entries.keys().map(|id| id.shape).collect(),
            entries,
            state: Mutex::default(),
            retained: AtomicUsize::new(0),
            most_retained: MOST_RETAINED,
        })
    }

    fn wants(&self, shape: &Shape) -> bool {
        self.shapes.contains(shape)
    }

    /// What the pack has for a texture of identity `id`. The first time a
    /// mapped one is asked for, its preparation starts on a thread of the
    /// pack's.
    pub(crate) fn replacement(&'static self, id: &Identity) -> Replacement {
        if !self.entries.contains_key(id) {
            return Replacement::None;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.get(id) {
            Some(State::Ready(prepared)) => return Replacement::Ready(prepared.clone()),
            Some(State::Preparing) => return Replacement::Pending,
            Some(State::Refused) => return Replacement::None,
            None => {}
        }
        state.insert(*id, State::Preparing);
        drop(state);
        let id = *id;
        let spawned = std::thread::Builder::new().name("texture pack".into()).spawn(move || self.work(id));
        if spawned.is_err() {
            tracing::warn!("texture pack: {id} keeps the title's texels: no thread to prepare its picture on");
            self.finish(id, None);
            return Replacement::None;
        }
        Replacement::Pending
    }

    /// The preparation of `id`'s picture, on the pack's thread (one at a
    /// time).
    fn work(&self, id: Identity) {
        static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(test)]
        tests::gate();
        let started = std::time::Instant::now();
        let Some(entry) = self.entries.get(&id) else {
            return self.finish(id, None);
        };
        let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.prepare(&id, entry)))
            .unwrap_or_else(|_| Err("its preparation failed".to_string()));
        match made {
            Ok(prepared) => {
                tracing::info!(
                    "texture pack: {id} takes the picture of entry {} ({:?}, {} levels, prepared in {} ms)",
                    entry.number,
                    entry.fit,
                    prepared.levels.len(),
                    started.elapsed().as_millis()
                );
                self.finish(id, Some(Arc::new(prepared)));
            }
            Err(why) => {
                tracing::warn!("texture pack: {id} keeps the title's texels: entry {}: {why}", entry.number);
                self.finish(id, None);
            }
        }
    }

    fn finish(&self, id: Identity, prepared: Option<Arc<Prepared>>) {
        let state = prepared.map_or(State::Refused, State::Ready);
        self.state.lock().unwrap_or_else(|e| e.into_inner()).insert(id, state);
        GENERATION.fetch_add(1, Ordering::Release);
    }

    /// The picture of `entry` made into a replacement for textures of `id`.
    /// The error is in fixed words.
    fn prepare(&self, id: &Identity, entry: &Entry) -> Result<Prepared, String> {
        let shape = id.shape;
        let encoding = Encoding::of(shape.format)
            .ok_or_else(|| format!("the texture's format ({}) has no encoder here", format_name(shape.format)))?;
        if !within(shape.width, shape.height) {
            return Err(format!("the texture is larger than the pack takes ({MOST_SIDE} a side, {MOST_PIXELS} texels)"));
        }
        // Its bytes are counted before any work is done, and given back if
        // it comes to nothing.
        let bytes = Prepared::bytes(&shape, encoding);
        if self
            .retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |r| (r + bytes <= self.most_retained).then_some(r + bytes))
            .is_err()
        {
            return Err(format!("the pictures prepared would take more than {} MiB", self.most_retained >> 20));
        }
        struct Counted<'a>(&'a AtomicUsize, usize, bool);
        impl Drop for Counted<'_> {
            fn drop(&mut self) {
                if !self.2 {
                    self.0.fetch_sub(self.1, Ordering::AcqRel);
                }
            }
        }
        let mut counted = Counted(&self.retained, bytes, false);
        let file = read_in(&self.folder, &entry.image, MOST_PICTURE_BYTES).map_err(|why| format!("its picture {why}"))?;
        let picture = decode(&file).map_err(|why| format!("its picture {why}"))?;
        drop(file);
        let prepared = Prepared::new(&picture, &shape, entry.fit, encoding);
        counted.2 = true;
        Ok(prepared)
    }

    /// Whether `id`'s preparation is over (tests).
    #[cfg(test)]
    pub(crate) fn settled(&self, id: &Identity) -> bool {
        !matches!(self.state.lock().unwrap_or_else(|e| e.into_inner()).get(id), Some(State::Preparing))
    }
}

/// Whether `name` is a bare file name: no folder, no drive, no stream.
fn plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    let single = matches!(
        (parts.next(), parts.next()),
        (Some(std::path::Component::Normal(n)), None) if n == std::ffi::OsStr::new(name)
    );
    single && !name.contains(['/', '\\', ':'])
}

/// The bytes of file `name` in `folder` (its links resolved), if it is a
/// plain file there, opened as such, of `most` bytes at most. The error is in
/// fixed words.
fn read_in(folder: &Path, name: &str, most: u64) -> Result<Vec<u8>, &'static str> {
    use std::io::Read;
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    const NOT_PLAIN: &str = "is not a plain file in the folder";
    if !plain_name(name) {
        return Err(NOT_PLAIN);
    }
    // A link is opened as itself, not followed, and refused.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(folder.join(name))
        .map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { "was not found" } else { "could not be opened" })?;
    let meta = file.metadata().map_err(|_| "could not be opened")?;
    if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 || !meta.is_file() || !in_folder(&file, folder) {
        return Err(NOT_PLAIN);
    }
    if meta.len() > most {
        return Err("is larger than the pack takes");
    }
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    file.take(most + 1).read_to_end(&mut bytes).map_err(|_| "could not be read")?;
    if bytes.len() as u64 > most {
        return Err("is larger than the pack takes");
    }
    Ok(bytes)
}

/// Whether the opened `file` lies in `folder` itself, as Windows names it
/// once every link is resolved.
fn in_folder(file: &std::fs::File, folder: &Path) -> bool {
    use std::os::windows::ffi::OsStringExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW;
    let mut name = vec![0u16; 32 * 1024];
    // SAFETY: an open handle and a buffer of the length given.
    let n = unsafe { GetFinalPathNameByHandleW(file.as_raw_handle(), name.as_mut_ptr(), name.len() as u32, 0) } as usize;
    if n == 0 || n >= name.len() {
        return false;
    }
    let path = PathBuf::from(std::ffi::OsString::from_wide(&name[..n]));
    path.parent()
        .is_some_and(|parent| parent.as_os_str().to_string_lossy().eq_ignore_ascii_case(&folder.as_os_str().to_string_lossy()))
}

/// The pack, from `VICTORY_TEXTURE_OVERRIDE` (read once).
#[cfg(not(test))]
fn active() -> Option<&'static Pack> {
    static PACK: std::sync::OnceLock<Option<Pack>> = std::sync::OnceLock::new();
    PACK.get_or_init(|| {
        let folder = std::env::var_os("VICTORY_TEXTURE_OVERRIDE").filter(|f| !f.is_empty())?;
        match Pack::load(Path::new(&folder)) {
            Ok(pack) => {
                tracing::info!("texture pack: on, {} textures mapped", pack.entries.len());
                Some(pack)
            }
            Err(why) => {
                tracing::warn!("texture pack: VICTORY_TEXTURE_OVERRIDE is set and not used: {why}");
                None
            }
        }
    })
    .as_ref()
}

/// Whether the pack or its log is on: off, staging and detiling are as
/// without it (read once).
#[cfg(not(test))]
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        let logged = log::level() != 0;
        #[cfg(not(feature = "diagnostics"))]
        let logged = false;
        active().is_some() || logged
    })
}

/// The pack a test puts in place (the environment is not read in tests).
#[cfg(test)]
static TEST_PACK: Mutex<Option<&'static Pack>> = Mutex::new(None);

#[cfg(test)]
fn active() -> Option<&'static Pack> {
    *TEST_PACK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(crate) fn on() -> bool {
    active().is_some()
}

/// What the pack does with a texture being detiled: hash every level (its
/// shape is mapped, or logged), and replace it if mapped.
pub(crate) struct Look {
    pack: Option<&'static Pack>,
}

/// The look for a texture of `shape`; none (nothing to do) while the pack
/// is off and nothing is logged.
pub(crate) fn look(shape: Shape) -> Option<Look> {
    let pack = active().filter(|p| p.wants(&shape));
    #[cfg(feature = "diagnostics")]
    let logged = log::wants(&shape);
    #[cfg(not(feature = "diagnostics"))]
    let logged = false;
    (pack.is_some() || logged).then_some(Look { pack })
}

impl Look {
    /// What the pack has for a texture of identity `id`.
    pub(crate) fn replacement(&self, id: &Identity) -> Replacement {
        self.pack.map_or(Replacement::None, |pack| pack.replacement(id))
    }
}

/// Calls `f` with a buffer of the pack's of `len` bytes, this thread's (the
/// staging buffer is write-combined memory, slow to read back).
pub(crate) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.clear();
        scratch.resize(len, 0);
        f(&mut scratch)
    })
}

/// A picture: RGBA, 8 bits a channel, rows of `width` texels.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({}x{})", self.width, self.height)
    }
}

impl Image {
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = (y as usize * self.width as usize + x as usize) * 4;
        [self.rgba[at], self.rgba[at + 1], self.rgba[at + 2], self.rgba[at + 3]]
    }
}

/// Decodes a PNG or JPEG file's bytes with Windows' own decoder for it (no
/// other decoder is asked), its size checked before its texels are made.
/// On the pack's thread, which has COM to itself. The error is in fixed
/// words.
fn decode(bytes: &[u8]) -> Result<Image, &'static str> {
    use windows::Win32::Graphics::Imaging::*;
    use windows::Win32::System::Com::*;
    const NOT_DECODED: &str = "could not be decoded";
    let decoder = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        CLSID_WICPngDecoder2
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        CLSID_WICJpegDecoder
    } else {
        return Err("is not a PNG or JPEG file");
    };
    // SAFETY: COM for this thread, undone before it returns; every object
    // made here is released first; the stream reads `bytes`, which outlive
    // it.
    unsafe {
        if CoInitializeEx(None, COINIT_MULTITHREADED).is_err() {
            return Err(NOT_DECODED);
        }
        let decoded = (|| -> Result<Image, &'static str> {
            let factory: IWICImagingFactory =
                CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER).map_err(|_| NOT_DECODED)?;
            let stream = factory.CreateStream().map_err(|_| NOT_DECODED)?;
            stream.InitializeFromMemory(bytes).map_err(|_| NOT_DECODED)?;
            let decoder: IWICBitmapDecoder = CoCreateInstance(&decoder, None, CLSCTX_INPROC_SERVER).map_err(|_| NOT_DECODED)?;
            decoder.Initialize(&stream, WICDecodeMetadataCacheOnDemand).map_err(|_| NOT_DECODED)?;
            let frame = decoder.GetFrame(0).map_err(|_| NOT_DECODED)?;
            let (mut width, mut height) = (0, 0);
            frame.GetSize(&mut width, &mut height).map_err(|_| NOT_DECODED)?;
            if !within(width, height) {
                return Err("is larger than the pack takes");
            }
            let rgba = WICConvertBitmapSource(&GUID_WICPixelFormat32bppRGBA, &frame).map_err(|_| NOT_DECODED)?;
            let mut texels = vec![0u8; width as usize * height as usize * 4];
            rgba.CopyPixels(std::ptr::null(), width * 4, &mut texels).map_err(|_| NOT_DECODED)?;
            Ok(Image { width, height, rgba: texels })
        })();
        CoUninitialize();
        decoded
    }
}

/// `picture` fitted to `width` x `height`.
fn fitted(picture: &Image, width: u32, height: u32, fit: Fit) -> Image {
    let (w0, h0) = (f64::from(picture.width), f64::from(picture.height));
    let (w, h) = (f64::from(width), f64::from(height));
    match fit {
        Fit::Cover => {
            let scale = (w / w0).max(h / h0);
            let (sw, sh) = (w / scale, h / scale);
            resample(picture, [(w0 - sw) / 2.0, (h0 - sh) / 2.0, sw, sh], width, height)
        }
        Fit::Letterbox => {
            let scale = (w / w0).min(h / h0);
            let inner_w = ((w0 * scale).round() as u32).clamp(1, width);
            let inner_h = ((h0 * scale).round() as u32).clamp(1, height);
            let inner = resample(picture, [0.0, 0.0, w0, h0], inner_w, inner_h);
            let mut out = Image {
                width,
                height,
                rgba: [0, 0, 0, 255].repeat(width as usize * height as usize),
            };
            let (x0, y0) = ((width - inner_w) / 2, (height - inner_h) / 2);
            let row = inner_w as usize * 4;
            for y in 0..inner_h as usize {
                let to = ((y0 as usize + y) * width as usize + x0 as usize) * 4;
                out.rgba[to..to + row].copy_from_slice(&inner.rgba[y * row..][..row]);
            }
            out
        }
    }
}

/// The part of `src` at `[x, y, width, height]` (in texels, fractions
/// allowed) resampled onto `width` x `height` texels. One output row at a
/// time: no buffer beyond the output and a row of it.
fn resample(src: &Image, from: [f64; 4], width: u32, height: u32) -> Image {
    let xs = taps(src.width, from[0], from[2], width);
    let ys = taps(src.height, from[1], from[3], height);
    let (w, sw) = (width as usize, src.width as usize);
    let mut rgba = vec![0u8; w * height as usize * 4];
    let mut row = vec![[0f32; 4]; w];
    for (y, down) in ys.iter().enumerate() {
        row.fill([0.0; 4]);
        for &(sy, weight) in down {
            let source = &src.rgba[sy * sw * 4..][..sw * 4];
            for (out, across) in row.iter_mut().zip(&xs) {
                let mut sum = [0f32; 4];
                for &(sx, w) in across {
                    for c in 0..4 {
                        sum[c] += w * f32::from(source[sx * 4 + c]);
                    }
                }
                for c in 0..4 {
                    out[c] += weight * sum[c];
                }
            }
        }
        for (x, texel) in row.iter().enumerate() {
            for c in 0..4 {
                rgba[(y * w + x) * 4 + c] = (texel[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Image { width, height, rgba }
}

/// For each of `n` outputs over the source span `[from, from + len)` of a
/// source `size` texels long: the source texels it takes and their weights
/// (summing to 1). A tent reaching one output's width either side of the
/// output's centre, and at least one texel (bilinear when enlarging);
/// texels past an edge are the edge's.
fn taps(size: u32, from: f64, len: f64, n: u32) -> Vec<Vec<(usize, f32)>> {
    let step = len / f64::from(n);
    let reach = step.max(1.0);
    let last = i64::from(size) - 1;
    (0..n)
        .map(|i| {
            let centre = from + (f64::from(i) + 0.5) * step;
            let first = (centre - reach - 0.5).floor() as i64;
            let end = (centre + reach - 0.5).ceil() as i64;
            let mut raw: Vec<(usize, f64)> = Vec::new();
            for j in first..=end {
                let weight = 1.0 - ((j as f64 + 0.5 - centre).abs() / reach);
                if weight <= 0.0 {
                    continue;
                }
                let at = j.clamp(0, last) as usize;
                // Clamped texels repeat next to each other: merged.
                match raw.last_mut() {
                    Some(tap) if tap.0 == at => tap.1 += weight,
                    _ => raw.push((at, weight)),
                }
            }
            let sum: f64 = raw.iter().map(|t| t.1).sum();
            raw.into_iter().map(|(at, w)| (at, (w / sum) as f32)).collect()
        })
        .collect()
}

/// How the texels of a format the pack writes are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Rgba,
    Bgra,
    Bgrx,
    Bc1,
    Bc3,
    Bc7,
}

impl Encoding {
    fn of(format: u32) -> Option<Encoding> {
        Some(match format {
            27..=29 => Encoding::Rgba,
            87 | 90 | 91 => Encoding::Bgra,
            88 | 92 | 93 => Encoding::Bgrx,
            70..=72 => Encoding::Bc1,
            76..=78 => Encoding::Bc3,
            97..=99 => Encoding::Bc7,
            _ => return None,
        })
    }

    /// Bytes of a 4x4 block (block-compressed formats).
    fn block_bytes(self) -> Option<usize> {
        match self {
            Encoding::Bc1 => Some(8),
            Encoding::Bc3 | Encoding::Bc7 => Some(16),
            _ => None,
        }
    }
}

/// A DXGI format's name (the ones the title's large textures use), or its
/// number.
fn format_name(format: u32) -> String {
    let name = match format {
        2 => "R32G32B32A32_FLOAT",
        10 => "R16G16B16A16_FLOAT",
        24 => "R10G10B10A2_UNORM",
        26 => "R11G11B10_FLOAT",
        27 => "R8G8B8A8_TYPELESS",
        28 => "R8G8B8A8_UNORM",
        29 => "R8G8B8A8_UNORM_SRGB",
        49 => "R8G8_UNORM",
        61 => "R8_UNORM",
        70 => "BC1_TYPELESS",
        71 => "BC1_UNORM",
        72 => "BC1_UNORM_SRGB",
        73 => "BC2_TYPELESS",
        74 => "BC2_UNORM",
        75 => "BC2_UNORM_SRGB",
        76 => "BC3_TYPELESS",
        77 => "BC3_UNORM",
        78 => "BC3_UNORM_SRGB",
        79 => "BC4_TYPELESS",
        80 => "BC4_UNORM",
        82 => "BC5_TYPELESS",
        83 => "BC5_UNORM",
        87 => "B8G8R8A8_UNORM",
        88 => "B8G8R8X8_UNORM",
        90 => "B8G8R8A8_TYPELESS",
        91 => "B8G8R8A8_UNORM_SRGB",
        92 => "B8G8R8X8_TYPELESS",
        93 => "B8G8R8X8_UNORM_SRGB",
        95 => "BC6H_UF16",
        96 => "BC6H_SF16",
        97 => "BC7_TYPELESS",
        98 => "BC7_UNORM",
        99 => "BC7_UNORM_SRGB",
        other => return format!("format {other}"),
    };
    name.to_string()
}

/// A replacement: every level of a texture, in its format, rows packed.
pub(crate) struct Prepared {
    levels: Vec<Level>,
}

struct Level {
    bytes: Vec<u8>,
    row_bytes: usize,
    rows: usize,
}

impl Prepared {
    fn new(picture: &Image, shape: &Shape, fit: Fit, encoding: Encoding) -> Prepared {
        let mut image = fitted(picture, shape.width, shape.height, fit);
        let mut levels = vec![encode(&image, encoding)];
        for level in 1..shape.mips.clamp(1, 32) {
            // Each level filtered down from the one above it.
            let (w, h) = ((shape.width >> level).max(1), (shape.height >> level).max(1));
            image = resample(&image, [0.0, 0.0, f64::from(image.width), f64::from(image.height)], w, h);
            levels.push(encode(&image, encoding));
        }
        Prepared { levels }
    }

    /// The bytes a replacement for a texture of `shape` keeps.
    fn bytes(shape: &Shape, encoding: Encoding) -> usize {
        (0..shape.mips.clamp(1, 32))
            .map(|level| {
                let (w, h) = ((shape.width >> level).max(1) as usize, (shape.height >> level).max(1) as usize);
                match encoding.block_bytes() {
                    Some(block) => w.div_ceil(4) * h.div_ceil(4) * block,
                    None => w * h * 4,
                }
            })
            .sum()
    }

    /// Whether every part of a staging buffer, `(level, bytes, pitch, rows)`,
    /// takes its level whole.
    pub(crate) fn fits(&self, parts: impl IntoIterator<Item = (u32, usize, usize, usize)>) -> bool {
        parts.into_iter().all(|(mip, len, pitch, rows)| {
            self.levels.get(mip as usize).is_some_and(|level| {
                level.rows == rows && rows > 0 && level.row_bytes <= pitch && (rows - 1) * pitch + level.row_bytes <= len
            })
        })
    }

    /// Writes level `mip` into a part of a staging buffer: `rows` rows (of
    /// texels, or of blocks) `pitch` bytes apart. False, and nothing
    /// written, if the part does not take it whole ([`Prepared::fits`]).
    pub(crate) fn write(&self, mip: u32, linear: &mut [u8], pitch: usize, rows: usize) -> bool {
        if !self.fits([(mip, linear.len(), pitch, rows)]) {
            return false;
        }
        let level = &self.levels[mip as usize];
        for (row, from) in level.bytes.chunks_exact(level.row_bytes).enumerate() {
            linear[row * pitch..][..level.row_bytes].copy_from_slice(from);
        }
        true
    }
}

/// `image` in `encoding`, rows packed.
fn encode(image: &Image, encoding: Encoding) -> Level {
    let (w, h) = (image.width as usize, image.height as usize);
    let Some(block_bytes) = encoding.block_bytes() else {
        let bytes = match encoding {
            Encoding::Rgba => image.rgba.clone(),
            Encoding::Bgra => image.rgba.chunks_exact(4).flat_map(|t| [t[2], t[1], t[0], t[3]]).collect(),
            _ => image.rgba.chunks_exact(4).flat_map(|t| [t[2], t[1], t[0], 255]).collect(),
        };
        return Level { bytes, row_bytes: w * 4, rows: h };
    };
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut bytes = vec![0u8; bw * bh * block_bytes];
    // Rows of blocks shared out among threads (done once per replacement).
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let rows_each = bh.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (part, chunk) in bytes.chunks_mut(rows_each * bw * block_bytes).enumerate() {
            s.spawn(move || {
                for (i, out) in chunk.chunks_exact_mut(block_bytes).enumerate() {
                    let n = part * rows_each * bw + i;
                    let texels = block(image, (n % bw) as u32, (n / bw) as u32);
                    match encoding {
                        Encoding::Bc1 => out.copy_from_slice(&bc1(&texels)),
                        Encoding::Bc3 => {
                            out[..8].copy_from_slice(&bc4(texels.map(|t| t[3])));
                            out[8..].copy_from_slice(&bc1(&texels));
                        }
                        _ => out.copy_from_slice(&bc7(&texels)),
                    }
                }
            });
        }
    });
    Level {
        bytes,
        row_bytes: bw * block_bytes,
        rows: bh,
    }
}

/// The 4x4 block at block `(bx, by)`, row by row; texels past the edge are
/// the edge's.
fn block(image: &Image, bx: u32, by: u32) -> [[u8; 4]; 16] {
    std::array::from_fn(|i| {
        let (x, y) = (bx * 4 + i as u32 % 4, by * 4 + i as u32 / 4);
        image.texel(x.min(image.width - 1), y.min(image.height - 1))
    })
}

type Texel = [f32; 4];

/// The line through a block's texels that holds them best (the principal
/// axis through their mean), as its two ends: where the texels' projections
/// onto it end.
fn line(texels: &[Texel; 16]) -> (Texel, Texel) {
    let mut mean = [0f32; 4];
    for t in texels {
        for c in 0..4 {
            mean[c] += t[c] / 16.0;
        }
    }
    let mut spread = [[0f32; 4]; 4];
    for t in texels {
        for a in 0..4 {
            for b in 0..4 {
                spread[a][b] += (t[a] - mean[a]) * (t[b] - mean[b]);
            }
        }
    }
    // Power iteration from the channel that spreads most.
    let start = (0..4).max_by(|&a, &b| spread[a][a].total_cmp(&spread[b][b])).unwrap_or(0);
    if spread[start][start] < 1e-3 {
        return (mean, mean);
    }
    let mut axis = spread[start];
    for _ in 0..16 {
        let next: Texel = std::array::from_fn(|a| (0..4).map(|b| spread[a][b] * axis[b]).sum());
        let norm = next.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-12 {
            break;
        }
        axis = next.map(|v| v / norm);
    }
    let norm = axis.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return (mean, mean);
    }
    axis = axis.map(|v| v / norm);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for t in texels {
        let along: f32 = (0..4).map(|c| (t[c] - mean[c]) * axis[c]).sum();
        lo = lo.min(along);
        hi = hi.max(along);
    }
    let end = |at: f32| -> Texel { std::array::from_fn(|c| (mean[c] + axis[c] * at).clamp(0.0, 255.0)) };
    (end(lo), end(hi))
}

/// The ends of the line that comes closest, in least squares, to texels
/// that each lie `t` (0 to 1) of the way from the first end to the second.
fn fit_ends(texels: &[Texel; 16], t: &[f32; 16]) -> Option<(Texel, Texel)> {
    let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
    let (mut x0, mut x1) = ([0f32; 4], [0f32; 4]);
    for (texel, &t) in texels.iter().zip(t) {
        let u = 1.0 - t;
        a += u * u;
        b += u * t;
        c += t * t;
        for ch in 0..4 {
            x0[ch] += u * texel[ch];
            x1[ch] += t * texel[ch];
        }
    }
    let det = a * c - b * b;
    if det.abs() < 1e-6 {
        return None;
    }
    let e0 = std::array::from_fn(|ch| ((c * x0[ch] - b * x1[ch]) / det).clamp(0.0, 255.0));
    let e1 = std::array::from_fn(|ch| ((a * x1[ch] - b * x0[ch]) / det).clamp(0.0, 255.0));
    Some((e0, e1))
}

/// Where each BC1 colour code lies between the two ends.
const BC1_T: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];

/// A BC1 colour block (four colours: the ends kept in decreasing order; the
/// alpha of the texels is not stored).
fn bc1(texels: &[[u8; 4]; 16]) -> [u8; 8] {
    let colour: [Texel; 16] = std::array::from_fn(|i| {
        let t = texels[i];
        [f32::from(t[0]), f32::from(t[1]), f32::from(t[2]), 0.0]
    });
    let (mut e0, mut e1) = line(&colour);
    let mut best = bc1_try(texels, e0, e1);
    for _ in 0..2 {
        let t = best.2.map(|i| BC1_T[usize::from(i)]);
        let Some(ends) = fit_ends(&colour, &t) else {
            break;
        };
        (e0, e1) = ends;
        let next = bc1_try(texels, e0, e1);
        if next.0 >= best.0 {
            break;
        }
        best = next;
    }
    let (_, [mut q0, mut q1], mut codes) = best;
    if q0 < q1 {
        std::mem::swap(&mut q0, &mut q1);
        codes = codes.map(|c| [1, 0, 3, 2][usize::from(c)]);
    }
    if q0 == q1 {
        // Three colours and black: code 0 is the colour.
        codes = [0; 16];
    }
    let bits = codes.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | u32::from(c) << (2 * i));
    let mut out = [0u8; 8];
    out[..2].copy_from_slice(&q0.to_le_bytes());
    out[2..4].copy_from_slice(&q1.to_le_bytes());
    out[4..].copy_from_slice(&bits.to_le_bytes());
    out
}

/// The 5:6:5 ends nearest `e0` and `e1`, the error of the block coded with
/// them, and each texel's code.
fn bc1_try(texels: &[[u8; 4]; 16], e0: Texel, e1: Texel) -> (u32, [u16; 2], [u8; 16]) {
    let q = |e: Texel| -> u16 {
        let r = (e[0] * 31.0 / 255.0).round().clamp(0.0, 31.0) as u16;
        let g = (e[1] * 63.0 / 255.0).round().clamp(0.0, 63.0) as u16;
        let b = (e[2] * 31.0 / 255.0).round().clamp(0.0, 31.0) as u16;
        r << 11 | g << 5 | b
    };
    let (q0, q1) = (q(e0), q(e1));
    let (c0, c1) = (from_565(q0), from_565(q1));
    let mix = |a: i32, b: i32| (2 * a + b + 1) / 3;
    let palette: [[i32; 3]; 4] = [
        c0,
        c1,
        std::array::from_fn(|c| mix(c0[c], c1[c])),
        std::array::from_fn(|c| mix(c1[c], c0[c])),
    ];
    let mut codes = [0u8; 16];
    let mut error = 0u32;
    for (i, t) in texels.iter().enumerate() {
        let (code, e) = (0..4)
            .map(|k| {
                let e: i32 = (0..3).map(|c| (palette[k][c] - i32::from(t[c])).pow(2)).sum();
                (k as u8, e as u32)
            })
            .min_by_key(|&(_, e)| e)
            .unwrap_or((0, 0));
        codes[i] = code;
        error += e;
    }
    (error, [q0, q1], codes)
}

/// A 5:6:5 colour in 8 bits a channel, as decoders widen it.
fn from_565(c: u16) -> [i32; 3] {
    let (r, g, b) = (i32::from(c >> 11 & 31), i32::from(c >> 5 & 63), i32::from(c & 31));
    [r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2]
}

/// A BC4 block (BC3's alpha): eight values from the largest to the smallest.
fn bc4(values: [u8; 16]) -> [u8; 8] {
    let (lo, hi) = (values.iter().copied().min().unwrap_or(0), values.iter().copied().max().unwrap_or(0));
    let mut out = [0u8; 8];
    out[0] = hi;
    out[1] = lo;
    if lo == hi {
        // Code 0 everywhere is the one value.
        return out;
    }
    let (a0, a1) = (i32::from(hi), i32::from(lo));
    let palette: [i32; 8] =
        std::array::from_fn(|k| match k { 0 => a0, 1 => a1, k => ((8 - k as i32) * a0 + (k as i32 - 1) * a1 + 3) / 7 });
    let bits = values.iter().enumerate().fold(0u64, |acc, (i, &v)| {
        let code = (0..8).min_by_key(|&k| (palette[k] - i32::from(v)).abs()).unwrap_or(0);
        acc | (code as u64) << (3 * i)
    });
    out[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

/// BC7's 4-bit index weights (of 64).
const BC7_WEIGHTS: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// A BC7 block in mode 6: one line in RGBA, ends of 7 bits a channel and a
/// shared low bit each, 16 steps between them.
fn bc7(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let fl: [Texel; 16] = std::array::from_fn(|i| texels[i].map(f32::from));
    let (mut e0, mut e1) = line(&fl);
    let mut best = bc7_try(texels, e0, e1);
    for _ in 0..2 {
        let t = best.codes.map(|i| BC7_WEIGHTS[usize::from(i)] as f32 / 64.0);
        let Some(ends) = fit_ends(&fl, &t) else {
            break;
        };
        (e0, e1) = ends;
        let next = bc7_try(texels, e0, e1);
        if next.error >= best.error {
            break;
        }
        best = next;
    }
    let Bc7 { mut ends, mut low, mut codes, .. } = best;
    // The first texel's code has three bits stored: below 8, or the ends
    // swap (the weights of codes i and 15 - i add up to 64).
    if codes[0] >= 8 {
        ends.swap(0, 1);
        low.swap(0, 1);
        codes = codes.map(|c| 15 - c);
    }
    let mut bits = 0u128;
    let mut at = 0;
    let mut put = |value: u128, n: u32| {
        bits |= value << at;
        at += n;
    };
    put(1 << 6, 7);
    for c in 0..4 {
        put(u128::from(ends[0][c]), 7);
        put(u128::from(ends[1][c]), 7);
    }
    put(u128::from(low[0]), 1);
    put(u128::from(low[1]), 1);
    put(u128::from(codes[0]), 3);
    for &c in &codes[1..] {
        put(u128::from(c), 4);
    }
    bits.to_le_bytes()
}

struct Bc7 {
    error: u32,
    /// The ends' 7-bit channels and low bits.
    ends: [[u8; 4]; 2],
    low: [u8; 2],
    codes: [u8; 16],
}

/// Of the four choices of low bits, the one that codes the block with
/// least error with ends nearest `e0` and `e1`.
fn bc7_try(texels: &[[u8; 4]; 16], e0: Texel, e1: Texel) -> Bc7 {
    let mut best: Option<Bc7> = None;
    for low in [[0u8, 0], [0, 1], [1, 0], [1, 1]] {
        let q = |e: Texel, p: u8| e.map(|v| ((v - f32::from(p)) / 2.0).round().clamp(0.0, 127.0) as u8);
        let ends = [q(e0, low[0]), q(e1, low[1])];
        let v0 = ends[0].map(|c| i32::from(c << 1 | low[0]));
        let v1 = ends[1].map(|c| i32::from(c << 1 | low[1]));
        let palette: [[i32; 4]; 16] = std::array::from_fn(|k| {
            let w = BC7_WEIGHTS[k];
            std::array::from_fn(|c| ((64 - w) * v0[c] + w * v1[c] + 32) >> 6)
        });
        let mut codes = [0u8; 16];
        let mut error = 0u32;
        for (i, t) in texels.iter().enumerate() {
            let (code, e) = (0..16)
                .map(|k| {
                    let e: i32 = (0..4).map(|c| (palette[k][c] - i32::from(t[c])).pow(2)).sum();
                    (k as u8, e as u32)
                })
                .min_by_key(|&(_, e)| e)
                .unwrap_or((0, 0));
            codes[i] = code;
            error += e;
        }
        if best.as_ref().is_none_or(|b| error < b.error) {
            best = Some(Bc7 { error, ends, low, codes });
        }
    }
    best.unwrap_or(Bc7 { error: u32::MAX, ends: [[0; 4]; 2], low: [0; 2], codes: [0; 16] })
}

/// Whether a draw into a target of `width` x `height` through `viewport`
/// (`[x, y, width, height]`) covers most of the screen: a target of at least
/// 1280x720, three quarters of it inside the viewport.
#[cfg(feature = "diagnostics")]
pub(crate) fn covers_screen(width: u64, height: u32, viewport: [f32; 4]) -> bool {
    let (w, h) = (width as f32, height as f32);
    if width < 1280 || height < 720 {
        return false;
    }
    let [x, y, vw, vh] = viewport;
    let across = ((x + vw).min(w) - x.max(0.0)).max(0.0);
    let down = ((y + vh).min(h) - y.max(0.0)).max(0.0);
    across * down >= 0.75 * w * h
}

/// Whether the log is on (`VICTORY_TEXTURE_OVERRIDE_LOG=1` or `2`).
#[cfg(feature = "diagnostics")]
pub(crate) fn logging() -> bool {
    log::level() != 0
}

/// A draw over most of the screen reads title texture `resource`: its
/// identity is logged if it is large and has not been (diagnostics builds,
/// `VICTORY_TEXTURE_OVERRIDE_LOG`). Nothing is made.
#[cfg(feature = "diagnostics")]
pub(crate) fn drawn_over_screen(resource: usize) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    if log::level() == 0 {
        return;
    }
    let Some(texture) = crate::resource::host_made(resource as crate::object::This) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.DepthOrArraySize != 1 || d.Width < 512 || d.Height < 512 {
        return;
    }
    let lines = log::seen().drawn(resource, crate::frame::current());
    for line in lines {
        tracing::info!("{line}");
    }
}

/// Title texture `resource` was detiled and its top level has identity `id`
/// (diagnostics builds: for the log).
#[cfg(feature = "diagnostics")]
pub(crate) fn detiled(resource: usize, id: &Identity) {
    let level = log::level();
    if level == 0 {
        return;
    }
    let lines = log::seen().detiled(resource, *id, crate::frame::current(), level >= 2);
    for line in lines {
        tracing::info!("{line}");
    }
}

/// The title released `resource`.
#[cfg(feature = "diagnostics")]
pub(crate) fn forget(resource: usize) {
    if log::level() != 0 {
        log::seen().forget(resource);
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) mod log {
    use super::{Identity, Shape};
    use std::collections::{HashMap, HashSet};

    /// Pictures of one texture logged at most: past them it is a movie's
    /// surface (or a slideshow's), said once.
    pub(crate) const MOST_PICTURES: u32 = 32;
    /// Lines of uploads (`VICTORY_TEXTURE_OVERRIDE_LOG=2`) at most.
    const MOST_UPLOADS: usize = 2000;
    /// A detiling counts as the one a draw over the screen read if it comes
    /// at most this many frames after the draw (it runs at the submission
    /// of the draw's list).
    const DRAW_TO_DETILING: u64 = 1;

    /// `VICTORY_TEXTURE_OVERRIDE_LOG`: 0 (off), 1, or 2 (uploads too).
    pub(super) fn level() -> u8 {
        static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *LEVEL.get_or_init(|| match std::env::var("VICTORY_TEXTURE_OVERRIDE_LOG").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        })
    }

    /// Whether a texture of `shape` is hashed for the log.
    pub(super) fn wants(shape: &Shape) -> bool {
        level() != 0 && shape.width >= 512 && shape.height >= 512
    }

    pub(super) fn seen() -> std::sync::MutexGuard<'static, Seen> {
        static SEEN: std::sync::LazyLock<std::sync::Mutex<Seen>> = std::sync::LazyLock::new(Default::default);
        SEEN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What was logged, and what is known.
    #[derive(Default)]
    pub(crate) struct Seen {
        /// Each texture's identity at its last detiling.
        identity: HashMap<usize, Identity>,
        /// The textures a draw over most of the screen read: the frame one
        /// last did, and the pictures of each logged.
        drawn: HashMap<usize, (u64, u32)>,
        /// Identities logged as drawn, and as uploaded.
        logged: HashSet<Identity>,
        uploaded: HashSet<Identity>,
    }

    impl Seen {
        /// A draw over most of the screen reads `resource` at `frame`: what
        /// it holds now, if known, is logged.
        pub(crate) fn drawn(&mut self, resource: usize, frame: u64) -> Vec<String> {
            self.drawn.entry(resource).or_insert((frame, 0)).0 = frame;
            match self.identity.get(&resource).copied() {
                Some(id) => self.picture(resource, id, frame),
                None => Vec::new(),
            }
        }

        /// `resource` was detiled at `frame` and is `id` now: logged as drawn
        /// if a draw over most of the screen read it just before.
        pub(crate) fn detiled(&mut self, resource: usize, id: Identity, frame: u64, uploads: bool) -> Vec<String> {
            self.identity.insert(resource, id);
            let mut lines = Vec::new();
            if uploads && self.uploaded.len() < MOST_UPLOADS && self.uploaded.insert(id) {
                lines.push(format!("texture pack: {id} ({}) uploaded, frame {frame}", id.described()));
            }
            let read = self.drawn.get(&resource).is_some_and(|&(at, _)| at <= frame && frame - at <= DRAW_TO_DETILING);
            if read {
                lines.extend(self.picture(resource, id, frame));
            }
            lines
        }

        fn picture(&mut self, resource: usize, id: Identity, frame: u64) -> Vec<String> {
            if self.logged.contains(&id) {
                return Vec::new();
            }
            let Some((_, pictures)) = self.drawn.get_mut(&resource) else {
                return Vec::new();
            };
            if *pictures > MOST_PICTURES {
                return Vec::new();
            }
            *pictures += 1;
            if *pictures > MOST_PICTURES {
                return vec![format!(
                    "texture pack: a texture of {} drawn over most of the screen has shown more than {MOST_PICTURES} pictures by frame {frame}: a movie's surface or a slideshow's; its next pictures are not logged",
                    id.described()
                )];
            }
            self.logged.insert(id);
            vec![format!("texture pack: {id} ({}) drawn over most of the screen, frame {frame}", id.described())]
        }

        /// The title released `resource` (its address may be another's next).
        pub(crate) fn forget(&mut self, resource: usize) {
            self.identity.remove(&resource);
            self.drawn.remove(&resource);
        }
    }
}

#[cfg(test)]
mod tests;
