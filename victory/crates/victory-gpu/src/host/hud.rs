//! The live panel: with `VICTORY_LIVE_HUD` on (a live switch, off by default), a small box of
//! text in the presented frame's top-left corner shows the live switches in force, the last
//! change and the frame rate.
//!
//! The text is built at most once a second ([`lines`]) from the registry's pairs and defaults,
//! the last change and the latest stats sample; a 5 by 7 font written in this file draws it
//! ([`raster`]); and the list that presents the frame copies it onto the back buffer ([`hook`]),
//! from a texture of the back buffer's own format, refilled only when the text changes. The
//! present path leaves the back buffer in the copy-destination state for its own copy of the
//! frame and puts it back in the present state after this, so the panel needs no transition of
//! its own for it.
//!
//! With the switch off the hook returns before reading or making anything. With it on and the
//! text unchanged, a frame costs one copy of the panel's box and one lock of its state: no
//! allocation, no wait on the card and no pipeline.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{Host, HostList, transition};

/// A glyph's pixels: 5 columns by 7 rows in a 6 by 8 cell, so letters have a pixel of space.
const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;
const CELL_W: u32 = 6;
const CELL_H: u32 = 8;
/// The box's margin round the text, in pixels at scale 1.
const PAD: u32 = 2;
/// How far the box's corner sits from the back buffer's, in pixels.
const INSET: u32 = 8;
/// The panel's limits: 44 characters a line, 14 lines.
const MAX_COLS: usize = 44;
const MAX_LINES: usize = 14;
/// A change is shown for this long after it was applied.
const SHOWN: Duration = Duration::from_secs(20);
/// The text is built at most this often.
const REFRESH: Duration = Duration::from_secs(1);
/// The box, its text, and the line of the last change, in RGBA.
const BOX: [u8; 4] = [12, 14, 18, 255];
const TEXT: [u8; 4] = [232, 234, 238, 255];
const CHANGED: [u8; 4] = [255, 192, 64, 255];
/// The back buffer formats the panel is drawn in, as the present path's `swap_format` gives them:
/// the two 8-bit layouts and the 10-bit one.
const R8G8B8A8: i32 = 28;
const B8G8R8A8: i32 = 87;
const R10G10B10A2: i32 = 24;

/// The font, written here and copied from nowhere: 5 by 7, a row to a byte, the leftmost column
/// the highest bit. `A`-`Z`, `0`-`9`, `_`, `.`, `:`, `-`, `>`, `/`, `%`, `(`, `)`, and the space
/// last. Lower case shows its upper case.
const GLYPHS: [[u8; GLYPH_H as usize]; 46] = [
    [
        0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
    ], // A
    [
        0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
    ], // B
    [
        0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
    ], // C
    [
        0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
    ], // D
    [
        0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
    ], // E
    [
        0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
    ], // F
    [
        0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
    ], // G
    [
        0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
    ], // H
    [
        0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ], // I
    [
        0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
    ], // J
    [
        0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
    ], // K
    [
        0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
    ], // L
    [
        0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
    ], // M
    [
        0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
    ], // N
    [
        0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
    ], // O
    [
        0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
    ], // P
    [
        0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
    ], // Q
    [
        0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
    ], // R
    [
        0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
    ], // S
    [
        0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
    ], // T
    [
        0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
    ], // U
    [
        0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
    ], // V
    [
        0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
    ], // W
    [
        0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
    ], // X
    [
        0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
    ], // Y
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
    ], // Z
    [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ], // 0
    [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ], // 1
    [
        0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
    ], // 2
    [
        0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
    ], // 3
    [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ], // 4
    [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ], // 5
    [
        0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ], // 6
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ], // 7
    [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ], // 8
    [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
    ], // 9
    [
        0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
    ], // _
    [
        0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
    ], // .
    [
        0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000,
    ], // :
    [
        0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
    ], // -
    [
        0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000,
    ], // >
    [
        0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
    ], // /
    [
        0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
    ], // %
    [
        0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
    ], // (
    [
        0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
    ], // )
    [
        0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
    ], // the space
];

/// The glyph of `c`: lower case shows its upper case, and a character the font does not name has
/// none.
fn glyph(c: char) -> Option<[u8; GLYPH_H as usize]> {
    let c = c.to_ascii_uppercase();
    let index = match c {
        'A'..='Z' => c as u8 - b'A',
        '0'..='9' => 26 + (c as u8 - b'0'),
        '_' => 36,
        '.' => 37,
        ':' => 38,
        '-' => 39,
        '>' => 40,
        '/' => 41,
        '%' => 42,
        '(' => 43,
        ')' => 44,
        ' ' => 45,
        _ => return None,
    };
    Some(GLYPHS[usize::from(index)])
}

/// The latest stats sample, as the panel's first line shows it.
#[derive(Clone, Copy)]
pub(crate) struct Sample {
    pub(crate) frame_ms: f64,
    pub(crate) fps: f64,
}

/// The latest sample's two numbers, as bits, and whether one has been taken.
static SAMPLE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static SAMPLE_TAKEN: AtomicBool = AtomicBool::new(false);

/// Keeps the latest stats sample for the panel's first line; nothing with the switch off, which
/// is what the stats file's write calls.
pub(crate) fn note_sample(frame_ms: f64, fps: f64) {
    if !crate::live::live_hud() {
        return;
    }
    SAMPLE[0].store(frame_ms.to_bits(), Ordering::Relaxed);
    SAMPLE[1].store(fps.to_bits(), Ordering::Relaxed);
    SAMPLE_TAKEN.store(true, Ordering::Relaxed);
}

/// The latest stats sample, if the first second has passed.
fn sample() -> Option<Sample> {
    SAMPLE_TAKEN.load(Ordering::Relaxed).then(|| Sample {
        frame_ms: f64::from_bits(SAMPLE[0].load(Ordering::Relaxed)),
        fps: f64::from_bits(SAMPLE[1].load(Ordering::Relaxed)),
    })
}

/// The panel's text: the frame rate, every switch away from its default, `ALL DEFAULTS` when
/// there is none, and the last change for 20 seconds after it. Pure: its inputs are all it reads.
fn lines(
    pairs: &[(&'static str, String)],
    defaults: &[(&'static str, String)],
    last: Option<&crate::live::Change>,
    sample: Option<Sample>,
    now: Instant,
) -> Vec<String> {
    let mut text = Vec::with_capacity(MAX_LINES);
    text.push(match sample {
        Some(sample) => fit(format!(
            "LIVE  {:.1} MS  {:.0} FPS",
            sample.frame_ms, sample.fps
        )),
        None => "LIVE  - MS  - FPS".to_owned(),
    });
    let at_default = |name: &str, value: &str| {
        defaults
            .iter()
            .any(|(named, default)| *named == name && default == value)
    };
    let mut shown = false;
    for (name, value) in pairs {
        if at_default(name, value) {
            continue;
        }
        shown = true;
        text.push(fit(format!("{} {}", short(name), value)));
    }
    if !shown {
        text.push("ALL DEFAULTS".to_owned());
    }
    if let Some(last) = last {
        let age = now.saturating_duration_since(last.at);
        if age <= SHOWN {
            text.push(fit(format!(
                "> {} {} -> {}  {}S AGO",
                short(last.name),
                last.old,
                last.new,
                age.as_secs()
            )));
        }
    }
    text.truncate(MAX_LINES);
    text
}

/// The name the panel shows: the registry's, without `VICTORY_`.
fn short(name: &str) -> &str {
    name.strip_prefix("VICTORY_").unwrap_or(name)
}

/// `text` as a line: at most [`MAX_COLS`] characters.
fn fit(text: String) -> String {
    if text.chars().count() <= MAX_COLS {
        return text;
    }
    text.chars().take(MAX_COLS).collect()
}

/// The panel's pixels for `text`: a dark box, the text in light, a line that begins with `>` (the
/// last change's) in the second colour, four bytes to a pixel (RGBA) and `width * height * 4`
/// bytes long. Pure.
fn raster(lines: &[String], scale: u32) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let rows = lines.len().min(MAX_LINES);
    let cols = lines
        .iter()
        .take(rows)
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
        .min(MAX_COLS);
    let width = (cols as u32 * CELL_W + 2 * PAD) * scale;
    let height = (rows as u32 * CELL_H + 2 * PAD) * scale;
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&BOX);
    }
    for (row, line) in lines.iter().take(rows).enumerate() {
        let colour = if line.starts_with('>') { CHANGED } else { TEXT };
        for (col, c) in line.chars().take(MAX_COLS).enumerate() {
            let Some(bits_of) = glyph(c) else {
                continue;
            };
            for (gy, &bits) in bits_of.iter().enumerate() {
                for gx in 0..GLYPH_W {
                    if bits & (1 << (GLYPH_W - 1 - gx)) == 0 {
                        continue;
                    }
                    let x = (PAD + col as u32 * CELL_W + gx) * scale;
                    let y = (PAD + row as u32 * CELL_H + gy as u32) * scale;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            put(&mut pixels, width, x + dx, y + dy, colour);
                        }
                    }
                }
            }
        }
    }
    (width, height, pixels)
}

/// One pixel of `pixels` (RGBA) set to `colour`.
fn put(pixels: &mut [u8], width: u32, x: u32, y: u32, colour: [u8; 4]) {
    let at = (y as usize * width as usize + x as usize) * 4;
    pixels[at..at + 4].copy_from_slice(&colour);
}

/// One pixel's bytes in a texture of `format`, from RGBA, or `None` for a format the panel is not
/// drawn in.
fn pixel_bytes(format: DXGI_FORMAT, [r, g, b, a]: [u8; 4]) -> Option<[u8; 4]> {
    Some(match format.0 {
        R8G8B8A8 => [r, g, b, a],
        B8G8R8A8 => [b, g, r, a],
        R10G10B10A2 => {
            let wide = |value: u8| u32::from(value) * 1023 / 255;
            let word =
                wide(r) | (wide(g) << 10) | (wide(b) << 20) | ((u32::from(a) * 3 / 255) << 30);
            word.to_le_bytes()
        }
        _ => return None,
    })
}

/// Whether the panel can be drawn in `format`.
fn encodable(format: DXGI_FORMAT) -> bool {
    [R8G8B8A8, B8G8R8A8, R10G10B10A2].contains(&format.0)
}

/// Says once that the back buffer's format is not one the panel has.
fn warn_unsupported(format: DXGI_FORMAT) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            format = format.0,
            "live panel: the back buffer's format is not one the panel is drawn in (8-bit and 10-bit colour only); nothing is drawn"
        );
    }
}

/// The panel's scale: 2 at 1440p and below, 3 above.
fn scale_for(height: u32) -> u32 {
    if height > 1440 { 3 } else { 2 }
}

/// What the panel is drawn on: the host's device, the host list presenting the frame (with the
/// back buffer in the copy-destination state; its upload buffers are held until it resets), and
/// the back buffer's format and size.
struct Frame<'a> {
    host: &'a Host,
    list: &'a HostList,
    back: &'a ID3D12Resource,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
}

/// The panel's texture and the text it holds, and when that text was built.
struct Panel {
    format: DXGI_FORMAT,
    scale: u32,
    texture: ID3D12Resource,
    /// The upload buffer the texture was last filled from. The list that recorded the fill
    /// holds it until its fence-safe reset, so replacing this reference (or the whole panel)
    /// cannot free a buffer the card may still read; it is kept here to know the texture is
    /// filled.
    upload: Option<ID3D12Resource>,
    text: Vec<String>,
    /// The box the texture holds, in pixels; 0 when the panel did not fit and nothing is drawn.
    width: u32,
    height: u32,
    built: Instant,
}

static PANEL: Mutex<Option<Panel>> = Mutex::new(None);

/// Draws the panel on the frame when the switch is on; `false`, with nothing read or touched,
/// when it is off.
pub(crate) fn hook(
    host: &Host,
    list: &HostList,
    back: &ID3D12Resource,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
) -> bool {
    if !crate::live::live_hud() {
        return false;
    }
    paint(&Frame {
        host,
        list,
        back,
        format,
        width,
        height,
    });
    true
}

/// Whether the panel can draw on this frame at all: a format it knows and a device that is not
/// gone.
fn supported(frame: &Frame) -> bool {
    if !encodable(frame.format) {
        warn_unsupported(frame.format);
        return false;
    }
    // SAFETY: a plain query; a removed device is asked nothing more.
    unsafe { frame.host.device.GetDeviceRemovedReason() }.is_ok()
}

/// The panel's frame: the text is built and drawn again when a second has passed, the format or
/// the scale moved, and the box copied every frame otherwise.
fn paint(frame: &Frame) {
    if !supported(frame) {
        return;
    }
    let scale = scale_for(frame.height);
    let now = Instant::now();
    let due = {
        let panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
        panel.as_ref().is_none_or(|panel| {
            panel.format != frame.format
                || panel.scale != scale
                || now.saturating_duration_since(panel.built) >= REFRESH
        })
    };
    if due && let Some(text) = text_now(now) {
        draw(frame, text);
        return;
    }
    // Not due, or a recording thread was scanning a registry list: copy the text the panel
    // already holds, if it is in this frame's format and scale and fits.
    let panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(panel) = panel.as_ref().filter(|panel| {
        panel.format == frame.format && panel.scale == scale && fits(panel, frame)
    }) else {
        return;
    };
    record(frame, panel);
}

/// The panel's text now, or `None` while a recording thread holds a registry list (the
/// presenting thread must not wait for a scan of a long skip list; the last text stays).
fn text_now(now: Instant) -> Option<Vec<String>> {
    let pairs = crate::live::pairs_try()?;
    let defaults = crate::live::defaults();
    let last = crate::live::last_change();
    Some(lines(&pairs, &defaults, last.as_ref(), sample(), now))
}

/// The panel's work for `text`: the texture made or filled when the text, the format or the scale
/// changed, then the box copied onto the back buffer. `false` when nothing was drawn.
fn draw(frame: &Frame, text: Vec<String>) -> bool {
    if !supported(frame) {
        return false;
    }
    let scale = scale_for(frame.height);
    let now = Instant::now();
    let mut panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
    if panel
        .as_ref()
        .is_none_or(|panel| panel.format != frame.format || panel.scale != scale)
    {
        let Some(texture) = make_texture(frame, scale) else {
            return false;
        };
        if let Some(old) = panel.take() {
            // A list sent ahead may still name the old texture; the present path drains such
            // lists before a resize or format change, and this holds it until this list resets
            // as the same rule for every deferred resource.
            frame.list.keep(&old.texture);
        }
        *panel = Some(Panel {
            format: frame.format,
            scale,
            texture,
            upload: None,
            text: Vec::new(),
            width: 0,
            height: 0,
            built: now,
        });
    }
    let panel = panel.as_mut().expect("the panel was just made");
    panel.built = now;
    let (width, height, pixels) = raster(&text, scale);
    if panel.text != text
        || panel.width != width
        || panel.height != height
        || panel.upload.is_none()
    {
        let Some(upload) = fill(frame, panel, width, height, &pixels) else {
            panel.width = 0;
            panel.height = 0;
            return false;
        };
        panel.upload = Some(upload);
        panel.text = text;
        panel.width = width;
        panel.height = height;
    }
    // The box the texture holds stays recorded even when it does not fit this frame: growing
    // the buffer back shows the panel on the next present, not at the next one-second refresh.
    if !fits(panel, frame) {
        return false;
    }
    record(frame, panel);
    true
}

/// Whether the panel's box fits the back buffer with its corner's inset.
fn fits(panel: &Panel, frame: &Frame) -> bool {
    panel.width > 0
        && panel.height > 0
        && INSET + panel.width <= frame.width
        && INSET + panel.height <= frame.height
}

/// The largest panel the limits allow at `scale`, in the back buffer's format and the copy-source
/// state: text of any size fits it, and it is made again only when the format or the scale moves.
fn make_texture(frame: &Frame, scale: u32) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from((MAX_COLS as u32 * CELL_W + 2 * PAD) * scale),
        Height: (MAX_LINES as u32 * CELL_H + 2 * PAD) * scale,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: frame.format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a texture; it starts as a copy source, which is what it is read
    // as until its first fill moves it.
    let made = unsafe {
        frame.host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            None,
            &mut texture,
        )
    };
    match made {
        Ok(()) => texture,
        Err(e) => {
            tracing::warn!(error = %e, "live panel: its texture could not be made; nothing is drawn");
            None
        }
    }
}

/// Copies `pixels` (RGBA, `width` by `height`) into `panel`'s texture through an upload buffer,
/// in the back buffer's format, and gives the texture back in the copy-source state. The
/// recording list holds the upload until its fence-safe reset, so a later fill may replace the
/// panel's own reference while the card still reads the old one. The upload comes back for the
/// panel to know when its texture is filled.
fn fill(
    frame: &Frame,
    panel: &Panel,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Option<ID3D12Resource> {
    // SAFETY: a plain query of a live texture.
    let desc = unsafe { panel.texture.GetDesc() };
    let mut placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: outputs for one subresource of a live texture.
    unsafe {
        frame.host.device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut placed),
            None,
            None,
            Some(&mut total),
        );
    }
    if placed.Footprint.Width < width || placed.Footprint.Height < height {
        return None;
    }
    let (upload, mapped) = frame.list.upload(frame.host, total)?;
    // SAFETY: a mapped upload buffer of `total` bytes just made; the rows written are inside the
    // footprint, each `width` pixels of four bytes at its row's start.
    unsafe {
        let base = mapped.cast::<u8>().add(placed.Offset as usize);
        for row in 0..height as usize {
            let to = base.add(row * placed.Footprint.RowPitch as usize);
            for col in 0..width as usize {
                let at = (row * width as usize + col) * 4;
                let rgba = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];
                let pixel = pixel_bytes(panel.format, rgba)?;
                std::ptr::copy_nonoverlapping(pixel.as_ptr(), to.add(col * 4), 4);
            }
        }
        upload.Unmap(0, None);
    }
    let to = subresource(&panel.texture);
    let from = rows(&upload, placed);
    let boxed = D3D12_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: width,
        bottom: height,
        back: 1,
    };
    // SAFETY: the presenting list, the panel's texture and the upload's rows; the box is inside
    // both; the list is the only writer of the texture and the state moves and returns.
    unsafe {
        frame.list.list.ResourceBarrier(&[transition(
            &panel.texture,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        frame
            .list
            .list
            .CopyTextureRegion(&to, 0, 0, 0, &from, Some(&boxed));
        frame.list.list.ResourceBarrier(&[transition(
            &panel.texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
    Some(upload)
}

/// Copies the panel's box onto the back buffer's top-left corner. The back buffer is in the state
/// the present path copies the frame in (copy destination), and it puts it back to present after.
fn record(frame: &Frame, panel: &Panel) {
    let to = subresource(frame.back);
    let from = subresource(&panel.texture);
    let boxed = D3D12_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: panel.width,
        bottom: panel.height,
        back: 1,
    };
    // SAFETY: the presenting list; both textures live; the box is inside both; the panel's
    // texture is in the copy-source state and the back buffer in the copy-destination one.
    unsafe {
        frame
            .list
            .list
            .CopyTextureRegion(&to, INSET, INSET, 0, &from, Some(&boxed));
    }
}

/// A copy location naming the whole of a texture's one subresource.
fn subresource(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    }
}

/// A copy location naming an upload buffer's rows.
fn rows(
    upload: &ID3D12Resource,
    placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(upload) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: placed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panel is one static: the tests that draw on it take turns.
    static PANEL_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The registry's defaults: one name and value a switch, every switch at its default.
    fn at_defaults() -> Vec<(&'static str, String)> {
        crate::live::defaults()
    }

    /// `pairs` with the switch called `name` set to `value`.
    fn set(pairs: &mut [(&'static str, String)], name: &str, value: &str) {
        let (_, held) = pairs
            .iter_mut()
            .find(|(named, _)| *named == name)
            .expect("a switch of the registry");
        *held = value.to_owned();
    }

    /// With every switch at its default the panel says so, under the frame rate.
    #[test]
    fn no_switches_shows_the_rate_and_all_defaults() {
        let pairs = at_defaults();
        let sample = Sample {
            frame_ms: 16.7,
            fps: 60.0,
        };
        assert_eq!(
            lines(&pairs, &pairs, None, Some(sample), Instant::now()),
            ["LIVE  16.7 MS  60 FPS", "ALL DEFAULTS"]
        );
        // Before the first second there is no sample: dashes, not a wrong rate.
        assert_eq!(
            lines(&pairs, &pairs, None, None, Instant::now())[0],
            "LIVE  - MS  - FPS"
        );
    }

    /// A switch away from its default is one line, `NAME VALUE`, the `VICTORY_` gone.
    #[test]
    fn switches_away_from_their_defaults_are_shown_without_the_prefix() {
        let defaults = at_defaults();
        let mut pairs = defaults.clone();
        set(&mut pairs, "VICTORY_TEXTURE_LOW", "60");
        set(&mut pairs, "LIVE_SKIP_DRAW", "a,b");
        let sample = Sample {
            frame_ms: 8.0,
            fps: 120.0,
        };
        assert_eq!(
            lines(&pairs, &defaults, None, Some(sample), Instant::now()),
            [
                // The registry's one table is alphabetical.
                "LIVE  8.0 MS  120 FPS",
                "LIVE_SKIP_DRAW a,b",
                "TEXTURE_LOW 60"
            ]
        );
    }

    /// A change is a line of its own for 20 seconds after it was applied, and gone after that.
    #[test]
    fn a_fresh_change_is_shown_with_its_age_and_an_old_one_is_not() {
        let defaults = at_defaults();
        let mut pairs = defaults.clone();
        set(&mut pairs, "VICTORY_TEXTURE_LOW", "60");
        let now = Instant::now();
        let change = crate::live::Change {
            name: "VICTORY_TEXTURE_LOW",
            old: "70".to_owned(),
            new: "60".to_owned(),
            at: now - Duration::from_secs(12),
        };
        let fresh = lines(&pairs, &defaults, Some(&change), None, now);
        assert_eq!(
            fresh.last().map(String::as_str),
            Some("> TEXTURE_LOW 70 -> 60  12S AGO")
        );
        // At exactly 20 seconds it still shows; a second later it is gone.
        let last = lines(&pairs, &defaults, Some(&change), None, change.at + SHOWN);
        assert_eq!(
            last.last().map(String::as_str),
            Some("> TEXTURE_LOW 70 -> 60  20S AGO")
        );
        let gone = lines(
            &pairs,
            &defaults,
            Some(&change),
            None,
            change.at + SHOWN + REFRESH,
        );
        assert!(!gone.iter().any(|line| line.starts_with('>')));
    }

    /// The limits hold: 14 lines, each 44 characters, whatever the values are.
    #[test]
    fn lines_are_at_most_44_characters_and_14_lines() {
        let defaults = at_defaults();
        // More switches than lines, with values longer than a line.
        let pairs: Vec<(&'static str, String)> = (0..20)
            .map(|i| (defaults[i % defaults.len()].0, "X".repeat(90)))
            .collect();
        let now = Instant::now();
        let change = crate::live::Change {
            name: "VICTORY_TEXTURE_LOW",
            old: "O".repeat(60),
            new: "N".repeat(60),
            at: now,
        };
        let sample = Sample {
            frame_ms: 1234.5,
            fps: 12345.0,
        };
        let text = lines(&pairs, &defaults, Some(&change), Some(sample), now);
        assert!(text.len() <= MAX_LINES, "{} lines", text.len());
        assert!(text.iter().all(|line| line.chars().count() <= MAX_COLS));
        assert!(text[0].chars().count() <= MAX_COLS);
    }

    /// The box's size for a known text, and the pixels a glyph lights.
    #[test]
    fn raster_sizes_a_known_text_and_sets_the_pixels_of_a_glyph() {
        let (width, height, pixels) = raster(&["AB".to_owned(), "C".to_owned()], 2);
        assert_eq!(
            (width, height),
            ((2 * CELL_W + 2 * PAD) * 2, (2 * CELL_H + 2 * PAD) * 2)
        );
        assert_eq!(pixels.len(), (width * height * 4) as usize);

        // Scale 1: the box's corner is dark; `A`'s first row lights its middle three pixels.
        let (width, _, pixels) = raster(&["A".to_owned()], 1);
        let pixel = |x: u32, y: u32| {
            let at = (y * width + x) as usize * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        assert_eq!(pixel(0, 0), BOX, "the box's corner");
        assert_eq!(pixel(PAD, PAD), BOX, "A's first row, first column");
        assert_eq!(pixel(PAD + 1, PAD), TEXT, "A's first row, second column");

        // A line that begins with `>` is the changed line, in the second colour.
        let (width, _, pixels) = raster(&[">".to_owned()], 1);
        let at = ((PAD * width + PAD) * 4) as usize;
        assert_eq!(
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]],
            CHANGED
        );
    }

    /// Every character the font names has a glyph, and lower case shows its upper case.
    #[test]
    fn every_character_the_font_names_has_a_glyph() {
        let named = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-/>%() ";
        assert_eq!(named.chars().count(), GLYPHS.len());
        for c in named.chars() {
            assert!(glyph(c).is_some(), "no glyph for {c:?}");
            assert_eq!(
                glyph(c.to_ascii_lowercase()),
                glyph(c),
                "lower case {c:?} shows its upper case"
            );
        }
        assert!(glyph('?').is_none(), "not in the font");
        assert_eq!(raster(&["a".to_owned()], 1), raster(&["A".to_owned()], 1));
    }

    /// The pixels are encoded for each of the formats the panel draws in.
    #[test]
    fn pixels_are_encoded_for_the_three_formats() {
        let rgba = [10, 20, 30, 255];
        assert_eq!(
            pixel_bytes(DXGI_FORMAT_R8G8B8A8_UNORM, rgba),
            Some([10, 20, 30, 255])
        );
        assert_eq!(
            pixel_bytes(DXGI_FORMAT_B8G8R8A8_UNORM, rgba),
            Some([30, 20, 10, 255])
        );
        let wide = pixel_bytes(DXGI_FORMAT_R10G10B10A2_UNORM, rgba).expect("10 bits");
        let word = u32::from_le_bytes(wide);
        assert_eq!(word & 1023, 10 * 1023 / 255);
        assert_eq!((word >> 10) & 1023, 20 * 1023 / 255);
        assert_eq!((word >> 20) & 1023, 30 * 1023 / 255);
        assert_eq!(word >> 30, 3);
        assert_eq!(
            pixel_bytes(DXGI_FORMAT_R16G16B16A16_FLOAT, rgba),
            None,
            "16-bit float is not one the panel draws in"
        );
    }

    /// With the switch off the hook returns before reading or recording anything.
    #[test]
    fn with_the_switch_off_the_hook_returns_before_touching_anything() {
        if crate::live::live_hud() {
            return; // a run with the switch on is not this test
        }
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let back = make_target(host, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16).expect("a target");
        let list = host.new_list().expect("a list");
        assert!(
            !hook(host, &list, &back, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16),
            "the switch is off"
        );
        assert!(list.untouched(), "nothing was recorded");
    }

    /// A texture of `format` and the given size, in the copy-destination state a present leaves
    /// the back buffer in.
    fn make_target(
        host: &Host,
        format: DXGI_FORMAT,
        width: u32,
        height: u32,
    ) -> Option<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a texture.
        unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut texture,
            )
        }
        .ok()?;
        texture
    }

    /// The panel on the software adapter: its box covers the target's corner, and a pixel outside
    /// keeps the target's own colour.
    #[test]
    fn the_panel_is_copied_onto_a_target_and_reads_back() {
        let _alone = PANEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let (width, height) = (320u32, 128u32);
        let target = make_target(host, format, width, height).expect("a target");
        let list = host.new_list().expect("a list");
        // The target's own colour, as the frame copy would leave it.
        let own = [1u8, 2, 3, 255];
        // SAFETY: a plain query of a live texture.
        let desc = unsafe { target.GetDesc() };
        let mut placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: outputs for one subresource of a live texture.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut placed),
                None,
                None,
                Some(&mut total),
            );
        }
        let (fill, mapped) = host.upload_buffer(total).expect("an upload buffer");
        // SAFETY: a mapped upload buffer of `total` bytes just made; every pixel is four bytes.
        unsafe {
            for at in (0..total as usize).step_by(4) {
                std::ptr::copy_nonoverlapping(own.as_ptr(), mapped.cast::<u8>().add(at), 4);
            }
            let to = subresource(&target);
            let from = rows(&fill, placed);
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        let text = vec!["LIVE  1.0 MS  1 FPS".to_owned(), "ALL DEFAULTS".to_owned()];
        let (panel_width, panel_height, _) = raster(&text, 2);
        let frame = Frame {
            host,
            list: &list,
            back: &target,
            format,
            width,
            height,
        };
        assert!(draw(&frame, text), "the panel is drawn");
        let readback = host.read_back_buffer(total).expect("a read-back buffer");
        // SAFETY: the presenting list; live textures; the copy is the whole target.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &target,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = rows(&readback, placed);
            let from = subresource(&target);
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: total as usize,
        };
        // SAFETY: the copies finished (execute waits); the buffer is `total` bytes.
        unsafe {
            readback
                .Map(0, Some(&range), Some(&mut data))
                .expect("the read-back maps");
        }
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), total as usize) };
        let pixel = |x: u32, y: u32| {
            let at = placed.Offset as usize
                + y as usize * placed.Footprint.RowPitch as usize
                + x as usize * 4;
            [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
        };
        assert_eq!(pixel(INSET + 1, INSET + 1), BOX, "inside the box's margin");
        assert!(
            pixel(INSET + panel_width - 1, INSET + panel_height - 1) != own,
            "the box's far corner is the panel's, not the target's"
        );
        assert_eq!(
            pixel(width - 2, height - 2),
            own,
            "outside the box the target keeps its colour"
        );
        // SAFETY: a mapping of ours, no longer read.
        unsafe { readback.Unmap(0, None) };
    }

    /// A present recorded ahead holds its upload until the list's fence-safe reset: the text
    /// changing while that list has not run replaces the panel's reference, not the list's.
    #[test]
    fn an_ahead_present_keeps_the_upload_after_the_text_changes() {
        let _alone = PANEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let target = make_target(host, format, 320, 128).expect("a target");
        let ahead = host.new_list().expect("an ahead list");
        let later = host.new_list().expect("a later list");
        let ahead_frame = Frame {
            host,
            list: &ahead,
            back: &target,
            format,
            width: 320,
            height: 128,
        };
        assert!(draw(&ahead_frame, vec!["ONE".to_owned()]), "the first text is drawn");
        let old = {
            let panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
            panel
                .as_ref()
                .and_then(|panel| panel.upload.clone())
                .expect("the panel's upload")
        };
        // The text changes before the ahead list has run: the panel's own reference goes.
        let later_frame = Frame {
            host,
            list: &later,
            back: &target,
            format,
            width: 320,
            height: 128,
        };
        assert!(draw(&later_frame, vec!["TWO".to_owned()]), "the second text is drawn");
        assert!(
            crate::host::references(&old) > 1,
            "the ahead list the earlier present was recorded in still holds the old upload"
        );
        // It runs after the change, and the upload stays until the list's reset.
        ahead.close();
        host.execute(&[&ahead]);
        assert!(
            crate::host::references(&old) > 1,
            "the list still holds the upload until it is reset"
        );
        ahead.reset();
        assert_eq!(
            crate::host::references(&old),
            1,
            "only this test's reference is left after the reset"
        );
    }

    /// Growing the buffer back after a refresh that found it too small shows the panel at once:
    /// the cached box is kept, so the next present copies it, not the next one-second refresh.
    #[test]
    fn the_panel_returns_as_soon_as_the_buffer_grows_back() {
        let _alone = PANEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        // Room for any text the limits allow (44 characters by 14 lines at scale 2), whatever
        // another test's last change puts in it.
        let big = make_target(host, format, 640, 256).expect("a big target");
        let small = make_target(host, format, 20, 20).expect("a small target");
        let list = host.new_list().expect("a list");
        let big_frame = Frame {
            host,
            list: &list,
            back: &big,
            format,
            width: 640,
            height: 256,
        };
        let small_frame = Frame {
            host,
            list: &list,
            back: &small,
            format,
            width: 20,
            height: 20,
        };
        // The panel at the big size; then the text refresh falls due on the small buffer.
        assert!(
            draw(
                &big_frame,
                vec!["LIVE  1.0 MS  1 FPS".to_owned(), "ALL DEFAULTS".to_owned()]
            ),
            "the panel is drawn on the big buffer"
        );
        {
            let mut panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
            let panel = panel.as_mut().expect("the panel was made and fits the big buffer");
            assert!(panel.width > 0 && panel.height > 0);
            // The small buffer arrives when a refresh is due: the old code cleared the box here.
            panel.built = Instant::now() - REFRESH;
        }
        paint(&small_frame);
        {
            let panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
            let panel = panel.as_ref().expect("the panel is kept while it does not fit");
            assert!(
                panel.width > 0 && panel.height > 0,
                "the box is kept while the buffer is too small"
            );
        }
        paint(&big_frame);
        let panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
        let panel = panel.as_ref().expect("the panel is still there");
        assert!(
            panel.width > 0 && panel.height > 0 && fits(panel, &big_frame),
            "the grown buffer shows the panel at once, not at the next refresh"
        );
    }

    /// With the switch off the hook and the stats sample take no panel lock: they return while
    /// another thread holds them, and record nothing.
    #[test]
    fn with_the_switch_off_the_hook_and_the_sample_take_no_lock() {
        if crate::live::live_hud() {
            return; // a run with the switch on is not this test
        }
        let _alone = PANEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let back = make_target(host, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16).expect("a target");
        let list = host.new_list().expect("a list");
        let timed_out = std::sync::Arc::new(AtomicBool::new(false));
        let (locked, held) = std::sync::mpsc::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let holder = {
            let timed_out = timed_out.clone();
            std::thread::spawn(move || {
                let _panel = PANEL.lock().unwrap_or_else(|e| e.into_inner());
                let _change = crate::live::hold_change_lock();
                locked.send(()).unwrap();
                if released.recv_timeout(Duration::from_secs(10)).is_err() {
                    timed_out.store(true, Ordering::Relaxed);
                }
            })
        };
        held.recv_timeout(Duration::from_secs(10))
            .expect("the panel's locks are held");
        note_sample(16.0, 60.0);
        assert!(
            !hook(host, &list, &back, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16),
            "the switch is off"
        );
        let _ = release.send(());
        holder.join().unwrap();
        assert!(
            !timed_out.load(Ordering::Relaxed),
            "the off path waited on one of the panel's locks"
        );
        assert!(list.untouched(), "nothing was recorded");
    }
}
