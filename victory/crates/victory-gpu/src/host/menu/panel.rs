//! The settings menu drawn on its panel: the scaled art copied, the bar
//! under the row picked, and the lettering blended by its coverage (what
//! `draw` in `scripts/overlay-art/menu_assets.py` does), all on the
//! processor; then written out in the swap chain's format for today's copy.

use windows::Win32::Graphics::Dxgi::Common::*;

use super::art::{self, Lettering, Scaled};

/// A line of the list.
pub(super) enum Entry {
    Heading(&'static str),
    Blank,
    /// A setting (with a value) or an action (without).
    Line { label: &'static str, mark: &'static str, value: String, picked: bool },
}

/// What the panel shows: the list, the picked row's help (one or more
/// lines), a note, the keys, and the repository line.
pub(super) struct Page {
    pub(super) entries: Vec<Entry>,
    pub(super) help: Vec<&'static str>,
    pub(super) note: Option<String>,
    pub(super) keys: &'static str,
    pub(super) repo: &'static str,
}

/// The repository line: in the header, under the title, dark on its light gradient.
const REPO_AT: (f32, f32) = (0.075, 0.236);
const REPO_COLOUR: [u8; 3] = [58, 22, 74];
/// The note (a change, a save): as bright as the labels.
const NOTE_COLOUR: [u8; 3] = art::LABEL;
/// The help and note lines' right edge (as far in as their left edge is from the bar's).
const TEXT_RIGHT: f32 = art::BAR_RIGHT - (art::X_HELP - art::BAR_LEFT);

/// Space before an entry's baseline and after it, as fractions of the height.
fn spacing(entry: &Entry) -> (f32, f32) {
    match entry {
        Entry::Heading(_) => (art::HEADING_GAP_BEFORE, art::HEADING_PITCH),
        Entry::Blank => (0.0, art::ROW_PITCH / 2.0),
        Entry::Line { .. } => (0.0, art::ROW_PITCH),
    }
}

/// The baselines of `entries[lo..=hi]` from the list's top, if the last is at most `bottom` (fractions).
fn baselines(entries: &[Entry], lo: usize, hi: usize, bottom: f32) -> Option<Vec<f32>> {
    let mut y = art::FIRST_BASELINE_Y;
    let mut out = Vec::with_capacity(hi + 1 - lo);
    for entry in &entries[lo..=hi] {
        let (before, after) = spacing(entry);
        let base = y + before;
        if base > bottom + 1e-6 {
            return None;
        }
        out.push(base);
        y = base + after;
    }
    Some(out)
}

/// The entries shown, `lo..=hi`, with `bottom` the lowest baseline the list
/// may use: the picked line with its group's heading above it, then grown a
/// line at a time below and above while they fit, then moved down while a
/// line at the top is without its heading (and the picked line stays).
pub(super) fn window(entries: &[Entry], bottom: f32) -> (usize, usize) {
    let (mut lo, mut hi) = grown(entries, bottom);
    let at = entries.iter().position(|e| matches!(e, Entry::Line { picked: true, .. })).unwrap_or(0);
    while lo > 0
        && hi > at
        && matches!(entries[lo], Entry::Line { .. })
        && matches!(entries[lo - 1], Entry::Heading(_))
        && baselines(entries, lo - 1, hi - 1, bottom).is_some()
    {
        lo -= 1;
        hi -= 1;
    }
    (lo, hi)
}

fn grown(entries: &[Entry], bottom: f32) -> (usize, usize) {
    let at = entries.iter().position(|e| matches!(e, Entry::Line { picked: true, .. })).unwrap_or(0);
    let head = entries[..=at].iter().rposition(|e| !matches!(e, Entry::Line { .. })).unwrap_or(at);
    let (mut lo, mut hi) = (at, at);
    while lo > head && baselines(entries, lo - 1, hi, bottom).is_some() {
        lo -= 1;
    }
    loop {
        let mut grew = false;
        if hi + 1 < entries.len() && baselines(entries, lo, hi + 1, bottom).is_some() {
            hi += 1;
            grew = true;
        }
        if lo > 0 && baselines(entries, lo - 1, hi, bottom).is_some() {
            lo -= 1;
            grew = true;
        }
        if !grew {
            return (lo, hi);
        }
    }
}

/// `text` cut into lines of at most `width` pixels, at spaces.
pub(super) fn wrapped(font: &Lettering, text: &str, width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ') {
        let longer = if line.is_empty() { word.to_owned() } else { format!("{line} {word}") };
        if !line.is_empty() && font.width(&longer) > width {
            lines.push(std::mem::replace(&mut line, word.to_owned()));
        } else {
            line = longer;
        }
    }
    lines.push(line);
    lines
}

/// The lines under the list, top first: the help, then the note.
fn under(scaled: &Scaled, page: &Page) -> Vec<(String, [u8; 3])> {
    let width = (TEXT_RIGHT - art::X_HELP) * scaled.size.0 as f32;
    let mut lines: Vec<(String, [u8; 3])> =
        page.help.iter().flat_map(|h| wrapped(&scaled.small, h, width)).map(|l| (l, art::SOFT)).collect();
    if let Some(note) = &page.note {
        lines.extend(wrapped(&scaled.small, note, width).into_iter().map(|l| (l, NOTE_COLOUR)));
    }
    lines.retain(|(l, _)| !l.is_empty());
    lines
}

/// The lowest baseline the list may use (a fraction) with `n` lines under it.
fn list_bottom(n: usize) -> f32 {
    let pitch = art::KEYS_BASELINE_Y - art::HELP_BASELINE_Y;
    art::LAST_BASELINE_Y - n.saturating_sub(1) as f32 * pitch
}

/// Draws `page` on `canvas` (the panel's size, four bytes a pixel, red first).
pub(super) fn compose(scaled: &Scaled, page: &Page, canvas: &mut Vec<u8>) {
    canvas.clear();
    canvas.extend_from_slice(&scaled.panel);
    let mut c = Canvas { pixels: canvas, size: (scaled.size.0 as usize, scaled.size.1 as usize) };
    let (w, h) = (scaled.size.0 as f32, scaled.size.1 as f32);
    let under = under(scaled, page);
    let (lo, hi) = window(&page.entries, list_bottom(under.len()));
    let bases = baselines(&page.entries, lo, hi, 2.0).unwrap_or_default();
    for (entry, base) in page.entries[lo..=hi].iter().zip(bases) {
        let y = base * h;
        match entry {
            Entry::Heading(text) => c.text(&scaled.small, &text.to_uppercase(), art::X_HEADING * w, y, art::HEADING, false),
            Entry::Blank => {}
            Entry::Line { label, mark, value, picked } => {
                if *picked {
                    let (x0, x1) = (art::BAR_LEFT * w, art::BAR_RIGHT * w);
                    let top = y - art::BAR_TOP_ABOVE_BASELINE * h;
                    c.fill(x0, x1, top, top + art::BAR_HEIGHT * h, art::BAR_FILL);
                    let under = y + art::BAR_UNDERLINE_BELOW_BASELINE * h;
                    c.fill(x0, x1, under, under + (art::BAR_UNDERLINE_HEIGHT * h).max(1.0), art::BAR_UNDERLINE);
                }
                let x = art::X_LABEL * w;
                c.text(&scaled.row, label, x, y, art::LABEL, false);
                if !mark.is_empty() {
                    c.text(&scaled.row, mark, x + scaled.row.width(label), y, art::SOFT, false);
                }
                if value.is_empty() {
                    continue;
                }
                if *picked {
                    let right = art::X_PICKED_VALUE_RIGHT * w;
                    c.text(&scaled.row, value, right, y, art::PICKED_VALUE, true);
                    // The left arrow at its place, or left of a value that reaches past it.
                    let gap = scaled.row.width(" ");
                    let left = (art::X_ARROW_LEFT * w).min(right - scaled.row.width(value) - gap - scaled.row.width("\u{2039}"));
                    c.text(&scaled.row, "\u{2039}", left, y, art::VALUE, false);
                    c.text(&scaled.row, "\u{203a}", art::X_ARROW_RIGHT * w, y, art::VALUE, false);
                } else {
                    c.text(&scaled.row, value, art::X_VALUE_RIGHT * w, y, art::VALUE, true);
                }
            }
        }
    }
    let pitch = (art::KEYS_BASELINE_Y - art::HELP_BASELINE_Y) * h;
    let last = art::HELP_BASELINE_Y * h;
    for (i, (line, colour)) in under.iter().enumerate() {
        let y = last - (under.len() - 1 - i) as f32 * pitch;
        c.text(&scaled.small, line, art::X_HELP * w, y, *colour, false);
    }
    c.text(&scaled.small, page.keys, art::X_HELP * w, art::KEYS_BASELINE_Y * h, art::SOFT, false);
    c.text(&scaled.small, page.repo, REPO_AT.0 * w, REPO_AT.1 * h, REPO_COLOUR, false);
}

/// Where a line's text ends and its value begins, in pixels at the panel's
/// size, for the tests: (label and mark's right end, value's left end).
#[cfg(test)]
pub(super) fn spans(scaled: &Scaled, label: &str, mark: &str, value: &str, picked: bool) -> (f32, f32) {
    let w = scaled.size.0 as f32;
    let end = art::X_LABEL * w + scaled.row.width(label) + scaled.row.width(mark);
    let right = if picked { art::X_PICKED_VALUE_RIGHT } else { art::X_VALUE_RIGHT } * w;
    (end, right - scaled.row.width(value))
}

/// How wide the help and note lines may be, in pixels, and the keys line's width.
#[cfg(test)]
pub(super) fn widths(scaled: &Scaled, page: &Page) -> (f32, f32) {
    ((TEXT_RIGHT - art::X_HELP) * scaled.size.0 as f32, scaled.small.width(page.keys))
}

/// How many lines sit under the list.
#[cfg(test)]
pub(super) fn lines_under(scaled: &Scaled, page: &Page) -> (usize, f32) {
    let n = under(scaled, page).len();
    (n, list_bottom(n))
}

struct Canvas<'a> {
    pixels: &'a mut Vec<u8>,
    size: (usize, usize),
}

impl Canvas<'_> {
    fn fill(&mut self, x0: f32, x1: f32, y0: f32, y1: f32, colour: [u8; 3]) {
        let (w, h) = self.size;
        let clamp = |v: f32, most: usize| (v.round().max(0.0) as usize).min(most);
        let (x0, x1, y0, y1) = (clamp(x0, w), clamp(x1, w), clamp(y0, h), clamp(y1, h));
        for y in y0..y1 {
            for px in self.pixels[(y * w + x0) * 4..(y * w + x1) * 4].chunks_exact_mut(4) {
                px[..3].copy_from_slice(&colour);
            }
        }
    }

    /// `text` with its pen starting at `x` (ending there if `right`), standing on `baseline`.
    fn text(&mut self, font: &Lettering, text: &str, x: f32, baseline: f32, colour: [u8; 3], right: bool) {
        let (w, h) = (self.size.0 as i64, self.size.1 as i64);
        let ((cw, ch), origin_x, base) = font.metrics();
        let mut pen = if right { x - font.width(text) } else { x };
        let dy = baseline.round() as i64 - base.round() as i64;
        for c in text.chars() {
            let Some(i) = font.index(c) else { continue };
            let dx = pen.round() as i64 - origin_x.round() as i64;
            let cover = font.cover(i);
            for gy in 0..ch as i64 {
                let y = dy + gy;
                if !(0..h).contains(&y) {
                    continue;
                }
                let row = &cover[gy as usize * cw..(gy as usize + 1) * cw];
                for (gx, &a) in row.iter().enumerate() {
                    let x = dx + gx as i64;
                    if a == 0 || !(0..w).contains(&x) {
                        continue;
                    }
                    let at = (y * w + x) as usize * 4;
                    let (a, keep) = (u32::from(a), 255 - u32::from(a));
                    for (was, new) in self.pixels[at..at + 3].iter_mut().zip(colour) {
                        *was = ((u32::from(*was) * keep + u32::from(new) * a + 127) / 255) as u8;
                    }
                }
            }
            pen += font.advance(i);
        }
    }
}

/// A half-precision float's bits, for values from 0 to 1.
fn half(v: f32) -> u16 {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7F_FFFF;
    if v <= 0.0 || exp < -10 {
        return 0;
    }
    if exp <= 0 {
        let m = mantissa | 0x80_0000;
        let shift = (14 - exp) as u32;
        return ((m >> shift) + ((m >> (shift - 1)) & 1)) as u16;
    }
    ((exp as u32) << 10 | mantissa >> 13) as u16 + ((mantissa >> 12) & 1) as u16
}

/// The light of an sRGB byte (for the floating-point swap chain, whose values are linear).
fn linear(v: u8) -> f32 {
    let v = f32::from(v) / 255.0;
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Writes `canvas` (`size`, four bytes a pixel, red first) into `out`, rows
/// `pitch` bytes apart, as pixels of `format`. False for a format it does not write.
pub(super) fn write_out(canvas: &[u8], size: (u32, u32), format: DXGI_FORMAT, out: &mut [u8], pitch: usize) -> bool {
    let (w, h) = (size.0 as usize, size.1 as usize);
    let rows = canvas.chunks_exact(w * 4).take(h).zip(out.chunks_mut(pitch));
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => rows.for_each(|(from, to)| to[..w * 4].copy_from_slice(from)),
        DXGI_FORMAT_B8G8R8A8_UNORM => rows.for_each(|(from, to)| {
            for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(4)) {
                q.copy_from_slice(&[p[2], p[1], p[0], 0xFF]);
            }
        }),
        DXGI_FORMAT_R10G10B10A2_UNORM => rows.for_each(|(from, to)| {
            let ten = |v: u8| (u32::from(v) * 1023 + 127) / 255;
            for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(4)) {
                q.copy_from_slice(&(ten(p[0]) | ten(p[1]) << 10 | ten(p[2]) << 20 | 3 << 30).to_le_bytes());
            }
        }),
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            static HALVES: std::sync::OnceLock<[[u8; 2]; 256]> = std::sync::OnceLock::new();
            let halves = HALVES.get_or_init(|| std::array::from_fn(|v| half(linear(v as u8)).to_le_bytes()));
            let one = half(1.0).to_le_bytes();
            rows.for_each(|(from, to)| {
                for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(8)) {
                    q[..2].copy_from_slice(&halves[usize::from(p[0])]);
                    q[2..4].copy_from_slice(&halves[usize::from(p[1])]);
                    q[4..6].copy_from_slice(&halves[usize::from(p[2])]);
                    q[6..].copy_from_slice(&one);
                }
            });
        }
        _ => return false,
    }
    true
}

/// Bytes a pixel of `format` takes, among those [`write_out`] writes.
pub(super) fn pixel_bytes(format: DXGI_FORMAT) -> Option<u32> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_of_what_the_panel_writes_are_exact() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(1.0), 0x3C00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(0.25), 0x3400);
        // A small one, below the normal range: 2^-20.
        assert_eq!(half(1.0 / 1_048_576.0), 0x0010);
        assert_eq!(half(linear(255)), 0x3C00);
    }

    #[test]
    fn every_format_is_written_as_its_pixels() {
        let canvas = [10u8, 128, 255, 255, 0, 0, 0, 255];
        let mut out = vec![0u8; 2 * 8 + 3];
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R8G8B8A8_UNORM, &mut out, 19));
        assert_eq!(&out[..8], &canvas);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_B8G8R8A8_UNORM, &mut out, 19));
        assert_eq!(&out[..4], &[255, 128, 10, 255]);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R10G10B10A2_UNORM, &mut out, 19));
        assert_eq!(u32::from_le_bytes([out[0], out[1], out[2], out[3]]), 40 | 514 << 10 | 1023 << 20 | 3 << 30);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R16G16B16A16_FLOAT, &mut out, 19));
        assert_eq!(&out[4..8], &[0x00, 0x3C, 0x00, 0x3C]);
        assert_eq!(&out[8..16], &[0, 0, 0, 0, 0, 0, 0x00, 0x3C]);
        assert!(!write_out(&canvas, (2, 1), DXGI_FORMAT_R32G32B32A32_FLOAT, &mut out, 19));
    }
}
