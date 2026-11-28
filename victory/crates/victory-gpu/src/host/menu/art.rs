//! The settings menu's art: the panel and two sizes of lettering, made by
//! `scripts/overlay-art/` in the outer repository (our own art), read from
//! QOI files in a folder and scaled to the size the menu takes; and where
//! things sit on the panel, as fractions of its width and height (that
//! folder's `layout.json`, written out here as constants).
//!
//! The QOI files are not in this repository: its asset policy takes no image
//! files. They are read from `VICTORY_MENU_ART`, else from the folder
//! `menu-art` beside the program; without them the menu is drawn in the
//! dot letters as before.

use std::path::{Path, PathBuf};

use super::qoi;

/// One size of lettering: a grid of cells in a coverage image, each character
/// at its origin (`origin_x` from the cell's left, standing on `baseline`
/// from its top), and how far each moves the pen. The characters are 32 to
/// 126, then `extra`.
pub(super) struct FontSpec {
    pub(super) image: &'static str,
    pub(super) size: (u32, u32),
    pub(super) columns: u32,
    pub(super) cell: (u32, u32),
    pub(super) origin_x: f32,
    pub(super) baseline: f32,
    pub(super) first: u32,
    pub(super) extra: [char; 4],
    pub(super) advance: [f32; 99],
}

// From layout.json (outer repository, scripts/overlay-art/assets/, 2026-11-28).
#[rustfmt::skip]
mod layout {
    use super::FontSpec;
    pub(in crate::host::menu) const PANEL_IMAGE: &str = "panel.qoi";
    pub(in crate::host::menu) const PANEL: (u32, u32) = (1600, 1200);
    pub(in crate::host::menu) const ROW_FONT: FontSpec = FontSpec {
        image: "font-row.qoi",
        size: (832, 406),
        columns: 16,
        cell: (52, 58),
        origin_x: 6.0,
        baseline: 40.0,
        first: 32,
        extra: ['\u{2039}', '\u{203a}', '\u{b7}', '\u{d7}'],
        advance: [10.56, 10.3, 13.45, 22.19, 20.6, 25.84, 23.75, 7.14, 12.27, 12.27, 15.85, 22.19, 8.32, 12.27, 8.32, 10.68, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 8.32, 8.32, 22.19, 22.19, 22.19, 12.65, 38.38, 25.35, 21.39, 26.9, 28.5, 19.0, 17.82, 28.12, 27.7, 9.5, 9.5, 24.93, 18.62, 29.68, 29.68, 31.27, 19.38, 31.27, 22.95, 17.4, 22.95, 26.9, 22.95, 39.6, 26.9, 22.95, 24.55, 12.65, 10.68, 12.65, 17.82, 19.0, 12.65, 16.23, 19.0, 16.64, 19.38, 18.2, 9.5, 16.23, 19.0, 8.32, 8.32, 18.2, 8.32, 29.3, 19.0, 20.98, 19.0, 19.0, 15.05, 14.63, 12.65, 19.0, 16.64, 27.32, 19.0, 16.64, 15.85, 12.65, 9.88, 12.65, 22.19, 11.89, 11.89, 9.88, 22.19],
    };
    pub(in crate::host::menu) const SMALL_FONT: FontSpec = FontSpec {
        image: "font-small.qoi",
        size: (704, 315),
        columns: 16,
        cell: (44, 45),
        origin_x: 6.0,
        baseline: 31.0,
        first: 32,
        extra: ['\u{2039}', '\u{203a}', '\u{b7}', '\u{d7}'],
        advance: [8.34, 8.13, 10.62, 17.52, 16.26, 20.4, 18.75, 5.64, 9.69, 9.69, 12.51, 17.52, 6.57, 9.69, 6.57, 8.43, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 6.57, 6.57, 17.52, 17.52, 17.52, 9.99, 30.3, 20.01, 16.89, 21.24, 22.5, 15.0, 14.07, 22.2, 21.87, 7.5, 7.5, 19.68, 14.7, 23.43, 23.43, 24.69, 15.3, 24.69, 18.12, 13.74, 18.12, 21.24, 18.12, 31.26, 21.24, 18.12, 19.38, 9.99, 8.43, 9.99, 14.07, 15.0, 9.99, 12.81, 15.0, 13.14, 15.3, 14.37, 7.5, 12.81, 15.0, 6.57, 6.57, 14.37, 6.57, 23.13, 15.0, 16.56, 15.0, 15.0, 11.88, 11.55, 9.99, 15.0, 13.14, 21.57, 15.0, 13.14, 12.51, 9.99, 7.8, 9.99, 17.52, 9.39, 9.39, 7.8, 17.52],
    };
    pub(in crate::host::menu) const FIRST_BASELINE_Y: f32 = 0.31333;
    pub(in crate::host::menu) const ROW_PITCH: f32 = 0.04167;
    pub(in crate::host::menu) const HEADING_GAP_BEFORE: f32 = 0.01;
    pub(in crate::host::menu) const HEADING_PITCH: f32 = 0.03917;
    pub(in crate::host::menu) const LAST_BASELINE_Y: f32 = 0.84167;
    pub(in crate::host::menu) const X_HEADING: f32 = 0.0675;
    pub(in crate::host::menu) const X_LABEL: f32 = 0.0875;
    pub(in crate::host::menu) const X_VALUE_RIGHT: f32 = 0.91;
    pub(in crate::host::menu) const X_PICKED_VALUE_RIGHT: f32 = 0.9;
    pub(in crate::host::menu) const X_ARROW_LEFT: f32 = 0.6275;
    pub(in crate::host::menu) const X_ARROW_RIGHT: f32 = 0.9275;
    pub(in crate::host::menu) const X_HELP: f32 = 0.0675;
    pub(in crate::host::menu) const BAR_LEFT: f32 = 0.0525;
    pub(in crate::host::menu) const BAR_RIGHT: f32 = 0.9475;
    pub(in crate::host::menu) const BAR_TOP_ABOVE_BASELINE: f32 = 0.032;
    pub(in crate::host::menu) const BAR_HEIGHT: f32 = 0.04333;
    pub(in crate::host::menu) const BAR_UNDERLINE_BELOW_BASELINE: f32 = 0.012;
    pub(in crate::host::menu) const BAR_UNDERLINE_HEIGHT: f32 = 0.00233;
    pub(in crate::host::menu) const BAR_FILL: [u8; 3] = [22, 104, 118];
    pub(in crate::host::menu) const BAR_UNDERLINE: [u8; 3] = [70, 240, 238];
    pub(in crate::host::menu) const HELP_BASELINE_Y: f32 = 0.89333;
    pub(in crate::host::menu) const KEYS_BASELINE_Y: f32 = 0.93833;
    pub(in crate::host::menu) const HEADING: [u8; 3] = [255, 122, 188];
    pub(in crate::host::menu) const LABEL: [u8; 3] = [240, 245, 255];
    pub(in crate::host::menu) const VALUE: [u8; 3] = [120, 255, 244];
    pub(in crate::host::menu) const PICKED_VALUE: [u8; 3] = [255, 255, 255];
    pub(in crate::host::menu) const SOFT: [u8; 3] = [168, 180, 214];
}
pub(super) use layout::*;

/// Where the art is read from: `VICTORY_MENU_ART`, else `menu-art` beside the program.
pub(super) fn folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("VICTORY_MENU_ART").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    Some(std::env::current_exe().ok()?.parent()?.join("menu-art"))
}

/// The art as made: decoded, at the panel's own size.
pub(super) struct Art {
    panel: qoi::Image,
    row: qoi::Image,
    small: qoi::Image,
}

impl Art {
    /// The three images in `dir`, checked against the layout above.
    pub(super) fn read(dir: &Path) -> Result<Art, String> {
        let image = |name: &str| {
            let data = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
            qoi::decode(&data).ok_or_else(|| format!("{name}: not a QOI image"))
        };
        Art::new(image(PANEL_IMAGE)?, image(ROW_FONT.image)?, image(SMALL_FONT.image)?)
    }

    pub(super) fn new(panel: qoi::Image, row: qoi::Image, small: qoi::Image) -> Result<Art, String> {
        let check = |what: &str, image: &qoi::Image, size: (u32, u32)| {
            if (image.width, image.height) == size {
                Ok(())
            } else {
                Err(format!("{what}: {} x {}, the layout has {} x {}", image.width, image.height, size.0, size.1))
            }
        };
        check(PANEL_IMAGE, &panel, PANEL)?;
        check(ROW_FONT.image, &row, ROW_FONT.size)?;
        check(SMALL_FONT.image, &small, SMALL_FONT.size)?;
        Ok(Art { panel, row, small })
    }

    /// The art at `size` (from [`panel_size`]): the panel and the lettering scaled
    /// by the same factor with a box filter.
    pub(super) fn at(&self, size: (u32, u32)) -> Scaled {
        let k = size.1 as f32 / PANEL.1 as f32;
        let (w, h) = (size.0 as usize, size.1 as usize);
        let step = (PANEL.0 as f32 / size.0 as f32, PANEL.1 as f32 / size.1 as f32);
        let rgb = boxed(&self.panel.pixels, (PANEL.0 as usize, PANEL.1 as usize), self.panel.channels, (w, h), step);
        let mut panel = Vec::with_capacity(w * h * 4);
        for p in rgb.chunks_exact(self.panel.channels) {
            panel.extend_from_slice(&[p[0], p[1], p[2], 0xFF]);
        }
        Scaled { size, panel, row: Lettering::new(&self.row, &ROW_FONT, k), small: Lettering::new(&self.small, &SMALL_FONT, k) }
    }
}

/// The size the panel takes in a picture `width` by `height`: three quarters
/// of the height (at most 90% of the width), never larger than the art.
pub(super) fn panel_size((width, height): (u32, u32)) -> (u32, u32) {
    let (pw, ph) = (PANEL.0 as f32, PANEL.1 as f32);
    let h = (height as f32 * 0.75).min(width as f32 * 0.9 * ph / pw).min(ph).round().max(1.0);
    ((h * pw / ph).round().max(1.0) as u32, h as u32)
}

/// One size of lettering, scaled: every character's coverage in a cell of
/// the same size, where its origin sits in that cell, and its advance.
pub(super) struct Lettering {
    cell: (usize, usize),
    cover: Vec<u8>,
    origin_x: f32,
    baseline: f32,
    advance: [f32; 99],
    first: u32,
    extra: [char; 4],
}

impl Lettering {
    fn new(image: &qoi::Image, spec: &FontSpec, k: f32) -> Lettering {
        let (cw, ch) = (spec.cell.0 as usize, spec.cell.1 as usize);
        let cell = (((cw as f32) * k).ceil() as usize, ((ch as f32) * k).ceil() as usize);
        let stride = image.width as usize * image.channels;
        let mut cover = Vec::with_capacity(spec.advance.len() * cell.0 * cell.1);
        let mut one = vec![0u8; cw * ch];
        for i in 0..spec.advance.len() {
            let (cx, cy) = ((i % spec.columns as usize) * cw, (i / spec.columns as usize) * ch);
            for y in 0..ch {
                for x in 0..cw {
                    one[y * cw + x] = image.pixels[(cy + y) * stride + (cx + x) * image.channels];
                }
            }
            cover.extend(boxed(&one, (cw, ch), 1, cell, (1.0 / k, 1.0 / k)));
        }
        Lettering {
            cell,
            cover,
            origin_x: spec.origin_x * k,
            baseline: spec.baseline * k,
            advance: spec.advance.map(|a| a * k),
            first: spec.first,
            extra: spec.extra,
        }
    }

    /// The character's place among the cells, if the lettering has it.
    pub(super) fn index(&self, c: char) -> Option<usize> {
        let n = u32::from(c);
        if (self.first..self.first + 95).contains(&n) {
            return Some((n - self.first) as usize);
        }
        self.extra.iter().position(|e| *e == c).map(|i| 95 + i)
    }

    /// How far `text` moves the pen, in pixels.
    pub(super) fn width(&self, text: &str) -> f32 {
        text.chars().filter_map(|c| self.index(c)).map(|i| self.advance[i]).sum()
    }

    pub(super) fn advance(&self, i: usize) -> f32 {
        self.advance[i]
    }

    /// A cell's size, and where a character's origin and baseline sit in it.
    pub(super) fn metrics(&self) -> ((usize, usize), f32, f32) {
        (self.cell, self.origin_x, self.baseline)
    }

    pub(super) fn cover(&self, i: usize) -> &[u8] {
        let n = self.cell.0 * self.cell.1;
        &self.cover[i * n..(i + 1) * n]
    }

    fn bytes(&self) -> usize {
        self.cover.len()
    }
}

/// The art at one size: the panel (four bytes a pixel, red first, the fourth 255) and the lettering.
pub(super) struct Scaled {
    pub(super) size: (u32, u32),
    pub(super) panel: Vec<u8>,
    pub(super) row: Lettering,
    pub(super) small: Lettering,
}

impl Scaled {
    /// The bytes it holds.
    pub(super) fn bytes(&self) -> usize {
        self.panel.len() + self.row.bytes() + self.small.bytes()
    }
}

/// A box filter: `src` (`sw` by `sh`, `n` bytes a pixel) to `dw` by `dh`,
/// each pixel the mean of the area of the source it covers, `step` source
/// pixels to one on each axis (the area past the source's edge counts as
/// nothing, and is not counted).
fn boxed(src: &[u8], (sw, sh): (usize, usize), n: usize, (dw, dh): (usize, usize), step: (f32, f32)) -> Vec<u8> {
    // For each pixel on an axis: the first source pixel it covers, and how much of each.
    let spans = |d: usize, s: usize, step: f32| -> Vec<(usize, Vec<f32>)> {
        (0..d)
            .map(|i| {
                let (lo, hi) = (i as f32 * step, ((i + 1) as f32 * step).min(s as f32));
                if hi <= lo {
                    return (0, Vec::new());
                }
                let first = lo.floor() as usize;
                let weights: Vec<f32> =
                    (first..(hi.ceil() as usize).min(s)).map(|j| (hi.min(j as f32 + 1.0) - lo.max(j as f32)).max(0.0)).collect();
                let total: f32 = weights.iter().sum();
                (first, weights.into_iter().map(|w| w / total).collect())
            })
            .collect()
    };
    let (across, down) = (spans(dw, sw, step.0), spans(dh, sh, step.1));
    let mut wide = vec![0f32; sh * dw * n];
    for y in 0..sh {
        let row = &src[y * sw * n..(y + 1) * sw * n];
        let out = &mut wide[y * dw * n..(y + 1) * dw * n];
        for (x, (first, weights)) in across.iter().enumerate() {
            for (j, w) in weights.iter().enumerate() {
                let at = (first + j) * n;
                for c in 0..n {
                    out[x * n + c] += f32::from(row[at + c]) * w;
                }
            }
        }
    }
    let mut out = vec![0u8; dh * dw * n];
    let mut sum = vec![0f32; dw * n];
    for (y, (first, weights)) in down.iter().enumerate() {
        sum.iter_mut().for_each(|s| *s = 0.0);
        for (j, w) in weights.iter().enumerate() {
            let row = &wide[(first + j) * dw * n..(first + j + 1) * dw * n];
            for (s, v) in sum.iter_mut().zip(row) {
                *s += v * w;
            }
        }
        for (o, s) in out[y * dw * n..(y + 1) * dw * n].iter_mut().zip(&sum) {
            *o = (s + 0.5).min(255.0) as u8;
        }
    }
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Art of the layout's sizes, made here: a panel of one colour with a
    /// lighter square, and lettering whose every cell is covered where its
    /// character would stand (for the menu's tests; the real art is not in
    /// the repository).
    pub(in super::super) fn made() -> Art {
        let image = |(w, h): (u32, u32), at: &dyn Fn(u32, u32) -> [u8; 3]| qoi::Image {
            width: w,
            height: h,
            channels: 3,
            pixels: (0..h).flat_map(|y| (0..w).flat_map(move |x| at(x, y))).collect(),
        };
        let panel = image(PANEL, &|x, y| if (400..800).contains(&x) && (400..800).contains(&y) { [200, 180, 40] } else { [30, 24, 48] });
        let cells = |spec: &FontSpec| {
            let (cw, ch, base) = (spec.cell.0, spec.cell.1, spec.baseline as u32);
            move |x: u32, y: u32| {
                let (x, y) = (x % cw, y % ch);
                let v = if (8..cw / 2).contains(&x) && (base / 3..base).contains(&y) { 255 } else { 0 };
                [v; 3]
            }
        };
        Art::new(panel, image(ROW_FONT.size, &cells(&ROW_FONT)), image(SMALL_FONT.size, &cells(&SMALL_FONT)))
            .expect("art of the layout's sizes")
    }

    #[test]
    fn the_box_filter_keeps_a_flat_colour_and_averages_an_edge() {
        let flat = vec![77u8; 10 * 6 * 3];
        assert!(boxed(&flat, (10, 6), 3, (4, 3), (2.5, 2.0)).iter().all(|v| *v == 77));
        // Black then white halves, halved: a grey column where they meet, at 1.5 pixels a pixel.
        let edge = [0u8, 0, 0, 255, 255, 255];
        assert_eq!(boxed(&edge, (6, 1), 1, (4, 1), (1.5, 1.0)), [0, 0, 255, 255]);
        let mid: Vec<u8> = [0, 0, 0, 0, 255, 255, 255, 255].to_vec();
        assert_eq!(boxed(&mid, (8, 1), 1, (3, 1), (8.0 / 3.0, 1.0)), [0, 128, 255]);
    }

    #[test]
    fn the_panel_takes_three_quarters_of_the_height_and_never_more_than_its_art() {
        assert_eq!(panel_size((1280, 720)), (720, 540));
        assert_eq!(panel_size((1920, 1080)), (1080, 810));
        assert_eq!(panel_size((3840, 2160)), PANEL);
        // A narrow window: the width decides.
        let (w, h) = panel_size((800, 1200));
        assert!(w <= 720 && h <= 900, "{w} x {h}");
        let art = made();
        let scaled = art.at(panel_size((1920, 1080)));
        assert_eq!(scaled.panel.len(), 1080 * 810 * 4);
        assert_eq!(&scaled.panel[..4], &[30, 24, 48, 255]);
        // Lettering scaled by the same factor: an advance of 19 at full size is 19 * 0.675.
        let i = scaled.row.index('0').unwrap_or(0);
        assert!((scaled.row.advance(i) - 19.0 * 0.675).abs() < 1e-3);
        assert_eq!(scaled.row.index('\u{203a}'), Some(96));
        assert_eq!(scaled.row.index('\u{e9}'), None);
    }

    #[test]
    fn art_of_the_wrong_size_is_refused() {
        let wrong = qoi::Image { width: 2, height: 2, channels: 3, pixels: vec![0; 12] };
        let art = made();
        assert!(Art::new(wrong, art.row, art.small).is_err());
    }
}
