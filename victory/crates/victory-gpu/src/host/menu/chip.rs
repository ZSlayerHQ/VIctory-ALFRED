//! The chip: a small box in the settings panel's style for the frame-rate
//! counter, the details panel (F3) and the shaders' status. Its art is
//! [`CHIP_IMAGE`] in the menu's art folder, a nine-slice cut from the
//! panel's own corner and edges at half the panel's scale (its `layout.json`
//! entry `chip`): the four corners drawn as they are, the edges and the
//! centre repeated (`drawChip` in `scripts/overlay-art/chip.mjs` of the outer
//! repository), then scaled by the panel's factor, as the lettering is.
//! Drawn on the processor when its text changes and presented as the menu
//! is; without the art (or with `VICTORY_MENU_PANEL=0`) the callers draw
//! their dot letters instead.

use std::cell::RefCell;
use std::path::Path;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use super::art::{self, Lettering, SMALL_FONT};
use super::panel::Canvas;
use super::{LEAVE, UPLOADS};
use crate::host::picture::{self, Blend, Buffers};
use crate::host::{Host, HostList, qoi};

/// The chip's art in the menu's art folder.
pub(in crate::host) const CHIP_IMAGE: &str = "chip.qoi";
/// Its `layout.json` entry: the corners, drawn as they are, in the art's pixels
/// (what is between them, `edge` there, repeats), and from the chip's edge to its text.
const CORNER: u32 = 32;
const PADDING: f32 = 30.0;
/// From a line's top to its baseline, in the panel's pixels (the lettering's).
const ASCENT: f32 = 34.0;
/// From the window's corner to the chip, in the panel's pixels.
const MARGIN: f32 = 16.0;

/// Lines apart, in the panel's pixels: as the panel's help and key lines.
fn pitch() -> f32 {
    (art::KEYS_BASELINE_Y - art::HELP_BASELINE_Y) * art::PANEL.1 as f32
}

/// The panel's factor in a window `height` high (its lettering's and the chip's scale).
pub(in crate::host) fn scale_for(height: u32) -> f32 {
    art::panel_size((height.saturating_mul(2), height)).1 as f32 / art::PANEL.1 as f32
}

/// The chip's art, decoded: the chip and the small lettering.
pub(in crate::host) struct ChipArt {
    chip: qoi::Image,
    small: qoi::Image,
}

impl ChipArt {
    /// The art in `dir`.
    pub(in crate::host) fn read(dir: &Path) -> Result<ChipArt, String> {
        let image = |name: &str| {
            let data = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
            qoi::decode(&data).ok_or_else(|| format!("{name}: not a QOI image"))
        };
        ChipArt::new(image(CHIP_IMAGE)?, image(SMALL_FONT.image)?)
    }

    pub(in crate::host) fn new(chip: qoi::Image, small: qoi::Image) -> Result<ChipArt, String> {
        if chip.width <= 2 * CORNER || chip.height <= 2 * CORNER {
            return Err(format!("{CHIP_IMAGE}: {} x {}, no room for corners of {CORNER}", chip.width, chip.height));
        }
        if (small.width, small.height) != SMALL_FONT.size {
            return Err(format!("{}: {} x {}, the layout has {:?}", SMALL_FONT.image, small.width, small.height, SMALL_FONT.size));
        }
        Ok(ChipArt { chip, small })
    }

    /// The art for a window `height` high.
    pub(in crate::host) fn at(self, height: u32) -> Scaled {
        let k = scale_for(height);
        Scaled { k, small: Lettering::new(&self.small, &SMALL_FONT, k), chip: self.chip, background: RefCell::new(None) }
    }
}

/// The chip's art for one window height.
pub(in crate::host) struct Scaled {
    k: f32,
    chip: qoi::Image,
    small: Lettering,
    /// The chip without its text, at the size drawn last (sizes change seldom).
    background: RefCell<Option<((u32, u32), Vec<u8>)>>,
}

/// The chip's text colours: a first line near white, the others soft grey-blue (the settings panel's).
pub(in crate::host) const TEXT: [u8; 3] = art::LABEL;
pub(in crate::host) const SOFT: [u8; 3] = art::SOFT;

/// A line of a chip, and its colour.
pub(in crate::host) type Line = (String, [u8; 3]);

/// The art's pixel for pixel `at` of a chip `length` long, along an axis of
/// the art `art` long: the corners as they are, the edge between them
/// repeated (`drawChip`'s rule).
fn from(at: u32, length: u32, art: u32) -> u32 {
    let edge = art - 2 * CORNER;
    if at < CORNER {
        at
    } else if at >= length.saturating_sub(CORNER) {
        art - (length - at)
    } else {
        CORNER + (at - CORNER) % edge
    }
}

impl Scaled {
    fn px(&self, v: f32) -> u32 {
        (v * self.k).round().max(1.0) as u32
    }

    /// The chip's size for `lines`: the text and the padding around it, at least its corners.
    pub(in crate::host) fn size(&self, lines: &[Line]) -> (u32, u32) {
        let widest = lines.iter().map(|(l, _)| self.small.width(l)).fold(0.0, f32::max);
        let pad = self.px(PADDING);
        let least = self.px((2 * CORNER + 1) as f32);
        let w = widest.ceil() as u32 + 2 * pad;
        let h = (lines.len() as f32 * pitch() * self.k).round() as u32 + 2 * pad;
        (w.max(least), h.max(least))
    }

    /// The window's corner to the chip, in pixels.
    pub(in crate::host) fn margin(&self) -> u32 {
        self.px(MARGIN)
    }

    /// The chip without text at `size`: drawn at the art's own scale by
    /// [`from`], then scaled by the panel's factor with the box filter.
    fn background(&self, (w, h): (u32, u32)) -> Vec<u8> {
        let (aw, ah) = (((w as f32 / self.k).ceil() as u32).max(2 * CORNER + 1), ((h as f32 / self.k).ceil() as u32).max(2 * CORNER + 1));
        let n = self.chip.channels;
        let mut drawn = Vec::with_capacity(aw as usize * ah as usize * n);
        for y in 0..ah {
            let row = &self.chip.pixels[from(y, ah, self.chip.height) as usize * self.chip.width as usize * n..];
            for x in 0..aw {
                let at = from(x, aw, self.chip.width) as usize * n;
                drawn.extend_from_slice(&row[at..at + n]);
            }
        }
        let scaled = if (aw, ah) == (w, h) {
            drawn
        } else {
            picture::boxed(&drawn, (aw as usize, ah as usize), n, (w as usize, h as usize), (aw as f32 / w as f32, ah as f32 / h as f32))
        };
        picture::rgba(scaled, n)
    }

    /// Draws `lines` on a chip of their size into `canvas` (four bytes a pixel, red first).
    pub(in crate::host) fn compose(&self, lines: &[Line], canvas: &mut Vec<u8>) -> (u32, u32) {
        let size = self.size(lines);
        let mut background = self.background.borrow_mut();
        if background.as_ref().is_none_or(|(s, _)| *s != size) {
            *background = Some((size, self.background(size)));
        }
        canvas.clear();
        canvas.extend_from_slice(background.as_ref().map_or(&[][..], |(_, b)| b.as_slice()));
        let pad = self.px(PADDING) as f32;
        let mut c = Canvas::new(canvas, (size.0 as usize, size.1 as usize));
        for (i, (line, colour)) in lines.iter().enumerate() {
            let baseline = pad + (i as f32 * pitch() + ASCENT) * self.k;
            c.text(&self.small, line, pad, baseline, *colour, false);
        }
        size
    }

    fn bytes(&self) -> usize {
        let background = self.background.borrow().as_ref().map_or(0, |(_, b)| b.capacity());
        self.chip.pixels.capacity() + self.small.bytes() + background
    }
}

/// A chip on the picture: its art for the window's height, its buffers, and
/// what it shows. Let go of whole ([`ChipView::release`]) when nothing is shown.
pub(in crate::host) struct ChipView {
    /// Drawn on its art when that is there (`VICTORY_MENU_PANEL`, unless 0).
    allowed: bool,
    art: Option<(u32, Scaled)>,
    /// The art was looked for since the last release and is not there.
    missing: bool,
    buffers: Option<Buffers>,
    leaving: Vec<(u32, Buffers)>,
    canvas: Vec<u8>,
    shown: Vec<Line>,
    size: (u32, u32),
    /// Pictures made, for the tests and the log.
    pub(in crate::host) composed: u64,
}

impl ChipView {
    pub(in crate::host) fn new(allowed: bool) -> ChipView {
        ChipView {
            allowed,
            art: None,
            missing: false,
            buffers: None,
            leaving: Vec::new(),
            canvas: Vec::new(),
            shown: Vec::new(),
            size: (0, 0),
            composed: 0,
        }
    }

    /// Whether the chip can be drawn in a window `height` high: its art read
    /// (once a release: not looked for again while it is missing).
    pub(in crate::host) fn ready(&mut self, height: u32) -> bool {
        if !self.allowed || self.missing {
            return false;
        }
        if self.art.as_ref().is_some_and(|(h, _)| *h == height) {
            return true;
        }
        // (Read again for another height: what is kept is the art at one size.)
        self.art = None;
        let art = super::art::folder().ok_or_else(|| "no art folder".to_owned()).and_then(|dir| ChipArt::read(&dir));
        self.use_art(art, height)
    }

    /// Takes `art` (or its absence) for a window `height` high.
    pub(in crate::host) fn use_art(&mut self, art: Result<ChipArt, String>, height: u32) -> bool {
        match art {
            Ok(art) => {
                self.art = Some((height, art.at(height)));
                self.shown.clear();
                true
            }
            Err(e) => {
                static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(error = %e, "counter chip: no art; drawn in dot letters");
                }
                self.missing = true;
                false
            }
        }
    }

    /// The chip with `lines` for a back buffer `out` of `format`: made again
    /// only if its lines or the buffers changed. False if it cannot be drawn.
    pub(in crate::host) fn prepare(&mut self, blend: &mut Blend, host: &Host, out: (u32, u32), format: DXGI_FORMAT, lines: &[Line]) -> bool {
        let Some((_, scaled)) = &self.art else { return false };
        let alpha = scaled.chip.channels == 4;
        if alpha && !blend.ready(host, format) {
            return false;
        }
        let size = scaled.size(lines);
        let fits = self.buffers.as_ref().is_some_and(|b| b.format == format && b.alpha == alpha && b.size.0 >= size.0 && b.size.1 >= size.1);
        if !fits {
            if let Some(old) = self.buffers.take() {
                self.leaving.push((LEAVE, old));
            }
            let room = (size.0.next_multiple_of(64).min(out.0.max(size.0)), size.1.next_multiple_of(32));
            let Some(buffers) = Buffers::overlay(host, format, room, UPLOADS, alpha) else { return false };
            self.buffers = Some(buffers);
            self.shown.clear();
        }
        if self.shown.as_slice() != lines {
            let Some(buffers) = &mut self.buffers else { return false };
            self.size = scaled.compose(lines, &mut self.canvas);
            buffers.write_canvas(&self.canvas, self.size);
            self.shown = lines.to_vec();
            self.composed += 1;
        }
        true
    }

    /// Records the chip prepared last onto `back` (`out` large), at its margin from the top left.
    pub(in crate::host) fn draw(&mut self, blend: &Blend, host: &Host, list: &HostList, back: &ID3D12Resource, out: (u32, u32)) {
        let (Some((_, scaled)), Some(buffers)) = (&self.art, &mut self.buffers) else { return };
        let m = scaled.margin();
        if m >= out.0 || m >= out.1 {
            return;
        }
        let part = (self.size.0.min(out.0 - m), self.size.1.min(out.1 - m));
        buffers.draw(blend, host, list, back, part, (m, m));
    }

    /// Nothing shown: the art and the picture go now, the buffers once the card is done with them.
    pub(in crate::host) fn release(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            self.leaving.push((LEAVE, buffers));
        }
        self.art = None;
        self.missing = false;
        self.canvas = Vec::new();
        self.shown.clear();
    }

    /// A present: buffers let go are retired [`LEAVE`] presents later.
    pub(in crate::host) fn tick(&mut self) {
        for (left, _) in &mut self.leaving {
            *left = left.saturating_sub(1);
        }
        while let Some(i) = self.leaving.iter().position(|(left, _)| *left == 0) {
            crate::host::retire(self.leaving.swap_remove(i).1);
        }
    }

    /// Bytes held: on the processor's side, and buffers for the card (those leaving too).
    pub(in crate::host) fn held(&self) -> (u64, u64) {
        let cpu = self.art.as_ref().map_or(0, |(_, s)| s.bytes()) + self.canvas.capacity();
        let card = self.buffers.iter().chain(self.leaving.iter().map(|(_, b)| b)).map(Buffers::bytes).sum();
        (cpu as u64, card)
    }
}

#[cfg(test)]
pub(in crate::host) mod tests {
    use super::*;

    /// A chip made here, of the art's size (72 by 72): a pink rim, a cyan line
    /// inside it and dark glass, with a mark in each corner and on each edge's
    /// strip, so that what is drawn where can be told.
    pub(in crate::host) fn made_art() -> ChipArt {
        let (w, h) = (72u32, 72u32);
        let pixels = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let edge = x.min(y).min(w - 1 - x).min(h - 1 - y);
                    match edge {
                        0..=3 => [255, 122, 188],
                        4..=5 => [70, 240, 238],
                        _ => [20, 18, (x + y) as u8],
                    }
                })
            })
            .collect();
        let chip = qoi::Image { width: w, height: h, channels: 3, pixels };
        ChipArt::new(chip, art::tests::made_small()).expect("art of the right sizes")
    }

    fn lines(texts: &[&str]) -> Vec<Line> {
        texts.iter().map(|t| ((*t).to_owned(), art::LABEL)).collect()
    }

    #[test]
    fn the_corners_are_drawn_as_they_are_and_the_edges_repeat() {
        // At the art's own size: every pixel its own.
        assert_eq!((0..72).map(|x| from(x, 72, 72)).collect::<Vec<_>>(), (0..72).collect::<Vec<_>>());
        // Longer: the corners as they are, the 8 between them over and over (`drawChip`).
        let xs: Vec<u32> = (0..200).map(|x| from(x, 200, 72)).collect();
        assert_eq!(&xs[..32], &(0..32).collect::<Vec<_>>()[..]);
        assert_eq!(&xs[168..], &(40..72).collect::<Vec<_>>()[..]);
        assert_eq!(&xs[32..48], &[32, 33, 34, 35, 36, 37, 38, 39, 32, 33, 34, 35, 36, 37, 38, 39]);
        assert!(xs[32..168].iter().all(|x| (32..40).contains(x)));
        // Art that leaves no edge between its corners is refused.
        let small = qoi::Image { width: 64, height: 72, channels: 3, pixels: vec![0; 64 * 72 * 3] };
        assert!(ChipArt::new(small, art::tests::made_small()).is_err());
    }

    #[test]
    fn a_short_line_a_long_line_and_the_details_lines_lay_out() {
        let scaled = made_art().at(2160);
        assert!((scaled.k - 1.0).abs() < 1e-6, "{}", scaled.k);
        let short = scaled.size(&lines(&["FPS: 60.0"]));
        let long = scaled.size(&lines(&["FPS: 143.9  FG: 287.8", "PREPARING SHADERS (1234 left)"]));
        assert!(long.0 > short.0 && long.1 > short.1, "{short:?} {long:?}");
        // The details panel: ten lines and the status, each of them inside the chip.
        let details: Vec<String> = (0..11).map(|i| format!("LINE {i} SUBMIT BUSY 12.3  WAITS FOR GPU 45.6")).collect();
        let details = lines(&details.iter().map(String::as_str).collect::<Vec<_>>());
        let (w, h) = scaled.size(&details);
        let pad = scaled.px(PADDING) as f32;
        let widest = details.iter().map(|(l, _)| scaled.small.width(l)).fold(0.0, f32::max);
        assert!(pad + widest <= w as f32 - pad + 1.0);
        let last = pad + (10.0 * pitch() + ASCENT) * scaled.k;
        assert!(last < h as f32 - pad / 2.0, "{last} {h}");
        // Drawn: the art's corner at the chip's, the rim along its edges, lettering on it.
        let mut canvas = Vec::new();
        let size = scaled.compose(&details, &mut canvas);
        assert_eq!(size, (w, h));
        assert_eq!(canvas.len(), (w * h * 4) as usize);
        let px = |x: u32, y: u32| canvas[((y * w + x) * 4) as usize..][..3].to_vec();
        assert_eq!(px(0, 0), [255, 122, 188]);
        assert_eq!(px(w - 1, h - 1), [255, 122, 188]);
        assert_eq!(px(w / 2, 0), [255, 122, 188]);
        assert_eq!(px(w / 2, 4), [70, 240, 238]);
        let lit = canvas.chunks_exact(4).filter(|p| p[..3] == art::LABEL).count();
        assert!(lit > 100, "{lit}");
        // A second picture of the same size reuses its background.
        let mut again = Vec::new();
        scaled.compose(&lines(&["X"]), &mut again);
        // Smaller in a smaller window, the corners scaled with it.
        let small = made_art().at(720);
        assert!(small.size(&details).0 < w);
        assert_eq!(small.px((2 * CORNER + 1) as f32), (65.0 * scale_for(720)).round() as u32);
    }

    #[test]
    fn rgb_and_rgba_chips_keep_coverage_in_the_four_corner_slices() {
        use art::tests::{assert_corners_and_centre, rounded, write_image};
        use crate::host::menu::status::Status;

        let dir = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let dir = dir.path();
        let art = made_art();
        write_image(dir, SMALL_FONT.image, &art.small);
        write_image(dir, CHIP_IMAGE, &art.chip);
        let rgb = ChipArt::read(dir).unwrap().at(2160);
        assert_eq!(rgb.chip.channels, 3);
        write_image(dir, CHIP_IMAGE, &rounded(art.chip));
        let rgba = ChipArt::read(dir).unwrap().at(2160);
        assert_eq!(rgba.chip.channels, 4);

        let preparing = Status::Preparing(1234);
        let ready = Status::Ready(std::time::Instant::now());
        let status_line = |status: Status| (status.words().unwrap(), status.colour());
        let mut details = lines(&["FPS 60.0  FRAME 16.7 MS  LONGEST 20", "FRAME GEN OFF", "DRAWS 112  DROPPED 0",
            "SUBMIT BUSY 12.3  WAITS FOR GPU 45.6", "GPU BUSY 12.0  IDLE BEHIND US 3.0", "COPY BACK 0.2",
            "MB UP 3  TEXTURES 2  BACK 1", "RAM 1.0 GB  IN MEMORY 2.0", "VRAM 1.0/8.0 GB  SHARED 0.0", "THIS PANEL 100 US A FRAME"]);
        details.push(status_line(preparing));
        let layouts = [lines(&["FPS: 60.0"]), lines(&["FPS: 143.9  FG: 287.8"]), vec![status_line(preparing)], vec![status_line(ready)], details];
        for lines in &layouts {
            let mut old = Vec::new();
            let mut canvas = Vec::new();
            let (w, h) = rgba.compose(lines, &mut canvas);
            assert_eq!(rgb.compose(lines, &mut old), (w, h));
            assert_corners_and_centre(&old, (w, h), 255);
            assert_corners_and_centre(&canvas, (w, h), 0);
            assert!(canvas.chunks_exact(4).any(|p| p[3] == 128));
            for (i, (a, b)) in old.chunks_exact(4).zip(canvas.chunks_exact(4)).enumerate() {
                let (x, y) = (i as u32 % w, i as u32 / w);
                if b[3] == 255 {
                    assert_eq!(a, b);
                } else {
                    assert!(x.min(w - 1 - x) < CORNER && y.min(h - 1 - y) < CORNER);
                }
            }
        }

        let rgba = ChipArt::read(dir).unwrap().at(720);
        for lines in &layouts {
            let mut canvas = Vec::new();
            let size = rgba.compose(lines, &mut canvas);
            assert_corners_and_centre(&canvas, size, 0);
        }
    }

    #[test]
    fn without_art_or_switched_off_it_is_not_drawn() {
        let mut off = ChipView::new(false);
        assert!(!off.ready(1080));
        assert_eq!(off.held(), (0, 0));
        let mut missing = ChipView::new(true);
        assert!(!missing.use_art(Err("chip.qoi: not found".to_owned()), 1080));
        assert!(!missing.ready(1080), "not looked for again until released");
        assert_eq!(missing.held(), (0, 0));
        missing.release();
        assert!(!missing.missing);
    }

    #[test]
    fn unchanged_lines_are_not_drawn_again_and_nothing_is_held_when_released() {
        let Some(host) = crate::host::host() else {
            eprintln!("no host device: not run");
            return;
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
        let mut view = ChipView::new(true);
        let mut blend = Blend::default();
        assert!(view.use_art(Ok(made_art()), 2160));
        let out = (3840, 2160);
        let a = lines(&["FPS: 60.0"]);
        assert!(view.prepare(&mut blend, host, out, DXGI_FORMAT_B8G8R8A8_UNORM, &a));
        assert!(view.prepare(&mut blend, host, out, DXGI_FORMAT_B8G8R8A8_UNORM, &a));
        assert_eq!(view.composed, 1, "an unchanged frame makes nothing");
        let b = lines(&["FPS: 59.9", "SHADERS READY"]);
        assert!(view.prepare(&mut blend, host, out, DXGI_FORMAT_B8G8R8A8_UNORM, &b));
        assert_eq!(view.composed, 2);
        let (cpu, card) = view.held();
        assert!(cpu > 0 && card > 0);
        eprintln!("chip held at 3840 x 2160: {cpu} bytes on the processor's side, {card} for the card");
        view.release();
        assert_eq!(view.held().0, 0);
        for _ in 0..LEAVE {
            view.tick();
        }
        assert_eq!(view.held(), (0, 0));
    }

    /// One update of the chip at 3840 by 2160, for the details panel's lines
    /// and the counter's: `cargo test --profile diag -p victory-gpu --lib chip::tests::cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, in an optimised build"]
    fn cost() {
        use std::time::Instant;
        let scaled = match std::env::var_os("VICTORY_MENU_ART") {
            Some(dir) => match ChipArt::read(Path::new(&dir)) {
                Ok(art) => art.at(2160),
                Err(e) => {
                    println!("the art folder's chip: {e}; a chip made here");
                    made_art().at(2160)
                }
            },
            None => made_art().at(2160),
        };
        let details: Vec<String> = (0..11).map(|i| format!("LINE {i} SUBMIT BUSY 12.3  WAITS FOR GPU 45.6")).collect();
        for (what, texts) in [("counter", vec!["FPS: 143.9  FG: 287.8".to_owned(), "PREPARING SHADERS (1234 left)".to_owned()]), ("details", details)] {
            let lines = lines(&texts.iter().map(String::as_str).collect::<Vec<_>>());
            let mut canvas = Vec::new();
            let (mut compose, mut write) = (Vec::new(), Vec::new());
            for _ in 0..50 {
                let t = Instant::now();
                let size = scaled.compose(&lines, &mut canvas);
                compose.push(t.elapsed().as_secs_f64() * 1e3);
                let pitch = (size.0 * 4).next_multiple_of(256) as usize;
                let mut out = vec![0u8; pitch * size.1 as usize];
                let t = Instant::now();
                picture::write_out(&canvas, size, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM, &mut out, pitch);
                write.push(t.elapsed().as_secs_f64() * 1e3);
            }
            compose.sort_by(f64::total_cmp);
            write.sort_by(f64::total_cmp);
            let size = scaled.size(&lines);
            println!(
                "{what}: chip {} x {}, drawn in {:.3} ms, written out in {:.3} ms (medians); art and lettering {} bytes, picture {} bytes",
                size.0,
                size.1,
                compose[25],
                write[25],
                scaled.bytes(),
                canvas.capacity()
            );
            // `VICTORY_MENU_PROOF=<file.ppm>`: the details chip written there, to look at.
            if what == "details"
                && let Some(proof) = std::env::var_os("VICTORY_MENU_PROOF")
            {
                let mut ppm = format!("P6 {} {} 255
", size.0, size.1).into_bytes();
                ppm.extend(canvas.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
                std::fs::write(proof, ppm).expect("the proof written");
            }
        }
    }
}
