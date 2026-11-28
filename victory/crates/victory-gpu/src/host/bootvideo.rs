//! The boot video: `VICTORY_BOOT_VIDEO=<folder>` plays a video once over the
//! title's first screens, from the first present (its clock starting when
//! its first frame is up, a decode later), by the wall clock, then
//! the title's own screens show. It has no sound and the title's is not
//! touched. `VICTORY_BOOT_VIDEO_LOOPS=N` plays it N times (1 by default).
//!
//! The folder holds `video.json` (`frames`, `fps`, `width`, `height`,
//! `pattern` such as `frames/%04d.qoi`, `first`) and the frames, QOI images
//! of three or four bytes a pixel. The video is not in any repository: it is
//! read from that folder at run time.
//!
//! [`WORKERS`] threads decode ahead, in turn, into a ring of [`RING`] frames; a present takes the
//! newest frame due at its time (frames late are skipped, never queued),
//! scales it to the back buffer (letterboxed, nearest pixel: the menu's box
//! filter is too slow for 24 a second at 3840 by 2160, see the `cost` test)
//! and copies it as the settings menu copies its picture. A frame not ready
//! leaves the last one up. A frame that cannot be read ends the video with
//! one warning. At the end the workers stop and everything is let go (the
//! card's buffers [`LEAVE`] presents later).
//!
//! Unset, nothing is read, decoded, held or drawn: the present path loads
//! one atomic and returns.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use super::picture::{self, Buffers};
use super::{Host, HostList, qoi};

/// The same video over the world-loading screen (`VICTORY_BOOT_VIDEO_COVER=1`).
pub(super) mod cover;

/// Frames decoded ahead and being decoded, at most.
const RING: usize = 4;
/// Threads decoding: a frame's decode took up to 41.9 ms in a loaded start (a frame
/// lasts 41.7 at 24 a second); two keep the ring ahead of the presents.
const WORKERS: usize = 2;
/// Upload buffers written in turn, as the settings menu's.
const UPLOADS: usize = 4;
/// Presents the card's buffers are kept after the end, as the settings menu's.
const LEAVE: u32 = 8;

/// The video in a folder, from its `video.json`.
#[derive(Clone, Debug, PartialEq)]
struct Plan {
    dir: PathBuf,
    frames: u32,
    fps: f64,
    size: (u32, u32),
    pattern: String,
    first: u32,
    loops: u32,
}

/// A value in a flat JSON object: a number as written, or a string's contents.
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = text[at..].trim_start().strip_prefix(':')?.trim_start();
    if let Some(string) = rest.strip_prefix('"') {
        return string.split('"').next();
    }
    let end = rest.find(|c: char| c == ',' || c == '}' || c.is_whitespace()).unwrap_or(rest.len());
    Some(&rest[..end])
}

/// `pattern` with its `%d` (or `%04d`) given `n`.
fn numbered(pattern: &str, n: u32) -> Option<String> {
    let at = pattern.find('%')?;
    let rest = &pattern[at + 1..];
    let end = rest.find('d')?;
    let width: usize = if end == 0 { 0 } else { rest[..end].parse().ok()? };
    Some(format!("{}{n:0width$}{}", &pattern[..at], &rest[end + 1..]))
}

impl Plan {
    fn parse(dir: &Path, text: &str, loops: u32) -> Result<Plan, String> {
        let number = |key: &str| field(text, key).ok_or_else(|| format!("video.json has no {key}"));
        let whole = |key: &str| number(key)?.parse::<u32>().map_err(|e| format!("video.json: {key}: {e}"));
        let plan = Plan {
            dir: dir.to_path_buf(),
            frames: whole("frames")?,
            fps: number("fps")?.parse().map_err(|e| format!("video.json: fps: {e}"))?,
            size: (whole("width")?, whole("height")?),
            pattern: number("pattern")?.to_owned(),
            first: field(text, "first").map_or(Ok(0), |f| f.parse().map_err(|e| format!("video.json: first: {e}")))?,
            loops: loops.max(1),
        };
        if plan.frames == 0 || !plan.fps.is_finite() || plan.fps <= 0.0 || plan.size.0 == 0 || plan.size.1 == 0 || numbered(&plan.pattern, 0).is_none() {
            return Err(format!("video.json: nothing to play ({plan:?})"));
        }
        Ok(plan)
    }

    fn read(dir: &Path, loops: u32) -> Result<Plan, String> {
        let text = std::fs::read_to_string(dir.join("video.json")).map_err(|e| format!("video.json: {e}"))?;
        Plan::parse(dir, &text, loops)
    }

    /// Frames in all, the loops counted.
    fn total(&self) -> u64 {
        u64::from(self.frames) * u64::from(self.loops)
    }

    /// The file of frame `n` (counted over all the loops).
    fn file(&self, n: u64) -> PathBuf {
        let n = (n % u64::from(self.frames)) as u32 + self.first;
        self.dir.join(numbered(&self.pattern, n).unwrap_or_default())
    }

    /// The folder's name (never its whole path, for the log).
    fn name(&self) -> String {
        self.dir.file_name().map_or_else(|| "?".to_owned(), |n| n.to_string_lossy().into_owned())
    }
}

/// The video `var` names, if it is set: `None` unset (and nothing is read).
fn plan_from(var: Option<OsString>, loops: Option<OsString>) -> Option<Result<Plan, String>> {
    let dir = PathBuf::from(var.filter(|v| !v.is_empty())?);
    let loops = loops.and_then(|l| l.to_str()?.trim().parse().ok()).unwrap_or(1);
    Some(Plan::read(&dir, loops))
}

/// The frame shown `elapsed` seconds after the start: `None` once all are past.
fn frame_at(elapsed: f64, fps: f64, total: u64) -> Option<u64> {
    let n = (elapsed.max(0.0) * fps).floor() as u64;
    (n < total).then_some(n)
}

/// Where the picture sits in a back buffer `out`: as large as fits, its
/// shape kept, in the middle (black around it).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    out: (u32, u32),
    at: (u32, u32),
    size: (u32, u32),
}

fn fit(src: (u32, u32), out: (u32, u32)) -> Fit {
    let k = (f64::from(out.0) / f64::from(src.0)).min(f64::from(out.1) / f64::from(src.1));
    let size = (
        ((f64::from(src.0) * k).round() as u32).clamp(1, out.0.max(1)),
        ((f64::from(src.1) * k).round() as u32).clamp(1, out.1.max(1)),
    );
    Fit { out, at: (out.0.saturating_sub(size.0) / 2, out.1.saturating_sub(size.1) / 2), size }
}

/// Rows kept between frames: the source column of each column of the
/// picture, a row being made, it written in the format, and a black row.
#[derive(Default)]
struct Rows {
    xs: Vec<usize>,
    rgbx: Vec<u8>,
    line: Vec<u8>,
    black: Vec<u8>,
}

impl Rows {
    fn bytes(&self) -> usize {
        self.xs.capacity() * std::mem::size_of::<usize>() + self.rgbx.capacity() + self.line.capacity() + self.black.capacity()
    }
}

/// Writes `image` (red first) into `out` at `fit`, rows `pitch` bytes apart,
/// as pixels of `format`: each pixel the nearest of the image's, black around it.
fn write_frame(image: &qoi::Image, fit: &Fit, format: DXGI_FORMAT, out: &mut [u8], pitch: usize, rows: &mut Rows) {
    let (frame, src, n) = (&image.pixels, (image.width, image.height), image.channels);
    let Some(bytes) = picture::pixel_bytes(format) else { return };
    let (ow, oh) = (fit.out.0 as usize, fit.out.1 as usize);
    let row_bytes = ow * bytes as usize;
    if rows.black.len() != row_bytes || rows.rgbx.len() != ow * 4 {
        rows.rgbx = [0, 0, 0, 0xFF].repeat(ow);
        rows.black = vec![0; row_bytes];
        rows.line = vec![0; row_bytes];
        picture::write_row(&rows.rgbx, format, &mut rows.black);
    }
    let (sw, sh) = (u64::from(src.0), u64::from(src.1));
    let (fw, fh) = (u64::from(fit.size.0), u64::from(fit.size.1));
    rows.xs.clear();
    rows.xs.extend((0..fw).map(|x| ((2 * x + 1) * sw / (2 * fw)).min(sw - 1) as usize * n));
    let (left, top) = (fit.at.0 as usize, fit.at.1 as usize);
    let mut last = None;
    for y in 0..oh {
        let to = &mut out[y * pitch..y * pitch + row_bytes];
        if y < top || y >= top + fh as usize {
            to.copy_from_slice(&rows.black);
            continue;
        }
        let sy = ((2 * (y - top) as u64 + 1) * sh / (2 * fh)).min(sh - 1) as usize;
        if last != Some(sy) {
            let from = &frame[sy * src.0 as usize * n..(sy + 1) * src.0 as usize * n];
            for (px, &sx) in rows.rgbx[left * 4..(left + fw as usize) * 4].chunks_exact_mut(4).zip(&rows.xs) {
                px[..3].copy_from_slice(&from[sx..sx + 3]);
            }
            picture::write_row(&rows.rgbx, format, &mut rows.line);
            last = Some(sy);
        }
        to.copy_from_slice(&rows.line);
    }
}

/// What the workers and the presents share.
struct Shared {
    ring: Mutex<Ring>,
    wake: Condvar,
}

#[derive(Default)]
struct Ring {
    /// Frames decoded, by number (workers finish out of order).
    ready: BTreeMap<u64, qoi::Image>,
    /// Frames a worker is decoding now.
    decoding: usize,
    /// The next frame a worker takes.
    next: u64,
    /// The frame the last present was due to show.
    due: u64,
    stop: bool,
    /// The first frame that could not be read or decoded, and what it was.
    failed: Option<(u64, String)>,
    slowest_decode_ms: f64,
    /// The most bytes the ring held.
    most: usize,
}

/// A frame's pixels from its number (counted over all the loops).
type Decode = dyn Fn(u64) -> Result<qoi::Image, String> + Send + Sync;

/// The frame `n` of `plan`, read and decoded from its file.
fn decode_file(plan: &Plan, n: u64) -> Result<qoi::Image, String> {
    let file = plan.file(n);
    let name = file.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    std::fs::read(&file)
        .map_err(|e| e.to_string())
        .and_then(|data| qoi::decode(&data).ok_or_else(|| "not a QOI image".to_owned()))
        .and_then(|image| {
            if (image.width, image.height) == plan.size {
                Ok(image)
            } else {
                Err(format!("{} x {}, video.json has {} x {}", image.width, image.height, plan.size.0, plan.size.1))
            }
        })
        .map_err(|e| format!("{name}: {e}"))
}

/// One of the workers: takes the next frame (from the one due on) while the
/// ring has room for it, at most [`RING`] decoded and being decoded, until
/// the last, a failure, or `stop`.
fn work(total: u64, shared: &Shared, decode: &Decode) {
    loop {
        let n = {
            let mut ring = shared.ring.lock().unwrap_or_else(|e| e.into_inner());
            while ring.ready.len() + ring.decoding >= RING && !ring.stop && ring.failed.is_none() {
                ring = shared.wake.wait_timeout(ring, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
            }
            if ring.stop || ring.failed.is_some() {
                return;
            }
            let n = ring.next.max(ring.due);
            if n >= total {
                return;
            }
            ring.next = n + 1;
            ring.decoding += 1;
            n
        };
        let began = Instant::now();
        let frame = decode(n);
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let mut ring = shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        ring.decoding -= 1;
        match frame {
            Ok(image) => {
                ring.slowest_decode_ms = ring.slowest_decode_ms.max(ms);
                ring.ready.insert(n, image);
                ring.most = ring.most.max(ring.ready.values().map(|i| i.pixels.capacity()).sum());
            }
            Err(e) => {
                if ring.failed.as_ref().is_none_or(|(f, _)| n < *f) {
                    ring.failed = Some((n, e));
                }
            }
        }
        drop(ring);
        shared.wake.notify_all();
    }
}

/// What a present does.
enum Step {
    /// Shows this frame (its number, and it).
    Show(u64, qoi::Image),
    /// Keeps what is up (the frame due is not decoded yet).
    Keep,
    /// The video is over: played through, or a frame could not be read.
    End(Option<String>),
}

/// The presents' side of the video: the workers, and what was shown.
struct Player {
    plan: Plan,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    shown: u64,
    /// The last frame due.
    due: u64,
    /// [`Player::stop`] was called (the drop does not ask again).
    stopped: std::sync::atomic::AtomicBool,
}

impl Player {
    fn start(plan: Plan) -> Result<Player, String> {
        let of_plan = plan.clone();
        Player::start_with(plan, WORKERS, Arc::new(move |n| decode_file(&of_plan, n)))
    }

    /// The video with `workers` threads decoding by `decode`.
    fn start_with(plan: Plan, workers: usize, decode: Arc<Decode>) -> Result<Player, String> {
        let shared = Arc::new(Shared { ring: Mutex::new(Ring::default()), wake: Condvar::new() });
        let total = plan.total();
        let mut player = Player { plan, shared, workers: Vec::new(), shown: 0, due: 0, stopped: false.into() };
        for i in 0..workers.max(1) {
            let (shared, decode) = (Arc::clone(&player.shared), Arc::clone(&decode));
            let worker = std::thread::Builder::new()
                .name(format!("boot video {i}"))
                .spawn(move || work(total, &shared, &*decode))
                .map_err(|e| e.to_string());
            match worker {
                Ok(worker) => player.workers.push(worker),
                // One worker is enough to play it (more only keep it ahead).
                Err(e) if player.workers.is_empty() => {
                    return Err(e);
                }
                Err(_) => break,
            }
        }
        Ok(player)
    }

    /// A present `elapsed` seconds after the first: the newest frame due, if decoded.
    fn step(&mut self, elapsed: f64) -> Step {
        let Some(due) = frame_at(elapsed, self.plan.fps, self.plan.total()) else {
            return Step::End(None);
        };
        self.due = due;
        let mut ring = self.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        ring.due = due;
        // A frame that could not be read ends it once it is due, after those before it.
        if let Some((failed, _)) = &ring.failed
            && due >= *failed
        {
            let why = ring.failed.take().map(|(_, why)| why);
            ring.ready.clear();
            drop(ring);
            self.shared.wake.notify_all();
            return Step::End(why);
        }
        let later = ring.ready.split_off(&(due + 1));
        let newest = std::mem::replace(&mut ring.ready, later).into_iter().next_back();
        drop(ring);
        self.shared.wake.notify_all();
        match newest {
            Some((n, frame)) => {
                self.shown += 1;
                Step::Show(n, frame)
            }
            None => Step::Keep,
        }
    }

    /// Asks the workers to stop; each does after the frame it is decoding.
    fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        self.shared.ring.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.shared.wake.notify_all();
    }

    /// Frames shown, frames skipped (due and never shown), the slowest decode, the most bytes the ring held.
    fn counts(&self) -> (u64, u64, f64, usize) {
        let ring = self.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        let reached = (self.due + 1).min(self.plan.total());
        (self.shown, reached.saturating_sub(self.shown), ring.slowest_decode_ms, ring.most)
    }
}

impl Drop for Player {
    /// The workers stop however the player goes (an unwinding panic of the
    /// loading cover's presenter too), not only by [`Player::stop`].
    fn drop(&mut self) {
        if !self.stopped.load(Ordering::Relaxed) {
            self.stop();
        }
    }
}

/// The video while it plays.
struct Video {
    player: Player,
    /// When its first frame was shown: its clock starts there (frame 0 is
    /// decoded once the first present has asked for the video).
    began: Option<Instant>,
    /// Made at the first frame shown, for the back buffer then.
    buffers: Option<(Buffers, Fit)>,
    rows: Rows,
    slowest_scale_ms: f64,
}

impl Video {
    /// A present at `elapsed` seconds into a back buffer `out` of `format`:
    /// the next frame written if one is due. `Err`: the video is over.
    fn frame(&mut self, host: &Host, out: (u32, u32), format: DXGI_FORMAT, elapsed: f64) -> Result<(), Option<String>> {
        match self.player.step(elapsed) {
            Step::End(why) => Err(why),
            Step::Keep => Ok(()),
            Step::Show(n, frame) => {
                let first = self.buffers.is_none();
                if self.buffers.as_ref().is_some_and(|(b, f)| (b.format, f.out) != (format, out)) {
                    return Err(Some(format!("the window changed to {} x {}", out.0, out.1)));
                }
                if self.buffers.is_none() {
                    let buffers = Buffers::new(host, format, out, UPLOADS).ok_or_else(|| Some("no buffers for it".to_owned()))?;
                    self.buffers = Some((buffers, fit(self.player.plan.size, out)));
                }
                let Some((buffers, fit)) = &mut self.buffers else { return Ok(()) };
                let began = Instant::now();
                let pitch = buffers.pitch as usize;
                write_frame(&frame, fit, format, buffers.next(), pitch, &mut self.rows);
                self.slowest_scale_ms = self.slowest_scale_ms.max(began.elapsed().as_secs_f64() * 1e3);
                self.began.get_or_insert_with(Instant::now);
                if first {
                    let (processor_bytes, card_bytes) = self.held();
                    tracing::debug!(frame = n, after_ms = elapsed * 1e3, processor_bytes, card_bytes, "boot video: first frame up");
                }
                Ok(())
            }
        }
    }

    /// The bytes held: on the processor's side (the ring and the rows), and for the card.
    fn held(&self) -> (u64, u64) {
        let ring = self.player.shared.ring.lock().unwrap_or_else(|e| e.into_inner());
        let cpu = ring.ready.values().map(|i| i.pixels.capacity()).sum::<usize>() + self.rows.bytes();
        (cpu as u64, self.buffers.as_ref().map_or(0, |(b, _)| b.bytes()))
    }

    /// Over: the workers asked to stop, one line said, and what is left to let go.
    fn end(self, why: Option<String>) -> Ending {
        self.player.stop();
        let (shown, skipped, decode_ms, ring_most) = self.player.counts();
        let card = self.held().1;
        let name = self.player.plan.name();
        // The processor's side: the most the ring held, and the rows.
        let fields = (shown, skipped, decode_ms, self.slowest_scale_ms, ring_most + self.rows.bytes(), card);
        match why {
            None => {
                tracing::info!(
                    video = %name, shown = fields.0, skipped = fields.1, slowest_decode_ms = fields.2, slowest_scale_ms = fields.3,
                    processor_bytes = fields.4, card_bytes = fields.5, "boot video: played"
                );
                // Played to its end: the loading cover may watch from here.
                cover::boot_ended(true);
            }
            Some(why) => {
                tracing::warn!(
                    video = %name, shown = fields.0, skipped = fields.1, slowest_decode_ms = fields.2, slowest_scale_ms = fields.3,
                    processor_bytes = fields.4, card_bytes = fields.5, why = %why, "boot video: ended early"
                );
                cover::boot_ended(false);
            }
        }
        let mut player = self.player;
        Ending { workers: std::mem::take(&mut player.workers), buffers: self.buffers.map(|(b, _)| b), left: LEAVE }
    }
}

/// The video over: kept until the workers have stopped and the card is done with the buffers.
struct Ending {
    workers: Vec<JoinHandle<()>>,
    buffers: Option<Buffers>,
    left: u32,
}

impl Ending {
    /// A present: true once everything is let go.
    fn tick(&mut self) -> bool {
        while let Some(i) = self.workers.iter().position(JoinHandle::is_finished) {
            let _ = self.workers.swap_remove(i).join();
        }
        self.left = self.left.saturating_sub(1);
        if self.left == 0
            && let Some(buffers) = self.buffers.take()
        {
            super::retire(buffers);
        }
        self.workers.is_empty() && self.buffers.is_none()
    }
}

enum State {
    Playing(Box<Video>),
    Ending(Ending),
}

/// Not looked at yet, off (unset, not there, or over and let go), or playing or ending.
const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
static PHASE: AtomicU8 = AtomicU8::new(UNKNOWN);
static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Whether the video is off for good (the presents' first look).
fn off() -> bool {
    PHASE.load(Ordering::Relaxed) == OFF
}

/// A present (the back buffer `back`, `width` by `height`, in the copy
/// destination state): the video's frame over the title's picture while it plays.
pub(super) fn draw(host: &Host, list: &HostList, back: &ID3D12Resource, (width, height): (u32, u32), format: DXGI_FORMAT) {
    if off() {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if PHASE.load(Ordering::Relaxed) == UNKNOWN {
        let started = match plan_from(std::env::var_os("VICTORY_BOOT_VIDEO"), std::env::var_os("VICTORY_BOOT_VIDEO_LOOPS")) {
            None => None,
            Some(Err(e)) => {
                tracing::warn!(error = %e, "boot video: not played");
                None
            }
            Some(Ok(plan)) => {
                tracing::info!(video = %plan.name(), frames = plan.frames, fps = plan.fps, w = plan.size.0, h = plan.size.1, loops = plan.loops, "boot video: playing");
                match Player::start(plan) {
                    Ok(player) => {
                        // With the cover asked for, its keeper starts now, parked until the video ends.
                        cover::boot_starts(&player.plan);
                        Some(player)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "boot video: no worker; not played");
                        None
                    }
                }
            }
        };
        let Some(player) = started else {
            PHASE.store(OFF, Ordering::Relaxed);
            return;
        };
        *state = Some(State::Playing(Box::new(Video { player, began: None, buffers: None, rows: Rows::default(), slowest_scale_ms: 0.0 })));
        PHASE.store(ON, Ordering::Relaxed);
    }
    match state.take() {
        Some(State::Playing(mut video)) => {
            let elapsed = video.began.map_or(0.0, |b| b.elapsed().as_secs_f64());
            match video.frame(host, (width, height), format, elapsed) {
                Ok(()) => {
                    if let Some((buffers, _)) = &mut video.buffers {
                        buffers.copy_out(list, back, (width.min(buffers.size.0), height.min(buffers.size.1)), (0, 0));
                    }
                    *state = Some(State::Playing(video));
                }
                Err(why) => *state = Some(State::Ending(video.end(why))),
            }
        }
        Some(State::Ending(mut ending)) => {
            if ending.tick() {
                PHASE.store(OFF, Ordering::Relaxed);
            } else {
                *state = Some(State::Ending(ending));
            }
        }
        None => PHASE.store(OFF, Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    /// A folder of `n` frames `w` by `h`, frame `i` all of colour `[40 * i, 255 - 40 * i, 7]`, at 24 a second.
    fn folder(name: &str, n: u32, (w, h): (u32, u32)) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("victory-boot-video-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("frames")).expect("a folder for the test");
        let json = format!("{{\n  \"frames\": {n},\n  \"fps\": 24,\n  \"width\": {w},\n  \"height\": {h},\n  \"pattern\": \"frames/%04d.qoi\",\n  \"first\": 1\n}}\n");
        std::fs::write(dir.join("video.json"), json).expect("video.json");
        for i in 0..n {
            let pixels = [(40 * i) as u8, (255 - 40 * i) as u8, 7].repeat((w * h) as usize);
            let data = qoi::tests::encode(w, h, 3, &pixels);
            std::fs::write(dir.join(format!("frames/{:04}.qoi", i + 1)), data).expect("a frame");
        }
        dir
    }

    fn colour(i: u64) -> [u8; 3] {
        [(40 * i) as u8, (255 - 40 * i) as u8, 7]
    }

    /// Steps until a frame is shown or the video ends (the workers are threads of their own).
    fn shown_at(player: &mut Player, elapsed: f64) -> Step {
        let began = Instant::now();
        loop {
            match player.step(elapsed) {
                Step::Keep if began.elapsed() < Duration::from_secs(10) => std::thread::sleep(Duration::from_millis(1)),
                step => return step,
            }
        }
    }

    #[test]
    fn the_frame_for_a_time() {
        let fps = 24.0;
        assert_eq!(frame_at(0.0, fps, 241), Some(0));
        assert_eq!(frame_at(1.5 / fps, fps, 241), Some(1));
        assert_eq!(frame_at(240.5 / fps, fps, 241), Some(240));
        assert_eq!(frame_at(241.0 / fps, fps, 241), None);
        assert_eq!(frame_at(100.0, fps, 241), None);
        // Two loops: the second plays the files from the first again.
        let plan = Plan::parse(Path::new("v"), "{\"frames\": 241, \"fps\": 24, \"width\": 1920, \"height\": 1080, \"pattern\": \"frames/%04d.qoi\", \"first\": 1}", 2).expect("a plan");
        assert_eq!(plan.total(), 482);
        assert_eq!(frame_at(241.5 / fps, fps, plan.total()), Some(241));
        assert_eq!(plan.file(241), Path::new("v").join("frames/0001.qoi"));
        assert_eq!(plan.file(240), Path::new("v").join("frames/0241.qoi"));
        assert_eq!(frame_at(482.0 / fps, fps, plan.total()), None);
        assert_eq!(numbered("f%d.qoi", 7).as_deref(), Some("f7.qoi"));
    }

    #[test]
    fn the_picture_is_letterboxed() {
        // As wide as the back buffer, or as high: the shape kept, in the middle.
        assert_eq!(fit((1920, 1080), (3840, 2160)), Fit { out: (3840, 2160), at: (0, 0), size: (3840, 2160) });
        assert_eq!(fit((1920, 1080), (2560, 1080)), Fit { out: (2560, 1080), at: (320, 0), size: (1920, 1080) });
        assert_eq!(fit((1920, 1080), (1080, 1350)), Fit { out: (1080, 1350), at: (0, 371), size: (1080, 608) });
        // Written: black around the picture, the picture's colour inside.
        let frame = qoi::Image { width: 4, height: 2, channels: 3, pixels: [200u8, 100, 50].repeat(4 * 2) };
        let f = fit((4, 2), (8, 6));
        let pitch = 8 * 4;
        let mut out = vec![9u8; pitch * 6];
        write_frame(&frame, &f, DXGI_FORMAT_R8G8B8A8_UNORM, &mut out, pitch, &mut Rows::default());
        let px = |x: usize, y: usize| out[y * pitch + x * 4..y * pitch + x * 4 + 4].to_vec();
        assert_eq!(f, Fit { out: (8, 6), at: (0, 1), size: (8, 4) });
        assert_eq!(px(3, 0), [0, 0, 0, 255]);
        assert_eq!(px(3, 3), [200, 100, 50, 255]);
        assert_eq!(px(7, 5), [0, 0, 0, 255]);
    }

    #[test]
    fn four_frames_made_here_play_in_order() {
        let dir = folder("order", 4, (8, 4));
        let plan = Plan::read(&dir, 1).expect("the plan");
        assert_eq!(plan.name(), dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let mut player = Player::start(plan).expect("a worker");
        for i in 0..4u64 {
            match shown_at(&mut player, (i as f64 + 0.5) / 24.0) {
                Step::Show(n, frame) => {
                    assert_eq!(n, i);
                    assert_eq!(&frame.pixels[..3], &colour(i));
                }
                _ => panic!("frame {i} was not shown"),
            }
        }
        assert!(matches!(player.step(4.5 / 24.0), Step::End(None)));
        assert_eq!(player.counts().0, 4);
        player.stop();
        for worker in std::mem::take(&mut player.workers) {
            let _ = worker.join();
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_frame_ends_the_video_without_a_panic() {
        let dir = folder("missing", 4, (8, 4));
        std::fs::remove_file(dir.join("frames/0003.qoi")).expect("the third frame removed");
        let mut player = Player::start(Plan::read(&dir, 1).expect("the plan")).expect("a worker");
        assert!(matches!(shown_at(&mut player, 0.5 / 24.0), Step::Show(0, _)));
        assert!(matches!(shown_at(&mut player, 1.5 / 24.0), Step::Show(1, _)));
        match shown_at(&mut player, 2.5 / 24.0) {
            Step::End(Some(why)) => assert!(why.starts_with("0003.qoi"), "{why}"),
            _ => panic!("the video did not end at the missing frame"),
        }
        // The workers have stopped by themselves.
        assert!(!player.workers.is_empty());
        for worker in std::mem::take(&mut player.workers) {
            assert!(worker.join().is_ok());
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unset_nothing_is_read_and_a_missing_folder_plays_nothing() {
        assert_eq!(plan_from(None, None), None);
        assert_eq!(plan_from(Some(OsString::new()), None), None);
        let nowhere = std::env::temp_dir().join(format!("victory-boot-video-none-{}", std::process::id()));
        assert!(matches!(plan_from(Some(nowhere.into_os_string()), None), Some(Err(_))));
        // The presents' look while it is off: one atomic.
        PHASE.store(OFF, Ordering::Relaxed);
        assert!(off());
    }

    /// Plays a video of `times.len()` frames at 24 a second whose frame `n`
    /// takes `times[n]` ms to decode, with `workers` decoding and a present
    /// every `present_ms`, all `scale` times as fast as in a run: frames shown
    /// and skipped, and the most bytes the ring held.
    fn replay(times: &[f64], workers: usize, present_ms: f64, scale: f64) -> (u64, u64, usize) {
        let plan = Plan {
            dir: PathBuf::from("replay"),
            frames: times.len() as u32,
            fps: 24.0,
            size: (2, 2),
            pattern: "%d".to_owned(),
            first: 1,
            loops: 1,
        };
        let times = times.to_vec();
        let decode: Arc<Decode> = Arc::new(move |n| {
            std::thread::sleep(Duration::from_secs_f64(times[n as usize % times.len()] / 1e3 / scale));
            Ok(qoi::Image { width: 2, height: 2, channels: 3, pixels: vec![0; 12] })
        });
        let mut player = Player::start_with(plan, workers, decode).expect("workers");
        // The clock starts at the first frame shown, as `draw`'s does.
        let mut began: Option<Instant> = None;
        loop {
            match player.step(began.map_or(0.0, |b| b.elapsed().as_secs_f64() * scale)) {
                Step::End(_) => break,
                Step::Show(..) => {
                    began.get_or_insert_with(Instant::now);
                }
                Step::Keep => {}
            }
            std::thread::sleep(Duration::from_secs_f64(present_ms / 1e3 / scale));
        }
        player.stop();
        let (shown, skipped, _, most) = player.counts();
        for worker in std::mem::take(&mut player.workers) {
            let _ = worker.join();
        }
        (shown, skipped, most)
    }

    /// The rules of [`work`] and [`Player::step`] in a clock of their own
    /// (no threads, so the answer does not depend on the machine's load):
    /// `workers` take the next frame, from the one due on, while the frames
    /// decoded and being decoded are fewer than [`RING`]; frame `n` takes
    /// `times[n]` ms; a present comes at each of `presents` (ms) and takes the
    /// newest frame due; the clock starts at the first frame shown. Frames shown and skipped.
    fn simulate(times: &[f64], workers: usize, presents: &[f64]) -> (u64, u64) {
        let total = times.len() as u64;
        let (mut ready, mut busy) = (BTreeMap::<u64, ()>::new(), vec![None::<(u64, f64)>; workers]);
        let (mut next, mut due, mut shown, mut began) = (0u64, 0u64, 0u64, None::<f64>);
        let mut presents = presents.iter().copied().peekable();
        let mut t = 0.0;
        while let Some(&at) = presents.peek() {
            // Decodes finished by now, then free workers take frames.
            for slot in &mut busy {
                if let Some((n, done)) = *slot
                    && done <= t
                {
                    ready.insert(n, ());
                    *slot = None;
                }
            }
            for i in 0..workers {
                let decoding = busy.iter().filter(|b| b.is_some()).count();
                if busy[i].is_none() && ready.len() + decoding < RING {
                    let n = next.max(due);
                    if n < total {
                        busy[i] = Some((n, t + times[n as usize]));
                        next = n + 1;
                    }
                }
            }
            if at <= t {
                presents.next();
                let elapsed = began.map_or(0.0, |b| t - b) / 1e3;
                let Some(now) = frame_at(elapsed, 24.0, total) else { break };
                due = now;
                let later = ready.split_off(&(due + 1));
                if std::mem::replace(&mut ready, later).into_keys().next_back().is_some() {
                    shown += 1;
                    began.get_or_insert(t);
                }
            }
            t += 0.1;
        }
        (shown, (due + 1).min(total) - shown)
    }

    /// Decode times like those of the real frames in a loaded start: the real
    /// frames took 31 ms (median) here, and up to 41.9 ms in the start that
    /// skipped; 36.6 to 41.9 ms in a fixed order.
    fn loaded_times() -> Vec<f64> {
        (0..241u32).map(|n| 36.6 + f64::from((n * 37) % 54) / 10.0).collect()
    }

    /// Presents every `every` ms for `seconds`, after a first one at 0.
    fn steady(every: f64, seconds: f64) -> Vec<f64> {
        (0..).map(|i| f64::from(i) * every).take_while(|t| *t < seconds * 1e3).collect()
    }

    #[test]
    fn decode_times_of_a_loaded_start_skip_no_frame() {
        let times = loaded_times();
        assert!(times.iter().copied().fold(0.0, f64::max) >= 41.7, "some decodes are longer than a frame");
        assert_eq!(simulate(&times, WORKERS, &steady(1000.0 / 60.0, 12.0)), (241, 0));
        assert_eq!(simulate(&times, WORKERS, &steady(1000.0 / 30.0, 12.0)), (241, 0));
        // Every decode longer than a frame: one worker falls behind and skips, two do not.
        let slow = vec![45.0; 241];
        assert!(simulate(&slow, 1, &steady(1000.0 / 60.0, 14.0)).1 > 0);
        assert_eq!(simulate(&slow, WORKERS, &steady(1000.0 / 60.0, 14.0)), (241, 0));
        // What decoding cannot help: the title presenting nothing for a while
        // (the start that skipped had presents at 0, 229 and 682 ms): frames due
        // in the gap are passed over (the clock is the wall clock).
        let gap: Vec<f64> = [0.0, 229.0].into_iter().chain(steady(1000.0 / 60.0, 12.0).into_iter().map(|t| t + 682.0)).collect();
        let (shown, skipped) = simulate(&times, WORKERS, &gap);
        assert!(skipped > 0 && skipped < 12, "{shown} shown, {skipped} skipped");
    }

    /// The real frames' decode times (`VICTORY_BOOT_VIDEO`), measured, then
    /// played back with one worker and with [`WORKERS`], as they are and as
    /// slow as in the loaded start (their slowest 41.9 ms): skips and the
    /// ring's bytes. `cargo test --profile diag -p victory-gpu --lib
    /// bootvideo::tests::replay_of_the_real_frames -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, in an optimised build, with the real frames"]
    fn replay_of_the_real_frames() {
        let Some(dir) = std::env::var_os("VICTORY_BOOT_VIDEO") else {
            println!("VICTORY_BOOT_VIDEO unset: nothing to measure");
            return;
        };
        let plan = Plan::read(Path::new(&dir), 1).expect("the plan");
        let mut times = Vec::new();
        let mut bytes = 0;
        for n in 0..u64::from(plan.frames) {
            let t = Instant::now();
            let image = decode_file(&plan, n).expect("a frame");
            times.push(t.elapsed().as_secs_f64() * 1e3);
            bytes = bytes.max(image.pixels.capacity());
        }
        let slowest = times.iter().copied().fold(0.0, f64::max);
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        println!("{} frames decoded: median {:.1} ms, slowest {slowest:.1} ms; a frame {bytes} bytes", times.len(), sorted[times.len() / 2]);
        let loaded: Vec<f64> = times.iter().map(|t| t * 41.9 / slowest).collect();
        for (what, times) in [("as measured", &times), ("as slow as the loaded start", &loaded)] {
            for workers in [1, WORKERS] {
                let (shown, skipped, most) = replay(times, workers, 1000.0 / 60.0, 1.0);
                // (The replay's frames are 2 by 2, 12 bytes each.)
                println!("{what}, {workers} worker(s): {shown} shown, {skipped} skipped; the ring held at most {} frames ({} bytes of these)", most / 12, most / 12 * bytes);
            }
        }
    }

    #[test]
    fn nothing_is_held_after_the_end() {
        let Some(host) = super::super::host() else {
            eprintln!("no host device: not run");
            return;
        };
        let dir = folder("held", 3, (16, 9));
        let player = Player::start(Plan::read(&dir, 1).expect("the plan")).expect("a worker");
        let mut video = Video { player, began: None, buffers: None, rows: Rows::default(), slowest_scale_ms: 0.0 };
        let out = (64, 36);
        let began = Instant::now();
        while video.buffers.is_none() && began.elapsed() < Duration::from_secs(10) {
            video.frame(host, out, DXGI_FORMAT_B8G8R8A8_UNORM, 0.01).expect("playing");
            std::thread::sleep(Duration::from_millis(1));
        }
        let (cpu, card) = video.held();
        assert!(cpu > 0 && card == 256 * 36 * (UPLOADS as u64 + 1), "{cpu} {card}");
        // The frame shown, in the upload buffer: frame 0's colour, as blue, green, red.
        let (buffers, _) = video.buffers.as_ref().expect("buffers");
        // SAFETY: the upload buffer just written.
        let written = unsafe { std::slice::from_raw_parts(buffers.uploads[buffers.turn].1, 4) };
        assert_eq!(written, [7, 255, 0, 255]);
        let why = video.frame(host, out, DXGI_FORMAT_B8G8R8A8_UNORM, 1.0).expect_err("over");
        assert_eq!(why, None);
        let mut ending = video.end(why);
        let began = Instant::now();
        let mut presents = 0;
        while !ending.tick() {
            presents += 1;
            assert!(began.elapsed() < Duration::from_secs(10), "not let go");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(presents + 1 >= LEAVE as usize, "kept {LEAVE} presents for the card");
        assert!(ending.buffers.is_none() && ending.workers.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// What the video costs: each frame read and decoded, and scaled and written
    /// at 3840 by 2160 for each format, with the menu's box filter beside it.
    /// With `VICTORY_BOOT_VIDEO`: its frames (else frames made here). Run in an
    /// optimised profile:
    /// `cargo test --profile diag -p victory-gpu --lib bootvideo::tests::cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, in an optimised build"]
    fn cost() {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let dir = std::env::var_os("VICTORY_BOOT_VIDEO").map_or_else(|| folder("cost", 4, (1920, 1080)), PathBuf::from);
        let plan = Plan::read(&dir, 1).expect("the plan");
        let mut decode = Vec::new();
        let mut frames = Vec::new();
        for n in (0..u64::from(plan.frames)).step_by((plan.frames as usize / 12).max(1)) {
            let t = Instant::now();
            let image = qoi::decode(&std::fs::read(plan.file(n)).expect("a frame")).expect("decoded");
            decode.push(ms(t));
            frames.push(image);
        }
        decode.sort_by(f64::total_cmp);
        println!("{} frames read and decoded: median {:.1} ms, slowest {:.1} ms", decode.len(), decode[decode.len() / 2], decode[decode.len() - 1]);
        let src = plan.size;
        for out in [(3840, 2160), (2560, 1440), (1920, 1080)] {
            let f = fit(src, out);
            for format in [DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT] {
                let pitch = (out.0 * picture::pixel_bytes(format).unwrap_or(4)).next_multiple_of(256) as usize;
                let mut buffer = vec![0u8; pitch * out.1 as usize];
                let mut rows = Rows::default();
                let mut times: Vec<f64> = frames
                    .iter()
                    .map(|frame| {
                        let t = Instant::now();
                        write_frame(frame, &f, format, &mut buffer, pitch, &mut rows);
                        ms(t)
                    })
                    .collect();
                times.sort_by(f64::total_cmp);
                println!(
                    "{out:?} {format:?}: scaled and written median {:.2} ms, slowest {:.2} ms; card's buffers {} bytes",
                    times[times.len() / 2],
                    times[times.len() - 1],
                    pitch * out.1 as usize * (UPLOADS + 1)
                );
            }
            if out == (3840, 2160) {
                let t = Instant::now();
                let step = (src.0 as f32 / out.0 as f32, src.1 as f32 / out.1 as f32);
                let boxed = picture::boxed(&frames[0].pixels, (src.0 as usize, src.1 as usize), 3, (out.0 as usize, out.1 as usize), step);
                println!("  the menu's box filter to {out:?}: {:.1} ms for one frame (before writing it out)", ms(t));
                drop(boxed);
            }
        }
        let ring = frames.iter().take(RING).map(|f| f.pixels.capacity()).sum::<usize>();
        println!("the ring of {RING} frames: {ring} bytes");
        PHASE.store(OFF, Ordering::Relaxed);
        let t = Instant::now();
        let mut n = 0u32;
        for _ in 0..10_000_000 {
            n += u32::from(std::hint::black_box(off()));
        }
        println!("unset, a present's look at the video: {:.2} ns ({n})", ms(t) * 1e6 / 1e7);
        if std::env::var_os("VICTORY_BOOT_VIDEO").is_none() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
