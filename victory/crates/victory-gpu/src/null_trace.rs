//! The null-read trace (`VICTORY_NULL_TRACE=1`; off by default; diagnostics
//! builds): per frame, how often a draw or dispatch reads zeros, or a slot
//! that is not its own, where no other line names the shader, with the
//! frame's first case of each kind as a sample. Logging only: nothing it
//! looks at is copied, bound or made otherwise.
//!
//! 1. A view table entry past the title slots the host view heap mirrors
//!    (`HostHeap::count`, fewer than the title's when the host caps the
//!    heap): `State::tables`. An entry the shader's used resources name
//!    makes its table one made afresh in dynamic slots; one a shader picks
//!    by index in a table not made afresh reads the host slot that lies
//!    there (a dynamic slot of another draw's, or the null slot).
//! 2. A texture slot passed over unmapped (`State::heap_textures`): its
//!    host slot is made null.
//! 3. A table entry a draw reads left null (`State::table_entry`): no
//!    view at the title's slot, or a buffer view the host refused (whose
//!    slot the mirror made null, though it gives the view back).
//! 4. A shader taking the address of a buffer view that is not there
//!    (`State::host_word`): it reads 0.

use std::sync::{Mutex, OnceLock};

/// What a counted case is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    PastHeap = 0,
    TextureAway = 1,
    BufferNull = 2,
    NoAddress = 3,
}

const KINDS: [Kind; 4] = [Kind::PastHeap, Kind::TextureAway, Kind::BufferNull, Kind::NoAddress];

impl Kind {
    fn what(self) -> &'static str {
        match self {
            Kind::PastHeap => "view table entries draws use past the host view heap",
            Kind::TextureAway => "texture slots passed over unmapped, made null",
            Kind::BufferNull => "table entries draws read left null (no view there, or a buffer view the host refused)",
            Kind::NoAddress => "addresses taken of buffer views that are not there, read as 0",
        }
    }
}

/// Whether the trace is on (read once).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NULL_TRACE").is_some_and(|v| v == "1"))
}

/// A frame's line for one kind: how many, and the first.
#[derive(Debug, PartialEq, Eq)]
struct Line {
    frame: u64,
    kind: Kind,
    count: u64,
    first: String,
}

/// The frame being counted: per kind, its count and first sample.
struct Tally {
    frame: Option<u64>,
    counts: [u64; 4],
    firsts: [Option<String>; 4],
}

impl Tally {
    const fn new() -> Self {
        Tally {
            frame: None,
            counts: [0; 4],
            firsts: [None, None, None, None],
        }
    }

    /// Counts one `kind` at `frame`; `sample` is made only for the frame's
    /// first of its kind. A new frame gives back the frame before's lines,
    /// one per kind it had.
    fn add(&mut self, frame: u64, kind: Kind, sample: impl FnOnce() -> String) -> Vec<Line> {
        let mut done = Vec::new();
        if self.frame != Some(frame) {
            if let Some(last) = self.frame {
                done = self.lines(last);
            }
            self.frame = Some(frame);
        }
        let k = kind as usize;
        if self.counts[k] == 0 {
            self.firsts[k] = Some(sample());
        }
        self.counts[k] += 1;
        done
    }

    /// The lines of `frame`, and counting begun again.
    fn lines(&mut self, frame: u64) -> Vec<Line> {
        KINDS
            .into_iter()
            .filter_map(|kind| {
                let k = kind as usize;
                let count = std::mem::take(&mut self.counts[k]);
                let first = self.firsts[k].take().unwrap_or_default();
                (count > 0).then_some(Line { frame, kind, count, first })
            })
            .collect()
    }
}

static TALLY: Mutex<Tally> = Mutex::new(Tally::new());

/// Counts one `kind` in the current frame when the trace is on; `sample`
/// (the pipeline's shaders and where) is made for the frame's first only.
/// The frame before's lines are logged when the first case of a later
/// frame comes ([`note_at`]).
pub(crate) fn note(kind: Kind, sample: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    note_at(crate::frame::current(), kind, sample);
}

/// [`note`] in `frame`. The frame before's lines are logged by the bounded
/// reporting thread (`crate::host::report`), not here: notes are made while
/// a draw is recorded, which holds the list's state, and a log that blocks
/// would hold it too (as world's indirect lines, f43659f7). When the
/// reporter is behind they are dropped and counted.
fn note_at(frame: u64, kind: Kind, sample: impl FnOnce() -> String) {
    let lines = TALLY.lock().unwrap_or_else(|e| e.into_inner()).add(frame, kind, sample);
    if !lines.is_empty() {
        crate::host::report(Box::new(move || log(lines)));
    }
}

/// Logs a frame's lines (tests also keep them, with the thread that logged
/// them, [`LOGGED`]).
fn log(lines: Vec<Line>) {
    for line in lines {
        tracing::info!(
            frame = line.frame,
            count = line.count,
            first = %line.first,
            "null trace: {}",
            line.kind.what()
        );
        #[cfg(test)]
        LOGGED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((std::thread::current().name().map(str::to_owned), line.first));
    }
}

/// Tests: each line's first case, with the name of the thread that logged it.
#[cfg(test)]
static LOGGED: Mutex<Vec<(Option<String>, String)>> = Mutex::new(Vec::new());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_gives_one_line_per_kind_with_its_count_and_first_case() {
        let mut t = Tally::new();
        let mut made = 0;
        let mut sample = |s: &str| {
            made += 1;
            s.to_owned()
        };
        assert!(t.add(7, Kind::TextureAway, || sample("slot 1")).is_empty());
        assert!(t.add(7, Kind::TextureAway, || sample("slot 2")).is_empty());
        assert!(t.add(7, Kind::NoAddress, || sample("root 3")).is_empty());
        assert!(t.add(7, Kind::TextureAway, || sample("slot 3")).is_empty());
        // Only the first of each kind in a frame is made.
        assert_eq!(made, 2);
        let lines = t.add(9, Kind::PastHeap, || "slot 800000".to_owned());
        assert_eq!(
            lines,
            [
                Line { frame: 7, kind: Kind::TextureAway, count: 3, first: "slot 1".into() },
                Line { frame: 7, kind: Kind::NoAddress, count: 1, first: "root 3".into() },
            ]
        );
        // The new frame counts afresh, its first case its own.
        let lines = t.add(10, Kind::BufferNull, || "root 0".to_owned());
        assert_eq!(lines, [Line { frame: 9, kind: Kind::PastHeap, count: 1, first: "slot 800000".into() }]);
    }

    /// A frame's lines are logged by the bounded reporting thread, not
    /// where the case that ends the frame is noted: a note is made while a
    /// draw is recorded, which holds the list's state, and a log that
    /// blocks would hold it too (world's f43659f7, for its indirect lines).
    #[test]
    fn lines_are_logged_by_the_reporter_not_where_they_are_noted() {
        let first = "the reporter's null-trace test";
        note_at(u64::MAX - 11, Kind::PastHeap, || first.to_owned());
        note_at(u64::MAX - 10, Kind::PastHeap, String::new);
        crate::host::reports_settled();
        let logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        let ours: Vec<_> = logged.iter().filter(|(_, f)| f == first).collect();
        assert_eq!(ours.len(), 1, "logged once: {logged:?}");
        assert_eq!(ours[0].0.as_deref(), Some("victory-reports"), "logged where it was noted");
    }

    #[test]
    fn kinds_without_cases_give_no_line() {
        let mut t = Tally::new();
        assert!(t.add(1, Kind::BufferNull, String::new).is_empty());
        let lines = t.add(2, Kind::BufferNull, String::new);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].kind, Kind::BufferNull);
        assert!(t.lines(2).iter().all(|l| l.kind == Kind::BufferNull && l.count == 1));
        assert!(t.lines(2).is_empty());
    }
}
