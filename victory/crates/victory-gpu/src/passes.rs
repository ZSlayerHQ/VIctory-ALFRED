//! Diagnostics: the card's time by the title's passes, and our own work's
//! share of it (`VICTORY_GPU_PASSES=1`, which turns the stamps of
//! [`crate::stamps`] on at level 1 too; off otherwise, and nothing is
//! stamped or counted then).
//!
//! A pass is a run of one host list's title calls of one kind: compute
//! dispatches, or draws and mesh dispatches after the same binding of
//! targets by the title (each binding, a view of another mip or slice too,
//! begins one). Before a call whose kind or binding differ from the pass
//! under way is recorded, before any work of ours for it, a timestamp ends
//! that pass and begins the next ([`crate::stamps::Stamps::pass`], at the top
//! of `draw::State::bind`); the list's opening and close are its own pair's
//! stamps. A pass counts only if a call of it was recorded
//! ([`crate::stamps::Stamps::pass_called`], after each draw or dispatch): one
//! whose call was dropped (its bind refused, or a draw found it could not
//! run) leaves its stretch in no pass, and the line says how many. So a
//! pass runs from the completion mark before its first call to the next
//! pass's, or to its list's close: queue time, with the stalls and
//! barriers between the marks and the work of ours recorded for its
//! calls, but our clamp groups and texture uploads in it left out (they
//! have pairs of their own, and count as ours). The stretch from the
//! list's opening to its first pass is in no pass, as is the stretch after
//! a mark that had no query heap, up to the next usable one; and a frame
//! with such a mark, or with an upload's pair lost, is left out.
//!
//! A pass is told apart by what names it, the fragment of its title list's
//! recording it is in (a list cut at its waits records in several, from 0)
//! and its place among the passes so named in that fragment (from 1).
//! What names it: the title's label of its list, by a hash of it (the
//! list's `SetName`: the label itself is never kept or printed, it may
//! hold a path or the title's name), else its first call's program (stage
//! and shader ids, at most four of them printed; `harness/name-hashes.mjs`
//! gives the ids names: the names never come into the runtime), else
//! nothing. The title records no debug events through any list method we
//! lack: an unwritten slot logs "method not implemented" once, and no run
//! has. Passes alike in all three in different lists are one group: each
//! pass is a sample of its own (none are summed), and the median is over
//! them.
//!
//! The marks are read with the rest of the list's stamps once its fence
//! has passed, a few frames late and never waited for ([`crate::stamps`]).
//! Every send counts in the frame it went in, ours too: a frame is the
//! time from one present to the next ([`presented`]), its number taken
//! under the book's lock, as the lines take frames. A frame presented since
//! whose sends were all read is taken, oldest first, every five seconds
//! ([`Book::take`]); one with a send unread after 10 s is given up and left
//! out, as is one with a flaw in any of its sends (a pair or mark lost or
//! out of place, a send let go unread or left out). The lines ([`lines`]):
//! the passes with the largest median, and one with the title's and our
//! own figures: spans summed, so where two lists' work overlaps on the
//! queue it counts in each, and our share is of the time stamped.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::stamps::{KINDS, Kind, Program};

/// What names a pass ([`crate::passes`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum By {
    /// A hash of the title's label of its list (FNV-1a of its UTF-16
    /// units).
    Label(u32),
    /// Its first call's program ([`crate::stamps::program`]).
    Program(Program),
    /// Nothing: its first call's pipeline has no shader ids.
    Unnamed,
}

/// What tells a pass apart ([`crate::passes`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct PassName {
    pub(crate) by: By,
    /// The fragment of its title list's recording (from 0).
    pub(crate) fragment: u32,
    /// Its place among the passes so named in that fragment (from 1).
    pub(crate) nth: u32,
}

/// What one send read, for its frame.
#[derive(Default, Debug)]
pub(crate) struct Reading {
    /// A title list's passes and their ticks, our clamp groups and
    /// texture uploads in them left out ([`parts`]).
    pub(crate) passes: Vec<(PassName, u64)>,
    /// Its ticks in no pass, the same left out.
    pub(crate) unattributed: u64,
    /// Passes begun whose call was not recorded.
    pub(crate) uncalled: u32,
    /// Ticks of our own work read, by [`Kind`] (summed spans).
    pub(crate) ours: [u64; KINDS],
    /// A flaw that leaves the send's frame out.
    pub(crate) flawed: bool,
}

/// A title list's span cut at its marks ([`parts`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Cut {
    pub(crate) passes: Vec<(PassName, u64)>,
    /// Ticks in no pass.
    pub(crate) none: u64,
    /// Marks that could not be used.
    pub(crate) bad: u32,
    /// Passes begun whose call was not recorded.
    pub(crate) uncalled: u32,
}

/// A title list's own span `(begun, end)` cut at the marks that begin its
/// passes (`marks` in recorded order: each the stamp's value, `None` where
/// it had no query heap, the pass's name, and its calls recorded), with the
/// ticks of `inside` (our pairs within the list) left out of whatever
/// stretch they lie in. A mark is used only if it lies inside the span and
/// not before the mark before it; an unusable one leaves the stretch from
/// the mark before it to the next usable one in no pass. A pass with no
/// call recorded is none: its stretch is in no pass.
pub(crate) fn parts(
    (begun, end): (u64, u64),
    marks: impl IntoIterator<Item = (Option<u64>, PassName, u32)>,
    inside: &[(u64, u64)],
) -> Cut {
    let mut stretches: Vec<(Option<PassName>, u64, u64)> = Vec::new();
    let (mut from, mut owner, mut bad, mut uncalled) = (begun, None, 0, 0);
    for (at, name, calls) in marks {
        match at {
            Some(at) if at >= from && at <= end => {
                stretches.push((owner.take(), from, at));
                if calls == 0 {
                    uncalled += 1;
                }
                (from, owner) = (at, (calls > 0).then_some(name));
            }
            _ => {
                bad += 1;
                owner = None;
            }
        }
    }
    stretches.push((owner, from, end));
    let ours = |a: u64, b: u64| inside.iter().map(|&(x, y)| y.min(b).saturating_sub(x.max(a))).sum::<u64>();
    let (mut passes, mut none) = (Vec::new(), 0);
    for (owner, a, b) in stretches {
        let ticks = (b - a).saturating_sub(ours(a, b));
        match owner {
            Some(name) => passes.push((name, ticks)),
            None => none += ticks,
        }
    }
    Cut { passes, none, bad, uncalled }
}

/// One frame's figures as its sends are read.
#[derive(Default, Debug)]
struct Frame {
    /// Each pass's ticks, by name.
    passes: BTreeMap<PassName, Vec<u64>>,
    unattributed: u64,
    uncalled: u64,
    ours: [u64; KINDS],
    flawed: bool,
    /// Its sends not read yet, and when its first went (CPU's counter).
    unread: u32,
    first: u64,
}

impl Frame {
    fn title(&self) -> u64 {
        self.passes.values().flatten().sum::<u64>() + self.unattributed
    }

    fn ours(&self) -> u64 {
        self.ours.iter().sum()
    }
}

/// The frames being read, and what the lines took.
#[derive(Default)]
pub(crate) struct Book {
    frames: BTreeMap<u64, Frame>,
    /// Frames up to this one were taken: sends of them after count late.
    taken: Option<u64>,
    late: u64,
    /// When the next lines are due (CPU's counter).
    next: Option<u64>,
}

impl Book {
    /// A send went at `now` (CPU's counter) in `frame`.
    fn sent(&mut self, frame: u64, now: u64) {
        if self.taken.is_some_and(|t| frame <= t) {
            self.late += 1;
            return;
        }
        let f = self.frames.entry(frame).or_default();
        if f.unread == 0 && f.first == 0 {
            f.first = now;
        }
        f.unread += 1;
    }

    /// A send of `frame` was read. One of a frame taken already is not.
    fn read(&mut self, frame: u64, reading: Reading) {
        if self.taken.is_some_and(|t| frame <= t) {
            return;
        }
        // Read with no send noted: not known to be whole.
        let f = self.frames.entry(frame).or_insert_with(|| Frame { flawed: true, ..Frame::default() });
        f.unread = f.unread.saturating_sub(1);
        f.flawed |= reading.flawed;
        for (name, ticks) in reading.passes {
            f.passes.entry(name).or_default().push(ticks);
        }
        f.unattributed += reading.unattributed;
        f.uncalled += u64::from(reading.uncalled);
        for (a, b) in f.ours.iter_mut().zip(reading.ours) {
            *a += b;
        }
    }

    /// A send of `frame` was let go unread: the frame is left out.
    fn lost(&mut self, frame: u64) {
        self.read(frame, Reading { flawed: true, ..Reading::default() });
    }

    /// The frames before `current` (the frame under way) whose sends were
    /// all read, oldest first, up to the first that still waits for one;
    /// one whose first send went before `stale` (CPU's counter) and still
    /// waits is given up, as flawed. Returns them, and how many were given
    /// up.
    fn take(&mut self, current: u64, stale: u64) -> (Vec<Frame>, u64) {
        let (mut out, mut given_up) = (Vec::new(), 0);
        while let Some((&n, f)) = self.frames.first_key_value() {
            if n >= current || (f.unread > 0 && f.first >= stale) {
                break;
            }
            let (_, mut f) = self.frames.pop_first().expect("a frame");
            if f.unread > 0 {
                f.flawed = true;
                given_up += 1;
            }
            out.push(f);
            self.taken = Some(n);
        }
        (out, given_up)
    }
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

#[cfg(test)]
thread_local! {
    /// Tests: what this thread's sends give goes here.
    static SINK: std::cell::RefCell<Option<Book>> = const { std::cell::RefCell::new(None) };
}

fn with_book<R>(f: impl FnOnce(&mut Book) -> R) -> R {
    #[cfg(test)]
    {
        let mut f = Some(f);
        let mine = SINK.with(|sink| sink.borrow_mut().as_mut().map(|book| (f.take().expect("once"))(book)));
        if let Some(r) = mine {
            return r;
        }
        (f.take().expect("once"))(BOOK.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Book::default))
    }
    #[cfg(not(test))]
    f(BOOK.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Book::default))
}

/// A send went at `now` (CPU's counter): the frame it counts in, taken
/// under the book's lock.
pub(crate) fn sent(now: u64) -> u64 {
    with_book(|book| {
        let frame = frame();
        book.sent(frame, now);
        frame
    })
}

/// A send of `frame` was read.
pub(crate) fn read(frame: u64, reading: Reading) {
    with_book(|book| book.read(frame, reading));
}

/// A send of `frame` was let go unread.
pub(crate) fn lost(frame: u64) {
    with_book(|book| book.lost(frame));
}

/// Title lists the title labelled (`SetName`) so far.
pub(crate) static LABELLED: AtomicU64 = AtomicU64::new(0);

/// Frames presented so far: a send goes in the frame its number says.
static PRESENTED: AtomicU64 = AtomicU64::new(0);

/// A frame was presented.
pub(crate) fn presented() {
    PRESENTED.fetch_add(1, Ordering::Relaxed);
}

/// The frame under way: the frames presented so far.
pub(crate) fn frame() -> u64 {
    PRESENTED.load(Ordering::Relaxed)
}

/// A hash of a label's UTF-16 units (FNV-1a), which names its passes.
pub(crate) fn label_hash(units: impl IntoIterator<Item = u16>) -> u32 {
    units.into_iter().fold(0x811c_9dc5u32, |h, u| (h ^ u32::from(u)).wrapping_mul(0x0100_0193))
}

/// Every five seconds (`now` the CPU's counter, `cpu_frequency` its ticks a
/// second, `frequency` the card's): the frames read since, by pass.
pub(crate) fn line(now: u64, cpu_frequency: u64, frequency: u64) {
    let taken = with_book(|book| {
        let due = *book.next.get_or_insert(now + 5 * cpu_frequency);
        if now < due {
            return None;
        }
        book.next = Some(now + 5 * cpu_frequency);
        let (frames, given_up) = book.take(frame(), now.saturating_sub(10 * cpu_frequency));
        Some((frames, given_up, std::mem::take(&mut book.late)))
    });
    let Some((frames, given_up, late)) = taken else {
        return;
    };
    let labelled = LABELLED.load(Ordering::Relaxed);
    for line in lines(&frames, given_up, late, labelled, frequency, &name_text) {
        tracing::info!("{line}");
    }
}

/// A pass's name for its line: fixed words, hex numbers and shader ids.
fn name_text(name: &PassName) -> String {
    let by = match name.by {
        By::Label(hash) => format!("named by the title's label of its list, hashed #{hash:08x} (the label is never printed)"),
        By::Program(p) => match crate::stamps::program_text(p) {
            Some((stage, shaders)) => format!("named by its first program: stage {stage}, shaders={}", bounded(&shaders)),
            None => format!("named by its first program: number {p}, unknown"),
        },
        By::Unnamed => "unnamed (its first call's pipeline has no shader ids)".to_owned(),
    };
    format!("{by}, fragment {} of its list's recording, its pass {} so named there", name.fragment, name.nth)
}

/// A pipeline's shader ids as `[stage id, ...]`, at most four of them and
/// 200 characters: what is past them is counted, not printed.
pub(crate) fn bounded(shaders: &str) -> String {
    let inner = shaders.trim_start_matches('[').trim_end_matches(']');
    let all: Vec<&str> = inner.split(", ").filter(|s| !s.is_empty()).collect();
    let mut text: String = all.iter().take(4).copied().collect::<Vec<_>>().join(", ").chars().take(200).collect();
    if all.len() > 4 {
        text.push_str(&format!(", and {} more", all.len() - 4));
    }
    format!("[{text}]")
}

/// Passes the lines rank.
const TOP: usize = 20;

/// The middle of `values` (sorted here): the mean of the two middle ones
/// for an even count.
fn median(values: &mut [u64]) -> f64 {
    values.sort_unstable();
    match values.len() {
        0 => 0.0,
        n if n % 2 == 1 => values[n / 2] as f64,
        n => (values[n / 2 - 1] + values[n / 2]) as f64 / 2.0,
    }
}

/// The lines for `frames` (`given_up` of them given up unread, `late`
/// sends that went for frames already taken, `labelled` title lists the
/// title labelled, `frequency` the card's ticks a second, `name` a pass's
/// name in words): one a name for the [`TOP`] with the largest median of
/// their passes, then one with the title's and our own figures. With no
/// frame to count, one line that says why.
fn lines(frames: &[Frame], given_up: u64, late: u64, labelled: u64, frequency: u64, name: &dyn Fn(&PassName) -> String) -> Vec<String> {
    use std::fmt::Write;
    const HEAD: &str = "card time by pass (VICTORY_GPU_PASSES=1)";
    let counted: Vec<&Frame> = frames.iter().filter(|f| !f.flawed).collect();
    let left_out = frames.len() - counted.len();
    let mut why = format!(
        "{left_out} frames left out (a stamp or mark lost or out of place, or a send of theirs left out, let go or unread: \
         {given_up} of them given up after 10 s)"
    );
    if late > 0 {
        let _ = write!(why, ", {late} sends that went for frames already counted");
    }
    if counted.is_empty() {
        return vec![format!("{HEAD}: no frames to count since the last line: {why}; no figures")];
    }
    let m = counted.len();
    let ms = |ticks: f64| ticks * 1e3 / frequency.max(1) as f64;
    // By name: each pass's ticks, and the frames it ran in.
    let mut by: BTreeMap<&PassName, (Vec<u64>, usize)> = BTreeMap::new();
    for f in &counted {
        for (n, ticks) in &f.passes {
            let e = by.entry(n).or_default();
            e.0.extend_from_slice(ticks);
            e.1 += 1;
        }
    }
    let mut ranked: Vec<(&PassName, f64, usize, usize, u64)> = by
        .into_iter()
        .map(|(n, (mut ticks, ran))| {
            let sum = ticks.iter().sum::<u64>();
            (n, median(&mut ticks), ticks.len(), ran, sum)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.4.cmp(&a.4)).then(a.0.cmp(b.0)));
    let mut out = Vec::with_capacity(TOP + 1);
    for (rank, &(n, mid, passes, ran, sum)) in ranked.iter().take(TOP).enumerate() {
        out.push(format!(
            "{HEAD}: rank {}: {:.3} ms median a pass over {passes} passes in {ran} of {m} frames, {:.3} ms a frame over all {m}, \
             {}",
            rank + 1,
            ms(mid),
            ms(sum as f64 / m as f64),
            name(n)
        ));
    }
    let mut title: Vec<u64> = counted.iter().map(|f| f.title()).collect();
    let mut ours: Vec<u64> = counted.iter().map(|f| f.ours()).collect();
    let (title_sum, ours_sum) = (title.iter().sum::<u64>(), ours.iter().sum::<u64>());
    let kind = |k: Kind| ms(counted.iter().map(|f| f.ours[k as usize]).sum::<u64>() as f64 / m as f64);
    let passes: usize = counted.iter().flat_map(|f| f.passes.values()).map(Vec::len).sum();
    let uncalled: u64 = counted.iter().map(|f| f.uncalled).sum();
    let share = if title_sum + ours_sum > 0 { ours_sum as f64 * 100.0 / (title_sum + ours_sum) as f64 } else { 0.0 };
    out.push(format!(
        "{HEAD}: over {m} frames ({why}); title lists {:.2} ms median a frame, {:.1} passes a frame under {} names, {:.3} ms a \
         frame of it in no pass, {:.1} passes a frame begun whose call was not recorded (in no pass); our own work {:.2} ms \
         median a frame, as a mean a frame: clamp groups {:.2}, texture uploads {:.2}, preludes {:.2}, relay fetches {:.2}, \
         present copies {:.2}, hash lists {:.2}, compare lists {:.2}, synchronous fetches {:.2}; ours {share:.1}% of the time \
         stamped (ours over ours and the title lists', spans summed over the frames counted: work that overlaps on the queue \
         counts in each, and work not stamped is in neither); passes are named by a hash of the title's label of their list \
         where it set one ({labelled} lists labelled so far), else by their first program, and told apart by fragment and \
         place; a pass is a run of one list's calls of one kind (compute, or drawing after one binding of targets), from the \
         completion mark before its first call to the next pass's or its list's close: queue time with the stalls and \
         barriers between the marks, our clamp groups and texture uploads in it left out; a median is over the passes of a \
         name, each its own sample",
        ms(median(&mut title)),
        passes as f64 / m as f64,
        ranked.len(),
        ms(counted.iter().map(|f| f.unattributed).sum::<u64>() as f64 / m as f64),
        uncalled as f64 / m as f64,
        ms(median(&mut ours)),
        kind(Kind::Clamp),
        kind(Kind::Upload),
        kind(Kind::Prelude),
        kind(Kind::Relay),
        kind(Kind::Present),
        kind(Kind::Hash),
        kind(Kind::Compare),
        kind(Kind::Fetch),
    ));
    out
}

/// Tests: this thread's sends give to its own book while this lives.
#[cfg(test)]
pub(crate) struct Sink;

#[cfg(test)]
impl Sink {
    pub(crate) fn new() -> Sink {
        SINK.with(|s| *s.borrow_mut() = Some(Book::default()));
        Sink
    }

    /// Every frame read so far whatever the frame under way is, and the
    /// lines for them (the card's `frequency`).
    pub(crate) fn lines(&self, frequency: u64) -> Vec<String> {
        let frames = with_book(|book| book.take(u64::MAX, 0).0);
        lines(&frames, 0, 0, 0, frequency, &name_text)
    }

    /// The passes read so far by name over the frames (their ticks), whether
    /// any frame had a flaw, our work by kind, and the passes begun whose
    /// call was not recorded (not taken).
    pub(crate) fn passes(&self) -> (BTreeMap<PassName, Vec<u64>>, bool, [u64; KINDS], u64) {
        with_book(|book| {
            let mut all: BTreeMap<PassName, Vec<u64>> = BTreeMap::new();
            let (mut ours, mut flawed, mut uncalled) = ([0; KINDS], false, 0);
            for f in book.frames.values() {
                for (n, t) in &f.passes {
                    all.entry(*n).or_default().extend_from_slice(t);
                }
                for (a, b) in ours.iter_mut().zip(f.ours) {
                    *a += b;
                }
                flawed |= f.flawed;
                uncalled += f.uncalled;
            }
            (all, flawed, ours, uncalled)
        })
    }
}

#[cfg(test)]
impl Drop for Sink {
    fn drop(&mut self) {
        SINK.with(|s| *s.borrow_mut() = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(n: u32) -> PassName {
        PassName { by: By::Unnamed, fragment: 0, nth: n }
    }

    /// A list's span is cut at its marks: the stretch before the first is
    /// in no pass, each mark's pass runs to the next or the close, and our
    /// pairs inside leave their ticks out of the stretch they lie in.
    #[test]
    fn a_list_is_cut_at_its_marks_and_our_pairs_are_left_out() {
        let marks = [(Some(10), p(1), 1), (Some(40), p(2), 3), (Some(70), p(3), 1)];
        let cut = parts((0, 100), marks, &[(20, 30), (65, 75)]);
        assert_eq!((cut.bad, cut.uncalled, cut.none), (0, 0, 10));
        assert_eq!(cut.passes, [(p(1), 30 - 10), (p(2), 30 - 5), (p(3), 30 - 5)]);
        let total: u64 = cut.passes.iter().map(|x| x.1).sum::<u64>() + cut.none + 10 + 10;
        assert_eq!(total, 100, "the passes, no pass and our pairs are the span");
    }

    /// A mark with no query heap, or out of place, leaves the stretch from
    /// the mark before it to the next usable one in no pass, and says so.
    #[test]
    fn a_lost_or_misplaced_mark_leaves_its_stretch_in_no_pass() {
        let marks = [(Some(10), p(1), 1), (None, p(2), 1), (Some(50), p(3), 1), (Some(40), p(4), 1), (Some(120), p(5), 1)];
        let cut = parts((0, 100), marks, &[]);
        assert_eq!(cut.bad, 3, "the lost one, one before the mark before it, one past the close");
        // Pass 1 to the lost mark, pass 3 to the misplaced one: no pass.
        assert!(cut.passes.is_empty(), "{cut:?}");
        assert_eq!(cut.none, 100);
        // A lost mark after a usable one: the pass before it loses its
        // stretch, the next usable mark's pass has its own.
        let cut = parts((0, 100), [(Some(10), p(1), 1), (None, p(2), 1), (Some(50), p(3), 1)], &[]);
        assert_eq!(cut, Cut { passes: vec![(p(3), 50)], none: 50, bad: 1, uncalled: 0 });
    }

    /// A pass whose call was not recorded is none: its stretch is in no
    /// pass, counted apart (Sol's review of the run diagnostics: a refused
    /// bind began and named a pass that never ran).
    #[test]
    fn a_pass_without_a_recorded_call_is_none() {
        let cut = parts((0, 100), [(Some(10), p(1), 2), (Some(40), p(2), 0), (Some(60), p(3), 1)], &[]);
        assert_eq!(cut, Cut { passes: vec![(p(1), 30), (p(3), 40)], none: 10 + 20, bad: 0, uncalled: 1 });
    }

    /// A frame is taken once the frame under way is past it and all its
    /// sends were read; one waiting is not, and the frames after it wait
    /// too, until it is given up; a send for a frame taken counts late.
    #[test]
    fn a_frame_is_taken_once_its_sends_are_read() {
        let mut book = Book::default();
        book.sent(1, 100);
        book.sent(1, 101);
        book.sent(2, 102);
        let reading = |ticks| Reading { passes: vec![(p(1), ticks)], ..Reading::default() };
        book.read(1, reading(5));
        book.read(2, reading(7));
        assert!(book.take(3, 0).0.is_empty(), "frame 1 waits for a send, frame 2 behind it");
        book.read(1, reading(6));
        assert!(book.take(1, 0).0.is_empty(), "frame 1 is still under way");
        let (frames, given_up) = book.take(3, 0);
        assert_eq!((frames.len(), given_up), (2, 0));
        assert_eq!(frames[0].passes[&p(1)], [5, 6], "two passes, two samples: not summed");
        book.sent(2, 200);
        assert_eq!(book.late, 1);
        book.sent(4, 300);
        let (frames, given_up) = book.take(5, 301);
        assert_eq!((frames.len(), given_up), (1, 1), "unread past the stale moment: given up");
        assert!(frames[0].flawed);
    }

    /// A send's frame is taken under the book's lock, as the lines take
    /// frames: one sent after a frame was taken goes in a later frame, never
    /// in the one taken (Sol's review: the number was read before the
    /// lock, and a send could land in a frame already printed).
    #[test]
    fn a_send_after_a_take_goes_in_a_later_frame() {
        let _sink = Sink::new();
        let first = sent(1);
        read(first, Reading::default());
        presented();
        let taken = with_book(|book| book.take(frame(), 0).0.len());
        assert_eq!(taken, 1);
        let second = sent(2);
        assert!(second > first, "{second} after {first}");
        assert_eq!(with_book(|book| book.late), 0);
    }

    /// The lines rank names by the median of their passes, each pass a
    /// sample, leave out a flawed frame, and give our share of the summed
    /// spans.
    #[test]
    fn the_lines_rank_by_median_and_give_our_share() {
        let frame = |a: &[u64], b: Option<u64>, ours: u64, flawed: bool| {
            let mut f = Frame { flawed, ..Frame::default() };
            f.passes.insert(p(1), a.to_vec());
            if let Some(b) = b {
                f.passes.insert(p(2), vec![b]);
            }
            f.ours[Kind::Hash as usize] = ours;
            f
        };
        // Pass 2 runs in two of three frames, at 4 and 6 ms; pass 1 twice in
        // the first frame (1 and 3 ms), then at 2 and 9.
        let frames = [
            frame(&[1_000, 3_000], Some(4_000), 5_000, false),
            frame(&[2_000], None, 5_000, false),
            frame(&[9_000], Some(6_000), 5_000, false),
            frame(&[50_000], Some(50_000), 0, true),
        ];
        let lines = lines(&frames, 0, 0, 0, 1_000_000, &|n| format!("{n:?}"));
        assert_eq!(lines.len(), 3, "{lines:#?}");
        assert!(lines[0].starts_with("card time by pass (VICTORY_GPU_PASSES=1): rank 1: 5.000 ms median a pass over 2 passes in 2 of 3 frames, 3.333 ms a frame over all 3, "), "{}", lines[0]);
        assert!(lines[1].starts_with("card time by pass (VICTORY_GPU_PASSES=1): rank 2: 2.500 ms median a pass over 4 passes in 3 of 3 frames, 5.000 ms a frame over all 3, "), "{}", lines[1]);
        // Title 8+2+15 = 25 ms, ours 15: 15/40.
        assert!(lines[2].contains("over 3 frames (1 frames left out"), "{}", lines[2]);
        assert!(lines[2].contains("title lists 8.00 ms median a frame, 2.0 passes a frame under 2 names"), "{}", lines[2]);
        assert!(lines[2].contains("our own work 5.00 ms median a frame"), "{}", lines[2]);
        assert!(lines[2].contains("ours 37.5% of the time stamped"), "{}", lines[2]);
        let none = super::lines(&frames[3..], 1, 2, 0, 1_000_000, &|n| format!("{n:?}"));
        assert_eq!(none.len(), 1);
        assert!(none[0].contains("no frames to count since the last line: 1 frames left out") && none[0].contains("2 sends that went for frames already counted"), "{}", none[0]);
    }

    /// A pass's name is fixed words, a hash and bounded shader ids: no
    /// label text (it may hold a path or the title's name), at most four
    /// shader ids (Sol's review: a label survived as text, and a stream of
    /// repeated shader parts printed whole).
    #[test]
    fn a_name_prints_no_label_and_at_most_four_shaders() {
        let label = label_hash("a fixture label".encode_utf16());
        let text = name_text(&PassName { by: By::Label(label), fragment: 1, nth: 2 });
        assert_eq!(
            text,
            format!(
                "named by the title's label of its list, hashed #{label:08x} (the label is never printed), fragment 1 of its \
                 list's recording, its pass 2 so named there"
            )
        );
        assert_eq!(bounded("[vertex a, pixel b]"), "[vertex a, pixel b]");
        assert_eq!(bounded("[compute a, compute b, compute c, compute d, compute e, compute f]"), "[compute a, compute b, compute c, compute d, and 2 more]");
        assert!(bounded(&format!("[compute {}]", "x".repeat(500))).len() <= 202);
    }
}

/// The passes on the software adapter: real lists, real marks, read after
/// their fence.
#[cfg(test)]
mod tests_on_the_card {
    use super::{By, PassName, Sink};
    use crate::host::{HostList, test_host};
    use crate::pipeline::host::HostPipeline;
    use crate::stamps::Kind;
    use crate::stamps::program_tests_on_the_card::{compute, dispatch};
    use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

    /// The stamps at level 1 with the passes on, and this thread's sinks.
    fn sinks() -> (crate::stamps::Sink, Sink) {
        let stamps = crate::stamps::Sink::new();
        crate::stamps::passes_for_test(true);
        (stamps, Sink::new())
    }

    /// A title call as the draw state records one: its pass's mark (`key`),
    /// the call, and the call noted.
    fn call(list: &HostList, p: &HostPipeline, key: u64, buffer: &ID3D12Resource, groups: u32) {
        list.pass_mark(p, key, None, 0);
        dispatch(list, p, buffer, groups);
        list.pass_called();
    }

    fn named(p: &HostPipeline, nth: u32) -> PassName {
        PassName { by: By::Program(p.program()), fragment: 0, nth }
    }

    /// A heavy compute pass, a pass after a binding of targets with a light
    /// dispatch in it, and the heavy one again: three passes told apart, the
    /// heavy name's two by their place, each a sample of its own with ticks
    /// of its own, the heavy two together no shorter than the light one, and
    /// the passes, the stretch in no pass and our upload in it make the
    /// list's span exactly; the passes fit inside the span, and their sum
    /// reaches each one of them (what the card counts between them in no
    /// pass differs by adapter, so no nearer bound holds on every card).
    #[test]
    fn a_list_s_passes_are_told_apart_and_add_up() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (stamps, book) = sinks();
        let (heavy, light) = (compute(20_000, "[compute pass heavy test]"), compute(1, "[compute pass light test]"));
        let second = compute(1, "[compute pass second light test]");
        let buffer = host.uav_buffer(4096).unwrap();
        let (big, other) = (host.uav_buffer(8 << 20).unwrap(), host.uav_buffer(8 << 20).unwrap());
        let list = host.new_list().unwrap();
        call(&list, &heavy, 0, &buffer, 64);
        call(&list, &light, 1, &buffer, 1);
        // An upload of ours in that pass: its time is ours, not the pass's.
        let begun = list.upload_stamp_begin();
        assert!(matches!(begun, Some(Some(_))), "an upload's pair with the passes on");
        // SAFETY: an open list; two live buffers of 8 MiB.
        unsafe { list.list.CopyBufferRegion(&other, 0, &big, 0, 8 << 20) };
        list.upload_stamp_end(begun);
        // The same binding again: no new pass. A second one-step program, not the heavy one: the heavy program
        // is a dependent chain, so one group of it takes as long as sixty-four on a card that runs them all at
        // once, and the "light" pass would then cost as much as a heavy one and the ranking below be a coin toss.
        call(&list, &second, 1, &buffer, 1);
        call(&list, &heavy, 0, &buffer, 64);
        list.close();
        host.execute(&[&list]);
        let span = stamps.spans().iter().find(|s| s.0 == Kind::List).map(|s| s.2 - s.1).expect("the list's pair");
        let upload: u64 = stamps.spans().iter().filter(|s| s.0 == Kind::Upload).map(|s| s.2 - s.1).sum();
        assert!(upload > 0, "the upload's pair: {:?}", stamps.spans());
        let (passes, flawed, ours, uncalled) = book.passes();
        assert!(!flawed && uncalled == 0, "{passes:?}");
        assert_eq!(ours[Kind::Upload as usize], upload);
        assert_eq!(passes.len(), 3, "{passes:?}");
        // Three distinct passes, in the order they ran, each a sample of its
        // own: the heavy name's two told apart by place, the light one's by
        // its program.
        let names = [named(&heavy, 1), named(&light, 1), named(&heavy, 2)];
        assert!(names.iter().all(|n| passes.get(n).is_some_and(|ticks| ticks.len() == 1)), "{passes:?}");
        let (h1, l, h2) = (passes[&names[0]][0], passes[&names[1]][0], passes[&names[2]][0]);
        assert!(h1 > 0 && l > 0 && h2 > 0, "each pass has ticks of its own: {h1}, {l}, {h2}");
        // Not "the heavy ones over four times the light one": each pass has
        // a fixed cost of its own, so that ratio does not hold on every
        // card; the heavy two together are still no shorter than the light
        // one.
        assert!(h1 + h2 >= l, "heavy {h1} and {h2}, light {l}");
        let none: u64 = super::with_book(|book| book.frames.values().map(|f| f.unattributed).sum());
        assert_eq!(h1 + h2 + l + none + upload, span, "the passes, no pass and our upload against the list's span");
        // The passes need not fill the list's span: what lies between them —
        // the driver's command processing, the timestamp queries and their
        // resolve — is in no pass (`none` above), and how much that is
        // depends on the adapter (on the laptop's Intel it is about a third
        // to a half of the span; on NVIDIA about a tenth). What holds on
        // every card is the decomposition above, and: the passes fit inside
        // the span (a hundredth's tolerance for rounding between the pairs),
        // the heavy two together are no shorter than the light one, and the
        // sum is at least any one of its terms — the floor asserted here is
        // the light pass's own time, which a sum of non-negative parts must
        // reach whatever the card counts between them.
        let sum = h1 + h2 + l;
        assert!(sum <= span + span / 100, "the passes {sum}, the list's span {span}");
        assert!(sum >= l && sum >= h1 && sum >= h2, "the passes {sum}: heavy {h1}, light {l}, heavy {h2}");
        let lines = book.lines(test_host().unwrap().clock_calibration().unwrap().frequency);
        assert!(lines[0].contains("rank 1: ") && lines[0].contains("shaders=[compute pass heavy test]"), "{lines:#?}");
        assert!(lines.last().unwrap().contains("over 1 frames (0 frames left out"), "{lines:#?}");
    }

    /// A mark that finds no query heap leaves its frame out, and the line
    /// says so.
    #[test]
    fn a_lost_mark_leaves_its_frame_out() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (_stamps, book) = sinks();
        let (a, b) = (compute(1, "[compute lost mark a test]"), compute(1, "[compute lost mark b test]"));
        let buffer = host.uav_buffer(4096).unwrap();
        let list = host.new_list().unwrap();
        for _ in 1..255 {
            list.stamp_begin();
        }
        call(&list, &a, 0, &buffer, 1);
        crate::stamps::no_chunks(true);
        list.pass_mark(&b, 1, None, 0);
        crate::stamps::no_chunks(false);
        dispatch(&list, &b, &buffer, 1);
        list.pass_called();
        list.close();
        host.execute(&[&list]);
        assert!(book.passes().1, "a frame with a lost mark counted");
        let lines = book.lines(1_000_000);
        assert_eq!(lines.len(), 1, "{lines:#?}");
        assert!(lines[0].contains("no frames to count since the last line: 1 frames left out"), "{}", lines[0]);
    }

    /// An upload whose first stamp finds no query heap is a lost pair: its
    /// frame is left out, not admitted with the upload in the pass around
    /// it (Sol's review: the lost beginning was not counted).
    #[test]
    fn an_upload_without_its_first_stamp_leaves_its_frame_out() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (_stamps, book) = sinks();
        let p = compute(1, "[compute lost upload test]");
        let buffer = host.uav_buffer(4096).unwrap();
        let list = host.new_list().unwrap();
        // The last slot of the first query heap, so the end has a heap.
        for _ in 1..255 {
            list.stamp_begin();
        }
        call(&list, &p, 0, &buffer, 1);
        crate::stamps::no_chunks(true);
        let begun = list.upload_stamp_begin();
        crate::stamps::no_chunks(false);
        assert_eq!(begun, Some(None), "stamping, with no slot");
        list.upload_stamp_end(begun);
        list.close();
        host.execute(&[&list]);
        assert!(book.passes().1, "a frame with a lost upload pair counted");
    }

    /// A pass whose call was not recorded is none (a draw dropped after its
    /// bind began a pass): its stretch is in no pass, said in the line.
    #[test]
    fn a_pass_whose_call_was_dropped_is_none() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (_stamps, book) = sinks();
        let (a, b) = (compute(1, "[compute kept test]"), compute(1, "[compute dropped test]"));
        let buffer = host.uav_buffer(4096).unwrap();
        let list = host.new_list().unwrap();
        call(&list, &a, 0, &buffer, 1);
        list.pass_mark(&b, 1, None, 0);
        call(&list, &a, 0, &buffer, 1);
        list.close();
        host.execute(&[&list]);
        let (passes, flawed, _, uncalled) = book.passes();
        assert!(!flawed);
        assert_eq!(uncalled, 1);
        assert_eq!(passes.keys().copied().collect::<Vec<_>>(), [named(&a, 1), named(&a, 2)]);
        let lines = book.lines(1_000_000);
        assert!(lines.last().unwrap().contains("1.0 passes a frame begun whose call was not recorded"), "{lines:#?}");
    }

    /// With the passes off a mark and an upload stamp nothing at level 1,
    /// and nothing reaches the book; a label's hash names the passes, with
    /// their fragment.
    #[test]
    fn off_nothing_is_marked_and_a_label_names_the_passes() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let p = compute(1, "[compute pass off test]");
        let buffer = host.uav_buffer(4096).unwrap();
        {
            let _stamps = crate::stamps::Sink::new();
            let book = Sink::new();
            let list = host.new_list().unwrap();
            let opened = list.stamp_slots();
            list.pass_mark(&p, 0, None, 0);
            assert!(list.upload_stamp_begin().is_none());
            assert_eq!(list.stamp_slots(), opened, "a mark at level 1 without the passes");
            dispatch(&list, &p, &buffer, 1);
            list.pass_called();
            list.close();
            host.execute(&[&list]);
            assert!(book.passes().0.is_empty());
        }
        let (_stamps, book) = sinks();
        let list = host.new_list().unwrap();
        list.pass_mark(&p, 0, Some(0xabcd), 3);
        dispatch(&list, &p, &buffer, 1);
        list.pass_called();
        list.close();
        host.execute(&[&list]);
        let (passes, flawed, _, _) = book.passes();
        assert!(!flawed);
        assert_eq!(passes.keys().copied().collect::<Vec<_>>(), [PassName { by: By::Label(0xabcd), fragment: 3, nth: 1 }]);
    }
}

/// Diagnostics: what the passes cost on this adapter. Run by name with
/// `--ignored --nocapture`.
#[cfg(test)]
mod cost_on_the_card {
    use super::Sink;
    use crate::host::test_host;
    use crate::stamps::Kind;
    use crate::stamps::program_tests_on_the_card::{compute, dispatch};

    /// 2,000 trivial dispatches in one list at level 1, with the passes off
    /// and on: a pass every 20 calls (100 passes, a frame's order), and a
    /// pass at every call (the worst case). Recording, sent to read, and
    /// the list's span on the queue, median of 5.
    #[test]
    #[ignore = "a measurement: run it by name with --ignored --nocapture"]
    fn passes_cost() {
        const CALLS: u32 = 2_000;
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let p = compute(1, "[compute pass cost test]");
        let buffer = host.uav_buffer(4096).unwrap();
        let frequency = host.clock_calibration().unwrap().frequency as f64;
        let run = |every: Option<u32>| -> [f64; 3] {
            let stamps = crate::stamps::Sink::new();
            crate::stamps::passes_for_test(every.is_some());
            let _book = Sink::new();
            let started = std::time::Instant::now();
            let list = host.new_list().unwrap();
            for i in 0..CALLS {
                if let Some(every) = every {
                    list.pass_mark(&p, u64::from(i / every) + 1, None, 0);
                }
                dispatch(&list, &p, &buffer, 1);
                list.pass_called();
            }
            list.close();
            let recorded = started.elapsed().as_secs_f64() * 1e3;
            let sent = std::time::Instant::now();
            host.execute(&[&list]);
            let ran = sent.elapsed().as_secs_f64() * 1e3;
            let span = stamps.spans().iter().find(|s| s.0 == Kind::List).map_or(0, |s| s.2 - s.1) as f64 * 1e3 / frequency;
            [recorded, ran, span]
        };
        let cases = [(None, "passes off"), (Some(20), "a pass every 20 calls"), (Some(1), "a pass at every call")];
        for (every, _) in cases {
            run(every);
        }
        for (every, what) in cases {
            let runs: Vec<[f64; 3]> = (0..5).map(|_| run(every)).collect();
            let median = |i: usize| {
                let mut v: Vec<f64> = runs.iter().map(|r| r[i]).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            eprintln!(
                "{what}, {CALLS} trivial dispatches in one list (median of 5): recording {:.2} ms, sent to read {:.2} ms, the \
                 list's span on the queue {:.2} ms",
                median(0),
                median(1),
                median(2)
            );
        }
    }
}
