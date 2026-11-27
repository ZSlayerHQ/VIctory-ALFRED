//! The binding trace (`VICTORY_BIND_TRACE=<pixel shader ids>`, 16 hex
//! digits each, separated by commas; off by default; diagnostics builds):
//! for the first draws of each of those pixel shaders, one line per
//! texture the draw's shaders read, with what the title's heap and the
//! host's hold for it: the title slot and whether the host heap mirrors
//! it, the texture's memory, formats, size, mips and tile mode, its host
//! copy, and when its contents were last read from title memory.
//!
//! A draw counts once per set of bindings: draws whose textures are the
//! same views in the same slots as a draw already traced are passed over
//! (a character's body comes in many meshlet batches). Each shader gets
//! `VICTORY_BIND_TRACE_DRAWS` traced draws (3; at most 64), from frame
//! `VICTORY_BIND_TRACE_FROM` on (0).
//!
//! The textures are those the draw names: table entries of its shaders'
//! used resources, entries picked by a root constant, and the slots that
//! words of its small constant and buffer tables name (a shader may pick
//! its textures by an index read from memory; such a word may be a small
//! number that names a slot by chance, so these lines say "named by a
//! word").
//!
//! Reads only: nothing it looks at is made, uploaded, bound or counted as
//! used. The lines wait for the next present and are logged there, not
//! while a list is recorded.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Mutex, OnceLock};

/// Ids at most.
const MOST_IDS: usize = 16;
/// Traced draws per id, by default and at most.
const DRAWS: u32 = 3;
const MOST_DRAWS: u32 = 64;
/// Lines waiting for a present at most; more are counted, not kept.
const MOST_PENDING: usize = 4096;

/// Which draws are traced.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Filter {
    ids: Vec<String>,
    draws: u32,
    from: u64,
}

impl Filter {
    /// From the switch's value (`ids`), the draws per id and the first
    /// frame. Ids that are not 16 hex digits are left out.
    pub(crate) fn parse(ids: &str, draws: Option<&str>, from: Option<&str>) -> Filter {
        let mut out: Vec<String> = Vec::new();
        for id in ids.split(',').map(str::trim).filter(|id| !id.is_empty()) {
            let id = id.to_ascii_lowercase();
            if id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit()) && !out.contains(&id) {
                out.push(id);
            }
        }
        out.truncate(MOST_IDS);
        let draws = draws
            .and_then(|d| d.trim().parse::<u32>().ok())
            .map_or(DRAWS, |d| d.clamp(1, MOST_DRAWS));
        let from = from.and_then(|f| f.trim().parse::<u64>().ok()).unwrap_or(0);
        Filter {
            ids: out,
            draws,
            from,
        }
    }

    /// Where `pixel` (a pixel shader's hash) is in the list.
    fn position(&self, pixel: &str) -> Option<usize> {
        self.ids.iter().position(|id| id == pixel)
    }
}

fn filter() -> &'static Filter {
    static FILTER: OnceLock<Filter> = OnceLock::new();
    FILTER.get_or_init(|| {
        let ids = std::env::var("VICTORY_BIND_TRACE").unwrap_or_default();
        let draws = std::env::var("VICTORY_BIND_TRACE_DRAWS").ok();
        let from = std::env::var("VICTORY_BIND_TRACE_FROM").ok();
        let f = Filter::parse(&ids, draws.as_deref(), from.as_deref());
        if !ids.trim().is_empty() {
            tracing::info!(ids = ?f.ids, draws = f.draws, from = f.from, "bind trace: on for these pixel shaders");
        }
        f
    })
}

/// Whether the trace is on (some id given).
pub(crate) fn on() -> bool {
    !filter().ids.is_empty()
}

/// Per id: the draws traced, and the bindings seen (a key per draw).
#[derive(Debug, Default)]
struct Seen {
    traced: Vec<u32>,
    keys: Vec<Vec<u64>>,
}

impl Seen {
    /// Whether a draw of `pixel` at `frame` may be traced at all.
    fn wants(&self, f: &Filter, pixel: &str, frame: u64) -> bool {
        frame >= f.from
            && f.position(pixel)
                .is_some_and(|i| self.traced.get(i).copied().unwrap_or(0) < f.draws)
    }

    /// The number (from 1) of a draw of `pixel` at `frame` with bindings
    /// `key` if it is traced: one of the first `f.draws` with bindings not
    /// seen before.
    fn take(&mut self, f: &Filter, pixel: &str, frame: u64, key: u64) -> Option<u32> {
        if !self.wants(f, pixel, frame) {
            return None;
        }
        let i = f.position(pixel)?;
        if self.traced.len() < f.ids.len() {
            self.traced.resize(f.ids.len(), 0);
            self.keys.resize(f.ids.len(), Vec::new());
        }
        if self.keys[i].contains(&key) {
            return None;
        }
        self.keys[i].push(key);
        self.traced[i] += 1;
        Some(self.traced[i])
    }
}

static SEEN: Mutex<Seen> = Mutex::new(Seen {
    traced: Vec::new(),
    keys: Vec::new(),
});

/// Whether a draw of pixel shader `pixel` may be traced now (cheap: the
/// draw's bindings are looked at only then).
pub(crate) fn wants(pixel: &str) -> bool {
    on() && SEEN.lock().unwrap_or_else(|e| e.into_inner()).wants(
        filter(),
        pixel,
        crate::frame::current(),
    )
}

/// The draw's number if a draw of `pixel` with bindings `key` is traced.
pub(crate) fn take(pixel: &str, key: u64) -> Option<u32> {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).take(
        filter(),
        pixel,
        crate::frame::current(),
        key,
    )
}

/// What the trace saw of a title texture's contents being read from title
/// memory (its staging buffer detiled, then copied into its host copy):
/// the last time, its bytes, whether they were all zero, whether the
/// detiling failed, how many times; and the last check at a submission
/// that samples it (a check that finds nothing new reads nothing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadNote {
    pub(crate) detiled: Option<u64>,
    pub(crate) bytes: u64,
    pub(crate) zero: bool,
    pub(crate) failed: bool,
    pub(crate) detiles: u32,
    pub(crate) checked: Option<u64>,
    pub(crate) checks: u32,
}

impl UploadNote {
    fn text(note: Option<&UploadNote>) -> String {
        let Some(n) = note else {
            return "never read from title memory since the trace began".into();
        };
        let mut text = match n.detiled {
            Some(frame) => format!(
                "detiled at frame {frame} ({} times, {} bytes{}{})",
                n.detiles,
                n.bytes,
                if n.zero { ", all zero" } else { "" },
                if n.failed { ", detiling failed" } else { "" }
            ),
            None => "never detiled".into(),
        };
        if let Some(frame) = n.checked {
            let _ = write!(text, "; checked at frame {frame} ({} checks)", n.checks);
        }
        text
    }
}

static UPLOADS: Mutex<Option<HashMap<usize, UploadNote>>> = Mutex::new(None);

/// Title texture `resource`'s staging buffer was detiled from `bytes` of
/// title memory (`zero`: all of them zero; `failed`: the detiling stopped).
pub(crate) fn note_detile(resource: usize, bytes: u64, zero: bool, failed: bool) {
    let frame = crate::frame::current();
    let mut map = UPLOADS.lock().unwrap_or_else(|e| e.into_inner());
    let n = map
        .get_or_insert_with(HashMap::new)
        .entry(resource)
        .or_default();
    n.detiled = Some(frame);
    n.bytes = bytes;
    n.zero = zero;
    n.failed = failed;
    n.detiles += 1;
}

/// Title texture `resource`'s memory was checked at a submission.
pub(crate) fn note_check(resource: usize) {
    let frame = crate::frame::current();
    let mut map = UPLOADS.lock().unwrap_or_else(|e| e.into_inner());
    let n = map
        .get_or_insert_with(HashMap::new)
        .entry(resource)
        .or_default();
    n.checked = Some(frame);
    n.checks += 1;
}

/// The title released `resource`: a later texture at its address is
/// another.
pub(crate) fn forget(resource: usize) {
    if let Some(map) = UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        map.remove(&resource);
    }
}

/// What the trace saw of title texture `resource`'s contents.
pub(crate) fn upload(resource: usize) -> Option<UploadNote> {
    UPLOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()?
        .get(&resource)
        .copied()
}

/// A title texture as the trace saw it (`resource` 0: a placed view for
/// which no title texture was made yet).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Texture {
    pub(crate) resource: usize,
    pub(crate) address: u64,
    /// The resource description's words: dimension, width, height, depth
    /// or array size, mips, samples, format, layout field, flags.
    pub(crate) dimension: u32,
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) mips: u32,
    pub(crate) samples: u32,
    pub(crate) format: u32,
    pub(crate) layout: u32,
    pub(crate) flags: u32,
    /// The swizzle mode its memory is detiled with (`None`: no layout).
    pub(crate) mode: Option<u32>,
    /// The view's format, first mip and mip count.
    pub(crate) view_format: u32,
    pub(crate) first_mip: u32,
    pub(crate) view_mips: u32,
    /// Its host copy in a few words.
    pub(crate) host: String,
    pub(crate) title_fills: bool,
    pub(crate) host_wrote: bool,
    /// The texture the host writes in the same memory whose texels it
    /// shows (`crate::resource::alias_of`).
    pub(crate) alias: Option<usize>,
    /// A texture the host renders to or writes whose memory overlaps this
    /// one's, in a few words.
    pub(crate) under: Option<String>,
    /// Its staging buffer: uploaded by a list, detiled since.
    pub(crate) staged: Option<(bool, bool)>,
    pub(crate) upload: Option<UploadNote>,
    /// Its title memory: mapped bytes and fingerprints.
    pub(crate) memory: String,
}

/// One texture a traced draw reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bound {
    pub(crate) stage: &'static str,
    /// Where the shader finds it.
    pub(crate) source: String,
    /// Its title slot, and how many title slots the host heap mirrors.
    pub(crate) slot: Option<u32>,
    pub(crate) mirrored: u32,
    /// What the shader reads in the host heap for it.
    pub(crate) host_slot: String,
    /// The view in a few words.
    pub(crate) view: String,
    pub(crate) texture: Option<Texture>,
    /// The queue of textures picked by index, of its slot.
    pub(crate) queue: Option<String>,
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

impl Bound {
    /// Its line, for draw `n` of pixel shader `pixel`.
    pub(crate) fn text(&self, pixel: &str, n: u32) -> String {
        let mut t = format!("{pixel} draw {n} {} {}", self.stage, self.source);
        match self.slot {
            Some(slot) => {
                let _ = write!(
                    t,
                    " | title slot {slot} {}",
                    if slot < self.mirrored {
                        format!("(mirrored, of {})", self.mirrored)
                    } else {
                        format!("(PAST the {} the host heap mirrors)", self.mirrored)
                    }
                );
            }
            None => t.push_str(" | no title slot"),
        }
        let _ = write!(t, " | host slot: {} | view: {}", self.host_slot, self.view);
        if let Some(x) = &self.texture {
            if x.resource == 0 {
                let _ = write!(
                    t,
                    " | texture: none made for this memory yet at {:#x}",
                    x.address
                );
            } else {
                let _ = write!(t, " | texture {:#x} at {:#x}", x.resource, x.address);
            }
            let _ = write!(
                t,
                " dim {} {}x{}x{} mips {} samples {} f{} layout {:#x} mode {} flags {:#x}",
                x.dimension,
                x.width,
                x.height,
                x.depth,
                x.mips,
                x.samples,
                x.format,
                x.layout,
                x.mode.map_or_else(|| "none".to_string(), |m| m.to_string()),
                x.flags
            );
            let _ = write!(
                t,
                " | view f{} mips {}+{}",
                x.view_format, x.first_mip, x.view_mips
            );
            let _ = write!(
                t,
                " | host copy: {} | title fills: {} host wrote: {} alias of: {} under a host target: {}",
                x.host,
                yes(x.title_fills),
                yes(x.host_wrote),
                x.alias
                    .map_or_else(|| "none".to_string(), |a| format!("{a:#x}")),
                x.under.as_deref().unwrap_or("none")
            );
            let staged = match x.staged {
                Some((uploaded, dirty)) => format!(
                    "uploaded by a list {}, detiled since {}",
                    yes(uploaded),
                    yes(dirty)
                ),
                None => "none".into(),
            };
            let _ = write!(
                t,
                " | staging: {staged} | contents: {} | {}",
                UploadNote::text(x.upload.as_ref()),
                x.memory
            );
        }
        if let Some(q) = &self.queue {
            let _ = write!(t, " | queue: {q}");
        }
        t
    }
}

/// A line waiting for the present, with the title texture whose contents
/// it tells of (0: none): at the present that texture's contents are told
/// again, as they are by then (the draw's own upload runs at submission).
struct Pending {
    text: String,
    texture: usize,
}

#[derive(Default)]
struct Lines {
    pending: Vec<Pending>,
    dropped: u64,
}

impl Lines {
    fn add(&mut self, lines: Vec<(String, usize)>) {
        for (text, texture) in lines {
            if self.pending.len() < MOST_PENDING {
                self.pending.push(Pending { text, texture });
            } else {
                self.dropped += 1;
            }
        }
    }
}

static LINES: Mutex<Lines> = Mutex::new(Lines {
    pending: Vec::new(),
    dropped: 0,
});

/// Keeps a traced draw's lines for the next present.
pub(crate) fn queue(lines: Vec<(String, usize)>) {
    LINES.lock().unwrap_or_else(|e| e.into_inner()).add(lines);
}

/// At the present of `frame`: logs the lines kept since the last.
pub(crate) fn flush(frame: u64) {
    if !on() {
        return;
    }
    let (pending, dropped) = {
        let mut lines = LINES.lock().unwrap_or_else(|e| e.into_inner());
        (
            std::mem::take(&mut lines.pending),
            std::mem::take(&mut lines.dropped),
        )
    };
    for p in pending {
        let mut text = p.text;
        if p.texture != 0 {
            let _ = write!(
                text,
                " | contents by the present: {}",
                UploadNote::text(upload(p.texture).as_ref())
            );
        }
        tracing::info!(frame, "bind trace: {text}");
    }
    if dropped > 0 {
        tracing::info!(frame, dropped, "bind trace: lines past the limit left out");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_sixteen_hex_digits_and_the_counts_are_bounded() {
        let f = Filter::parse(
            " 0123456789ABCDEF, fedcba9876543210,bad,,0123,0123456789abcdef",
            None,
            None,
        );
        assert_eq!(f.ids, ["0123456789abcdef", "fedcba9876543210"]);
        assert_eq!((f.draws, f.from), (DRAWS, 0));
        assert_eq!(Filter::parse("", Some("5"), Some("3600")).draws, 5);
        assert_eq!(Filter::parse("", Some("5"), Some("3600")).from, 3600);
        assert_eq!(Filter::parse("", Some("1000"), None).draws, MOST_DRAWS);
        assert_eq!(
            Filter::parse("", Some("0"), Some("x")),
            Filter {
                ids: vec![],
                draws: 1,
                from: 0
            }
        );
        let many: Vec<String> = (0..40).map(|i| format!("{i:016x}")).collect();
        assert_eq!(
            Filter::parse(&many.join(","), None, None).ids.len(),
            MOST_IDS
        );
        assert_eq!(f.position("fedcba9876543210"), Some(1));
        assert_eq!(f.position("fedcba987654321"), None, "a prefix is not an id");
    }

    #[test]
    fn each_id_gets_its_draws_with_bindings_not_seen_before() {
        let f = Filter::parse("aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb", Some("2"), Some("10"));
        let mut s = Seen::default();
        assert_eq!(
            s.take(&f, "aaaaaaaaaaaaaaaa", 9, 1),
            None,
            "before the first frame"
        );
        assert_eq!(s.take(&f, "cccccccccccccccc", 10, 1), None, "not listed");
        assert_eq!(s.take(&f, "aaaaaaaaaaaaaaaa", 10, 1), Some(1));
        assert_eq!(
            s.take(&f, "aaaaaaaaaaaaaaaa", 10, 1),
            None,
            "the same bindings again"
        );
        assert_eq!(
            s.take(&f, "bbbbbbbbbbbbbbbb", 11, 1),
            Some(1),
            "each id on its own"
        );
        assert!(s.wants(&f, "aaaaaaaaaaaaaaaa", 12));
        assert_eq!(s.take(&f, "aaaaaaaaaaaaaaaa", 12, 2), Some(2));
        assert!(!s.wants(&f, "aaaaaaaaaaaaaaaa", 12), "its draws are done");
        assert_eq!(s.take(&f, "aaaaaaaaaaaaaaaa", 13, 3), None);
        assert!(!Seen::default().wants(&Filter::parse("", None, None), "aaaaaaaaaaaaaaaa", 0));
    }

    #[test]
    fn a_texture_line_says_where_it_is_and_what_its_contents_are() {
        let b = Bound {
            stage: "pixel",
            source: "word of s1+56 at +0x1c names it".into(),
            slot: Some(93022),
            mirrored: 737_855,
            host_slot: "the title slot's, made from this view".into(),
            view: "placed texture".into(),
            texture: Some(Texture {
                resource: 0x1000,
                address: 0x7_0000_0000,
                dimension: 3,
                width: 1024,
                height: 512,
                depth: 1,
                mips: 11,
                samples: 1,
                format: 71,
                layout: 0x105,
                mode: Some(5),
                view_format: 72,
                first_mip: 2,
                view_mips: 9,
                host: "made 1024x512x1 mips 11 f71 flags 0x0".into(),
                title_fills: true,
                alias: Some(0x2000),
                staged: Some((true, false)),
                upload: Some(UploadNote {
                    detiled: Some(3500),
                    bytes: 4096,
                    zero: true,
                    detiles: 1,
                    checked: Some(3600),
                    checks: 7,
                    ..Default::default()
                }),
                memory: "memory=0x700000000 mapped=4096/4096".into(),
                ..Default::default()
            }),
            queue: Some("not queued".into()),
        };
        let text = b.text("0123456789abcdef", 2);
        for part in [
            "0123456789abcdef draw 2 pixel word of s1+56 at +0x1c names it",
            "title slot 93022 (mirrored, of 737855)",
            "host slot: the title slot's, made from this view | view: placed texture",
            "texture 0x1000 at 0x700000000 dim 3 1024x512x1 mips 11 samples 1 f71 layout 0x105 mode 5 flags 0x0",
            "view f72 mips 2+9",
            "host copy: made 1024x512x1 mips 11 f71 flags 0x0 | title fills: yes host wrote: no alias of: 0x2000 under a host target: none",
            "staging: uploaded by a list yes, detiled since no",
            "contents: detiled at frame 3500 (1 times, 4096 bytes, all zero); checked at frame 3600 (7 checks)",
            "memory=0x700000000 mapped=4096/4096 | queue: not queued",
        ] {
            assert!(text.contains(part), "{part:?} not in {text:?}");
        }
        let past = Bound {
            slot: Some(800_000),
            mirrored: 737_855,
            ..Default::default()
        }
        .text("x", 1);
        assert!(
            past.contains("title slot 800000 (PAST the 737855 the host heap mirrors)"),
            "{past}"
        );
        let none = Bound {
            texture: Some(Texture {
                address: 0x10,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            none.text("x", 1)
                .contains("texture: none made for this memory yet at 0x10")
        );
        assert!(
            none.text("x", 1)
                .contains("contents: never read from title memory since the trace began")
        );
    }

    #[test]
    fn lines_past_the_limit_are_counted_not_kept() {
        let mut l = Lines::default();
        l.add((0..MOST_PENDING + 5).map(|i| (i.to_string(), 0)).collect());
        assert_eq!((l.pending.len(), l.dropped), (MOST_PENDING, 5));
    }
}
