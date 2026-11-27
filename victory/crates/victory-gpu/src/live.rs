//! Live switches: `VICTORY_LIVE=<file>` names a control file read at frame
//! boundaries so a run's diagnostics can change without a restart.
//!
//! The file holds one `NAME=value` per line, `#` starts a comment; names
//! the registry does not know are said once and ignored, a line without
//! `=` makes the whole file count as not written yet (writers write a
//! temporary file and rename it), and a value of the wrong shape leaves
//! that switch's old value. The file is looked at at most twice a second,
//! and only when its modified time has moved since the last read.
//!
//! A change applies only at a frame boundary (the present that counts a
//! frame). Each one logs `live: <NAME> <old> -> <new> at frame <n>` and the
//! segment it ends logs `live segment <k>: frames <a>..<b>, mean <ms>, p50
//! <ms>, p95 <ms>, skipped <d> draws and <s> dispatches, in force:
//! <NAME=value ...>` from the frame times the runtime measures at the same
//! boundary (the line's whole shape is [`segment_line`]'s: it also says the frames left
//! out of the mean as transition, as light and as long, and what began and ended it).
//! `LIVE_MARK=<n>` changes no switch: a new number only ends the segment under way, so
//! a plan's start begins a segment of its own. The
//! last segment is logged at the run's end by any path the runtime sees; the segment
//! under way is logged as `live segment <k> open: ...` with each frame line and is in
//! the stats file, for a run that is killed from outside.
//!
//! Switches whose value is captured into a long-lived object, or that
//! change the memory model's guards, are deliberately not here: see the
//! README's `VICTORY_SKIP_RT_FEED` and `VICTORY_VIEW_ASK`.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use crate::scripted_keys::{ScriptedKey, live_keys_script};

/// One slot per switch's scalar value (flags and numbers); the two lists have their own.
const SCALARS: usize = COUNT;
/// How often the control file may be looked at.
const CHECK: Duration = Duration::from_millis(500);

/// What a switch's value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// On or off (`on`/`off`, or the old spellings `1`/`0`, `true`/`false`,
    /// `yes`/`no`; an empty value is off).
    OnOff,
    /// A whole number: a texture watermark in percent (1 to 100).
    Integer,
    /// Comma-separated ids of pipelines to skip.
    List,
    /// A timed key sequence (`LIVE_KEYS`), read whole by the shared parser.
    Keys,
    /// A mark (`LIVE_MARK`): a whole number, zero or more, that changes nothing in the run. A
    /// new number ends the segment under way, so a plan's start begins a segment of its own.
    Mark,
}

/// A live switch: the name the control file and the log use, and its kind.
pub(crate) struct Spec {
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
}

/// The registry's switches in ONE table, alphabetical by name (a test says
/// so): one entry each, `CONST => "NAME", Kind, start value`. The slot of
/// each switch is its position in this list, worked out when the macro
/// expands, so a switch added anywhere renumbers nothing by hand; the
/// start value is its environment default (the two skip lists exist only
/// here and start empty).
macro_rules! switches {
    ($( $ident:ident => $name:literal, $kind:ident, $start:expr; )*) => {
        switches!(@count (0usize); $($ident),*);
        switches!(@slots (0usize); $($ident),*);
        /// The switches, in slot order.
        pub(crate) const SPECS: [Spec; COUNT] = [ $(Spec { name: $name, kind: Kind::$kind }),* ];
        /// The start value of the switch at `index`, from `get` (the
        /// environment in a run): today's defaults and spellings.
        fn start_at(index: usize, get: &dyn Fn(&str) -> Option<OsString>) -> Value {
            match index {
                $( $ident => ($start)(get), )*
                _ => Value::Flag(false),
            }
        }
    };
    (@count ($n:expr); ) => {
        const COUNT: usize = $n;
    };
    (@count ($n:expr); $ident:ident $(, $($rest:ident),*)?) => {
        switches!(@count ($n + 1usize); $($($rest),*)?);
    };
    (@slots ($i:expr); ) => {};
    (@slots ($i:expr); $ident:ident $(, $($rest:ident),*)?) => {
        pub(crate) const $ident: usize = $i;
        switches!(@slots ($i + 1usize); $($($rest),*)?);
    };
}

switches! {
    LIVE_KEYS => "LIVE_KEYS", Keys, |_get: &dyn Fn(&str) -> Option<OsString>| Value::Keys(Keys::default());
    MARK => "LIVE_MARK", Mark, |_get: &dyn Fn(&str) -> Option<OsString>| Value::Number(0);
    SKIP_DISPATCH => "LIVE_SKIP_DISPATCH", List, |_get: &dyn Fn(&str) -> Option<OsString>| Value::List(Vec::new());
    SKIP_DRAW => "LIVE_SKIP_DRAW", List, |_get: &dyn Fn(&str) -> Option<OsString>| Value::List(Vec::new());
    EXACT_BACK => "VICTORY_EXACT_BACK", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_EXACT_BACK").as_deref() == Some(OsStr::new("1")));
    FETCH_QUEUE => "VICTORY_FETCH_QUEUE", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_FETCH_QUEUE").as_deref().is_some_and(|v| flag_on(&v.to_string_lossy())));
    KEEP_TEXTURES => "VICTORY_KEEP_TEXTURES", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_KEEP_TEXTURES").as_deref().is_none_or(|v| v != OsStr::new("0")));
    LAND_SCAN => "VICTORY_LAND_SCAN", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_LAND_SCAN").as_deref() == Some(OsStr::new("1")));
    LATE_WAITS => "VICTORY_LATE_WAITS", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_LATE_WAITS").as_deref() == Some(OsStr::new("1")));
    LIVE_HUD => "VICTORY_LIVE_HUD", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_LIVE_HUD").as_deref().is_some_and(|v| flag_on(&v.to_string_lossy())));
    TEXTURE_HIGH => "VICTORY_TEXTURE_HIGH", Integer,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Number(percent(get("VICTORY_TEXTURE_HIGH").as_deref(), 85));
    TEXTURE_LOW => "VICTORY_TEXTURE_LOW", Integer,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Number(percent(get("VICTORY_TEXTURE_LOW").as_deref(), 70));
    TEXTURE_MADE => "VICTORY_TEXTURE_MADE", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_TEXTURE_MADE").as_deref().is_some_and(|v| flag_on(&v.to_string_lossy())));
    TEXTURE_RECHECK => "VICTORY_TEXTURE_RECHECK", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_TEXTURE_RECHECK").as_deref().is_some_and(|v| flag_on(&v.to_string_lossy())));
    UPLOAD_POOL => "VICTORY_UPLOAD_POOL", OnOff,
        |get: &dyn Fn(&str) -> Option<OsString>| Value::Flag(get("VICTORY_UPLOAD_POOL").as_deref().is_some_and(|v| flag_on(&v.to_string_lossy())));
}

/// A live key sequence: the text as the log prints it and its presses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Keys {
    text: String,
    presses: Vec<ScriptedKey>,
}

/// A switch's value. Flags and numbers live in atomics, lists in mutexes,
/// and key sequences in one of their own with their playback state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Value {
    Flag(bool),
    Number(i64),
    List(Vec<String>),
    Keys(Keys),
}

impl Value {
    /// The value as the control file writes it and the log prints it:
    /// `on`/`off`, a decimal number, a comma-joined list (empty if so), or
    /// the key sequence's text.
    pub(crate) fn text(&self) -> String {
        match self {
            Value::Flag(true) => "on".to_owned(),
            Value::Flag(false) => "off".to_owned(),
            Value::Number(number) => number.to_string(),
            Value::List(list) => list.join(","),
            Value::Keys(keys) => keys.text.clone(),
        }
    }
}

/// Whether `text` says on: anything but `0`, `false`, `off` or `no` (any
/// case), and an empty value off; surrounding space is ignored.
fn flag_on(text: &str) -> bool {
    let text = text.trim();
    !(text.is_empty()
        || text.eq_ignore_ascii_case("0")
        || text.eq_ignore_ascii_case("false")
        || text.eq_ignore_ascii_case("off")
        || text.eq_ignore_ascii_case("no"))
}

/// A percent value: a trimmed whole number clamped to 1 to 100, or
/// `default` where the text is not one.
fn percent(value: Option<&OsStr>, default: i64) -> i64 {
    value
        .and_then(OsStr::to_str)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(default, |value| value.clamp(1, 100) as i64)
}

/// [`std::env::var_os`] as the start values' lookup.
fn env_os(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// The start values from `get` (the environment in a run), in slot order.
fn start_values(get: &dyn Fn(&str) -> Option<OsString>) -> Vec<Value> {
    (0..COUNT).map(|index| start_at(index, get)).collect()
}

/// A value of `spec`'s kind from the text after `=`, or `None` if the text
/// is not one (the switch keeps its old value).
fn parse_value(spec: &Spec, text: &str) -> Option<Value> {
    match spec.kind {
        Kind::OnOff => Some(Value::Flag(flag_on(text))),
        Kind::Integer => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|number| Value::Number(number.clamp(1, 100))),
        Kind::Mark => text.trim().parse::<i64>().ok().filter(|number| *number >= 0).map(Value::Number),
        Kind::List => Some(Value::List(
            text.split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect(),
        )),
        Kind::Keys => {
            let text = text.trim();
            if text.is_empty() {
                // Empty cancels what has not played: the switch's own line
                // in `store` drops the playback state.
                return Some(Value::Keys(Keys::default()));
            }
            match live_keys_script(text) {
                Ok(presses) => Some(Value::Keys(Keys {
                    text: text.to_owned(),
                    presses,
                })),
                Err(reason) => {
                    // One warning, and the old value (and its playback)
                    // stays; `parse` does not add the generic bad-value
                    // warning for this kind.
                    tracing::warn!(
                        name = "LIVE_KEYS",
                        value = text,
                        reason = reason.reason(),
                        "live: LIVE_KEYS is refused; it keeps its old value"
                    );
                    None
                }
            }
        }
    }
}

/// Says an unknown switch name once, then ignores it.
fn note_unknown(name: &str) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|seen| seen == name) {
        seen.push(name.to_owned());
        tracing::warn!(name, "live: the control file names a switch there is none of; ignored");
    }
}

/// Says a value of the wrong shape once per reading (the file is read only
/// when its modified time moves, so this cannot repeat every frame).
fn note_bad(name: &str, text: &str) {
    tracing::warn!(
        name,
        value = text.trim(),
        "live: the control file gives this switch a value of the wrong shape; it keeps its old value"
    );
}

/// Says a control file that is not whole once, and waits for its writer's
/// rename.
fn note_unparsed() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!("live: the control file is not whole yet (a line without `=`); ignored until it parses");
    }
}

/// The assignments `text` asks for, or `Err` while it is not a whole file
/// (a non-comment line without `=`). Unknown names and bad values are said
/// and left out, so they leave the old values alone.
fn parse(text: &str) -> Result<Vec<(usize, Value)>, ()> {
    let mut given = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err(());
        };
        let name = name.trim();
        if name.is_empty() {
            return Err(());
        }
        let Some(index) = SPECS.iter().position(|spec| spec.name == name) else {
            note_unknown(name);
            continue;
        };
        match parse_value(&SPECS[index], value) {
            Some(value) => given.push((index, value)),
            // A refused key sequence has already logged its one warning.
            None if SPECS[index].kind == Kind::Keys => {}
            None => note_bad(name, value),
        }
    }
    Ok(given)
}

/// Applies `given` to `values`, with each change's old value. A value that
/// is already in force is no change: it is not applied, logged or made to
/// close a segment.
fn apply(values: &mut [Value], given: Vec<(usize, Value)>) -> Vec<(usize, Value, Value)> {
    let mut changes = Vec::new();
    for (index, new) in given {
        let old = values[index].clone();
        if old != new {
            values[index] = new.clone();
            changes.push((index, old, new));
        }
    }
    changes
}

/// The start values in one place, then the current ones, so a reading of
/// the file can be applied to a copy and compared before anything changes.
static INIT: OnceLock<()> = OnceLock::new();
static SCALAR: [AtomicI64; SCALARS] = [const { AtomicI64::new(0) }; SCALARS];
static SKIP_DISPATCH_LIST: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SKIP_DRAW_LIST: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SKIP_DISPATCH_LEN: AtomicUsize = AtomicUsize::new(0);
static SKIP_DRAW_LEN: AtomicUsize = AtomicUsize::new(0);
/// The `LIVE_KEYS` value in force (the last one parsed, empty or not).
static KEYS_VALUE: Mutex<Option<Keys>> = Mutex::new(None);
/// The sequence playing now, if one is: its start is the frame boundary
/// that applied it, so its times count from there.
static KEYS_ACTIVE: Mutex<Option<ActiveKeys>> = Mutex::new(None);
/// The number the next run of a live sequence gets.
static KEYS_NEXT: AtomicU64 = AtomicU64::new(1);

/// A live sequence playing: the id the input side names when it is done,
/// the frame boundary that applied it, and its presses.
struct ActiveKeys {
    id: u64,
    start: Instant,
    presses: Arc<[ScriptedKey]>,
    count: usize,
}

/// A live sequence as the input side reads it: the run it belongs to, when
/// its clock started, and its presses.
#[derive(Clone)]
pub struct KeysRun {
    pub id: u64,
    pub start: Instant,
    pub presses: Arc<[ScriptedKey]>,
    pub count: usize,
}

/// The live sequence playing now, if one is. The input side evaluates it
/// from [`KeysRun::start`] and calls [`keys_done`] when every press has
/// ended.
pub fn keys() -> Option<KeysRun> {
    KEYS_ACTIVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|active| KeysRun {
            id: active.id,
            start: active.start,
            presses: Arc::clone(&active.presses),
            count: active.count,
        })
}

/// Logs a finished live sequence once and stops it. `id` is the run the
/// input side evaluated, so a sequence replaced meanwhile is left alone.
pub fn keys_done(id: u64) {
    // The frame the input poll saw the end at: the last boundary applied.
    let frame = path()
        .map(|path| {
            state(path.clone())
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .segment
                .last
        })
        .unwrap_or(0);
    let mut active = KEYS_ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(run) = active.as_ref().filter(|run| run.id == id) else {
        return;
    };
    tracing::info!("{}", done_line(frame, run.count));
    *active = None;
}

/// The line a finished live sequence logs.
fn done_line(frame: u64, presses: usize) -> String {
    format!("live keys: sequence done at frame {frame}, {presses} presses")
}

/// Draws then dispatches skipped by the live switches, over the run.
static SKIPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Segments closed so far ([`segment`]).
static SEGMENT: AtomicU64 = AtomicU64::new(0);

/// Tests: whether the registry's start values have been read (the off path must not make it).
#[cfg(test)]
pub(crate) fn started() -> bool {
    INIT.get().is_some()
}

fn init() {
    INIT.get_or_init(|| {
        for (index, value) in start_values(&env_os).into_iter().enumerate() {
            store(index, value);
        }
    });
}

/// Writes one value into the registry. Only the control file (at a frame
/// boundary) and the start write.
fn store(index: usize, value: Value) {
    match value {
        Value::Flag(on) => SCALAR[index].store(i64::from(on), Ordering::Relaxed),
        Value::Number(number) => SCALAR[index].store(number, Ordering::Relaxed),
        Value::List(list) => {
            let (behind, len) = match index {
                SKIP_DISPATCH => (&SKIP_DISPATCH_LIST, &SKIP_DISPATCH_LEN),
                SKIP_DRAW => (&SKIP_DRAW_LIST, &SKIP_DRAW_LEN),
                _ => return,
            };
            let length = list.len();
            *behind.lock().unwrap_or_else(|e| e.into_inner()) = list;
            len.store(length, Ordering::SeqCst);
        }
        Value::Keys(keys) => {
            debug_assert_eq!(index, LIVE_KEYS);
            // A non-empty value starts its run here, at the frame boundary
            // that applied it; empty has no run and cancels the last one.
            let active = (!keys.presses.is_empty()).then(|| ActiveKeys {
                id: KEYS_NEXT.fetch_add(1, Ordering::Relaxed),
                start: Instant::now(),
                count: keys.presses.len(),
                presses: Arc::from(keys.presses.clone()),
            });
            *KEYS_VALUE.lock().unwrap_or_else(|e| e.into_inner()) = Some(keys);
            *KEYS_ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = active;
        }
    }
}

/// The registry's current values, for a file reading to be applied to. With `wait` false it
/// gives `None` rather than wait for a list's lock (a recording thread scanning a long skip
/// list), which is what the presenting thread's snapshot does.
fn snapshot_with(wait: bool) -> Option<Vec<Value>> {
    init();
    let flag = |index: usize| Value::Flag(SCALAR[index].load(Ordering::Relaxed) != 0);
    let list = |behind: &Mutex<Vec<String>>| {
        let held = if wait {
            behind.lock().unwrap_or_else(|e| e.into_inner())
        } else {
            behind.try_lock().ok()?
        };
        Some(Value::List(held.clone()))
    };
    let keys = if wait {
        KEYS_VALUE.lock().unwrap_or_else(|e| e.into_inner()).clone()
    } else {
        KEYS_VALUE.try_lock().ok()?.clone()
    };
    let mut values = Vec::with_capacity(COUNT);
    for (index, spec) in SPECS.iter().enumerate() {
        values.push(match spec.kind {
            Kind::OnOff => flag(index),
            Kind::Integer | Kind::Mark => Value::Number(SCALAR[index].load(Ordering::Relaxed)),
            Kind::List => match index {
                SKIP_DISPATCH => list(&SKIP_DISPATCH_LIST)?,
                SKIP_DRAW => list(&SKIP_DRAW_LIST)?,
                _ => return None,
            },
            Kind::Keys => Value::Keys(keys.clone().unwrap_or_default()),
        });
    }
    Some(values)
}

/// [`snapshot_with`], taking every lock.
fn snapshot() -> Vec<Value> {
    snapshot_with(true).expect("a waiting lock succeeds")
}

/// Whether a run has `VICTORY_LIVE` set: off, nothing below reads a file,
/// records a frame time or logs a segment.
pub(crate) fn on() -> bool {
    path().is_some()
}

/// The control file, from `VICTORY_LIVE` (an empty value is off); said
/// once, when first asked.
fn path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = std::env::var_os("VICTORY_LIVE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        if let Some(path) = &path {
            tracing::info!(
                file = %path.display(),
                "live switches: the control file is read at frame boundaries, at most twice a second"
            );
        }
        path
    })
    .as_ref()
}

/// Whether the texture retention policy is on (`VICTORY_KEEP_TEXTURES`).
pub(crate) fn keep_textures() -> bool {
    init();
    SCALAR[KEEP_TEXTURES].load(Ordering::Relaxed) != 0
}

/// The idle sweep's low watermark in percent (`VICTORY_TEXTURE_LOW`).
pub(crate) fn texture_low() -> u64 {
    init();
    SCALAR[TEXTURE_LOW].load(Ordering::Relaxed) as u64
}

/// The idle sweep's high watermark in percent (`VICTORY_TEXTURE_HIGH`).
pub(crate) fn texture_high() -> u64 {
    init();
    SCALAR[TEXTURE_HIGH].load(Ordering::Relaxed) as u64
}

/// Whether late frame waits sleep to the next tick, as before
/// (`VICTORY_LATE_WAITS`).
pub(crate) fn late_waits() -> bool {
    init();
    SCALAR[LATE_WAITS].load(Ordering::Relaxed) != 0
}

/// Whether `id` is in `list`: the whole pipeline id as the log names it
/// (`[compute 0123...]`) or one of its shader ids.
fn listed(behind: &Mutex<Vec<String>>, len: &AtomicUsize, id: &str) -> bool {
    if len.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let list = behind.lock().unwrap_or_else(|e| e.into_inner());
    list.iter().any(|token| token_matches(token, id))
}

/// Whether `token` names `id`: the id whole (the brackets the log puts
/// round it are optional), or one of the ids inside it.
fn token_matches(token: &str, id: &str) -> bool {
    let trim = |text: &str| text.trim().trim_matches(['[', ']']).trim().to_owned();
    let token = trim(token);
    if token.is_empty() {
        return false;
    }
    if trim(id).eq_ignore_ascii_case(&token) {
        return true;
    }
    id.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| !part.is_empty() && part.eq_ignore_ascii_case(&token))
}

/// Whether the skip list for dispatches (`dispatch`) or for draws has anything in it:
/// a load, no lock.
pub(crate) fn skipping(dispatch: bool) -> bool {
    init();
    let len = if dispatch { &SKIP_DISPATCH_LEN } else { &SKIP_DRAW_LEN };
    len.load(Ordering::Relaxed) != 0
}

/// Whether a compute dispatch of pipeline `id` is skipped
/// (`LIVE_SKIP_DISPATCH`).
pub(crate) fn skip_dispatch(id: &str) -> bool {
    init();
    listed(&SKIP_DISPATCH_LIST, &SKIP_DISPATCH_LEN, id)
}

/// Whether a landing looks the copies in the PC's memory through while the card fetch is in
/// flight (live; starts from `VICTORY_LAND_SCAN`, off).
pub(crate) fn land_scan() -> bool {
    init();
    SCALAR[LAND_SCAN].load(Ordering::Relaxed) != 0
}

/// Whether a merge compares 32 bytes at a time and stores only the words that differ (live;
/// starts from `VICTORY_EXACT_BACK`, off).
pub(crate) fn exact_back() -> bool {
    init();
    SCALAR[EXACT_BACK].load(Ordering::Relaxed) != 0
}

/// Whether upload fills run on a pool (live; starts from `VICTORY_UPLOAD_POOL`, off).
pub(crate) fn upload_pool() -> bool {
    init();
    SCALAR[UPLOAD_POOL].load(Ordering::Relaxed) != 0
}

/// Whether the live panel is drawn on the presented frame (live; starts from `VICTORY_LIVE_HUD`,
/// off). One relaxed atomic load, without starting the registry: with the panel off, the hook
/// and the stats sample must not allocate, lock or format anything. The registry's start values
/// are read by the other switches at startup (a frame wait or a retention sweep among them), so
/// an environment value reaches this flag before the first present; a control-file change
/// writes it at its frame boundary. Once the flag is on, the panel's first paint initialises
/// what it needs ([`pairs_try`]).
pub(crate) fn live_hud() -> bool {
    SCALAR[LIVE_HUD].load(Ordering::Relaxed) != 0
}

/// Whether the title textures `texture_at` makes are counted (live; starts
/// from `VICTORY_TEXTURE_MADE`, off; read by the diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn texture_made() -> bool {
    init();
    SCALAR[TEXTURE_MADE].load(Ordering::Relaxed) != 0
}

/// Whether an upload's source is looked at again after 2, 8 and 30 frames
/// (live; starts from `VICTORY_TEXTURE_RECHECK`, off).
pub(crate) fn texture_recheck() -> bool {
    init();
    SCALAR[TEXTURE_RECHECK].load(Ordering::Relaxed) != 0
}

/// The registry's default values, name and text, for the live panel's "not at its default" lines:
/// the start values with no environment variable set.
pub(crate) fn defaults() -> Vec<(&'static str, String)> {
    SPECS
        .iter()
        .zip(start_values(&|_| None))
        .map(|(spec, value)| (spec.name, value.text()))
        .collect()
}

/// The last change the control file applied: the switch, the values either side of it, and the
/// moment of the frame boundary that applied it.
#[derive(Clone)]
pub(crate) struct Change {
    pub(crate) name: &'static str,
    pub(crate) old: String,
    pub(crate) new: String,
    pub(crate) at: Instant,
}

/// The last change applied, for the live panel: `None` before the control file changes anything.
static LAST_CHANGE: Mutex<Option<Change>> = Mutex::new(None);

/// The last change the control file applied, if any.
pub(crate) fn last_change() -> Option<Change> {
    LAST_CHANGE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Tests: the change history's lock, held from another thread to show that the paths taken
/// while the panel is off do not take it.
#[cfg(test)]
pub(crate) fn hold_change_lock() -> std::sync::MutexGuard<'static, Option<Change>> {
    LAST_CHANGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tests: serialises the ones that hold a registry list's lock against the ones that read the
/// panel's text, so neither makes the other's snapshot give up.
#[cfg(test)]
pub(crate) static REGISTRY_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Keeps the last change for the live panel. Called only when a reading changed a value, so the
/// frame boundary's work is unchanged while the file says nothing new.
fn remember_change(name: &'static str, old: &Value, new: &Value, at: Instant) {
    *LAST_CHANGE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Change {
        name,
        old: old.text(),
        new: new.text(),
        at,
    });
}

/// Whether the copy-back's fetch goes to a copy queue of its own (live; starts from
/// `VICTORY_FETCH_QUEUE`, off). A plan takes the mode only when no plan is out, and keeps it
/// to its landing's end (`host/fetch.rs`).
pub(crate) fn fetch_queue() -> bool {
    init();
    SCALAR[FETCH_QUEUE].load(Ordering::Relaxed) != 0
}

/// The segments closed so far: what the live stats file calls the segment.
pub(crate) fn segment() -> u64 {
    SEGMENT.load(Ordering::Relaxed)
}

/// Whether switch `index` is one the stats and segment lines list now: the panel's own switch
/// is left out while the panel is off, so a run with it off writes the same lines as one
/// without it.
fn listed_in_lines(index: usize) -> bool {
    index != LIVE_HUD || SCALAR[LIVE_HUD].load(Ordering::Relaxed) != 0
}

/// The values as the stats line's `live` map and the segment's `in force` list, name and
/// text, the panel's switch left out while it is off.
fn pairs_of(values: Vec<Value>) -> Vec<(&'static str, String)> {
    SPECS
        .iter()
        .zip(values)
        .enumerate()
        .filter(|(index, _)| listed_in_lines(*index))
        .map(|(_, (spec, value))| (spec.name, value.text()))
        .collect()
}

/// Every live switch's current value, name and text, for the live stats file.
pub(crate) fn pairs() -> Vec<(&'static str, String)> {
    pairs_of(snapshot())
}

/// As [`pairs`], for the presenting thread: `None` instead of waiting for a lock a recording
/// thread holds while scanning a long skip list. The panel then keeps the text it shows and
/// tries again at the next refresh.
pub(crate) fn pairs_try() -> Option<Vec<(&'static str, String)>> {
    snapshot_with(false).map(pairs_of)
}

/// Whether a draw of pipeline `id` is skipped (`LIVE_SKIP_DRAW`).
pub(crate) fn skip_draw(id: &str) -> bool {
    init();
    listed(&SKIP_DRAW_LIST, &SKIP_DRAW_LEN, id)
}

/// Counts one skipped call (its `dispatch`-ness chooses the counter), for
/// the next segment line.
pub(crate) fn skipped(dispatch: bool) {
    SKIPPED[usize::from(dispatch)].fetch_add(1, Ordering::Relaxed);
}

fn skip_totals() -> [u64; 2] {
    [
        SKIPPED[0].load(Ordering::Relaxed),
        SKIPPED[1].load(Ordering::Relaxed),
    ]
}

/// Warns, once per skip switch, when its list first becomes non-empty: the
/// picture may break while one is set.
fn warn_first_use(index: usize, old: &Value, new: &Value) {
    static WARNED: [std::sync::atomic::AtomicBool; 2] =
        [const { std::sync::atomic::AtomicBool::new(false) }; 2];
    let (slot, name, what) = match index {
        SKIP_DISPATCH => (0, "LIVE_SKIP_DISPATCH", "compute dispatches"),
        SKIP_DRAW => (1, "LIVE_SKIP_DRAW", "draws"),
        _ => return,
    };
    let empty = |value: &Value| matches!(value, Value::List(list) if list.is_empty());
    if empty(old) && !empty(new) && !WARNED[slot].swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "{name} is set: {what} of the listed pipelines are skipped; the picture may break while a list is set"
        );
    }
}

/// One step of the control file: the modified time it was read at and its
/// text, or `None` when there is nothing new (or the file is missing or
/// not readable yet).
struct Watcher {
    path: PathBuf,
    /// The modified time last read and parsed; `None` until one is.
    modified: Option<SystemTime>,
    /// When the file was last looked at (at most twice a second).
    checked: Option<Instant>,
}

impl Watcher {
    fn new(path: PathBuf) -> Watcher {
        Watcher {
            path,
            modified: None,
            checked: None,
        }
    }

    /// Whether the file may be looked at now, and mark it looked at.
    fn due(&mut self, now: Instant) -> bool {
        if self
            .checked
            .is_some_and(|checked| now.saturating_duration_since(checked) < CHECK)
        {
            return false;
        }
        self.checked = Some(now);
        true
    }

    /// Takes the file when its modified time has moved: `(modified, text)`.
    /// A missing file, or one that cannot be read, is ignored until a later
    /// look.
    fn take(&mut self, now: Instant) -> Option<(SystemTime, String)> {
        if !self.due(now) {
            return None;
        }
        #[cfg(test)]
        POLLS.fetch_add(1, Ordering::Relaxed);
        let modified = std::fs::metadata(&self.path)
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        let path = self.path.clone();
        self.check(modified, || std::fs::read_to_string(&path).ok())
    }

    /// [`Watcher::take`]'s rule apart from the file system, for tests: a
    /// moved modified time reads; the same one does not.
    fn check(
        &mut self,
        modified: Option<SystemTime>,
        read: impl FnOnce() -> Option<String>,
    ) -> Option<(SystemTime, String)> {
        let modified = modified?;
        if self.modified == Some(modified) {
            return None;
        }
        let text = read()?;
        self.modified = Some(modified);
        Some((modified, text))
    }

    /// A reading that did not parse whole: its modified time is not taken,
    /// so the file is read again at the next look.
    fn forget(&mut self) {
        self.modified = None;
    }
}

/// The frames after a change that a segment leaves out of its mean and percentiles: the
/// present that applies a change is a boundary on the presenting thread only. Submissions
/// already sent ahead and lists the title recorded before it still run under the old value in
/// the frame after, and the first use of a path makes its objects in the frame after that (the
/// copy queue and its memory, a pipeline); the third is one to spare. They are counted and
/// their mean is said, never dropped.
pub(crate) const TRANSITION_FRAMES: u64 = 3;

/// A frame with fewer draws than this is not the scene: a load, a menu or a pause (the
/// harness's settled watch asks for as many). The runtime cannot tell which of the three, so
/// it guesses none: the frame is left out of the mean and percentiles as "light", counted,
/// with the light frames' own mean beside it.
pub(crate) const LIGHT_DRAWS: u64 = 1000;

/// A frame longer than this, in ms, is left out of the mean and percentiles as "long" (a
/// stall, a load with the scene still drawn), counted, with the long frames' own mean.
pub(crate) const LONG_MS: f64 = 1000.0;

/// What began a segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Begun {
    /// The run's start: nothing cut a segment before it, and it holds the loading.
    #[default]
    Start,
    /// A change of a switch: its first frames are transition.
    Change,
    /// A mark alone (`LIVE_MARK`): nothing changed, so no frame is transition.
    Mark,
}

/// A count of frames left out, and their times' sum in ms.
type Left = (u64, f64);

/// A segment's frames: what its line and the stats file's `segment_open` say. Each frame is
/// in exactly one of four: transition, light, long, or counted, asked in that order.
#[derive(Clone, Default)]
struct Segment {
    /// The first frame the segment holds, and the last.
    first: Option<u64>,
    last: u64,
    begun: Begun,
    /// The frame times it counts, in ms.
    times: Vec<f64>,
    /// The first [`TRANSITION_FRAMES`] after a change.
    transition: Left,
    /// Frames under [`LIGHT_DRAWS`] draws.
    light: Left,
    /// Frames over [`LONG_MS`].
    long: Left,
}

impl Segment {
    /// A present counted `frame`, `milliseconds` after the one before, with `draws` run on
    /// the host in it.
    fn frame(&mut self, frame: u64, milliseconds: f64, draws: u64) {
        self.last = frame;
        if self.first.is_none() {
            self.first = Some(frame);
        }
        let seen = self.times.len() as u64 + self.transition.0 + self.light.0 + self.long.0;
        let left = if self.begun == Begun::Change && seen < TRANSITION_FRAMES {
            &mut self.transition
        } else if draws < LIGHT_DRAWS {
            &mut self.light
        } else if milliseconds > LONG_MS {
            &mut self.long
        } else {
            self.times.push(milliseconds);
            return;
        };
        *left = (left.0 + 1, left.1 + milliseconds);
    }

    /// Whether it holds a frame.
    fn any(&self) -> bool {
        self.first.is_some() && (!self.times.is_empty() || self.transition.0 + self.light.0 + self.long.0 > 0)
    }

    /// Its figures, with the calls skipped in it.
    fn figures(&self, skipped: [u64; 2]) -> Figures {
        let (mean, p50, p95) = if self.times.is_empty() { (0.0, 0.0, 0.0) } else { stats(&self.times) };
        let mean_of = |left: Left| (left.0, if left.0 > 0 { left.1 / left.0 as f64 } else { 0.0 });
        Figures {
            counted: self.times.len() as u64,
            transition: mean_of(self.transition),
            light: mean_of(self.light),
            long: mean_of(self.long),
            mean,
            p50,
            p95,
            begun: self.begun,
            skipped_draws: skipped[0],
            skipped_dispatches: skipped[1],
        }
    }
}

/// A segment's figures: its frame times in ms, the frames left out of them by kind (a count
/// and their mean), and calls skipped in it.
#[derive(Clone, Copy, Default)]
struct Figures {
    counted: u64,
    transition: Left,
    light: Left,
    long: Left,
    mean: f64,
    p50: f64,
    p95: f64,
    begun: Begun,
    skipped_draws: u64,
    skipped_dispatches: u64,
}

/// What ended a segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ended {
    /// A change of a switch.
    Change,
    /// A mark alone (`LIVE_MARK`): a plan's start, which changes no switch.
    Mark,
    /// The run's end, by a path the runtime sees.
    RunEnd,
    /// Nothing yet: the segment under way, as of the line.
    NotYet,
}

/// Mean, p50 and p95 of `times` (ms), nearest rank.
fn stats(times: &[f64]) -> (f64, f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    (mean, nearest(&sorted, 50), nearest(&sorted, 95))
}

/// The `percent`th value by nearest rank (sorted, not empty).
fn nearest(sorted: &[f64], percent: usize) -> f64 {
    let rank = (sorted.len() * percent).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// The marker an applied change logs.
fn marker_line(name: &str, old: &Value, new: &Value, frame: u64) -> String {
    format!("live: {name} {} -> {} at frame {frame}", old.text(), new.text())
}

/// The line a closed segment logs; with [`Ended::NotYet`], the line the segment under way
/// logs with each frame line (`live segment <k> open: ...`), so that a run ended from outside
/// (a kill, which no code in the process sees) has its last stretch in the log up to then.
fn segment_line(k: u64, first: u64, last: u64, figures: Figures, ended: Ended, in_force: &[String]) -> String {
    format!(
        "live segment {k}{}: frames {first}..{last}, {} counted, {} excluded as transition (the first {TRANSITION_FRAMES} \
         after a change; their mean {:.1}), {} excluded as light (under {LIGHT_DRAWS} draws: a load, a menu or a pause, the \
         runtime cannot tell which; their mean {:.1}), {} excluded as long (over {LONG_MS:.0} ms; their mean {:.1}), mean \
         {:.1}, p50 {:.1}, p95 {:.1} of the counted, begun at {}, {}, skipped {} draws and {} dispatches, in force: {}",
        if ended == Ended::NotYet { " open" } else { "" },
        figures.counted,
        figures.transition.0,
        figures.transition.1,
        figures.light.0,
        figures.light.1,
        figures.long.0,
        figures.long.1,
        figures.mean,
        figures.p50,
        figures.p95,
        match figures.begun {
            Begun::Start => "the start of the run",
            Begun::Change => "a change",
            Begun::Mark => "a mark",
        },
        match ended {
            Ended::Change => "ended by a change",
            Ended::Mark => "ended by a mark",
            Ended::RunEnd => "ended by the end of the run",
            Ended::NotYet => "not ended (the segment under way, as of this line)",
        },
        figures.skipped_draws,
        figures.skipped_dispatches,
        in_force.join(" ")
    )
}

/// The stats file's `segment_open`: the segment under way, as of the object.
fn segment_json(k: u64, first: u64, last: u64, figures: Figures) -> String {
    format!(
        "{{\"segment\":{k},\"first_frame\":{first},\"last_frame\":{last},\"counted\":{},\"transition_excluded\":{},\
         \"transition_mean_ms\":{:.2},\"light_excluded\":{},\"light_mean_ms\":{:.2},\"long_excluded\":{},\
         \"long_mean_ms\":{:.2},\"mean_ms\":{:.2},\"p50_ms\":{:.2},\"p95_ms\":{:.2},\"begun\":\"{}\"}}",
        figures.counted,
        figures.transition.0,
        figures.transition.1,
        figures.light.0,
        figures.light.1,
        figures.long.0,
        figures.long.1,
        figures.mean,
        figures.p50,
        figures.p95,
        match figures.begun {
            Begun::Start => "start",
            Begun::Change => "change",
            Begun::Mark => "mark",
        },
    )
}

/// What the registry holds, as the segment line's `in force` list (the panel's switch left out
/// while it is off).
fn in_force() -> Vec<String> {
    SPECS
        .iter()
        .zip(snapshot())
        .enumerate()
        .filter(|(index, _)| listed_in_lines(*index))
        .map(|(_, (spec, value))| format!("{}={}", spec.name, value.text()))
        .collect()
}

/// The segment under way.
struct State {
    watcher: Watcher,
    segment: Segment,
    /// The skip counters when the segment began.
    start_skips: [u64; 2],
    /// Segments closed so far (the next one's number).
    closed: u64,
}

impl State {
    fn new(path: PathBuf) -> State {
        State { watcher: Watcher::new(path), segment: Segment::default(), start_skips: skip_totals(), closed: 0 }
    }
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state(path: PathBuf) -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::new(path)))
}

/// A presented frame: `frame` its number, `milliseconds` since the last
/// present, `draws` run on the host in it. Reads the control file when due
/// and applies what moved.
pub(crate) fn frame(frame: u64, milliseconds: f64, draws: u64) {
    let Some(path) = path() else {
        return;
    };
    let mut state = state(path.clone()).lock().unwrap_or_else(|e| e.into_inner());
    state.segment.frame(frame, milliseconds, draws);
    let now = Instant::now();
    poll(&mut state, now, frame);
}

/// Reads and applies the control file if due, at this frame boundary.
fn poll(state: &mut State, now: Instant, frame: u64) {
    let Some((_, text)) = state.watcher.take(now) else {
        return;
    };
    let Ok(given) = parse(&text) else {
        state.watcher.forget(); // not whole: read again at the next look
        note_unparsed();
        return;
    };
    let mut values = snapshot();
    let changes = apply(&mut values, given);
    if changes.is_empty() {
        return;
    }
    if let Some(line) = close(state, frame, ended_by(&changes)) {
        tracing::info!("{line}");
    }
    for (index, old, new) in changes {
        store(index, new.clone());
        tracing::info!("{}", marker_line(SPECS[index].name, &old, &new, frame));
        warn_first_use(index, &old, &new);
        // The panel's history: nothing locked or formatted for it while it is off. A change
        // that turns it on is kept (the store above comes first); one that turns it off is not.
        if live_hud() {
            remember_change(SPECS[index].name, &old, &new, now);
        }
    }
}

/// What ends the segment when `changes` are applied: a mark, if nothing but marks changed.
fn ended_by(changes: &[(usize, Value, Value)]) -> Ended {
    if changes.iter().all(|(index, ..)| SPECS[*index].kind == Kind::Mark) { Ended::Mark } else { Ended::Change }
}

/// Closes the segment under way and starts the next one at the following
/// frame: its line, if it held any frame.
fn close(state: &mut State, frame: u64, ended: Ended) -> Option<String> {
    let totals = skip_totals();
    let line = state.segment.any().then(|| {
        state.closed += 1;
        SEGMENT.store(state.closed, Ordering::Relaxed);
        let skipped = [totals[0] - state.start_skips[0], totals[1] - state.start_skips[1]];
        segment_line(
            state.closed,
            state.segment.first.unwrap_or(frame),
            state.segment.last,
            state.segment.figures(skipped),
            ended,
            &in_force(),
        )
    });
    // What ended this one began the next: its first frames are transition only after a change.
    let begun = match ended {
        Ended::Change => Begun::Change,
        Ended::Mark => Begun::Mark,
        Ended::RunEnd | Ended::NotYet => Begun::Start,
    };
    state.segment = Segment { last: frame, begun, ..Segment::default() };
    state.start_skips = totals;
    line
}

/// The segment under way as of now, not closed: its line ([`Ended::NotYet`]) and the stats
/// file's object. None while it holds no frame.
fn open(state: &State) -> Option<(String, String)> {
    let segment = &state.segment;
    let first = segment.first.filter(|_| segment.any())?;
    let totals = skip_totals();
    let figures = segment.figures([totals[0] - state.start_skips[0], totals[1] - state.start_skips[1]]);
    let k = state.closed + 1;
    Some((
        segment_line(k, first, segment.last, figures, Ended::NotYet, &in_force()),
        segment_json(k, first, segment.last, figures),
    ))
}

/// The segment under way, for the log, with each frame line: a run that is ended from
/// outside (the harness's timer kills the process, which nothing in it sees) never reaches
/// [`finish`], and its last stretch is in the newest of these lines and in the stats file.
pub(crate) fn open_line() -> Option<String> {
    path()?;
    let state = STATE.get()?.lock().unwrap_or_else(|e| e.into_inner());
    open(&state).map(|(line, _)| line)
}

/// The segment under way, for the stats file's `segment_open`; `null` with the switch unset
/// or before its first frame.
pub(crate) fn open_json() -> String {
    let open = || {
        path()?;
        let state = STATE.get()?.lock().unwrap_or_else(|e| e.into_inner());
        open(&state).map(|(_, json)| json)
    };
    open().unwrap_or_else(|| "null".to_owned())
}

/// The final segment, at the run's end by any path the runtime sees (the title's exit, its
/// `ExitProcess` or `TerminateProcess` of itself, the window's close, a host GPU failure):
/// the same line, for the last stretch. A second call finds no frame and says nothing.
pub(crate) fn finish() {
    let Some(path) = path() else {
        return;
    };
    let mut state = state(path.clone()).lock().unwrap_or_else(|e| e.into_inner());
    let frame = state.segment.last;
    if let Some(line) = close(&mut state, frame, Ended::RunEnd) {
        tracing::info!("{line}");
    }
}

#[cfg(test)]
static POLLS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<Value> {
        (0..COUNT).map(|index| start_at(index, &|_| None)).collect()
    }

    /// The one table is alphabetical, and every switch's start value has the
    /// shape its kind declares.
    #[test]
    fn the_table_is_alphabetical_and_every_start_matches_its_kind() {
        assert!(
            SPECS.windows(2).all(|two| two[0].name < two[1].name),
            "the registry's names are alphabetical: {:?}",
            SPECS.iter().map(|spec| spec.name).collect::<Vec<_>>()
        );
        let mut kinds = 0;
        for (index, spec) in SPECS.iter().enumerate() {
            match (spec.kind, start_at(index, &|_| None)) {
                (Kind::OnOff, Value::Flag(_)) | (Kind::Integer | Kind::Mark, Value::Number(_)) => kinds += 1,
                (Kind::List, Value::List(_)) | (Kind::Keys, Value::Keys(_)) => kinds += 1,
                (kind, value) => panic!("{}: a {:?} starts as {value:?}", spec.name, kind),
            }
        }
        assert_eq!(kinds, COUNT, "every switch has a start");
    }

    /// The README's live list names every switch of the table, in its order.
    #[test]
    fn the_readme_lists_every_switch_in_table_order() {
        let readme = include_str!("../../../README.md");
        let row = readme
            .lines()
            .find(|line| line.starts_with("| `VICTORY_LIVE=<file>` |"))
            .expect("the live row");
        let names: Vec<&str> = SPECS.iter().map(|spec| spec.name).collect();
        let mut listed: Vec<&str> = Vec::new();
        for (at, _) in row.match_indices('`') {
            let Some(text) = row[at + 1..].split('`').next() else {
                continue;
            };
            // The row names a switch again later (a note on one of them):
            // the list is the first time each appears.
            if names.contains(&text) && !listed.contains(&text) {
                listed.push(text);
            }
        }
        assert_eq!(listed, names, "the README's live list, in the table's order");
    }

    /// Comments (whole-line and trailing), unknown names and bad values:
    /// the last two are left out, so the old values stay.
    #[test]
    fn parsing_takes_comments_and_leaves_unknown_names_and_bad_values_alone() {
        let text = "\
# a comment
VICTORY_KEEP_TEXTURES = off  # trailing

VICTORY_TEXTURE_LOW=wat
NOT_A_SWITCH=1
LIVE_SKIP_DRAW=alpha, beta ,
";
        let given = parse(text).expect("it parses");
        assert_eq!(
            given,
            vec![
                (KEEP_TEXTURES, Value::Flag(false)),
                (SKIP_DRAW, Value::List(vec!["alpha".to_owned(), "beta".to_owned()])),
            ]
        );
        let mut values = fixture();
        let changes = apply(&mut values, given);
        assert_eq!(
            changes,
            vec![
                (KEEP_TEXTURES, Value::Flag(true), Value::Flag(false)),
                (
                    SKIP_DRAW,
                    Value::List(Vec::new()),
                    Value::List(vec!["alpha".to_owned(), "beta".to_owned()])
                ),
            ]
        );
        assert_eq!(values[TEXTURE_LOW], Value::Number(70), "a bad value keeps the old one");
        // A line without `=`: the file is not whole yet.
        assert!(parse("VICTORY_TEXTURE_LOW=60\nhalf a line").is_err());
    }

    /// A change is a change once: applying the same reading again neither
    /// changes nor returns anything.
    #[test]
    fn the_same_value_twice_is_one_change() {
        let mut values = fixture();
        let given = parse("VICTORY_TEXTURE_LOW=60").expect("it parses");
        assert_eq!(
            apply(&mut values, given),
            vec![(TEXTURE_LOW, Value::Number(70), Value::Number(60))]
        );
        let again = parse("VICTORY_TEXTURE_LOW=60").expect("it parses");
        assert!(apply(&mut values, again).is_empty(), "no second change");
        assert_eq!(values[TEXTURE_LOW], Value::Number(60));
    }

    /// A reading that flips one of the three copy-back / pool switches is a change like any
    /// other, applied at the frame boundary that read it.
    #[test]
    fn a_copy_back_switch_flip_is_a_change() {
        let mut values = fixture();
        let given = parse("VICTORY_UPLOAD_POOL=on
VICTORY_EXACT_BACK=1").expect("it parses");
        assert_eq!(
            apply(&mut values, given),
            vec![
                (UPLOAD_POOL, Value::Flag(false), Value::Flag(true)),
                (EXACT_BACK, Value::Flag(false), Value::Flag(true)),
            ]
        );
    }

    /// The start values are the environment's: the default spellings, the
    /// watermarks' defaults and the two empty lists.
    #[test]
    fn the_start_values_come_from_the_environment() {
        use std::collections::HashMap;
        let env: HashMap<&str, OsString> = [
            ("VICTORY_KEEP_TEXTURES", "0"),
            ("VICTORY_TEXTURE_LOW", "40"),
            ("VICTORY_TEXTURE_HIGH", "95"),
            ("VICTORY_LATE_WAITS", "1"),
            ("VICTORY_LAND_SCAN", "1"),
            ("VICTORY_EXACT_BACK", "0"),
            ("VICTORY_UPLOAD_POOL", "1"),
            ("VICTORY_LIVE_HUD", "1"),
            ("VICTORY_FETCH_QUEUE", "1"),
        ]
        .into_iter()
        .map(|(name, value)| (name, OsString::from(value)))
        .collect();
        let get = |name: &str| env.get(name).cloned();
        let values = start_values(&get);
        assert_eq!(values[KEEP_TEXTURES], Value::Flag(false));
        assert_eq!(values[TEXTURE_LOW], Value::Number(40));
        assert_eq!(values[TEXTURE_HIGH], Value::Number(95));
        assert_eq!(values[LATE_WAITS], Value::Flag(true));
        assert_eq!(values[SKIP_DISPATCH], Value::List(Vec::new()));
        assert_eq!(values[SKIP_DRAW], Value::List(Vec::new()));
        assert_eq!(values[LAND_SCAN], Value::Flag(true));
        assert_eq!(values[EXACT_BACK], Value::Flag(false));
        assert_eq!(values[UPLOAD_POOL], Value::Flag(true));
        assert_eq!(values[LIVE_HUD], Value::Flag(true));
        assert_eq!(values[FETCH_QUEUE], Value::Flag(true));
        let none = |_: &str| None;
        let defaults = start_values(&none);
        assert_eq!(defaults[KEEP_TEXTURES], Value::Flag(true), "on unless 0");
        assert_eq!(defaults[TEXTURE_LOW], Value::Number(70));
        assert_eq!(defaults[TEXTURE_HIGH], Value::Number(85));
        assert_eq!(defaults[LATE_WAITS], Value::Flag(false), "off unless 1");
        assert_eq!(defaults[LIVE_HUD], Value::Flag(false), "off unless 1");
        assert_eq!(defaults[FETCH_QUEUE], Value::Flag(false), "the copy queue's fetch is off by default");
        // Clamped as the old parse clamped them.
        let wide = |name: &str| (name == "VICTORY_TEXTURE_LOW").then(|| OsString::from("400"));
        assert_eq!(start_values(&wide)[TEXTURE_LOW], Value::Number(100));
    }

    /// The panel's defaults are the start values with no environment variable set, one for each
    /// switch, the panel's own included (off).
    #[test]
    fn the_defaults_are_the_values_with_no_environment() {
        let defaults = defaults();
        assert_eq!(defaults.len(), COUNT);
        assert_eq!(defaults[LIVE_HUD], ("VICTORY_LIVE_HUD", "off".to_owned()));
        assert_eq!(defaults[TEXTURE_LOW], ("VICTORY_TEXTURE_LOW", "70".to_owned()));
        assert_eq!(defaults[KEEP_TEXTURES], ("VICTORY_KEEP_TEXTURES", "on".to_owned()));
    }

    /// The panel's own switch is left out of the stats map and the segment's `in force` list
    /// while it is off, so the lines match a run without the panel.
    #[test]
    fn the_panel_is_not_listed_while_it_is_off() {
        if live_hud() {
            return; // a run with the panel on is not this test
        }
        let _alone = REGISTRY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        init();
        assert!(!pairs().iter().any(|(name, _)| *name == "VICTORY_LIVE_HUD"));
        assert!(
            !in_force()
                .iter()
                .any(|pair| pair.starts_with("VICTORY_LIVE_HUD="))
        );
    }

    /// The presenting thread's snapshot does not wait behind a recording thread scanning a
    /// skip list: it comes back empty, and the panel keeps the text it shows until the next
    /// refresh.
    #[test]
    fn the_presenter_does_not_wait_while_the_registry_lock_is_held() {
        let _alone = REGISTRY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        init();
        let (locked, held) = std::sync::mpsc::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            let _guard = SKIP_DRAW_LIST.lock().unwrap_or_else(|e| e.into_inner());
            locked.send(()).unwrap();
            let _ = released.recv_timeout(Duration::from_secs(10));
        });
        held.recv_timeout(Duration::from_secs(10))
            .expect("the recording thread holds the list");
        let started = Instant::now();
        assert!(pairs_try().is_none(), "the presenter's snapshot did not wait");
        assert!(started.elapsed() < Duration::from_secs(5));
        release.send(()).unwrap();
        holder.join().unwrap();
        assert!(pairs_try().is_some(), "the snapshot is taken once the list is free");
    }

    /// The last change is kept for the panel with the moment the boundary applied it.
    #[test]
    fn the_last_change_is_kept_with_the_moment_it_applied() {
        let now = Instant::now();
        remember_change("VICTORY_TEXTURE_LOW", &Value::Number(70), &Value::Number(60), now);
        let change = last_change().expect("a change was kept");
        assert_eq!(change.name, "VICTORY_TEXTURE_LOW");
        assert_eq!((change.old.as_str(), change.new.as_str()), ("70", "60"));
        assert_eq!(change.at, now);
    }

    /// The real registry, in a process of its own: the start values are
    /// read from the environment variables there.
    #[test]
    fn the_registry_starts_from_the_environment_variables() {
        let name = "live::tests::start_values_child";
        let mut command = std::process::Command::new(std::env::current_exe().expect("the test binary"));
        command
            .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
            .env_remove("VICTORY_LIVE")
            .env_remove("VICTORY_LIVE_HUD")
            .env_remove("VICTORY_FETCH_QUEUE")
            .env("VICTORY_KEEP_TEXTURES", "0")
            .env("VICTORY_TEXTURE_LOW", "40")
            .env("VICTORY_TEXTURE_HIGH", "95")
            .env("VICTORY_LATE_WAITS", "1");
        let out = command.output().expect("the child ran");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success() && text.contains("1 passed"), "{text}");
    }

    /// The stats sample with the panel off is one relaxed load: it does not start the registry
    /// (which would allocate and read the environment). A child process, so no earlier test
    /// has started it.
    #[test]
    fn the_off_sample_does_not_start_the_registry() {
        let name = "live::tests::off_sample_child";
        let mut command = std::process::Command::new(std::env::current_exe().expect("the test binary"));
        command
            .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
            .env_remove("VICTORY_LIVE")
            .env_remove("VICTORY_LIVE_HUD");
        let out = command.output().expect("the child ran");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success() && text.contains("1 passed"), "{text}");
    }

    #[test]
    #[ignore = "a child process of the test above"]
    fn off_sample_child() {
        assert!(!started(), "the registry has not started");
        crate::host::note_sample(16.0, 60.0);
        assert!(!started(), "the sample with the panel off started the registry");
        assert!(!live_hud(), "off unless 1");
    }

    #[test]
    #[ignore = "a child process of the test above"]
    fn start_values_child() {
        assert!(!on(), "VICTORY_LIVE is unset");
        assert!(!keep_textures());
        assert_eq!(texture_low(), 40);
        assert_eq!(texture_high(), 95);
        assert!(late_waits());
        assert!(!skip_dispatch("[compute 0123456789abcdef]"));
        assert!(!skip_draw("anything"));
        assert!(!live_hud(), "off unless 1");
        assert!(!fetch_queue(), "off unless 1");
    }

    /// With `VICTORY_LIVE` unset the file is not touched, and the registry
    /// keeps its start values.
    #[test]
    fn without_victory_live_no_file_is_touched() {
        if std::env::var_os("VICTORY_LIVE").is_some() {
            return; // a run with the switch is not this test
        }
        assert!(!on());
        let before = POLLS.load(Ordering::Relaxed);
        frame(1, 16.0, 0);
        assert_eq!(POLLS.load(Ordering::Relaxed), before, "no file looked at");
        let keep = std::env::var_os("VICTORY_KEEP_TEXTURES");
        assert_eq!(keep_textures(), keep.as_deref().is_none_or(|v| v != "0"));
        let low = std::env::var_os("VICTORY_TEXTURE_LOW");
        assert_eq!(texture_low(), percent(low.as_deref(), 70) as u64);
        let high = std::env::var_os("VICTORY_TEXTURE_HIGH");
        assert_eq!(texture_high(), percent(high.as_deref(), 85) as u64);
    }

    /// The modified time is the signal: a moved one is read, the same one
    /// is not, even when the text would say otherwise.
    #[test]
    fn a_change_is_seen_once_and_only_when_the_modified_time_moves() {
        let mut watcher = Watcher::new(PathBuf::from("unused"));
        let t1 = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let t2 = SystemTime::UNIX_EPOCH + Duration::from_secs(2);
        assert_eq!(
            watcher.check(Some(t1), || Some("a".to_owned())),
            Some((t1, "a".to_owned()))
        );
        assert_eq!(
            watcher.check(Some(t1), || panic!("not read at the same modified time")),
            None
        );
        assert_eq!(
            watcher.check(Some(t2), || Some("b".to_owned())),
            Some((t2, "b".to_owned()))
        );
        assert_eq!(watcher.check(None, || panic!("a missing file has no reading")), None);
    }

    /// At most twice a second: a look 100 ms after another does not happen.
    #[test]
    fn the_file_is_looked_at_at_most_twice_a_second() {
        let mut watcher = Watcher::new(PathBuf::from("unused"));
        let now = Instant::now();
        assert!(watcher.due(now));
        assert!(!watcher.due(now + Duration::from_millis(100)));
        assert!(!watcher.due(now + Duration::from_millis(499)));
        assert!(watcher.due(now + Duration::from_millis(500)));
    }

    /// The segment statistics on a hand-made list of frame times.
    #[test]
    fn segment_statistics_of_a_hand_made_list() {
        assert_eq!(stats(&[10.0, 20.0, 30.0, 40.0]), (25.0, 20.0, 40.0));
        assert_eq!(stats(&[16.0]), (16.0, 16.0, 16.0));
        let many: Vec<f64> = (1..=100).map(f64::from).collect();
        let (mean, p50, p95) = stats(&many);
        assert_eq!(mean, 50.5);
        assert_eq!((p50, p95), (50.0, 95.0));
    }

    /// A segment's line up to the calls skipped in it and the switches in force (other tests
    /// change both in the one registry).
    fn figures_of(line: &str) -> &str {
        line.split(", skipped ").next().expect("a line")
    }

    /// The line shapes a harness parses are fixed here. (Corrected twice. With the first
    /// timing work: a segment says the frames it left out as transition and what began and
    /// ended it. With the check of that work: frames under 1000 draws and frames over 1000 ms
    /// are left out of the mean and percentiles too, each kind counted with its own mean,
    /// where they were counted and still in the mean.)
    #[test]
    fn the_marker_and_segment_lines_have_their_fixed_shape() {
        assert_eq!(
            marker_line("VICTORY_TEXTURE_LOW", &Value::Number(70), &Value::Number(60), 13),
            "live: VICTORY_TEXTURE_LOW 70 -> 60 at frame 13"
        );
        let mut state = test_state();
        state.segment.frame(9, 200.0, 7000);
        close(&mut state, 9, Ended::Change).expect("the segment before");
        let frames = [
            (10, 300.0, 7000), // the three after the change
            (11, 320.0, 7000),
            (12, 310.0, 7000),
            (13, 90.0, 20), // a light frame
            (14, 1500.0, 7000), // two long frames
            (15, 1500.0, 7000),
            (16, 10.0, 7000),
            (17, 20.0, 7000),
            (18, 30.0, 7000),
            (19, 40.0, 7000),
        ];
        for (frame, ms, draws) in frames {
            state.segment.frame(frame, ms, draws);
        }
        let (open, json) = open(&state).expect("the segment under way");
        assert_eq!(
            figures_of(&open),
            "live segment 2 open: frames 10..19, 4 counted, 3 excluded as transition (the first 3 after a change; their mean \
             310.0), 1 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their \
             mean 90.0), 2 excluded as long (over 1000 ms; their mean 1500.0), mean 25.0, p50 20.0, p95 40.0 of the counted, \
             begun at a change, not ended (the segment under way, as of this line)"
        );
        assert_eq!(
            json,
            "{\"segment\":2,\"first_frame\":10,\"last_frame\":19,\"counted\":4,\"transition_excluded\":3,\
             \"transition_mean_ms\":310.00,\"light_excluded\":1,\"light_mean_ms\":90.00,\"long_excluded\":2,\
             \"long_mean_ms\":1500.00,\"mean_ms\":25.00,\"p50_ms\":20.00,\"p95_ms\":40.00,\"begun\":\"change\"}"
        );
        let closed = close(&mut state, 19, Ended::Change).expect("its line");
        assert_eq!(figures_of(&closed), figures_of(&open).replace(" open:", ":").replace("not ended (the segment under way, as of this line)", "ended by a change"));
        assert!(closed.contains(", skipped ") && closed.contains(" dispatches, in force: "), "{closed}");
    }

    /// A state of the test's own: no control file is ever read (its path names none).
    fn test_state() -> State {
        State::new(PathBuf::from("victory-live-test-no-such-file"))
    }

    /// The first frames after a change are left out of a segment's mean and percentiles and
    /// counted; the first segment of a run, begun by no change, leaves none out as transition;
    /// a frame with few draws (a load, a menu or a pause) and a long frame are left out of
    /// the mean too, each counted with its own mean. (Corrected: this fixture said a 1500 ms
    /// load frame belonged in the first segment's mean, 633.3 ms for two frames of 200.)
    #[test]
    fn a_segment_leaves_its_transition_light_and_long_frames_out_and_says_so() {
        let mut state = test_state();
        // The run's first segment: a load (few draws, long), the scene, then one long frame.
        state.segment.frame(1, 1500.0, 20);
        state.segment.frame(2, 200.0, 7000);
        state.segment.frame(3, 200.0, 7000);
        state.segment.frame(4, 1200.0, 7000);
        let first = close(&mut state, 4, Ended::Change).expect("a line for the first segment");
        assert!(
            first.contains(": frames 1..4, 2 counted, 0 excluded as transition (the first 3 after a change; their mean 0.0), 1 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their mean 1500.0), 1 excluded as long (over 1000 ms; their mean 1200.0), mean 200.0, p50 200.0, p95 200.0 of the counted, begun at the start of the run, ended by a change, "),
            "{first}"
        );
        // The next: a 740 ms frame right after the change, two more, then the steady ones.
        for (frame, ms) in [(5, 740.0), (6, 300.0), (7, 250.0), (8, 220.0), (9, 230.0)] {
            state.segment.frame(frame, ms, 7000);
        }
        let second = close(&mut state, 9, Ended::Change).expect("a line for the second segment");
        assert!(
            second.contains(": frames 5..9, 2 counted, 3 excluded as transition (the first 3 after a change; their mean 430.0), 0 excluded as light ("),
            "{second}"
        );
        assert!(second.contains("mean 225.0, p50 220.0, p95 230.0 of the counted, begun at a change, ended by a change, "), "{second}");
    }

    /// A plan's start cuts a segment: `LIVE_MARK=<n>` in the control file changes no switch
    /// but its own number, ends the segment under way, and the segment it begins leaves no
    /// frame out as transition (nothing changed). So a plan's first arm no longer runs from
    /// the start of the run.
    #[test]
    fn a_mark_cuts_a_segment_and_leaves_no_frame_out() {
        let index = SPECS.iter().position(|spec| spec.name == "LIVE_MARK").expect("LIVE_MARK is a live switch");
        let mut values = fixture();
        let changes = apply(&mut values, parse("LIVE_MARK=7\n").expect("the file parses"));
        assert_eq!(changes.iter().map(|change| change.0).collect::<Vec<_>>(), [index], "the mark alone changed");
        assert_eq!(ended_by(&changes), Ended::Mark);
        // With a switch beside it, it is a change like any other.
        let both = apply(&mut fixture(), parse("LIVE_MARK=8\nVICTORY_TEXTURE_LOW=60\n").expect("the file parses"));
        assert_eq!(ended_by(&both), Ended::Change);
        let mut state = test_state();
        state.segment.frame(1, 900.0, 7000);
        state.segment.frame(2, 200.0, 7000);
        let before = close(&mut state, 2, ended_by(&changes)).expect("the stretch before the plan");
        assert!(before.contains("begun at the start of the run, ended by a mark, "), "{before}");
        for (frame, ms) in [(3, 210.0), (4, 220.0), (5, 230.0)] {
            state.segment.frame(frame, ms, 7000);
        }
        let arm = close(&mut state, 5, Ended::Change).expect("the plan's first arm");
        assert!(arm.contains(": frames 3..5, 3 counted, 0 excluded as transition ("), "{arm}");
        assert!(arm.contains("mean 220.0, p50 220.0, p95 230.0 of the counted, begun at a mark, ended by a change, "), "{arm}");
    }

    /// A run ended by the harness's timer is killed from outside: nothing in the process
    /// runs, and [`finish`] is never called. Its last segment is written all the same: the
    /// line of the segment under way, logged with each frame line, and the stats file's
    /// `segment_open` hold it up to the last report, with the same figures the closing line
    /// would have given for those frames.
    #[test]
    fn a_run_ended_by_the_timer_has_its_last_segment_written() {
        let mut state = test_state();
        state.segment.frame(1, 200.0, 7000);
        close(&mut state, 1, Ended::Change).expect("the segment before the last");
        assert_eq!(state.closed, 1);
        // The last arm: its frames up to the last report before the kill.
        for (frame, ms) in [(2, 700.0), (3, 300.0), (4, 300.0), (5, 240.0), (6, 236.0), (7, 244.0)] {
            state.segment.frame(frame, ms, 7000);
        }
        let (line, json) = open(&state).expect("the segment under way is written while it runs");
        assert!(
            line.starts_with("live segment 2 open: frames 2..7, 3 counted, 3 excluded as transition (the first 3 after a change; their mean 433.3), 0 excluded as light ("),
            "{line}"
        );
        assert!(line.contains("mean 240.0, p50 240.0, p95 244.0 of the counted, begun at a change, not ended (the segment under way, as of this line)"), "{line}");
        assert!(json.starts_with("{\"segment\":2,\"first_frame\":2,\"last_frame\":7,\"counted\":3,\"transition_excluded\":3,"), "{json}");
        assert!(json.contains("\"mean_ms\":240.00,\"p50_ms\":240.00,\"p95_ms\":244.00"), "{json}");
        // Writing it closed nothing: the segment goes on, and its number is not taken.
        assert_eq!(state.closed, 1);
        // The same frames, had the run ended by a path the runtime sees.
        let closed = close(&mut state, 7, Ended::RunEnd).expect("the last segment's line");
        assert_eq!(
            figures_of(&closed),
            figures_of(&line).replace(" open:", ":").replace("not ended (the segment under way, as of this line)", "ended by the end of the run")
        );
        // Ended, a second end says nothing: no segment is logged twice.
        assert_eq!(close(&mut state, 7, Ended::RunEnd), None);
        assert!(open(&state).is_none());
    }

    /// A live key test's serial number: the playback state is one registry,
    /// so these tests take turns.
    static KEYS_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The `LIVE_KEYS` line applied as a frame boundary does; `true` when it
    /// changed the value.
    fn set_live_keys(text: &str) -> bool {
        let _alone = REGISTRY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut values = snapshot();
        let given = parse(&format!("LIVE_KEYS={text}")).expect("the file is whole");
        let changes = apply(&mut values, given);
        for (index, _, new) in changes.clone() {
            store(index, new);
        }
        !changes.is_empty()
    }

    /// Drops the key playback state so tests do not see each other's.
    fn reset_live_keys() {
        let _alone = REGISTRY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        *KEYS_VALUE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *KEYS_ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// A changed non-empty value starts one run, with its presses; applying
    /// the same value again changes nothing and starts nothing.
    #[test]
    fn a_live_keys_change_plays_once_and_the_same_value_again_does_nothing() {
        let _turn = KEYS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_live_keys();
        assert!(set_live_keys("0:W,2:W"), "a new sequence is a change");
        let first = keys().expect("the sequence plays");
        assert_eq!(first.count, 2);
        assert_eq!(first.presses[0].key, 0x57);
        assert_eq!(first.presses[1].at, Duration::from_secs(2));
        assert!(!set_live_keys("0:W,2:W"), "the same value is no change");
        let again = keys().expect("the first run is still the one playing");
        assert_eq!((again.id, again.start), (first.id, first.start), "not restarted");
        // A finished run is not replayed by writing the same value: the
        // value in force is still that sequence.
        keys_done(first.id);
        assert!(!set_live_keys("0:W,2:W"), "still the same value");
        assert!(keys().is_none(), "and nothing starts");
        reset_live_keys();
    }

    /// An empty value cancels what has not played: the run is gone at once.
    #[test]
    fn an_empty_live_keys_value_cancels_the_unplayed_sequence() {
        let _turn = KEYS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_live_keys();
        assert!(set_live_keys("0:W,60:W"));
        assert!(keys().is_some(), "playing before the cancel");
        assert!(set_live_keys(""), "empty is a change");
        assert!(keys().is_none(), "the unplayed presses are cancelled");
        assert_eq!(snapshot()[LIVE_KEYS], Value::Keys(Keys::default()));
        reset_live_keys();
    }

    /// A refused sequence keeps the old value and its playback; the limits
    /// refuse the same way.
    #[test]
    fn a_refused_live_sequence_keeps_the_old_state() {
        let _turn = KEYS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_live_keys();
        assert!(set_live_keys("0:W,2:W"));
        let old = keys().expect("playing").id;
        assert!(!set_live_keys("0:Space"), "an unknown key is refused");
        assert!(!set_live_keys("0:W:wat"), "a malformed hold is refused");
        assert!(!set_live_keys("0:W,,2:W"), "an empty entry is refused");
        let too_many = vec!["0:W"; 65].join(",");
        assert!(!set_live_keys(&too_many), "65 presses are refused");
        assert!(!set_live_keys("121:W:0"), "past 120 seconds is refused");
        assert!(!set_live_keys("0:W:121"), "a hold past 120 seconds is refused");
        assert_eq!(keys().expect("the old run is still playing").id, old);
        assert_eq!(
            snapshot()[LIVE_KEYS].text(),
            "0:W,2:W",
            "the old value stays in force"
        );
        reset_live_keys();
    }

    /// The done line a driver parses has its fixed shape, and a finished run
    /// is stopped once.
    #[test]
    fn the_done_line_has_its_fixed_shape_and_a_run_stops_once() {
        assert_eq!(
            done_line(1234, 2),
            "live keys: sequence done at frame 1234, 2 presses"
        );
        let _turn = KEYS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_live_keys();
        assert!(set_live_keys("0:W"));
        let id = keys().expect("playing").id;
        keys_done(id);
        assert!(keys().is_none(), "done stops the run");
        keys_done(id); // a second report for the same run is ignored
        assert!(keys().is_none());
        reset_live_keys();
    }

    /// With no run, the input side has nothing to play.
    #[test]
    fn no_live_keys_run_means_nothing_to_play() {
        let _turn = KEYS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_live_keys();
        assert!(keys().is_none());
        keys_done(0);
        assert!(keys().is_none());
    }

    /// A listed id is the whole pipeline id (brackets optional) or one of
    /// its shader ids; not a part of one.
    #[test]
    fn a_listed_pipeline_id_matches_itself_or_a_shader_id() {
        let id = "[compute 0123456789abcdef]";
        assert!(token_matches("[compute 0123456789abcdef]", id));
        assert!(token_matches("compute 0123456789abcdef", id));
        assert!(token_matches("0123456789abcdef", id));
        assert!(token_matches("0123456789ABCDEF", id));
        assert!(!token_matches("0123456789abcde", id), "not a prefix");
        assert!(!token_matches("fedcba9876543210", id));
        assert!(!token_matches("", id));
    }
}
