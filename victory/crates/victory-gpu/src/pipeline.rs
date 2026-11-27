//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates tens of thousands of pipelines in its first
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their bytes, translations
/// kept and the bytes of their host code. None of them is ever let go.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

/// Host pipelines made and loaded from the library since the start: what
/// tells, once a scene has settled, whether pipeline states are being made
/// again (the 600-frame line and the live stats file; `host/background.rs`).
#[cfg(windows)]
pub(crate) fn library_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (library::MADE.load(Relaxed), library::LOADED.load(Relaxed))
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let (full, id) = victory_shader::container_id(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(full)
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader {
                stage,
                hash: id,
                container: bytes.to_vec(),
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// Puts the translation of a container of the title's into the cache on
/// disk, and with `compute` its host pipeline into the driver's, before any
/// run wants them. Whether it translated, and whether it has a pipeline
/// (`None`: not asked for, or left out).
#[cfg(windows)]
pub fn warm(stage: &'static str, container: &[u8], compute: bool) -> (bool, Option<bool>) {
    let full = blake3::hash(container);
    // Not through the table, which lets nothing go.
    let shader = Arc::new(Shader {
        stage,
        hash: full.to_hex()[..16].to_owned(),
        container: container.to_vec(),
        translated: OnceLock::new(),
    });
    let translated = shader.translated().is_some();
    let made = (translated && compute).then(|| make::warm(vec![shader])).flatten();
    (translated, made)
}

/// Tests: a compute shader already translated to `host`, its console
/// program reading `used`, for [`host::HostPipeline::new`] to make a
/// pipeline of as it makes the title's.
#[cfg(all(windows, test))]
pub(crate) fn test_translated_program(
    host: victory_shader::emit::HostShader,
    used: Vec<victory_shader::UsedResource>,
) -> Arc<Shader> {
    Arc::new(Shader {
        stage: "compute",
        hash: "translated test".to_owned(),
        container: Vec::new(),
        translated: OnceLock::from(Some(Translated { host, used, stores: true })),
    })
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let done = translate(&self.container);
                // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
                // translated (`.dxil`), by the hash crash reports name it with
                if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let stem = format!("{}.{}", self.hash, self.stage);
                    let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &self.container);
                    if let Ok(t) = &done {
                        let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &t.host.dxil);
                    }
                }
                if let Ok(t) = &done {
                    KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KEPT[3].fetch_add(t.host.dxil.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let host = crate::shader_cache::host_shader(container)?;
    let stores = programs.is_empty() || programs.iter().any(stores);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    Ok(Translated { host, used, stores })
}

/// The compute pipelines taken for a ray-tracing structure builder's
/// stages ([`of_builder`]) run **all or none**. Such stages feed each
/// other through memory: a stage that runs on what a dropped stage never
/// wrote can compute where it stores from garbage. The structures it
/// builds lie in plain title memory, and a root view has no size, so
/// nothing stops such a store from landing on whatever lies next to them.
///
/// So a builder dispatch runs only while every builder pipeline the title
/// has created has a host pipeline; the first builder dispatch makes them
/// all on the spot instead of over the next frames (a stage "still being
/// made" would be a dropped stage). One that cannot be made turns the
/// builder off for the rest of the run.
///
/// shortcut: and only with `VICTORY_BVH_BUILDER=1`, until a run has shown
/// the whole builder and the host side of title memory by address
/// (`memory.rs`) working on a card; nothing reads the structures before
/// the programs that walk them translate.
#[cfg(windows)]
struct Builder {
    pipelines: Vec<Arc<make::Pending>>,
    /// Every one of `pipelines` has a host pipeline.
    whole: bool,
    /// One could not be made.
    off: bool,
}

#[cfg(windows)]
static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pipelines: Vec::new(),
    whole: false,
    off: false,
});

/// Whether `shaders` are a pipeline of the structure builder.
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    use victory_shader::{Container, FourCC};
    let [shader] = shaders else { return false };
    if shader.stage != "compute" {
        return false;
    }
    let Ok(c) = Container::parse(&shader.container) else {
        return false;
    };
    c.part(FourCC::DXIL)
        .and_then(|part| victory_shader::native_programs(part.data).ok())
        .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    is_builder(this) && bvh_quarantine()
}

/// Whether `VICTORY_BVH_QUARANTINE=1` is set: the builder's stores then run undeclared, and
/// the copy-back's fetch stays on the drawing queue (`host/fetch.rs`).
#[cfg(windows)]
pub(crate) fn bvh_quarantine() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs from here on.
#[cfg(windows)]
pub(crate) fn builder_off(why: &'static str) {
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if !b.off {
        b.off = true;
        tracing::warn!(why, "the ray-tracing structure builder is turned off for the rest of the run: its stages run all or none");
    }
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`Builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    if !*ALLOWED.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")) {
        return false;
    }
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if b.off {
        return false;
    }
    if b.whole {
        return true;
    }
    let missing = b.pipelines.iter().filter(|p| !make::now(p)).count();
    if missing > 0 {
        b.off = true;
        tracing::warn!(
            missing,
            of = b.pipelines.len(),
            "the ray-tracing structure builder stays off: not all of its programs have a host pipeline, and its stages run all or none"
        );
        return false;
    }
    tracing::info!(
        pipelines = b.pipelines.len(),
        "the ray-tracing structure builder runs: all of its programs have host pipelines"
    );
    b.whole = true;
    true
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// One of the ray-tracing structure builder's ([`Builder`]).
    #[cfg(windows)]
    builder: bool,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// The id of title pipeline `this` as the log names it: its stages and
/// shader ids, `[compute 0123..]` (`None` for no pipeline). The live skip
/// switches ([`crate::live`]) match their lists against this.
pub(crate) fn live_id(this: This) -> Option<String> {
    (!this.is_null()).then(|| format!("{:?}", pipeline(this).shaders))
}

/// The stages and hashes of title pipeline `this`, as `[mesh 0123..,
/// pixel 4567..]` (diagnostics); "no pipeline" for none.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shaders_of(this: This) -> String {
    if this.is_null() {
        return "no pipeline".to_owned();
    }
    let names: Vec<String> = pipeline(this).shaders.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
    format!("[{}]", names.join(", "))
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    crate::capture::op(0, || format!("DROPPED {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    let builder = of_builder(&shaders);
    #[cfg(windows)]
    if builder {
        // Another stage: the group is looked at again at its next dispatch.
        let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
        b.pipelines.push(host.clone());
        b.whole = false;
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Lets the pipeline library's mapped file go from memory when no pipeline
/// has loaded for a while (`library::trim`); called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// Tests: holds a library call open, as a pipeline being made does, until dropped.
#[cfg(test)]
pub(crate) fn hold_call_for_tests() -> impl Drop {
    library::call_for_tests()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

/// Tests: containers of the console's made here, with one native vertex
/// program each (GFX10 code and its root tables, as `victory_shader::native`
/// reads them), and title pipelines made of them as the title makes them,
/// from a stream, with their host pipelines made: what draws reach through
/// the bind path (`crate::draw::State::bind`) and its classification of
/// programs that store ([`stores`]).
#[cfg(all(test, windows, feature = "diagnostics"))]
pub(crate) mod test_programs {
    use super::*;
    use victory_shader::{FourCC, ResourceType, RootParam, RootType, UsedResource};

    /// A container holding one native vertex program: wave32, `vgprs`
    /// VGPRs, `code`, the root layout `layout` and the resources `used`, and
    /// stage data with no inputs and no outputs.
    pub(crate) fn vertex_container(vgprs: u32, code: &[u32], layout: &[RootParam], used: &[UsedResource]) -> Vec<u8> {
        const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        const RECORD: usize = 0x54;
        const STAGE_DATA: usize = 0x200;
        // The payload's record table: LS HS ES GS VS PS, mesh, compute.
        const VERTEX: usize = 0x28;
        let root = |r: RootType| match r {
            RootType::Constants => 0u8,
            RootType::Descriptor => 1,
            RootType::Table => 3,
            _ => unimplemented!("a root type the tests do not make"),
        };
        let mut payload = 6u64.to_le_bytes().to_vec();
        payload.resize(0x68, 0);
        for at in [0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        let record = payload.len();
        payload[VERTEX..VERTEX + 8].copy_from_slice(&(record as u64).to_le_bytes());
        payload.resize(record + RECORD + STAGE_DATA, 0);
        let tables = payload.len();
        for p in layout {
            let mut e = [0u8; 20];
            e[0] = root(p.root);
            e[4..8].copy_from_slice(&p.table_offset_dwords.to_le_bytes());
            e[8..12].copy_from_slice(&p.sgprs.to_le_bytes());
            e[12..16].copy_from_slice(&p.base_register.to_le_bytes());
            (e[16], e[17]) = (p.user_data, p.slot);
            payload.extend_from_slice(&e);
        }
        let used_at = payload.len();
        for u in used {
            let mut e = [0u8; 20];
            e[0] = match u.kind {
                ResourceType::Srv => 0,
                ResourceType::Sampler => 1,
                ResourceType::Cbv => 2,
                ResourceType::Uav => 3,
                ResourceType::Other(n) => n,
            };
            e[1] = root(u.root);
            e[4..8].copy_from_slice(&u.table_offset_dwords.to_le_bytes());
            e[8..12].copy_from_slice(&u.lower_bound.to_le_bytes());
            e[12..16].copy_from_slice(&u.stride.to_le_bytes());
            (e[16], e[17], e[18]) = (u.dwords, u.user_data, u.slot);
            payload.extend_from_slice(&e);
        }
        let code_at = payload.len();
        payload.extend(code.iter().flat_map(|w| w.to_le_bytes()));
        let put = |payload: &mut Vec<u8>, at: usize, bytes: &[u8]| payload[record + at..record + at + bytes.len()].copy_from_slice(bytes);
        put(&mut payload, 0x00, &(code_at as u64).to_le_bytes());
        put(&mut payload, 0x08, &(4 * code.len() as u64).to_le_bytes());
        put(&mut payload, 0x14, &(1u32 << 5).to_le_bytes());
        put(&mut payload, 0x28, &(tables as u64).to_le_bytes());
        put(&mut payload, 0x30, &(used_at as u64).to_le_bytes());
        put(&mut payload, 0x38, &(layout.len() as u32).to_le_bytes());
        put(&mut payload, 0x3c, &(used.len() as u32).to_le_bytes());
        put(&mut payload, 0x50, &(vgprs | 16 << 20).to_le_bytes());
        let mut fssm = b"FSSM".to_vec();
        fssm.extend_from_slice(&0x101u32.to_le_bytes());
        fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        fssm.extend_from_slice(&payload);
        let mut packed = vec![0u8; zlib_rs::compress_bound(fssm.len())];
        let (stream, rc) = zlib_rs::compress_slice(&mut packed, &fssm, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        let mut blob = b"XBCH".to_vec();
        blob.extend_from_slice(&3u32.to_le_bytes());
        blob.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        blob.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
        blob.extend_from_slice(stream);
        // The part: its program header, the index of one blob, the blob.
        let mut part = vec![0u8; 0x18];
        part.extend_from_slice(b"XDAT");
        part.extend_from_slice(&1u32.to_le_bytes());
        part.extend_from_slice(&0x24u32.to_le_bytes());
        let at = part.len() + 12;
        for v in [0x0d, at as u32, blob.len() as u32] {
            part.extend_from_slice(&v.to_le_bytes());
        }
        part.extend_from_slice(&blob);
        // The container: its header (no digest), the part's offset, the part.
        let total = 32 + 4 + 8 + part.len();
        let mut out = b"DXBC".to_vec();
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&36u32.to_le_bytes());
        out.extend_from_slice(&FourCC::DXIL.0);
        out.extend_from_slice(&(part.len() as u32).to_le_bytes());
        out.extend_from_slice(&part);
        out
    }

    /// A title graphics pipeline of the vertex program in `vs`, drawing
    /// `topology` (a primitive topology type) with no targets and no depth
    /// test, made from a stream as the title makes one; its host pipeline
    /// made on this thread (`None` if it was refused).
    pub(crate) fn graphics(vs: &[u8], topology: u32) -> (This, Option<&'static host::HostPipeline>) {
        let mut stream = Vec::new();
        // A subobject: its type, then its data from `from` on, `size` bytes in all.
        let mut sub = |kind: u32, from: usize, data: &[u8], size: usize| {
            let at = stream.len();
            stream.resize(at + size, 0);
            stream[at..at + 4].copy_from_slice(&kind.to_le_bytes());
            stream[at + from..at + from + data.len()].copy_from_slice(data);
        };
        let code = [(vs.as_ptr() as u64).to_le_bytes(), (vs.len() as u64).to_le_bytes()].concat();
        sub(1, 8, &code, 24);
        sub(14, 4, &topology.to_le_bytes(), 8);
        // Depth-stencil: no depth or stencil test; comparisons "always", stencil
        // operations "keep".
        let words: [u32; 13] = [0, 0, 8, 0, 0xffff, 1, 1, 1, 8, 1, 1, 1, 8];
        let depth: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        sub(11, 4, &depth, 56);
        let desc = StreamDesc { size: stream.len(), stream: stream.as_ptr() };
        let mut p = std::ptr::null();
        assert_eq!(create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p), crate::object::S_OK);
        let made = make::now(&pipeline(p).host);
        (p, made.then(|| host_pipeline(p)).flatten())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// A stream assembled for this test, its subobjects in an order of the test's own, each
    /// where the sizes of those before it put it (the sizes written out here, not taken
    /// from [`subobject`]): topology, a pixel shader, a sample description, the render
    /// target formats, a mesh shader, the rasterizer state and the sample mask.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream: Vec<u8> = Vec::new();
        // Appends a zeroed subobject of `kind` and `size` bytes; its offset.
        let add = |stream: &mut Vec<u8>, kind: u32, size: usize| {
            let at = stream.len();
            stream.resize(at + size, 0);
            stream[at..at + 4].copy_from_slice(&kind.to_le_bytes());
            at
        };
        let put = |stream: &mut Vec<u8>, at: usize, v: u32| stream[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let shader = |stream: &mut Vec<u8>, at: usize, code: &[u8]| {
            stream[at + 8..at + 16].copy_from_slice(&(code.as_ptr() as u64).to_le_bytes());
            stream[at + 16..at + 24].copy_from_slice(&(code.len() as u64).to_le_bytes());
        };
        // Topology type 2 (lines).
        let at = add(&mut stream, 14, 8);
        put(&mut stream, at + 4, 2);
        let at = add(&mut stream, 2, 24);
        shader(&mut stream, at, &ps);
        add(&mut stream, 17, 16);
        // Two render targets, of formats 10 and 24.
        let at = add(&mut stream, 15, 40);
        put(&mut stream, at + 4, 10);
        put(&mut stream, at + 8, 24);
        put(&mut stream, at + 36, 2);
        let at = add(&mut stream, 24, 24);
        shader(&mut stream, at, &ms);
        add(&mut stream, 10, 48);
        add(&mut stream, 9, 8);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        // In the stream's order.
        assert_eq!(shaders(p), [("pixel", ps.to_vec()), ("mesh", ms.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 2);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0, g.targets.RTFormats[1].0),
                (2, 10, 24)
            );
        }
    }
}
