//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots, and the count the call reads is the clamp's, in the scratch
//! buffer after the records ([`count_at`]), not the title's.
//!
//! The title's argument and count buffers are host copies of title memory
//! ([`crate::shadow`]), which the title's commands also read (vertices,
//! indices, constants, views, picked from a heap by index or not) and store
//! into, in the same lists as the clamps. Legacy resource states cannot say
//! where a clamp's reads stand among those: a buffer takes on a state where
//! it is first used (a read state, or unordered access for a store) until
//! the lists sent together are done, nothing the commands say tells which (a
//! view may be bound and not used; a shader that picks buffers by index
//! names none), and a transition needs the state it starts from.
//!
//! So the clamps read the copies through shader resource views and order
//! those reads with enhanced barriers ([`Reads`]), where the device has them
//! ([`crate::host::Host::enhanced_barriers`]). A buffer has no layout, and a
//! buffer barrier says only what must be done and visible before what:
//! before a pass, every earlier store into the copies it reads and every
//! copy into them (any stage) is done and visible to its loads; after it, a
//! later store or copy waits for its loads. No state is set, so none is
//! guessed. Our own buffers (the scratch the clamps write, the budget, the
//! slots) keep their legacy transitions.
//!
//! shortcut: where the device has no enhanced barriers (or with
//! `VICTORY_ENHANCED_BARRIERS=0`) each call is clamped alone (no runs), its
//! copies read through shader resource views in whatever state the title's
//! commands left them, as before the clamp of runs: a read after a store of
//! the same submission finds a copy in unordered access, which a read does
//! not allow.
//!
//! Each execution of a list has a budget of its own (2^27
//! vertex instances or threads, groups counted as 64 threads, mesh groups
//! as 128), which bounds what one execution can cost the card: active
//! records past it run empty, saturating without 32-bit wraparound so a
//! list of thousands of garbage records cannot run for seconds either. A
//! list a submission names twice runs twice, the second time once the
//! first is done (`Host::execute`), each from a full budget. The budget is in
//! the card's memory, zeroed there where the list first uses it, by
//! transitions of the list's own ([`reset_budget_on`], [`note_budget_on`]):
//! in the PC's memory each clamp's atomics on it crossed the bus, about 3.2
//! of a clamped call's 7.8 microseconds on an RTX 5080 (the benchmark
//! below, p against p1).
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0, flags = DATA_VOLATILE), SRV(t1, flags = DATA_VOLATILE), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: one dispatch, written as
    // SPLIT records of at most 65,535 groups in x each.
    uint has_count;
    uint count_offset;
};
// The title's argument copy and the copy its count is in: read here, never
// written, through shader resource views (the module's comment). Volatile:
// the title's commands store into them after the pass, which a view of
// static data forbids while it is bound.
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
        // The count the call reads, after the records.
        if (id.x == 0) {
            dst_args.Store(records * stride, eff_count);
        }
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && asked.x <= SPLIT * 65535u) {
            g.x = asked.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is out
        // of range (a slot not yet filled, as far as the clamp can tell):
        // empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_at + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_for_parts(device, split())
}

/// The clamp on `device`, making a dispatch in parts of `parts` records
/// ([`source`]).
fn make_for_parts(device: &ID3D12Device, parts: u32) -> windows::core::Result<Clamp> {
    make_from(device, &source(parts))
}

/// The line of [`SOURCE`] that says how many parts a dispatch in parts
/// has ([`bound_split`]), as it is at the default.
const SPLIT_LINE: &str = "static const uint SPLIT = 2;";

/// The clamp's text for `parts` parts: [`SOURCE`] itself at the default
/// ([`SPLIT`]), else with its [`SPLIT_LINE`] saying `parts`. Compiled once
/// a process ([`clamp`]); nothing keys a cache on it.
fn source(parts: u32) -> std::borrow::Cow<'static, str> {
    if parts == SPLIT {
        std::borrow::Cow::Borrowed(SOURCE)
    } else {
        std::borrow::Cow::Owned(SOURCE.replacen(SPLIT_LINE, &format!("static const uint SPLIT = {parts};"), 1))
    }
}

/// The clamp of runs' pass ([`batched`]), on `device`.
fn make_batch_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, BATCH)
}

/// A pass of ours with its root signature inside, compiled from `source`.
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`,
/// with the count the call reads at [`count_at`] when `count` is given.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let size = if count.is_some() { count_at(spec.1, spec.2) + 4 } else { bytes };
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(size)?);
    counted(list, src.0, spec.0, stamped(list, || record_on(c, list, &dst, budget, src, count, spec, 0))).then_some(dst)
}

/// Where in [`bound`]'s scratch buffer the count of a call with a count
/// buffer is (`records` records of `stride` bytes): after the records.
/// `ExecuteIndirect` reads it there, in the indirect-argument state the
/// clamp leaves the buffer in, and never the title's count, whose state is
/// the title's (the module's comment).
pub(crate) fn count_at(stride: u32, records: u32) -> u64 {
    u64::from(stride) * u64::from(records)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones written are whatever the buffer holds, and each record clamped
/// charges the list's budget when it is clamped: a record over the most a
/// mesh dispatch runs (65,535 groups), clamped to that most, is charged a
/// sixteenth of the budget, so sixteen of them emptied every draw of the
/// list after them (run 93: 43 to 153 such records a frame). Here a record
/// that asks for more than 65,535 groups is out of range: it is left empty
/// and charges nothing; a record clamped alone is clamped and charged as
/// before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let recorded = stamped(list, || record_on(c, list, &dst, budget, src, None, spec, AHEAD));
    counted(list, src.0, spec.0, recorded).then_some(dst)
}

/// The parts [`bound_split`] makes of a dispatch by default, the most
/// groups of each, and the bytes between them.
pub(crate) const SPLIT: u32 = 2;
pub(crate) const PART: u32 = 65_535;
pub(crate) const PART_BYTES: u64 = 12;
/// The most parts [`split`] takes: sixteen parts are 1,048,560 groups,
/// under the most one dispatch runs (the clamp's `GROUPS`, 2^20).
pub(crate) const SPLIT_MOST: u32 = 16;
const _: () = assert!(SPLIT_MOST * PART <= 1 << 20, "the parts must fit what one dispatch runs");

/// The parts of a dispatch in parts ([`bound_split`]):
/// `VICTORY_SPLIT_PARTS=<n>`, read once, 2 to 16; unset (or anything
/// else, with one warning) [`SPLIT`]. At 3840x2160 a screen-sized pass
/// asks for about 259,200 groups, which takes 4.
pub(crate) fn split() -> u32 {
    static PARTS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PARTS.get_or_init(|| {
        let value = std::env::var_os("VICTORY_SPLIT_PARTS").map(|v| v.to_string_lossy().into_owned());
        let (parts, taken) = split_from(value.as_deref());
        if !taken {
            tracing::warn!(value = value.as_deref().unwrap_or(""), "VICTORY_SPLIT_PARTS takes 2 to 16: 2 parts");
        }
        parts
    })
}

/// The parts `value` (the setting's text, if set) asks for, and whether it
/// was taken: unset is [`SPLIT`] and taken; anything but a number from
/// [`SPLIT`] to [`SPLIT_MOST`] is [`SPLIT`] and not.
fn split_from(value: Option<&str>) -> (u32, bool) {
    match value.map(|v| v.trim().parse::<u32>()) {
        None => (SPLIT, true),
        Some(Ok(n)) if (SPLIT..=SPLIT_MOST).contains(&n) => (n, true),
        Some(_) => (SPLIT, false),
    }
}

/// The scratch bytes of a dispatch in `parts` parts: a record each.
fn split_bytes(parts: u32) -> u64 {
    u64::from(parts) * PART_BYTES
}

/// [`bound`] for one dispatch that a program with a group base runs
/// (`victory_shader::emit::USER_GROUP_BASE`): [`split`] records follow each
/// other in the scratch buffer, of which the first has the first [`PART`]
/// groups in x and each next the [`PART`] after (empty if there are none).
/// The console runs more groups in one dispatch than D3D12 takes (the title's final
/// gather of indirect light, several groups for each screen tile); clamped
/// to what the host runs, the tiles at the bottom of the screen kept what
/// they showed before. A dispatch asking for more than the parts hold is
/// clamped to what one dispatch runs, as before.
pub(crate) fn bound_split(list: &HostList, src: (&ID3D12Resource, u64)) -> Option<ID3D12Resource> {
    let spec = (2, 12, 1);
    validate_layout(spec, src.1, None)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let dst = list.indirect_scratch(split_bytes(split()))?;
    let recorded = stamped(list, || record_on(c, list, &dst, budget, src, None, spec, IN_PARTS));
    counted(list, src.0, spec.0, recorded).then_some(dst)
}

/// Executes the `parts` records [`bound_split`] made in `scratch` with
/// `signature` (one dispatch, records [`PART_BYTES`] apart), each after
/// setting the program's group base (root parameter and dword `at`) to its
/// first group, the base set back to 0 after. A part with no groups is
/// executed as it is, an empty dispatch.
///
/// # Safety
///
/// `list` is open, with the program's root signature, pipeline and
/// bindings set; `scratch` holds `parts` records, in the indirect-argument
/// state.
pub(crate) unsafe fn execute_parts(
    list: &ID3D12GraphicsCommandList,
    signature: &ID3D12CommandSignature,
    scratch: &ID3D12Resource,
    (param, dword): (u32, u32),
    parts: u32,
) {
    // SAFETY: as the caller promises.
    unsafe {
        for part in 0..parts {
            list.SetComputeRoot32BitConstant(param, part * PART, dword);
            list.ExecuteIndirect(signature, 1, scratch, u64::from(part) * PART_BYTES, None, 0);
        }
        list.SetComputeRoot32BitConstant(param, 0, dword);
    }
}

/// `record`, a clamp-and-barrier group recorded into `list`, between a pair
/// of timestamps with `VICTORY_GPU_STAMPS=1` (`crate::stamps`): its span on
/// the queue, its two waits for every earlier shader write included. A
/// group that recorded nothing (`record` said false) is no pair.
fn stamped(list: &HostList, record: impl FnOnce() -> bool) -> bool {
    // Named for outside GPU tools too (`crate::events`).
    crate::events::begin(&list.list, crate::events::CLAMP);
    #[cfg(feature = "diagnostics")]
    let begun = list.stamp_begin();
    let recorded = record();
    #[cfg(feature = "diagnostics")]
    list.stamp_end(crate::stamps::Kind::Clamp, begun, recorded);
    crate::events::end(&list.list);
    recorded
}

/// A clamp `recorded` for one call reading records of `kind` from `src`:
/// a call of the list's census of runs (diagnostics, `crate::runs`; a
/// dispatch ends its run).
fn counted(list: &HostList, src: &ID3D12Resource, kind: u32, recorded: bool) -> bool {
    // A call clamped alone is charged where it is: the batch open before it
    // (the clamp of runs, [`batched`]) ends, so the calls after it are
    // charged after it. Nothing is recorded for that.
    if recorded {
        list.batch_end();
    }
    #[cfg(feature = "diagnostics")]
    if recorded {
        let copy = windows::core::Interface::as_raw(src) as usize;
        list.runs(|runs| runs.call(copy, kind == 2));
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (list, src, kind);
    recorded
}

/// Bits of the clamp's `has_count`.
const AHEAD: u32 = 2;
const IN_PARTS: u32 = 4;

#[cfg(test)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_as(c, l, dst, budget, src, count, spec, 0)
}

/// [`record_clamp`]; `how`: [`AHEAD`], the records are a window clamped
/// ahead of their draws ([`bound_ahead`]); [`IN_PARTS`], one dispatch in
/// parts ([`bound_split`]). On a list of ours (the tests', the
/// benchmark's): its reads of `src` and `count` ordered with enhanced
/// barriers where the list has them ([`Reads`]).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    how: u32,
) -> bool {
    let Some(pass) = pass(dst, budget, src, count, spec, how) else {
        return false;
    };
    let barriers: Option<ID3D12GraphicsCommandList7> = windows::core::Interface::cast(l).ok();
    record_pass(c, l, dst, &pass, &Reads::new([src.0].into_iter().chain(count.map(|c| c.0)), barriers.as_ref()));
    true
}

/// Records the clamp of `spec`'s records from `src` (bounded by `count`)
/// into `dst`, charging `budget` (`how`: [`AHEAD`], a window clamped ahead
/// of its draws; [`IN_PARTS`], one dispatch in parts), on a host list: its
/// reads of `src` and `count` ordered with enhanced barriers where the list
/// has them ([`Reads`]). False if [`pass`] refuses.
#[allow(clippy::too_many_arguments)]
fn record_on(
    c: &Clamp,
    list: &HostList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    how: u32,
) -> bool {
    let Some(pass) = pass(dst, budget, src, count, spec, how) else {
        return false;
    };
    let reads = Reads::new([src.0].into_iter().chain(count.map(|c| c.0)), list.enhanced());
    record_pass(c, &list.list, dst, &pass, &reads);
    true
}

/// Records `pass` into `l` between its barriers: `dst` goes `COMMON ->
/// UNORDERED_ACCESS -> INDIRECT_ARGUMENT`, and the pass's reads of the
/// title's copies are ordered as `reads` says (whether or not it has
/// records). Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves
/// the waits for earlier work out (the two for every earlier shader write,
/// and the copies' barriers; the scratch's state changes stay).
fn record_pass(c: &Clamp, l: &ID3D12GraphicsCommandList, dst: &ID3D12Resource, pass: &Pass, reads: &Reads<'_>) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [uav_barrier(), transition(dst, COMMON, UNORDERED)];
    let second = [uav_barrier(), transition(dst, UNORDERED, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)];
    let skip = usize::from(!waits);
    // SAFETY: an open list; a live scratch destination in the common state.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip..]);
        if waits {
            reads.before();
        }
        pass.record(c, l);
        l.ResourceBarrier(&second[skip..]);
        if waits {
            reads.after();
        }
    }
}

const UNORDERED: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
const COMMON: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COMMON;

/// What a buffer of the title's copies has done before a clamp pass reads
/// it, and may do after, that orders with the pass's loads: stores
/// (unordered access, by any shader or clear) and copies into it (title
/// copies, query resolves, the host's refreshes). Reads need no order with
/// the pass's reads.
const WRITES: D3D12_BARRIER_ACCESS =
    D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 | D3D12_BARRIER_ACCESS_COPY_DEST.0);

/// The title's copies a clamp pass reads, each once, and the list's
/// enhanced-barrier interface where its reads of them are ordered by
/// enhanced barriers (the module's comment; `None`: the legacy fallback,
/// nothing recorded for them).
struct Reads<'a> {
    copies: Vec<&'a ID3D12Resource>,
    barriers: Option<&'a ID3D12GraphicsCommandList7>,
}

impl<'a> Reads<'a> {
    fn new(
        copies: impl IntoIterator<Item = &'a ID3D12Resource>,
        barriers: Option<&'a ID3D12GraphicsCommandList7>,
    ) -> Reads<'a> {
        Reads { copies: distinct(copies), barriers }
    }

    /// Before the pass: every earlier store into the copies and copy into
    /// them, at any stage, done and visible to its loads.
    fn before(&self) {
        self.record(
            (D3D12_BARRIER_SYNC_ALL, WRITES),
            (D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE),
        );
    }

    /// After the pass: a later store into the copies or copy into them, at
    /// any stage, waits for its loads.
    fn after(&self) {
        self.record(
            (D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE),
            (D3D12_BARRIER_SYNC_ALL, WRITES),
        );
    }

    /// One buffer barrier a copy, the whole buffer, from `before` (sync,
    /// access) to `after`.
    fn record(
        &self,
        before: (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS),
        after: (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS),
    ) {
        let Some(l) = self.barriers else {
            return;
        };
        let barriers: Vec<D3D12_BUFFER_BARRIER> = self
            .copies
            .iter()
            .map(|copy| D3D12_BUFFER_BARRIER {
                SyncBefore: before.0,
                SyncAfter: after.0,
                AccessBefore: before.1,
                AccessAfter: after.1,
                // SAFETY: the barrier holds no reference; the copy lives
                // through the call.
                pResource: unsafe { std::mem::transmute_copy(*copy) },
                Offset: 0,
                Size: u64::MAX,
            })
            .collect();
        let group = D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: barriers.len() as u32,
            Anonymous: D3D12_BARRIER_GROUP_0 { pBufferBarriers: barriers.as_ptr() },
        };
        // SAFETY: an open list; live buffers.
        unsafe { l.Barrier(&[group]) };
    }
}

/// `copies`, each once.
fn distinct<'a>(copies: impl IntoIterator<Item = &'a ID3D12Resource>) -> Vec<&'a ID3D12Resource> {
    use windows::core::Interface;
    let mut seen: Vec<&ID3D12Resource> = Vec::new();
    for copy in copies {
        if !seen.iter().any(|s| s.as_raw() == copy.as_raw()) {
            seen.push(copy);
        }
    }
    seen
}

/// A clamp pass, validated and addressed: what [`record_clamp_as`] binds
/// and dispatches between its barriers.
struct Pass {
    constants: [u32; 6],
    src: u64,
    count: u64,
    dst: u64,
    budget: u64,
    records: u32,
}

/// The pass that clamps `spec`'s records from `src` (bounded by `count`)
/// into `dst`, charging `budget` (see [`record_clamp_as`]); `None` if a
/// buffer or offset is not on its four-byte line, or the records do not
/// fit.
fn pass(
    dst: &ID3D12Resource,
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    how: u32,
) -> Option<Pass> {
    let bytes = validate_layout((kind, stride, records), src_offset, count.map(|(_, off)| off))?;
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return None;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => (src_base.checked_add(src_offset)?, 0),
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return None;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => (base.checked_add(off)?, 1u32, 0u32),
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    Some(Pass {
        constants: [kind, stride, records, args_offset, has_count | how, count_offset],
        src: src_gpu,
        count: count_gpu,
        dst: dst_gpu,
        budget,
        records,
    })
}

impl Pass {
    /// Binds the clamp and records its dispatch (nothing for no records):
    /// replaces the list's compute root signature and pipeline. The
    /// caller orders it with what writes the source and reads `dst`, and
    /// with what accesses the source and count after it (the module's
    /// comment).
    fn record(&self, c: &Clamp, l: &ID3D12GraphicsCommandList) {
        if self.records == 0 {
            return;
        }
        // SAFETY: an open list; our root signature and pipeline; live
        // source, count, scratch destination, and budget buffers.
        unsafe {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, self.constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, self.src);
            l.SetComputeRootShaderResourceView(2, self.count);
            l.SetComputeRootUnorderedAccessView(3, self.dst);
            l.SetComputeRootUnorderedAccessView(4, self.budget);
            l.Dispatch(self.records.div_ceil(64), 1, 1);
        }
    }
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// The bytes of a list's budget the clamp uses: the work charged, then its
/// two notes on mesh dispatches (the largest group count a record asked
/// for, the groups asked in all).
pub(crate) const BUDGET_BYTES: u64 = 12;

/// Where a list first uses its budget `budget` (in the card's memory, in
/// the common state: as made, and as [`note_budget_on`] leaves it): moved
/// to copy destination, zeroed by a copy from `zeros` (upload memory,
/// which stays in its one state), then left in unordered access for the
/// clamps to charge. Recorded in the list, so each execution of it starts
/// at zero, by transitions of its own and not by the promotion and decay
/// of states between executions.
pub(crate) fn reset_budget_on(l: &ID3D12GraphicsCommandList, budget: &ID3D12Resource, zeros: &ID3D12Resource) {
    // SAFETY: an open list; live buffers, `zeros` in upload memory.
    unsafe {
        l.ResourceBarrier(&[transition(budget, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        l.CopyBufferRegion(budget, 0, zeros, 0, BUDGET_BYTES);
        l.ResourceBarrier(&[transition(
            budget,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
}

/// Where a list that ran [`reset_budget_on`] ends: the budget's words
/// copied into `notes`, in the PC's memory, where the CPU reads what the
/// clamps noted; then both back in the common state, where the next
/// execution starts from.
pub(crate) fn note_budget_on(l: &ID3D12GraphicsCommandList, budget: &ID3D12Resource, notes: &ID3D12Resource) {
    let (common, unordered) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    // SAFETY: an open list; live buffers, the budget in unordered access
    // since its reset, the notes in the common state.
    unsafe {
        l.ResourceBarrier(&[transition(budget, unordered, source), transition(notes, common, dest)]);
        l.CopyBufferRegion(notes, 0, budget, 0, BUDGET_BYTES);
        l.ResourceBarrier(&[transition(budget, source, common), transition(notes, dest, common)]);
    }
}

/// Whether the calls of a run share one clamp pass ([`batched`]): on by
/// default; `VICTORY_CLAMP_RUNS=0` turns it off, and every call is clamped
/// alone, as before.
///
/// A guest may draw most of a frame as one-record indirect calls, and each
/// had a clamp of its own between two waits (thousands a frame, tens of
/// milliseconds, measured). The census of runs (`crate::runs`) found three
/// quarters of them in a few runs of 64 calls or more, with nothing between
/// their calls that can write buffer memory; most runs are one call each,
/// a title dispatch computing the record just before. In the window view
/// of the timing runs the clamps took a sixth of the time a frame with it
/// that they took without, the picture the same.
pub(crate) fn runs_on() -> bool {
    #[cfg(test)]
    if let Some(on) = RUNS_FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| runs_from(std::env::var_os("VICTORY_CLAMP_RUNS").as_deref()))
}

/// [`runs_on`] for the switch's value: on unless it is `0`.
fn runs_from(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

#[cfg(test)]
thread_local! {
    /// Tests: [`runs_on`] on this thread, whatever the switch says.
    pub(crate) static RUNS_FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests: host lists on this thread have no enhanced barriers
    /// ([`HostList::enhanced`]): the legacy fallback.
    pub(crate) static LEGACY_FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The bytes of a call's slot: records of this stride or less are batched.
pub(crate) const SLOT: u64 = 64;
/// The calls of a run's first batch; a batch that fills hands the run on
/// to one twice its size, up to [`MOST`].
const FIRST: u32 = 4;
pub(crate) const MOST: u32 = 1024;
/// The argument copies one pass reads (its root views t1 to t4:
/// [`record_batch`]).
const SOURCES: usize = 4;
/// The work list's bytes before its records, and a record's.
const HEAD: u64 = 16;
const ENTRY: u64 = 16;
/// The slot buffer's first size; one found short is made twice as large
/// at the list's next recording.
const SLOT_BYTES: u64 = 256 << 10;

/// Since the last frame line, with [`runs_on`]: passes, the calls in them,
/// passes of one call.
static RUN_COUNTS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// A host list's clamp of runs ([`batched`]): the batch open and the slot
/// buffer its calls read their clamped records from.
#[derive(Default)]
pub(crate) struct Batching {
    /// The slot buffer (the card's memory), kept across resets, and its bytes.
    slots: Option<(ID3D12Resource, u64)>,
    /// The bytes the next recording's slot buffer is made with, if more.
    want: u64,
    /// In this recording: the slot bytes taken; whether a pass left the
    /// slot buffer in the indirect-argument state; whether it was short.
    used: u64,
    argument: bool,
    short: bool,
    open: Option<Open>,
    /// Since the recording began: passes, the calls in them, passes of one
    /// call ([`RUN_COUNTS`]).
    pub(crate) counts: [u64; 3],
}

/// A batch whose pass is recorded and whose calls are still coming.
struct Open {
    /// The work list, in the list's upload memory: the count of records,
    /// then one a call. The pass reads it when the list runs, by when the
    /// list is closed and every call of the batch is in it.
    work: usize,
    capacity: u32,
    calls: u32,
    /// The argument copies the pass reads, by place (t1 on).
    sources: Vec<ID3D12Resource>,
    /// The slot buffer and the offset of the batch's first slot.
    slots: ID3D12Resource,
    base: u64,
}

impl Batching {
    /// The batch open ends ([`HostList::batch_end`]): the run is over.
    pub(crate) fn end(&mut self) {
        if let Some(open) = self.open.take() {
            self.ended(&open);
        }
    }

    fn ended(&mut self, open: &Open) {
        let counts = [1, u64::from(open.calls), u64::from(open.calls == 1)];
        for (i, n) in counts.into_iter().enumerate() {
            self.counts[i] += n;
            RUN_COUNTS[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The list is reset, every execution of it done: what the recording
    /// held goes; the slot buffer stays (one found short is let go, and a
    /// larger one made at the next recording's first batch).
    pub(crate) fn reset(&mut self) {
        let bytes = self.slots.as_ref().map_or(0, |s| s.1);
        let want = if self.short { bytes.max(SLOT_BYTES) * 2 } else { self.want };
        let slots = if self.short { None } else { self.slots.take() };
        *self = Batching { slots, want, ..Default::default() };
    }

    /// The slot buffer, made (or made larger) before the recording's first
    /// batch only: once a pass has named it, it stays.
    fn slot_buffer(&mut self) -> Option<(ID3D12Resource, u64)> {
        let want = self.want.max(SLOT_BYTES);
        if self.used == 0 && !self.argument && self.slots.as_ref().is_none_or(|s| s.1 < want) {
            let buffer = host()?.uav_buffer(want)?;
            crate::host::name(&buffer, || format!("indirect run slots {want:#x}"));
            self.slots = Some((buffer, want));
        }
        self.slots.clone()
    }
}

/// A call of a run, with [`runs_on`]: its record (`kind`, `stride` bytes
/// at `src`) clamped by its batch's pass into a slot of its own: the slot
/// buffer, the slot's offset, and whether a pass was recorded for it (it
/// replaced the list's compute root signature and pipeline). `None`: not
/// batched (a kind or layout the pass does not take, or nothing to make
/// one with); the caller clamps it alone, as before.
///
/// A run's records are not known when its first call is recorded, but the
/// pass need not know them then: it is recorded there, one group, and
/// reads the records' places and their count from a work list in the
/// list's upload memory, which each later call of the batch fills on the
/// CPU. The list runs only once it is closed, so the pass finds every
/// call of its batch. No cut is made and nothing is guessed. A batch goes
/// on only across the title's commands known to leave buffer memory as it
/// is (`crate::list`'s `KEEP_RUNS`: state, barriers, query marks, draws
/// whose shaders cannot store, the run's own calls); any other command
/// ends it ([`HostList::batch_end`]), as does the list's end. It also ends
/// when it is full (the run's next is twice as large), and at a call whose
/// argument copy it does not read (the next reads that one and those it
/// read, four at most). Its calls read their slots with nothing between
/// them: the waits are the pass's alone.
///
/// The budget keeps its meaning: the pass charges the batch's records in
/// call order against the list's budget (each execution its own), records
/// past it run empty, saturating; a running sum over the records, capped
/// just past the budget, takes the place of an atomic a record, which made
/// a pass over many records slow (benchmark x: 0.7 microseconds a record).
pub(crate) fn batched(
    list: &HostList,
    (src, offset): (&ID3D12Resource, u64),
    (kind, stride): (u32, u32),
) -> Option<(ID3D12Resource, u64, bool)> {
    use windows::core::Interface;
    if !matches!(kind, 0 | 1 | 10) || u64::from(stride) > SLOT {
        return None;
    }
    validate_layout((kind, stride, 1), offset, None)?;
    let at = u32::try_from(offset).ok().filter(|at| at.checked_add(stride).is_some())?;
    // SAFETY: a live buffer.
    if !unsafe { src.GetGPUVirtualAddress() }.is_multiple_of(4) {
        return None;
    }
    let mut b = list.batching();
    let raw = src.as_raw();
    let place = |open: &Open| open.sources.iter().position(|s| s.as_raw() == raw);
    let taken = b.open.as_ref().filter(|o| o.calls < o.capacity).and_then(place);
    let (which, recorded) = match taken {
        Some(which) => (which, false),
        None => {
            // The run's next batch, or its first.
            let (capacity, mut sources) = match b.open.take() {
                Some(open) => {
                    b.ended(&open);
                    let full = open.calls == open.capacity;
                    (if full { (open.capacity * 2).min(MOST) } else { open.capacity }, open.sources)
                }
                None => (FIRST, Vec::new()),
            };
            if !sources.iter().any(|s| s.as_raw() == raw) {
                if sources.len() == SOURCES {
                    sources.remove(0);
                }
                sources.push(src.clone());
            }
            let which = sources.iter().position(|s| s.as_raw() == raw)?;
            begin(list, &mut b, capacity, sources)?;
            (which, true)
        }
    };
    let open = b.open.as_mut()?;
    let call = open.calls;
    // SAFETY: the work list holds `capacity` records after its head; the
    // list's upload memory, alive until it is reset.
    unsafe {
        let entry = (open.work as *mut u8).add((HEAD + u64::from(call) * ENTRY) as usize).cast::<u32>();
        for (k, word) in [which as u32 | (kind << 8), at, stride, 0].into_iter().enumerate() {
            entry.add(k).write_volatile(word);
        }
        (open.work as *mut u32).write_volatile(call + 1);
    }
    open.calls += 1;
    let made = (open.slots.clone(), open.base + u64::from(call) * SLOT, recorded);
    drop(b);
    // A call of the census of runs (`crate::runs`), as a call clamped alone.
    #[cfg(feature = "diagnostics")]
    list.runs(|runs| runs.call(raw as usize, false));
    Some(made)
}

/// `crate::draw`'s choice for a one-record call without a count buffer: with
/// [`runs_on`] (the default) and enhanced barriers (the module's comment),
/// its run's batch ([`batched`]); without (or if none can be made) `None`,
/// and the caller clamps it alone with [`bound`], recording what it
/// recorded before the clamp of runs.
pub(crate) fn in_run(list: &HostList, src: (&ID3D12Resource, u64), spec: (u32, u32)) -> Option<(ID3D12Resource, u64, bool)> {
    if !runs_on() || list.enhanced().is_none() {
        return None;
    }
    batched(list, src, spec)
}

/// A batch of `capacity` calls begins: its slots taken, its work list made,
/// its pass recorded into `list`. `None` if one cannot be made.
fn begin(list: &HostList, b: &mut Batching, capacity: u32, sources: Vec<ID3D12Resource>) -> Option<()> {
    let c = batch_clamp()?;
    let budget = list.budget()?;
    let bytes = u64::from(capacity) * SLOT;
    let (slots, size) = b.slot_buffer()?;
    if b.used + bytes > size {
        b.short = true;
        return None;
    }
    let (work_gpu, work) = list.scratch(HEAD + u64::from(capacity) * ENTRY)?;
    // SAFETY: the list's upload memory: no records yet.
    unsafe { work.cast::<u32>().write_volatile(0) };
    // SAFETY: live buffers.
    let (views, dst) = unsafe {
        let views: [u64; SOURCES] = std::array::from_fn(|i| sources[i.min(sources.len() - 1)].GetGPUVirtualAddress());
        (views, slots.GetGPUVirtualAddress() + b.used)
    };
    let from = if b.argument { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT } else { D3D12_RESOURCE_STATE_COMMON };
    let reads = Reads::new(&sources, list.enhanced());
    if !stamped(list, || record_batch(c, &list.list, (&slots, from), budget, dst, work_gpu, (views, &reads))) {
        return None;
    }
    let base = b.used;
    (b.used, b.argument) = (b.used + bytes, true);
    b.open = Some(Open { work: work as usize, capacity, calls: 0, sources, slots, base });
    Some(())
}

/// Records a batch's pass: a wait for every earlier shader write (the
/// dispatch that wrote the records), the slot buffer from `from`
/// to unordered access, and the barriers before the pass's reads of the
/// title's copies `reads` says; the pass, one group, over the work list at
/// `work` into the slots from `dst` on, charging `budget`; a wait, the
/// slot buffer to indirect argument, and the barriers after the reads.
/// `VICTORY_BARRIER_TEST=indirect` leaves the waits out, as for a call's
/// own clamp.
///
/// The argument copies at `views`, by place, are read through shader
/// resource views (t1 to t4), as a call's own clamp reads its copy; a
/// place past the last copy names the last again.
fn record_batch(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (slots, from): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    dst: u64,
    work: u64,
    (views, reads): ([u64; SOURCES], &Reads<'_>),
) -> bool {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [uav_barrier(), transition(slots, from, UNORDERED)];
    let second = [uav_barrier(), transition(slots, UNORDERED, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)];
    let skip = usize::from(!waits);
    // SAFETY: an open list; our root signature and pipeline; live buffers.
    unsafe {
        l.ResourceBarrier(&first[skip..]);
        if waits {
            reads.before();
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootShaderResourceView(0, work);
        l.SetComputeRootUnorderedAccessView(1, dst);
        l.SetComputeRootUnorderedAccessView(2, budget);
        for (i, view) in views.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(3 + i as u32, view);
        }
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&second[skip..]);
        if waits {
            reads.after();
        }
    }
    true
}

/// Where a host list ends: its batch ends, and its slot buffer goes back to
/// the common state, where its next execution starts from.
pub(crate) fn batch_close(l: &ID3D12GraphicsCommandList, b: &mut Batching) {
    b.end();
    if std::mem::take(&mut b.argument)
        && let Some((slots, _)) = &b.slots
    {
        let (argument, common) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
        // SAFETY: an open list; the slot buffer, in indirect argument since
        // the last pass.
        unsafe { l.ResourceBarrier(&[transition(slots, argument, common)]) };
    }
}

fn batch_clamp() -> Option<&'static Clamp> {
    static BATCH_CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    BATCH_CLAMP
        .get_or_init(|| {
            let made = host().ok_or_else(windows::core::Error::empty).and_then(|h| make_batch_for(&h.device));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no clamp of runs: their calls are clamped one by one");
                    None
                }
            }
        })
        .as_ref()
}

/// The frame line's clamp of runs, with [`runs_on`] (the default).
pub(crate) fn runs_line(frames: f64) {
    if !runs_on() {
        return;
    }
    let [passes, calls, ones] =
        RUN_COUNTS.each_ref().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames.max(1.0));
    tracing::info!(
        "clamped runs per frame (the clamp of runs; VICTORY_CLAMP_RUNS=0 turns it off): {passes:.1} batches (a clamp pass each; a long run takes several), \
         {calls:.0} calls in them, {ones:.1} batches of one call; cuts made 0, wasted 0 (none: a batch's pass is recorded \
         at its first call and reads its calls when the list runs)"
    );
}

/// The clamp of runs' pass ([`batched`]): one group of 256 threads, four
/// records each at most ([`MOST`]), in call order.
const BATCH: &str = r#"
#define RS "SRV(t0), UAV(u0), UAV(u1), SRV(t1, flags = DATA_VOLATILE), SRV(t2, flags = DATA_VOLATILE), SRV(t3, flags = DATA_VOLATILE), SRV(t4, flags = DATA_VOLATILE)"
// The count of records at 0; from 16 on, a record's argument copy (0 to
// 3) and kind (from bit 8), its offset there and its stride. Record i goes
// to slot i, 64 bytes apart.
ByteAddressBuffer work : register(t0);
RWByteAddressBuffer dst : register(u0);
// As the clamp's: the work so far, then the mesh records' notes.
RWByteAddressBuffer budget : register(u1);
// The argument copies: read here, never written, through shader resource
// views of volatile data, as the clamp reads its own.
ByteAddressBuffer src0 : register(t1);
ByteAddressBuffer src1 : register(t2);
ByteAddressBuffer src2 : register(t3);
ByteAddressBuffer src3 : register(t4);
static const uint BUDGET = 1u << 27;
// Sums are capped just past the budget: adding stays exact up to it and
// never wraps.
static const uint CAP = BUDGET + 1;
static const uint THREADS = 256;
static const uint EACH = 4;
static const uint SLOT = 64;
groupshared uint sums[THREADS];
groupshared uint start;
groupshared uint largest_asked;
groupshared uint all_asked;

uint word(uint which, uint at) {
    if (which == 0) return src0.Load(at);
    if (which == 1) return src1.Load(at);
    if (which == 2) return src2.Load(at);
    return src3.Load(at);
}

uint capped(uint a, uint b) {
    return min(a + b, CAP);
}

struct Record {
    uint which;
    uint kind;
    uint at;
    uint stride;
    uint cost;
    uint2 counts;
    uint3 groups;
    uint asked;
};

// As the clamp bounds one record (not ahead of its draw).
Record measure(uint i) {
    uint4 e = work.Load4(16 + i * 16);
    Record r;
    r.which = e.x & 0xffu;
    r.kind = e.x >> 8;
    r.at = e.y;
    r.stride = e.z;
    r.counts = uint2(0, 0);
    r.groups = uint3(0, 0, 0);
    r.asked = 0;
    if (r.kind <= 1) {
        uint2 asked = uint2(word(r.which, r.at), word(r.which, r.at + 4));
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        r.cost = count * instances;
        r.counts = uint2(count, instances);
    } else {
        uint3 g = uint3(word(r.which, r.at), word(r.which, r.at + 4), word(r.which, r.at + 8));
        r.asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        r.cost = g.x * g.y * g.z * 128u;
        r.groups = g;
    }
    return r;
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint t : SV_GroupIndex) {
    uint n = min(work.Load(0), THREADS * EACH);
    if (t == 0) {
        uint was;
        budget.InterlockedOr(0, 0, was);
        start = was;
        largest_asked = 0;
        all_asked = 0;
    }
    uint first = t * EACH;
    uint last = min(first + EACH, n);
    uint mine = 0;
    uint largest = 0;
    uint all = 0;
    for (uint i = first; i < last; i++) {
        Record r = measure(i);
        mine = capped(mine, r.cost);
        if (r.kind > 1) {
            largest = max(largest, r.asked);
            all += min(r.asked, 1u << 24);
        }
    }
    sums[t] = mine;
    GroupMemoryBarrierWithGroupSync();
    // Each thread's sum and those before it (in call order).
    for (uint d = 1; d < THREADS; d <<= 1) {
        uint v = sums[t];
        if (t >= d) {
            v = capped(v, sums[t - d]);
        }
        GroupMemoryBarrierWithGroupSync();
        sums[t] = v;
        GroupMemoryBarrierWithGroupSync();
    }
    InterlockedMax(largest_asked, largest);
    InterlockedAdd(all_asked, all);
    // A record past the budget, counting it and all before it, runs empty.
    uint before = capped(start, t == 0 ? 0 : sums[t - 1]);
    for (uint j = first; j < last; j++) {
        Record q = measure(j);
        uint after = capped(before, q.cost);
        bool over = q.cost > 0 && after > BUDGET;
        uint to = j * SLOT;
        if (q.kind <= 1) {
            dst.Store2(to, over ? uint2(0, 0) : q.counts);
            for (uint w = 8; w < q.stride; w += 4) {
                dst.Store(to + w, word(q.which, q.at + w));
            }
        } else {
            dst.Store3(to, over ? uint3(0, 0, 0) : q.groups);
            for (uint m = 12; m < q.stride; m += 4) {
                dst.Store(to + m, word(q.which, q.at + m));
            }
        }
        before = after;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0 && n > 0) {
        uint was;
        budget.InterlockedMax(0, min(capped(start, sums[THREADS - 1]), BUDGET), was);
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
}
"#;

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: D3D12 test harness setup on a device of our own.
            unsafe {
                // The software adapter, by the one rule's software request; none where
                // DXGI does not say it is software.
                let device = crate::host::test_warp_device()?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                // Today's two parts, whatever `VICTORY_SPLIT_PARTS` says.
                let clamp = make_for_parts(&device, SPLIT).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::hash::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(PIECES.div_ceil(64), 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if crate::host::test_host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place,
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`, and
    /// (d) the count the calls read (`count_at`, after the records in
    ///     `dst_args`, as `crate::draw` reads it) is `min(max, count)`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let counted_at = count_at(STRIDE, MAX);
        let dst_buf = warp.default_uav(counted_at + 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(&sig, MAX, &dst_buf, 0, &dst_buf, counted_at);
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes
        // and the count, then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list
                .CopyBufferRegion(&readback_buf, u64::from(STRIDE * 6), &dst_buf, counted_at, 4);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(&sig, MAX, &dst_buf, 0, &dst_buf, counted_at);
        }
        done_on(&warp.list, &dst_buf);
        // Call #2's count as it read it.
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            let (common, source) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            warp.list.ResourceBarrier(&[transition(&dst_buf, common, source)]);
            warp.list
                .CopyBufferRegion(&readback_buf, u64::from(STRIDE * 6) + 4, &dst_buf, counted_at, 4);
            warp.list.ResourceBarrier(&[transition(&dst_buf, source, common)]);
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 6 + 2,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
        // The counts the calls read: the title's 3 and 2, under `MAX`.
        assert_eq!(clamped_words[36..38], [3, 2], "the counts after the records");
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// One dispatch in parts: groups past what the host runs in one
    /// dimension go into the second record; what nothing wrote stays
    /// clamped, or empty.
    #[test]
    fn warp_large_dispatch_is_made_in_parts() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let cases: [([u32; 3], [u32; 6]); 6] = [
            ([115_200, 1, 1], [65_535, 1, 1, 49_665, 1, 1]),
            ([65_535, 1, 1], [65_535, 1, 1, 0, 1, 1]),
            ([65_536, 1, 1], [65_535, 1, 1, 1, 1, 1]),
            ([5, 3, 2], [5, 3, 2, 0, 3, 2]),
            ([0, 1, 1], [0, 1, 1, 0, 1, 1]),
            // Past both parts: as before, what one dispatch runs.
            ([200_000, 1, 1], [65_535, 1, 1, 0, 1, 1]),
        ];
        for (asked, parts) in cases {
            let (src_buf, src_ptr, _) = warp.shared_buffer(16);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(64);
            let dst_buf = warp.default_uav(64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, 64);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (2, 12, 1),
                IN_PARTS,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list and live buffers.
            unsafe { warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 24) };
            warp.submit_and_wait();
            // SAFETY: the mapping holds 24 bytes the copy wrote.
            let made: [u32; 6] = unsafe { readback_ptr.cast::<[u32; 6]>().read_volatile() };
            assert_eq!(made, parts, "{asked:?}");
        }
    }

    /// A program of ours with a group base, as a translated one takes it
    /// (`victory_shader::emit::USER_GROUP_BASE`): one root constant, the
    /// number of its first group, and a count for each group that ran.
    const GROUPS_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Base : register(b0) { uint base; };
RWByteAddressBuffer ran : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    uint prev;
    ran.InterlockedAdd((base + g.x) * 4u, 1u, prev);
}
"#;

    /// What one dispatch in parts did ([`ran_in_parts`]): how many times
    /// each group ran, and the records the clamp made, a part each.
    struct Ran {
        counts: Vec<u32>,
        records: Vec<[u32; 3]>,
    }

    impl Ran {
        /// (groups below `first` run once, groups run otherwise than that):
        /// each group below `first` once, none at or past it.
        fn against(&self, first: u32) -> (usize, usize) {
            let once = self.counts[..first as usize].iter().filter(|&&n| n == 1).count();
            let wrong = self.counts.iter().enumerate().filter(|&(i, &n)| n != u32::from(i < first as usize)).count();
            (once, wrong)
        }
    }

    /// One one-dimensional dispatch of `asked` groups, clamped in `parts`
    /// parts by a clamp made for them, and executed part by part as
    /// `crate::draw` does ([`execute_parts`]) by [`GROUPS_CS`]: what ran,
    /// for `slots` groups.
    fn ran_in_parts(warp: &mut WarpHarness, parts: u32, asked: [u32; 3], slots: usize) -> Ran {
        let clamp = make_for_parts(&warp.device, parts).expect("the clamp must compile on WARP");
        let program = compile_cs(&warp.device, GROUPS_CS).expect("the groups program must compile on WARP");
        let signature = warp.signature(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32, PART_BYTES as u32);
        let bytes = split_bytes(parts);
        let (src_buf, src_ptr, _) = warp.shared_buffer(16);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_ran_buf, ran_ptr, ran_gpu) = warp.shared_buffer(slots as u64 * 4);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(bytes);
        let dst_buf = warp.default_uav(bytes);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(ran_ptr, 0, slots * 4);
            std::ptr::write_bytes(readback_ptr, 0xEE, bytes as usize);
        }
        assert!(record_clamp_as(&clamp, &warp.list, &dst_buf, budget_gpu, (&src_buf, 0), None, (2, 12, 1), IN_PARTS));
        // SAFETY: an open list; the program's own root signature and
        // pipeline, its buffer bound; the scratch holds `parts` records in
        // the indirect-argument state.
        unsafe {
            warp.list.SetComputeRootSignature(&program.root);
            warp.list.SetPipelineState(&program.pso);
            warp.list.SetComputeRootUnorderedAccessView(1, ran_gpu);
            execute_parts(&warp.list, &signature, &dst_buf, (0, 0), parts);
        }
        done_on(&warp.list, &dst_buf);
        // SAFETY: an open list; the scratch is copied out for the check.
        unsafe {
            let (common, source) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            warp.list.ResourceBarrier(&[transition(&dst_buf, common, source)]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, bytes);
            warp.list.ResourceBarrier(&[transition(&dst_buf, source, common)]);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; the mappings hold `slots` counts and
        // `parts` records.
        unsafe {
            Ran {
                counts: std::slice::from_raw_parts(ran_ptr.cast::<u32>().cast_const(), slots).to_vec(),
                records: std::slice::from_raw_parts(readback_ptr.cast::<[u32; 3]>().cast_const(), parts as usize).to_vec(),
            }
        }
    }

    /// The groups of a large one-dimensional dispatch all run, once each,
    /// with 4 parts and with 8 (`VICTORY_SPLIT_PARTS`): 259,200 is a
    /// screen-sized pass at 3840x2160 (8 groups for each 16x16 tile),
    /// 131,071 one more than two parts hold.
    #[test]
    fn warp_split_runs_every_group_asked() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let mut seen = Vec::new();
        for parts in [4, 8] {
            for asked in [259_200u32, 131_071] {
                let ran = ran_in_parts(&mut warp, parts, [asked, 1, 1], asked as usize + 1024);
                let (once, wrong) = ran.against(asked);
                seen.push((parts, asked, once, wrong));
            }
        }
        let want: Vec<_> = seen.iter().map(|&(parts, asked, ..)| (parts, asked, asked as usize, 0)).collect();
        assert_eq!(seen, want, "(parts, asked, groups run once, groups run otherwise)");
    }

    /// At the default, two parts, as before the setting: the same two
    /// dispatches are cut to what one dispatch runs, 131,070 runs whole.
    #[test]
    fn warp_split_default_cuts_as_before() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        for (asked, first) in [(259_200u32, 65_535u32), (131_071, 65_535), (131_070, 131_070)] {
            let ran = ran_in_parts(&mut warp, SPLIT, [asked, 1, 1], 262_144);
            assert_eq!(ran.against(first), (first as usize, 0), "{asked} asked");
            assert_eq!(ran.records.len(), 2);
        }
    }

    /// One group more than the parts hold: cut to what one dispatch runs,
    /// as before; exactly what they hold runs whole.
    #[test]
    fn warp_split_past_the_parts_is_cut_as_before() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        for parts in [4u32, 8] {
            let most = parts * PART;
            let whole = ran_in_parts(&mut warp, parts, [most, 1, 1], most as usize + 1024);
            assert_eq!(whole.against(most), (most as usize, 0), "{most} asked of {parts} parts");
            let past = ran_in_parts(&mut warp, parts, [most + 1, 1, 1], most as usize + 1024);
            assert_eq!(past.against(PART), (PART as usize, 0), "{} asked of {parts} parts", most + 1);
            let mut want = vec![[0, 1, 1]; parts as usize];
            want[0] = [PART, 1, 1];
            assert_eq!(past.records, want, "{} asked of {parts} parts", most + 1);
        }
    }

    /// A part with no groups left is a record of none (x 0, y and z as
    /// asked), executed as an empty dispatch: no group runs for it, as
    /// today's second part of a dispatch of 65,535 or fewer.
    #[test]
    fn warp_split_parts_with_no_groups_are_empty() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        for parts in [SPLIT, 8] {
            for (asked, first) in [([65_535, 1, 1], [65_535, 1, 1]), ([5, 3, 2], [5, 3, 2]), ([0, 1, 1], [0, 1, 1])] {
                let ran = ran_in_parts(&mut warp, parts, asked, 262_144);
                let mut want = vec![[0, first[1], first[2]]; parts as usize];
                want[0] = first;
                assert_eq!(ran.records, want, "{asked:?} in {parts} parts");
                if asked[1] == 1 {
                    assert_eq!(ran.against(asked[0]), (asked[0] as usize, 0), "{asked:?} in {parts} parts");
                }
            }
        }
    }

    /// `VICTORY_SPLIT_PARTS`: unset is two parts; 2 to 16 are taken;
    /// anything else is two parts and not taken (one warning, [`split`]).
    #[test]
    fn split_parts_setting_takes_two_to_sixteen() {
        assert_eq!(split_from(None), (2, true));
        for (text, parts) in [("2", 2), ("4", 4), (" 8 ", 8), ("16", 16)] {
            assert_eq!(split_from(Some(text)), (parts, true), "{text:?}");
        }
        for text in ["0", "1", "17", "-4", "four", "", "4.0", "99999999999"] {
            assert_eq!(split_from(Some(text)), (2, false), "{text:?}");
        }
        assert_eq!(split_bytes(SPLIT), 24, "today's scratch: two records");
        assert_eq!(split_bytes(SPLIT_MOST) / PART_BYTES, 16);
    }

    /// FNV-1a, 64 bits: a fingerprint of a text.
    fn fnv(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    }

    /// The clamp's text at the default is the text before the setting,
    /// byte for byte (its length and fingerprint, taken before the change);
    /// another part count changes the one line and nothing else.
    #[test]
    fn split_parts_default_is_todays_clamp_text() {
        assert_eq!((SOURCE.len(), fnv(SOURCE)), (TODAY_LEN, TODAY_FNV), "the clamp's text changed");
        assert_eq!(source(SPLIT), SOURCE);
        assert_eq!(SOURCE.matches(SPLIT_LINE).count(), 1);
        for parts in SPLIT + 1..=SPLIT_MOST {
            let text = source(parts);
            let line = format!("static const uint SPLIT = {parts};");
            assert_eq!(text.matches(&line).count(), 1, "{parts}");
            assert_eq!(text.replacen(&line, SPLIT_LINE, 1), SOURCE, "{parts}");
        }
    }

    /// The length and fingerprint ([`fnv`]) of [`SOURCE`]: the text on
    /// `world` at 3b2ccf7be, before the part count was a setting, with the
    /// one comment of the mesh records' check reworded since (no code of
    /// the clamp changed).
    const TODAY_LEN: usize = 5022;
    const TODAY_FNV: u64 = 0x66f8_f707_8a79_a562;

    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged; the unwritten slots, which the test fills
    /// with its own word over the most a dispatch runs, come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        // The test's own word for every slot nothing wrote: over 65,535, so
        // out of range for a mesh record.
        const UNWRITTEN: u32 = 70_000;
        let total = (STRIDE * RECORDS) as usize;
        let mut src = UNWRITTEN.to_le_bytes().repeat(total / 4);
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                if ahead { AHEAD } else { 0 },
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], UNWRITTEN, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, UNWRITTEN);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let _lane = crate::inplace::warp_lane();
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = crate::host::test_host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// A clamp group whose recording was refused before it recorded
    /// anything (as for a source address it cannot use) is no pair; one
    /// that recorded is (`VICTORY_GPU_STAMPS=1`; Sol's review of the frame
    /// timers: the refused one was counted as a group).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_refused_clamp_group_is_no_pair() {
        use crate::stamps::{Kind, Sink};
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        let list = host.new_list().unwrap();
        assert!(!stamped(&list, || false));
        assert!(stamped(&list, || true));
        // Through the production validation (Sol's third read of the frame
        // timers): a source offset off its four-byte line is refused before
        // anything is stamped; an aligned one is clamped, between a pair.
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([3, 1, 1, 0, 1, 1]) };
        let slots = list.stamp_slots();
        assert!(bound(&list, (&src, 2), None, (10, 12, 2)).is_none(), "an unaligned source clamped");
        assert_eq!(list.stamp_slots(), slots, "a refused clamp stamped");
        let scratch = bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        done(&list, &scratch);
        assert_eq!(list.stamp_slots(), slots + 2, "an accepted clamp is a pair");
        list.close();
        host.execute(&[&list]);
        let spans = sink.spans();
        assert_eq!(spans.iter().filter(|s| s.0 == Kind::Clamp).count(), 2, "{spans:?}");
        assert_eq!(sink.flaws().lost, 0);
    }
}

/// The clamp of runs ([`batched`]) on host lists of the runtime's, through
/// the functions the draw path calls (`in_run`, `bound`, `done`,
/// `HostList::batch_end` where the title's dispatch, copy or storing draw
/// is recorded), on the software adapter.
#[cfg(test)]
mod run_tests {
    use super::*;
    use crate::host::{HostList, test_host};
    #[cfg(feature = "diagnostics")]
    use crate::object::This;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    /// The switch, on this thread, until dropped.
    struct Switch(Option<bool>);
    impl Switch {
        fn set(on: bool) -> Switch {
            Switch(RUNS_FORCED.with(|f| f.replace(Some(on))))
        }
    }
    impl Drop for Switch {
        fn drop(&mut self) {
            RUNS_FORCED.with(|f| f.set(self.0));
        }
    }

    /// A buffer in the PC's memory with `words` from offset 0 on.
    fn filled(words: &[(u64, &[u32])]) -> ID3D12Resource {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (buffer, data) = host.shared_buffer(1 << 16).unwrap();
        for (at, record) in words {
            // SAFETY: inside the mapping.
            unsafe { std::ptr::copy_nonoverlapping(record.as_ptr(), data.add(*at as usize).cast::<u32>(), record.len()) };
        }
        buffer
    }

    /// Copies `bytes` of `from` at `at` (in the indirect-argument state)
    /// into `out` at `to`.
    fn copy_out(list: &HostList, from: &ID3D12Resource, at: u64, out: &ID3D12Resource, to: u64, bytes: u64) {
        let (argument, source) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        // SAFETY: an open list; live buffers.
        unsafe {
            list.list.ResourceBarrier(&[transition(from, argument, source)]);
            list.list.CopyBufferRegion(out, to, from, at, bytes);
            list.list.ResourceBarrier(&[transition(from, source, argument)]);
        }
    }

    /// A call of a run: its record clamped in its batch, or alone (`bound`)
    /// when not batched, copied into `out` at `to`: whether batched.
    fn call(list: &HostList, src: (&ID3D12Resource, u64), spec: (u32, u32), out: &ID3D12Resource, to: u64) -> bool {
        match in_run(list, src, spec) {
            Some((slots, at, _)) => {
                copy_out(list, &slots, at, out, to, u64::from(spec.1));
                true
            }
            None => {
                let scratch = bound(list, src, None, (spec.0, spec.1, 1)).expect("the clamp");
                copy_out(list, &scratch, 0, out, to, u64::from(spec.1));
                done(list, &scratch);
                false
            }
        }
    }

    fn words(data: *mut u8, at: usize, n: usize) -> Vec<u32> {
        // SAFETY: inside the mapping; read once the card is done.
        unsafe { std::slice::from_raw_parts(data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// A compute pass of ours, made from `source` (its root signature
    /// inside), on the host device.
    fn compute(source: &str) -> Clamp {
        let host = test_host().unwrap();
        let code = crate::host::compile(source, s!("main"), s!("cs_5_1")).expect("our pass compiles");
        // SAFETY: plain object creation from our own code.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                    ..Default::default()
                })
                .unwrap();
            Clamp { root, pso }
        }
    }

    /// Writes record `i` (four words) at `i * 32` of the view at u0 for each
    /// of `n` threads: `base + i`, 1, 0, 0. A title dispatch's work.
    const WRITER: &str = r#"
#define RS "UAV(u0), RootConstants(num32BitConstants=2, b0)"
RWByteAddressBuffer records : register(u0);
cbuffer What : register(b0) { uint base; uint n; };
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint i : SV_DispatchThreadID) {
    if (i < n) {
        records.Store4(i * 32, uint4(base + i, 1, 0, 0));
    }
}
"#;

    /// Records a title-like dispatch of [`WRITER`] into `list`: records
    /// `first` on, `n` of them, `base + i` each, into `buffer` from `at`.
    fn write_records(list: &HostList, writer: &Clamp, buffer: &ID3D12Resource, at: u64, (base, n): (u32, u32)) {
        // SAFETY: an open list; our pass; a live buffer.
        unsafe {
            list.list.SetComputeRootSignature(&writer.root);
            list.list.SetPipelineState(&writer.pso);
            list.list.SetComputeRootUnorderedAccessView(0, buffer.GetGPUVirtualAddress() + at);
            list.list.SetComputeRoot32BitConstants(1, 2, [base, n].as_ptr().cast(), 0);
            list.list.Dispatch(n.div_ceil(64), 1, 1);
        }
        // As `State::dispatch` does once the title's dispatch is recorded.
        list.batch_end();
    }

    /// A run of six draws whose records a dispatch just wrote, each drawing
    /// as many points as its record asks, counted per draw by its vertex
    /// program: each call reads its own record, as the dispatch left it;
    /// the run takes two passes (four calls, then the next batch of eight).
    #[test]
    fn a_run_s_draws_each_draw_their_own_record_as_the_dispatch_wrote_it() {
        let _on = Switch::set(true);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        const COUNTER: &str = r#"
#define RS "UAV(u0), RootConstants(num32BitConstants=1, b0)"
RWByteAddressBuffer hits : register(u0);
cbuffer Call : register(b0) { uint call; };
[RootSignature(RS)]
float4 main(uint v : SV_VertexID) : SV_Position {
    uint was;
    hits.InterlockedAdd(call * 4, 1, was);
    return float4(0, 0, 0, 1);
}
"#;
        let vs = crate::host::compile(COUNTER, s!("main"), s!("vs_5_1")).expect("the counter compiles");
        // SAFETY: plain object creation from our own code and descriptions.
        let (root, pso, signature) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: true.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                })
                .unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() };
            let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature).unwrap();
            (root, pso, signature.unwrap())
        };
        // Before the dispatch, records that ask for nothing.
        let src = filled(&[]);
        let (hits, counted) = host.shared_buffer(256).unwrap();
        // SAFETY: inside the mapping.
        unsafe { std::ptr::write_bytes(counted, 0, 256) };
        let writer = compute(WRITER);
        let list = host.new_list().unwrap();
        write_records(&list, &writer, &src, 0, (11, 6));
        for i in 0..6u32 {
            let (slots, at, recorded) = in_run(&list, (&src, u64::from(i) * 32), (0, 16)).expect("batched");
            // SAFETY: an open list; our pipeline; the slot as the batch left it.
            unsafe {
                if recorded || i == 0 {
                    list.list.SetGraphicsRootSignature(&root);
                    list.list.SetPipelineState(&pso);
                    list.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                    list.list.SetGraphicsRootUnorderedAccessView(0, hits.GetGPUVirtualAddress());
                }
                list.list.SetGraphicsRoot32BitConstant(1, i, 0);
                list.list.ExecuteIndirect(&signature, 1, &slots, at, None, 0);
            }
        }
        list.close();
        assert_eq!(list.batching().counts, [2, 6, 0], "passes, calls in them, passes of one call");
        host.execute(&[&list]);
        assert_eq!(words(counted, 0, 6), [11, 12, 13, 14, 15, 16], "each draw drew its own record, as the dispatch wrote it");
        list.reset();
    }

    /// A run whose records ask for more than the budget partway: in call
    /// order the first two are kept (the second takes it to the budget
    /// exactly), the two after run empty with their other words kept, the
    /// same bytes as the clamp makes of them one by one; and each execution
    /// of the list is charged from a budget of its own.
    #[test]
    fn a_run_past_the_budget_empties_its_later_calls_in_order_as_one_by_one() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let most = [u32::MAX, u32::MAX, 0, 0, 0];
        let src = filled(&[(0, &most), (32, &most), (64, &most), (96, &[3, 1, 7, 8, 9])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        // Each call's record: five words, 32 bytes apart from `at` on.
        let slots5 = |at: usize| (0..4).map(|i| words(seen, at + i * 32, 5)).collect::<Vec<_>>();
        let record = |list: &HostList, to: u64| -> Vec<bool> {
            (0..4).map(|i| call(list, (&src, i * 32), (1, 20), &out, to + i * 32)).collect()
        };
        let alone = {
            let _off = Switch::set(false);
            let list = host.new_list().unwrap();
            assert_eq!(record(&list, 0), [false; 4]);
            list.close();
            host.execute(&[&list]);
            list.reset();
            slots5(0)
        };
        let _on = Switch::set(true);
        let list = host.new_list().unwrap();
        assert_eq!(record(&list, 1024), [true; 4], "the run is batched");
        list.close();
        for run in 0..2 {
            // SAFETY: inside the mapping, between executions.
            unsafe { std::ptr::write_bytes(seen.add(1024), 0xEE, 128) };
            host.execute(&[&list]);
            let batched = slots5(1024);
            assert_eq!(batched, alone, "execution {run}: the same records as clamped one by one");
            let kept: Vec<&[u32]> = batched.iter().map(|r| &r[..2]).collect();
            assert_eq!(kept, [[1 << 24, 4], [1 << 24, 4], [0, 0], [0, 0]], "execution {run}");
            assert_eq!(batched[3][2..], [7, 8, 9], "the emptied record's other words are kept");
        }
        assert_eq!(list.batching().counts, [1, 4, 0]);
        list.reset();
    }

    /// A call clamped alone in the middle of a run (one with a count
    /// buffer, say) is charged in its place: the batch before it ends
    /// there, so the call after it is charged after it. Two records of half
    /// the budget, the second clamped alone, take it to the budget
    /// exactly; the small one after them runs empty, as one by one.
    #[test]
    fn a_call_clamped_alone_in_a_run_is_charged_in_its_place() {
        let _on = Switch::set(true);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let most = [u32::MAX, u32::MAX, 0, 0, 0];
        let src = filled(&[(0, &most), (32, &most), (64, &[3, 1, 7, 8, 9]), (128, &[1])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        assert!(call(&list, (&src, 0), (1, 20), &out, 0));
        let alone = bound(&list, (&src, 32), Some((&src, 128)), (1, 20, 1)).expect("the clamp");
        copy_out(&list, &alone, 0, &out, 32, 20);
        done(&list, &alone);
        assert!(call(&list, (&src, 64), (1, 20), &out, 64));
        list.close();
        host.execute(&[&list]);
        let made: Vec<Vec<u32>> = (0..3).map(|i| words(seen, i * 32, 5)).collect();
        assert_eq!(made[0][..2], [1 << 24, 4]);
        assert_eq!(made[1][..2], [1 << 24, 4], "the call clamped alone, charged second");
        assert_eq!(made[2], [0, 0, 7, 8, 9], "the call after it, charged third, past the budget");
        assert_eq!(list.batching().counts, [2, 2, 2], "a batch before the call clamped alone and one after");
        list.reset();
    }

    /// A run whose calls read two argument copies, in turn: a pass of the
    /// first copy for the first call, then one pass of both for the rest,
    /// every call its own record.
    #[test]
    fn a_run_over_two_argument_copies_is_clamped_by_one_pass_of_both() {
        let _on = Switch::set(true);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let a = filled(&[(0, &[5, 1, 1]), (32, &[7, 1, 1])]);
        let b = filled(&[(0, &[6, 1, 1]), (32, &[8, 1, 1])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        let calls = [(&a, 0), (&b, 0), (&a, 32), (&b, 32), (&a, 0)];
        for (i, (copy, at)) in calls.into_iter().enumerate() {
            assert!(call(&list, (copy, at), (10, 12), &out, i as u64 * 16), "call {i} batched");
        }
        list.close();
        assert_eq!(list.batching().counts, [2, 5, 1], "the first call's pass, then one of both copies for the four after");
        host.execute(&[&list]);
        let made: Vec<Vec<u32>> = (0..5).map(|i| words(seen, i * 16, 3)).collect();
        assert_eq!(made, [[5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1], [5, 1, 1]].map(|r| r.to_vec()));
        list.reset();
    }

    /// A draw that can store ends the run (`State::bind` ends the batch):
    /// one writes a later call's record between two calls, and that call
    /// reads the record as rewritten, in a batch of its own. The writer here
    /// is a dispatch of ours followed by the batch's end, as the bind
    /// records it; a translated draw through the bind path itself is
    /// [`a_translated_draw_that_stores_ends_the_run_and_the_call_after_reads_what_it_wrote`].
    #[test]
    fn a_draw_that_stores_ends_the_run_and_the_next_call_reads_what_it_wrote() {
        let _on = Switch::set(true);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let src = filled(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let writer = compute(WRITER);
        let list = host.new_list().unwrap();
        assert!(call(&list, (&src, 0), (10, 12), &out, 0));
        // The storing draw: it rewrites the record at 32 (to 9), and its
        // binding ends the batch, as `State::bind` does for a pipeline that
        // can store to buffer memory.
        // SAFETY: an open list; our pass; a live buffer.
        unsafe {
            list.list.SetComputeRootSignature(&writer.root);
            list.list.SetPipelineState(&writer.pso);
            list.list.SetComputeRootUnorderedAccessView(0, src.GetGPUVirtualAddress() + 32);
            list.list.SetComputeRoot32BitConstants(1, 2, [9u32, 1].as_ptr().cast(), 0);
            list.list.Dispatch(1, 1, 1);
        }
        list.batch_end();
        assert!(call(&list, (&src, 32), (10, 12), &out, 16));
        list.close();
        host.execute(&[&list]);
        assert_eq!(words(seen, 0, 3), [5, 1, 1]);
        // The store wrote 9, 1, 0: clamped as it is (its z of 0 runs it empty).
        assert_eq!(words(seen, 16, 3), [9, 1, 0], "the call after the store reads the record it wrote");
        assert_eq!(list.batching().counts, [2, 2, 2]);
        list.reset();
    }

    /// A batch begun for a run that turns out to be one call (a dispatch
    /// right after it): one pass,
    /// as a call's own clamp, no list cut and nothing wasted but the slots
    /// it did not use; the next run begins at the first size again.
    #[test]
    fn a_run_of_one_call_costs_one_pass_and_the_next_run_starts_small() {
        let _on = Switch::set(true);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let src = filled(&[(0, &[3, 1, 1]), (32, &[4, 1, 1])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let writer = compute(WRITER);
        let scratch = filled(&[]);
        let list = host.new_list().unwrap();
        assert!(call(&list, (&src, 0), (10, 12), &out, 0));
        write_records(&list, &writer, &scratch, 0, (1, 1));
        let (_, second, recorded) = in_run(&list, (&src, 32), (10, 12)).expect("batched");
        assert!(recorded, "the next run's own pass");
        assert_eq!(second, u64::from(FIRST) * SLOT, "after the first batch's four slots");
        let (slots, _, _) = in_run(&list, (&src, 32), (10, 12)).expect("batched");
        copy_out(&list, &slots, second, &out, 16, 12);
        list.close();
        assert_eq!(list.batching().counts, [2, 3, 1]);
        host.execute(&[&list]);
        assert_eq!((words(seen, 0, 3), words(seen, 16, 3)), (vec![3, 1, 1], vec![4, 1, 1]));
        list.reset();
    }

    /// A vertex program of the console's that stores 9 at byte 32 of the
    /// buffer its root UAV (the title's root parameter 0) names: the view's
    /// address made a buffer descriptor, then the store.
    #[cfg(feature = "diagnostics")]
    const STORES_9_AT_32: [u32; 8] = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0301, // s_mov_b32 s5, s1
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x7E02_0289, // v_mov_b32 v1, 9
        0xE070_0020,
        0x8001_0100, // buffer_store_dword v1, off, s[4:7], 0 offset:32
        0xBF81_0000, // s_endpgm
    ];

    /// The two title pipelines of the fixtures below, translated and made
    /// as the title's are (`crate::pipeline::test_programs`): one whose
    /// vertex program stores nothing, and one whose program is
    /// [`STORES_9_AT_32`], as the pipelines' classification of their
    /// programs sees them. Points, no targets.
    #[cfg(feature = "diagnostics")]
    fn title_pipelines() -> (This, This) {
        use crate::pipeline::test_programs::{graphics, vertex_container};
        use victory_shader::{ResourceType, RootParam, RootType, UsedResource};
        let layout = [RootParam { root: RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 0, slot: 0 }];
        let used = [UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 0,
            slot: 0,
        }];
        const POINTS: u32 = 1;
        let (plain, made) = graphics(&vertex_container(4, &[0xBF81_0000], &[], &[]), POINTS);
        let plain_stores = made.expect("a host pipeline for the program that stores nothing").stages[0].translated().stores;
        let (storing, made) = graphics(&vertex_container(4, &STORES_9_AT_32, &layout, &used), POINTS);
        let storing_stores = made.expect("a host pipeline for the program that stores").stages[0].translated().stores;
        assert_eq!((plain_stores, storing_stores), (false, true), "the programs as the classification sees them");
        (plain, storing)
    }

    /// Title memory of a title buffer of 64 KiB (the process's own, as the
    /// title's is) with `records` in it: its address.
    #[cfg(feature = "diagnostics")]
    fn title_buffer(records: &[(u64, &[u32])]) -> u64 {
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: a fresh allocation, never freed: lists may hold its copy.
        let at = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!at.is_null());
        for (offset, words) in records {
            // SAFETY: inside the allocation.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), at.add(*offset as usize).cast::<u32>(), words.len()) };
        }
        crate::shadow::register(at as usize, at as u64, 1 << 16);
        at as u64
    }

    /// The slot the run's last call reads its clamped record from.
    #[cfg(feature = "diagnostics")]
    fn last_slot(list: &HostList) -> (ID3D12Resource, u64) {
        let b = list.batching();
        let open = b.open.as_ref().expect("a batch open");
        (open.slots.clone(), open.base + u64::from(open.calls - 1) * SLOT)
    }

    /// What [`around_a_storing_draw`] saw: each call's record as the call
    /// reads it, the list's batches (passes, calls in them, passes of one
    /// call), and the word at 32 of the title's memory once the list ran.
    #[cfg(feature = "diagnostics")]
    struct Seen {
        records: Vec<Vec<u32>>,
        counts: [u64; 3],
        title: u32,
    }

    /// On a title list, through its draw state as the title's calls reach
    /// it (`SetPipelineState`, `ExecuteIndirect`, `DrawInstanced`): with
    /// `before`, an indirect draw of the record at 0 ([5, 1, 1, 0]) by
    /// `plain`; a draw of one point by `storing`, its root UAV the
    /// argument buffer (it stores 9 at 32); an indirect draw of the record
    /// at 32 ([6, 1, 1, 0] before the store) by `plain`. The argument
    /// buffer's copy is one already ([`argument_buffer`]). The list goes
    /// through the title's `Close` and a title queue.
    #[cfg(feature = "diagnostics")]
    fn around_a_storing_draw(plain: This, storing: This, before: bool) -> Seen {
        use crate::object::slot;
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let memory = argument_buffer(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])], plain);
        let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let title = crate::list::test_list();
        let mut slots = Vec::new();
        crate::list::test_state(title, |list, s| {
            s.set_heaps(&heaps);
            s.set_topology(list, 1);
            if before {
                s.set_pipeline(plain);
                assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the call before the draw");
                slots.push(last_slot(list));
            }
            s.set_pipeline(storing);
            s.set_view(false, 0, memory);
            s.draw(list, 1, 1, 0, 0);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory + 32, None), "the call after the draw");
            slots.push(last_slot(list));
            for (i, (buffer, at)) in slots.iter().enumerate() {
                copy_out(list, buffer, *at, &out, 16 * i as u64, 16);
            }
        });
        let close = slot::<extern "system" fn(This) -> windows::core::HRESULT>(title, 0x48);
        assert!(close(title).is_ok(), "the title's list closes");
        let counts = crate::list::host_list(title).expect("its host list").batching().counts;
        let queue = crate::queue::test_queue();
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &title);
        crate::host::settle_for_present();
        let records = (0..slots.len()).map(|i| words(seen, 16 * i, 4)).collect();
        // SAFETY: the title's memory; the list has run and its writes are back.
        let title_word = unsafe { (memory as *const u32).add(8).read_volatile() };
        crate::shadow::unregister(memory as usize, memory);
        Seen { records, counts, title: title_word }
    }

    /// A translated draw that stores to the argument buffer, between two
    /// calls of a run, ends the run where `State::bind` binds it (the
    /// pipeline's classification: it can store to buffer memory): the call
    /// after it is clamped by a pass of its own, recorded after the draw,
    /// and reads the record the draw wrote. With the classification forced
    /// to "stores nothing", the bind leaves the batch open: both calls are
    /// clamped by the one pass recorded before the draw, and the call after
    /// it reads the record as it was.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_translated_draw_that_stores_ends_the_run_and_the_call_after_reads_what_it_wrote() {
        let _on = Switch::set(true);
        let (plain, storing) = title_pipelines();
        let seen = around_a_storing_draw(plain, storing, true);
        assert_eq!(seen.title, 9, "the draw's store, copied back into title memory");
        assert_eq!(seen.records[0], [5, 1, 1, 0], "the call before the draw");
        assert_eq!(seen.records[1], [9, 1, 1, 0], "the call after the draw reads the record the draw wrote");
        assert_eq!(seen.counts, [2, 2, 2], "a pass for each call: the draw's bind ended the first batch");
        let was = crate::draw::STORES_FORCED.replace(Some(false));
        let forced = around_a_storing_draw(plain, storing, true);
        crate::draw::STORES_FORCED.set(was);
        assert_eq!(forced.title, 9, "forced: the draw stores all the same");
        assert_eq!(forced.records[1], [6, 1, 1, 0], "forced: the call after the draw reads the record as it was");
        assert_eq!(forced.counts, [1, 2, 0], "forced: one pass, before the draw, for both calls");
    }

    /// Runs [`an_argument_buffer_a_draw_wrote_is_read_in_its_own_state_child`]
    /// in a process of its own, with the D3D12 debug layer, which must be
    /// on before the first device is made (a machine without it says so,
    /// loudly, and checks the rest).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn an_argument_buffer_a_draw_wrote_is_read_in_its_own_state_by_the_next_batch() {
        in_debug_child("indirect::run_tests::an_argument_buffer_a_draw_wrote_is_read_in_its_own_state_child", true, 1, false);
    }

    /// Runs the ignored tests `name` names (exactly, or all whose names
    /// begin with it) in a process of their own, one after the other, with
    /// the D3D12 debug layer, which must be on before the first device is
    /// made, and with `gbv` GPU-based validation with its state tracking
    /// (`VICTORY_D3D12_GBV=states`): `n` of them must pass.
    #[cfg(feature = "diagnostics")]
    fn in_debug_child(name: &str, exact: bool, n: usize, gbv: bool) {
        let exe = std::env::current_exe().expect("the test binary");
        let mut args = vec![name, "--ignored", "--test-threads", "1", "--nocapture"];
        if exact {
            args.insert(0, "--exact");
        }
        let mut child = std::process::Command::new(exe);
        child.args(args).env("VICTORY_D3D12_DEBUG", "1").env_remove("VICTORY_GPU_TIMING");
        if gbv {
            child.env("VICTORY_D3D12_GBV", "states");
        } else {
            child.env_remove("VICTORY_D3D12_GBV");
        }
        let out = child.output().expect("the child ran");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        for line in text.lines().filter(|l| l.contains("NOT RUN")) {
            eprintln!("{line}");
        }
        assert!(out.status.success() && text.contains(&format!("{n} passed")), "{text}");
    }

    /// The child of the test above. The storing draw writes the argument
    /// buffer's host copy through an unordered access view (a one-view
    /// table, as the bind path binds a root UAV), which the debug layer
    /// does not follow (a descriptor's access); the bind moves the copy, an
    /// argument copy, to unordered access by a transition of the list's own
    /// (the module's comment), and the layer checks the batch pass that
    /// clamps the call after the draw against that.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the test above"]
    fn an_argument_buffer_a_draw_wrote_is_read_in_its_own_state_child() {
        let _on = Switch::set(true);
        let layer = crate::host::debug_layer_on();
        if !layer {
            eprintln!("NOT RUN (no debug layer): the D3D12 debug layer's check of the batch pass's views; the rest is checked");
        }
        let (plain, storing) = title_pipelines();
        let earlier = crate::host::take_debug_errors();
        let seen = around_a_storing_draw(plain, storing, false);
        let said = crate::host::take_debug_errors();
        assert!(earlier.is_empty(), "the debug layer, before the list: {earlier:#?}");
        assert_eq!(seen.records[0], [9, 1, 1, 0], "the call after the draw reads the record the draw wrote");
        assert_eq!(seen.counts, [1, 1, 1]);
        if layer {
            assert!(said.is_empty(), "the debug layer: {said:#?}");
        }
    }

    /// Runs the `states_child_` tests below in a process of their own, with
    /// the D3D12 debug layer: the clamps read an argument copy, and its
    /// count, in every state a list or the lists before it in a submission
    /// can leave the copy in, and leave it in one the list's next access
    /// can be made in.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_clamps_read_argument_copies_in_states_the_debug_layer_accepts() {
        in_debug_child("indirect::run_tests::states_child_", false, 4, false);
    }

    /// Whether the debug layer is on (said, if not), its errors so far
    /// taken.
    #[cfg(feature = "diagnostics")]
    fn layer_on() -> bool {
        let layer = crate::host::debug_layer_on();
        if !layer {
            eprintln!("NOT RUN (no debug layer): the D3D12 debug layer's check of the clamps' states; the rest is checked");
        }
        assert!(crate::host::take_debug_errors().is_empty(), "the debug layer, before the list");
        layer
    }

    /// The debug layer's errors since [`layer_on`]: none.
    #[cfg(feature = "diagnostics")]
    fn layer_said_nothing(layer: bool) {
        let said = crate::host::take_debug_errors();
        if layer {
            assert!(said.is_empty(), "the debug layer: {said:#?}");
        }
    }

    /// On a title list, through its draw state, as [`around_a_storing_draw`]
    /// records: `body` is given the list, its state (heaps set, points),
    /// and the address of title memory of a title buffer of 64 KiB holding
    /// `records`, whose host copy is an argument copy already (a call of an
    /// earlier submission by `plain` read it: [`argument_buffer`]). The
    /// list goes through the title's `Close` and a title queue. Its batches
    /// (passes, calls in them, passes of one call).
    #[cfg(feature = "diagnostics")]
    fn on_title_list(
        records: &[(u64, &[u32])],
        plain: This,
        body: impl FnOnce(&HostList, &mut crate::draw::State, u64),
    ) -> [u64; 3] {
        use crate::object::slot;
        let memory = argument_buffer(records, plain);
        let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
        let title = crate::list::test_list();
        crate::list::test_state(title, |list, s| {
            s.set_heaps(&heaps);
            s.set_topology(list, 1);
            body(list, s, memory);
        });
        let close = slot::<extern "system" fn(This) -> windows::core::HRESULT>(title, 0x48);
        assert!(close(title).is_ok(), "the title's list closes");
        let counts = crate::list::host_list(title).expect("its host list").batching().counts;
        let queue = crate::queue::test_queue();
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &title);
        crate::host::settle_for_present();
        crate::shadow::unregister(memory as usize, memory);
        counts
    }

    /// The clamp of a call alone (the clamp of runs off; and a call with a
    /// count buffer, always clamped alone) after a draw of the list stored
    /// into the argument copy: it reads the copy and the count in it in the
    /// state the store's bind left, and the call reads its count where that
    /// state allows.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_argument_copies_in_states_the_debug_layer_accepts"]
    fn states_child_a_call_clamped_alone_reads_a_copy_a_shader_stored_into() {
        let _off = Switch::set(false);
        let layer = layer_on();
        let (plain, storing) = title_pipelines();
        let counts = on_title_list(&[(0, &[5, 1, 1, 0]), (48, &[1])], plain, |list, s, memory| {
            s.set_pipeline(storing);
            s.set_view(false, 0, memory);
            s.draw(list, 1, 1, 0, 0);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the call");
            assert!(s.execute_indirect(list, (0, 16, 1), memory, Some(memory + 48)), "the call with a count");
        });
        assert_eq!(counts, [0; 3], "no batch");
        layer_said_nothing(layer);
    }

    /// A batch read the argument copy before a draw of the list stores into
    /// it: the store must still find the copy in a state it can be made in
    /// (the common state, from which it is promoted, or unordered access),
    /// and the call after it reads the record the store wrote.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_argument_copies_in_states_the_debug_layer_accepts"]
    fn states_child_a_draw_stores_into_a_copy_a_batch_read_before_it() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, storing) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let counts = on_title_list(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])], plain, |list, s, memory| {
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the call before the draw");
            let (slots, at) = last_slot(list);
            copy_out(list, &slots, at, &out, 0, 16);
            s.set_pipeline(storing);
            s.set_view(false, 0, memory);
            s.draw(list, 1, 1, 0, 0);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory + 32, None), "the call after the draw");
            let (slots, at) = last_slot(list);
            copy_out(list, &slots, at, &out, 16, 16);
        });
        assert_eq!((words(seen, 0, 4), words(seen, 16, 4)), (vec![5, 1, 1, 0], vec![9, 1, 1, 0]));
        assert_eq!(counts, [2, 2, 2]);
        layer_said_nothing(layer);
    }

    /// A list before this one in the submission stored into the argument
    /// copy (the lists of one submission run as one: nothing goes back to
    /// the common state between them), and this list's first batch reads
    /// what it stored.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_argument_copies_in_states_the_debug_layer_accepts"]
    fn states_child_a_batch_reads_a_copy_an_earlier_list_stored_into() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let writer = compute(WRITER);
        let a = filled(&[(0, &[7, 7, 7, 7])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let (first, second) = (host.new_list().unwrap(), host.new_list().unwrap());
        write_records(&first, &writer, &a, 0, (5, 1));
        assert!(call(&second, (&a, 0), (0, 16), &out, 0), "batched");
        first.close();
        second.close();
        host.execute(&[&first, &second]);
        assert_eq!(words(seen, 0, 4), [5, 1, 0, 0], "the record the first list stored");
        first.reset();
        second.reset();
        layer_said_nothing(layer);
    }

    /// A batch reads the argument copy a draw stored into, and then a call
    /// with a count buffer in the same copy is clamped alone: both read the
    /// copy, one after the other, in the state it is in.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_argument_copies_in_states_the_debug_layer_accepts"]
    fn states_child_a_call_clamped_alone_after_a_batch_of_the_same_copy() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, storing) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let records: [(u64, &[u32]); 3] = [(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0]), (48, &[1])];
        let counts = on_title_list(&records, plain, |list, s, memory| {
            s.set_pipeline(storing);
            s.set_view(false, 0, memory);
            s.draw(list, 1, 1, 0, 0);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory + 32, None), "the batched call");
            let (slots, at) = last_slot(list);
            copy_out(list, &slots, at, &out, 0, 16);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, Some(memory + 48)), "the call with a count");
        });
        assert_eq!(words(seen, 0, 4), [9, 1, 1, 0], "the batched call reads the record the draw wrote");
        assert_eq!(counts, [1, 1, 1]);
        layer_said_nothing(layer);
    }

    /// Runs the `reads_child_` tests below in a process of their own, with
    /// the D3D12 debug layer and GPU-based validation with its state
    /// tracking: a title draw reads its indices from an argument copy,
    /// which leaves the copy in a read state for the rest of the lists sent
    /// with it (promoted from the common state: the layer follows a read of
    /// the input assembler's), and a clamp reads the copy after that read.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept() {
        in_debug_child("indirect::run_tests::reads_child_", false, 5, true);
    }

    /// Where the fixtures' indexed draws find their indices in the argument
    /// buffer (three 32-bit zeros), and the format of those (PC numbering).
    #[cfg(feature = "diagnostics")]
    const INDICES: u64 = 256;
    #[cfg(feature = "diagnostics")]
    const R32_UINT: u32 = 42;

    /// A title draw of three points by `plain` whose indices are in the
    /// argument buffer at `memory`: the input assembler reads them from its
    /// host copy.
    #[cfg(feature = "diagnostics")]
    fn read_indices(list: &HostList, s: &mut crate::draw::State, plain: This, memory: u64) {
        s.set_pipeline(plain);
        s.set_index_buffer(Some(&crate::draw::IndexView::test(memory + INDICES, 12, R32_UINT)));
        s.draw_indexed(list, 3, 1, 0, 0, 0);
    }

    /// Copies `bytes` of the list's scratch buffer for calls clamped alone
    /// (in the common state between calls) into `out` at `to`: what the
    /// call recorded just before reads.
    #[cfg(feature = "diagnostics")]
    fn scratch_out(list: &HostList, out: &ID3D12Resource, to: u64, bytes: u64) {
        let scratch = list.indirect_scratch(bytes).expect("the list's scratch buffer");
        let (common, source) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        // SAFETY: an open list; live buffers, the scratch buffer in the
        // common state (`done`).
        unsafe {
            list.list.ResourceBarrier(&[transition(&scratch, common, source)]);
            list.list.CopyBufferRegion(out, to, &scratch, 0, bytes);
            list.list.ResourceBarrier(&[transition(&scratch, source, common)]);
        }
    }

    /// A title list's commands: given the list and its state (heaps set,
    /// points).
    #[cfg(feature = "diagnostics")]
    type Body<'a> = Box<dyn FnOnce(&HostList, &mut crate::draw::State) + 'a>;

    /// Title lists, one a body, closed and sent together in one submission
    /// of a title queue (the lists of one submission run as one: no state
    /// goes back to common between them), which is waited for.
    #[cfg(feature = "diagnostics")]
    fn send_title_lists(bodies: Vec<Body<'_>>) {
        use crate::object::slot;
        let titles: Vec<This> = bodies
            .into_iter()
            .map(|body| {
                let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
                let title = crate::list::test_list();
                crate::list::test_state(title, |list, s| {
                    s.set_heaps(&heaps);
                    s.set_topology(list, 1);
                    body(list, s);
                });
                let close = slot::<extern "system" fn(This) -> windows::core::HRESULT>(title, 0x48);
                assert!(close(title).is_ok(), "the title's list closes");
                title
            })
            .collect();
        let queue = crate::queue::test_queue();
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, titles.len() as u32, titles.as_ptr());
        crate::host::settle_for_present();
    }

    /// Title memory of a title buffer of 64 KiB holding `records`, whose
    /// host copy a call of an earlier submission read its record from, as
    /// the title's argument buffers have from their first frame on.
    #[cfg(feature = "diagnostics")]
    fn argument_buffer(records: &[(u64, &[u32])], plain: This) -> u64 {
        let memory = title_buffer(records);
        send_title_lists(vec![Box::new(|list, s| {
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the earlier submission's call");
        })]);
        memory
    }

    /// The clamp of a call alone (the clamp of runs off), between a call
    /// clamped alone and a draw that read its indices from the same copy:
    /// both calls read their records.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept"]
    fn reads_child_a_call_clamped_alone_after_a_draw_read_the_copy() {
        let _off = Switch::set(false);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let memory = argument_buffer(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])], plain);
        send_title_lists(vec![Box::new(|list, s| {
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the call before the draw");
            scratch_out(list, &out, 0, 16);
            read_indices(list, s, plain, memory);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory + 32, None), "the call after the draw");
            scratch_out(list, &out, 16, 16);
            assert_eq!(list.batching().counts, [0; 3], "no batch");
        })]);
        crate::shadow::unregister(memory as usize, memory);
        assert_eq!((words(seen, 0, 4), words(seen, 16, 4)), (vec![5, 1, 1, 0], vec![6, 1, 1, 0]));
        layer_said_nothing(layer);
    }

    /// A batch whose first call comes after a draw that read its indices
    /// from the argument copy, in the same list.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept"]
    fn reads_child_a_batch_after_a_draw_read_the_copy() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let memory = argument_buffer(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])], plain);
        send_title_lists(vec![Box::new(|list, s| {
            read_indices(list, s, plain, memory);
            s.set_pipeline(plain);
            for (i, at) in [0, 32].into_iter().enumerate() {
                assert!(s.execute_indirect(list, (0, 16, 1), memory + at, None), "a call after the draw");
                let (slots, at) = last_slot(list);
                copy_out(list, &slots, at, &out, 16 * i as u64, 16);
            }
            assert_eq!(list.batching().counts, [0, 0, 0], "the batch is open");
        })]);
        crate::shadow::unregister(memory as usize, memory);
        assert_eq!((words(seen, 0, 4), words(seen, 16, 4)), (vec![5, 1, 1, 0], vec![6, 1, 1, 0]));
        layer_said_nothing(layer);
    }

    /// A batch reads copy A; a draw reads its indices from A; a call from
    /// copy B begins the next batch, which carries A and reads it too; the
    /// call after it, from A, is that batch's.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept"]
    fn reads_child_a_batch_carries_a_copy_a_draw_read() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let a = argument_buffer(&[(0, &[5, 1, 1, 0]), (32, &[6, 1, 1, 0])], plain);
        let b = argument_buffer(&[(0, &[7, 1, 1, 0])], plain);
        let mut sources = 0;
        send_title_lists(vec![Box::new(|list, s| {
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), a, None), "the call from A");
            let (slots, at) = last_slot(list);
            copy_out(list, &slots, at, &out, 0, 16);
            read_indices(list, s, plain, a);
            s.set_pipeline(plain);
            for (i, at) in [b, a + 32].into_iter().enumerate() {
                assert!(s.execute_indirect(list, (0, 16, 1), at, None), "a call after the draw");
                let (slots, at) = last_slot(list);
                copy_out(list, &slots, at, &out, 16 * (i as u64 + 1), 16);
            }
            sources = list.batching().open.as_ref().map_or(0, |o| o.sources.len());
        })]);
        crate::shadow::unregister(a as usize, a);
        crate::shadow::unregister(b as usize, b);
        assert_eq!(sources, 2, "the second batch carries A");
        let seen: Vec<Vec<u32>> = (0..3).map(|i| words(seen, 16 * i, 4)).collect();
        assert_eq!(seen, [[5, 1, 1, 0], [7, 1, 1, 0], [6, 1, 1, 0]]);
        layer_said_nothing(layer);
    }

    /// A call with a count buffer (clamped alone, the clamp of runs on or
    /// off) whose records and count are in the copy a draw read its indices
    /// from: the call reads its record and the count.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept"]
    fn reads_child_a_call_with_a_count_after_a_draw_read_the_copy() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let memory = argument_buffer(&[(0, &[5, 1, 1, 0]), (48, &[1])], plain);
        send_title_lists(vec![Box::new(|list, s| {
            read_indices(list, s, plain, memory);
            s.set_pipeline(plain);
            assert!(s.execute_indirect(list, (0, 16, 1), memory, Some(memory + 48)), "the call with a count");
            scratch_out(list, &out, 0, count_at(16, 1) + 4);
        })]);
        crate::shadow::unregister(memory as usize, memory);
        assert_eq!(words(seen, 0, 5), [5, 1, 1, 0, 1], "the record, then the count the call reads");
        layer_said_nothing(layer);
    }

    /// Two lists sent together: a draw of the first reads its indices from
    /// the argument copy, and a batch of the second reads it.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the_clamps_read_a_copy_a_title_draw_read_in_states_the_layers_accept"]
    fn reads_child_a_batch_after_a_list_whose_draw_read_the_copy() {
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let memory = argument_buffer(&[(0, &[5, 1, 1, 0])], plain);
        send_title_lists(vec![
            Box::new(|list, s| read_indices(list, s, plain, memory)),
            Box::new(|list, s| {
                s.set_pipeline(plain);
                assert!(s.execute_indirect(list, (0, 16, 1), memory, None), "the call on the second list");
                let (slots, at) = last_slot(list);
                copy_out(list, &slots, at, &out, 0, 16);
            }),
        ]);
        crate::shadow::unregister(memory as usize, memory);
        assert_eq!(words(seen, 0, 4), [5, 1, 1, 0]);
        layer_said_nothing(layer);
    }

    /// A console vertex program that stores `words` (byte offset, value)
    /// into the buffer its root UAV (the title's root parameter 0) names,
    /// as [`STORES_9_AT_32`] stores its one word.
    #[cfg(feature = "diagnostics")]
    fn stores_program(words: &[(u32, u32)]) -> Vec<u32> {
        // The view's address made a buffer descriptor, s[4:7].
        let mut code = vec![0xBE84_0300, 0xBE85_0301, 0xBE86_03C1, 0xBE87_0380];
        for &(at, value) in words {
            assert!(at < 4096, "a store's offset fits its instruction");
            // v_mov_b32 v1, value; buffer_store_dword v1, off, s[4:7], 0 offset:at
            code.extend([0x7E02_02FF, value, 0xE070_0000 | at, 0x8001_0100]);
        }
        code.push(0xBF81_0000);
        code
    }

    /// A title pipeline of points whose vertex program is `code`, with one
    /// view of `kind` at the title's root parameter `slot` (`root`: a root
    /// descriptor or a table), made as [`title_pipelines`] makes its own.
    #[cfg(feature = "diagnostics")]
    fn title_pipeline(code: &[u32], view: Option<(victory_shader::ResourceType, victory_shader::RootType, u8)>) -> This {
        use crate::pipeline::test_programs::{graphics, vertex_container};
        use victory_shader::{RootParam, RootType, UsedResource};
        let (layout, used) = match view {
            Some((kind, root, slot)) => {
                let sgprs = if root == RootType::Table { 1 } else { 2 };
                let param = RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data: 0, slot };
                let dwords = if root == RootType::Table { 8 } else { 2 };
                let view = UsedResource { kind, root, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords, user_data: 0, slot };
                (vec![param], vec![view])
            }
            None => (Vec::new(), Vec::new()),
        };
        let (made, host) = graphics(&vertex_container(4, code, &layout, &used), 1);
        assert!(host.is_some(), "a host pipeline for the program");
        made
    }

    /// Adds one, atomically, to the word at 0 of the view at u0: an
    /// execution of the list it is recorded in, counted.
    #[cfg(feature = "diagnostics")]
    const COUNTS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint was;
    hits.InterlockedAdd(0, 1, was);
}
"#;

    /// Records [`COUNTS`] (made as `counter`) into `list` over `hits`.
    #[cfg(feature = "diagnostics")]
    fn count_execution(list: &HostList, counter: &Clamp, hits: &ID3D12Resource) {
        // SAFETY: an open list; our pass; a live buffer.
        unsafe {
            list.list.SetComputeRootSignature(&counter.root);
            list.list.SetPipelineState(&counter.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
        }
    }

    /// The title's wait for a memory value on list `this`, met (no
    /// address): the list is cut there, as with waits cut into lists.
    #[cfg(feature = "diagnostics")]
    fn wait_met(this: This) {
        use crate::object::slot;
        const EQUAL: u32 = 3;
        // The 32-bit wait: 0x238 in older titles' lists.
        let wait = crate::list::test_method(this, 71);
        slot::<extern "system" fn(This, u64, u32, u64)>(this, wait)(this, 0, EQUAL, 0);
    }

    /// Closes the title's list `this` and sends `named` (lists, this one
    /// among them) in one submission of a title queue, waited for.
    #[cfg(feature = "diagnostics")]
    fn close_and_send(this: This, named: &[This]) {
        use crate::object::slot;
        let close = slot::<extern "system" fn(This) -> windows::core::HRESULT>(this, 0x48);
        assert!(close(this).is_ok(), "the title's list closes");
        let queue = crate::queue::test_queue();
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, named.len() as u32, named.as_ptr());
        crate::host::settle_for_present();
    }

    /// Runs the `accept_child_` tests below in a process of their own, with
    /// the D3D12 debug layer and GPU-based validation with its state
    /// tracking.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_title_stream_sent_twice_reads_what_its_stores_left_in_states_the_layers_accept() {
        in_debug_child("indirect::run_tests::accept_child_", false, 1, true);
    }

    /// The acceptance stream (Astra's brainstorm, round five), on one title
    /// list sent twice in one submission ([L, L]): a copy no clamp has read
    /// before is read as indices and as a root constant buffer; a draw
    /// stores two records and a count into it through a root unordered
    /// access view, the second record poisoned (0xFFFFFFFF instances of one
    /// vertex: half the budget, clamped); a run of two calls reads them; a
    /// draw stores again at the same offsets; the run again; a call with a
    /// count over four records, the second poisoned and the two past the
    /// count poisoned (read, they would cost the budget twice over). Each
    /// execution counts itself. Both run, and each call reads the clamped
    /// record the stores before it left. The first store rewrites all the
    /// second changed, so both executions clamp alike; the second reads the
    /// first poisoned record in full only from a budget of its own (one
    /// kept from the first would empty it).
    /// shortcut: Astra's stream stores through a view picked from a heap by
    /// index, has a wait cut between two uses of one constant table, and is
    /// recycled while it runs: here the store is through a root view (the
    /// clamp's barriers do not depend on which view stored), the wait cut
    /// has a test of its own
    /// ([`a_wait_cut_gives_the_part_after_it_its_own_constant_table`]), and
    /// the submission is waited for.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of a_title_stream_sent_twice_reads_what_its_stores_left_in_states_the_layers_accept"]
    fn accept_child_a_title_stream_sent_twice_clamps_what_its_stores_left() {
        use victory_shader::{ResourceType, RootType};
        let _on = Switch::set(true);
        let layer = layer_on();
        let (plain, _) = title_pipelines();
        let root_uav = Some((ResourceType::Uav, RootType::Descriptor, 0));
        let first = title_pipeline(&stores_program(&[(0x000, 4), (0x010, 1), (0x014, u32::MAX), (0x180, 2)]), root_uav);
        let again = title_pipeline(&stores_program(&[(0x000, 6), (0x014, 3)]), root_uav);
        let reader = title_pipeline(&[0xBF81_0000], Some((ResourceType::Cbv, RootType::Descriptor, 0)));
        let poison: &[u32] = &[u32::MAX, u32::MAX, 0xDEAD, 0xBEEF];
        let memory = title_buffer(&[
            (0x000, &[3, 1, 0, 0]),
            (0x010, &[0, u32::MAX, 0, 0]),
            (0x100, &[2, 1, 0, 0]),
            (0x110, &[0, u32::MAX, 0, 0]),
            (0x120, poison),
            (0x130, poison),
            (0x180, &[9]),
        ]);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let (hits, counted) = host.shared_buffer(256).unwrap();
        let counter = compute(COUNTS);
        let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
        let title = crate::list::test_list();
        crate::list::test_state(title, |list, s| {
            s.set_heaps(&heaps);
            s.set_topology(list, 1);
            read_indices(list, s, plain, memory);
            s.set_pipeline(reader);
            s.set_view(false, 0, memory);
            s.draw(list, 1, 1, 0, 0);
        });
        // Calls through the title's `ExecuteIndirect`, each record's slot
        // copied out after it.
        let run = |stores: This, first_out: u64| {
            crate::list::test_state(title, |list, s| {
                s.set_pipeline(stores);
                s.set_view(false, 0, memory);
                s.draw(list, 1, 1, 0, 0);
                s.set_pipeline(plain);
            });
            for (n, at) in [0x000u64, 0x010].into_iter().enumerate() {
                assert!(crate::list::test_indirect(title, (0, 16, 1), memory + at, None), "a call of the run");
                crate::list::test_state(title, |list, _| {
                    let (slots, place) = last_slot(list);
                    copy_out(list, &slots, place, &out, first_out + 16 * n as u64, 16);
                });
            }
        };
        run(first, 0);
        run(again, 32);
        assert!(crate::list::test_indirect(title, (0, 16, 4), memory + 0x100, Some(memory + 0x180)), "the call with a count");
        crate::list::test_state(title, |list, _| {
            scratch_out(list, &out, 64, count_at(16, 4) + 4);
            count_execution(list, &counter, &hits);
        });
        close_and_send(title, &[title, title]);
        crate::shadow::unregister(memory as usize, memory);
        assert_eq!(words(counted, 0, 1), [2], "both executions of the list ran");
        let records: Vec<Vec<u32>> = (0..6).map(|i| words(seen, 16 * i, 4)).collect();
        let clamped = 1 << 26;
        assert_eq!(
            records,
            [[4, 1, 0, 0], [1, clamped, 0, 0], [6, 1, 0, 0], [1, 3, 0, 0], [2, 1, 0, 0], [0, clamped, 0, 0]],
            "the run, the run after the second store, the call with a count"
        );
        assert_eq!(words(seen, 64 + count_at(16, 4) as usize, 1), [2], "the count the store left");
        layer_said_nothing(layer);
    }

    /// A wait cut between two draws through the same table of constants
    /// (Neon's fourth review, finding 1): the part after the wait makes the
    /// table again, its constants copied into its own upload memory. A
    /// table kept from the part before would name that part's memory,
    /// which is reset with its part (possibly before this one runs).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_wait_cut_gives_the_part_after_it_its_own_constant_table() {
        use victory_shader::{ResourceType, RootType};
        let table_reader = title_pipeline(&[0xBF81_0000], Some((ResourceType::Cbv, RootType::Table, 1)));
        let constants = title_buffer(&[(0, &[1, 2, 3, 4])]);
        let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
        let handle = crate::descriptor::heap(heaps[0]).handle(0).expect("a slot of the heap");
        let view = [constants as u32, (constants >> 32) as u32, 256, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &view, handle);
        let title = crate::list::test_list();
        let mut fills = Vec::new();
        crate::list::test_state(title, |list, s| {
            s.set_heaps(&heaps);
            s.set_topology(list, 1);
            s.set_pipeline(table_reader);
            s.set_table(false, 1, handle);
            s.draw(list, 1, 1, 0, 0);
            fills.push(list.fills());
        });
        wait_met(title);
        assert!(crate::list::cut_list(title), "the list is cut at the wait");
        crate::list::test_state(title, |list, s| {
            s.draw(list, 1, 1, 0, 0);
            fills.push(list.fills());
        });
        close_and_send(title, &[title]);
        crate::shadow::unregister(constants as usize, constants);
        assert_eq!(fills, [1, 1], "each part copies the table's constants into its own memory");
    }

    /// A list cut at a wait and sent twice in one submission ([L, L]) runs
    /// twice, both parts (Neon's fourth review, finding 4, with a wait of
    /// the title's): the part after the wait counts each execution. It runs
    /// once, here and on `world`: `crate::list::detach_parts` drops the
    /// second naming of a list in parts (its shortcut), as each part's host
    /// list goes with one held run of the queue only.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "fails, here and on world: a list in parts named twice runs once (crate::list::detach_parts' shortcut); a job of its own"]
    fn a_list_cut_at_a_wait_and_sent_twice_runs_twice() {
        let (plain, _) = title_pipelines();
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (hits, counted) = host.shared_buffer(256).unwrap();
        let counter = compute(COUNTS);
        let heaps = [crate::descriptor::test_heap(false, 16), crate::descriptor::test_heap(true, 16)];
        let title = crate::list::test_list();
        crate::list::test_state(title, |list, s| {
            s.set_heaps(&heaps);
            s.set_topology(list, 1);
            s.set_pipeline(plain);
            s.draw(list, 1, 1, 0, 0);
        });
        wait_met(title);
        crate::list::test_state(title, |list, _| count_execution(list, &counter, &hits));
        close_and_send(title, &[title, title]);
        assert_eq!(words(counted, 0, 1), [2], "both executions of the part after the wait ran");
    }

    /// The clamp of runs is on by default, and `VICTORY_CLAMP_RUNS=0` turns
    /// it off; with it off the draw path's choice is no batch: the call is
    /// clamped alone by `bound`, as before the clamp of runs, and the list
    /// makes nothing of it.
    #[test]
    fn with_the_switch_off_every_call_is_clamped_alone_as_before() {
        use std::ffi::OsStr;
        assert!(runs_from(None), "on by default");
        assert!(!runs_from(Some(OsStr::new("0"))), "VICTORY_CLAMP_RUNS=0 turns it off");
        assert!(runs_from(Some(OsStr::new("1"))), "VICTORY_CLAMP_RUNS=1 leaves it on");
        let _off = Switch::set(false);
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let src = filled(&[(0, &[3, 1, 1])]);
        let list = host.new_list().unwrap();
        assert!(in_run(&list, (&src, 0), (10, 12)).is_none(), "a batch with the switch off");
        let scratch = bound(&list, (&src, 0), None, (10, 12, 1)).expect("the clamp");
        assert_eq!(scratch.as_raw(), list.indirect_scratch(64).unwrap().as_raw(), "the list's own scratch buffer");
        done(&list, &scratch);
        list.close();
        let b = list.batching();
        assert!(b.counts == [0; 3] && b.open.is_none() && b.slots.is_none() && !b.argument, "the list made something of a batch");
        drop(b);
        host.execute(&[&list]);
        list.reset();
    }

    /// Without enhanced barriers (a device without them, or
    /// `VICTORY_ENHANCED_BARRIERS=0`; here forced on this thread) the calls
    /// of a run are clamped alone, as before the clamp of runs, each
    /// reading its record through a shader resource view.
    #[test]
    fn without_enhanced_barriers_each_call_of_a_run_is_clamped_alone() {
        let _on = Switch::set(true);
        let was = LEGACY_FORCED.with(|f| f.replace(true));
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let src = filled(&[(0, &[3, 1, 0, 0]), (16, &[0, u32::MAX, 0, 0])]);
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        assert!(list.enhanced().is_none(), "no enhanced barriers on this thread");
        for (n, at) in [0u64, 16].into_iter().enumerate() {
            assert!(in_run(&list, (&src, at), (0, 16)).is_none(), "a batch without enhanced barriers");
            let scratch = bound(&list, (&src, at), None, (0, 16, 1)).expect("the clamp");
            copy_out(&list, &scratch, 0, &out, 16 * n as u64, 16);
            done(&list, &scratch);
        }
        list.close();
        assert!(list.batching().counts == [0; 3], "the list made a batch");
        host.execute(&[&list]);
        list.reset();
        LEGACY_FORCED.with(|f| f.set(was));
        assert_eq!((words(seen, 0, 4), words(seen, 16, 4)), (vec![3, 1, 0, 0], vec![0, 1 << 26, 0, 0]));
    }
}

// What one small call costs this card, bare and with one thing changed or
/// added: a measurement, not a check (it asserts only that the card
/// answered). A timing run of a lit scene has thousands of
/// small commands a frame at microseconds each, and no log says
/// whether that is what we put around a command or what the card takes for
/// one. Each variant here is 10,000 calls into a 2560x1440 target, timed on
/// the card by one pair of timestamps around the whole batch (not one per
/// command, which would time the stamps); the fastest of three batches is
/// printed. The variants from (j) on are a synthetic workload of the
/// runtime's own: mesh dispatches, one record at a time through
/// `ExecuteIndirect`, each after our clamp of its record, with root
/// arguments set for each call; p1
/// to p4 split the clamp's cost (its buffers' memory, its waits, its
/// transitions), p5 to p7 are the runtime's calls with the budget in the
/// card's memory, reset and read back once a list (`HostList::budget`),
/// against p1 and in many lists, x times one clamp of all the records before the calls,
/// x1 to x256 the clamp of runs (the default; `VICTORY_CLAMP_RUNS=0` off) in runs of 1 to 256 calls,
/// and y and z a draw whose constants are read from an upload buffer or
/// from the card's memory.
///
/// Run by name, no game needed, on the card:
/// `cargo test -p victory-gpu --release --locked -- --ignored bare_command_cost --nocapture`
/// The device is the adapter rule's (`Rig::new`): with `VICTORY_D3D12_WARP`
/// that is the software adapter, which a measurement of the card refuses, so
/// the test is skipped there. The mesh programs are compiled by the PC's `dxc.exe`
/// (`VICTORY_PC_DXC`, else the newest Windows SDK's, as the mesh pipeline
/// tests find it); without it, or without mesh shaders, the mesh variants
/// say they were skipped.
#[cfg(test)]
mod bare_cost {
    use std::path::{Path, PathBuf};

    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Calls in a batch.
    const DRAWS: u32 = 10_000;
    /// [`Call::Runs`]' lengths.
    const RUN_LENGTHS: [u32; 4] = [1, 16, 64, 256];
    /// The bytes from one mesh record to the next: a benchmark parameter
    /// (any multiple of 4 from the 12 of `D3D12_DISPATCH_MESH_ARGUMENTS` on;
    /// here each record is followed by 4 bytes of padding), and an indexed
    /// draw's record (`D3D12_DRAW_INDEXED_ARGUMENTS`).
    const MESH_STRIDE: u32 = 16;
    const INDEXED_STRIDE: u32 = 20;
    /// Where the count (1) follows the mesh records.
    const COUNT_AT: u64 = DRAWS as u64 * MESH_STRIDE as u64;
    /// The clamp's record kind for a mesh dispatch (the PC argument type).
    const KIND_MESH: u32 = 10;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// The mesh variants' programs (shader model 6.5, by `dxc.exe`) under
    /// one root signature of the benchmark's own, its parameters in this
    /// order: a descriptor table, 6 root constants, a shader view, a
    /// constant view, a second table and an unordered view. Each mesh group
    /// draws one small triangle (about 45 pixels) in its own cell of a
    /// 32x32 grid, so that 1,024 groups shade no pixel twice. The root views and the second
    /// table are read only under a constant the variants never set: named,
    /// so the program uses them, and never read; `read_mesh` and
    /// `read_texel` read the constant view for real.
    const MESH: &str = r#"
#define RS "DescriptorTable(SRV(t0)), RootConstants(num32BitConstants=6, b0), SRV(t1), CBV(b1), DescriptorTable(SRV(t3)), UAV(u0)"
cbuffer Own : register(b0) { uint4 k0; uint2 k1; };
cbuffer More : register(b1) { uint4 more; };
ByteAddressBuffer first_view : register(t1);
RWByteAddressBuffer written : register(u0);
Texture2D<float4> image : register(t0);
Texture2D<float4> other : register(t3);
struct V { float4 p : SV_Position; };
struct Sent { uint group; };
groupshared Sent to_send;

float4 corner(uint g, uint i) {
    uint extra = 0;
    if (k1.y == 0x5eed5eedu) {
        extra = first_view.Load(0) + written.Load(0) + more.x
            + asuint(other.Load(int3(0, 0, 0)).x);
    }
    float2 cell = float2(g % 32, g / 32) * 0.05 - 0.8;
    float2 p = float2((i << 1) & 2, i & 2) * 0.005 + cell + float2(k0.xy) * 1e-7 + float(extra);
    return float4(p, 0.5, 1.0);
}

[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(uint3 g : SV_GroupID, out vertices V v[3], out indices uint3 t[1]) {
    SetMeshOutputCounts(3, 1);
    v[0].p = corner(g.x, 0);
    v[1].p = corner(g.x, 1);
    v[2].p = corner(g.x, 2);
    t[0] = uint3(0, 1, 2);
}

[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void sent_mesh(uint3 g : SV_GroupID, in payload Sent s, out vertices V v[3], out indices uint3 t[1]) {
    SetMeshOutputCounts(3, 1);
    v[0].p = corner(g.x + s.group, 0);
    v[1].p = corner(g.x + s.group, 1);
    v[2].p = corner(g.x + s.group, 2);
    t[0] = uint3(0, 1, 2);
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void amplify() {
    to_send.group = k0.z;
    DispatchMesh(1, 1, 1, to_send);
}

[RootSignature(RS)]
float4 texel(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}

[RootSignature(RS)]
float4 flat(float4 at : SV_Position) : SV_Target {
    return float4(1.0, 1.0, 1.0, 1.0);
}

// The mesh program and the texel's pixel program, both reading the
// constant view (b1) for real: its words (zero) move the triangle and are
// added to the colour.
[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void read_mesh(uint3 g : SV_GroupID, out vertices V v[3], out indices uint3 t[1]) {
    SetMeshOutputCounts(3, 1);
    float4 moved = float4(asfloat(more.xy), 0.0, 0.0);
    v[0].p = corner(g.x, 0) + moved;
    v[1].p = corner(g.x, 1) + moved;
    v[2].p = corner(g.x, 2) + moved;
    t[0] = uint3(0, 1, 2);
}

[RootSignature(RS)]
float4 read_texel(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0)) + asfloat(more.zwzw);
}
"#;

    /// What one call of a batch is.
    #[derive(Clone, Copy, PartialEq)]
    enum Call {
        /// The one-triangle draw of (a), with `Between` recorded before it.
        Draw(Between),
        /// The same triangle as an indexed draw: direct, or one record a
        /// call through `ExecuteIndirect` (record after record of one
        /// buffer).
        Indexed { indirect: bool },
        /// `DispatchMesh(groups, 1, 1)`; `bound`: the benchmark's own root
        /// arguments ([`MeshRig::set_roots`]) set before every call, to time
        /// setting them.
        Mesh { pipe: Pipe, groups: u32, bound: bool },
        /// One mesh record of one group a call through `ExecuteIndirect`
        /// (the mesh signature, [`MESH_STRIDE`]), record after record of
        /// one buffer in the card's memory; `count`: with a count buffer
        /// (holding 1).
        MeshIndirect { count: bool },
        /// The same as the runtime issues it for a title call
        /// (`crate::draw`'s `run_indirect`): our clamp of the record (one
        /// record, one group) into the scratch buffer at 0, the mesh
        /// pipeline set again, the call reading the scratch buffer, under
        /// the clamp's barriers. The runtime's source records (its copies
        /// of small title buffers) and budget are in the PC's memory.
        Clamped(Clamping),
        /// Clamp once, draw many: before the batch's first call one clamp
        /// pass over all [`DRAWS`] records (source and budget in the card's
        /// memory) into a copy with one record a slot, then one transition
        /// to indirect argument and the mesh pipeline again (timed apart);
        /// each call is then [`Call::MeshIndirect`]'s, reading the next
        /// record of the copy, with nothing between.
        ClampOnce,
        /// The clamp of runs (the default, [`super::batched`]): the
        /// calls in runs of `length`, the first of each behind one pass
        /// ([`super::record_batch`]) over the run's records in the PC's
        /// memory into slots of their own, then the mesh pipeline again;
        /// each call (t)'s root arguments and the call reading its slot,
        /// with nothing between.
        Runs { length: u32 },
        /// [`Pipe::Constants`]' `DispatchMesh(1, 1, 1)`, its constant view
        /// set before each call to the next 256-byte slice of a buffer: an
        /// upload buffer (where the runtime writes a draw's constants), or
        /// with `card` one in the card's memory.
        Constants { card: bool },
    }

    impl Call {
        fn meshes(self) -> bool {
            !matches!(self, Call::Draw(_) | Call::Indexed { .. })
        }
    }

    /// The mesh pipelines, all under [`MESH`]'s root signature, all into
    /// the depth texture.
    #[derive(Clone, Copy, PartialEq)]
    enum Pipe {
        /// The mesh program and a pixel program that reads one texel
        /// through the first table, into the colour target; depth tested,
        /// not written (as (a)'s pipeline).
        Plain = 0,
        /// The same behind an amplification program of one group that
        /// launches one mesh group.
        Amplified = 1,
        /// The mesh program and no pixel program, no colour target; depth
        /// tested and written.
        DepthOnly = 2,
        /// The same with a pixel program that writes a constant colour
        /// into the colour target.
        DepthAndColour = 3,
        /// As `Plain`, its mesh and pixel programs both reading the
        /// constant view.
        Constants = 4,
    }

    /// How the clamp before each call is recorded, and where its buffers
    /// are (the variants p to w).
    #[derive(Clone, Copy, PartialEq)]
    struct Clamping {
        /// The clamp's source records in the card's memory, not the PC's.
        source_on_card: bool,
        /// Its budget in the card's memory, not the PC's.
        budget_on_card: bool,
        barriers: Barriers,
        /// The benchmark's own root arguments ([`MeshRig::set_roots`]) set
        /// after the pipeline, to time them with the clamped call.
        bound: bool,
        /// The budget (in the card's memory) as a host list keeps it: reset
        /// before each list's first call and its words copied into the PC's
        /// memory after its last ([`super::reset_budget_on`],
        /// [`super::note_budget_on`]).
        per_list: bool,
    }

    /// The barriers around a clamp and its call.
    #[derive(Clone, Copy, PartialEq)]
    enum Barriers {
        /// As the runtime records them ([`super::record_clamp`] and
        /// [`super::done_on`]): a wait for every earlier shader write and
        /// the scratch buffer from common to unordered access; the pass; a
        /// wait, the scratch to indirect argument and the source back to
        /// common; after the call, the scratch back to common.
        Runtime,
        /// The same three transitions, without the two waits.
        NoWaits,
        /// The two waits, and two transitions a call instead of three: the
        /// scratch from indirect argument to unordered access with the
        /// first wait (from common for a batch's first call), and back to
        /// indirect argument with the second; none after the call (back to
        /// common once, after a batch's last). The least legal: the pass
        /// writes and the call reads the same bytes, and a buffer moves
        /// out of a state it was moved into only by a transition (implicit
        /// promotion is from common only, and a write state promoted into
        /// holds to the end of the list).
        TwoTransitions,
    }

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
        /// The adapter is the software one (`VICTORY_D3D12_WARP`).
        software: bool,
        /// Three 16-bit indices, 0 to 2, for (a)'s triangle drawn indexed.
        index: D3D12_INDEX_BUFFER_VIEW,
        indexed_signature: ID3D12CommandSignature,
        /// [`DRAWS`] indexed records of that triangle.
        indexed_args: ID3D12Resource,
        /// What the mesh variants need, or why there is none.
        mesh: Result<MeshRig, &'static str>,
    }

    /// The mesh variants' objects.
    struct MeshRig {
        list: ID3D12GraphicsCommandList6,
        root: ID3D12RootSignature,
        /// By [`Pipe`].
        pipelines: [ID3D12PipelineState; 5],
        signature: ID3D12CommandSignature,
        /// [`DRAWS`] mesh records of one group, then the count (1) at
        /// [`COUNT_AT`]: in the card's memory.
        args: ID3D12Resource,
        /// The runtime's clamp, made on this device.
        clamp: super::Clamp,
        /// As a list's scratch buffer for clamped records
        /// (`HostList::indirect_scratch`): the card's memory, 64 KiB.
        scratch: ID3D12Resource,
        /// [`Call::ClampOnce`]'s clamped copy of all the records, one a
        /// slot: the card's memory.
        clamped: ID3D12Resource,
        /// [`Call::Runs`]: the clamp of runs' pass made on this device, a work
        /// list a run for each of [`RUN_LENGTHS`] (the PC's memory, as the
        /// runtime's upload memory), and a slot a call (the card's memory).
        batch: super::Clamp,
        runs: Vec<ID3D12Resource>,
        run_slots: ID3D12Resource,
        /// The clamp's source records (those of `args`) and its budget: in
        /// the PC's memory (`Host::shared_buffer`), then in the card's.
        sources: [ID3D12Resource; 2],
        budgets: [ID3D12Resource; 2],
        /// For [`Clamping::per_list`]: zeros in upload memory to reset the
        /// budget in the card's memory from, and the PC-memory words its
        /// own are copied into.
        zeros: ID3D12Resource,
        notes: ID3D12Resource,
        /// [`Call::Constants`]' [`DRAWS`] slices of 256 bytes (zero): an
        /// upload buffer, then one in the card's memory.
        constant_buffers: [ID3D12Resource; 2],
        /// Two sets of the benchmark's own root arguments (alternated call
        /// by call): 6 constants; a constant view, a shader view and an
        /// unordered view (`views`, in that order); two tables.
        constants: [[u32; 6]; 2],
        views: [[u64; 3]; 2],
        tables: [[D3D12_GPU_DESCRIPTOR_HANDLE; 2]; 2],
    }

    impl MeshRig {
        /// Sets root arguments `set` ([`MESH`]'s root parameters 0 to 5, in
        /// their order).
        fn set_roots(&self, l: &ID3D12GraphicsCommandList, set: usize) {
            let views = self.views[set];
            // SAFETY: an open list with the mesh root signature; the views
            // name live buffers and the tables the view heap that is set.
            unsafe {
                l.SetGraphicsRootDescriptorTable(0, self.tables[set][0]);
                l.SetGraphicsRoot32BitConstants(1, 6, self.constants[set].as_ptr().cast(), 0);
                l.SetGraphicsRootShaderResourceView(2, views[1]);
                l.SetGraphicsRootConstantBufferView(3, views[0]);
                l.SetGraphicsRootDescriptorTable(4, self.tables[set][1]);
                l.SetGraphicsRootUnorderedAccessView(5, views[2]);
            }
        }
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    /// A buffer in the PC's memory that shaders read and write and the CPU
    /// maps, as `Host::shared_buffer` makes them (the runtime's copies of
    /// title buffers, and a list's budget), in the common state.
    fn shared(device: &ID3D12Device, size: u64) -> (ID3D12Resource, *mut u8) {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain buffer creation and a mapping kept for the buffer's
        // life.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out)
                .unwrap();
            let out = out.unwrap();
            out.Map(0, None, Some(&mut data)).unwrap();
            (out, data.cast())
        }
    }

    /// A buffer in the card's memory holding `words`, copied by `list`
    /// from an upload buffer kept in `kept` (buffers are in the common
    /// state between lists, and move to what a list reads them as).
    fn filled(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        kept: &mut Vec<ID3D12Resource>,
        words: &[u32],
        flags: D3D12_RESOURCE_FLAGS,
    ) -> ID3D12Resource {
        let bytes = std::mem::size_of_val(words) as u64;
        let upload = buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let target = buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, flags, D3D12_RESOURCE_STATE_COMMON);
        let mut data = std::ptr::null_mut();
        // SAFETY: an upload buffer of `bytes`, mapped while written; an
        // open list copying it.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), words.len());
            upload.Unmap(0, None);
            list.CopyBufferRegion(&target, 0, &upload, 0, bytes);
        }
        kept.push(upload);
        target
    }

    /// A one-argument command signature, as `crate::draw`'s
    /// `host_signature` makes them (no root arguments).
    fn signature(device: &ID3D12Device, kind: D3D12_INDIRECT_ARGUMENT_TYPE, stride: u32) -> Option<ID3D12CommandSignature> {
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut out: Option<ID3D12CommandSignature> = None;
        // SAFETY: a description of one argument.
        unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut out) }.ok()?;
        out
    }

    /// The PC's `dxc.exe`: `VICTORY_PC_DXC`, else the newest Windows SDK's
    /// (beside the validator the translator loads).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?.with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    /// Entry `entry` of `source` (a file) for `profile`; the compiler's
    /// complaint is printed if it refuses.
    fn dxil(dxc: &Path, source: &Path, entry: &str, profile: &str) -> Option<Vec<u8>> {
        let out = source.with_file_name(format!("{entry}.dxil"));
        let made = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", entry, "-Fo"])
            .arg(&out)
            .arg(source)
            .output()
            .ok()?;
        if !made.status.success() {
            eprintln!("our {entry} program did not compile: {}", String::from_utf8_lossy(&made.stderr).trim());
            return None;
        }
        std::fs::read(out).ok()
    }

    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

    #[repr(C)]
    struct MeshStream {
        root: Sub<*mut std::ffi::c_void>,
        amplification: Sub<D3D12_SHADER_BYTECODE>,
        mesh: Sub<D3D12_SHADER_BYTECODE>,
        pixel: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }

    /// A mesh pipeline of `stages` (amplification, mesh, pixel; an empty
    /// one absent): one colour target if it has a pixel program, the depth
    /// texture tested and, if `writes_depth`, written.
    fn mesh_pipeline(
        device: &ID3D12Device2,
        root: &ID3D12RootSignature,
        stages: [&[u8]; 3],
        writes_depth: bool,
    ) -> windows::core::Result<ID3D12PipelineState> {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: if bytes.is_empty() { std::ptr::null() } else { bytes.as_ptr().cast() },
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut targets = D3D12_RT_FORMAT_ARRAY::default();
        if !stages[2].is_empty() {
            targets.NumRenderTargets = 1;
            targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        let mut stream = MeshStream {
            root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
            amplification: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, code(stages[0])),
            mesh: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(stages[1])),
            pixel: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(stages[2])),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, u32::MAX),
            raster: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
            ),
            depth: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
                D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: if writes_depth { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO },
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
            ),
            targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, targets),
            dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_D32_FLOAT),
            samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }),
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<MeshStream>(),
            pPipelineStateSubobjectStream: (&raw mut stream).cast(),
        };
        // SAFETY: the stream holds whole subobjects pointing at live data.
        unsafe { device.CreatePipelineState(&desc) }
    }

    /// The mesh variants' objects on `device`, their buffers filled by
    /// `list` (open, run before the first batch); `views` is the view heap
    /// the batches set (its slots 1 and 2 hold views for the tables),
    /// `words` and `reads` buffers the root views name.
    fn mesh_rig(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        kept: &mut Vec<ID3D12Resource>,
        (views, words, reads): (&ID3D12DescriptorHeap, u64, u64),
    ) -> Result<MeshRig, &'static str> {
        const NONE: &str = "no mesh shaders";
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_5,
        };
        // SAFETY: each query writes its own structure of the given size.
        let supported = unsafe {
            device
                .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, (&raw mut options).cast(), size_of_val(&options) as u32)
                .is_ok()
                && device
                    .CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, (&raw mut model).cast(), size_of_val(&model) as u32)
                    .is_ok()
        };
        if !supported
            || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
            || model.HighestShaderModel.0 < D3D_SHADER_MODEL_6_5.0
        {
            return Err(NONE);
        }
        let device2: ID3D12Device2 = device.cast().map_err(|_| NONE)?;
        let list6: ID3D12GraphicsCommandList6 = list.cast().map_err(|_| NONE)?;

        let dxc = dxc().ok_or("no PC dxc.exe (VICTORY_PC_DXC or a Windows SDK)")?;
        let dir = tempfile::tempdir().map_err(|_| "no folder to compile in")?;
        let source = dir.path().join("mesh.hlsl");
        std::fs::write(&source, MESH).map_err(|_| "no folder to compile in")?;
        let program = |entry: &str, profile: &str| dxil(&dxc, &source, entry, profile).ok_or("our mesh programs did not compile");
        let (mesh, sent_mesh, amplify) = (program("mesh", "ms_6_5")?, program("sent_mesh", "ms_6_5")?, program("amplify", "as_6_5")?);
        let (texel, flat) = (program("texel", "ps_6_5")?, program("flat", "ps_6_5")?);
        let (read_mesh, read_texel) = (program("read_mesh", "ms_6_5")?, program("read_texel", "ps_6_5")?);

        // SAFETY: plain object creation on our device.
        let root: ID3D12RootSignature = unsafe { device.CreateRootSignature(0, &mesh[..]) }.map_err(|_| "no mesh root signature")?;
        let made = |stages: [&[u8]; 3], writes_depth: bool| {
            mesh_pipeline(&device2, &root, stages, writes_depth).map_err(|_| "a mesh pipeline was refused")
        };
        let pipelines = [
            made([&[], &mesh, &texel], false)?,
            made([&amplify, &sent_mesh, &texel], false)?,
            made([&[], &mesh, &[]], true)?,
            made([&[], &mesh, &flat], true)?,
            made([&[], &read_mesh, &read_texel], false)?,
        ];
        let signature = signature(device, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH, MESH_STRIDE).ok_or("no mesh command signature")?;
        let clamp = super::make_for(device).map_err(|_| "no clamp")?;
        let batch = super::make_batch_for(device).map_err(|_| "no clamp of runs")?;
        // Run j of a length takes records j * length on; its work list is
        // the pass's: the count, then each record's place.
        let runs = RUN_LENGTHS
            .iter()
            .map(|&length| {
                let each = super::HEAD + u64::from(length) * super::ENTRY;
                let (work, data) = shared(device, u64::from(DRAWS.div_ceil(length)) * each);
                for j in 0..DRAWS.div_ceil(length) {
                    let calls = length.min(DRAWS - j * length);
                    let mut list = vec![calls, 0, 0, 0];
                    for i in 0..calls {
                        list.extend([KIND_MESH << 8, (j * length + i) * MESH_STRIDE, MESH_STRIDE, 0]);
                    }
                    // SAFETY: inside the mapping: run j's list.
                    unsafe { std::ptr::copy_nonoverlapping(list.as_ptr(), data.add((u64::from(j) * each) as usize).cast::<u32>(), list.len()) };
                }
                work
            })
            .collect();

        // The records: one group each, padded to the stride, then the count.
        let mut record = vec![0u32; (MESH_STRIDE / 4) as usize];
        record[..3].copy_from_slice(&[1, 1, 1]);
        let records: Vec<u32> = (0..DRAWS).flat_map(|_| record.iter().copied()).collect();
        let mut with_count = records.clone();
        with_count.push(1);
        let args = filled(device, list, kept, &with_count, D3D12_RESOURCE_FLAG_NONE);
        let (in_pc, data) = shared(device, COUNT_AT);
        // SAFETY: the mapping holds `COUNT_AT` bytes, the records' own.
        unsafe { std::ptr::copy_nonoverlapping(records.as_ptr(), data.cast::<u32>(), records.len()) };
        let on_card = filled(device, list, kept, &records, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        // Budgets start at zero (committed memory is zeroed) and are never
        // zeroed again but by the variants that reset the card's once a
        // list: a record clamped charges one group (128), a batch 1.28
        // million, under a hundredth of the 2^27 past which records run
        // empty, and each budget serves at most fifteen batches here.
        let budgets = [
            shared(device, 256).0,
            buffer(device, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ];
        // Zero, as committed memory is made.
        let zeros = buffer(device, 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let notes = shared(device, 256).0;
        let on_card_uav = |bytes| {
            buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)
        };
        let (scratch, clamped) = (on_card_uav(64 << 10), on_card_uav(COUNT_AT.next_multiple_of(64 << 10)));
        let run_slots = on_card_uav(u64::from(DRAWS) * super::SLOT);
        // Zero, as committed memory is made.
        let slices = u64::from(DRAWS) * 256;
        let constant_buffers = [
            buffer(device, slices, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ),
            buffer(device, slices, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON),
        ];

        // SAFETY: plain queries of a live heap.
        let (table, step) = unsafe {
            (
                views.GetGPUDescriptorHandleForHeapStart(),
                u64::from(device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)),
            )
        };
        let slot = |n: u64| D3D12_GPU_DESCRIPTOR_HANDLE { ptr: table.ptr + n * step };
        Ok(MeshRig {
            list: list6,
            root,
            pipelines,
            signature,
            args,
            clamp,
            scratch,
            clamped,
            batch,
            runs,
            run_slots,
            sources: [in_pc, on_card],
            budgets,
            zeros,
            notes,
            constant_buffers,
            constants: [[0; 6], [1, 0, 0, 0, 0, 0]],
            views: [[reads, reads + 512, words], [reads + 256, reads + 768, words + 256]],
            tables: [[slot(1), slot(2)], [slot(2), slot(1)]],
        })
    }

    impl Rig {
        /// The adapter the layer takes (the one rule, `VICTORY_ADAPTER` and
        /// `VICTORY_D3D12_WARP` included); `None` without one, on a software
        /// adapter, or on one DXGI does not describe (a measurement of the
        /// card only; `VICTORY_REQUIRE_GPU` fails those instead).
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            let software = crate::host::on_warp();
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let switch = crate::host::on_warp().then_some("VICTORY_D3D12_WARP");
                let made = victory_shader::adapter::test_device(victory_shader::adapter::Asked::Hardware, switch).ok()?;
                let card = made.adapter.filter(|a| !a.software)?.description;
                let device = made.device;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    32,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                // A batch's start and end, and (`Call::ClampOnce`) the end
                // of what precedes its calls.
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 3,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (three views of
                // the small texture each: the first for (a)'s table, the
                // others for the mesh variants' two) and samplers (none
                // used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                let view_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
                for views in &view_heaps {
                    let mut at = views.GetCPUDescriptorHandleForHeapStart();
                    for _ in 0..3 {
                        device.CreateShaderResourceView(&image, None, at);
                        at.ptr += view_step;
                    }
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture
                // cleared, and the buffers below filled.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut kept = vec![first_target, second_target, image];

                // What the root views of the mesh variants name (never
                // read) and the indices of the indexed draws (at 2048).
                let reads = buffer(&device, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                let mut data = std::ptr::null_mut();
                reads.Map(0, None, Some(&mut data)).ok()?;
                data.cast::<u8>().add(2048).cast::<[u16; 3]>().write_unaligned([0, 1, 2]);
                reads.Unmap(0, None);
                let index = D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: reads.GetGPUVirtualAddress() + 2048,
                    SizeInBytes: 6,
                    Format: DXGI_FORMAT_R16_UINT,
                };
                let indexed_signature = signature(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, INDEXED_STRIDE)?;
                let records: Vec<u32> = (0..DRAWS).flat_map(|_| [3, 1, 0, 0, 0]).collect();
                let indexed_args = filled(&device, &list, &mut kept, &records, D3D12_RESOURCE_FLAG_NONE);
                let addresses = (&view_heaps[0], words.GetGPUVirtualAddress(), reads.GetGPUVirtualAddress());
                let mesh = mesh_rig(&device, &list, &mut kept, addresses);
                kept.push(reads);
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: kept,
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                    software,
                    index,
                    indexed_signature,
                    indexed_args,
                    mesh,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list of `call`s starts with: a new list has nothing
        /// set.
        fn begin(&self, call: Call) {
            self.begin_draws();
            let l = &self.list;
            let pipe = match call {
                Call::Draw(_) => return,
                Call::Indexed { .. } => {
                    // SAFETY: an open list; the view names a live buffer.
                    unsafe { l.IASetIndexBuffer(Some(&self.index)) };
                    return;
                }
                Call::Mesh { pipe, .. } => pipe,
                Call::Constants { .. } => Pipe::Constants,
                Call::MeshIndirect { .. } | Call::Clamped(_) | Call::ClampOnce | Call::Runs { .. } => Pipe::Plain,
            };
            let Ok(m) = &self.mesh else { return };
            // SAFETY: an open list and live objects and views; the depth
            // texture is in the write state between batches.
            unsafe {
                l.SetGraphicsRootSignature(&m.root);
                l.SetPipelineState(&m.pipelines[pipe as usize]);
                m.set_roots(l, 0);
                match pipe {
                    Pipe::DepthOnly => l.OMSetRenderTargets(0, None, false, Some(&self.depth_written)),
                    Pipe::DepthAndColour => {
                        l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_written));
                    }
                    Pipe::Plain | Pipe::Amplified | Pipe::Constants => {}
                }
            }
            if call == Call::ClampOnce {
                self.clamp_all(m);
            }
            // As a host list does where it first clamps (`HostList::budget`).
            if let Call::Clamped(how) = call
                && how.per_list
            {
                super::reset_budget_on(l, &m.budgets[1], &m.zeros);
            }
        }

        /// What every list of `call`s ends with, before its last timestamp:
        /// for [`Clamping::per_list`] the budget's words copied out, as a
        /// host list does where it closes (`HostList::close`).
        fn end(&self, call: Call) {
            if let (Call::Clamped(how), Ok(m)) = (call, &self.mesh)
                && how.per_list
            {
                super::note_budget_on(&self.list, &m.budgets[1], &m.notes);
            }
        }

        /// [`Call::ClampOnce`]'s work before its calls: one clamp pass over
        /// all the records into the copy (implicitly promoted from common to
        /// unordered access, as a buffer may be), one transition to
        /// indirect argument, the mesh pipeline again, and the timestamp
        /// that ends the clamp's time.
        fn clamp_all(&self, m: &MeshRig) {
            let l = &self.list;
            // SAFETY: live buffers.
            let budget = unsafe { m.budgets[1].GetGPUVirtualAddress() };
            let pass = super::pass(&m.clamped, budget, (&m.sources[1], 0), None, (KIND_MESH, MESH_STRIDE, DRAWS), 0)
                .expect("the clamp refused records of ours");
            pass.record(&m.clamp, l);
            let (unordered, argument) = (D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            // SAFETY: an open list; the copy, promoted to unordered access
            // by the pass; our pipeline and query heap.
            unsafe {
                l.ResourceBarrier(&[transition(&m.clamped, unordered, argument)]);
                l.SetPipelineState(&m.pipelines[Pipe::Plain as usize]);
                l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 2);
            }
        }

        /// What every list of (a)'s draws starts with.
        fn begin_draws(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// Why this rig cannot make `call`s, if it cannot.
        fn refuses(&self, call: Call) -> Option<&'static str> {
            match (&self.mesh, call) {
                (Err(why), call) if call.meshes() => Some(why),
                // The software adapter (Windows 11, 26200) removes its
                // device at the first such dispatch; the card runs the
                // title's mesh programs without a pixel stage.
                (_, Call::Mesh { pipe: Pipe::DepthOnly, .. }) if self.software => {
                    Some("the software adapter loses its device on a mesh pipeline without a pixel program")
                }
                _ => None,
            }
        }

        /// Records [`Call::Clamped`]'s call `n`: the clamp of record `n`
        /// into the scratch buffer at 0 under `how`'s barriers, the mesh
        /// pipeline again, `how`'s root arguments, the call.
        fn clamped(&self, m: &MeshRig, how: Clamping, n: u32) {
            let l = &self.list;
            let (common, unordered, argument) = (
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            );
            let source = (&m.sources[usize::from(how.source_on_card)], u64::from(n) * u64::from(MESH_STRIDE));
            let scratch = &m.scratch;
            let spec = (KIND_MESH, MESH_STRIDE, 1);
            // SAFETY: an open list with the mesh pipeline's state; live
            // buffers; the scratch buffer is in common before a batch and
            // after each call but under `TwoTransitions`, where it is in
            // indirect argument between calls.
            unsafe {
                let budget = m.budgets[usize::from(how.budget_on_card)].GetGPUVirtualAddress();
                let pass = || super::pass(scratch, budget, source, None, spec, 0).expect("the clamp refused a record of ours");
                match how.barriers {
                    Barriers::Runtime => {
                        let clamped = super::record_clamp(&m.clamp, l, scratch, budget, source, None, spec);
                        assert!(clamped, "the clamp refused a record of ours");
                    }
                    Barriers::NoWaits => {
                        let pass = pass();
                        l.ResourceBarrier(&[transition(scratch, common, unordered)]);
                        pass.record(&m.clamp, l);
                        l.ResourceBarrier(&[transition(scratch, unordered, argument)]);
                    }
                    Barriers::TwoTransitions => {
                        let pass = pass();
                        let before = if n == 0 { common } else { argument };
                        l.ResourceBarrier(&[super::uav_barrier(), transition(scratch, before, unordered)]);
                        pass.record(&m.clamp, l);
                        l.ResourceBarrier(&[super::uav_barrier(), transition(scratch, unordered, argument)]);
                    }
                }
                l.SetPipelineState(&m.pipelines[Pipe::Plain as usize]);
                if how.bound {
                    m.set_roots(l, (n % 2) as usize);
                }
                l.ExecuteIndirect(&m.signature, 1, scratch, 0, None, 0);
                if how.barriers != Barriers::TwoTransitions || n + 1 == DRAWS {
                    super::done_on(l, scratch);
                }
            }
        }

        /// Records [`Call::Runs`]' call `n`: at each run's first call the
        /// run's pass (the slots from common, or from indirect argument after
        /// the batch's first run) and the mesh pipeline again; then (t)'s root
        /// arguments and the call from its slot; after the batch's last, the
        /// slots back to common.
        fn run_call(&self, m: &MeshRig, length: u32, n: u32) {
            let l = &self.list;
            let (common, argument) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            let at = u64::from(n) * super::SLOT;
            // SAFETY: an open list with the mesh pipeline's state; live
            // buffers; the work lists hold every run's records.
            unsafe {
                if n.is_multiple_of(length) {
                    let place = RUN_LENGTHS.iter().position(|&x| x == length).expect("a length of RUN_LENGTHS");
                    let each = super::HEAD + u64::from(length) * super::ENTRY;
                    let work = m.runs[place].GetGPUVirtualAddress() + u64::from(n / length) * each;
                    let from = if n == 0 { common } else { argument };
                    let (budget, src) = (m.budgets[1].GetGPUVirtualAddress(), m.sources[0].GetGPUVirtualAddress());
                    let dst = m.run_slots.GetGPUVirtualAddress() + at;
                    let barriers: Option<ID3D12GraphicsCommandList7> = windows::core::Interface::cast(l).ok();
                    let reads = super::Reads::new(std::slice::from_ref(&m.sources[0]), barriers.as_ref());
                    super::record_batch(&m.batch, l, (&m.run_slots, from), budget, dst, work, ([src; 4], &reads));
                    l.SetPipelineState(&m.pipelines[Pipe::Plain as usize]);
                }
                m.set_roots(l, (n % 2) as usize);
                l.ExecuteIndirect(&m.signature, 1, &m.run_slots, at, None, 0);
                if n + 1 == DRAWS {
                    l.ResourceBarrier(&[transition(&m.run_slots, argument, common)]);
                }
            }
        }

        /// Records call `n` of a batch of `call`s, after [`Rig::begin`].
        fn record(&self, call: Call, n: u32) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let l = &self.list;
            let odd = (n % 2) as usize;
            // SAFETY: an open list with everything set; live objects; the
            // records read are those of call `n`, inside their buffers.
            // Every state moved is moved back within the batch (the draws
            // of a list are an even number).
            unsafe {
                let between = match call {
                    Call::Draw(between) => between,
                    Call::Indexed { indirect: false } => return l.DrawIndexedInstanced(3, 1, 0, 0, 0),
                    Call::Indexed { indirect: true } => {
                        let at = u64::from(n) * u64::from(INDEXED_STRIDE);
                        return l.ExecuteIndirect(&self.indexed_signature, 1, &self.indexed_args, at, None, 0);
                    }
                    Call::Mesh { groups, bound, .. } => {
                        let Ok(m) = &self.mesh else { return };
                        if bound {
                            m.set_roots(l, odd);
                        }
                        return m.list.DispatchMesh(groups, 1, 1);
                    }
                    Call::MeshIndirect { count } => {
                        let Ok(m) = &self.mesh else { return };
                        let at = u64::from(n) * u64::from(MESH_STRIDE);
                        // Without a count buffer its offset must be 0.
                        let (counted, count_at) = if count { (Some(&m.args), COUNT_AT) } else { (None, 0) };
                        return l.ExecuteIndirect(&m.signature, 1, &m.args, at, counted, count_at);
                    }
                    Call::Clamped(how) => {
                        let Ok(m) = &self.mesh else { return };
                        return self.clamped(m, how, n);
                    }
                    Call::Runs { length } => {
                        let Ok(m) = &self.mesh else { return };
                        return self.run_call(m, length, n);
                    }
                    Call::ClampOnce => {
                        let Ok(m) = &self.mesh else { return };
                        let at = u64::from(n) * u64::from(MESH_STRIDE);
                        l.ExecuteIndirect(&m.signature, 1, &m.clamped, at, None, 0);
                        if n + 1 == DRAWS {
                            let (argument, common) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
                            l.ResourceBarrier(&[transition(&m.clamped, argument, common)]);
                        }
                        return;
                    }
                    Call::Constants { card } => {
                        let Ok(m) = &self.mesh else { return };
                        let slice = m.constant_buffers[usize::from(card)].GetGPUVirtualAddress() + u64::from(n) * 256;
                        l.SetGraphicsRootConstantBufferView(3, slice);
                        return m.list.DispatchMesh(1, 1, 1);
                    }
                };
                match between {
                    Between::Nothing => {}
                    Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                    Between::BarrierPair => {
                        l.ResourceBarrier(&[transition(&self.other, target, read)]);
                        l.ResourceBarrier(&[transition(&self.other, read, target)]);
                    }
                    Between::DepthMove => {
                        let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                        l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                    }
                    Between::Heaps => {
                        l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                        l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                    }
                    Between::Targets => {
                        l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                    }
                    Between::Compute | Between::ComputeWithWaits => {
                        let waits = between == Between::ComputeWithWaits;
                        if waits {
                            l.ResourceBarrier(&[super::uav_barrier()]);
                        }
                        l.SetComputeRootSignature(&self.work_root);
                        l.SetPipelineState(&self.work);
                        l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                        l.Dispatch(1, 1, 1);
                        if waits {
                            l.ResourceBarrier(&[super::uav_barrier()]);
                        }
                        l.SetPipelineState(&self.pipelines[0]);
                    }
                }
                l.DrawInstanced(3, 1, 0, 0);
            }
        }

        /// One batch of [`DRAWS`] `call`s in `lists` lists run one by one:
        /// the microseconds a call took on the card, the milliseconds
        /// recording took, and for [`Call::ClampOnce`] the microseconds its
        /// work before the calls took (the calls' time is from its end).
        fn batch(&mut self, call: Call, lists: u32) -> (f64, f64, Option<f64>) {
            let up_front = call == Call::ClampOnce;
            let stamped = if up_front { 3 } else { 2 };
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin(call);
                for _ in 0..DRAWS / lists {
                    self.record(call, drawn);
                    drawn += 1;
                }
                self.end(call);
                // SAFETY: an open list; our queue, query heap and read-back
                // buffer.
                unsafe {
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, stamped, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let bytes = 8 * stamped as usize;
            let range = D3D12_RANGE { Begin: 0, End: bytes };
            // SAFETY: a read-back buffer the finished list resolved
            // `stamped` timestamps into.
            let stamps = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), stamped as usize).to_vec();
                self.back.Unmap(0, None);
                stamps
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            let us = |from: u64, to: u64| to.saturating_sub(from) as f64 * 1e6 / frequency as f64;
            let (calls_from, before) = if up_front {
                (stamps[2], Some(us(stamps[0], stamps[2])))
            } else {
                (stamps[0], None)
            };
            (us(calls_from, stamps[1]) / f64::from(drawn.max(1)), recording, before)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        let software = if rig.software { " (the software adapter: its figures mean nothing)" } else { "" };
        eprintln!(
            "bare command cost on {}{software}: {DRAWS} calls a batch into {WIDTH}x{HEIGHT}, the fastest of three batches",
            rig.card
        );
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Call::Draw(Between::Nothing), 1);
        let draw = Call::Draw;
        let mesh = |pipe, groups, bound| Call::Mesh { pipe, groups, bound };
        let clamped = |source_on_card, budget_on_card, barriers, bound| {
            Call::Clamped(Clamping { source_on_card, budget_on_card, barriers, bound, per_list: false })
        };
        // A title call as the runtime records it since its budget is in the
        // card's memory: (p1), its budget reset and read back once a list.
        let runtime = |bound| {
            Call::Clamped(Clamping {
                source_on_card: false,
                budget_on_card: true,
                barriers: Barriers::Runtime,
                bound,
                per_list: true,
            })
        };
        let variants = [
            ("a", "nothing changes between draws", draw(Between::Nothing), 1),
            ("b", "another pipeline every draw", draw(Between::Pipeline), 1),
            ("c", "a barrier pair on a second texture every draw", draw(Between::BarrierPair), 1),
            ("d", "the tested depth texture moved between write and read every draw", draw(Between::DepthMove), 1),
            ("e", "another pair of descriptor heaps every draw", draw(Between::Heaps), 1),
            ("f", "another colour target every draw", draw(Between::Targets), 1),
            ("g", "nothing changes, in 100 lists run one by one", draw(Between::Nothing), 100),
            ("h", "a compute pass of ours of one group before every draw", draw(Between::Compute), 1),
            ("i", "the same pass between two waits for earlier shader writes", draw(Between::ComputeWithWaits), 1),
            (
                "j",
                "a mesh program (one group, one triangle) with (a)'s pixel program, no amplification: DispatchMesh(1, 1, 1)",
                mesh(Pipe::Plain, 1, false),
                1,
            ),
            ("k", "(j) behind an amplification program of one group that launches one mesh group", mesh(Pipe::Amplified, 1, false), 1),
            (
                "l",
                "(j)'s call through ExecuteIndirect (mesh signature, records 16 bytes apart), one record a call, the next record of one buffer on the card",
                Call::MeshIndirect { count: false },
                1,
            ),
            ("m", "(l) with a count buffer on the card holding 1", Call::MeshIndirect { count: true }, 1),
            ("n", "(a)'s triangle drawn indexed, direct: DrawIndexedInstanced(3, 1, 0, 0, 0)", Call::Indexed { indirect: false }, 1),
            (
                "o",
                "(n) through ExecuteIndirect (indexed signature, 20-byte records), one record a call, the next record of one buffer",
                Call::Indexed { indirect: true },
                1,
            ),
            (
                "p",
                "(l) as the runtime issues a title call: our clamp of the record (wait for shader writes, scratch common to unordered, clamp root signature and pipeline, 6 constants, 4 root views, Dispatch(1, 1, 1), wait, scratch to indirect argument), (j)'s pipeline again, the call from the scratch buffer, scratch back to common; source and budget in the PC's memory, as the runtime's were before its budget went to the card",
                clamped(false, false, Barriers::Runtime, false),
                1,
            ),
            ("q", "(p) with the clamp's source records and budget in the card's memory", clamped(true, true, Barriers::Runtime, false), 1),
            (
                "p1",
                "(p) with only the clamp's budget in the card's memory (source records in the PC's)",
                clamped(false, true, Barriers::Runtime, false),
                1,
            ),
            (
                "p2",
                "(p) with only the clamp's source records in the card's memory (budget in the PC's)",
                clamped(true, false, Barriers::Runtime, false),
                1,
            ),
            (
                "p3",
                "(q) without the clamp's two waits for shader writes (its three transitions stay)",
                clamped(true, true, Barriers::NoWaits, false),
                1,
            ),
            (
                "p4",
                "(q) with two transitions a call, not three: scratch indirect argument (common for the first call) to unordered with the first wait, unordered to indirect argument with the second, none after the call (back to common once, after the last)",
                clamped(true, true, Barriers::TwoTransitions, false),
                1,
            ),
            (
                "p5",
                "(p1) as the runtime records it now: the budget in the card's memory, zeroed there by a copy and a transition before the list's first call, its words copied into the PC's memory after its last (source records in the PC's)",
                runtime(false),
                1,
            ),
            ("p6", "(p5) in 100 lists run one by one, each with its own reset and copy of the budget", runtime(false), 100),
            ("p7", "(p1) in 100 lists run one by one, without them: against (p6), what a list's reset and copy cost", clamped(false, true, Barriers::Runtime, false), 100),
            (
                "r",
                "depth only: (j)'s mesh program, no pixel program, no colour target, depth tested and written: DispatchMesh(1, 1, 1)",
                mesh(Pipe::DepthOnly, 1, false),
                1,
            ),
            ("s", "(r) with a pixel program writing a constant colour into a colour target", mesh(Pipe::DepthAndColour, 1, false), 1),
            (
                "t",
                "(j) with 2 descriptor tables, 6 root constants and 3 root views (shader, constant, unordered) set before every call, in the root signature's order (two sets, alternated)",
                mesh(Pipe::Plain, 1, true),
                1,
            ),
            ("u", "(j) with 64 groups a call: DispatchMesh(64, 1, 1)", mesh(Pipe::Plain, 64, false), 1),
            ("v", "(j) with 1,024 groups a call: DispatchMesh(1024, 1, 1)", mesh(Pipe::Plain, 1024, false), 1),
            (
                "w",
                "(p) with (t)'s root arguments set after the pipeline: all we issued for a title call with the budget in the PC's memory",
                clamped(false, false, Barriers::Runtime, true),
                1,
            ),
            ("w1", "(p5) with (t)'s root arguments set after the pipeline: all we issue for a title call now", runtime(true), 1),
            (
                "x",
                "clamp once, draw many: before the calls one clamp pass of 157 groups over all 10,000 records (source and budget on the card) into a copy with one record a slot, one transition to indirect argument, the pipeline again; then (l)'s call reading the next record of the copy, nothing between",
                Call::ClampOnce,
                1,
            ),
            (
                "x1",
                "the clamp of runs (the default), runs of 1 call: before each run's first call one pass (a wait, the slots to unordered access, one group summing the run's records' costs in call order, a wait, the slots to indirect argument) and the pipeline again; every call (t)'s root arguments and the call from its own slot, nothing between; source records in the PC's memory, budget on the card: against (w1)",
                Call::Runs { length: 1 },
                1,
            ),
            ("x16", "(x1) in runs of 16 calls", Call::Runs { length: 16 }, 1),
            ("x64", "(x1) in runs of 64 calls", Call::Runs { length: 64 }, 1),
            ("x256", "(x1) in runs of 256 calls", Call::Runs { length: 256 }, 1),
            (
                "y",
                "(j) whose mesh and pixel programs read their constant view, set before every call to the next 256-byte slice of an upload buffer",
                Call::Constants { card: false },
                1,
            ),
            ("z", "(y) with the constant buffer in the card's memory", Call::Constants { card: true }, 1),
        ];
        let mut least = Vec::new();
        for (letter, what, call, lists) in variants {
            if let Some(why) = rig.refuses(call) {
                eprintln!("({letter}) {what}: skipped: {why}");
                continue;
            }
            let mut runs: Vec<(f64, f64, Option<f64>)> = (0..3).map(|_| rig.batch(call, lists)).collect();
            let mut before: Vec<f64> = runs.iter().filter_map(|run| run.2).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            before.sort_by(f64::total_cmp);
            let before = match before[..] {
                [least, second, third] => {
                    format!("; the clamp before them {least:.2} microseconds (the three: {least:.2}, {second:.2}, {third:.2})")
                }
                _ => String::new(),
            };
            eprintln!(
                "({letter}) {what}: {:.2} microseconds a call (the three: {:.2}, {:.2}, {:.2}){before}; recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
