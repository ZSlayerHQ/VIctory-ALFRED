//! End-to-end tests of 16-bit typed buffer loads (`buffer_load_format_d16_*`)
//! through views of each kind of 16-bit format: synthetic GFX10 programs
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! software adapter (WARP) only.
//!
//! On GFX10 a typed load converts by the format of the buffer's descriptor
//! (the same value `buffer_load_format_*` gives a 32-bit register), and
//! `_d16` packs each channel into 16 bits: a float or normalized value as a
//! half, an integer as its low 16 bits. So through a 16-bit UINT, SINT or
//! FLOAT view the element's own 16 bits come back; through a UNORM or SNORM
//! view, the half of the normalized value.
//!
//! Each program has one thread per element: it clears v1 (and v2), loads
//! element `thread` through the typed view at table entry 1, and stores the
//! registers as dwords through the raw UAV at entry 0. (The register is
//! cleared first so that the test says nothing of what a load of an odd
//! number of channels leaves in the last high half.)

use windows::Win32::Graphics::Dxgi::Common::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset`.
const fn s_load_dwordx4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (offset & 0x1F_FFFF),
    ]
}

/// VOP1 `v_mov_b32 v<vdst>, 0`.
const fn v_clear(vdst: u8) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | 0x80
}

/// VOP2 `v_lshlrev_b32 v<vdst>, <shift>, v<vsrc>`.
const fn v_lshlrev_b32(vdst: u8, shift: u8, vsrc: u8) -> u32 {
    0x1A << 25 | (vdst as u32) << 17 | (vsrc as u32) << 9 | 0x80 | shift as u32
}

/// MUBUF `op v<vdata>, v<vaddr>, s[srsrc:srsrc+3], 0 idxen`.
const fn mubuf_idxen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// MUBUF `op v<vdata>, v<vaddr>, s[srsrc:srsrc+3], 0 offen offset:byte_off`.
const fn mubuf_offen(op: u32, vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const BUFFER_STORE_DWORD: u32 = 0x1C;
const BUFFER_LOAD_FORMAT_D16_X: u32 = 0x80;
const BUFFER_LOAD_FORMAT_D16_XYZW: u32 = 0x83;

/// The 16-bit patterns the views hold, element by element: one, the largest
/// positive integer, the sign bit alone, all ones; then, for the float view,
/// a quiet NaN, a signalling NaN, the largest denormal and the smallest
/// negative denormal.
const PATTERNS: [u16; 8] = [0x0001, 0x7FFF, 0x8000, 0xFFFF, 0x7E00, 0x7C01, 0x03FF, 0x8001];

/// [`PATTERNS`] as the buffer's dwords, low half first.
fn pattern_dwords() -> Vec<u32> {
    PATTERNS.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16).collect()
}

/// The program of the module's description, `threads` threads wide, loading
/// with `op` (`_d16_x` into v1, or `_d16_xyzw` into v1 and v2) and storing
/// v1 at dword `thread` (`_x`) or v1 and v2 at dwords `2 * thread` on.
fn program(op: u32, threads: u32) -> HostShader {
    translated(&native_of(op, threads))
}

/// The console program [`program`] translates.
fn native_of(op: u32, threads: u32) -> NativeProgram {
    let wide = op == BUFFER_LOAD_FORMAT_D16_XYZW;
    let mut words = vec![s_mov_b32(1, 0x80)]; // s[0:1]: the table
    words.extend_from_slice(&s_load_dwordx4(8, 0, 0)); // entry 0: the raw UAV
    words.extend_from_slice(&s_load_dwordx4(12, 0, 32)); // entry 1: the typed view
    words.extend([v_clear(1), v_clear(2)]);
    words.extend_from_slice(&mubuf_idxen(op, 1, 0, 12)); // v1 (v2) = element v0
    words.push(v_lshlrev_b32(3, if wide { 3 } else { 2 }, 0));
    words.extend_from_slice(&mubuf_offen(BUFFER_STORE_DWORD, 1, 3, 8, 0));
    if wide {
        words.extend_from_slice(&mubuf_offen(BUFFER_STORE_DWORD, 2, 3, 8, 4));
    }
    words.push(S_ENDPGM);

    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: [(ResourceType::Uav, 0), (ResourceType::Srv, 8)]
            .map(|(kind, table_offset_dwords)| UsedResource {
                kind,
                root: RootType::Table,
                table_offset_dwords,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            })
            .to_vec(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // user SGPRs: s0
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A d16 load whose descriptor the program loads from the view heap at an
/// index it computes (`ResourceHeap`, s0 the heap's address): the heap's
/// first descriptor, read at element v0 into v4.
fn heap_native() -> NativeProgram {
    let mut words = vec![s_mov_b32(1, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 0, 0));
    words.push(v_clear(4));
    words.extend_from_slice(&mubuf_idxen(BUFFER_LOAD_FORMAT_D16_X, 4, 0, 8));
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A d16 load through a descriptor an `s_cselect` picks between two of a
/// table's (the table at s0, the choice by the root constant s1), stored to
/// LDS: the descriptor is no load outright.
fn selected_native() -> NativeProgram {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE200_2000,
        0x8004_0100, // buffer_load_format_d16_x v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let param = |root, user_data: u8| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Table, 0), param(RootType::Constants, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1 | 1 << 15, threads: [32, 1, 1] }), // s0, s1; 512 bytes of LDS
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` translated (and validated by `dxil.dll` when present).
fn translated(native: &NativeProgram) -> HostShader {
    let native = native.clone();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs [`program`] on the software adapter with [`PATTERNS`] behind a typed
/// view of `format` (elements of `bytes` bytes) and gives the UAV's `dwords`
/// dwords; None, and a line, without the software adapter.
fn loaded(op: u32, format: DXGI_FORMAT, bytes: u32, threads: u32, dwords: u32) -> Option<Vec<u32>> {
    let shader = program(op, threads);
    let mut roots = Vec::new();
    if super::d16_integer_views_on() {
        // With the switch on the load reads its view's kind from the
        // host's slot table: the fixture binds one, as the host does.
        assert!(
            shader.root_cbvs.is_empty() && shader.root_srvs == [super::SLOT_STRIDES] && shader.root_uavs.is_empty(),
            "with VICTORY_D16_INTEGER_VIEWS=1 the program reads the slot table and no other root view: {:?} {:?} {:?}",
            shader.root_cbvs,
            shader.root_srvs,
            shader.root_uavs
        );
        roots.push((1, Root::View { srv: true, data: slot_table(&[(1, slot_word(format, bytes))]) }));
    } else {
        assert!(
            shader.root_cbvs.is_empty() && shader.root_srvs.is_empty() && shader.root_uavs.is_empty(),
            "the program asks for a root view this fixture does not bind: {:?} {:?} {:?}",
            shader.root_cbvs,
            shader.root_srvs,
            shader.root_uavs
        );
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no SM 6.6 warp device; skipped");
        return None;
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    // The table starts at heap index 0.
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        assert_eq!(u.root, RootType::Table, "{u:?}");
        ud[u.index as usize] = 0;
    }
    roots.insert(0, (0, Root::Constants(ud)));
    let data = pattern_dwords();
    let views = [View::RawUav(dwords), View::Buffer { data: &data, stride: Stride::Format(format, bytes) }];
    let dispatch = Dispatch { roots, views: &views, samplers: &[], groups: [1, 1, 1] };
    let out = host.dispatch(&root, &pso, &dispatch).unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

/// The slot table's dword the host writes, with the switch on, for a typed
/// buffer view of `format` (elements of `bytes` bytes): the stride, and an
/// integer format's kind bits.
fn slot_word(format: DXGI_FORMAT, bytes: u32) -> u32 {
    use super::{VIEW_INTEGER, VIEW_SIGNED};
    let kind = match format {
        DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16G16B16A16_UINT => VIEW_INTEGER,
        DXGI_FORMAT_R16_SINT => VIEW_INTEGER | VIEW_SIGNED,
        _ => 0,
    };
    bytes | kind
}

/// A slot table of 64 dwords: these slots' words, the others 0 (slot 0's
/// raw view among them: no kind).
fn slot_table(words: &[(usize, u32)]) -> Vec<u32> {
    let mut table = vec![0u32; 64];
    for &(slot, word) in words {
        table[slot] = word;
    }
    table
}

/// `f` with `VICTORY_D16_INTEGER_VIEWS=1` for this thread.
fn with_kinds<T>(f: impl FnOnce() -> T) -> T {
    let _on = super::switched(&super::TEST_D16_INTEGER_VIEWS, true);
    f()
}

/// The lines of a comparison, one per element: the pattern, what was read,
/// what the console reads.
fn table(got: &[u32], want: &[u32]) -> String {
    let line = |(k, (g, w)): (usize, (&u32, &u32))| {
        let mark = if g == w { "" } else { "  <-- differs" };
        format!("\n  element {k} ({:#06x}): read {g:#010x}, the console reads {w:#010x}{mark}", PATTERNS[k])
    };
    got.iter().zip(want).enumerate().map(line).collect()
}

/// A 16_UINT view: the element's 16 bits, as they are, with
/// `VICTORY_D16_INTEGER_VIEWS=1`. Without the switch every element reads 0
/// (WARP, [`off_a_d16_load_through_an_integer_view_reads_what_world_reads`]):
/// the lowering reads the view through a float-typed handle and narrows with
/// `legacyF32ToF16`, and the integer's bits arrive as a float denormal,
/// whose half is 0.
#[test]
fn a_d16_load_through_a_uint16_view_reads_the_elements_bits() {
    let Some(got) = with_kinds(|| loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_UINT, 2, 8, 8)) else { return };
    let want: Vec<u32> = PATTERNS.iter().map(|&p| u32::from(p)).collect();
    assert_eq!(got, want, "{}", table(&got, &want));
}

/// A 16_16_16_16_UINT view, all four channels (two registers): each
/// channel's 16 bits, two to a register, low half first, with the switch.
/// Without it all four dwords read 0, for the reason of the one-channel
/// test.
#[test]
fn a_d16_load_of_four_channels_through_a_uint16_view_reads_the_elements_bits() {
    let Some(got) = with_kinds(|| loaded(BUFFER_LOAD_FORMAT_D16_XYZW, DXGI_FORMAT_R16G16B16A16_UINT, 8, 2, 4)) else {
        return;
    };
    let want = pattern_dwords();
    assert_eq!(got, want, "read {got:#010x?}, the console reads {want:#010x?}");
}

/// A 16_SINT view: the sign-extended integer's low 16 bits, the element's
/// own bits again, with the switch. Without it (WARP) a positive element
/// reads 0 and a negative one 0xFFFF: the sign-extended integer's bits
/// arrive as a float, a denormal (half 0) or a negative NaN (half 0xFFFF on
/// WARP).
#[test]
fn a_d16_load_through_a_sint16_view_reads_the_elements_bits() {
    let Some(got) = with_kinds(|| loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_SINT, 2, 8, 8)) else { return };
    let want: Vec<u32> = PATTERNS.iter().map(|&p| u32::from(p)).collect();
    assert_eq!(got, want, "{}", table(&got, &want));
}

/// Off is off, by what is read: without the switch an integer view's d16
/// load reads what `world` reads (WARP), the zeros of the faces bug: a
/// 16_UINT view 0, one channel or four; a 16_SINT view 0 for a positive
/// element and 0xFFFF for a negative one.
#[test]
fn off_a_d16_load_through_an_integer_view_reads_what_world_reads() {
    let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_UINT, 2, 8, 8) else { return };
    assert_eq!(got, [0; 8], "R16_UINT, switch off: {}", table(&got, &[0; 8]));
    let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_XYZW, DXGI_FORMAT_R16G16B16A16_UINT, 8, 2, 4) else { return };
    assert_eq!(got, [0; 4], "R16G16B16A16_UINT, switch off");
    let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_SINT, 2, 8, 8) else { return };
    let want: Vec<u32> = PATTERNS.iter().map(|&p| if p & 0x8000 != 0 { 0xFFFF } else { 0 }).collect();
    assert_eq!(got, want, "R16_SINT, switch off: {}", table(&got, &want));
}

/// With the switch a d16 load reads its view's kind from the slot table
/// whatever picked the descriptor: loaded from the view heap at an index
/// the program computes, or picked by `s_cselect` between two of a table's
/// (our own fixtures on job/fork-accuracy-r2, 6514f1b89 and dbc6ecd74).
/// Without it the selected one does not ask for the table; the heap-picked
/// one does on `world` too (every access picked from the heap reads it, for
/// its stride and to have the host ready the heap's buffers), so there only
/// the module tells the two apart. Off, both modules are world's
/// ([`off_each_d16_program_emits_worlds_module`]).
#[test]
fn a_d16_load_reads_the_view_kind_whatever_picked_the_descriptor() {
    for (name, native, world_reads) in [("heap-picked", heap_native(), true), ("selected", selected_native(), false)] {
        let off = translated(&native);
        assert_eq!(off.root_srvs.contains(&super::SLOT_STRIDES), world_reads, "{name}, switch off: {:?}", off.root_srvs);
        let on = with_kinds(|| translated(&native));
        assert_ne!(on.dxil, off.dxil, "{name}: the switch changes the module");
        assert!(
            on.root_srvs.contains(&super::SLOT_STRIDES),
            "{name}, switch on: the kind is read from the slot table: {:?}",
            on.root_srvs
        );
    }
}

/// A 16_UNORM view: the half of `x / 65535`. 1/65535 is 256.004 steps of
/// the smallest half (0x0100), 32768/65535 is just above 0.5 (0x3800) and
/// 65535/65535 is 1.0 (0x3C00), whichever way the narrowing rounds.
/// 32767/65535 lies between the halves 0x37FF and 0x3800 (0.5), nearer the
/// second: which of the two the console gives is its rounding rule for a
/// formatted `_d16` load, not established natively, so either is taken here
/// and the one read is printed.
#[test]
fn a_d16_load_through_a_unorm16_view_reads_the_half_of_the_normalized_value() {
    // With the switch off and on: a normalized view keeps its conversion.
    for on in [false, true] {
        let _on = super::switched(&super::TEST_D16_INTEGER_VIEWS, on);
        let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_UNORM, 2, 4, 4) else { return };
        eprintln!("unorm16 0x7fff as a half (switch {on}): {:#06x} (0x37ff truncated, 0x3800 to nearest)", got[1]);
        let mut want = vec![0x0100, 0x3800, 0x3800, 0x3C00];
        if got[1] == 0x37FF {
            want[1] = 0x37FF;
        }
        assert_eq!(got, want, "switch {on}: {}", table(&got, &want));
    }
}

/// A 16_SNORM view: the half of `max(x, -32767) / 32767`. 1/32767 is 512.016
/// steps of the smallest half (0x0200, and 0x8200 for -1/32767); 32767 is
/// 1.0 (0x3C00) and -32768 clamps to -1.0 (0xBC00): none depends on how the
/// narrowing rounds.
#[test]
fn a_d16_load_through_a_snorm16_view_reads_the_half_of_the_normalized_value() {
    for on in [false, true] {
        let _on = super::switched(&super::TEST_D16_INTEGER_VIEWS, on);
        let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_SNORM, 2, 4, 4) else { return };
        let want = [0x0200, 0x3C00, 0xBC00, 0x8200];
        assert_eq!(got, want, "switch {on}: {}", table(&got, &want));
    }
}

/// A 16_FLOAT view: the half itself. Widening a half to 32 bits and
/// narrowing it again is exact for every number, the denormals and -0
/// among them, so those are compared bit for bit. What the console leaves of
/// a NaN's payload on the way (and whether it quiets a signalling one) is
/// not established natively: a NaN must come back a NaN of the same sign,
/// and the bits read are printed.
#[test]
fn a_d16_load_through_a_float16_view_reads_the_half() {
    for on in [false, true] {
        let _on = super::switched(&super::TEST_D16_INTEGER_VIEWS, on);
        let Some(got) = loaded(BUFFER_LOAD_FORMAT_D16_X, DXGI_FORMAT_R16_FLOAT, 2, 8, 8) else { return };
        let nan = |h: u32| h & 0x7C00 == 0x7C00 && h & 0x03FF != 0;
        let mut want: Vec<u32> = PATTERNS.iter().map(|&p| u32::from(p)).collect();
        for (k, w) in want.iter_mut().enumerate() {
            if nan(*w) {
                eprintln!("float16 NaN {w:#06x} read as {:#06x} (switch {on})", got[k]);
                if nan(got[k]) && got[k] >> 15 == *w >> 15 {
                    *w = got[k];
                }
            }
        }
        assert_eq!(got, want, "switch {on}: {}", table(&got, &want));
    }
}

/// Every d16 program of these fixtures, by name: the six views' three
/// programs, the heap-picked load and the selected one.
fn d16_programs() -> [(&'static str, NativeProgram); 5] {
    [
        ("d16_x, 8 threads", native_of(BUFFER_LOAD_FORMAT_D16_X, 8)),
        ("d16_x, 4 threads", native_of(BUFFER_LOAD_FORMAT_D16_X, 4)),
        ("d16_xyzw, 2 threads", native_of(BUFFER_LOAD_FORMAT_D16_XYZW, 2)),
        ("d16_x, heap-picked", heap_native()),
        ("d16_x, selected", selected_native()),
    ]
}

/// The first 16 hex digits of the BLAKE3 hash of `bytes`.
fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex()[..16].to_owned()
}

/// Off is off, by bytes: with `VICTORY_D16_INTEGER_VIEWS` unset (and the
/// test's thread-local untouched) each d16 program emits `world`'s module,
/// byte for byte. The digests were taken from `world`'s translator
/// (`dc3d04bb5`, which this branch's first commit carries unchanged) by
/// this test printing them before any translator change.
#[test]
fn off_each_d16_program_emits_worlds_module() {
    const WORLD: [(&str, &str); 5] = [
        ("d16_x, 8 threads", "6b38243baad9b011"),
        ("d16_x, 4 threads", "32325160d4de7a1a"),
        ("d16_xyzw, 2 threads", "c3c2b32a9d1bf6ab"),
        ("d16_x, heap-picked", "52ed2f8a1086d037"),
        ("d16_x, selected", "3e1b1ebd73c5b67f"),
    ];
    let got: Vec<(&str, String)> =
        d16_programs().iter().map(|(name, native)| (*name, digest(&translated(native).dxil))).collect();
    for (name, d) in &got {
        eprintln!("{name}: {d}");
    }
    let want: Vec<(&str, String)> = WORLD.iter().map(|&(n, d)| (n, d.to_owned())).collect();
    assert_eq!(got, want, "a d16 program's module differs from world's");
}

/// Step d: with the switch the integer view's branch reads through an
/// integer-typed handle (`bufferLoad.i32`, annotated `0x405`), so its bits
/// do not depend on a driver passing them through a float read, which is
/// unproven on NVIDIA; the float view's branch keeps the float-typed one
/// (`bufferLoad.f32`, `0x409`) and `world`'s narrowing. `dxil.dll` must
/// take both annotations of the one descriptor (each program here is
/// validated as it is translated). Off: the float-typed handle alone, as on
/// `world`.
#[test]
fn an_integer_views_d16_load_reads_through_an_integer_typed_handle() {
    if validator::Validator::load(validator::Validator::default_path().as_deref()).is_err() {
        eprintln!("NOT RUN: no dxil.dll, the annotations are not validated");
        return;
    }
    let loads = |ops: Vec<String>| -> Vec<String> { ops.into_iter().filter(|o| o.starts_with("bufferLoad")).collect() };
    for (name, native) in [
        ("d16_x", native_of(BUFFER_LOAD_FORMAT_D16_X, 8)),
        ("d16_xyzw", native_of(BUFFER_LOAD_FORMAT_D16_XYZW, 2)),
        ("heap-picked", heap_native()),
        ("selected", selected_native()),
    ] {
        let _ = super::shader::take_ops();
        let _ = translated(&native);
        assert_eq!(loads(super::shader::take_ops()), ["bufferLoad.f32"], "{name}, switch off");
        let on = with_kinds(|| translated(&native));
        let mut got = loads(super::shader::take_ops());
        got.sort();
        assert_eq!(got, ["bufferLoad.f32", "bufferLoad.i32"], "{name}, switch on");
        assert_ne!(on.dxil.len(), 0);
    }
}
