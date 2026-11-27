//! End-to-end tests of the slot map (`slot_map`, `VICTORY_SLOT_MAP=1`):
//! synthetic GFX10 programs lowered with the map on, validated by `dxil.dll`
//! when present, and run on the host GPU and WARP with a hand-built
//! directory and pool. The host heap holds the typed nulls at 0..64 and the
//! views the map names from 64 on. The host slots equal to the title slots
//! the programs form hold poison views: a translation that reads host slot =
//! title slot (the mirror's rule) reads poison and writes the poison UAV.

use std::collections::BTreeSet;

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::slot_map::{CHUNK_SHIFT, CHUNK_SLOTS, DIR_ENTRIES, MAP_DIR, MAP_POOL, TEST_ON, entry};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Slot-map classes: Texture2D SRV, raw buffer SRV, raw buffer UAV.
const TEX: u32 = 2;
const RAW: u32 = 11;
const RAW_UAV: u32 = 11 | 32;

/// Where the fixtures' tables start in the title heap: the last slot of the
/// first chunk, so their entries lie on both sides of a chunk boundary.
const TABLE: u32 = CHUNK_SLOTS - 1;

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset` (signed, 21 bits).
const fn s_load_dwordx4(dst: u8, base: u8, offset: i32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (offset as u32 & 0x1F_FFFF),
    ]
}

/// MUBUF `op vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const BUFFER_LOAD_DWORD: u32 = 0x0C;
const BUFFER_STORE_DWORD: u32 = 0x1C;

/// The directory and pool that map each `(title slot, host slot, class)`:
/// the title chunks named get pool chunks from 1 on; chunk 0 is all misses.
fn map(entries: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let chunks: BTreeSet<u32> = entries.iter().map(|e| e.0 >> CHUNK_SHIFT).collect();
    let mut directory = vec![0u32; DIR_ENTRIES as usize];
    let mut pool = vec![0u32; (chunks.len() + 1) * CHUNK_SLOTS as usize];
    for (n, &c) in chunks.iter().enumerate() {
        directory[c as usize] = n as u32 + 1;
    }
    for &(t, h, class) in entries {
        let at = directory[(t >> CHUNK_SHIFT) as usize] << CHUNK_SHIFT | (t & (CHUNK_SLOTS - 1));
        pool[at as usize] = entry(h, class);
    }
    (directory, pool)
}

/// A host heap of `len` slots: the typed nulls at 0..64, `views` at their
/// slots, a null texture view in every other.
fn heap(len: usize, views: Vec<(usize, View<'_>)>) -> Vec<View<'_>> {
    let mut slots: Vec<Option<View>> = (0..len).map(|_| None).collect();
    for (at, v) in views {
        slots[at] = Some(v);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.unwrap_or(View::Null(if i < 64 { i as u32 } else { TEX })))
        .collect()
}

/// The root parameter of the root SRV `t<slot>` of `shader`: user data is
/// parameter 0, root views follow by slot, then kind
/// (`Host::root_signature_for`).
fn root_srv(shader: &HostShader, slot: u8) -> Option<u32> {
    let mut views: Vec<(u8, u8)> = shader.root_cbvs.iter().map(|&r| (r, 0)).collect();
    views.extend(shader.root_srvs.iter().map(|&r| (r, 1)));
    views.extend(shader.root_uavs.iter().map(|&r| (r, 2)));
    views.sort_unstable();
    views.dedup();
    views.iter().position(|&v| v == (slot, 1)).map(|p| p as u32 + 1)
}

/// Runs `shader` with user data `ud`, the map `(directory, pool)` and the
/// `more` root SRVs (slot, dwords) it reads, and `views` as the heap, on
/// the hardware GPU and on WARP (each skipped without a device); per one
/// that ran, every UAV in heap order.
fn run(
    shader: &HostShader,
    ud: &[u32],
    (directory, pool): &(Vec<u32>, Vec<u32>),
    views: &[View],
    more: &[(u8, &[u32])],
) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut roots = vec![(0, Root::Constants(ud.to_vec()))];
        let srvs = [(MAP_DIR, &directory[..]), (MAP_POOL, &pool[..])];
        for &(slot, data) in srvs.iter().chain(more) {
            if let Some(param) = root_srv(shader, slot) {
                roots.push((
                    param,
                    Root::View {
                        srv: true,
                        data: data.to_vec(),
                    },
                ));
            }
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots,
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words));
    }
    assert!(!out.is_empty(), "no SM 6.6 device");
    out
}

/// A root parameter of `sgprs` user SGPRs from s`user_data`, the title's
/// root parameter `slot`.
fn param(root: RootType, user_data: u8, sgprs: u32, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    }
}

/// Lowers a one-thread wave32 compute program with root constants at s0..s3
/// and the view table at s4, whose entry 0 is a raw UAV and entries from 1
/// on raw buffer SRVs; with `heap`, the view heap's base at s6.
fn lower_table_program(words: &[u32], heap: bool) -> HostShader {
    lower_program(words, heap, ResourceType::Srv, 1)
}

/// [`lower_table_program`] with entries from 1 on of kind `kind`, and
/// `threads` threads in x (the thread's index in v0).
fn lower_program(words: &[u32], heap: bool, kind: ResourceType, threads: u32) -> HostShader {
    let mut layout = vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)];
    if heap {
        layout.push(param(RootType::ResourceHeap, 6, 1, 2));
    }
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used: [(ResourceType::Uav, 0), (kind, 8)]
            .map(|(kind, table_offset_dwords)| UsedResource {
                kind,
                root: RootType::Table,
                table_offset_dwords,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            })
            .to_vec(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: (if heap { 7 } else { 5 }) << 1, // user SGPRs: s0..s4 (s6)
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
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

/// The user data of a program of [`lower_table_program`]'s layout whose
/// table starts at title slot `table`, with `constants` at s0..s3.
fn table_ud(shader: &HostShader, table: u32, constants: [u32; 4]) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Table => table * 32,
            RootType::Constants => constants[u.dword as usize],
            // A view's title address as a number: none of these programs uses it.
            RootType::Other(super::USER_VIEW_ADDRESS) | RootType::Other(0x102) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    ud
}

/// One program reads a table on both sides of a chunk boundary, the entry
/// 32 bytes before its base, a fixed entry three past the last one named,
/// an entry the title never wrote and one holding a texture, each as a raw
/// buffer, and stores what it read through the table's UAV. Each read goes
/// through the map: the named views read exactly, the title's own
/// neighbour too, the unmapped and the incompatible entry read zeros, and
/// the views at host slots equal to the title slots are never touched.
#[test]
fn table_reads_go_through_the_map_and_misses_read_zeros() {
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table
    for (dst, offset) in [(8, 0), (12, 32), (16, 64), (20, -32), (24, 160), (28, 96), (32, 128)] {
        words.extend_from_slice(&s_load_dwordx4(dst, 4, offset));
    }
    for (v, srsrc) in [(1, 12), (2, 16), (3, 20), (4, 24), (5, 28), (6, 32)] {
        words.extend_from_slice(&mubuf(BUFFER_LOAD_DWORD, v, srsrc, 0));
    }
    for v in 1..=6u8 {
        words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, v, 8, 4 * (u16::from(v) - 1)));
    }
    words.push(S_ENDPGM);
    let shader = lower_table_program(&words, false);

    // Host slots: the UAV, four buffers, a texture; poison at the title's numbers.
    let ([a, b, c, d], tex) = ([[0xA0], [0xB0], [0xC0], [0xD0]], [[5.0f32, 0.0, 0.0, 1.0]]);
    let poison: Vec<[u32; 1]> = (0..8).map(|k| [0xBAD0 + k]).collect();
    let mut views = vec![
        (64, View::RawUav(8)),
        (65, raw(&a)),
        (66, raw(&b)),
        (67, raw(&c)),
        (68, raw(&d)),
        (69, View::Texture(1, 1, &tex)),
    ];
    let t = TABLE as usize;
    for k in 0..7 {
        views.push((t - 1 + k, if k == 1 { View::RawUav(8) } else { raw(&poison[k]) }));
    }
    let views = heap(t + 6, views);
    let m = map(&[
        (TABLE - 1, 67, RAW),
        (TABLE, 64, RAW_UAV),
        (TABLE + 1, 65, RAW),
        (TABLE + 2, 66, RAW),
        (TABLE + 4, 69, TEX),
        (TABLE + 5, 68, RAW),
    ]);
    for (label, uavs) in run(&shader, &table_ud(&shader, TABLE, [0; 4]), &m, &views, &[]) {
        assert_eq!(uavs[0][..6], [0xA0, 0xB0, 0xC0, 0xD0, 0, 0], "{label}: the map's UAV");
        assert_eq!(uavs[1], [0; 8], "{label}: the UAV at the title's slot number");
    }
    assert!(root_srv(&shader, MAP_DIR).is_some(), "the shader reads the directory");
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], s<soffset>`.
const fn s_load_dwordx4_by(dst: u8, base: u8, soffset: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (soffset as u32) << 25,
    ]
}

/// A program picks a buffer from the view heap by the title slot in a root
/// constant (`ResourceHeap`) and loads element 1 of it, indexed: the handle
/// and the element's stride both come from the host slot the map names
/// (the host's table of strides is by host slot). Through the title slot,
/// the stride is the poison buffer's, and so is the handle.
#[test]
fn a_buffer_picked_from_the_heap_goes_through_the_map_with_its_host_slots_stride() {
    TEST_ON.set(true);
    const PICKED: u32 = 3000;
    let mut words = vec![s_mov_b32(5, 0x80), s_mov_b32(7, 0x80)]; // s[4:5] the table, s[6:7] the heap
    words.extend_from_slice(&s_load_dwordx4(8, 4, 0));
    // s_lshl_b32 s12, s0, 5: the picked title slot's byte offset
    words.push(0x8000_0000 | 0x1E << 23 | 12 << 16 | 0x85 << 8);
    words.extend_from_slice(&s_load_dwordx4_by(16, 6, 12));
    words.push(0x7E00_0000 | 0x001 << 9 | 0x81); // v_mov_b32 v0, 1
    // buffer_load_dword v1, v0, s[16:19], 0 idxen
    words.extend_from_slice(&[0x38 << 26 | BUFFER_LOAD_DWORD << 18 | 1 << 13, 0x80 << 24 | (16 >> 2) << 16 | 1 << 8]);
    words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, 1, 8, 0));
    words.push(S_ENDPGM);
    let shader = lower_table_program(&words, true);
    let (data, poison) = ([0xE0, 0xE1, 0xE2, 0xE3], [0xBAD0, 0xBAD1, 0xBAD2, 0xBAD3]);
    let t = TABLE as usize;
    let p = PICKED as usize;
    let views = heap(
        p + 1,
        vec![(64, View::RawUav(4)), (70, raw(&data)), (t, View::RawUav(4)), (p, raw(&poison))],
    );
    let mut strides = vec![0u32; p + 1];
    strides[70] = 8;
    strides[p] = 4;
    let m = map(&[(TABLE, 64, RAW_UAV), (PICKED, 70, RAW)]);
    let ud = table_ud(&shader, TABLE, [PICKED, 0, 0, 0]);
    for (label, uavs) in run(&shader, &ud, &m, &views, &[(super::SLOT_STRIDES, &strides)]) {
        assert_eq!(uavs[0][0], 0xE2, "{label}: element 1 at the host slot's stride of 8 bytes");
        assert_eq!(uavs[1], [0; 4], "{label}: the UAV at the title's slot number");
    }
    assert!(root_srv(&shader, super::SLOT_STRIDES).is_some(), "the shader reads the strides");
}

/// SMEM `s_buffer_load_dword*` (`op`: 8 one dword, 10 four) of
/// `s[dst..]` from the buffer descriptor `s[base:base+3]` at `offset` plus
/// `s<soffset>` (125: none).
const fn s_buffer_load(op: u32, dst: u8, base: u8, offset: u32, soffset: u8) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (soffset as u32) << 25 | (offset & 0x1F_FFFF),
    ]
}

/// MUBUF `op vdata, v<vaddr>, s[srsrc:srsrc+3], 0 offen`.
const fn mubuf_offen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A constant buffer in a table, read at a fixed row (`s_buffer_load_dwordx4`
/// at byte 16), at a byte the program computes (`s_buffer_load_dword` at a
/// root constant's offset) and per lane (`buffer_load_dword` at each lane's
/// own dword): with the map on, each read is a raw buffer read of the view
/// the map names (the host makes a constant buffer view raw: its records
/// start on 16 bytes, not D3D12's 256), exact for every lane.
#[test]
fn a_table_constant_buffer_is_read_as_a_raw_buffer_at_fixed_computed_and_per_lane_bytes() {
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table
    words.extend_from_slice(&s_load_dwordx4(8, 4, 0)); // entry 0: the UAV
    words.extend_from_slice(&s_load_dwordx4(12, 4, 32)); // entry 1: the constant buffer
    words.extend_from_slice(&s_buffer_load(10, 16, 12, 16, 125)); // s[16:19] = bytes 16..32
    words.extend_from_slice(&s_buffer_load(8, 20, 12, 0, 0)); // s20 = the dword at byte s0
    words.push(0x3402_0082); // v_lshlrev_b32 v1, 2, v0
    words.extend_from_slice(&mubuf_offen(BUFFER_LOAD_DWORD, 2, 1, 12)); // v2 = the lane's dword
    words.extend_from_slice(&mubuf_offen(BUFFER_STORE_DWORD, 2, 1, 8));
    for (n, s) in (16..=20u8).enumerate() {
        words.push(0x7E00_0000 | 3 << 17 | 0x001 << 9 | u32::from(s)); // v_mov_b32 v3, s<s>
        words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, 3, 8, 128 + 4 * n as u16));
    }
    words.push(S_ENDPGM);
    let shader = lower_program(&words, false, ResourceType::Cbv, 32);
    let data: Vec<u32> = (0..64).map(|i| 0xC000 + i).collect();
    let poison: Vec<u32> = (0..64).map(|i| 0xBAD0 + i).collect();
    let t = TABLE as usize;
    let views = heap(
        t + 2,
        vec![
            (64, View::RawUav(40)),
            (65, raw(&data)),
            (t, View::RawUav(40)),
            (t + 1, View::Cbv(&poison)),
        ],
    );
    let m = map(&[(TABLE, 64, RAW_UAV), (TABLE + 1, 65, RAW)]);
    let ud = table_ud(&shader, TABLE, [36, 0, 0, 0]);
    let mut want: Vec<u32> = (0..32).map(|l| 0xC000 + l).collect();
    want.extend([0xC004, 0xC005, 0xC006, 0xC007, 0xC009]);
    for (label, uavs) in run(&shader, &ud, &m, &views, &[]) {
        assert_eq!(uavs[0][..37], want[..], "{label}: lanes 0..32, then the row at byte 16 and the dword at 36");
        assert_eq!(uavs[1], [0; 40], "{label}: the UAV at the title's slot number");
    }
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], s<soffset> offset:offset`.
const fn s_load_dwordx4_by_at(dst: u8, base: u8, soffset: u8, offset: u32) -> [u32; 2] {
    let [w0, w1] = s_load_dwordx4_by(dst, base, soffset);
    [w0, w1 | (offset & 0x1F_FFFF)]
}

/// A one-dword raw read (`buffer_load_dword v1, off, s[16:19]`) of the buffer
/// descriptor at s[16:19], stored through the table's UAV at s[8:11].
fn load_and_store(words: &mut Vec<u32>) {
    words.extend_from_slice(&mubuf(BUFFER_LOAD_DWORD, 1, 16, 0));
    words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, 1, 8, 0));
    words.push(S_ENDPGM);
}

/// A table entry picked by a root constant (`RootIndexedView`: entry
/// `1 + s0`, a byte offset `s0 << 5` beside a fixed 32): its title slot goes
/// through the map.
#[test]
fn a_table_entry_picked_by_a_root_constant_goes_through_the_map() {
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4, 0));
    words.push(0x8000_0000 | 0x1E << 23 | 12 << 16 | 0x85 << 8); // s_lshl_b32 s12, s0, 5
    words.extend_from_slice(&s_load_dwordx4_by_at(16, 4, 12, 32));
    load_and_store(&mut words);
    let shader = lower_table_program(&words, false);
    assert!(!shader.root_indexed_views.is_empty(), "the pick is a root-indexed view");
    let (data, poison) = ([0x5A], [0xBAD0]);
    let t = TABLE as usize;
    let views = heap(
        t + 5,
        vec![(64, View::RawUav(1)), (70, raw(&data)), (t, View::RawUav(1)), (t + 4, raw(&poison))],
    );
    let m = map(&[(TABLE, 64, RAW_UAV), (TABLE + 4, 70, RAW)]);
    for (label, uavs) in run(&shader, &table_ud(&shader, TABLE, [3, 0, 0, 0]), &m, &views, &[]) {
        assert_eq!(uavs[0], [0x5A], "{label}: entry 4 through the map");
        assert_eq!(uavs[1], [0], "{label}: the UAV at the title's slot number");
    }
}

/// Two descriptors of a table, one chosen by SCC (`s_cselect_b64` per half,
/// a `Source::Select`): the chosen title slot goes through the map, either way.
#[test]
fn a_descriptor_chosen_by_scc_goes_through_the_map() {
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)];
    for (dst, offset) in [(8, 0), (12, 32), (24, 64)] {
        words.extend_from_slice(&s_load_dwordx4(dst, 4, offset));
    }
    words.push(0xBF06_8000); // s_cmp_eq_u32 s0, 0
    words.push(0x8000_0000 | 0x0B << 23 | 16 << 16 | 24 << 8 | 12); // s_cselect_b64 s[16:17], s[12:13], s[24:25]
    words.push(0x8000_0000 | 0x0B << 23 | 18 << 16 | 26 << 8 | 14); // s_cselect_b64 s[18:19], s[14:15], s[26:27]
    load_and_store(&mut words);
    let shader = lower_table_program(&words, false);
    let (a, b, poison) = ([0xA1], [0xB1], [0xBAD0]);
    let t = TABLE as usize;
    let views = heap(
        t + 3,
        vec![
            (64, View::RawUav(1)),
            (65, raw(&a)),
            (66, raw(&b)),
            (t, View::RawUav(1)),
            (t + 1, raw(&poison)),
            (t + 2, raw(&poison)),
        ],
    );
    let m = map(&[(TABLE, 64, RAW_UAV), (TABLE + 1, 65, RAW), (TABLE + 2, 66, RAW)]);
    for (s0, want) in [(0, 0xA1), (1, 0xB1)] {
        for (label, uavs) in run(&shader, &table_ud(&shader, TABLE, [s0, 0, 0, 0]), &m, &views, &[]) {
            assert_eq!(uavs[0], [want], "{label}: s0 = {s0}");
            assert_eq!(uavs[1], [0], "{label}: the UAV at the title's slot number");
        }
    }
}

/// A buffer descriptor whose record count the code reads as a number
/// (dword 2, as the console compiler tests a buffer's emptiness) is
/// answered by the size of the view the map names: 64 bytes here, where
/// the view at the title's slot number holds 32.
#[test]
fn a_size_read_from_a_descriptor_is_the_mapped_views() {
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4, 0));
    words.extend_from_slice(&s_load_dwordx4(12, 4, 32));
    words.push(0x7E00_0000 | 1 << 17 | 0x001 << 9 | 14); // v_mov_b32 v1, s14
    words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, 1, 8, 0));
    words.push(S_ENDPGM);
    let shader = lower_table_program(&words, false);
    let (data, poison) = ([7u32; 16], [9u32; 8]);
    let t = TABLE as usize;
    let views = heap(
        t + 2,
        vec![(64, View::RawUav(1)), (65, raw(&data)), (t, View::RawUav(1)), (t + 1, raw(&poison))],
    );
    let m = map(&[(TABLE, 64, RAW_UAV), (TABLE + 1, 65, RAW)]);
    for (label, uavs) in run(&shader, &table_ud(&shader, TABLE, [0; 4]), &m, &views, &[]) {
        assert_eq!(uavs[0], [64], "{label}: the mapped view's bytes");
        assert_eq!(uavs[1], [0], "{label}: the UAV at the title's slot number");
    }
}

/// Title memory read by address takes the memory copy's view, a host slot
/// the host owns (`USER_MEMORY_BASE` plus the copy): it never reads the map.
/// Here the program's one map lookup is its table's UAV.
#[test]
fn a_memory_copy_is_a_host_slot_and_bypasses_the_map() {
    use super::slot_map::TEST_LOOKUPS;
    TEST_ON.set(true);
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4, 0));
    words.push(0x7E00_0000 | 2 << 17 | 0x001 << 9); // v_mov_b32 v2, s0
    words.push(0x7E00_0000 | 3 << 17 | 0x001 << 9 | 1); // v_mov_b32 v3, s1
    words.extend_from_slice(&[0xDC30_8000, 1 << 24 | 0x7D << 16 | 2]); // global_load_dword v1, v[2:3], off
    words.extend_from_slice(&mubuf(BUFFER_STORE_DWORD, 1, 8, 0));
    words.push(S_ENDPGM);
    TEST_LOOKUPS.set(0);
    let shader = lower_table_program(&words, false);
    assert!(shader.root_srvs.contains(&super::SLOT_MEMORY), "read through the directory of title memory");
    assert!(
        shader.user_data.iter().any(|u| u.root == RootType::Other(super::USER_MEMORY_BASE)),
        "the copies' first host slot is the host's word"
    );
    assert_eq!(TEST_LOOKUPS.get(), 1, "only the table's UAV reads the map");
}

/// A raw buffer SRV over `data`.
fn raw(data: &[u32]) -> View<'_> {
    View::Buffer {
        data,
        stride: Stride::Raw,
    }
}

/// The waterfall programs of `waterfall_tests` (lanes picking one of two
/// textures by their own descriptors: joined by control flow, at a first
/// lane's offset, loaded or picked per lane) with their table on both
/// sides of a chunk boundary: each lane's title slot goes through the map,
/// non-uniform, and the textures at the title's numbers are not sampled.
#[test]
fn per_lane_descriptors_go_through_the_map() {
    use crate::resource::tests::{WATERFALL_INDEXED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    TEST_ON.set(true);
    let (first, second, poison) = ([[1.0f32, 0.0, 0.0, 1.0]], [[2.0f32, 0.0, 0.0, 1.0]], [[9.0f32, 0.0, 0.0, 1.0]]);
    let t = TABLE as usize;
    let views = heap(
        t + 3,
        vec![
            (64, View::Texture(1, 1, &first)),
            (65, View::Texture(1, 1, &second)),
            (66, View::RawUav(32)),
            (t, View::Texture(1, 1, &poison)),
            (t + 1, View::Texture(1, 1, &poison)),
            (t + 2, View::RawUav(32)),
        ],
    );
    let m = map(&[(TABLE, 64, TEX), (TABLE + 1, 65, TEX), (TABLE + 2, 66, RAW_UAV)]);
    let cases: [(&[u32], fn(usize) -> f32); 3] = [
        (WATERFALL_JOINED_WAVE32, |l| if l < 16 { 1.0 } else { 2.0 }),
        (WATERFALL_INDEXED_WAVE32, |l| [1.0, 2.0][l & 1]),
        (WATERFALL_JOINED_MIXED_WAVE32, |l| match l {
            0..16 => [1.0, 2.0][l & 1],
            16..24 => 1.0,
            _ => 2.0,
        }),
    ];
    for (n, (words, want)) in cases.into_iter().enumerate() {
        let shader = super::waterfall_tests::build(words, false);
        // Both tables at the user data; the sampler table at 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if u.slot == 0 {
                ud[u.index as usize] = TABLE * 32;
            }
        }
        let want: Vec<f32> = (0..32).map(want).collect();
        for (label, uavs) in run(&shader, &ud, &m, &views, &[]) {
            let red: Vec<f32> = uavs[0].iter().map(|&w| f32::from_bits(w)).collect();
            assert_eq!(red, want, "{label}, program {n}");
            assert_eq!(uavs[1], [0; 32], "{label}, program {n}: the UAV at the title's slot number");
        }
    }
}
