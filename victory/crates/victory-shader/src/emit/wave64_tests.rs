//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
pub(super) fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
pub(super) fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
pub(super) fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves are at most `host_wave` lanes.
pub(super) fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    on_the_card_of(&host, shader, constants)
}

/// [`on_the_card`] on `host`.
fn on_the_card_of(host: &Host, shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// Whether a device runs a program of `lanes`-lane waves: `Err` says why
/// not. `software`: the device is the software adapter; `(min, max)`: its
/// waves' lanes.
fn wave_gate(software: bool, (min, max): (u32, u32), lanes: u32) -> Result<(), String> {
    if software {
        return Err("the device is the software adapter".to_owned());
    }
    if !(min..=max).contains(&lanes) {
        return Err(format!("the device's waves have {min} to {max} lanes, not this program's {lanes}"));
    }
    Ok(())
}

/// The hardware device a program of `lanes`-lane waves is run on, the one
/// [`wave_gate`] asks about; None, and a line saying why, without one.
pub(super) fn card_with_waves_of(lanes: u32) -> Option<Host> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    // The software adapter is a skip of its own. On a card, a query that
    // fails is a failure, never a silent skip of the dispatches (Sol's
    // re-check, new fault 6).
    let software = host.is_software();
    let range = if software {
        (0, 0)
    } else {
        wave_lanes(&host).unwrap_or_else(|e| panic!("the device's wave sizes could not be asked (OPTIONS1): {e}"))
    };
    match wave_gate(software, range, lanes) {
        Ok(()) => Some(host),
        Err(why) => {
            eprintln!("{why}; not run");
            None
        }
    }
}

/// The software adapter is not a card. A device runs a 32-lane program
/// when 32 is among its wave sizes (the program asks for `WaveSize(32)`),
/// not only when its narrowest wave has 32 lanes: a device of 8 to 32
/// lanes runs it (Sol's review, finding 14).
#[test]
fn the_wave_gate_takes_every_device_with_the_programs_wave_size() {
    assert!(wave_gate(true, (32, 32), 32).is_err(), "the software adapter");
    assert!(wave_gate(false, (4, 4), 32).is_err(), "waves of 4");
    assert!(wave_gate(false, (64, 64), 32).is_err(), "waves of 64 only");
    assert_eq!(wave_gate(false, (32, 32), 32), Ok(()));
    assert_eq!(wave_gate(false, (32, 64), 32), Ok(()));
    assert_eq!(wave_gate(false, (8, 32), 32), Ok(()), "8 to 32 lanes");
    assert_eq!(wave_gate(false, (16, 64), 32), Ok(()), "16 to 64 lanes");
}

/// The fewest and the most lanes a wave of `host`'s device has.
fn wave_lanes(host: &Host) -> Result<(u32, u32), windows::core::Error> {
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1};
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: a plain query that fills a structure of the given size.
    unsafe {
        host.device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut options).cast(), size_of_val(&options) as u32)?
    };
    Ok((options.WaveLaneCountMin, options.WaveLaneCountMax))
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as a program may keep
/// its lanes' masks and its node numbers on one stack. The mask arrives as the ballot of its
/// lanes. The program is written for this test.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // The program is a 32-lane wave's (it asks for WaveSize(32)): a device
    // without that wave size (the software adapter's waves have 4 lanes)
    // runs it as several waves, and a ballot of 32 lanes is no mask of one.
    // Asked of the device it then runs on.
    let Some(host) = card_with_waves_of(32) else { return };
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card_of(&host, &shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along (a
/// program written for this test). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// A sum of doubles over a console wave of 64 lanes, made in a whole-wave section: the
/// threads below s2 take part, each with float(t) + s3 widened to a double, the others with
/// 0.0. Inside the section each lane adds the partial sum of a partner lane read through
/// DPP16 (`quad_perm`, then `row_xmask`), then of the other row of its half
/// (`v_permlanex16_b32`), then of the other half (`v_readlane_b32`); the sum leaves the
/// section in s[28:29] and is stored as a float. The section is a sum over every lane; the
/// widening before it is a double outside a section, so the program is refused.
#[test]
fn a_wave_sum_of_doubles_is_the_host_waves_sum() {
    // Encoded field by field from the instruction formats (SOP1, VOP1, VOP2, VOPC, VOP3,
    // DPP16, MUBUF); registers, operands and the order of the steps are this test's choice.
    const VCC: u32 = 106;
    const EXEC: u32 = 126;
    const ZERO: u32 = 128;
    const MINUS_ONE: u32 = 193;
    const DPP16: u32 = 0xFA;
    // v[10:11] the lane's double, v[12:13] its partial sum, v[14:15] a partner's; EXEC of
    // the branch in s[22:23], of the section in s[24:25]; the upper half's sum in s[26:27],
    // the wave's in s[28:29]
    const TERM: u32 = 10;
    const ACC: u32 = 12;
    const PEER: u32 = 14;
    const BRANCH: u32 = 22;
    const WHOLE: u32 = 24;
    const UPPER: u32 = 26;
    const SUM: u32 = 28;
    let sop1 = |op: u32, sdst: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
    let vop1 = |op: u32, vdst: u32, src0: u32| 0x3F << 25 | vdst << 17 | op << 9 | src0;
    let vop3 = |op: u32, vdst: u32, src: [u32; 3]| {
        [0xD400_0000 | op << 16 | vdst, src[0] | src[1] << 9 | src[2] << 18]
    };
    let v = |n: u32| 256 + n;
    let int = |k: u32| 128 + k; // an inline integer, 0..=64
    let add_f64 = |src0: u32| vop3(0x164, ACC, [src0, v(ACC), 0]); // v_add_f64 v[12:13], src0, v[12:13]
    let mut words = vec![
        vop1(0x06, 9, v(0)),                       // v_cvt_f32_u32 v9, v0
        0x03 << 25 | 9 << 17 | 9 << 9 | 3,         // v_add_f32 v9, s3, v9
        vop1(0x10, TERM, v(9)),                    // v_cvt_f64_f32 v[10:11], v9
        vop1(0x01, 8, 2),                          // v_mov_b32 v8, s2
        0x3E << 25 | 0xC1 << 17 | 8 << 9 | v(0),   // v_cmp_lt_u32 vcc, v0, v8
        sop1(0x24, BRANCH, VCC),                   // s_and_saveexec_b64 s[22:23], vcc
        sop1(0x25, WHOLE, MINUS_ONE),              // s_or_saveexec_b64 s[24:25], -1
    ];
    for h in 0..2 {
        // v_cndmask_b32 v12+h, 0, v10+h, s[24:25]
        words.extend(vop3(0x101, ACC + h, [ZERO, v(TERM + h), WHOLE]));
    }
    // DPP16 controls: quad_perm [2,3,0,1], quad_perm [1,0,3,2], row_xmask:4, row_xmask:8
    let in_a_row = [0x4E, 0xB1, 0x164, 0x168];
    for ctrl in in_a_row {
        for h in 0..2 {
            // v_mov_b32 v14+h, v12+h <ctrl> row_mask:0xf bank_mask:0xf
            words.extend([vop1(0x01, PEER + h, DPP16), 0xF << 28 | 0xF << 24 | ctrl << 8 | (ACC + h)]);
        }
        words.extend(add_f64(v(PEER)));
    }
    // the other row of the half, every lane reading its lane 15:
    // v_permlanex16_b32 v14+h, v12+h, -1, -1
    let other_row = [0, 1].map(|h| vop3(0x378, PEER + h, [v(ACC + h), MINUS_ONE, MINUS_ONE]));
    words.extend(other_row.concat());
    words.extend(add_f64(v(PEER)));
    // the other half: lane 40's sum added in every lane, then lane 5's total
    for h in 0..2 {
        words.extend(vop3(0x360, UPPER + h, [v(ACC + h), int(40), 0])); // v_readlane_b32 s26+h, v12+h, 40
    }
    words.extend(add_f64(UPPER));
    for h in 0..2 {
        words.extend(vop3(0x360, SUM + h, [v(ACC + h), int(5), 0])); // v_readlane_b32 s28+h, v12+h, 5
    }
    words.extend([
        sop1(0x04, EXEC, WHOLE),       // s_mov_b64 exec, s[24:25]
        vop1(0x0F, 1, SUM),            // v_cvt_f32_f64 v1, s[28:29]
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
        sop1(0x04, EXEC, BRANCH),      // s_mov_b64 exec, s[22:23]
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let count = |n: &str| p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == n).count();
    // an addition per step in a row, one for the other row and one for the other half; a
    // permute per dword; one narrowing
    assert_eq!(
        (count("v_add_f64"), count("v_permlanex16_b32"), count("v_cvt_f32_f64")),
        (in_a_row.len() + 2, other_row.len(), 1)
    );
    // the section leaves the sum over every lane of `s[24:25] ? v[10:11] : 0.0` in s[28:29]
    let saves = |i: &ir::Inst| i.name() == "s_or_saveexec_b64";
    let block = p.blocks.iter().position(|b| b.insts.iter().any(saves)).expect("the section");
    let insts = &p.blocks[block].insts;
    let start = insts.iter().position(saves).expect("the section");
    let written = |r: ir::Reg| insts.iter().find_map(|i| i.write(r)).expect("a register written");
    let reg = |n: u32| u8::try_from(n).expect("a register number");
    let (lo, hi) = (written(ir::Reg::V(reg(TERM))), written(ir::Reg::V(reg(TERM + 1))));
    let mask = written(ir::Reg::S(reg(WHOLE)));
    let sections = super::wave::sections(&p, block);
    let section = sections.get(&start).expect("a whole-wave section at s_or_saveexec_b64");
    for (high, dword) in [(false, SUM), (true, SUM + 1)] {
        assert_eq!(
            section.shapes.get(&written(ir::Reg::S(reg(dword)))),
            Some(&super::wave::Shape::ActiveF64(lo, hi, mask, high)),
            "s{dword}: the sum's {} dword",
            if high { "high" } else { "low" }
        );
    }
    // The doubles around the wave section (the widening) wait for the program's float mode,
    // which does not reach the translator (the second review of job/land-2c): the program is
    // refused.
    let why = lowered_on(&native, &p, &HashSet::new(), 64).err().map(|e| e.to_string());
    assert!(
        why.as_deref().is_some_and(|w| w.contains("a double operation whose float mode is not known")),
        "{why:?}"
    );
}
