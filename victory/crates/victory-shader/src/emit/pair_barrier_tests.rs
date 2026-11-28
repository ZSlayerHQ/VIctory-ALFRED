//! The meetings of a console wave's two host waves as group barriers (board W35 P2,
//! `VICTORY_PAIR_MEETINGS=barrier`): thread groups of two console waves of 64, each run as two
//! host waves of 32 ([`lower::with_pair_meetings`]). Each fixture is one of the design note's
//! correctness checks. Where values are compared, the console's are the 64-lane model's
//! ([`super::wave_model`], one run per console wave); the translation's come from a device
//! with 32-lane waves, which the software adapter is not (G-127): there the comparison is
//! skipped with a line, and the shape of the emitted program is what is checked. A program
//! with a group barrier on a cycle of its control flow goes to the software adapter only
//! (G-106).

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::wave64_tests::native_of;
use super::{HostShader, lower, validator};
use crate::native::{NativeProgram, RootType, StageIo};
use crate::{ShaderError, io, ir, resource, structure};

/// `v_lshlrev_b32 v1, 2, v0`: the thread's dword of the UAV.
const DWORD_OF_T: u32 = 0x3402_0082;
/// `v_add_nc_u32 v2, s0, v0`: v2 = s0 + t.
const V2_IS_S0_PLUS_T: u32 = 0x4A04_0000;
/// `v_readlane_b32 s7, v2, 63`: lane 63, in the upper host wave of the console wave.
const READ_LANE_63: [u32; 2] = [0xD760_0007, 0x0001_7F02];
/// `v_readlane_b32 s7, v2, 0`: lane 0, in the lower host wave.
const READ_LANE_0: [u32; 2] = [0xD760_0007, 0x0001_0102];
/// `v_add_nc_u32 v2, s7, v2`.
const ADD_S7: u32 = 0x4A04_0407;
/// `buffer_store_dword v2, v1, s[8:11], 0 offen`.
const STORE_V2: [u32; 2] = [0xE070_1000, 0x8002_0201];
/// `s_mov_b64 s[14:15], exec` and back.
const SAVE_EXEC: u32 = 0xBE8E_047E;
const RESTORE_EXEC: u32 = 0xBEFE_040E;
/// `v_mov_b32 v5, 0`, `v_add_nc_u32 v5, 1, v5`, `v_cmpx_gt_u32 exec, v7, v5`: a lane's rounds.
const V5_IS_0: u32 = 0x7E0A_0280;
const V5_PLUS_1: u32 = 0x4A0A_0A81;
const ROUNDS_LEFT: u32 = 0x7DA8_0B07;

/// The program lowered for a host of 32-lane waves, its console waves meeting on their own,
/// with the branches closing `wide` taken wave-wide and the meetings as group barriers when
/// `barrier`: the wave size it asks for, the shader, and its group barriers on a cycle of its
/// control flow.
fn paired(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    barrier: bool,
) -> Result<(Option<u32>, HostShader, u32), ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let host = lower::TEST_HOST_WAVE.replace(Some(32));
    let form = super::TEST_PAIR_BARRIERS.replace(Some(barrier));
    let lowered = lower::with_pair_meetings(|| {
        lower::lower(&lower::Parts {
            native,
            p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
    });
    super::TEST_PAIR_BARRIERS.set(form);
    lower::TEST_HOST_WAVE.set(host);
    let lowered = lowered?;
    let shader = HostShader {
        stage: crate::native::HwStage::Cs,
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
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let looped = super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote");
    Ok((lowered.wave_size, shader, looped))
}

/// A group of 128 threads (two console waves) running `body`, with `lds` bytes of LDS.
fn group_of_two(body: &[u32], lds: u32) -> (NativeProgram, ir::Program) {
    let mut native = native_of(body, true, 128);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= lds.div_ceil(512) << 15;
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    (native, p)
}

/// The block closed by the program's first `name`.
fn block_ending(p: &ir::Program, name: &str) -> usize {
    (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .unwrap_or_else(|| panic!("no block ends in {name}"))
}

/// The UAV the console leaves: the 64-lane model run once per console wave, the second with
/// v0 = 64 + lane (`v_add_nc_u32 v0, 64, v0` before the program), each keeping its own dwords.
fn console(native: &NativeProgram, constants: &[u32]) -> Vec<u32> {
    let words: Vec<u32> = native.code.chunks(4).map(|b| u32::from_le_bytes(b.try_into().expect("a word"))).collect();
    let mut out = super::wave_model::console(&words, constants, 128);
    let second: Vec<u32> = [0x4A00_00C0].into_iter().chain(words).collect();
    out[64..].copy_from_slice(&super::wave_model::console(&second, constants, 128)[64..]);
    out
}

/// The first 128 dwords of the UAV after one group of `shader` ran with root constants s0,
/// s1, on the default adapter, or on the software adapter only with `software`; None (and a
/// line) where the adapter does not take it.
fn run(shader: &HostShader, constants: [u32; 2], software: bool) -> Option<Vec<u32>> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(software) else {
        eprintln!("no SM 6.6 device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the adapter does not take this program (no waves of 32?): {e}; not compared");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(&root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(ud))],
            views: &[View::RawUav(128)],
            samplers: &[],
            groups: [1, 1, 1],
        })
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][..128].to_vec())
}

fn refused_as(r: Result<(Option<u32>, HostShader, u32), ShaderError>, want: &str, name: &str) {
    match r {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, want, "{name}"),
        Err(e) => panic!("{name}: refused for another reason: {e}"),
        Ok((wave, _, looped)) => panic!("{name}: emitted (wave {wave:?}, {looped} group barriers on a cycle), not refused"),
    }
}

/// Check 1: two console waves read lane 63 and then lane 0 of their own console wave (each an
/// exchange between the halves) in code every thread runs. As group barriers: one barrier per
/// meeting, none on a cycle of the emitted program (the spin form waits in a loop), and the
/// console's values.
#[test]
fn meetings_in_group_uniform_code_are_one_barrier_each_and_no_loop() {
    let mut body = vec![DWORD_OF_T, V2_IS_S0_PLUS_T];
    body.extend(READ_LANE_63);
    body.push(ADD_S7);
    body.extend(READ_LANE_0);
    body.push(ADD_S7);
    body.extend(STORE_V2);
    let (native, p) = group_of_two(&body, 0);
    let (wave, shader, looped) = paired(&native, &p, &HashSet::new(), true).expect("emitted as group barriers");
    assert_eq!(wave, Some(32));
    assert_eq!(looped, 0, "a group barrier on a loop of the emitted program: the meeting still spins");
    for s0 in [0u32, 1000, 0xFFFF_FF00] {
        let Some(got) = run(&shader, [s0, 0], false) else { return };
        assert_eq!(got, console(&native, &[s0, 0, 0, 0]), "s0 = {s0:#x}");
    }
    eprintln!("ran: both console waves' exchanges, as the console runs them");
}

/// Check 2, Astra's counter-examples: a per-lane loop with an exchange of LDS between the
/// halves each round (lane t writes its sum and adds lane t ^ 63's), its back edge taken per
/// console wave. `rounds` sets v7 to the lane's rounds: a back edge only the upper half takes
/// (1 and 2 rounds), unequal trips between the halves of one console wave (2 and 4), unequal
/// trips between the console waves (2 and 4). The loop's latch decides per console wave, which
/// the group need not share: refused as group barriers (P3 would take it group-wide).
#[test]
fn meetings_in_loops_a_console_wave_goes_round_its_own_way_are_refused_as_barriers() {
    let cases: [(&str, [u32; 4]); 3] = [
        // v7 = ((t >> 5) & 1) + 1
        ("a back edge one half takes", [0x2C0E_0085, 0x360E_0E81, 0x4A0E_0E81, 0x4A0E_0E80]),
        // v7 = ((t >> 5) & 1) * 2 + 2
        ("unequal trips of the halves", [0x2C0E_0085, 0x360E_0E81, 0x340E_0E81, 0x4A0E_0E82]),
        // v7 = (t >> 6) * 2 + 2
        ("unequal trips of the console waves", [0x2C0E_0086, 0x340E_0E81, 0x4A0E_0E82, 0x4A0E_0E80]),
    ];
    for (name, rounds) in cases {
        let mut body = vec![
            DWORD_OF_T,
            0x3A06_00BF, // v_xor_b32 v3, 63, v0
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3: the partner's dword
            V2_IS_S0_PLUS_T,
        ];
        body.extend(rounds);
        body.extend([
            V5_IS_0,
            SAVE_EXEC,
            0xD834_0000,
            0x0000_0201, // loop: ds_write_b32 v1, v2
            0xD8D8_0000,
            0x0400_0003, // ds_read_b32 v4, v3
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x4A04_0902, // v_add_nc_u32 v2, v2, v4
            V5_PLUS_1,
            ROUNDS_LEFT,
            0xBF89_FFF7, // s_cbranch_execnz loop
            RESTORE_EXEC,
        ]);
        body.extend(STORE_V2);
        let (native, p) = group_of_two(&body, 512);
        let wide = HashSet::from([block_ending(&p, "s_cbranch_execnz")]);
        assert!(paired(&native, &p, &wide, false).is_ok(), "{name}: the spin form emits it");
        refused_as(paired(&native, &p, &wide, true), lower::PAIR_UNPROVEN, name);
    }
}

/// Check 3, a wait after a loop whose exit is not group-uniform. Lane t adds 1 to v2 for
/// (t >> 6) * 2 + 2 rounds, then reads lane 63. With the latch taken per console wave
/// (`s_cbranch_execnz` wave-wide), its decision is an exchange of the halves inside the loop,
/// which the console waves go round 2 and 4 times: refused. With the latch per lane each host
/// thread goes round on its own, and every thread reaches the read after the loop once: that
/// stays emitted, with the console's values (the note's check says "refused"; this reading of
/// it, which D3D12's rule allows, is for the lead to confirm).
#[test]
fn a_meeting_after_a_loop_whose_exit_is_not_group_uniform() {
    let mut body = vec![
        DWORD_OF_T,
        V2_IS_S0_PLUS_T,
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0
        0x340E_0E81, // v_lshlrev_b32 v7, 1, v7
        0x4A0E_0E82, // v_add_nc_u32 v7, 2, v7
        V5_IS_0,
        SAVE_EXEC,
        0x4A04_0481, // loop: v_add_nc_u32 v2, 1, v2
        V5_PLUS_1,
        ROUNDS_LEFT,
        0xBF89_FFFC, // s_cbranch_execnz loop
        RESTORE_EXEC,
    ];
    body.extend(READ_LANE_63);
    body.push(ADD_S7);
    body.extend(STORE_V2);
    let (native, p) = group_of_two(&body, 0);
    let wide = HashSet::from([block_ending(&p, "s_cbranch_execnz")]);
    assert!(paired(&native, &p, &wide, false).is_ok(), "the spin form emits it");
    refused_as(paired(&native, &p, &wide, true), lower::PAIR_UNPROVEN, "the latch per console wave");
    let (wave, shader, looped) =
        paired(&native, &p, &HashSet::new(), true).expect("the latch per lane: emitted as group barriers");
    assert_eq!((wave, looped), (Some(32), 0));
    for s0 in [0u32, 1000] {
        let Some(got) = run(&shader, [s0, 0], false) else { return };
        assert_eq!(got, console(&native, &[s0, 0, 0, 0]), "s0 = {s0:#x}");
    }
}

/// Check 3, a wait after a partial return at the loop limit: a loop closed by a plain jump
/// (`s_branch`) that each lane leaves when its rounds are done (`s_cbranch_execz` out), then a
/// read of lane 63. At the iteration limit a thread still in the loop returns
/// (`abort_block`) while the others go on to the read's barrier: refused as group barriers,
/// though the read's block is reached by every thread that is left.
#[test]
fn a_meeting_after_a_loop_that_may_return_threads_at_its_limit_is_refused_as_a_barrier() {
    let mut body = vec![
        DWORD_OF_T,
        V2_IS_S0_PLUS_T,
        0x2C0E_0084, // v_lshrrev_b32 v7, 4, v0
        0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7: rounds (t >> 4) + 1
        V5_IS_0,
        SAVE_EXEC,
        ROUNDS_LEFT, // loop:
        0xBF88_0003, // s_cbranch_execz done
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        V5_PLUS_1,
        0xBF82_FFFB, // s_branch loop
        RESTORE_EXEC, // done:
    ];
    body.extend(READ_LANE_63);
    body.push(ADD_S7);
    body.extend(STORE_V2);
    let (native, p) = group_of_two(&body, 0);
    assert!(paired(&native, &p, &HashSet::new(), false).is_ok(), "the spin form emits it");
    refused_as(paired(&native, &p, &HashSet::new(), true), lower::PAIR_CUT, "a return at the limit");
}

/// Check 4: a read of lane 63 behind a scalar branch on a dword the program loads from its
/// UAV (`s_buffer_load_dword`, `s_cmp_eq_u32`, `s_cbranch_scc1`). Writable memory may hold
/// another value when the other console wave loads it, and the two then set opposite SCCs:
/// not group-uniform, refused as group barriers. The same branch on a root constant is.
#[test]
fn a_meeting_behind_a_scalar_branch_on_a_per_wave_load_is_refused_as_a_barrier() {
    let program = |load: &[u32], test: u32| {
        let mut body = vec![DWORD_OF_T, V2_IS_S0_PLUS_T];
        body.extend(load);
        body.extend([test, 0xBF85_0003]); // s_cbranch_scc1 past the read
        body.extend(READ_LANE_63);
        body.push(ADD_S7);
        body.extend(STORE_V2);
        group_of_two(&body, 0)
    };
    // s_buffer_load_dword s20, s[8:11], 0 and s_waitcnt lgkmcnt(0); s_cmp_eq_u32 s20, 0
    let (native, p) = program(&[0xF420_0504, 0xFA00_0000, 0xBF8C_C07F], 0xBF06_8014);
    assert!(paired(&native, &p, &HashSet::new(), false).is_ok(), "the spin form emits it");
    refused_as(paired(&native, &p, &HashSet::new(), true), lower::PAIR_UNPROVEN, "a branch on a loaded dword");
    // s_cmp_eq_u32 s1, 0: a root constant, alike in the whole group
    let (native, p) = program(&[], 0xBF06_8001);
    let (_, _, looped) = paired(&native, &p, &HashSet::new(), true).expect("a branch on a root constant: emitted");
    assert_eq!(looped, 0);
}

/// Check 5: one exchange (a read of lane 63) in a loop of s1 turns on a scalar counter, which
/// every thread of the group goes round alike. As group barriers its one meeting a turn is one
/// barrier (the spin form's is three, two of them around its poll) and the exchange buffer
/// flips each turn; the values are the console's. The barrier stands on the loop, so the run
/// is on the software adapter only (G-106), which has no 32-lane waves (G-127): there the
/// values are not compared.
#[test]
fn an_exchange_in_a_loop_of_several_turns_flips_its_buffer_each_turn() {
    let mut body = vec![
        DWORD_OF_T,
        V2_IS_S0_PLUS_T,
        0xBE94_0380, // s_mov_b32 s20, 0
    ];
    body.extend(READ_LANE_63); // loop:
    body.extend([
        ADD_S7,
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_0114, // s_cmp_lt_u32 s20, s1
        0xBF85_FFFA, // s_cbranch_scc1 loop
    ]);
    body.extend(STORE_V2);
    let (native, p) = group_of_two(&body, 0);
    let (_, _, spin) = paired(&native, &p, &HashSet::new(), false).expect("the spin form emits it");
    let (wave, shader, looped) = paired(&native, &p, &HashSet::new(), true).expect("emitted as group barriers");
    assert_eq!(wave, Some(32));
    assert_eq!(looped, 1, "one group barrier on the loop (the spin form has {spin})");
    for (s0, turns) in [(0u32, 1u32), (1000, 3), (7, 4)] {
        let Some(got) = run(&shader, [s0, turns], true) else { return };
        assert_eq!(got, console(&native, &[s0, turns, 0, 0]), "s0 = {s0:#x}, {turns} turns");
    }
}
