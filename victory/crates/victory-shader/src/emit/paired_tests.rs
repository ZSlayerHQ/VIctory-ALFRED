//! The paired form ([`super::paired`]): console wave64 compute programs on a host of 32-lane
//! waves, each console wave run by one host wave whose threads hold two lanes each. Every
//! program here is one `world`'s forms refuse on such a host, or translate only with each half
//! of the wave on its own; each is translated with the paired form off (as `world` does) and on,
//! validated by `dxil.dll` when present, and run on the test device when it takes waves of 32.
//! The software adapter does not (its waves have 4 lanes): there the run is reported skipped.
//! Every instruction word below is assembled for its test in the console's shape; none is a
//! dump of a title program.
//! after the translation and the validator.

use super::host::{Dispatch, Root, View};
use super::{HostShader, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A wave64 compute program of `threads` threads (the thread index in v0) with 512 bytes of
/// LDS and `vgprs` VGPRs: a root SRV at s[0:1] (user data 0) and the table of the output UAV
/// at s2 (user data 2).
fn native(words: &[u32], threads: u32, vgprs: u16) -> NativeProgram {
    let param = |root, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, root, dwords, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let mut code = words.to_vec();
    code.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs,
        layout: vec![param(RootType::Descriptor, 0, 0), param(RootType::Table, 2, 1)],
        used: vec![
            used(ResourceType::Srv, RootType::Descriptor, 2, 0, 0),
            used(ResourceType::Uav, RootType::Table, 4, 2, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 3 << 1, // 512 bytes of LDS; 3 user SGPRs: s0..s2
            threads: [threads, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A host whose waves are all 32 lanes (an RTX 5090's answer).
const HOST_32: super::HostCaps = super::HostCaps::Known(super::HostWaves { min: 32, max: 32, wave_ops: true });

/// `program` as the runtime translates it for [`HOST_32`], with the paired form on or off
/// (off: as `world` translates it), by [`super::emit_for`] with every form it tries.
fn emitted(program: &NativeProgram, pairs: bool) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&program.code, program.wave64, program.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &program.layout, program.stage, Some(&st))?;
    let io = io::raise(&p, program)?;
    let on = super::TEST_PAIRS.replace(Some(pairs));
    let shader = super::emit_for(&c, program, &p, &st, &io, &res, HOST_32).map(|(s, _)| s);
    super::TEST_PAIRS.set(on);
    shader
}

/// The paired form alone ([`super::paired`]), on a host of 32-lane waves.
fn paired_only(program: &NativeProgram) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&program.code, program.wave64, program.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &program.layout, program.stage, Some(&st))?;
    let io = io::raise(&p, program)?;
    let host = lower::TEST_HOST_WAVE.replace(Some(32));
    let shader = super::paired(&c, program, &p, &io, &res).map(|(s, _)| s);
    lower::TEST_HOST_WAVE.set(host);
    shader
}

/// The wave size and the thread count of an emitted container (`PSV0`'s fewest and most
/// lanes; its group size from the runtime info's `NumThreads`).
fn wave_and_threads(dxil: &[u8]) -> (u32, u32) {
    let c = Container::parse(dxil).expect("the emitted container");
    let psv = c.part(FourCC::PSV0).expect("a PSV0 part").data;
    let u32_at = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().expect("4 bytes"));
    let (min, max) = (u32_at(4 + 16), u32_at(4 + 20));
    assert_eq!(min, max, "one wave size");
    // (PSV0 runtime info 2 holds NumThreadsX, Y, Z after the 24 bytes of info 0 and 12 of 1)
    let size = u32_at(0) as usize;
    let threads = if size >= 48 { u32_at(4 + 36) * u32_at(4 + 40) * u32_at(4 + 44) } else { 0 };
    (min, threads)
}


/// What `world`'s forms make of a program on a host of 32-lane waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum World {
    /// Refused.
    Refuses,
    /// Translated, in a form other than the paired one (two host waves to a console wave,
    /// each acting on its own where its exact forms refuse the program, or with the float
    /// sums in the host's order).
    Translates,
}

/// `program` translated in the paired form (`world` being what `world`'s forms make of it),
/// validated when `dxil.dll` is there, and run on the test device with `srv` behind the root
/// SRV: one output word per thread. `None` where the device takes no waves of 32 (the
/// software adapter), reported skipped.
fn paired_run(name: &str, program: &NativeProgram, world: World, srv: &[u32]) -> Option<Vec<u32>> {
    let StageIo::Cs(cs) = program.io else { unreachable!() };
    let group = cs.threads.iter().product::<u32>();
    match (world, emitted(program, false)) {
        (World::Refuses, r) => assert!(r.is_err(), "{name}: world's forms refuse it on a 32-lane host"),
        (World::Translates, Ok(s)) => {
            assert_ne!(wave_and_threads(&s.dxil).1, group / 2, "{name}: world's form is not the paired one")
        }
        (World::Translates, Err(e)) => panic!("{name}: world's forms refuse it: {e}"),
    }
    // (with the switch on, the paired form where `world`'s forms refuse the program or are not
    // exact: else `world`'s form stays)
    let on = emitted(program, true).unwrap_or_else(|e| panic!("{name}: not translated with the switch on: {e}"));
    if world == World::Refuses {
        assert_eq!(wave_and_threads(&on.dxil).1, group / 2, "{name}: the switch on gives the paired form");
    }
    let shader = paired_only(program).unwrap_or_else(|e| panic!("{name}: not translated paired: {e}"));
    let (wave, threads) = wave_and_threads(&shader.dxil);
    assert_eq!(wave, 32, "{name}: one host wave of 32 for each console wave");
    assert_eq!(threads, group / 2, "{name}: two console lanes a thread");
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = super::valu_tests::hardware_host(name)?;
    let (root_sig, pso) = match host.compute_pipeline(&shader) {
        Ok(p) => p,
        Err(e) if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") => {
            eprintln!("{name}: skipped on WARP ({e})");
            return None;
        }
        Err(e) => panic!("{name}: compute_pipeline: {e}"),
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let mut roots = vec![(0, Root::Constants(ud))];
    for n in 0..shader.root_srvs.len() {
        roots.push((1 + n as u32, Root::View { srv: true, data: srv.to_vec() }));
    }
    let views = [View::RawUav(group)];
    let got = host
        .dispatch(&root_sig, &pso, &Dispatch { roots, views: &views, samplers: &[], groups: [1, 1, 1] })
        .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
    Some(got[0].clone())
}

/// The console's wave of 64 lanes in step through a loop each lane leaves after
/// `(t & 7) + 1` passes: `step` is one pass of the lanes still in it, `acc` their values.
fn console_loop(acc: &mut [u32; 64], mut step: impl FnMut(u32, &[bool; 64], &mut [u32; 64])) {
    let mut active = [true; 64];
    for pass in 0.. {
        if !active.iter().any(|&a| a) {
            break;
        }
        step(pass, &active, acc);
        for (t, a) in active.iter_mut().enumerate() {
            *a = *a && (t as u32 & 7) > pass;
        }
    }
}

/// LDS shared across the lanes of one console wave with no barrier, in a loop each lane leaves
/// after `(t & 7) + 1` passes: each pass, every lane still in it writes its sum to its word and
/// adds the word of lane `t + 32` (the other lane of its own host thread; a lane that has left
/// keeps its last word there). The console runs the wave's lanes in step; the host wave meets
/// between the accesses, the loop being the wave's.
const LDS_LOOP: [u32; 27] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x4A06_00A0, // v_add_nc_u32 v3, 32, v0
    0x3606_06BF, // v_and_b32 v3, 63, v3
    0x3406_0682, // v_lshlrev_b32 v3, 2, v3
    0x7E08_0300, // v_mov_b32 v4, v0
    0x360C_0087, // v_and_b32 v6, 7, v0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0xBE94_047E, // s_mov_b64 s[20:21], exec
    0xD834_0000,
    0x0000_0401, // ds_write_b32 v1, v4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD8D8_0000,
    0x0500_0003, // ds_read_b32 v5, v3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x4A08_0905, // v_add_nc_u32 v4, v5, v4
    0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7
    0x7D82_0F06, // v_cmp_lt_u32 vcc, v6, v7
    0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
    0xBF89_FFF5, // s_cbranch_execnz to the loop head
    0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x8004_0401, // buffer_store_dword v4, v1, s[16:19], 0 offen
];

fn lds_loop_on_the_console() -> Vec<u32> {
    let mut acc: [u32; 64] = std::array::from_fn(|t| t as u32);
    let mut lds = [0u32; 64];
    console_loop(&mut acc, |_, active, acc| {
        for t in 0..64 {
            if active[t] {
                lds[t] = acc[t];
            }
        }
        let read: Vec<u32> = (0..64).map(|t| lds[(t + 32) & 63]).collect();
        for t in 0..64 {
            if active[t] {
                acc[t] = acc[t].wrapping_add(read[t]);
            }
        }
    });
    acc.to_vec()
}

#[test]
fn paired_lanes_share_lds_across_the_halves_in_a_lane_loop() {
    let program = native(&LDS_LOOP, 64, 16);
    if let Some(got) = paired_run("lds loop", &program, World::Translates, &[0; 64]) {
        assert_eq!(got, lds_loop_on_the_console());
    }
}

/// The same program in a group of two console waves: the host wave of each would have to
/// meet its own threads between the accesses, and a group barrier there waits for the other
/// wave too, which goes its own way. Refused, not fenced.
#[test]
fn paired_lanes_refuse_lds_shared_without_a_barrier_in_a_group_of_two_waves() {
    let program = native(&LDS_LOOP, 128, 16);
    match paired_only(&program) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::PAIRED_LDS_APART),
        other => panic!("not refused for its LDS: {:?}", other.map(|_| ())),
    }
}

/// Lane reads that reach the other half: `v_readlane_b32` of lane 40, `v_readfirstlane_b32`
/// with only lanes 38..63 in EXEC (the first is an upper lane), and `ds_bpermute_b32` from
/// lane `63 - t`. Every thread's word: `(63 - t) | 40 << 8 | 38 << 16`.
const LANE_READS: [u32; 25] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xD760_0014,
    0x0001_5100, // v_readlane_b32 s20, v0, 40
    0x7D82_00A5, // v_cmp_lt_u32 vcc, 37, v0
    0xBE96_246A, // s_and_saveexec_b64 s[22:23], vcc
    0x7E2A_0500, // v_readfirstlane_b32 s21, v0
    0xBEFE_0416, // s_mov_b64 exec, s[22:23]
    0x4C06_00BF, // v_sub_nc_u32 v3, 63, v0
    0x3406_0682, // v_lshlrev_b32 v3, 2, v3
    0xDACC_0000,
    0x0200_0003, // ds_bpermute_b32 v2, v3, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E0A_0214, // v_mov_b32 v5, s20
    0x340A_0A88, // v_lshlrev_b32 v5, 8, v5
    0x7E0C_0215, // v_mov_b32 v6, s21
    0x340C_0C90, // v_lshlrev_b32 v6, 16, v6
    0x380A_0D05, // v_or_b32 v5, v5, v6
    0x380A_0505, // v_or_b32 v5, v5, v2
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x8004_0501, // buffer_store_dword v5, v1, s[16:19], 0 offen
];

#[test]
fn paired_lane_reads_reach_the_other_half() {
    let program = native(&LANE_READS, 64, 16);
    let want: Vec<u32> = (0..64u32).map(|t| (63 - t) | 40 << 8 | 38 << 16).collect();
    if let Some(got) = paired_run("lane reads", &program, World::Refuses, &[0; 64]) {
        assert_eq!(got, want);
    }
}

/// Lane masks read as data in a loop each lane leaves after `(t & 7) + 1` passes: each pass,
/// the lanes still in it add the bit count of EXEC and the first lane of `t & 3 == 1` among
/// them (shifted by 8), and XOR in EXEC's high dword.
const MASK_LOOP: [u32; 26] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x3606_0083, // v_and_b32 v3, 3, v0
    0x360C_0087, // v_and_b32 v6, 7, v0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0x7E08_0280, // v_mov_b32 v4, 0
    0xBE94_047E, // s_mov_b64 s[20:21], exec
    0xBE98_107E, // s_bcnt1_i32_b64 s24, exec
    0x7D84_0681, // v_cmp_eq_u32 vcc, 1, v3
    0xBE99_146A, // s_ff1_i32_b64 s25, vcc
    0x8F19_8819, // s_lshl_b32 s25, s25, 8
    0x8018_1918, // s_add_u32 s24, s24, s25
    0x7E10_027F, // v_mov_b32 v8, exec_hi
    0x4A08_0818, // v_add_nc_u32 v4, s24, v4
    0x3A08_0908, // v_xor_b32 v4, v8, v4
    0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7
    0x7D82_0F06, // v_cmp_lt_u32 vcc, v6, v7
    0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
    0xBF89_FFF4, // s_cbranch_execnz to the loop head
    0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x8004_0401, // buffer_store_dword v4, v1, s[16:19], 0 offen
];

fn mask_loop_on_the_console() -> Vec<u32> {
    let mut acc = [0u32; 64];
    console_loop(&mut acc, |_, active, acc| {
        let exec: u64 = (0..64).filter(|&t| active[t]).map(|t| 1u64 << t).sum();
        let vcc: u64 = (0..64).filter(|&t| active[t] && t & 3 == 1).map(|t| 1u64 << t).sum();
        let first = if vcc == 0 { u32::MAX } else { vcc.trailing_zeros() };
        let s24 = exec.count_ones().wrapping_add(first << 8);
        for t in 0..64 {
            if active[t] {
                acc[t] = acc[t].wrapping_add(s24) ^ (exec >> 32) as u32;
            }
        }
    });
    acc.to_vec()
}

#[test]
fn paired_lane_masks_are_the_two_halves_ballots() {
    let program = native(&MASK_LOOP, 64, 16);
    if let Some(got) = paired_run("mask loop", &program, World::Translates, &[0; 64]) {
        assert_eq!(got, mask_loop_on_the_console());
    }
}

/// Two console waves in a group: in a lane branch (lanes 0..39 of each wave) each lane takes
/// the thread index of lane `39 - l` of its wave by `ds_bpermute_b32` (lanes 0..7 read upper
/// lanes 32..39), the others 999; after the join every lane writes its word to LDS, the group
/// meets at the program's `s_barrier`, and each lane reads the other wave's word of its lane.
/// The waves are numbered at entry (`wave_base`); no other barrier is added.
const TWO_WAVES: [u32; 29] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7E04_02FF,
    0x0000_03E7, // v_mov_b32 v2, 999
    0x3606_00BF, // v_and_b32 v3, 63, v0
    0x7D88_06A8, // v_cmp_gt_u32 vcc, 40, v3
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0005, // s_cbranch_execz past the body
    0x4C08_06A7, // v_sub_nc_u32 v4, 39, v3
    0x3408_0882, // v_lshlrev_b32 v4, 2, v4
    0xDACC_0000,
    0x0200_0004, // ds_bpermute_b32 v2, v4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    0xD834_0000,
    0x0000_0201, // ds_write_b32 v1, v2
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0x3A06_00C0, // v_xor_b32 v3, 64, v0
    0x3406_0682, // v_lshlrev_b32 v3, 2, v3
    0xD8D8_0000,
    0x0500_0003, // ds_read_b32 v5, v3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x8004_0501, // buffer_store_dword v5, v1, s[16:19], 0 offen
];

#[test]
fn paired_waves_meet_at_the_programs_barrier_after_a_lane_branch() {
    let program = native(&TWO_WAVES, 128, 16);
    let word = |u: u32| if u & 63 < 40 { (u & 64) + 39 - (u & 63) } else { 999 };
    let want: Vec<u32> = (0..128u32).map(|t| word(t ^ 64)).collect();
    if let Some(got) = paired_run("two waves", &program, World::Refuses, &[0; 128]) {
        assert_eq!(got, want);
    }
}

/// A wave sum of floats in the steps of [`super::wave::reduction`], its first two steps
/// swapped: lanes two apart are added first, then neighbours, then the rest of the butterfly.
/// Lane 0 holds 2^25, lane 1 1.0, lane 2 -2^25, the others 0: in this order the sum is 1.0
/// (`0x3f80_0000`), neighbours first it is 0 (2^25 + 1 rounds to 2^25).
fn float_sum() -> Vec<u32> {
    use super::wave::reduction::{NEIGHBOURS, TWO_APART, V_ADD_F32, WORK, dpp8, words};
    let mut sum = words(V_ADD_F32, 7, 7); // v7 over the wave into s7
    (sum[4], sum[6]) = (dpp8(TWO_APART, WORK), dpp8(NEIGHBOURS, WORK));
    let mut w = vec![
        0x3412_0082, // v_lshlrev_b32 v9, 2, v0
        0xBE88_0300, // s_mov_b32 s8, s0
        0xBE89_0301, // s_mov_b32 s9, s1
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xE030_1000,
        0x8002_0709, // buffer_load_dword v7, v9, s[8:11], 0 offen
        0xBF8C_0000, // s_waitcnt 0
    ];
    w.extend(sum);
    w.extend([
        0x7E0A_0207, // v_mov_b32 v5, s7
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8004_0509, // buffer_store_dword v5, v9, s[16:19], 0 offen
    ]);
    w
}

/// The butterfly over the lane bits `order` of `terms` (one per lane, 64), as f32 additions.
fn butterfly(terms: &[f32], order: &[u32]) -> f32 {
    let mut v = terms.to_vec();
    for &d in order {
        v = (0..64).map(|l| v[l] + v[l ^ (1 << d)]).collect();
    }
    v[0]
}

#[test]
fn a_paired_float_sum_is_added_in_the_consoles_order() {
    let program = native(&float_sum(), 64, 16);
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    let found = super::wave::sections(&p, 0);
    let section = found.values().next().expect("the section");
    let total = p.blocks[0].insts.iter().rfind(|i| i.name() == "v_readlane_b32").unwrap().write(ir::Reg::S(7)).unwrap();
    assert_eq!(section.butterflies.get(&total).map(Vec::as_slice), Some(&[1, 0, 2, 3, 4, 5][..]), "lanes two apart first");
    let mut terms = [0f32; 64];
    terms[..3].copy_from_slice(&[33_554_432.0, 1.0, -33_554_432.0]);
    assert_eq!(butterfly(&terms, &[1, 0, 2, 3, 4, 5]).to_bits(), 0x3F80_0000);
    assert_eq!(butterfly(&terms, &[0, 1, 2, 3, 4, 5]).to_bits(), 0, "these terms tell the two orders apart");
    let mut srv = vec![0u32; 64];
    for (s, t) in srv.iter_mut().zip(terms) {
        *s = t.to_bits();
    }
    if let Some(got) = paired_run("float sum", &program, World::Translates, &srv) {
        assert_eq!(got, vec![0x3F80_0000; 64]);
    }
}

/// The order the paired form replays is the one a sum is added in: the compiler's own
/// (neighbours first) gives the butterfly over bits 0 to 5.
#[test]
fn a_float_sum_in_the_compilers_order_is_the_butterfly_over_bits_0_to_5() {
    let mut words = float_sum();
    // (the first two steps back in the compiler's order)
    words.swap(12, 14);
    let program = native(&words, 64, 16);
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    let found = super::wave::sections(&p, 0);
    let section = found.values().next().expect("the section");
    let total = p.blocks[0].insts.iter().rfind(|i| i.name() == "v_readlane_b32").unwrap().write(ir::Reg::S(7)).unwrap();
    assert_eq!(section.butterflies.get(&total).map(Vec::as_slice), Some(&[0, 1, 2, 3, 4, 5][..]));
}

/// [`emitted`], with the form the ladder handed over (`super::take_wave_forms`): the first
/// that translated for waves of 32; `None` where none of the ladder's forms did.
fn emitted_in(program: &NativeProgram, pairs: bool) -> (HostShader, Option<String>) {
    let was = super::TEST_WAVE_FORMS.replace(true);
    let shader = emitted(program, pairs);
    let forms = super::take_wave_forms();
    super::TEST_WAVE_FORMS.set(was);
    let shader = shader.unwrap_or_else(|e| panic!("not translated (pairs {pairs}): {e}\n  {}", forms.join("\n  ")));
    let taken = forms
        .iter()
        .filter_map(|f| f.rsplit_once(": wave "))
        .find(|(_, lanes)| *lanes != "64")
        .map(|(form, _)| form.to_owned());
    (shader, taken)
}

/// Where both the meeting form and the paired form take a program, the meeting form (exact
/// and reviewed) stands, the paired form on or off: the meeting's float sum over the wave
/// behind a lane branch (`meet_tests`), which the paired form takes too and was tried for
/// first of all forms (a wave sum of floats, added in the console's order) before the meeting
/// form added such sums in the program's order. With the paired form on, the translation is
/// the one it is with the paired form off, and it is the meeting's.
#[test]
fn the_meeting_form_stands_where_the_paired_form_also_would() {
    use super::meet_tests as m;
    let mut body = m::lane_terms(&[(0, 33_554_432.0f32), (1, 1.0), (2, -33_554_432.0)]);
    body.extend(m::wave_sum_f32_v7_to_s7(false));
    body.push(m::vop1(m::V_MOV_B32, 10, m::sgpr(7))); // v10 = the sum
    body.extend(m::store(10, 0));
    let program = m::native(&body, false);
    let paired = paired_only(&program).expect("the paired form takes it");
    assert_eq!(wave_and_threads(&paired.dxil), (32, 32), "the paired form: a host wave of 32, half the threads");
    let (off, taken_off) = emitted_in(&program, false);
    let taken_off = taken_off.expect("a form of the ladder, the paired form off");
    assert!(taken_off.starts_with("meeting"), "the paired form off: {taken_off}");
    assert_eq!(m::required_wave(&off.dxil), None, "the meeting form asks no wave size");
    let (on, taken_on) = emitted_in(&program, true);
    assert_eq!(taken_on.as_deref(), Some(taken_off.as_str()), "the paired form on: the same form");
    assert!(on.dxil == off.dxil, "the paired form on: the same translation");
}

#[test]
fn paired_lanes_refuse_a_group_that_does_not_fill_its_waves() {
    let program = native(&LANE_READS, 96, 16);
    match paired_only(&program) {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "paired lanes in a thread group that does not fill its console waves")
        }
        other => panic!("not refused: {:?}", other.map(|_| ())),
    }
}

/// A program of 129 VGPRs needs 258 registers a thread for its two lanes: over NVIDIA's 255.
#[test]
fn paired_lanes_refuse_a_program_over_255_registers_a_thread() {
    for (vgprs, refused) in [(127, false), (128, true), (129, true)] {
        let r = paired_only(&native(&LANE_READS, 64, vgprs));
        match r {
            Err(ShaderError::Unsupported { what, .. }) if refused => assert_eq!(what, "paired lanes over 255 registers a thread"),
            Ok(_) if !refused => {}
            other => panic!("{vgprs} VGPRs: {:?}", other.map(|_| ())),
        }
    }
}

/// Only console wave64 compute programs are paired: a pixel program of 64 lanes (derivatives
/// and helper lanes are its quads') and a compute program of 32-lane waves are not, with the
/// switch on.
#[test]
fn a_pixel_program_and_a_wave32_program_are_not_paired() {
    let host = HOST_32;
    let on = super::TEST_PAIRS.replace(Some(true));
    let mut ps = native(&LANE_READS, 64, 16);
    ps.stage = HwStage::Ps;
    assert!(!super::pairs_for(&ps, host));
    let mut cs32 = native(&LANE_READS, 64, 16);
    cs32.wave64 = false;
    assert!(!super::pairs_for(&cs32, host));
    assert!(super::pairs_for(&native(&LANE_READS, 64, 16), host));
    super::TEST_PAIRS.set(on);
}

/// D3D12 fixes a program's wave size but promises no more about how a group is cut into
/// waves: only a host whose every wave has 32 lanes takes the paired form, and the program
/// checks at its entry that its group runs as the form takes it
/// (`Lower::paired_entry_check`). A host of 4 to 32 lanes (`VICTORY_HOST_MAX_WAVE=32`), of
/// 32 to 64, or one not known does not.
#[test]
fn only_a_host_of_32_lane_waves_takes_the_paired_form() {
    let on = super::TEST_PAIRS.replace(Some(true));
    let cs = native(&LANE_READS, 64, 16);
    let host = |min, max| super::HostCaps::Known(super::HostWaves { min, max, wave_ops: true });
    assert!(super::pairs_for(&cs, host(32, 32)));
    assert!(!super::pairs_for(&cs, super::HostCaps::at_most(32)));
    assert!(!super::pairs_for(&cs, host(32, 64)));
    assert!(!super::pairs_for(&cs, super::HostCaps::Unknown));
    super::TEST_PAIRS.set(on);
}

/// On unless `VICTORY_WAVE_PAIRS` is set to an off spelling; with it off a program the
/// paired form takes is translated as before, or refused as before.
#[test]
fn the_off_spellings_turn_the_paired_form_off_and_it_is_on_by_default() {
    for off in ["", "0", "false", "OFF", "no", " 0 ", "  "] {
        assert!(!super::pairs_value_on(std::ffi::OsStr::new(off)), "{off:?} is off");
    }
    for on in ["1", "true", "on", " 1 ", "2"] {
        assert!(super::pairs_value_on(std::ffi::OsStr::new(on)), "{on:?} is on");
    }
    if std::env::var_os("VICTORY_WAVE_PAIRS").is_none() {
        assert!(super::pairs_on(), "on by default");
    }
    let kept = super::TEST_PAIRS.replace(Some(false));
    assert!(!super::pairs_on(), "off for a test's thread");
    super::TEST_PAIRS.set(kept);
    assert!(emitted(&native(&LANE_READS, 64, 16), false).is_err());
}

/// A thread group of more threads than a dword counts is refused, as `world` refuses it,
/// with the paired form off and on (the merge's review, 1: `Lower::new` multiplied the group
/// unchecked for the paired form's `one_wave` even with the form off, and panicked).
#[test]
fn a_group_past_a_dword_of_threads_is_refused_with_pairs_off_and_on() {
    let mut program = native(&LANE_READS, 64, 16);
    let StageIo::Cs(cs) = &mut program.io else { unreachable!() };
    cs.threads = [65536, 65536, 1];
    for pairs in [false, true] {
        match emitted(&program, pairs) {
            Err(ShaderError::Unsupported { .. }) => {}
            Err(e) => panic!("pairs {pairs}: refused otherwise: {e}"),
            Ok(_) => panic!("pairs {pairs}: translated"),
        }
    }
    assert!(matches!(paired_only(&program), Err(ShaderError::Unsupported { .. })), "the paired form alone refuses it");
}
