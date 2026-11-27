//! Scalar pairs parked in two VGPR lanes that hold other scalars elsewhere (`ir::unspill`,
//! `ir::lane_pairs`), and the helpers the tests of console waves of 64 on 32-lane hosts share
//! with [`super::early_out_tests`]: a program's form as the host gets it, and its dispatch on
//! the software adapter only (which makes no pipeline of `WaveSize(32)`: reported skipped
//! there).
//!
//! The webs are off by default (`VICTORY_WAVE_WEBS=1`), and so are the stricter rules of what
//! a park is and of masks in partial waves (`VICTORY_WAVE_GUARDS=1`): each test here sets, for
//! its own thread and until it ends, the switch its rule is behind. What the translator does
//! with neither set is `world`'s, asserted in [`super::off_is_off_tests`].

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::wave64_tests::native_of;
use super::{HostShader, lower, switched, validator};
use crate::native::{NativeProgram, RootType};
use crate::{io, ir, resource, structure};

pub(super) fn validated(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
}

/// `p`'s parts for [`super::emit`] and [`super::emit_program`].
pub(super) fn parts(native: &NativeProgram, p: &ir::Program) -> (structure::Structured, resource::Resources, io::Io) {
    let st = structure::structure(p).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    (st, res, io)
}

/// One group of `shader` on `host` with s0, s1 = `constants`: the first `dwords` dwords of
/// the UAV.
pub(super) fn one_group(host: &Host, shader: &HostShader, constants: [u32; 2], dwords: u32) -> Result<Vec<u32>, String> {
    let (root, pso) = host.compute_pipeline(shader).map_err(|e| format!("compute_pipeline: {e}"))?;
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &[View::RawUav(dwords)], samplers: &[], groups: [1, 1, 1] },
    )?;
    Ok(out[0].clone())
}

/// `p` (a form of `native`) as [`super::emit_program`] hands it to a host of 32-lane waves,
/// and the wave size it asks for: 32 for a program with exchanges (without that a device
/// runs it in waves of its own size and its exchanges pair the wrong lanes; the software
/// adapter's waves have 4), none for one whose lanes all decide alone.
pub(super) fn emitted(native: &NativeProgram, p: &ir::Program) -> (HostShader, Option<u32>) {
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let (st, res, io) = parts(native, p);
    let shader = super::emit_program(&c, native, p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let (shader, wave) = shader.expect("the form translates");
    assert!(matches!(wave, None | Some(32)), "for host waves of 32: {wave:?}");
    assert_eq!(super::host_wave_tests::required_wave(&shader.dxil), wave, "the container asks for that");
    validated(&shader);
    (shader, wave)
}

/// `shader` (from [`emitted`]) on the software adapter, never on a card (its group
/// barriers): every case's 128 dwords against the console's. Reported skipped where the
/// adapter makes no pipeline of waves of 32.
pub(super) fn on_the_software_adapter(name: &str, shader: &HostShader, cases: &[([u32; 2], Vec<u32>)]) {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("{name}: no SM 6.6 software device; skipped");
        return;
    };
    if let Err(e) = host.compute_pipeline(shader) {
        eprintln!("{name}: skipped on the software adapter, which has no waves of 32 ({e})");
        return;
    }
    for (constants, expected) in cases {
        let got = one_group(&host, shader, *constants, 128).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(&got, expected, "{name}: s0, s1 = {constants:?}");
    }
    eprintln!("{name}: every thread of every case as the console stores it, on the software adapter");
}

/// `v_writelane_b32 vN, sX, K`.
pub(super) fn writelane(n: u32, x: u32, lane: u32) -> Vec<u32> {
    vec![0xD761_0000 | n, (128 + lane) << 9 | x]
}

/// `v_readlane_b32 sX, vN, K`.
pub(super) fn readlane(x: u32, n: u32, lane: u32) -> Vec<u32> {
    vec![0xD760_0000 | x, (128 + lane) << 9 | (256 + n)]
}

fn parked(words: &[Vec<u32>]) -> ir::Program {
    let native = super::wave64_tests::native(&words.concat());
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

pub(super) fn instructions<'a>(p: &'a ir::Program, name: &str) -> Vec<&'a ir::Inst> {
    p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).collect()
}

pub(super) fn regs<T>(of: &[(ir::Reg, T)]) -> Vec<ir::Reg> {
    of.iter().map(|r| r.0).collect()
}

#[test]
fn a_pair_parked_in_lanes_that_hold_other_scalars_elsewhere_moves_as_a_pair() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    // s[20:21] goes into lanes 4 and 5 of v10 and comes back into s[22:23]; later lane 4
    // holds s30 for a while: no lane of the two is the pair's alone
    let p = parked(&[
        writelane(10, 20, 4),
        writelane(10, 21, 5),
        vec![0x7E04_0281], // v_mov_b32 v2, 1
        readlane(22, 10, 4),
        readlane(23, 10, 5),
        writelane(10, 30, 4),
        readlane(31, 10, 4),
    ]);
    let wide = instructions(&p, "s_mov_b64");
    let [save, restore] = wide[..] else { panic!("one pair save and one pair restore: {} moves", wide.len()) };
    assert_eq!(regs(&save.reads), [ir::Reg::S(20), ir::Reg::S(21)]);
    assert_eq!(regs(&restore.writes), [ir::Reg::S(22), ir::Reg::S(23)]);
    assert_eq!(regs(&restore.reads), regs(&save.writes));
    // the single scalar has its own lane register, not the pair's
    let narrow = instructions(&p, "s_mov_b32");
    let lone_save = narrow.iter().find(|i| regs(&i.reads) == [ir::Reg::S(30)]).expect("the save of s30");
    let lone_restore = narrow.iter().find(|i| regs(&i.writes) == [ir::Reg::S(31)]).expect("the restore into s31");
    assert_eq!(regs(&lone_restore.reads), regs(&lone_save.writes));
    assert!(regs(&lone_save.writes).iter().all(|r| !regs(&save.writes).contains(r)));
}

#[test]
fn a_parked_pair_another_save_or_restore_shares_stays_two_scalars() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    let cases = [
        (
            "a lone restore of the lane",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
                readlane(24, 10, 4),
            ],
        ),
        (
            "another save of the lane on one path to the restore",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                vec![0xBF06_8001, 0xBF85_0002], // s_cmp_eq_u32 s1, 0 and s_cbranch_scc1 over the save
                writelane(10, 30, 4),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
            ],
        ),
    ];
    for (name, words) in cases {
        let p = parked(&words);
        assert!(instructions(&p, "s_mov_b64").is_empty(), "{name}");
        assert!(!instructions(&p, "s_mov_b32").is_empty(), "{name}");
    }
}

/// Two console waves. s20 = 7 is parked in lane 0 of v10 and restored, and between the two an
/// M0-relative read (`v_movrels_b32 v5, v9` with M0 = 1) reads v10 in every lane: lane 0 of
/// each console wave sees the parked 7. Each thread stores what its lane read if it is lane 0
/// of its wave, else 0. Taking the park for a scalar move leaves nothing in the lane for the
/// relative read to see.
pub(super) fn a_lane_parked_under_a_relative_read() -> (NativeProgram, Vec<u32>) {
    let words = [
        vec![0xBE94_0387], // s_mov_b32 s20, 7
        writelane(10, 20, 0),
        vec![
            0xBEFC_0381, // s_mov_b32 m0, 1
            0x7E0A_8709, // v_movrels_b32 v5, v9: v5 = v[9 + m0] = v10
        ],
        readlane(21, 10, 0),
        vec![
            0x3602_00BF, // v_and_b32 v1, 63, v0: the lane in the console wave
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x020A_0A80, // v_cndmask_b32 v5, 0, v5, vcc
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0x38 << 26 | 0x1C << 18 | 1 << 12,
            0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
        ],
    ];
    let expected = (0..128u32).map(|t| if t % 64 == 0 { 7 } else { 0 }).collect();
    (native_of(&words.concat(), true, 128), expected)
}

/// [`a_lane_parked_under_a_relative_read`]: the park stays a lane write, so the relative read
/// finds the scalar. (The software adapter, without waves of 32, skips the dispatch; the card
/// runs it in [`on_a_card_a_relative_read_of_a_parked_lane_reads_the_scalar`].)
#[test]
fn a_parked_lane_an_m0_relative_read_reaches_reads_the_scalar_parked_there() {
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    let (native, expected) = a_lane_parked_under_a_relative_read();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(instructions(&p, "v_writelane_b32").len(), 1, "the park stays in the lane");
    let (shader, _) = emitted(&native, &p);
    on_the_software_adapter("the relative read of a parked lane", &shader, &[([0, 0], expected)]);
}

/// [`a_lane_parked_under_a_relative_read`] on a card with 32-lane waves. Ignored as the other
/// card dispatches of console waves of 64 are (no barrier here waits on a branch, but the
/// exchange form has group barriers):
///   cargo test -p victory-shader --locked -- --ignored --test-threads=1 --nocapture on_a_card_a_relative
#[test]
#[ignore = "dispatches a program with group barriers on the hardware device"]
fn on_a_card_a_relative_read_of_a_parked_lane_reads_the_scalar() {
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = super::wave64_tests::card_with_waves_of(32).expect("a hardware device with 32-lane waves");
    let (native, expected) = a_lane_parked_under_a_relative_read();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let (shader, _) = emitted(&native, &p);
    assert_eq!(one_group(&host, &shader, [0, 0], 128).expect("dispatch"), expected);
}

/// Neon's review, follow-up: a lane parked in a VGPR at or past the program's count (16 here,
/// a count too low for the code) is no park. An M0-relative read from v15 names v15 up to the
/// count only (`ir::effects`), so v16 would stay a park the read never makes ordinary, and
/// the read would miss the scalar parked there.
#[test]
fn a_lane_past_the_programs_vgprs_is_no_park() {
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    let words = [
        vec![0xBE94_0387], // s_mov_b32 s20, 7
        writelane(16, 20, 0),
        vec![
            0xBEFC_0381, // s_mov_b32 m0, 1
            0x7E0A_870F, // v_movrels_b32 v5, v15: v5 = v[15 + m0] = v16
        ],
        readlane(21, 16, 0),
    ];
    let native = native_of(&words.concat(), true, 64);
    assert_eq!(native.vgprs, 16);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(instructions(&p, "v_writelane_b32").len(), 1, "the lane write stays");
    assert_eq!(instructions(&p, "v_readlane_b32").len(), 1, "the lane read stays");
}

/// Two console waves; in `if (t < s0)` EXEC is narrowed to t < s1, saved, parked in lanes 4
/// and 5 of v10 and taken back with the registers used in between: a saved EXEC parked in
/// lanes of a VGPR. Lane 4 holds s30 before the `if`, so lane 4 is no
/// pair's alone. Every thread stores at its own dword: 0 with t >= s0, else 3 with t < s1, 2
/// without.
pub(super) fn a_mask_parked_in_lanes_shared_with_a_scalar() -> NativeProgram {
    a_mask_parked_beside(&[])
}

/// [`a_mask_parked_in_lanes_shared_with_a_scalar`] with `extra` words inside the inner `if`.
fn a_mask_parked_beside(extra: &[u32]) -> NativeProgram {
    let words = [
        vec![0xBE9E_0389], // s_mov_b32 s30, 9
        writelane(10, 30, 4),
        readlane(31, 10, 4),
        vec![
            0x7E0A_0280, // v_mov_b32 v5, 0
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0010, // s_cbranch_execz past the if
            0x7D88_0001, // v_cmp_gt_u32 vcc, s1, v0
            0xBE8E_246A, // s_and_saveexec_b64 s[14:15], vcc
        ],
        writelane(10, 14, 4),
        vec![0x7E0A_0281], // v_mov_b32 v5, 1
        writelane(10, 15, 5),
        vec![
            0xBE8E_0385, // s_mov_b32 s14, 5
            0xBE8F_0386, // s_mov_b32 s15, 6
        ],
        readlane(14, 10, 4),
        vec![0x7E0C_0280], // v_mov_b32 v6, 0
        readlane(15, 10, 5),
        vec![
            0xBEFE_040E, // s_mov_b64 exec, s[14:15]
            0x380A_0A82, // v_or_b32 v5, 2, v5
        ],
        extra.to_vec(),
        vec![
            0xBEFE_040C, // past the if: s_mov_b64 exec, s[12:13]
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0x38 << 26 | 0x1C << 18 | 1 << 12,
            0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
        ],
    ];
    native_of(&words.concat(), true, 128)
}

/// What [`a_mask_parked_in_lanes_shared_with_a_scalar`] stores in each of 128 threads.
fn parked_mask_expected(below: u32, under: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| match (t < below, t < under) {
            (false, _) => 0,
            (true, true) => 3,
            (true, false) => 2,
        })
        .collect()
}

const PARKED_MASK_CASES: [(u32, u32); 6] = [(128, 128), (128, 40), (128, 100), (40, 10), (40, 64), (65, 70)];

/// [`a_mask_parked_in_lanes_shared_with_a_scalar`]: with its two lanes restored as two
/// scalars the saved EXEC is a number to the lowering (a ballot where it was made, which two
/// host waves must exchange, inside the `if`): refused with that program's message.
/// Restored as the pair it is, it is the mask: no ballot, no exchange.
#[test]
fn a_mask_parked_in_lanes_shared_with_a_scalar_comes_back_as_the_mask_it_is() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    let native = a_mask_parked_in_lanes_shared_with_a_scalar();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    for host_wave in [32, 64] {
        let s = super::wave64_tests::lowered_on(&native, &p, &HashSet::new(), host_wave)
            .unwrap_or_else(|e| panic!("host waves of {host_wave}: {e}"));
        validated(&s);
    }
    let wide = instructions(&p, "s_mov_b64");
    assert!(
        wide.iter().any(|i| regs(&i.writes) == [ir::Reg::S(14), ir::Reg::S(15)]),
        "the parked EXEC comes back by one pair move"
    );
    let cases: Vec<_> = PARKED_MASK_CASES.iter().map(|&(b, u)| ([b, u], parked_mask_expected(b, u))).collect();
    // (no exchange is left: every lane decides alone, in waves of any size)
    let (shader, wave) = emitted(&native, &p);
    assert_eq!(wave, None, "no wave size");
    on_the_software_adapter("the parked mask", &shader, &cases);
}

/// One console wave of 40 threads (a partial wave). On `s1 == 0` EXEC is copied to s[14:15]
/// and parked in lanes 4 and 5 of v10, else s[30:31] = -1 is, the same lanes, restored into
/// s[20:21]. Then, again on `s1 == 0`, EXEC is set from s[20:21]; every thread stores s21 at
/// its own dword. The console stores EXEC's high dword (0xFF: lanes 32-39) with `s1 == 0`,
/// else the number 0xFFFF_FFFF.
fn a_web_of_exec_and_a_number_read_as_data(one_move: bool) -> NativeProgram {
    let minus_one: [u32; 2] = if one_move {
        [0xBE9E_04C1, 0xBF80_0000] // s_mov_b64 s[30:31], -1; s_nop
    } else {
        [0xBE9E_03C1, 0xBE9F_03C1] // s_mov_b32 s30, -1; s_mov_b32 s31, -1
    };
    let words = [
        vec![
            minus_one[0],
            minus_one[1],
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF85_0005, // s_cbranch_scc1 to the park of EXEC
        ],
        writelane(10, 30, 4),
        writelane(10, 31, 5),
        vec![
            0xBF82_0005, // s_branch past the park of EXEC
            0xBE8E_047E, // s_mov_b64 s[14:15], exec
        ],
        writelane(10, 14, 4),
        writelane(10, 15, 5),
        readlane(20, 10, 4),
        readlane(21, 10, 5),
        vec![
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF84_0001, // s_cbranch_scc0 over the next
            0xBEFE_0414, // s_mov_b64 exec, s[20:21]
            0x7E0A_0215, // v_mov_b32 v5, s21
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0x38 << 26 | 0x1C << 18 | 1 << 12,
            0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
        ],
    ];
    native_of(&words.concat(), true, 40)
}

/// `native` (one console wave of 40 threads) in the meeting form, which asks no wave size,
/// on the software adapter: each case's 40 dwords as the console stores them, unless the form
/// refuses the program for reading a lane mask as data.
fn in_a_partial_wave(name: &str, native: &NativeProgram, cases: &[([u32; 2], Vec<u32>)], must_translate: bool) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let (st, res, io) = parts(native, &p);
    let met = lower::with_meeting(|| super::emit_program(&c, native, &p, &st, &io, &res));
    lower::TEST_HOST_WAVE.set(was);
    match met {
        Ok((shader, wave)) => {
            assert_eq!(wave, None, "{name}: the meeting form asks no wave size");
            validated(&shader);
            let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let Ok(host) = Host::new(true) else {
                eprintln!("{name}: no SM 6.6 software device; skipped");
                return;
            };
            for (constants, expected) in cases {
                let got = one_group(&host, &shader, *constants, 40).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(&got, expected, "{name}: s0, s1 = {constants:?}");
            }
        }
        Err(crate::ShaderError::Unsupported { what, .. }) if !must_translate => assert_eq!(what, lower::LANE_MASK, "{name}"),
        Err(e) => panic!("{name}: {e}"),
    }
}

/// Neon's review, must-fix 2, and Sol's third review, item 3: [`a_web_of_exec_and_a_number_read_as_data`]
/// in a partial wave. The restored pair is a mask on one way and a number on the other; read
/// as data it was the ballot of the lanes' bits, and a partial wave has no lanes 40-63 to give
/// the number's bits there, whether -1 comes by two dword moves (an opaque mask) or by one
/// pair move (the mask of all lanes, which says nothing of where it came from).
#[test]
fn a_web_of_exec_and_a_number_in_a_partial_wave_is_read_as_the_number() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    for (name, one_move) in [("-1 by two moves", false), ("-1 by one move", true)] {
        let native = a_web_of_exec_and_a_number_read_as_data(one_move);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert!(
            instructions(&p, "s_mov_b64").iter().any(|i| regs(&i.writes) == [ir::Reg::S(20), ir::Reg::S(21)]),
            "{name}: the web comes back by one pair move"
        );
        in_a_partial_wave(name, &native, &[([0, 0], vec![0xFF; 40]), ([0, 1], vec![u32::MAX; 40])], false);
    }
}

/// Sol's third review, item 3: a mask operation that sets bits above the wave's threads.
/// `s_not_b64 s[20:21], exec` in a wave of 40 threads is 0xFFFF_FF00 in its high dword on the
/// console, where every thread's own bit is clear (a ballot gives 0). EXEC itself as data
/// (`s_mov_b64`) is 0xFF there, as its ballot is: that one must translate.
#[test]
fn a_mask_with_bits_above_a_partial_waves_threads_is_read_as_its_number() {
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    let store = [
        0x7E0A_0215, // v_mov_b32 v5, s21
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ];
    let with = |op: u32| native_of(&[&[op][..], &store].concat(), true, 40);
    in_a_partial_wave("EXEC as data", &with(0xBE94_047E), &[([0, 0], vec![0xFF; 40])], true); // s_mov_b64 s[20:21], exec
    in_a_partial_wave("not EXEC", &with(0xBE94_087E), &[([0, 0], vec![0xFFFF_FF00; 40])], false); // s_not_b64 s[20:21], exec
}

/// Sol's third review, item 1 (`f133e35efed2a48a`): the approximate forms (each half-wave on
/// its own) take a program as `world` parks it ([`ir::build_without_webs`]) unless
/// `VICTORY_WAVE_PAIRS_APPROX=1`. That program has the parked EXEC of
/// [`a_mask_parked_in_lanes_shared_with_a_scalar`] back as two scalars, as on `world`; the
/// program with its web has it back by one pair move. (Which form the selection then takes for
/// `f133` is the pack's to show.)
#[test]
fn without_webs_a_parked_mask_comes_back_as_world_has_it() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    let native = a_mask_parked_in_lanes_shared_with_a_scalar();
    let pair = |p: &ir::Program| instructions(p, "s_mov_b64").iter().any(|i| regs(&i.writes) == [ir::Reg::S(14), ir::Reg::S(15)]);
    let webbed = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let world = ir::build_without_webs(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(webbed.webs && pair(&webbed), "with its web: one pair move");
    assert!(!world.webs && !pair(&world), "as world: two scalars");
}

/// [`a_web_of_exec_and_a_number_read_as_data`] with lane 4 of v10 holding s24 for a while
/// first: the two lanes are then no pair over the whole program (`world` restores s[20:21] as
/// two scalars, numbers both, and stores the console's dwords), and only a web makes the
/// restored pair one register, a mask on one way and the number -1 on the other.
fn a_true_web_of_exec_and_a_number(one_move: bool) -> NativeProgram {
    let mut native = a_web_of_exec_and_a_number_read_as_data(one_move);
    let shared = [writelane(10, 24, 4), readlane(25, 10, 4)].concat();
    // (after the three words `native_of` puts first)
    native.code.splice(12..12, shared.iter().flat_map(|w| w.to_le_bytes()));
    native
}

/// What `VICTORY_WAVE_WEBS=1` brings with it: the rule of masks in partial waves
/// (`super::mask_bounds_on`). [`a_true_web_of_exec_and_a_number`] without a web stores the
/// console's dwords in a partial wave (asserted here, with no switch set: 0xFF with `s1 == 0`,
/// else -1). With its web the pair is a mask, whose ballot in a wave of 40 threads has no
/// lanes for the number's bits 40-63: the webs alone, without that rule, stored 0xFF for -1.
/// With the webs on it is refused as a mask read as data (or stores the console's dwords).
#[test]
fn with_the_webs_alone_a_web_of_exec_and_a_number_in_a_partial_wave_is_read_as_the_number() {
    let cases = [([0, 0], vec![0xFF; 40]), ([0, 1], vec![u32::MAX; 40])];
    for (name, one_move) in [("-1 by two moves", false), ("-1 by one move", true)] {
        let native = a_true_web_of_exec_and_a_number(one_move);
        let world = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert!(!world.webs && instructions(&world, "s_mov_b64").iter().all(|i| regs(&i.writes) != [ir::Reg::S(20), ir::Reg::S(21)]));
        in_a_partial_wave(&format!("{name}, no switch set"), &native, &cases, true);
        let _webs = switched(&super::TEST_WAVE_WEBS, true);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert!(p.webs, "{name}: a web");
        assert!(
            instructions(&p, "s_mov_b64").iter().any(|i| regs(&i.writes) == [ir::Reg::S(20), ir::Reg::S(21)]),
            "{name}: the web comes back by one pair move"
        );
        in_a_partial_wave(&format!("{name}, the webs alone"), &native, &cases, false);
    }
}

/// The paired form of [`super::paired`] gives a host thread two console lanes, so its group
/// has half the console's threads: 32 for one console wave, 96 for three. Counted against 64
/// lanes that made a group of an odd number of console waves a partial wave, and with
/// `VICTORY_WAVE_GUARDS=1` its mask read as data was refused (`Lower::above_threads`), where
/// `world` translates it (both lanes of every thread give their bits). The paired form takes
/// whole console waves only: with the guards it translates, to the module it is without.
#[test]
fn in_the_paired_form_a_group_of_an_odd_number_of_console_waves_is_no_partial_wave() {
    let words = [
        0xBE94_087E, // s_not_b64 s[20:21], exec
        0x7E0A_0215, // v_mov_b32 v5, s21
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ];
    for threads in [64, 128, 192, 320] {
        let native = native_of(&words, true, threads);
        let paired = || {
            let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
            let bytes = crate::dxbc::build(&[]);
            let c = crate::Container::parse(&bytes).expect("container");
            let was = lower::TEST_HOST_WAVE.replace(Some(32));
            let (_, res, io) = parts(&native, &p);
            let r = super::paired(&c, &native, &p, &io, &res);
            lower::TEST_HOST_WAVE.set(was);
            r
        };
        let (world, wave) = paired().unwrap_or_else(|e| panic!("{threads} threads, no switch set: {e}"));
        assert_eq!(wave, Some(32), "{threads} threads: one host wave a console wave");
        validated(&world);
        let _guards = switched(&super::TEST_WAVE_GUARDS, true);
        let (guarded, _) = paired().unwrap_or_else(|e| panic!("{threads} threads, with the guards: {e}"));
        assert!(guarded.dxil == world.dxil, "{threads} threads: the guards change a group of whole waves");
    }
}

/// `native` with its root parameters in a spilled user-data table (`crate::spill`): two user
/// SGPRs, the table's address in s0 and `s1 & 0xff`; the root constants at user-data index 16
/// and the UAV's table at 20, which the code now loads from the table into the registers the
/// rest of it reads (s0, s1 and s4).
fn spilled(mut native: NativeProgram) -> NativeProgram {
    let table: [u32; 7] = [
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0xF400_0101,
        0xFA00_0010, // s_load_dword s4, s[2:3], 0x10
        0xF404_0001,
        0xFA00_0000, // s_load_dwordx2 s[0:1], s[2:3], 0
    ];
    native.code.splice(0..0, table.iter().flat_map(|w| w.to_le_bytes()));
    native.layout.iter_mut().for_each(|e| e.user_data += 16);
    native.used.iter_mut().for_each(|u| u.user_data += 16);
    let crate::native::StageIo::Cs(cs) = &mut native.io else { unreachable!() };
    cs.rsrc2 = 2 << 1;
    native
}

/// With a web in the program and `VICTORY_WAVE_PAIRS_APPROX` off, the approximate forms take
/// the program as `world` parks it. That is the code and the rewrite the main path takes
/// (`spill::build`: the patched code, the loads of a spilled user-data table as entry
/// registers) less the webs, not the bare `native.code`: a compute program that keeps its
/// root parameters in a table lost the rewrite there, its descriptors then loads through a
/// pointer no form traces, and the approximate tries fell back to the program with its web.
///
/// A far early-out (which the forms do not leave out by default, so the program goes to the
/// approximate tries) over a saved EXEC parked in lanes shared with a scalar (the web) and a
/// count of VCC's lanes, in a program with a spilled table: what the selection hands over is
/// what it hands over for the program built without webs, form and module, and not what it
/// hands over with the webs in the approximate forms.
#[test]
fn the_approximate_forms_of_a_spilled_program_with_a_web_take_its_table_rewritten() {
    let _webs = switched(&super::TEST_WAVE_WEBS, true);
    let _joins = switched(&ir::joins::TEST_JOINS, Some(false));
    let body = [
        vec![0xBE9E_0389], // s_mov_b32 s30, 9
        writelane(10, 30, 4),
        readlane(31, 10, 4),
        vec![
            0x7D88_0001, // v_cmp_gt_u32 vcc, s1, v0
            0xBE8E_246A, // s_and_saveexec_b64 s[14:15], vcc
        ],
        writelane(10, 14, 4),
        writelane(10, 15, 5),
        vec![
            0xBE8E_0385, // s_mov_b32 s14, 5
            0xBE8F_0386, // s_mov_b32 s15, 6
        ],
        readlane(14, 10, 4),
        readlane(15, 10, 5),
        super::early_out_tests::GATE.to_vec(),
        vec![0xBEFE_040E], // s_mov_b64 exec, s[14:15]
    ];
    let native = spilled(super::early_out_tests::far_early_out_over(&body.concat(), &[], false));
    assert!(crate::spill::is_spilled(&native));
    let p = crate::spill::build(&native).expect("spill::build");
    assert!(p.webs, "the parked EXEC is a web");
    // as `world` parks it: the same code, the same rewrite, no web
    let mut q = ir::build_without_webs(&native.patched_code(), native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(crate::spill::user_data(&mut q, &native), 2, "both loads of the table are rewritten");
    let selected = |p: &ir::Program| {
        let bytes = crate::dxbc::build(&[]);
        let c = crate::Container::parse(&bytes).expect("container");
        let _host = switched(&lower::TEST_HOST_WAVE, Some(32));
        let _forms = switched(&super::TEST_WAVE_FORMS, true);
        let (st, res, io) = parts(&native, p);
        let shader = super::emit(&c, &native, p, &st, &io, &res);
        let forms = super::take_wave_forms();
        (shader.unwrap_or_else(|e| panic!("not translated: {e}\n  {}", forms.join("\n  "))), forms)
    };
    let (taken, forms) = selected(&p);
    let (world, world_forms) = selected(&q);
    let last = forms.last().expect("the forms tried");
    assert!(last.starts_with("local waves (APPROXIMATE)") && last.ends_with(": wave 32"), "an approximate form is taken\n  {}", forms.join("\n  "));
    assert_eq!(Some(last), world_forms.last(), "the form taken\n  {}", forms.join("\n  "));
    assert!(taken.dxil == world.dxil, "the approximate form is not of the program as world parks it\n  {}", forms.join("\n  "));
    // (and the web is what the two differ by: asked for, the approximate forms keep it)
    let _approx = switched(&super::TEST_PAIRS_APPROX, true);
    let (with_web, _) = selected(&p);
    assert!(with_web.dxil != world.dxil, "the web changes nothing here");
}
