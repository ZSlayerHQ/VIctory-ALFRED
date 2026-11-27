//! With none of this branch's switches set (`VICTORY_WAVE_WEBS`, `VICTORY_WAVE_GUARDS`; the
//! tests' thread-locals untouched) the translator does what `world` does: one synthetic
//! program for each rule the two switches gate, each reaching the changed code and asserting
//! `world`'s result there. (What the switches turn on is asserted where the rules are tested:
//! [`super::lane_pair_tests`], [`super::early_out_tests`].) `VICTORY_WAVE_WEBS` gates the webs
//! (L1, and with them which program the approximate forms take, L6) and brings the rule of
//! masks in partial waves (L5) with it; `VICTORY_WAVE_GUARDS` gates the rules of what a park
//! is (L2, L3), of an early-out's end (L4) and L5. Translated only; nothing here is
//! dispatched.

use super::early_out_tests::{a_short_early_out_to, branches_named};
use super::lane_pair_tests::{a_lane_parked_under_a_relative_read, instructions, parts, readlane, regs, validated, writelane};
use super::wave64_tests::native_of;
use super::lower;
use crate::ir;

fn built(native: &crate::native::NativeProgram) -> ir::Program {
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

/// s[20:21] parked in lanes 4 and 5 of v10 and restored into s[22:23]; later lane 4 holds s30
/// for a while, so the two lanes are no pair over the whole program.
fn a_pair_in_shared_lanes() -> Vec<Vec<u32>> {
    vec![
        writelane(10, 20, 4),
        writelane(10, 21, 5),
        vec![0x7E04_0281], // v_mov_b32 v2, 1
        readlane(22, 10, 4),
        readlane(23, 10, 5),
        writelane(10, 30, 4),
        readlane(31, 10, 4),
    ]
}

/// L1 (`ir::unspill`, `ir::lane_pairs`). `world` has no lane-pair web: a pair parked in lanes
/// that hold another scalar elsewhere comes back as two scalars, each lane one register by
/// another name, never as one `s_mov_b64`.
#[test]
fn off_a_pair_parked_in_shared_lanes_comes_back_as_two_scalars() {
    let native = native_of(&a_pair_in_shared_lanes().concat(), true, 64);
    let p = built(&native);
    assert!(instructions(&p, "v_writelane_b32").is_empty() && instructions(&p, "v_readlane_b32").is_empty(), "unspilled");
    let wide = instructions(&p, "s_mov_b64").len();
    assert_eq!(wide, 0, "no switch set: {wide} pair moves where world has two scalars");
    assert!(!p.webs, "no switch set: a web was made");
    // each half through its own lane register, and lane 4's register is s30's later too
    let narrow = instructions(&p, "s_mov_b32");
    let through = |from: u8, to: u8| {
        let save = narrow.iter().find(|i| regs(&i.reads) == [ir::Reg::S(from)]).expect("the save");
        let restore = narrow.iter().find(|i| regs(&i.writes) == [ir::Reg::S(to)]).expect("the restore");
        assert_eq!(regs(&restore.reads), regs(&save.writes), "s{from} comes back in s{to}");
        regs(&save.writes)
    };
    let (low, high, lone) = (through(20, 22), through(21, 23), through(30, 31));
    assert_eq!(low, lone, "lane 4 is one register whatever it holds");
    assert_ne!(low, high);
    assert_eq!(p, ir::build_without_webs(&native.code, native.wave64, native.vgprs).expect("ir::build"));
}

/// L1, the register codes. A web takes two spare codes of its own before the single lanes
/// take theirs, so a program that parks as many scalars as `world` has spare codes for was
/// left wholly unspilled. [`a_pair_in_shared_lanes`] and then `n` more lanes parked, for the
/// largest `n` the program without webs still unspills (one more lane and it does not: the
/// codes have run out): with no switch set it is unspilled all the same.
#[test]
fn off_a_program_that_uses_every_spare_code_is_still_unspilled() {
    let program = |n: usize| {
        let mut words = a_pair_in_shared_lanes();
        for k in 0..n as u32 {
            let (reg, lane) = (16 + k / 32, 2 * (k % 32));
            words.push(writelane(reg, 30, lane));
            words.push(readlane(31, reg, lane));
        }
        let mut native = native_of(&words.concat(), true, 64);
        native.vgprs = 64;
        native
    };
    let unspilled = |p: &ir::Program| instructions(p, "v_writelane_b32").is_empty();
    let world = |n: usize| {
        let native = program(n);
        unspilled(&ir::build_without_webs(&native.code, native.wave64, native.vgprs).expect("ir::build"))
    };
    let most = (0..400).take_while(|&n| world(n)).last().expect("the pair alone is unspilled");
    assert!((100..399).contains(&most), "the spare codes run out at some count: {most}");
    assert!(!world(most + 1), "one more lane than there are codes: left as it is");
    let p = built(&program(most));
    let left = instructions(&p, "v_writelane_b32").len();
    assert_eq!(left, 0, "no switch set: {left} lane writes left in a program world unspills ({most} lanes and the pair)");
}

/// L2 (`ir::unspill`'s `parked`). On `world` a lane write and read of a VGPR at or past the
/// program's count (16 here; v16) is a park like any other: two scalar moves.
#[test]
fn off_a_lane_past_the_programs_vgprs_is_a_park() {
    let words = [
        vec![0xBE94_0387], // s_mov_b32 s20, 7
        writelane(16, 20, 0),
        readlane(21, 16, 0),
    ];
    let native = native_of(&words.concat(), true, 64);
    assert_eq!(native.vgprs, 16);
    let p = built(&native);
    let left = instructions(&p, "v_writelane_b32").len() + instructions(&p, "v_readlane_b32").len();
    assert_eq!(left, 0, "no switch set: {left} lane accesses of v16 left where world parks");
    let narrow = instructions(&p, "s_mov_b32");
    let save = narrow.iter().find(|i| regs(&i.reads) == [ir::Reg::S(20)]).expect("the save of s20");
    let restore = narrow.iter().find(|i| regs(&i.writes) == [ir::Reg::S(21)]).expect("the restore into s21");
    assert_eq!(regs(&restore.reads), regs(&save.writes));
}

/// L3 (`ir::unspill`, the effects loop). On `world` an M0-relative read (`v_movrels_b32`)
/// does not make the VGPRs it names ordinary: the lane parked in v10 under such a read from v9
/// is still a park, a scalar move each way, and the relative read stays.
#[test]
fn off_a_lane_under_an_m0_relative_read_is_still_a_park() {
    let (native, _) = a_lane_parked_under_a_relative_read();
    let p = built(&native);
    assert_eq!(instructions(&p, "v_movrels_b32").len(), 1);
    let left = instructions(&p, "v_writelane_b32").len() + instructions(&p, "v_readlane_b32").len();
    assert_eq!(left, 0, "no switch set: {left} lane accesses of v10 left where world parks");
    let narrow = instructions(&p, "s_mov_b32");
    let save = narrow.iter().filter(|i| regs(&i.reads) == [ir::Reg::S(20)]).count();
    let restore = narrow.iter().filter(|i| regs(&i.writes) == [ir::Reg::S(21)]).count();
    assert_eq!((save, restore), (1, 1), "the park's two moves");
}

/// L4 (`ir::joins::ends_program`). On `world` any block of program-control words alone that
/// ends the program is an early-out's end, one that waits at a barrier or sends a message
/// too: the `s_cbranch_execz` to it is left out of both forms ("without early-outs",
/// "without skips").
#[test]
fn off_an_early_out_to_an_end_with_a_barrier_or_a_message_is_left_out() {
    for (name, end) in [("s_barrier", 0xBF8A_0000), ("s_sendmsg", 0xBF90_0003)] {
        let p = a_short_early_out_to(end);
        assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == name), "{name} in the code");
        assert_eq!(branches_named(&p, "s_cbranch_execz"), 1, "{name}");
        for (form, without) in [
            ("without early-outs", ir::without_early_outs(&p)),
            ("without skips", ir::without_skips(&p, false)),
            ("without skips, deep", ir::without_skips(&p, true)),
        ] {
            let without = without.unwrap_or_else(|| panic!("no switch set: {form} keeps the early-out to an end of {name}"));
            assert_eq!(branches_named(&without, "s_cbranch_execz"), 0, "{form}, an end of {name}");
        }
    }
}

/// L5 (`structure::bounded_masks`, `Lower::above_threads`). One console wave of 40 threads (a
/// partial wave) reads the high dword of `s_not_b64 s[20:21], exec` as data and stores it: a
/// mask dword that can hold bits above the wave's threads. On `world` its meeting form
/// translates (the dword as the ballot of the lanes that are there); it is not refused.
#[test]
fn off_an_unbounded_mask_as_data_in_a_partial_wave_translates() {
    let words = [
        0xBE94_087E, // s_not_b64 s[20:21], exec
        0x7E0A_0215, // v_mov_b32 v5, s21
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ];
    let native = native_of(&words, true, 40);
    let p = built(&native);
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let (st, res, io) = parts(&native, &p);
    let met = lower::with_meeting(|| super::emit_program(&c, &native, &p, &st, &io, &res));
    lower::TEST_HOST_WAVE.set(was);
    let (shader, wave) = met.unwrap_or_else(|e| panic!("no switch set: refused where world translates: {e}"));
    assert_eq!(wave, None, "the meeting form asks no wave size");
    validated(&shader);
}

/// Both switches are part of the shader cache's key: a translation cached with one unset is
/// not served where it is set.
#[test]
fn the_cache_key_holds_both_switches() {
    let key = |switch: &'static str, value: Option<&'static str>| {
        super::translation_fingerprint_with(|name| value.filter(|_| name == switch).map(Into::into))
    };
    for switch in ["VICTORY_WAVE_WEBS", "VICTORY_WAVE_GUARDS"] {
        assert!(super::translation_switches().any(|n| n == switch), "{switch} is a translation switch");
        assert_ne!(key(switch, None), key(switch, Some("1")), "{switch}");
    }
}
