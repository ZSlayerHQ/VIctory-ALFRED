//! Console wave64 compute programs on a host of 32-lane waves, in two control-flow shapes the
//! exact forms do not take on their own, built here for the tests, and the switch that lets the
//! forms take them (`VICTORY_WAVE_JOINS`, `ir::joins`): an early-out spelled as a far jump, and
//! a skip whose scalars come back through an inner join. Lowered, validated by `dxil.dll` when
//! present, and dispatched on the software adapter only; the dispatch on a card is an
//! ignored test of its own.

use std::collections::HashSet;

use super::lane_pair_tests::{emitted, on_the_software_adapter, one_group, parts, validated};
use super::wave64_tests::{card_with_waves_of, lowered, native, native_of};
use super::{HostShader, lower, switched};
use crate::native::NativeProgram;
use crate::{ShaderError, ir};

/// A wave64 program of 128 threads (two console waves) that leaves early when no lane of its
/// wave is below s0, as the console's compiler does when the end of the program is too far for
/// a branch: `s_cbranch_execnz` over a far jump to the end. `extra` are words that go into the
/// far jump's block before it, and `work_at_end` puts an instruction between the jump's target
/// and the end of the program. Then the gate: every thread stores 1, or 2 when some lane of its
/// console wave that is still running has the thread number s1 (the lanes that fail a test).
fn far_early_out(extra: &[u32], work_at_end: bool) -> NativeProgram {
    far_early_out_over(&GATE, extra, work_at_end)
}

/// The gate of [`far_early_out`].
pub(super) const GATE: [u32; 8] = [
    0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
    0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
    0xBF06_8010, // s_cmp_eq_u32 s16, 0
    0x8511_8281, // s_cselect_b32 s17, 1, 2
    0x7E02_0211, // v_mov_b32 v1, s17
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0x38 << 26 | 0x1C << 18 | 1 << 12,
    0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

/// [`far_early_out`] with `gate` for its body.
pub(super) fn far_early_out_over(gate: &[u32], extra: &[u32], work_at_end: bool) -> NativeProgram {
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, the way to the end
        0x820F_800F, // s_addc_u32 s15, s15, 0
        0xBE80_200E, // s_setpc_b64 s[14:15]
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | (extra.len() + jump.len()) as u32, // s_cbranch_execnz over the jump
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(&jump);
    words.extend_from_slice(gate);
    if work_at_end {
        words.push(0x7E06_0281); // v_mov_b32 v3, 1
    }
    native_of(&words, true, 128)
}

/// What the gate of [`far_early_out`] stores in each of 128 threads.
fn gate_expected(below: u32, failing: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| {
            let some_fail = failing < below && failing / 64 == t / 64;
            match (t < below, some_fail) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            }
        })
        .collect()
}

/// The (s0, s1) cases of [`far_early_out`]: threads below s0 are in bounds, thread s1 fails
/// the test. A failing thread of the upper half gives 2 to the lower half's threads as well,
/// and a failing thread out of bounds counts for nothing.
const GATE_CASES: [(u32, u32); 8] = [(128, 40), (128, 100), (128, 63), (40, 50), (40, 10), (33, 32), (1, 0), (0, 5)];

pub(super) fn branches_named(p: &ir::Program, name: &str) -> usize {
    p.blocks.iter().filter(|b| b.insts.last().is_some_and(|i| i.name() == name)).count()
}

/// What the runtime's retry ([`super::emit`]) gives for `native` on a host of 32-lane waves,
/// and what `form` alone gives there.
fn retried_and_form(native: &NativeProgram, form: &ir::Program) -> (Result<HostShader, ShaderError>, HostShader) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let (st, res, io) = parts(native, &p);
    let taken = super::emit(&c, native, &p, &st, &io, &res);
    let (st, res, io) = parts(native, form);
    let exact = super::emit_program(&c, native, form, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let (exact, wave) = exact.expect("the form translates");
    assert_eq!(wave, Some(32), "for host waves of 32");
    (taken, exact)
}

/// [`far_early_out`]: refused per lane; with the far early-out left out (by both forms that
/// leave early-outs out) every lane runs the gate, the two host waves of each console wave
/// exchange their halves of VCC, and the retry takes that form.
#[test]
fn a_far_early_out_is_left_out_and_the_gate_counts_the_lanes_of_both_halves() {
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1);
    lower::take_needs();
    let kept = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&kept, Err(ShaderError::Unsupported { what, .. }) if what.ends_with("where not every lane runs")),
        "{:?}",
        kept.err()
    );
    // by default (`VICTORY_WAVE_JOINS` unset) the forms see no early-out: the retry takes a
    // form in which each half-wave counts its own lanes
    let _joins = switched(&ir::joins::TEST_JOINS, Some(false));
    assert!(ir::without_early_outs(&p).is_none() && ir::without_skips(&p, true).is_none());
    ir::joins::TEST_JOINS.set(Some(true));
    for (name, form) in [("early-outs", ir::without_early_outs(&p)), ("skips", ir::without_skips(&p, false))] {
        let form = form.unwrap_or_else(|| panic!("without {name}: the far early-out stays"));
        assert_eq!(branches_named(&form, "s_cbranch_execnz"), 0, "without {name}");
    }
    let form = ir::without_early_outs(&p).expect("left out");
    validated(&lowered(&native, &form, &HashSet::new()).expect("lower"));
    let (taken, exact) = retried_and_form(&native, &form);
    assert!(taken.expect("the program translates").dxil == exact.dxil, "the retry took another form");
    ir::joins::TEST_JOINS.set(Some(false));
    let (by_default, _) = retried_and_form(&native, &form);
    ir::joins::TEST_JOINS.set(Some(true));
    assert!(by_default.is_ok_and(|s| s.dxil != exact.dxil), "the form is taken without being asked for");
    let cases: Vec<_> = GATE_CASES.iter().map(|&(b, f)| ([b, f], gate_expected(b, f))).collect();
    let (shader, wave) = emitted(&native, &form);
    assert_eq!(wave, Some(32), "the two host waves of a console wave exchange");
    on_the_software_adapter("the far early-out's gate", &shader, &cases);
}

/// What must stay: a far jump to the end that does any work on its way, or whose target does
/// work before the end, is no early-out.
#[test]
fn a_far_jump_that_does_work_is_not_an_early_out() {
    let _joins = switched(&ir::joins::TEST_JOINS, Some(true));
    for (name, native) in [
        ("work in the jump", far_early_out(&[0xBE94_0385], false)), // s_mov_b32 s20, 5
        ("work at the target", far_early_out(&[], true)),
    ] {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1, "{name}");
        assert!(ir::without_early_outs(&p).is_none(), "{name}");
        assert!(ir::without_skips(&p, false).is_none(), "{name}");
        assert!(ir::without_skips(&p, true).is_none(), "{name}");
    }
}

/// `s_cbranch_execz` to a block that ends the program after `end` (one SOPP word): an
/// early-out only where that block does nothing another wave or the hardware sees on the way
/// out. A barrier there is one the console's leaving wave waits at, a message one it sends.
pub(super) fn a_short_early_out_to(end: u32) -> ir::Program {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz to the end
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x7E0C_0282, // v_mov_b32 v6, 2
        0xBF81_0000, // s_endpgm
        end,
    ];
    let native = native_of(&words, true, 64);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

/// Neon's review, follow-up: an end that waits at a barrier or sends a message is no pure end
/// of the program, and the branch to it stays; an end of `s_nop` (which `build` drops) is one.
/// With `VICTORY_WAVE_GUARDS=1`; without, `world`'s rule (`off_is_off_tests`).
#[test]
fn an_end_with_a_barrier_or_a_message_is_no_early_outs_end() {
    let _guards = switched(&super::TEST_WAVE_GUARDS, true);
    for (name, end, left_out) in [
        ("s_nop", 0xBF80_0000, true),
        ("s_barrier", 0xBF8A_0000, false),
        ("s_sendmsg", 0xBF90_0003, false),
    ] {
        let p = a_short_early_out_to(end);
        let named = p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == name);
        assert!(named || left_out, "{name} in the code");
        assert_eq!(branches_named(&p, "s_cbranch_execz"), 1, "{name}");
        assert_eq!(ir::without_early_outs(&p).is_some(), left_out, "an end of {name}");
    }
}

/// What the forms do not prove (the review of job/shadow-mask, finding 1): a body that puts
/// EXEC back from the mask saved before the far early-out and then stores. The console's wave
/// with no lane below s0 has ended at the branch and stores nothing; walked with EXEC clear,
/// the body revives every lane and each stores 7. The forms leave this early-out out all the
/// same, as they do the short spelling's (`s_cbranch_execz` to the end): open until a pass
/// proves that EXEC stays within the lanes that entered.
#[test]
#[ignore = "open: the forms do not see EXEC put back behind an early-out they leave out"]
fn a_far_early_out_whose_body_puts_exec_back_and_stores_is_kept() {
    let _joins = switched(&ir::joins::TEST_JOINS, Some(true));
    let body = [
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0287, // v_mov_b32 v1, 7
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let native = far_early_out_over(&body, &[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1);
    assert!(ir::without_early_outs(&p).is_none(), "without early-outs");
    assert!(ir::without_skips(&p, true).is_none(), "without skips");
}

/// `if (t < s0) { park s[20:21]; if (s1 == 0) {..} else {..}; take it back }` and a read of
/// s[20:21] after the join, as the console's compiler leaves a parked scalar across an inner
/// branch. `then` is the inner branch's first arm; the second one leaves the parked scalar alone.
fn parked_across_an_inner_branch(then: u32) -> ir::Program {
    let body = [
        0xBE96_0414, // s_mov_b64 s[22:23], s[20:21]
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0002, // s_cbranch_scc1 to the second arm
        then,
        0xBF82_0001, // s_branch to the join
        0xBE98_0382, // s_mov_b32 s24, 2
        0xBE94_0416, // s_mov_b64 s[20:21], s[22:23]
    ];
    let mut words = vec![
        0xBE94_0406, // s_mov_b64 s[20:21], s[6:7]
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
    ];
    words.extend_from_slice(&body);
    words.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0214, // v_mov_b32 v1, s20
    ]);
    let native = native(&words);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

#[test]
fn a_skip_whose_parked_scalars_pass_an_inner_branch_unchanged_is_left_out() {
    let p = parked_across_an_inner_branch(0xBE96_0314); // s_mov_b32 s22, s20
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    let _joins = switched(&ir::joins::TEST_JOINS, Some(false));
    assert!(ir::without_skips(&p, true).is_none(), "by default");
    ir::joins::TEST_JOINS.set(Some(true));
    // (s22 comes back through a phi of the inner join whose two arguments are copies of one
    // value: only `deep` follows it)
    assert!(ir::without_skips(&p, false).is_none());
    let deep = ir::without_skips(&p, true).expect("left out");
    assert_eq!(branches_named(&deep, "s_cbranch_execz"), 0);
}

#[test]
fn a_skip_whose_parked_scalar_an_inner_branch_changes_is_kept() {
    let _joins = switched(&ir::joins::TEST_JOINS, Some(true));
    let p = parked_across_an_inner_branch(0xBE96_0387); // s_mov_b32 s22, 7
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    assert!(ir::without_skips(&p, false).is_none());
    assert!(ir::without_skips(&p, true).is_none());
}

/// [`far_early_out`] in the form the translation takes, on a card with 32-lane waves: every
/// thread of every case. Ignored: the program has group barriers, and a dispatch whose
/// threads wait at a barrier the others never reach takes the display with it. Run only on a
/// machine with such a card and nothing to lose on its screen:
///   cargo test -p victory-shader --locked -- --ignored --test-threads=1 --nocapture on_a_card_a_far
#[test]
#[ignore = "dispatches a program with group barriers on the hardware device"]
fn on_a_card_a_far_early_outs_gate_stores_what_the_console_stores() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = card_with_waves_of(32).expect("a hardware device with 32-lane waves");
    let _joins = switched(&ir::joins::TEST_JOINS, Some(true));
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let form = ir::without_early_outs(&p).expect("left out");
    let (shader, _) = emitted(&native, &form);
    for &(b, f) in &GATE_CASES {
        assert_eq!(one_group(&host, &shader, [b, f], 128).expect("dispatch"), gate_expected(b, f), "gate {b}, {f}");
    }
}
