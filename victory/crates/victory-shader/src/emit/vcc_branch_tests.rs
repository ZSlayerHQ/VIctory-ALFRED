//! `s_cbranch_vccz` and `vccnz`, which the console takes for the whole wave: VCCZ is "VCC is
//! zero in all 64 bits", so with one lane's VCC bit set every lane of EXEC runs the region
//! the branch skips otherwise. Read per lane by default, as on `world` (right where VCC is
//! alike in every lane of EXEC or the region runs no lane outside VCC); taken for the wave
//! with `VICTORY_WAVE_VCC=1` ([`super::wave_vcc_on`]), which these tests set where they say
//! so. The wave's decision is checked in the meeting form with its vote asked for
//! (`VICTORY_WAVE_MEET_WIDE=1`), which asks no wave size of the host and runs on the software
//! adapter (its group barriers stand where every thread comes); the form for 64-lane hosts is
//! translated and validated, and the exchange of 32-lane halves is declined. Never
//! dispatched on a card here.

use std::collections::HashSet;

use super::lane_pair_tests::{one_group, parts, validated};
use super::wave64_tests::{lowered_on, native};
use super::{HostShader, host::Host, lower};
use crate::{ir, structure};

const V_MOV_V5_0: u32 = 0x7E0A_0280;
const V_MOV_V5_2: u32 = 0x7E0A_0282;
/// `s_cbranch_vccz` over the next `n` words.
const S_CBRANCH_VCCZ: u32 = 0xBF86_0000;

/// One console wave of 64 threads: v5 = 0; VCC = `compare`; `s_cbranch_vccz` over `region`
/// (which sets v5 = 2 where it runs); every thread stores v5 at its own dword.
fn program(compare: &[u32], region: &[u32]) -> crate::native::NativeProgram {
    let mut words = vec![V_MOV_V5_0];
    words.extend_from_slice(compare);
    words.push(S_CBRANCH_VCCZ | region.len() as u32);
    words.extend_from_slice(region);
    words.extend([
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ]);
    native(&words)
}

/// `compare`, `s_cbranch_vccz` over `v_mov_b32 v5, 2`: the region runs in all of EXEC.
fn plain(compare: &[u32]) -> crate::native::NativeProgram {
    program(compare, &[V_MOV_V5_2])
}

/// The program lowered for a host of 64-lane waves (the per-lane form: no exchange).
fn lowered(native: &crate::native::NativeProgram) -> Result<HostShader, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    lowered_on(native, &p, &HashSet::new(), 64)
}

/// The blocks whose VCC branch the structure keeps for the wave.
fn wave_wide(native: &crate::native::NativeProgram) -> HashSet<usize> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    structure::structure(&p).expect("structure").wide
}

/// `native` with `VICTORY_WAVE_VCC=1`: its VCC branch the wave's (as on `world` without),
/// translated and validated for a host of 64-lane waves, declined for one of 32-lane waves
/// (no exchange), and its meeting form, which the software adapter runs.
fn the_waves_branch(native: &crate::native::NativeProgram) -> HostShader {
    assert!(wave_wide(native).is_empty(), "without the switch: per lane, as on world");
    with_vcc(|| {
        assert_eq!(wave_wide(native).len(), 1, "the VCC branch is the wave's");
        validated(&lowered(native).expect("for 64-lane waves"));
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let bytes = crate::dxbc::build(&[]);
        let c = crate::Container::parse(&bytes).expect("container");
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        let (st, res, io) = parts(native, &p);
        let exchanged = super::emit_program(&c, native, &p, &st, &io, &res);
        lower::TEST_HOST_WAVE.set(was);
        assert!(exchanged.is_err(), "no exchange of the halves for a VCC branch");
        met(native)
    })
}

/// `shader` on the software adapter, one group: the 64 dwords against `expected` per s0, s1.
fn on_the_software_adapter(name: &str, shader: &HostShader, cases: &[([u32; 2], Vec<u32>)]) {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("{name}: no SM 6.6 software device; skipped");
        return;
    };
    for (constants, expected) in cases {
        let got = one_group(&host, shader, *constants, 64).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(&got, expected, "{name}: s0, s1 = {constants:?}");
    }
    eprintln!("{name}: every thread as the console stores it, on the software adapter");
}

/// What the console stores: 2 in every thread when some thread's compare holds, else 0.
fn the_wave_goes_in(any: bool) -> Vec<u32> {
    vec![if any { 2 } else { 0 }; 64]
}

/// VCC = (t < s0): VCC a subset of EXEC, a bit apart in each lane, and the region runs in all
/// of EXEC. With s0 = 40 the console runs it in all 64 threads; read per lane, threads 40-63
/// skipped it and stored 0. With s0 = 0 VCC is zero and EXEC is not: no thread runs it.
#[test]
fn a_vccz_over_work_in_lanes_outside_vcc_is_the_waves_branch() {
    let native = plain(&[0x7D88_0000]); // v_cmp_gt_u32 vcc, s0, v0
    let shader = the_waves_branch(&native);
    let cases: Vec<_> = [0u32, 1, 40, 64].iter().map(|&s0| ([s0, 0], the_wave_goes_in(s0 > 0))).collect();
    on_the_software_adapter("vccz, VCC within EXEC", &shader, &cases);
}

/// VCC = (t == s0): one lane's bit. With s0 = 5 it is in the lower 32-lane half only, with
/// s0 = 40 in the upper half only (a host wave of 32 holds no bit of it in the other half);
/// either way every thread of the console wave runs the region. With s0 = 64 VCC is zero, EXEC
/// all lanes: none does.
#[test]
fn a_vccz_on_one_lanes_bit_in_either_half_is_the_waves_branch() {
    let native = plain(&[0x7D84_0000]); // v_cmp_eq_u32 vcc, s0, v0
    let shader = the_waves_branch(&native);
    let cases: Vec<_> = [5u32, 40, 0, 63, 64].iter().map(|&s0| ([s0, 0], the_wave_goes_in(s0 < 64))).collect();
    on_the_software_adapter("vccz, one lane's bit", &shader, &cases);
}

/// VCC = (s0 > s1) through a VGPR copy of s1: alike in every lane, so each lane's own bit is
/// the wave's decision. Read per lane by default, as on `world`: translated, and right.
#[test]
fn a_vccz_on_a_compare_alike_in_every_lane_stays_per_lane() {
    let native = plain(&[
        0x7E02_0201, // v_mov_b32 v1, s1
        0x7D88_0200, // v_cmp_gt_u32 vcc, s0, v1
    ]);
    assert!(wave_wide(&native).is_empty(), "the branch stays per lane");
    let shader = lowered(&native).expect("a compare of scalars is the wave's");
    validated(&shader);
    let cases: Vec<_> = [(0u32, 0u32), (5, 3), (3, 5), (64, 0)].iter().map(|&(a, b)| ([a, b], the_wave_goes_in(a > b))).collect();
    on_the_software_adapter("vccz on a scalar compare", &shader, &cases);
}

/// VCC = (t < s0) apart in each lane, but the region narrows EXEC to VCC before its work
/// (`s_and_saveexec_b64`) and puts it back after: no lane outside VCC runs anything there, so
/// skipping it per lane is the console. Read per lane by default, as on `world`: translated,
/// and right (threads below s0 store 2).
#[test]
fn a_vccz_over_a_region_that_narrows_exec_to_vcc_stays_per_lane() {
    let native = program(
        &[0x7D88_0000], // v_cmp_gt_u32 vcc, s0, v0
        &[
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            V_MOV_V5_2,
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        ],
    );
    assert!(wave_wide(&native).is_empty(), "the branch stays per lane");
    let shader = lowered(&native).expect("the region runs in VCC's lanes only");
    validated(&shader);
    let cases: Vec<_> = [0u32, 1, 40, 64]
        .iter()
        .map(|&s0| ([s0, 0], (0..64u32).map(|t| if t < s0 { 2 } else { 0 }).collect()))
        .collect();
    on_the_software_adapter("vccz over a region in VCC", &shader, &cases);
}

/// Sol's third review, item 1: `program(compare, region)`'s shapes the per-lane proofs took
/// for exact. A scalar written in the region (`s_mov_b32 s20, 7` under VCC = (t == 0)) is the
/// wave's: the console stores 7 in every thread. A select by a pair's lane bit
/// (`v_cndmask_b32 v1, 0, 1, s[12:13]` with s[12:13] = 1) is 1 in lane 0 alone, so VCC =
/// (v1 == 1) is lane 0's bit, and the console runs the region (v5 = 2) in every thread. With
/// `VICTORY_WAVE_VCC=1` both are the wave's branch, run in the meeting form on the software
/// adapter; without, the branch is read per lane as on `world`.
#[test]
fn what_the_per_lane_proofs_took_for_exact_is_the_waves_branch() {
    let scalar = native(&[
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        S_CBRANCH_VCCZ | 1,
        0xBE94_0387, // s_mov_b32 s20, 7
        0x7E0A_0214, // v_mov_b32 v5, s20
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ]);
    let select = plain(&[
        0xBE8C_0381, // s_mov_b32 s12, 1
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xD501_0001,
        0x80 | 0x81 << 9 | 12 << 18, // v_cndmask_b32 v1, 0, 1, s[12:13]
        0x7D84_0281, // v_cmp_eq_u32 vcc, 1, v1
    ]);
    for (name, native, expected) in [("a scalar in the region", &scalar, 7), ("a select by a lane bit", &select, 2)] {
        let shader = with_vcc(|| met(native));
        on_the_software_adapter(name, &shader, &[([0, 0], vec![expected; 64])]);
        assert!(wave_wide(native).is_empty(), "{name}: without the switch, per lane as on world");
        assert_eq!(with_vcc(|| wave_wide(native)).len(), 1, "{name}: with the switch, the wave's");
    }
}

/// Sol's third review, item 2: 128 threads (two console waves), VCC = (t == 0) and VCCZ to the
/// join, inside it VCC = (t == 1) and VCCZ to the same join over a vector write. Taken for the
/// wave in the exchanged form of a 32-lane host, only the first console wave comes to the
/// inner vote and its barriers. Without the switch the program is as on `world` (per lane, no
/// exchange); with it the exchange is declined. Translated only, never run.
#[test]
fn a_vote_some_console_waves_never_reach_is_no_exchange() {
    let mut words = vec![V_MOV_V5_0, 0x7D84_0080, S_CBRANCH_VCCZ | 3, 0x7D84_0081, S_CBRANCH_VCCZ | 1, V_MOV_V5_2];
    words.extend([
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 2, // buffer_store_dword v5, v2, s[8:11], 0 offen
    ]);
    let native = super::wave64_tests::native_of(&words, true, 128);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let on_32 = |native: &crate::native::NativeProgram| {
        let bytes = crate::dxbc::build(&[]);
        let c = crate::Container::parse(&bytes).expect("container");
        let was = lower::TEST_HOST_WAVE.replace(Some(32));
        let (st, res, io) = parts(native, &p);
        let r = super::emit_program(&c, native, &p, &st, &io, &res);
        lower::TEST_HOST_WAVE.set(was);
        r
    };
    let (_, wave) = on_32(&native).expect("as on world: translated per lane");
    assert_eq!(wave, None, "no exchange, no barrier");
    assert!(with_vcc(|| on_32(&native)).is_err(), "with the switch: the exchange is declined");
}

/// `f` with `VICTORY_WAVE_VCC=1`.
fn with_vcc<R>(f: impl FnOnce() -> R) -> R {
    let was = super::TEST_WAVE_VCC.replace(true);
    let r = f();
    super::TEST_WAVE_VCC.set(was);
    r
}

/// `native`'s meeting form with its vote asked for, for a 32-lane host: no wave size, so the
/// software adapter runs it.
fn met(native: &crate::native::NativeProgram) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&bytes).expect("container");
    let was = (lower::TEST_HOST_WAVE.replace(Some(32)), super::TEST_MEET_WIDE.replace(true));
    let (st, res, io) = parts(native, &p);
    let r = lower::with_meeting(|| super::emit_program(&c, native, &p, &st, &io, &res));
    lower::TEST_HOST_WAVE.set(was.0);
    super::TEST_MEET_WIDE.set(was.1);
    let (shader, wave) = r.expect("the meeting form translates");
    assert_eq!(wave, None, "the meeting form asks no wave size");
    validated(&shader);
    shader
}
