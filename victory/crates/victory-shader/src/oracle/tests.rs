//! The oracle against the reference's own worked examples and functional examples, and small
//! programs whose answers follow from the reference alone.

use super::compare::{compare, ulps};
use super::semantics::{self as sem, BufferAddress, BufferResource, Lane, Mode, SaluIn, Wave64Permute};
use super::{Group, Memory, Tol};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_BARRIER: u32 = 0xBF8A_0000;

fn lane(s: [u32; 3]) -> Lane {
    Lane { s: s.map(u64::from), mask: false, d: 0, lane: 0, mode: Mode(0x00F0), clamp: false, omod: 0 }
}

fn v(name: &str, s: [u32; 3]) -> u32 {
    sem::valu(name, &lane(s)).unwrap_or_else(|e| panic!("{name}: {e}")).d.expect("a result") as u32
}

/// 12.8's functional examples of the transcendental and reciprocal entries.
#[test]
fn functional_examples_of_the_reciprocal_and_transcendental_entries() {
    for (a, d) in [(0xff80_0000, 0x8000_0000), (0xc000_0000, 0xbf00_0000), (0x8000_0000, 0xff80_0000), (0, 0x7f80_0000), (0x7f80_0000, 0)] {
        assert_eq!(v("v_rcp_f32", [a, 0, 0]), d, "rcp({a:#x})");
    }
    for (a, d) in [(0x8000_0000u32, 0xff80_0000u32), (0, 0x7f80_0000), (0x4080_0000, 0x3f00_0000), (0x7f80_0000, 0)] {
        assert_eq!(v("v_rsq_f32", [a, 0, 0]), d, "rsq({a:#x})");
    }
    assert!(f32::from_bits(v("v_rsq_f32", [0xff80_0000, 0, 0])).is_nan());
    for (a, d) in [(0x8000_0000u32, 0x8000_0000u32), (0, 0), (0x4080_0000, 0x4000_0000), (0x7f80_0000, 0x7f80_0000)] {
        assert_eq!(v("v_sqrt_f32", [a, 0, 0]), d, "sqrt({a:#x})");
    }
    for (a, d) in [(0xff80_0000u32, 0u32), (0x8000_0000, 0x3f80_0000), (0x7f80_0000, 0x7f80_0000)] {
        assert_eq!(v("v_exp_f32", [a, 0, 0]), d, "exp({a:#x})");
    }
    for (a, d) in [(0x8000_0000u32, 0xff80_0000u32), (0, 0xff80_0000), (0x3f80_0000, 0), (0x7f80_0000, 0x7f80_0000)] {
        assert_eq!(v("v_log_f32", [a, 0, 0]), d, "log({a:#x})");
    }
    for (a, d) in [(0xff7f_ffffu32, 0u32), (0x8000_0000, 0x8000_0000), (0x3e80_0000, 0x3f80_0000)] {
        assert_eq!(v("v_sin_f32", [a, 0, 0]), d, "sin({a:#x})");
    }
    for (a, d) in [(0xff7f_ffffu32, 0x3f80_0000u32), (0x8000_0000, 0x3f80_0000)] {
        assert_eq!(v("v_cos_f32", [a, 0, 0]), d, "cos({a:#x})");
    }
    // cos(0.25 turn) is 0 to within the tolerance the comparison allows sine and cosine
    assert!(f32::from_bits(v("v_cos_f32", [0x3e80_0000, 0, 0])).abs() < 1e-7);
}

/// The scalar and vector bit-search and count examples (12.3, 12.8).
#[test]
fn functional_examples_of_the_bit_searches_and_counts() {
    let s = |name: &str, s0: u64| sem::salu(name, SaluIn { s0, ..SaluIn::default() }).expect(name).d.expect("a result") as u32;
    for (a, d) in [(0xaaaa_aaaau64, 1u32), (0x5555_5555, 0), (0, 0xffff_ffff), (0xffff_ffff, 0), (0x0001_0000, 16)] {
        assert_eq!(s("s_ff1_i32_b32", a), d);
    }
    for (a, d) in [(0u64, 0xffff_ffffu32), (0x0000_cccc, 16), (0xffff_3333, 0), (0x7fff_ffff, 1), (0x8000_0000, 0)] {
        assert_eq!(s("s_flbit_i32_b32", a), d);
    }
    for (a, d) in [(0u64, 0u32), (0xcccc_cccc, 16), (0xffff_ffff, 32)] {
        assert_eq!(s("s_bcnt1_i32_b32", a), d);
    }
    for (a, d) in [(1u64, 1u32), (0x7fff_ffff, 0x7fff_ffff), (0x8000_0000, 0x8000_0000), (0x8000_0001, 0x7fff_ffff), (0xffff_ffff, 1)] {
        assert_eq!(s("s_abs_i32", a), d);
    }
    for (a, d) in [(0u32, 0xffff_ffffu32), (0x8000_00ff, 0), (0x1000_00ff, 3), (0x0000_ffff, 16), (1, 31)] {
        assert_eq!(v("v_ffbh_u32", [a, 0, 0]), d);
    }
    for (a, d) in [(0u32, 0xffff_ffffu32), (0xff00_0001, 0), (0xff00_0008, 3), (0xffff_0000, 16), (0x8000_0000, 31)] {
        assert_eq!(v("v_ffbl_b32", [a, 0, 0]), d);
    }
    for (a, d) in [(0u32, 0xffff_ffffu32), (0x4000_0000, 1), (0x8000_0000, 1), (0x0fff_ffff, 4), (0xffff_0000, 16), (0xffff_fffe, 31), (0xffff_ffff, 0xffff_ffff)] {
        assert_eq!(v("v_ffbh_i32", [a, 0, 0]), d);
    }
    // v_mbcnt_lo/hi of -1 and 0: each lane's position (12.12's example)
    for l in [0u32, 5, 31, 32, 63] {
        let mut x = lane([u32::MAX, 0, 0]);
        x.lane = l;
        let lo = sem::valu("v_mbcnt_lo_u32_b32", &x).unwrap().d.unwrap() as u32;
        x.s[1] = u64::from(lo);
        assert_eq!(sem::valu("v_mbcnt_hi_u32_b32", &x).unwrap().d.unwrap() as u32, l);
    }
}

/// `V_MED3_F32`, `V_MIN_F32` and `V_MAX_F32` case by case (12.12, 12.7).
#[test]
fn medians_and_minimums_follow_their_cases() {
    let f = f32::to_bits;
    let nan = 0x7FC0_0000;
    assert_eq!(v("v_med3_f32", [f(1.0), f(2.0), f(3.0)]), f(2.0));
    assert_eq!(v("v_med3_f32", [f(3.0), f(1.0), f(2.0)]), f(2.0));
    // with a NaN: V_MIN3_F32 of the three, which passes over the NaN (IEEE mode clear)
    assert_eq!(v("v_med3_f32", [f(1.0), f(2.0), nan]), f(1.0));
    assert_eq!(v("v_med3_f32", [nan, f(5.0), nan]), f(5.0));
    // +0 and -0: min takes -0, max +0, whichever comes first
    assert_eq!(v("v_min_f32", [0, 0x8000_0000, 0]), 0x8000_0000);
    assert_eq!(v("v_max_f32", [0x8000_0000, 0, 0]), 0);
    // DX9's multiply: 0 times infinity is 0
    assert_eq!(v("v_mul_legacy_f32", [0, 0x7f80_0000, 0]), 0);
}

/// Half conversions round to nearest even and make half denormals (12.8).
#[test]
fn halves_round_to_nearest_even() {
    let m = Mode(0x00F0);
    assert_eq!(sem::f32_to_f16(m, 1.0f32.to_bits()), 0x3C00);
    assert_eq!(sem::f32_to_f16(m, 65504.0f32.to_bits()), 0x7BFF);
    assert_eq!(sem::f32_to_f16(m, 65520.0f32.to_bits()), 0x7C00, "half way to the next: up, to infinity");
    assert_eq!(sem::f32_to_f16(m, 2f32.powi(-24).to_bits()), 0x0001);
    assert_eq!(sem::f32_to_f16(m, 2f32.powi(-25).to_bits()), 0x0000, "a tie to even: down");
    assert_eq!(sem::f32_to_f16(m, (3.0 * 2f32.powi(-26)).to_bits()), 0x0001);
    assert_eq!(sem::f32_to_f16(Mode(0x0030), 2f32.powi(-24).to_bits()), 0x0000, "half denormals flushed");
    assert_eq!(sem::f32_to_f16(m, 1.000_976_6f32.to_bits()), 0x3C01);
    for h in [0x0001u16, 0x03FF, 0x3C00, 0xC000, 0x7BFF, 0x8001] {
        assert_eq!(sem::f32_to_f16(m, sem::f16_to_f32(h)), h, "{h:#x} there and back");
    }
}

/// 12.12's example of `V_PERMLANE16_B32`: a rotation within each row.
#[test]
fn permlane16_rotates_a_row_as_the_reference_shows() {
    let (s0, s1) = (0x8765_4321, 0x0fed_cba9);
    assert_eq!(sem::permlane_source(s0, s1, 0, false), 1);
    assert_eq!(sem::permlane_source(s0, s1, 14, false), 15);
    assert_eq!(sem::permlane_source(s0, s1, 15, false), 0);
    assert_eq!(sem::permlane_source(s0, s1, 16, false), 17);
    assert_eq!(sem::permlane_source(s0, s1, 31, false), 16);
    // x16: the other row of the pair (altrow = {row[1], ~row[0]})
    assert_eq!(sem::permlane_source(s0, s1, 15, true), 16);
    assert_eq!(sem::permlane_source(s0, s1, 31, true), 0);
    assert_eq!(sem::permlane_source(s0, s1, 47, true), 48);
}

/// 12.13.1's examples of `DS_SWIZZLE_B32` over the inputs 1..0x20.
#[test]
fn swizzles_give_the_reference_examples() {
    let out = |offset: u16| -> Vec<usize> { (0..32).map(|i| sem::swizzle_source(offset, i).unwrap() + 1).collect() };
    let fft0 = [1, 0x11, 9, 0x19, 5, 0x15, 0xd, 0x1d, 3, 0x13, 0xb, 0x1b, 7, 0x17, 0xf, 0x1f];
    assert_eq!(out(0xE000)[..16], fft0);
    let fft10 = [1, 9, 5, 0xd, 3, 0xb, 7, 0xf, 2, 0xa, 6, 0xe, 4, 0xc, 8, 0x10, 0x11, 0x19];
    assert_eq!(out(0xE010)[..18], fft10);
    assert_eq!(out(0xE01F), (1..=32).collect::<Vec<_>>(), "0x1f: no swizzle");
    let left: Vec<usize> = (2..=32).chain([1]).collect();
    assert_eq!(out(0xC000 | 1 << 5), left);
    let right: Vec<usize> = [32].into_iter().chain(1..=31).collect();
    assert_eq!(out(0xC000 | 1 << 10 | 1 << 5), right);
    // the masked rotate's example is not its pseudocode's: refused
    assert!(sem::swizzle_source(0xC000 | 1 << 5 | 1, 0).is_err());
    // SWAPX16 and BCASTX4 of thread 2
    assert_eq!(sem::swizzle_source(0x10 << 10 | 0x1f, 3).unwrap(), 19);
    assert_eq!(sem::swizzle_source(2 << 5 | 0x1c, 13).unwrap(), 14);
}

/// 12.13's examples of `DS_BPERMUTE_B32` and `DS_PERMUTE_B32` (four-lane waves there; the
/// first four lanes of a wave32 here).
#[test]
fn permutes_give_the_reference_examples() {
    let addr = [0u32, 0, 12, 4];
    let lane = |a: u32, l: usize| sem::permute_lane(a, l, 32, Wave64Permute::Unchosen).unwrap();
    // backward: VDST = {A, A, D, B}
    assert_eq!(addr.iter().enumerate().map(|(l, &a)| lane(a, l)).collect::<Vec<_>>(), [0, 0, 3, 1]);
    // a wave64 permute needs a reading chosen
    assert!(sem::permute_lane(4, 40, 64, Wave64Permute::Unchosen).is_err());
    assert_eq!(sem::permute_lane(4, 40, 64, Wave64Permute::Halves).unwrap(), 33);
    assert_eq!(sem::permute_lane(4, 40, 64, Wave64Permute::Whole).unwrap(), 1);
}

/// Table 35's range checks and the swizzled address formula (8.1.5).
#[test]
fn buffer_range_checks_follow_table_35() {
    let raw = BufferResource::decode([0x1000, 0, 64, 4 | 5 << 3 | 6 << 6 | 7 << 9 | 20 << 12 | 1 << 24 | 3 << 28]);
    assert_eq!((raw.base, raw.num_records, raw.oob_select, raw.format), (0x1000, 64, 3, 20));
    let at = |off: u32, so: u32| sem::buffer_address(&raw, 0, off, 0, so, 0);
    assert!(!sem::out_of_range(&raw, &at(60, 0), 4, 0));
    assert!(sem::out_of_range(&raw, &at(61, 0), 4, 0), "offset + payload past num_records");
    assert!(sem::out_of_range(&raw, &at(56, 8), 4, 8), "num_records reduced by the SGPR offset");
    assert_eq!(at(8, 16).address, 0x1000 + 16 + 8);
    // stride 0 with OOB_SELECT 0: every offset is at or past the stride
    let structured0 = BufferResource { oob_select: 0, ..raw };
    assert!(sem::out_of_range(&structured0, &at(0, 0), 4, 0));
    // the INVALID format on a resource that is not all zeros is refused, all zeros is unbound
    assert!(BufferResource::decode([0x1000, 0, u32::MAX, 0]).check().is_err());
    assert!(BufferResource::decode([0; 4]).check().is_ok());
    // swizzled: index_stride 8, stride 16, element 4
    let sw = BufferResource { swizzle: true, stride: 16, index_stride: 8, ..raw };
    let a: BufferAddress = sem::buffer_address(&sw, 9, 4, 0, 0, 0);
    // index_msb 1, index_lsb 1, offset_msb 1: (1*16 + 1*4) * 8 + 1*4 = 164
    assert_eq!(a.address, 0x1000 + 164);
}

/// The comparison keeps its classes apart.
#[test]
fn the_comparison_keeps_its_classes_apart() {
    let one = 1.0f32.to_bits();
    let approx = Tol { ulps: 2, ..Tol::EXACT };
    let d = compare(
        &[5, one, one, 7, 9],
        &[Tol::EXACT, approx, approx, Tol { unordered: true, ..Tol::EXACT }, Tol::UNDEFINED],
        &[5, one + 2, one + 3, 8, 1],
    );
    assert_eq!((d.exact, d.exact_differ.len()), (1, 0));
    assert_eq!((d.approximate, d.approximate_differ.len()), (2, 1));
    assert_eq!((d.unordered, d.undefined), (1, 1));
    assert_eq!(ulps(0, 0x8000_0000), 1);
    let d = compare(&[5], &[Tol::EXACT], &[6]);
    assert!(!d.agrees() && d.exact_differ[0].dword == 0);
}

// ---- small programs ----

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}
const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}
const fn ds(op: u32, vdst: u32, addr: u32, data0: u32) -> [u32; 2] {
    [0x36 << 26 | op << 18, vdst << 24 | data0 << 8 | addr]
}
/// `buffer_store_dword vdata, vaddr, s[8:11], 0 offen`
const fn store_at(vdata: u32, vaddr: u32) -> [u32; 2] {
    [0x38 << 26 | 0x1C << 18 | 1 << 12, 0x80 << 24 | 2 << 16 | vdata << 8 | vaddr]
}
const V_MOV: u32 = 1;
const V_LSHLREV: u32 = 26;
const V_SUB_NC: u32 = 38;

fn group(words: &[u32], threads: u32, lds: bool) -> (Group, Memory) {
    let mut memory = Memory::new();
    memory.map_dwords(0x1000, &vec![0; 256]);
    let rsrc = [0x1000, 0, 1024, 4 | 5 << 3 | 6 << 6 | 7 << 9 | 20 << 12 | 1 << 24 | 3 << 28];
    // s[8:11] the resource: user SGPRs s0..s11
    let mut user = vec![0; 8];
    user.extend(rsrc);
    let g = Group {
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        wave64: true,
        threads: [threads, 1, 1],
        group_id: [0; 3],
        rsrc2: 12 << 1 | u32::from(lds) << 15,
        user_sgprs: user,
        vgprs: 16,
        mode: Mode(0x00F0),
        permute: Wave64Permute::Unchosen,
        step_limit: 10_000,
    };
    (g, memory)
}

/// Two console waves exchange through LDS across `s_barrier`: every thread writes its id and
/// reads the id of thread 127 - id, which the other wave wrote. The oracle runs each wave to the
/// barrier first, so the second wave's writes are there.
#[test]
fn two_waves_meet_at_a_barrier() {
    let mut w = vec![vop2(V_LSHLREV, 1, 128 + 2, 0)]; // v1 = 4 id
    w.extend(ds(0x0D, 0, 1, 0)); // ds_write_b32 v1, v0
    w.push(S_BARRIER);
    w.extend([vop2(V_SUB_NC, 2, 255, 0), 127, vop2(V_LSHLREV, 2, 128 + 2, 2)]); // v2 = 4 (127 - id)
    w.extend(ds(0x36, 3, 2, 0)); // ds_read_b32 v3, v2
    w.extend(store_at(3, 1));
    w.push(S_ENDPGM);
    let (g, mut memory) = group(&w, 128, true);
    super::run(&g, &mut memory).expect("runs");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    for t in 0..128 {
        assert_eq!(out[t], 127 - t as u32, "thread {t}");
        assert!(tol[t].exact());
    }
    // without the barrier the first wave reads what the second has not written yet (undefined,
    // whichever comes first), and the second reads what the first wrote with nothing ordering
    // the two: which comes first is the hardware's, refused
    let no_barrier: Vec<u32> = w.iter().copied().filter(|&x| x != S_BARRIER).collect();
    let (g, mut memory) = group(&no_barrier, 128, true);
    let r = super::run(&g, &mut memory).expect_err("refused");
    assert_eq!((r.instruction.as_str(), r.reason.as_str()), ("ds_read_b32", super::WAVES_RACE));
}

/// The oracle refuses what the reference does not define, by name.
#[test]
fn what_the_reference_does_not_define_is_refused() {
    // v_mad_f32 (VOP3 0x141): not in the RDNA 2 reference
    let w = [0x35 << 26 | 0x141 << 16, 0x0101_0100, S_ENDPGM];
    let (g, mut memory) = group(&w, 64, false);
    let r = super::run(&g, &mut memory).expect_err("refused");
    assert_eq!(r.instruction, "v_mad_f32");
    assert!(!super::implements("v_mad_f32") && !super::implements("v_mac_f32") && !super::implements("image_sample_lz"));
    assert!(super::implements("v_readlane_b32") && super::implements("ds_bpermute_b32") && super::implements("s_and_saveexec_b64"));
    // a VGPR read nothing wrote and stored is undefined, not a value
    let mut w = vec![vop1(V_MOV, 5, 256 + 9), vop2(V_LSHLREV, 1, 128 + 2, 0)];
    w.extend(store_at(5, 1));
    w.push(S_ENDPGM);
    let (g, mut memory) = group(&w, 64, false);
    super::run(&g, &mut memory).expect("runs");
    assert!(memory.dwords(0x1000).unwrap().1[3].undefined);
}

// ---- the second review's findings (Sol, round one) and the laptop's card run ----

const fn sop1(op: u32, sdst: u32, src: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src
}
/// A VOP3 (or VOP3B, `sdst` in bits 14:8) instruction; `neg` in bits 31:29 of the second word.
const fn vop3(op: u32, vdst: u32, sdst: u32, src: [u32; 3], neg: u32) -> [u32; 2] {
    [0x35 << 26 | op << 16 | sdst << 8 | vdst, neg << 29 | src[2] << 18 | src[1] << 9 | src[0]]
}
const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}
const fn ds2(op: u32, vdst: u32, addr: u32, data: [u32; 2], offsets: [u32; 2]) -> [u32; 2] {
    [0x36 << 26 | op << 18 | offsets[1] << 8 | offsets[0], vdst << 24 | data[1] << 16 | data[0] << 8 | addr]
}
/// A MUBUF instruction through the resource at `s[rsrc:rsrc+3]`, SOFFSET 0.
const fn mubuf(op: u32, vdata: u32, vaddr: u32, rsrc: u32, offset: u32, offen: bool, idxen: bool) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | (idxen as u32) << 13 | (offen as u32) << 12 | offset,
        0x80 << 24 | (rsrc / 4) << 16 | vdata << 8 | vaddr,
    ]
}
const V_ADD_NC: u32 = 37;
const S_MOV_B32: u32 = 3;
const S_MOV_B64: u32 = 4;
const RAW_W3: u32 = 4 | 5 << 3 | 6 << 6 | 7 << 9 | 20 << 12 | 1 << 24 | 3 << 28;

/// The decoder's names of `words`, to keep a fixture's encodings honest.
fn names(words: &[u32]) -> Vec<&'static str> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < code.len() {
        let i = crate::gfx10::instruction(&code, at).expect("decodes");
        out.push(i.name().unwrap_or("?"));
        at += 4 * i.words.len();
    }
    out
}

/// Runs `words` as [`group`] lays it out, with s0..s7 from `user` and `threads` threads.
fn run_words(words: &[u32], threads: u32, lds: bool, user: &[u32]) -> Result<Memory, super::Refusal> {
    let (mut g, mut memory) = group(words, threads, lds);
    for (i, &u) in user.iter().enumerate() {
        g.user_sgprs[i] = u;
    }
    super::run(&g, &mut memory).map(|_| memory)
}

/// v1 = 4 thread, and `vdata` stored at the thread's own dword.
fn store_own(vdata: u32) -> Vec<u32> {
    let mut w = vec![vop2(V_LSHLREV, 1, 128 + 2, 0)];
    w.extend(store_at(vdata, 1));
    w
}

/// 12.1: `D.i = (S0.i >> S1.u[4:0]) & ((1 << S1.u[22:16]) - 1)`, sign-extended from the width:
/// the shift is of a signed source, so bits past bit 31 are its sign.
#[test]
fn s_bfe_i32_extracts_from_its_source_shifted_as_signed() {
    let bfe = |s0: u32, s1: u32| {
        sem::salu("s_bfe_i32", SaluIn { s0: u64::from(s0), s1: u64::from(s1), ..SaluIn::default() }).unwrap().d.unwrap() as u32
    };
    assert_eq!(bfe(0x8000_0000, 31 | 2 << 16), 0xFFFF_FFFF, "offset 31, width 2: the sign twice");
    assert_eq!(bfe(0x0000_0030, 4 | 2 << 16), 0xFFFF_FFFF, "0b11 as two signed bits");
    assert_eq!(bfe(0x0000_0010, 4 | 2 << 16), 1);
}

/// `D.u = ({S0, S1} >> (8 * count)) & 0xffffffff` for a count both readings share (0..3; the
/// counts past them are refused, below).
#[test]
fn v_alignbyte_shifts_by_eight_times_the_count() {
    assert_eq!(v("v_alignbyte_b32", [0x1122_3344, 0x5566_7788, 1]), 0x4455_6677);
}

/// 6.3 lists `V_MOV_FED_B32` without giving what it does: no plain move is assumed.
#[test]
fn v_mov_fed_b32_has_no_operation_in_the_reference() {
    assert!(sem::valu("v_mov_fed_b32", &lane([5, 0, 0])).is_err());
}

/// A minimum or maximum past a NaN gives the other operand; whether that operand is flushed
/// where the mode flushes denormals the entries do not say: refused there, kept where denormals
/// are kept.
#[test]
fn a_float_minimum_past_a_nan_beside_a_flushed_denormal_is_refused() {
    let flush = Lane { mode: Mode(0), ..lane([0x7FC0_0000, 1, 0]) };
    assert!(sem::valu("v_min_f32", &flush).is_err(), "qNaN and a denormal, flush both");
    assert!(sem::valu("v_max3_f32", &flush).is_err());
    assert_eq!(v("v_min_f32", [0x7FC0_0000, 1, 0]), 1, "denormals kept: the denormal");
    let f64nan = 0x7FF8_0000_0000_0000u64;
    let flush64 = Lane { s: [f64nan, 1, 0], mode: Mode(0), ..lane([0; 3]) };
    assert!(sem::valu("v_max_f64", &flush64).is_err());
}

/// 12.8's example: the cosine of a quarter turn is 0 (its bits, `0x0`); sine and cosine follow
/// the output denormal mode as every other result does.
#[test]
fn sine_and_cosine_flush_their_output_and_a_quarter_turn_is_exact() {
    assert_eq!(v("v_cos_f32", [0x3E80_0000, 0, 0]), 0, "cos(0.25 turn)");
    // input denormals allowed, output flushed: sin(2 pi 2^-140) is a denormal, so 0
    let l = Lane { mode: Mode(0x0010), ..lane([0x200, 0, 0]) };
    assert_eq!(sem::valu("v_sin_f32", &l).unwrap().d.unwrap() as u32, 0);
}

/// `V_RCP_IFLAG_F32` has its own entry, without the ordinary reciprocal's "denormals are
/// flushed": a denormal input or result is refused, a normal one is the reciprocal.
#[test]
fn v_rcp_iflag_f32_of_a_denormal_is_refused() {
    assert!(sem::valu("v_rcp_iflag_f32", &lane([1, 0, 0])).is_err());
    assert!(sem::valu("v_rcp_iflag_f32", &lane([0x7F00_0000, 0, 0])).is_err(), "1/2^127 is a denormal");
    assert_eq!(v("v_rcp_iflag_f32", [2.0f32.to_bits(), 0, 0]), 0.5f32.to_bits());
}

/// An approximate value is never matched by an infinity or a NaN one ULP away, nor an infinity
/// by the largest finite value.
#[test]
fn the_comparison_takes_no_infinity_or_nan_for_a_nearby_value() {
    let approx = Tol { ulps: 2, ..Tol::EXACT };
    for (a, b) in [(0x7F80_0000u32, 0x7F80_0001u32), (0x7F7F_FFFF, 0x7F80_0000), (0x7F80_0000, 0x7F7F_FFFF)] {
        let d = compare(&[a], &[approx], &[b]);
        assert!(!d.agrees(), "{a:#x} against {b:#x}");
    }
}

/// A read-back shorter or longer than the oracle's output does not agree.
#[test]
fn the_comparison_rejects_a_read_back_of_another_length() {
    assert!(!compare(&[1, 2], &[Tol::EXACT; 2], &[1]).agrees(), "a missing dword");
    assert!(!compare(&[1], &[Tol::EXACT], &[1, 2]).agrees(), "an extra dword");
    assert!(!compare(&[1, 2], &[Tol::EXACT], &[1, 2]).agrees(), "a dword without its trust");
}

/// `V_CVT_I32_F32(-1.0)` is the integer -1: its bits are a NaN's as a float, but it is an
/// integer and exact.
#[test]
fn an_integer_from_a_float_conversion_is_exact_and_no_nan() {
    let mut w = vec![vop1(V_MOV, 5, 243), vop1(8, 6, 256 + 5)]; // v5 = -1.0, v6 = cvt_i32(v5)
    w.extend(store_own(6));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[1], "v_cvt_i32_f32");
    let memory = run_words(&w, 64, false, &[]).expect("runs");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    assert_eq!(out[3], 0xFFFF_FFFF);
    assert!(tol[3].exact(), "{:?}", tol[3]);
}

/// 12.3: a `_B32` save-exec writes EXEC_LO; on a wave of 64 EXEC_HI keeps its lanes.
#[test]
fn a_32_bit_saveexec_on_a_wave64_keeps_exec_hi() {
    let mut w = vec![sop1(S_MOV_B32, 21, 193), sop1(0x3C, 20, 21), vop1(V_MOV, 5, 128 + 7)];
    w.extend(store_own(5));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[1], "s_and_saveexec_b32");
    let memory = run_words(&w, 64, false, &[]).expect("runs");
    let (out, _) = memory.dwords(0x1000).unwrap();
    assert_eq!((out[3], out[40]), (7, 7), "lanes 3 and 40 both still run");
}

/// 3.6.1: a destination past the program's VGPRs makes the instruction write nothing, its
/// carry-out (a VOP3B SGPR result) too.
#[test]
fn a_carry_out_beside_a_destination_past_the_programs_vgprs_is_not_written() {
    let mut w = vec![sop1(S_MOV_B32, 30, 128), sop1(S_MOV_B32, 31, 128)];
    w.extend(vop3(0x30F, 100, 30, [193, 193, 0], 0)); // v_add_co_u32 v100, s[30:31], -1, -1
    w.push(vop1(V_MOV, 5, 30)); // v5 = s30
    w.extend(store_own(5));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[2], "v_add_co_u32");
    let memory = run_words(&w, 64, false, &[]).expect("runs");
    assert_eq!(memory.dwords(0x1000).unwrap().0[0], 0, "s30 kept its 0");
}

/// 3.6.3: a 64-bit SGPR operand starts at an even SGPR.
#[test]
fn an_sgpr_pair_at_an_odd_register_is_refused() {
    let w = [sop1(S_MOV_B64, 9, 128), S_ENDPGM];
    assert_eq!(names(&w)[0], "s_mov_b64");
    let r = run_words(&w, 64, false, &[]).expect_err("refused");
    assert!(r.reason.contains("even"), "{r}");
}

/// 13.3.9's row and bank masks are given for VGPR writes: a compare through DPP, whose result
/// is a lane mask, is refused.
#[test]
fn a_compare_through_dpp_is_refused() {
    // v_cmp_eq_u32 vcc, v0 (quad_perm identity, row_mask 1, bank_mask 0xf), v0
    let w = [vopc(0xC2, 250, 0), 0xE4 << 8 | 0xF << 24 | 1 << 28, S_ENDPGM];
    assert_eq!(names(&w)[0], "v_cmp_eq_u32");
    let r = run_words(&w, 64, false, &[]).expect_err("refused");
    assert!(r.reason.contains("DPP"), "{r}");
}

/// The lane instructions take no input modifiers, and their lane select is an SGPR, M0 or an
/// inline integer 0..63 (12.12; AMD's RDNA 2 machine-readable ISA, `OPR_SSRC_LANESEL`): a
/// float constant there is refused.
#[test]
fn a_lane_read_with_a_modifier_or_another_lane_select_is_refused() {
    let read = |sel: u32, neg: u32| {
        let mut w = vop3(0x360, 20, 0, [256, sel, 0], neg).to_vec(); // v_readlane_b32 s20, v0, sel
        w.push(vop1(V_MOV, 5, 20));
        w.extend(store_own(5));
        w.push(S_ENDPGM);
        assert_eq!(names(&w)[0], "v_readlane_b32");
        run_words(&w, 64, false, &[])
    };
    assert_eq!(read(128 + 5, 0).expect("an inline 5").dwords(0x1000).unwrap().0[0], 5);
    assert!(read(128 + 5, 1).is_err(), "NEG on source 0");
    assert!(read(242, 0).is_err(), "1.0 as a lane select");
}

/// 10.4.3: a `DS_WRITE2` whose two offsets are one place makes one access, of DATA0.
#[test]
fn a_write2_at_one_place_writes_data0() {
    let mut w = vec![vop1(V_MOV, 1, 128), vop1(V_MOV, 2, 128 + 17), vop1(V_MOV, 3, 128 + 34)];
    w.extend(ds2(0x0E, 0, 1, [2, 3], [0, 0])); // ds_write2_b32 v1, v2, v3
    w.extend(ds(0x36, 4, 1, 0)); // ds_read_b32 v4, v1
    w.extend(store_own(4));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[3], "ds_write2_b32");
    let memory = run_words(&w, 64, true, &[]).expect("runs");
    assert_eq!(memory.dwords(0x1000).unwrap().0[0], 17);
}

/// 3.6.1: a destination range that runs past the program's VGPRs is checked before any access:
/// the oracle refuses it rather than write part of it, or change memory for a returning atomic.
#[test]
fn a_destination_past_the_programs_vgprs_is_refused_before_any_access() {
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(ds(0x76, 15, 1, 0)); // ds_read_b64 v[15:16], v1
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[1], "ds_read_b64");
    assert!(run_words(&w, 64, true, &[]).is_err(), "v16 is past the 16 VGPRs");
    let mut w = vec![vop1(V_MOV, 1, 128), vop1(V_MOV, 2, 129)];
    w.extend(ds(0x20, 20, 1, 2)); // ds_add_rtn_u32 v20, v1, v2
    w.push(S_ENDPGM);
    assert!(run_words(&w, 64, true, &[]).is_err(), "a returning atomic into v20");
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(mubuf(0x0D, 15, 1, 8, 0, true, false)); // buffer_load_dwordx2 v[15:16], v1, s[8:11] offen
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[1], "buffer_load_dwordx2");
    assert!(run_words(&w, 64, false, &[]).is_err(), "a buffer load into v[15:16]");
}

/// Buffer layouts the reference does not settle are refused: a swizzled stride that is no
/// multiple of 4 (8.1.5), a format access off its alignment (8.1.7), a swizzled format element
/// of more than one dword (the swizzle formula is per dword), and an unswizzled stride with
/// OOB_SELECT 3 (Tables 33 and 37 disagree on NumRecords' units there).
#[test]
fn buffer_layouts_the_reference_does_not_settle_are_refused() {
    let load = |op: u32, offset: u32, offen: bool, idxen: bool, rsrc: [u32; 4]| {
        let mut w = vec![vop1(V_MOV, 1, 128)];
        w.extend(mubuf(op, 5, if idxen { 0 } else { 1 }, 4, offset, offen, idxen));
        w.extend(store_own(5));
        w.push(S_ENDPGM);
        let mut user = vec![0; 4];
        user.extend(rsrc);
        run_words(&w, 64, false, &user)
    };
    let swizzled = |stride: u32, format: u32| {
        [0x1000, stride << 16 | 1 << 31, 64, (RAW_W3 & !(0x7F << 12)) | format << 12]
    };
    assert!(load(0x0C, 0, false, true, swizzled(6, 20)).is_err(), "a swizzled stride of 6");
    assert!(load(0x00, 2, true, false, [0x1000, 0, 1024, RAW_W3]).is_err(), "a format load at byte 2");
    assert!(load(0x03, 0, false, true, swizzled(16, 77)).is_err(), "a swizzled element of 16 bytes");
    assert!(load(0x0C, 0, false, true, [0x1000, 16 << 16, 64, RAW_W3]).is_err(), "stride 16, OOB_SELECT 3");
    // and the all-zero resource of 8.1.8 reads zeros through a format load too
    let memory = load(0x00, 0, true, false, [0; 4]).expect("an unbound resource reads 0");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    assert!(out[..64].iter().all(|&x| x == 0) && tol[5].exact());
}

/// An atomic whose address is past the LDS returns 0 to every lane that runs it, in any order:
/// exact, not order-dependent (the lanes do not race for a dword that is not there).
#[test]
fn returns_of_an_atomic_past_the_lds_are_exact_zeros() {
    let mut w = vec![vop1(V_MOV, 1, 255), 0x1_0000, vop1(V_MOV, 2, 129)];
    w.extend(ds(0x20, 5, 1, 2)); // ds_add_rtn_u32 v5, v1, v2
    w.extend(store_own(5));
    w.push(S_ENDPGM);
    let memory = run_words(&w, 64, true, &[]).expect("runs");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    for t in 0..64 {
        assert!(out[t] == 0 && tol[t].exact(), "thread {t}: {:#x} {:?}", out[t], tol[t]);
    }
}

/// Every lane exchanges its id into one LDS dword: which lands last is the hardware's, so what
/// the dword holds after is order-dependent.
#[test]
fn an_exchange_lanes_race_for_leaves_an_order_dependent_value() {
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(ds(0x0D, 0, 1, 1)); // ds_write_b32 v1, v1: the dword is 0 first
    w.extend(ds(0x2D, 5, 1, 0)); // ds_wrxchg_rtn_b32 v5, v1, v0
    w.extend(ds(0x36, 4, 1, 0)); // ds_read_b32 v4, v1
    w.extend(store_own(4));
    w.push(S_ENDPGM);
    let memory = run_words(&w, 64, true, &[]).expect("runs");
    let (_, tol) = memory.dwords(0x1000).unwrap();
    assert!(tol[0].unordered, "{:?}", tol[0]);
}

/// Lane 0 of each of two waves adds 1 to one LDS dword and gets the old value back, with no
/// barrier between: which wave comes first is the hardware's, and the oracle runs them in one
/// order: refused. The same adds without the returns, read after a barrier, are 2.
#[test]
fn returning_atomics_of_two_waves_on_one_dword_between_barriers_are_refused() {
    let program = |rtn: bool| {
        let mut w = vec![
            vop1(V_MOV, 1, 128),
            vop1(V_MOV, 2, 129),
            vop2(0x1B, 3, 128 + 63, 0), // v3 = thread & 63
            vopc(0xC2, 128, 3),         // vcc = (0 == v3)
        ];
        w.extend(ds(0x0D, 0, 1, 1)); // ds_write_b32 v1, v1: LDS dword 0 = 0, by every lane
        w.push(S_BARRIER);
        w.push(sop1(0x24, 20, 106)); // s_and_saveexec_b64 s[20:21], vcc
        if rtn {
            w.extend(ds(0x20, 5, 1, 2)); // ds_add_rtn_u32 v5, v1, v2
        } else {
            w.extend(ds(0x00, 0, 1, 2)); // ds_add_u32 v1, v2
        }
        w.push(sop1(S_MOV_B64, 126, 20)); // exec = s[20:21]
        w.push(S_BARRIER);
        w.extend(ds(0x36, 4, 1, 0)); // ds_read_b32 v4, v1
        w.extend(store_own(4));
        w.push(S_ENDPGM);
        w
    };
    let w = program(true);
    assert_eq!(names(&w)[6..8], ["s_and_saveexec_b64", "ds_add_rtn_u32"]);
    let r = run_words(&w, 128, true, &[]).expect_err("refused");
    assert!(r.reason.contains("wave"), "{r}");
    let memory = run_words(&program(false), 128, true, &[]).expect("commuting adds run");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    assert!(out[..128].iter().all(|&x| x == 2) && tol[100].exact(), "{:?} {:?}", &out[..4], tol[100]);
}

/// The laptop's card run: all 64 lanes store `0x100 + thread` to one dword. Which lane's store
/// lands is the hardware's (NVIDIA keeps lane 32's, the oracle ran lane 63's last): the dword
/// is order-dependent. A dword every lane stores the same value to stays exact.
#[test]
fn a_dword_many_lanes_store_to_is_order_dependent() {
    let mut w = vec![vop2(V_ADD_NC, 7, 255, 0), 0x100, vop1(V_MOV, 8, 128 + 5)];
    w.extend(mubuf(0x1C, 7, 0, 8, 0x100, false, false)); // buffer_store_dword v7, off, s[8:11] offset:256
    w.extend(mubuf(0x1C, 8, 0, 8, 0x200, false, false)); // buffer_store_dword v8, off, s[8:11] offset:512
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[2], "buffer_store_dword");
    let memory = run_words(&w, 64, false, &[]).expect("runs");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    assert!(tol[64].unordered, "dword 64: {:#x} {:?}", out[64], tol[64]);
    assert!(out[128] == 5 && tol[128].exact(), "dword 128: every lane stored 5");
    let choices = memory.choices(0x1000).unwrap();
    assert_eq!(choices[64], (0x100..0x140).collect::<Vec<u32>>(), "the values the 64 lanes stored");
    // the card's lane 32 is one of them; a value no lane stored is not
    let mut host = out.clone();
    host[64] = 0x120;
    let d = super::compare::compare_choices(&out, &tol, &choices, &host);
    assert!(d.agrees() && d.complete() && (d.unordered, d.unordered_checked) == (1, 1), "{}", d.line());
    host[64] = 0x99;
    assert!(!super::compare::compare_choices(&out, &tol, &choices, &host).agrees());
}

/// Every lane adds 1 to one dword through a returning buffer atomic: what each gets back is
/// order-dependent, the sum left is not (64). Past the range every lane gets 0, exactly.
#[test]
fn a_returning_buffer_add_leaves_an_exact_sum_and_past_the_range_returns_zero() {
    let add = |offset: u32| {
        let mut w = vec![vop1(V_MOV, 1, 255), offset, vop1(V_MOV, 5, 129)];
        let mut a = mubuf(0x32, 5, 1, 8, 0, true, false); // buffer_atomic_add v5, v1, s[8:11] offen glc
        a[0] |= 1 << 14;
        w.extend(a);
        w.extend([vop2(V_LSHLREV, 1, 128 + 2, 0)]);
        w.extend(store_at(5, 1)); // the return, at the thread's dword
        w.push(S_ENDPGM);
        assert_eq!(names(&w)[2], "buffer_atomic_add");
        run_words(&w, 64, false, &[]).expect("runs").dwords(0x1000).unwrap()
    };
    let (out, tol) = add(0x200); // dword 128
    assert!(out[128] == 64 && tol[128].exact(), "the sum: {:#x} {:?}", out[128], tol[128]);
    assert!(tol[3].unordered, "a return the lanes raced for");
    let (out, tol) = add(0x1_0000);
    assert!((0..64).all(|t| out[t] == 0 && tol[t].exact()), "{:?}", tol[0]);
}

/// A byte written to LDS reads back as written: the dword's other bytes, which nothing wrote,
/// do not make it undefined.
#[test]
fn a_byte_written_to_lds_reads_back_defined() {
    let mut w = vec![vop2(V_LSHLREV, 1, 128 + 2, 0), vop1(V_MOV, 2, 128 + 5)];
    w.extend(ds(0x1E, 0, 1, 2)); // ds_write_b8 v1, v2
    w.extend(ds(0x3A, 4, 1, 0)); // ds_read_u8 v4, v1
    w.extend(store_at(4, 1));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[3], "ds_read_u8");
    let memory = run_words(&w, 64, true, &[]).expect("runs");
    let (out, tol) = memory.dwords(0x1000).unwrap();
    assert!(out[7] == 5 && tol[7].exact(), "{:#x} {:?}", out[7], tol[7]);
}

/// Lanes 0 and 1 of one `ds_write_b64` overlap at bytes 4..8 with different values: which lands
/// there is the hardware's.
#[test]
fn overlapping_lds_writes_of_one_instruction_are_order_dependent() {
    let mut w = vec![vop2(V_LSHLREV, 1, 128 + 2, 0), vop1(V_MOV, 2, 256), vop1(V_MOV, 3, 256)];
    w.extend(ds(0x4D, 0, 1, 2)); // ds_write_b64 v1, v[2:3]
    w.extend(ds(0x36, 4, 1, 0)); // ds_read_b32 v4, v1
    w.extend(store_at(4, 1));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[3], "ds_write_b64");
    let memory = run_words(&w, 2, true, &[]).expect("runs");
    let (_, tol) = memory.dwords(0x1000).unwrap();
    assert!(tol[1].unordered, "lane 1 read bytes 4..8: {:?}", tol[1]);
}

/// An approximate value goes only where its tolerance still bounds the result: into a
/// multiplication, not into an addition (cancellation), a conversion to an integer or a compare.
#[test]
fn an_approximate_value_goes_only_where_its_tolerance_holds() {
    let after_rcp = |next: &[u32]| {
        let mut w = vec![vop1(V_MOV, 5, 246), vop1(0x2A, 6, 256 + 5)]; // v5 = 4.0, v6 = rcp(v5)
        w.extend(next);
        w.extend(store_own(7));
        w.push(S_ENDPGM);
        run_words(&w, 64, false, &[])
    };
    let memory = after_rcp(&[vop2(8, 7, 256 + 6, 5)]).expect("a product runs"); // v7 = v6 * v5
    assert!(memory.dwords(0x1000).unwrap().1[0].ulps > 0);
    assert!(after_rcp(&[vop2(3, 7, 243, 6)]).is_err(), "-1.0 + rcp(4): an addition");
    assert!(after_rcp(&[vop1(8, 7, 256 + 6)]).is_err(), "cvt_i32 of rcp(4)");
    let cmp = [vopc(0xC2, 256 + 6, 6), vop1(V_MOV, 7, 106)];
    assert!(after_rcp(&cmp).is_err(), "a compare of rcp(4)");
}

// ---- the second review's round two (Sol): false-exact cases ----

const S_CBRANCH_SCC1: u32 = 0xBF85_0000;

/// `run` was refused, for a reason with `why` in it.
fn refused(run: Result<Memory, super::Refusal>, why: &str) {
    match run {
        Err(r) => assert!(r.reason.contains(why), "refused for another reason than \"{why}\": {r}"),
        Ok(_) => panic!("not refused (\"{why}\")"),
    }
}

/// An EXEC_HI nothing wrote reaches no exact result: not through a 64-bit save-exec's SCC,
/// nor through EXECZ.
#[test]
fn an_exec_hi_nothing_wrote_reaches_no_exact_result() {
    let saveexec = [
        sop1(S_MOV_B32, 127, 40), // exec_hi = s40, which nothing wrote
        sop1(S_MOV_B32, 22, 193),
        sop1(S_MOV_B32, 23, 193),
        sop1(0x24, 20, 22), // s_and_saveexec_b64 s[20:21], s[22:23]
        S_CBRANCH_SCC1,
        S_ENDPGM,
    ];
    assert_eq!(names(&saveexec)[3], "s_and_saveexec_b64");
    refused(run_words(&saveexec, 64, false, &[]), "nothing wrote");
    let execz = [
        sop1(S_MOV_B32, 127, 40),
        sop1(S_MOV_B32, 20, 252),         // s20 = execz
        0xBF00_0000 | 6 << 16 | 128 << 8 | 20, // s_cmp_eq_u32 s20, 0
        S_CBRANCH_SCC1,
        S_ENDPGM,
    ];
    assert_eq!(names(&execz)[2], "s_cmp_eq_u32");
    refused(run_words(&execz, 64, false, &[]), "nothing wrote");
}

/// 7.5: the destination of a four-dword scalar load is a multiple of four SGPRs.
#[test]
fn a_four_dword_scalar_load_at_s2_is_refused() {
    let w = [0x3D << 26 | 0x02 << 18 | 2 << 6 | 4, 125 << 25, S_ENDPGM]; // s_load_dwordx4 s[2:5], s[8:9], 0
    assert_eq!(names(&w)[0], "s_load_dwordx4");
    refused(run_words(&w, 64, false, &[]), "multiple of 4");
}

/// A NaN made by a double or a half operation has bits the reference does not pin: never an
/// exact result.
#[test]
fn nans_of_doubles_and_halves_are_not_exact() {
    let mut w = vec![vop1(V_MOV, 2, 128), vop1(V_MOV, 3, 255), 0x7FF8_0001, vop1(V_MOV, 4, 128), vop1(V_MOV, 5, 255), 0x7FF8_0002];
    w.extend(vop3(0x164, 6, 0, [256 + 2, 256 + 4, 0], 0)); // v_add_f64 v[6:7], v[2:3], v[4:5]
    w.extend(store_own(6));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[4], "v_add_f64");
    refused(run_words(&w, 64, false, &[]), "a NaN of a double or a half");
    let mut w = vec![vop1(V_MOV, 5, 255), 0x7FC0_0001, vop1(0x0A, 6, 256 + 5)]; // v6 = cvt_f16(NaN)
    w.extend(store_own(6));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[1], "v_cvt_f16_f32");
    refused(run_words(&w, 64, false, &[]), "a NaN of a double or a half");
}

/// An approximate result at the edge of the normal range (where a host's rounding could make
/// it a denormal, then a flushed zero, and its reciprocal infinite) is refused.
#[test]
fn an_approximate_value_at_the_edge_of_the_normal_range_is_refused() {
    let mut w = vec![vop1(V_MOV, 5, 255), 0x7E80_0000, vop1(0x2A, 6, 256 + 5)]; // v6 = rcp(2^126) = 2^-126
    w.extend(store_own(6));
    w.push(S_ENDPGM);
    refused(run_words(&w, 64, false, &[]), "edge of the normal range");
}

/// A byte loaded from a dword the oracle holds approximate is an integer with no ULPs.
#[test]
fn a_byte_of_an_approximate_dword_is_refused() {
    let mut w = vec![vop1(V_MOV, 5, 246), vop1(0x2A, 6, 256 + 5), vop1(V_MOV, 1, 128)]; // v6 = rcp(4)
    w.extend(store_at(6, 1));
    w.extend(mubuf(0x08, 7, 1, 8, 0, true, false)); // buffer_load_ubyte v7, v1, s[8:11] offen
    w.extend(store_own(7));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[4], "buffer_load_ubyte");
    refused(run_words(&w, 64, false, &[]), "no ULPs");
}

/// 8.1.7: a format access of more than one channel is aligned to its element; byte 4 of an
/// 8-byte 32_32 element is aligned to the channel only, refused.
#[test]
fn a_two_channel_format_load_off_its_element_is_refused() {
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(mubuf(0x01, 5, 1, 4, 4, true, false)); // buffer_load_format_xy v[5:6], v1, s[4:7] offen offset:4
    w.extend(store_own(5));
    w.push(S_ENDPGM);
    let rsrc = [0, 0, 0, 0, 0x1000, 0, 1024, (RAW_W3 & !(0x7F << 12)) | 62 << 12];
    refused(run_words(&w, 64, false, &rsrc), "8-byte element");
}

/// 3.6.1: a memory instruction's VGPR source past the program's VGPRs has no defined value
/// (the VALU's fallback to v0 is not the memory instructions'): refused.
#[test]
fn a_memory_source_past_the_programs_vgprs_is_refused() {
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(ds(0x0D, 0, 1, 20)); // ds_write_b32 v1, v20
    w.extend(ds(0x36, 5, 1, 0)); // ds_read_b32 v5, v1
    w.extend(store_own(5));
    w.push(S_ENDPGM);
    refused(run_words(&w, 64, true, &[]), "a source v20");
    let mut w = vec![vop1(V_MOV, 1, 128)];
    w.extend(store_at(20, 1)); // buffer_store_dword v20, v1
    w.push(S_ENDPGM);
    refused(run_words(&w, 64, false, &[]), "a source v20");
}

/// Two waves' lane 0 add 1 and XOR 1 into one dword: each commutes with itself, not with the
/// other (0 or 2 is left): refused.
#[test]
fn two_different_commuting_atomics_on_one_dword_are_refused() {
    let mut w = vec![vop1(V_MOV, 1, 255), 0x200, vop1(V_MOV, 2, 129), vopc(0xC2, 128, 0), sop1(0x24, 20, 106)];
    w.extend(mubuf(0x32, 2, 1, 8, 0, true, false)); // buffer_atomic_add v2, v1 (thread 0)
    w.extend([sop1(S_MOV_B64, 126, 20), vopc(0xC2, 255, 0), 64, sop1(0x24, 20, 106)]);
    w.extend(mubuf(0x3B, 2, 1, 8, 0, true, false)); // buffer_atomic_xor v2, v1 (thread 64)
    w.extend([sop1(S_MOV_B64, 126, 20), S_ENDPGM]);
    assert_eq!(names(&w)[4], "buffer_atomic_add");
    refused(run_words(&w, 128, false, &[]), "mixed atomics");
}

/// Two lanes store 0x100 and 0x200 to one dword, then both store 1 to its byte 1: 0x101 or 0x201
/// is left, which no whole-dword choice gives: refused.
#[test]
fn a_raced_dword_partly_overwritten_is_refused() {
    let mut w = vec![vop1(V_MOV, 8, 255), 0x100, vop2(V_LSHLREV, 7, 256, 8), vop1(V_MOV, 9, 129)]; // v7 = 0x100 << thread
    w.extend(mubuf(0x1C, 7, 0, 8, 0x100, false, false));
    w.extend(mubuf(0x18, 9, 0, 8, 0x101, false, false)); // buffer_store_byte v9, off offset:0x101
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[4], "buffer_store_byte");
    refused(run_words(&w, 2, false, &[]), "overlap in part");
}

/// Lane 0 stores 1, a barrier, then lane 1 stores 5: whether the barrier orders the two stores
/// (the waits before it) is not modelled, and 1 is no answer to accept: refused.
#[test]
fn stores_of_two_lanes_across_a_barrier_are_refused() {
    let only = |n: u32| [vopc(0xC2, 128 + n, 0), sop1(0x24, 20, 106)];
    let mut w = vec![vop1(V_MOV, 7, 129), vop1(V_MOV, 8, 133)];
    w.extend(only(0));
    w.extend(mubuf(0x1C, 7, 0, 8, 0x100, false, false));
    w.extend([sop1(S_MOV_B64, 126, 20), S_BARRIER]);
    w.extend(only(1));
    w.extend(mubuf(0x1C, 8, 0, 8, 0x100, false, false));
    w.extend([sop1(S_MOV_B64, 126, 20), S_ENDPGM]);
    refused(run_words(&w, 2, false, &[]), "either side of a barrier");
}

/// 12.8: `V_CVT_RPI_I32_F32` is `floor(S0.f + 0.5)` of the value, not of a rounded single sum:
/// the float just below 0.5 gives 0.
#[test]
fn v_cvt_rpi_of_just_below_a_half_is_zero() {
    assert_eq!(v("v_cvt_rpi_i32_f32", [0x3EFF_FFFF, 0, 0]), 0);
}

/// SDWA with DST_UNUSED = PRESERVE keeps the old bits, and their trust: a byte written into a
/// VGPR nothing wrote leaves a dword that is not exact.
#[test]
fn an_sdwa_byte_into_an_unwritten_vgpr_is_not_exact() {
    let mut w = vec![vop1(V_MOV, 6, 249), 2 << 11 | 6 << 16]; // v_mov_b32_sdwa v6, v0 dst_sel:BYTE_0 dst_unused:PRESERVE
    w.extend(store_own(6));
    w.push(S_ENDPGM);
    assert_eq!(names(&w)[0], "v_mov_b32");
    let memory = run_words(&w, 64, false, &[]).expect("runs");
    assert!(memory.dwords(0x1000).unwrap().1[0].undefined);
}

/// `V_ALIGNBYTE_B32` at a count of 4 or more: the reference's 8 * S2[4:0] (RDNA1/2 12.12) and
/// RDNA3's 8 * S2[1:0] (16.12) disagree there: refused.
#[test]
fn v_alignbyte_of_a_count_of_4_or_more_is_refused() {
    for count in [4, 7, 8, 31] {
        assert!(sem::valu("v_alignbyte_b32", &lane([0x1122_3344, 0x5566_7788, count])).is_err(), "count {count}");
    }
    assert_eq!(v("v_alignbyte_b32", [0x1122_3344, 0x5566_7788, 3]), 0x2233_4455);
}

/// A comparison that could not be made dword by dword is not complete.
#[test]
fn a_shape_failure_is_not_complete() {
    let d = compare(&[1], &[Tol::EXACT], &[1, 2]);
    assert!(!d.agrees() && !d.complete(), "{}", d.line());
}

/// Coverage of the console's programs, offline: with `VICTORY_ORACLE_COVERAGE=<folder>` holding
/// `names.tsv` (group, name, occurrences, programs: one row per instruction of a group of
/// programs), writes `missing.tsv` there with the rows of the instructions the oracle does not
/// implement. Nothing is printed of either file.
#[test]
#[ignore = "an offline tool: run with VICTORY_ORACLE_COVERAGE set"]
fn coverage_of_a_census() {
    let Some(dir) = std::env::var_os("VICTORY_ORACLE_COVERAGE") else { return };
    let dir = std::path::PathBuf::from(dir);
    let names = std::fs::read_to_string(dir.join("names.tsv")).expect("names.tsv");
    let mut out = String::from("group\tname\toccurrences\tprograms\n");
    let (mut have, mut missing) = (0, 0);
    for row in names.lines().skip(1) {
        let name = row.split('\t').nth(1).unwrap_or("");
        if super::implements(name) {
            have += 1;
        } else {
            missing += 1;
            out.push_str(row);
            out.push('\n');
        }
    }
    std::fs::write(dir.join("missing.tsv"), out).expect("missing.tsv");
    eprintln!("coverage: {have} rows implemented, {missing} missing");
}
