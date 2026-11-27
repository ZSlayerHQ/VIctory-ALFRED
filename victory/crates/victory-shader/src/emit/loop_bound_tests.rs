//! Loops the console's own counter bounds past a stage's cap (`crate::bounds`): synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU (`VICTORY_WARP=1`: on WARP). A proved loop runs its console count; a loop
//! whose bound is read from data, or whose counter the body also changes, keeps the cap.

use super::bitcode::{Attr, Term};
use super::container::{self, SigElement};
use super::host::{Dispatch, Draw, Host, Root, View};
use super::shader::Shader;
use super::{HostShader, lower, validator};
use crate::bounds::{self, Bound};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC_LO: u32 = 126;
const VCC_LO: u32 = 106;
/// Inline constants: 0-64 and -1.
const fn k(n: u32) -> u32 {
    128 + n
}
const MINUS_ONE: u32 = 193;
const ONE_F: u32 = 0xF2;
/// A VGPR as a VALU source.
const fn v(n: u32) -> u32 {
    256 + n
}

const fn sop1(op: u32, dst: u32, src: u32) -> u32 {
    0xBE80_0000 | dst << 16 | op << 8 | src
}
const fn sop2(op: u32, dst: u32, a: u32, b: u32) -> u32 {
    0x8000_0000 | op << 23 | dst << 16 | b << 8 | a
}
const fn sopk(op: u32, dst: u32, imm: u32) -> u32 {
    0xB000_0000 | op << 23 | dst << 16 | imm
}
/// A branch at byte `at` to byte `to`.
const fn sopp(op: u32, at: u32, to: u32) -> u32 {
    0xBF80_0000 | op << 16 | ((to as i32 - at as i32 - 4) / 4) as u32 & 0xFFFF
}
const fn sopc(op: u32, a: u32, b: u32) -> u32 {
    0xBF00_0000 | op << 16 | b << 8 | a
}
const fn vop1(op: u32, dst: u32, src: u32) -> u32 {
    0x7E00_0000 | dst << 17 | op << 9 | src
}
const fn vop2(op: u32, dst: u32, a: u32, b: u32) -> u32 {
    op << 25 | dst << 17 | b << 9 | a
}
const fn vopc(op: u32, a: u32, b: u32) -> u32 {
    0x7C00_0000 | op << 17 | b << 9 | a
}

const S_MOV_B32: u32 = 0x03;
const S_ADD_I32: u32 = 2;
const S_CSELECT_B32: u32 = 10;
const S_AND_B32: u32 = 14;
const S_ANDN2_B32: u32 = 20;
const S_CMPK_GE_I32: u32 = 6;
const S_CMPK_GE_U32: u32 = 12;
const S_CMPK_LT_U32: u32 = 13;
const S_CMP_EQ_U32: u32 = 6;
const S_CMP_LT_U32: u32 = 10;
const S_ADD_U32: u32 = 0;
const S_BRANCH: u32 = 2;
const S_CBRANCH_SCC0: u32 = 4;
const S_CBRANCH_SCC1: u32 = 5;
const S_BARRIER: u32 = 0xBF8A_0000;
const S_CMP_LG_U32: u32 = 7;
const V_READFIRSTLANE_B32: u32 = 2;
const S_CBRANCH_EXECNZ: u32 = 9;
const V_MOV_B32: u32 = 1;
const V_CNDMASK_B32: u32 = 1;
const V_ADD_F32: u32 = 3;
const V_LSHLREV_B32: u32 = 0x1A;
const V_ADD_NC_U32: u32 = 0x25;
const V_CMP_GT_F32: u32 = 0x04;
const V_CMP_GT_U32: u32 = 0xC4;
const V_CMP_NE_U32: u32 = 0xC5;

/// `buffer_store_dword v1, v2, s[8:11], 0 offen`: each lane's v1 to dword `lane`
/// (v2 = v0 << 2 first).
const STORE_V1: [u32; 3] = [
    vop2(V_LSHLREV_B32, 2, k(2), 0),
    0x38 << 26 | 0x1C << 18 | 1 << 12,
    0x80 << 24 | 2 << 16 | 1 << 8 | 2,
];

/// A pixel program: s0 counts from 0 and the loop is left at its top once s0 >= 200; each
/// turn adds 1.0 to v0, which goes to target 0 after the loop. The back edge is the
/// body's `s_branch`, so a lane stopped at the cap leaves the program (it is discarded).
fn pixel_200() -> Vec<u32> {
    vec![
        sop1(S_MOV_B32, 0, k(0)),     // 0: s_mov_b32 s0, 0
        vop1(V_MOV_B32, 0, k(0)),     // 4: v_mov_b32 v0, 0
        sopk(S_CMPK_GE_U32, 0, 200),  // 8: s_cmpk_ge_u32 s0, 200
        sopp(S_CBRANCH_SCC1, 12, 28), // 12: s_cbranch_scc1 28
        vop2(V_ADD_F32, 0, ONE_F, 0), // 16: v_add_f32 v0, 1.0, v0
        sop2(S_ADD_I32, 0, 0, k(1)),  // 20: s_add_i32 s0, s0, 1
        sopp(S_BRANCH, 24, 8),        // 24: s_branch 8
        vop1(V_MOV_B32, 1, ONE_F),    // 28: v_mov_b32 v1, 1.0
        0xF800_180F,
        0x0100_0000, // 32: exp mrt0 v0, v0, v0, v1 done vm
        S_ENDPGM,
    ]
}

/// [`compute_two_exits`]: what the wave's count s22 moves by each turn, and the count at
/// which the whole wave leaves.
const COUNT_STEP: u32 = 2;
const COUNT_END: u32 = 12_000;
/// [`compute_two_exits`]: what a lane's sum v6 moves by each turn, and the shift that makes
/// the lane's own threshold of its thread index.
const SUM_STEP: u32 = 5;
const THRESHOLD_SHIFT: u32 = 11;

/// The turns thread `lane` of [`compute_two_exits`] runs on the console: to the wave's
/// bound, or to the first turn whose sum is past the lane's threshold if that comes first.
fn two_exit_turns(lane: u32) -> u32 {
    let by_count = COUNT_END.div_ceil(COUNT_STEP);
    let by_sum = (lane << THRESHOLD_SHIFT) / SUM_STEP + 1;
    by_count.min(by_sum)
}

/// After the compute preamble ([`compute`]): a wave loop with two ways out, written for
/// these tests. s22 counts [`COUNT_STEP`] a turn, and the whole wave leaves once it is at
/// [`COUNT_END`] (a scalar compare with an immediate, after the add). Each lane adds
/// [`SUM_STEP`] to its sum v6 a turn and leaves by itself once the sum is past its own
/// threshold v7, its thread index shifted by [`THRESHOLD_SHIFT`]. The lanes that stay go
/// back through a block that narrows EXEC to them (the back edge is that block's). The
/// sum is stored after the loop.
fn compute_two_exits() -> Vec<u32> {
    let at = 12; // the preamble's three dwords
    let mut w = vec![
        vop2(V_LSHLREV_B32, 7, k(THRESHOLD_SHIFT), 0), // v_lshlrev_b32 v7, 11, v0
        sop1(S_MOV_B32, 25, EXEC_LO),                  // s_mov_b32 s25, exec_lo
        vop1(V_MOV_B32, 6, k(0)),                      // v_mov_b32 v6, 0
        sop1(S_MOV_B32, 22, k(0)),                     // s_mov_b32 s22, 0
    ];
    let back = at + 4 * (w.len() as u32 + 1);
    let header = back + 4;
    w.push(sopp(S_BRANCH, back - 4, header)); // s_branch header
    w.extend([
        sop2(S_AND_B32, EXEC_LO, EXEC_LO, 24), // back: s_and_b32 exec_lo, exec_lo, s24
        vop2(V_ADD_NC_U32, 6, k(SUM_STEP), 6), // header: v_add_nc_u32 v6, 5, v6
        vopc(V_CMP_GT_U32, v(6), 7),           // v_cmp_gt_u32 vcc, v6, v7
        sop2(S_ADD_I32, 22, 22, k(COUNT_STEP)), // s_add_i32 s22, s22, 2
        sopk(S_CMPK_GE_I32, 22, COUNT_END),    // s_cmpk_ge_i32 s22, 12000
        // The lanes leaving: every lane at the wave's bound, else the ones VCC names.
        sop2(S_CSELECT_B32, 23, EXEC_LO, VCC_LO), // s_cselect_b32 s23, exec_lo, vcc_lo
        // The lanes staying, and SCC: some lane stays.
        sop2(S_ANDN2_B32, 24, EXEC_LO, 23), // s_andn2_b32 s24, exec_lo, s23
    ]);
    let branch = at + 4 * w.len() as u32;
    w.push(sopp(S_CBRANCH_SCC1, branch, back)); // s_cbranch_scc1 back
    w.push(sop1(S_MOV_B32, EXEC_LO, 25)); // s_mov_b32 exec_lo, s25
    w.push(vop1(V_MOV_B32, 1, v(6))); // v_mov_b32 v1, v6
    w.extend(STORE_V1);
    w
}

/// After the compute preamble: v5 is dword 0 of the UAV (zero) minus 1, and each lane
/// loops until its counter v6 equals v5: four billion turns on the console with that data.
/// Each turn adds 3 to v1, stored after the loop.
fn compute_from_buffer() -> Vec<u32> {
    let at = 12;
    let mut w = vec![
        0x38 << 26 | 0x0C << 18, // buffer_load_dword v5, off, s[8:11], 0
        0x80 << 24 | 2 << 16 | 5 << 8,
        vop2(V_ADD_NC_U32, 5, MINUS_ONE, 5), // v_add_nc_u32 v5, -1, v5
        vop1(V_MOV_B32, 6, k(0)),            // v_mov_b32 v6, 0
        vop1(V_MOV_B32, 1, k(0)),            // v_mov_b32 v1, 0
        sop1(S_MOV_B32, 15, EXEC_LO),        // s_mov_b32 s15, exec_lo
    ];
    let header = at + 4 * w.len() as u32;
    w.extend([
        vop2(V_ADD_NC_U32, 1, k(3), 1), // header: v_add_nc_u32 v1, 3, v1
        vop2(V_ADD_NC_U32, 6, k(1), 6), // v_add_nc_u32 v6, 1, v6
        vopc(V_CMP_NE_U32, v(6), 5),    // v_cmp_ne_u32 vcc, v6, v5
        sop2(S_AND_B32, EXEC_LO, EXEC_LO, VCC_LO), // s_and_b32 exec_lo, exec_lo, vcc_lo
    ]);
    let branch = at + 4 * w.len() as u32;
    w.push(sopp(S_CBRANCH_EXECNZ, branch, header)); // s_cbranch_execnz header
    w.push(sop1(S_MOV_B32, EXEC_LO, 15));
    w.extend(STORE_V1);
    w
}

/// After the compute preamble: a float counter v1 from 0.0 by 1.0 a turn while
/// 300.0 > v1, the loop narrowing EXEC to the lanes left.
fn compute_float_300() -> Vec<u32> {
    let at = 12;
    let mut w = vec![
        vop1(V_MOV_B32, 1, k(0)),     // v_mov_b32 v1, 0
        sop1(S_MOV_B32, 15, EXEC_LO), // s_mov_b32 s15, exec_lo
    ];
    let header = at + 4 * w.len() as u32;
    w.extend([
        vop2(V_ADD_F32, 1, ONE_F, 1), // header: v_add_f32 v1, 1.0, v1
        vopc(V_CMP_GT_F32, 0xFF, 1),  // v_cmp_gt_f32 vcc, 300.0, v1
        300f32.to_bits(),
        sop2(S_AND_B32, EXEC_LO, EXEC_LO, VCC_LO), // s_and_b32 exec_lo, exec_lo, vcc_lo
    ]);
    let branch = at + 4 * w.len() as u32;
    w.push(sopp(S_CBRANCH_EXECNZ, branch, header)); // s_cbranch_execnz header
    w.push(sop1(S_MOV_B32, EXEC_LO, 15));
    w.extend(STORE_V1);
    w
}

/// After the compute preamble: s12 counts to 5000, tested at the loop's top, and a turn
/// where s12 is even also adds s0 (a root constant, 0 here) to it: the body changes the
/// counter. Each turn adds 3 to v1, stored after the loop.
fn compute_counter_changed() -> Vec<u32> {
    let at = 12;
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)), // s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),  // v_mov_b32 v1, 0
    ];
    let header = at + 4 * w.len() as u32;
    let done = header + 4 * 8;
    let skip = header + 4 * 6;
    w.extend([
        sopk(S_CMPK_GE_U32, 12, 5000), // header: s_cmpk_ge_u32 s12, 5000
        sopp(S_CBRANCH_SCC1, header + 4, done), // s_cbranch_scc1 done
        vop2(V_ADD_NC_U32, 1, k(3), 1), // v_add_nc_u32 v1, 3, v1
        sop2(S_AND_B32, 17, 12, k(1)), // s_and_b32 s17, s12, 1
        sopp(S_CBRANCH_SCC1, header + 16, skip), // s_cbranch_scc1 skip
        sop2(S_ADD_I32, 12, 12, 0),    // s_add_i32 s12, s12, s0
        sop2(S_ADD_I32, 12, 12, k(1)), // skip: s_add_i32 s12, s12, 1
        sopp(S_BRANCH, header + 28, header), // s_branch header
    ]);
    assert_eq!(at + 4 * w.len() as u32, done);
    w.extend(STORE_V1);
    w
}

/// After the compute preamble: a block that ends both an outer and an inner loop's turns,
/// its branch's two arms back edges. The outer loop of header H runs until its count s12
/// equals root constant s0 (zero: four billion turns on the console); each of its turns
/// runs the inner loop of header I twice (s13 from 0 by 1 while below 2 at latch T, whose
/// taken arm is I's back edge and whose fall-through is H's). Each inner turn adds 1 to v1
/// and stores it.
fn compute_two_back_edges() -> Vec<u32> {
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)),      // 12: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),       // 16: v_mov_b32 v1, 0
        sopp(S_BRANCH, 20, 36),         // 20: s_branch H
        sop2(S_ADD_I32, 13, 13, k(1)),  // 24 T: s_add_i32 s13, s13, 1
        sopk(S_CMPK_LT_U32, 13, 2),     // 28: s_cmpk_lt_u32 s13, 2
        sopp(S_CBRANCH_SCC1, 32, 52),   // 32: s_cbranch_scc1 I (falls through to H)
        sop2(S_ADD_I32, 12, 12, k(1)),  // 36 H: s_add_i32 s12, s12, 1
        sopc(S_CMP_EQ_U32, 12, 0),      // 40: s_cmp_eq_u32 s12, s0
        sopp(S_CBRANCH_SCC1, 44, 72),   // 44: s_cbranch_scc1 X
        sop1(S_MOV_B32, 13, k(0)),      // 48: s_mov_b32 s13, 0
        vop2(V_ADD_NC_U32, 1, k(1), 1), // 52 I: v_add_nc_u32 v1, 1, v1
    ];
    w.extend(STORE_V1); // 56
    w.push(sopp(S_BRANCH, 68, 24)); // 68: s_branch T
    w.extend(STORE_V1); // 72 X
    w
}

/// After the compute preamble: a loop of header H whose latch T branches to H on both arms
/// (`s_cbranch_scc1` to the next instruction). H counts s12 and leaves once it equals root
/// constant s0 (zero: four billion turns on the console); each turn adds 1 to v1 and
/// stores it.
fn compute_both_arms_one_header() -> Vec<u32> {
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)),      // 12: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),       // 16: v_mov_b32 v1, 0
        sopp(S_BRANCH, 20, 32),         // 20: s_branch H
        sopc(S_CMP_LT_U32, 12, 0),      // 24 T: s_cmp_lt_u32 s12, s0
        sopp(S_CBRANCH_SCC1, 28, 32),   // 28: s_cbranch_scc1 H (falls through to H)
        vop2(V_ADD_NC_U32, 1, k(1), 1), // 32 H: v_add_nc_u32 v1, 1, v1
    ];
    w.extend(STORE_V1); // 36
    w.extend([
        sop2(S_ADD_I32, 12, 12, k(1)), // 48: s_add_i32 s12, s12, 1
        sopc(S_CMP_EQ_U32, 12, 0),     // 52: s_cmp_eq_u32 s12, s0
        sopp(S_CBRANCH_SCC1, 56, 64),  // 56: s_cbranch_scc1 X
        sopp(S_BRANCH, 60, 24),        // 60: s_branch T
    ]);
    w.extend(STORE_V1); // 64 X
    w
}

/// After the compute preamble: a `continue`. Header H adds 1 to v1 and to s12 and goes back
/// to itself while s12 < s16 = s0 - 1 (zero: four billion turns on the console); else C sets
/// v1 to 7, stores it, and jumps back to H. The console never gets to C, so stores nothing.
fn compute_continue() -> Vec<u32> {
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)),           // 12: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),            // 16: v_mov_b32 v1, 0
        sop2(S_ADD_U32, 16, 0, MINUS_ONE),   // 20: s_add_u32 s16, s0, -1
        vop2(V_ADD_NC_U32, 1, k(1), 1),      // 24 H: v_add_nc_u32 v1, 1, v1
        sop2(S_ADD_I32, 12, 12, k(1)),       // 28: s_add_i32 s12, s12, 1
        sopc(S_CMP_LT_U32, 12, 16),          // 32: s_cmp_lt_u32 s12, s16
        sopp(S_CBRANCH_SCC1, 36, 24),        // 36: s_cbranch_scc1 H
        vop1(V_MOV_B32, 1, k(7)),            // 40 C: v_mov_b32 v1, 7
    ];
    w.extend(STORE_V1); // 44
    w.push(sopp(S_BRANCH, 56, 24)); // 56: s_branch H
    w
}

/// After the compute preamble: an inner loop whose way out is an outer loop's back edge.
/// The outer loop of header O counts s12 and leaves once it equals root constant s0 (zero:
/// four billion turns on the console); each turn sets s13 to 0 and enters the inner loop
/// of header T, which goes back to O once s13 >= 5000 and otherwise through Inc, which adds
/// 1 to s13 and goes back to T while s13 < 0xF000. Past that, Y stores s13: the console's
/// s13 never passes 5000, so it never gets there. X stores 0.
fn compute_exit_through_outer_back_edge() -> Vec<u32> {
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)),       // 12: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),        // 16: v_mov_b32 v1, 0
        sop2(S_ADD_I32, 12, 12, k(1)),   // 20 O: s_add_i32 s12, s12, 1
        sopc(S_CMP_EQ_U32, 12, 0),       // 24: s_cmp_eq_u32 s12, s0
        sopp(S_CBRANCH_SCC1, 28, 76),    // 28: s_cbranch_scc1 X
        sop1(S_MOV_B32, 13, k(0)),       // 32: s_mov_b32 s13, 0
        sopk(S_CMPK_GE_U32, 13, 5000),   // 36 T: s_cmpk_ge_u32 s13, 5000
        sopp(S_CBRANCH_SCC1, 40, 20),    // 40: s_cbranch_scc1 O
        sop2(S_ADD_I32, 13, 13, k(1)),   // 44 Inc: s_add_i32 s13, s13, 1
        sopk(S_CMPK_LT_U32, 13, 0xF000), // 48: s_cmpk_lt_u32 s13, 0xf000
        sopp(S_CBRANCH_SCC1, 52, 36),    // 52: s_cbranch_scc1 T
        vop1(V_MOV_B32, 1, 13),          // 56 Y: v_mov_b32 v1, s13
    ];
    w.extend(STORE_V1); // 60
    w.push(S_ENDPGM); // 72
    w.extend(STORE_V1); // 76 X
    w
}

/// After the compute preamble: a nested pair. The inner loop (header I) adds 1 to v1 and
/// runs 5,000 turns by s13; the outer loop (header O) sets s13 to 0, runs the inner loop
/// and counts s12, going back while s12 < 2 (`proved`) or while s12 < s16 = s0 + 2 (a root
/// constant's sum, which the console's data decide). Both run two outer turns: v1 is 10,000.
fn compute_nested(proved: bool) -> Vec<u32> {
    let test = if proved {
        sopk(S_CMPK_LT_U32, 12, 2) // s_cmpk_lt_u32 s12, 2
    } else {
        sopc(S_CMP_LT_U32, 12, 16) // s_cmp_lt_u32 s12, s16
    };
    let mut w = vec![
        sop1(S_MOV_B32, 12, k(0)),      // 12: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),       // 16: v_mov_b32 v1, 0
        sop2(S_ADD_U32, 16, 0, k(2)),   // 20: s_add_u32 s16, s0, 2
        sop1(S_MOV_B32, 13, k(0)),      // 24 O: s_mov_b32 s13, 0
        vop2(V_ADD_NC_U32, 1, k(1), 1), // 28 I: v_add_nc_u32 v1, 1, v1
        sop2(S_ADD_I32, 13, 13, k(1)),  // 32: s_add_i32 s13, s13, 1
        sopk(S_CMPK_LT_U32, 13, 5000),  // 36: s_cmpk_lt_u32 s13, 5000
        sopp(S_CBRANCH_SCC1, 40, 28),   // 40: s_cbranch_scc1 I
        sop2(S_ADD_I32, 12, 12, k(1)),  // 44: s_add_i32 s12, s12, 1
        test,                           // 48
        sopp(S_CBRANCH_SCC1, 52, 24),   // 52: s_cbranch_scc1 O
    ];
    w.extend(STORE_V1); // 56
    w
}

/// After the compute preamble: each wave reads its first thread's index (s20) and goes
/// round header H while s12 < s16 (s16 = 4 billion for the first wave, 3 for the second),
/// by H's own `continue` and by C's branch back; then every thread waits at the title's
/// `s_barrier` and stores v1, the turns of H. Run as 64 threads (two console waves).
fn compute_cut_before_barrier() -> Vec<u32> {
    let mut w = vec![
        vop1(V_READFIRSTLANE_B32, 20, v(0)),       // 12: v_readfirstlane_b32 s20, v0
        sopc(S_CMP_EQ_U32, 20, k(0)),              // 16: s_cmp_eq_u32 s20, 0
        sop2(S_CSELECT_B32, 16, MINUS_ONE, k(3)),  // 20: s_cselect_b32 s16, -1, 3
        sop1(S_MOV_B32, 12, k(0)),                 // 24: s_mov_b32 s12, 0
        vop1(V_MOV_B32, 1, k(0)),                  // 28: v_mov_b32 v1, 0
        vop2(V_ADD_NC_U32, 1, k(1), 1),            // 32 H: v_add_nc_u32 v1, 1, v1
        sop2(S_ADD_I32, 12, 12, k(1)),             // 36: s_add_i32 s12, s12, 1
        sopc(S_CMP_LT_U32, 12, 16),                // 40: s_cmp_lt_u32 s12, s16
        sopp(S_CBRANCH_SCC1, 44, 32),              // 44: s_cbranch_scc1 H (continue)
        sopc(S_CMP_LT_U32, 12, 16),                // 48 C: s_cmp_lt_u32 s12, s16
        sopp(S_CBRANCH_SCC1, 52, 32),              // 52: s_cbranch_scc1 H
        S_BARRIER,                                 // 56 X: s_barrier
    ];
    w.extend(STORE_V1); // 60
    w
}

/// After the compute preamble: a latch T branching to header H on both arms by
/// `s_cbranch_scc0` (a negated condition), with VCC made from SCC on the way
/// (`s_cselect_b32 vcc_lo, exec_lo, 0`: every lane's bit where SCC holds, none where not),
/// so the two edges bring VCC in with different lane bits. SCC is s0 != 0: zero, so every
/// turn takes the taken edge with VCC empty. H stores v1 = VCC ? 7 : 0 (7 only on the first
/// turn, for the threads below 16) and goes back to T: four billion turns on the console.
fn compute_negated_both_arms() -> Vec<u32> {
    let mut w = vec![
        vopc(V_CMP_GT_U32, k(16), 0),            // 12: v_cmp_gt_u32 vcc, 16, v0
        vop1(V_MOV_B32, 4, k(7)),                // 16: v_mov_b32 v4, 7
        sopp(S_BRANCH, 20, 36),                  // 20: s_branch H
        sopc(S_CMP_LG_U32, 0, k(0)),             // 24 T: s_cmp_lg_u32 s0, 0
        sop2(S_CSELECT_B32, VCC_LO, EXEC_LO, k(0)), // 28: s_cselect_b32 vcc_lo, exec_lo, 0
        sopp(S_CBRANCH_SCC0, 32, 36),            // 32: s_cbranch_scc0 H (falls through to H)
        vop2(V_CNDMASK_B32, 1, k(0), 4),         // 36 H: v_cndmask_b32 v1, 0, v4, vcc
    ];
    w.extend(STORE_V1); // 40
    w.push(sopp(S_BRANCH, 52, 24)); // 52: s_branch T
    w
}

/// A wave32 compute program of 32 threads (the thread index in v0) with 4 root constants
/// in s0..s3 and a raw UAV's descriptor loaded to s[8:11] from the table at s4, then `body`.
fn compute(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
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
            rsrc2: 5 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn pixel(words: &[u32]) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The loops of `native` with their bounds.
fn loops(native: &NativeProgram) -> Vec<Bound> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    bounds::loops(&p, &st)
        .into_iter()
        .map(|l| l.bound)
        .collect()
}

fn validated(shader: HostShader) -> HostShader {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// `native` lowered, with its resources.
fn lowered(native: &NativeProgram) -> Result<(lower::Lowered, resource::Resources), ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    Ok((lowered, res))
}

fn compute_shader(native: &NativeProgram) -> HostShader {
    let (lowered, res) = lowered(native).expect("lower");
    validated(HostShader {
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

/// Each device's UAV of 32 dwords after one group of `native`, its root constants zero.
fn dispatched(native: &NativeProgram) -> Vec<(&'static str, Vec<u32>)> {
    let shader = compute_shader(native);
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
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    assert!(!out.is_empty(), "no device ran the program");
    out
}

fn target() -> SigElement {
    SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }
}

/// POSITION to SV_Position.
fn vertex_shader() -> HostShader {
    let mut s = Shader::new(false);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    for col in 0..4u8 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), undef];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let out = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
    }
    let element = |name: &str, kind, interp| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    s.inputs.push(element("POSITION", 0, 0));
    s.outputs.push(element("SV_Position", 3, 4));
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    }
}

/// Each device's 4x4 target after a full-screen triangle with `native` as its pixel
/// program (cleared to zero: a discarded pixel stays so).
fn drawn(native: &NativeProgram) -> Vec<(&'static str, Vec<[f32; 4]>)> {
    let isg1 = container::signature(&[], true);
    let osg1 = container::signature(&[target()], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).expect("container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let ps = validated(super::emit(&c, native, &p, &st, &io, &res).expect("emit"));
    let vs = vertex_shader();
    let vertices = [
        -1.0, -1.0, 0.5, 1.0, //
        -1.0, 3.0, 0.5, 1.0, //
        3.0, -1.0, 0.5, 1.0,
    ];
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
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: graphics pipeline failed: {e}"));
        let texels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
        out.push((label, texels));
    }
    assert!(!out.is_empty(), "no device ran the program");
    out
}

#[test]
fn counters_from_constants_to_a_constant_are_proved() {
    // 200 turns of the body, and the test once more: 201 turns of the header.
    assert_eq!(loops(&pixel(&pixel_200())), [Bound::Turns(201)]);
    // The count is tested after its add: COUNT_STEP on the first turn, COUNT_END on the
    // last. The lanes' own way out, by a sum against a threshold computed per lane, proves
    // nothing and takes nothing from the proof.
    assert_eq!(
        loops(&compute(&compute_two_exits())),
        [Bound::Turns(COUNT_END.div_ceil(COUNT_STEP))]
    );
    assert_eq!(COUNT_END.div_ceil(COUNT_STEP), 6000);
}

#[test]
fn a_count_read_from_a_buffer_or_changed_in_the_body_is_not_proved() {
    assert_eq!(
        loops(&compute(&compute_from_buffer())),
        [Bound::Data(bounds::Source::Memory)]
    );
    assert_eq!(
        loops(&compute(&compute_counter_changed())),
        [Bound::Computed]
    );
}

/// A float counter is counted for the census (its f32 run), never a proof.
#[test]
fn a_float_counter_is_counted_but_not_proved() {
    let native = compute(&compute_float_300());
    assert_eq!(loops(&native), [Bound::Float(300)]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    assert!(bounds::caps(&p, &st, true).is_empty());
}

/// The pixel cap is 128 turns: a 200-turn loop ran into it and its pixels were discarded.
#[test]
fn a_pixel_loop_of_200_turns_writes_its_result() {
    for (label, texels) in drawn(&pixel(&pixel_200())) {
        assert!(
            texels.iter().all(|t| *t == [200.0, 200.0, 200.0, 1.0]),
            "{label}: {:?}",
            &texels[..2]
        );
    }
}

/// The compute cap is 4,096 turns: lanes still in the loop then left the program and their
/// store was skipped. Every lane stores its sum: the lanes that leave by their own threshold
/// under the cap (threads 0 to 9) and past it (10 to 14, the first of them one turn past),
/// and the lanes the wave's bound sends out at 6,000 turns (15 to 31).
#[test]
fn a_compute_loop_of_6000_turns_stores_after_it() {
    let turns: Vec<u32> = (0..32).map(two_exit_turns).collect();
    // The three kinds of lane are all there, as the comment says.
    assert_eq!((turns[9], turns[10], turns[14], turns[15], turns[31]), (3687, 4097, 5735, 6000, 6000));
    let want: Vec<u32> = turns.iter().map(|t| SUM_STEP * t).collect();
    for (label, got) in dispatched(&compute(&compute_two_exits())) {
        assert_eq!(got, want, "{label}");
    }
}

/// With zeros for data the console runs this loop four billion times; the translation
/// still leaves it at the cap: 4,096 turns of 3, and the store after the loop.
#[test]
fn a_loop_bounded_by_a_buffer_keeps_the_cap() {
    for (label, got) in dispatched(&compute(&compute_from_buffer())) {
        assert_eq!(got, vec![3 * 4096; 32], "{label}");
    }
}

/// The console runs this loop 5,000 times (s0 is 0), but the translation cannot prove
/// that: lanes still reach the cap at 4,096 turns, leave the program and store nothing.
#[test]
fn a_loop_whose_body_changes_its_counter_keeps_the_cap() {
    for (label, got) in dispatched(&compute(&compute_counter_changed())) {
        assert_eq!(got, vec![0; 32], "{label}");
    }
}

/// A branch whose two arms are both back edges: the lowering counted the taken arm only, and
/// the other went back unchecked (the outer loop here ran on the host as long as the
/// console's four billion turns; never dispatched before). Each arm now checks its own
/// count. The inner loop's arm, at its cap, would go on down the outer's; the outer's arm,
/// whose other arm is in its loop, leaves the program at its cap: 4,096 outer turns of two
/// inner ones, the last store 8,192. With one header on both arms (whose counter missed an
/// entry: invalid DXIL) the lane leaves at the cap, 4,096 turns.
#[test]
fn a_branch_with_two_back_edges_is_taken_and_bounded() {
    let two = compute(&compute_two_back_edges());
    // The outer loop is the data's; the inner loop's way out is the outer back edge, whose
    // cap leaves the program.
    assert_eq!(
        loops(&two),
        [Bound::Data(bounds::Source::Input), Bound::Turns(2)]
    );
    for (label, got) in dispatched(&two) {
        assert_eq!(got, vec![8192; 32], "{label}");
    }
    for (label, got) in dispatched(&compute(&compute_both_arms_one_header())) {
        assert_eq!(got, vec![4096; 32], "{label}");
    }
}

/// Where a loop's way out is another loop's back edge, that loop's cap turned the way out
/// into another turn: once the outer cap (4,096 turns) acted, the lane went on into the
/// inner loop (Inc) past the 5,001 turns T's exit proved, and the inner loop's own cap
/// stopped it (Y stored s13 = 5,001): a cap acting on a loop called proved. The outer cap
/// now leaves the program there (Inc stays in the outer loop), so the proof stands and no
/// lane gets to Y or X: nothing is stored.
#[test]
fn an_outer_cap_cuts_the_lane_out_of_a_proved_inner_loop() {
    let native = compute(&compute_exit_through_outer_back_edge());
    for (label, got) in dispatched(&native) {
        assert_eq!(got, vec![0; 32], "{label}");
    }
    assert_eq!(
        loops(&native),
        [Bound::Data(bounds::Source::Input), Bound::Turns(5001)]
    );
}

/// A lane at the cap of a `continue` went down the branch's other arm, still in the loop:
/// C, which the console never reaches, stored 7. It now leaves the program there.
#[test]
fn a_lane_at_the_cap_of_a_continue_leaves_the_loop() {
    for (label, got) in dispatched(&compute(&compute_continue())) {
        assert_eq!(got, vec![0; 32], "{label}");
    }
}

/// A proved inner loop past the cap (5,000 turns) inside a proved outer loop (two turns),
/// and inside one whose bound is the console's data (two turns with these): the inner loop
/// runs its count on every outer turn.
#[test]
fn a_proved_loop_nested_in_a_proved_or_unproved_loop_runs_its_count() {
    assert_eq!(
        loops(&compute(&compute_nested(true))),
        [Bound::Turns(2), Bound::Turns(5000)]
    );
    assert_eq!(
        loops(&compute(&compute_nested(false))),
        [Bound::Data(bounds::Source::Input), Bound::Turns(5000)]
    );
    for proved in [true, false] {
        for (label, got) in dispatched(&compute(&compute_nested(proved))) {
            assert_eq!(got, vec![10_000; 32], "{label}, outer proved: {proved}");
        }
    }
}

/// The first wave goes round H to the cap, the others leave after 3 turns and wait at the
/// title's barrier. At the cap of H's `continue` (its other arm, C, in the loop) the lane
/// left the program (stored nothing), and the other waves waited at the barrier for a wave
/// that had gone (NVIDIA's wait is for every thread: the card is reset). In a program with a group
/// barrier the lane goes down the other arm as before, and C's own cap sends it out of the
/// loop to the barrier: 4,096 turns, then 3. Run on WARP only, with a timeout.
#[test]
fn a_lane_at_a_cap_before_a_barrier_does_not_leave_the_program() {
    let mut native = compute(&compute_cut_before_barrier());
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1,
        threads: [64, 1, 1],
    });
    let shader = compute_shader(&native);
    let (send, got) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(true) else {
            let _ = send.send(None);
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        let _ = send.send(Some(words[0].clone()));
    });
    match got.recv_timeout(std::time::Duration::from_secs(30)) {
        // (the host's waves: WARP's have 4 lanes, so the first is threads 0..3)
        Ok(Some(words)) => {
            assert!(words.iter().all(|&w| w == 4096 || w == 3), "{words:?}");
            assert_eq!((words[0], words[63]), (4096, 3), "{words:?}");
        }
        Ok(None) => eprintln!("no software adapter; skipped"),
        Err(_) => panic!("the dispatch did not finish in 30 s: a wave waits at the barrier"),
    }
}

/// A negated branch whose arms both go back to one header: the lowering swaps the arms, and
/// the counted form keyed the joins' edges in the swapped order, so H read the VCC of the
/// edge not taken (every lane's bit): 7 in every thread. Keyed by the console's edges, H
/// reads the empty VCC of the edge taken and the last store, at the cap, is 0.
#[test]
fn a_negated_branch_back_on_both_arms_joins_each_edges_mask() {
    for (label, got) in dispatched(&compute(&compute_negated_both_arms())) {
        assert_eq!(got, vec![0; 32], "{label}");
    }
}

/// The fixtures are the instructions their comments name.
#[test]
fn the_fixtures_decode_as_written() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap()
            .iter()
            .map(|i| i.text(false))
            .collect()
    };
    let pixel = text(&pixel_200());
    assert_eq!(pixel[2], "s_cmpk_ge_u32 s0, 0x00c8", "{pixel:#?}");
    assert_eq!(pixel[6], "s_branch label_0008", "{pixel:#?}");
    let two_exits = text(
        &compute(&compute_two_exits())
            .code
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    eprintln!("{two_exits:#?}");
    // The wave's bound, the lane's own threshold, the two made one mask of the lanes
    // leaving, and the block on the back edge that narrows EXEC.
    for line in [
        "s_cmpk_ge_i32 s22, 0x2ee0",
        "v_cmp_gt_u32 vcc_lo, v6, v7",
        "s_cselect_b32 s23, exec_lo, vcc_lo",
        "s_andn2_b32 s24, exec_lo, s23",
        "s_cbranch_scc1 label_0020",
        "s_and_b32 exec_lo, exec_lo, s24",
    ] {
        assert!(two_exits.contains(&line.to_owned()), "{line}: {two_exits:#?}");
    }
    assert_eq!(two_exits[7], "s_and_b32 exec_lo, exec_lo, s24", "{two_exits:#?}");
    eprintln!(
        "{:#?}",
        text(
            &compute(&compute_from_buffer())
                .code
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        )
    );
    eprintln!(
        "{:#?}",
        text(
            &compute(&compute_counter_changed())
                .code
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        )
    );
}
