//! A sampler control flow chooses among immediate ones (`sampler_choice.rs`): synthetic
//! GFX10 programs in the console compiler's form of `flag == 1.0 ? clamp : wrap` (a word
//! of a constant buffer compared with 1.0 by `s_cmp_lg_u32`, `s_cbranch_scc1` around the
//! moves of the one sampler's first word and of the other's, the remaining words set after
//! the join, the sample in the join's block), lowered to DXIL, validated by `dxil.dll`
//! when present, and run on the host GPU and WARP: each run samples with the sampler its
//! branch chose. Every other form is refused, each by its own reason.
//!
//! The texture is two texels wide, red 1.0 then 2.0, sampled with point filters at
//! u = 1.25: wrapped that is the first texel's centre, clamped the second texel.

use super::bitcode::{Attr, Term};
use super::container::SigElement;
use super::host::{Dispatch, Draw, Host, Root, View};
use super::shader::Shader;
use super::signature_order_tests::{emitted_inputs, float, interp};
use super::{HostShader, StaticSampler, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo,
    Usage, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// The sampler's first word clamped in x, y and z; wrapped it is 0.
const CLAMP: u32 = 0x92;
/// Point filters and no mip filter: the sampler where the flag is 1.0, then the other.
const POINT: [[u32; 4]; 2] = [[CLAMP, 0, 0, 0], [0; 4]];
/// Point filters and a point mip filter over every level.
const POINT_MIP: [[u32; 4]; 2] = [[CLAMP, 0x00FF_F000, 0x0400_0000, 0], [0, 0x00FF_F000, 0x0400_0000, 0]];
const TEXELS: [[f32; 4]; 2] = [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
/// What the sample reads with each sampler.
const WRAPPED: f32 = 1.0;
const CLAMPED: f32 = 2.0;
/// The flag that chooses the clamped sampler.
const SET: u32 = 0x3F80_0000;

const SCC1: u32 = 0xBF85_0000; // s_cbranch_scc1
const VCCZ: u32 = 0xBF86_0000; // s_cbranch_vccz
const SAMPLE: [u32; 2] = [0xF080_0F08, 0x0086_0C02]; // image_sample v[12:15], v[2:3], s[24:31], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
const SAMPLE_LZ: [u32; 2] = [0xF09C_0F08, 0x0086_0C02]; // image_sample_lz, the same operands

/// `s_mov_b32 s<reg>, value` as the console compiler writes it: 0 inline, a small number by
/// `s_movk_i32`, anything else as a literal.
fn mov(reg: u32, value: u32) -> Vec<u32> {
    match value {
        0 => vec![0xBE80_0380 | reg << 16],
        1..=0x7FFF => vec![0xB000_0000 | reg << 16 | value],
        _ => vec![0xBE80_03FF | reg << 16, value],
    }
}

/// The choice of `pair` into s[16:19]: `branch` to the second sampler's moves around the
/// first's (the words in which they differ), the words they share set after the join.
fn choice(branch: u32, pair: [[u32; 4]; 2]) -> Vec<u32> {
    let (mut first, mut second, mut shared) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..4 {
        let reg = 16 + k as u32;
        if pair[0][k] == pair[1][k] {
            shared.extend(mov(reg, pair[0][k]));
        } else {
            first.extend(mov(reg, pair[0][k]));
            second.extend(mov(reg, pair[1][k]));
        }
    }
    let mut words = vec![branch | (first.len() as u32 + 1)];
    words.extend(first);
    words.push(0xBF82_0000 | second.len() as u32); // s_branch -> join
    words.extend(second);
    words.extend(shared);
    words
}

fn table(user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    }
}

/// A flag that is no word of a constant buffer: one root constant in s2.
fn root_constant() -> RootParam {
    RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 2,
        slot: 0,
    }
}

fn views(kind: ResourceType, table_offset_dwords: u32, dwords: u8, slot: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot,
    }
}

/// A pixel program's sample and what stands around it.
#[derive(Clone, Copy, PartialEq)]
enum Pixel {
    /// Whole-quad mode, then `image_sample` at (1.25, 0.5) in every pixel.
    Plain,
    /// [`Pixel::Plain`] without whole-quad mode: the console's helper lanes then hold no
    /// coordinates, and its derivatives are of what those lanes held before.
    Unwhole,
    /// A shape of the test's own: whole-quad mode, then `image_sample` at the interpolated
    /// TEXCOORD0.xy in a section only the lanes with x above 0.5 run (`v_cmpx_lt_f32` and
    /// `s_cbranch_execz`), the choice inside it; the other lanes keep [`SENTINEL`].
    Section,
    /// [`Pixel::Section`] at coordinates the program computes before the section, where
    /// every lane of the quad runs (`v_mul_f32` by 2.0 and a move): the console's four
    /// lanes hold them, and they are no inputs of the shader.
    Outside,
    /// [`Pixel::Outside`] with the factor read from a VGPR the program wrote before it
    /// turned whole-quad mode on (`v_mov_b32` of 1.0, which the console's helper lanes did
    /// not run): the coordinate is written under the mode, from a value that is not.
    Before,
    /// [`Pixel::Section`] at a coordinate the program computes inside the section
    /// (`v_add_f32`): the lanes of the quad that sit the section out never compute it.
    Computed,
    /// Whole-quad mode, the interpolated TEXCOORD0.xy, whole-quad mode ended (`s_and_b32`
    /// of EXEC with the EXEC the program started with), and then the choice and
    /// `image_sample` at those coordinates in every pixel: the helper lanes hold the
    /// coordinates and do not run the sample.
    Ended,
    /// [`Pixel::Ended`] with U scaled, while whole-quad mode is on, by `exec_hi` of this
    /// wave32 program, into which a `v_readfirstlane_b32` of U was moved (`s_mov_b32
    /// exec_hi, s23`, `v_mul_f32 v2, exec_hi, v2`): EXEC as a number (Neon's seventh review).
    ExecHigh,
    /// [`Pixel::Ended`] with U scaled, while whole-quad mode is on, by a scalar
    /// `v_readfirstlane_b32` read from it: the lowering reads the first lane with a wave
    /// operation that leaves the helper lanes out.
    FirstLane,
    /// [`Pixel::Ended`] with U chosen, while whole-quad mode is on, by a branch on a scalar
    /// `v_readfirstlane_b32` read from it (`s_cmp_lg_u32` with 0, `s_cbranch_scc1`), one
    /// literal on each arm, joined before the mode ends: the helper lanes' first-lane read
    /// is undefined, and so is the arm they take.
    Diamond,
    /// [`Pixel::Ended`] with U swapped between the neighbours of each quad by
    /// `ds_swizzle_b32` (quad_perm 1, 0, 3, 2): the lowering reads the other lane with a
    /// wave operation that leaves the helper lanes out, within a quad too.
    Swizzle,
    /// [`Pixel::Ended`] with U set to 1.0, while whole-quad mode is on, in the lanes where it
    /// is above 0.5: `v_cmp_lt_f32`, `s_and_saveexec_b32` and `s_cbranch_scc0` on its SCC
    /// (the wave's "some lane"), EXEC put back at the join. Each lane goes by its own bit,
    /// helper lanes too, on the console and as lowered: no wave operation chooses.
    LaneArm,
    /// [`Pixel::Section`] with TEXCOORD0 marked flat in the stage data, its interpolation
    /// pair (`v_interp_p1_f32`, `v_interp_p2_f32`) as before.
    Flat,
    /// [`Pixel::Outside`] with the factor a word of the constant buffer (word 6, 2.0 in the
    /// draws), loaded before the section by `s_buffer_load_dword`.
    Scaled,
    /// [`Pixel::Ended`] with U set to 1.0, while whole-quad mode is on, behind
    /// `s_cbranch_vccnz` on a compare of U with its first lane's (`v_readfirstlane_b32`),
    /// EXEC narrowed to the lanes outside VCC first and put back in the arm and at the join.
    /// The console tests the whole of VCC; the lowering each lane's `VCC & EXEC`, which is
    /// clear in every lane (Neon's first hole).
    VccCancel,
    /// [`Pixel::Ended`] with U a scalar's copy: 0.5, set to 1.0 in an arm the lanes above 0.5
    /// enter by `s_and_saveexec_b32` and `s_cbranch_scc0`, copied after EXEC is put back.
    /// The console's arm runs for the wave if any lane enters, the scalar is the wave's;
    /// each lane's own as lowered (Neon's second hole).
    ArmScalar,
    /// [`Pixel::Outside`] with U doubled only where it is below 2.0 (pixels 0 to 3), by
    /// `v_cmp_gt_f32` and `v_cndmask_b32` under whole-quad mode: each lane selects by its
    /// own compare. Quads of doubled U take level 2, the others level 1.
    Selected,
    /// [`Pixel::Selected`] with the mask a constant (`s_mov_b32 vcc_lo, -1`), no compare.
    SelectedConstant,
    /// [`Pixel::Selected`] with the compare of U against `exec_hi` of this wave32 program,
    /// into which a `v_readfirstlane_b32` of U was moved (`v_cmp_gt_f32 vcc_lo, exec_hi,
    /// v2`): EXEC as a number in the compare (Neon's seventh review).
    SelectedExecHigh,
    /// [`Pixel::Selected`] with the compare's mask also the selected value of the lanes
    /// outside it (`v_cndmask_b32 v0, vcc_lo, v0, vcc_lo`): a mask as a number.
    SelectedMask,
    /// [`Pixel::Outside`] with U accumulated, under whole-quad mode, into a VGPR the program
    /// wrote before it turned the mode on (`v_mov_b32 v5, 1.0`, then `v_fmac_f32 v5, 2.0,
    /// v2`): the accumulator is an operand the console's helper lanes did not compute
    /// (Neon's sixth review).
    Accumulated,
    /// [`Pixel::Ended`] behind a branch, while whole-quad mode is on, on each lane's compare
    /// of U with 0.5 (`v_cmp_lt_f32`, `s_cbranch_vccz`) whose other way is a loop with no way
    /// out (`s_branch` to itself), at (1.25, 0.5) set past the loop: the lanes that enter it
    /// never sample (Neon's sixth review).
    Endless,
    /// [`Pixel::Ended`] after an if of the lanes above 0.5 (`s_and_saveexec_b32 s25`,
    /// `s_cbranch_execz`) whose both ways put EXEC back to s25 (`s_mov_b32`): the join's
    /// EXEC is a phi of two copies of one mask, which is that mask.
    Rejoined,
    /// [`Pixel::Rejoined`] with the skipping way putting nothing back: the join's EXEC is
    /// a phi of the compare's section and the copy of s25, two masks.
    TwoMasks,
    /// [`Pixel::Rejoined`] with the other way's EXEC `s_wqm_b32` of the program's own again
    /// (s22): equal to s25 as it runs, but another value, not a copy of it.
    OtherCopy,
    /// [`Pixel::Section`] entered by `v_cmp_lt_f32` and `s_and_saveexec_b32`, with
    /// `s_andn2_b32 s24` of the saved EXEC and VCC and `s_cbranch_scc1` past the sample
    /// inside it: the console skips the sample if any lane is outside VCC; the lanes'
    /// condition folds to a constant (Neon's third hole).
    Folded,
    /// [`Pixel::Section`] entered by `s_and_saveexec_b32` and `s_cbranch_execz` to a block
    /// that puts EXEC to the lanes outside VCC, the other way jumping over it: the choice
    /// and the sample after both, under an EXEC joined from the two (Neon's diamond).
    Joined,
    /// [`Pixel::Ended`] with no choice: the words of the first sampler of the pair moved in
    /// alone, which any program names an immediate sampler with, and the implicit sample.
    Single,
    /// These two words instead of `image_sample`, at (1.25, 0.5).
    Other([u32; 2]),
}

/// What a lane that sits out the section of [`Pixel::Section`] exports.
const SENTINEL: [f32; 4] = [7.0, 0.0, 0.0, 0.0];

/// A pixel program: the texture at entry 0 of the table in s0, sampled with implicit
/// derivatives with the sampler of `pair` that word 5 of the constant buffer at entry 1
/// chooses (loaded with its neighbours by `s_buffer_load_dwordx4`), to target 0.
fn pixel_shader(shape: Pixel, pair: [[u32; 4]; 2]) -> Result<HostShader, ShaderError> {
    let mut words = vec![
        0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
        0xF40C_0200,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
        0xF408_0100,
        0xFA00_0020, // s_load_dwordx4 s[4:7], s[0:1], 0x20
    ];
    let mut chosen = vec![
        0xF428_0802,
        0xFA00_0010, // s_buffer_load_dwordx4 s[32:35], s[4:7], 0x10
        0xBF07_F221, // s_cmp_lg_u32 s33, 1.0
    ];
    if shape == Pixel::Single {
        for (k, word) in pair[0].into_iter().enumerate() {
            chosen.extend(mov(16 + k as u32, word));
        }
    } else {
        chosen.extend(choice(SCC1, pair));
    }
    let selected = matches!(
        shape,
        Pixel::Selected | Pixel::SelectedConstant | Pixel::SelectedMask | Pixel::SelectedExecHigh
    );
    let outside = selected
        || matches!(
            shape,
            Pixel::Outside | Pixel::Before | Pixel::Scaled | Pixel::Accumulated
        );
    let ended = matches!(
        shape,
        Pixel::Ended
            | Pixel::ExecHigh
            | Pixel::FirstLane
            | Pixel::Diamond
            | Pixel::Swizzle
            | Pixel::LaneArm
            | Pixel::VccCancel
            | Pixel::ArmScalar
            | Pixel::Rejoined
            | Pixel::TwoMasks
            | Pixel::OtherCopy
            | Pixel::Endless
            | Pixel::Single
    );
    let section = outside
        || ended
        || matches!(
            shape,
            Pixel::Section | Pixel::Computed | Pixel::Flat | Pixel::Folded | Pixel::Joined
        );
    if matches!(shape, Pixel::Before | Pixel::Accumulated) {
        words.push(0x7E0A_02F2); // v_mov_b32 v5, 1.0
    }
    if section {
        words.extend([
            0xBE96_037E, // s_mov_b32 s22, exec_lo
            0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
        ]);
        words.extend(interp(2, 0, 0)); // v2 = TEXCOORD0.x
        words.extend(interp(3, 0, 1)); // v3 = TEXCOORD0.y
    }
    if shape == Pixel::FirstLane {
        words.extend([
            0x7E2E_0502, // v_readfirstlane_b32 s23, v2
            0x1004_0417, // v_mul_f32 v2, s23, v2
        ]);
    }
    if matches!(shape, Pixel::ExecHigh | Pixel::SelectedExecHigh) {
        words.extend([
            0x7E2E_0502, // v_readfirstlane_b32 s23, v2
            0xBEFF_0317, // s_mov_b32 exec_hi, s23
        ]);
    }
    if shape == Pixel::ExecHigh {
        words.push(0x1004_047F); // v_mul_f32 v2, exec_hi, v2
    }
    if shape == Pixel::Diamond {
        words.extend([
            0x7E2E_0502,  // v_readfirstlane_b32 s23, v2
            0xBF07_8017,  // s_cmp_lg_u32 s23, 0
            SCC1 | 2,     // s_cbranch_scc1 -> the second arm
            0x7E04_02F0,  // v_mov_b32 v2, 0.5
            0xBF82_0001,  // s_branch -> join
            0x7E04_02F2,  // v_mov_b32 v2, 1.0
        ]);
    }
    if shape == Pixel::Swizzle {
        words.extend([
            0xD8D4_80B1,
            0x0200_0002, // ds_swizzle_b32 v2, v2 offset:swizzle(QUAD_PERM, 1, 0, 3, 2)
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ]);
    }
    if shape == Pixel::LaneArm {
        words.extend([
            0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
            0xBE99_3C6A, // s_and_saveexec_b32 s25, vcc_lo
            0xBF84_0001, // s_cbranch_scc0 -> join
            0x7E04_02F2, // v_mov_b32 v2, 1.0
            0xBEFE_0319, // s_mov_b32 exec_lo, s25
        ]);
    }
    if shape == Pixel::VccCancel {
        words.extend([
            0x7E2E_0502, // v_readfirstlane_b32 s23, v2
            0x7C02_0417, // v_cmp_lt_f32 vcc_lo, s23, v2
            0xBE99_037E, // s_mov_b32 s25, exec_lo
            0x8A7E_6A7E, // s_andn2_b32 exec_lo, exec_lo, vcc_lo
            0xBF87_0002, // s_cbranch_vccnz -> join
            0xBEFE_0319, // s_mov_b32 exec_lo, s25
            0x7E04_02F2, // v_mov_b32 v2, 1.0
            0xBEFE_0319, // join: s_mov_b32 exec_lo, s25
        ]);
    }
    if shape == Pixel::ArmScalar {
        words.extend([
            0xBE9A_03F0, // s_mov_b32 s26, 0.5
            0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
            0xBE99_3C6A, // s_and_saveexec_b32 s25, vcc_lo
            0xBF84_0001, // s_cbranch_scc0 -> join
            0xBE9A_03F2, // s_mov_b32 s26, 1.0
            0xBEFE_0319, // join: s_mov_b32 exec_lo, s25
            0x7E04_021A, // v_mov_b32 v2, s26
        ]);
    }
    if matches!(shape, Pixel::Rejoined | Pixel::OtherCopy) {
        words.extend([
            0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
            0xBE99_3C6A, // s_and_saveexec_b32 s25, vcc_lo
            0xBF88_0003, // s_cbranch_execz -> the other way
            0x7E0A_0280, // v_mov_b32 v5, 0
            0xBEFE_0319, // s_mov_b32 exec_lo, s25
            0xBF82_0001, // s_branch -> join
            // the other way: s_mov_b32 exec_lo, s25 (or s_wqm_b32 exec_lo, s22)
            if shape == Pixel::Rejoined { 0xBEFE_0319 } else { 0xBEFE_0916 },
        ]);
    }
    if shape == Pixel::TwoMasks {
        words.extend([
            0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
            0xBE99_3C6A, // s_and_saveexec_b32 s25, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> join
            0x7E0A_0280, // v_mov_b32 v5, 0
            0xBEFE_0319, // s_mov_b32 exec_lo, s25
        ]);
    }
    if shape == Pixel::Endless {
        words.extend([
            0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
            VCCZ | 1,    // s_cbranch_vccz -> past the loop
            0xBF82_FFFF, // s_branch -> itself
            0x7E04_02FF,
            1.25f32.to_bits(), // past the loop: v_mov_b32 v2, 1.25
            0x7E06_02F0,       // v_mov_b32 v3, 0.5
        ]);
    }
    if shape == Pixel::Scaled {
        words.extend([
            0xF420_0902,
            0xFA00_0018, // s_buffer_load_dword s36, s[4:7], 0x18
        ]);
    }
    if ended {
        words.push(0x877E_167E); // s_and_b32 exec_lo, exec_lo, s22
        words.extend(chosen);
        words.extend([
            0xF080_0F08,
            0x0082_0402, // image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
        ]);
    } else if section {
        if selected {
            words.extend([
                // v_cmp_gt_f32 vcc_lo, 2.0, v2 (s_mov_b32 vcc_lo, -1; v_cmp_gt_f32 vcc_lo,
                // exec_hi, v2)
                match shape {
                    Pixel::SelectedConstant => 0xBEEA_03C1,
                    Pixel::SelectedExecHigh => 0x7C08_047F,
                    _ => 0x7C08_04F4,
                },
                0x1000_04F4, // v_mul_f32 v0, 2.0, v2
                // v_cndmask_b32 v0, v2, v0, vcc_lo (v0, vcc_lo, v0, vcc_lo)
                if shape == Pixel::SelectedMask { 0x0200_006A } else { 0x0200_0102 },
                0x7E02_0303, // v_mov_b32 v1, v3
            ]);
        } else if shape == Pixel::Accumulated {
            words.extend([
                0x560A_04F4, // v_fmac_f32 v5, 2.0, v2
                0x7E00_0305, // v_mov_b32 v0, v5
                0x7E02_0303, // v_mov_b32 v1, v3
            ]);
        } else if outside {
            words.extend([
                // v_mul_f32 v0, 2.0, v2 (v0, v5, v2: the factor written before the mode;
                // v0, s36, v2: the buffer's word)
                match shape {
                    Pixel::Before => 0x1000_0505,
                    Pixel::Scaled => 0x1000_0424,
                    _ => 0x1000_04F4,
                },
                0x7E02_0303, // v_mov_b32 v1, v3
            ]);
        }
        let computed = u32::from(shape == Pixel::Computed);
        words.extend([
            0x7E08_02FF,
            SENTINEL[0].to_bits(), // v_mov_b32 v4, 7.0
            0x7E0A_0280,           // v_mov_b32 v5, 0
            0x7E0C_0280,           // v_mov_b32 v6, 0
            0x7E0E_0280,           // v_mov_b32 v7, 0
        ]);
        let skip = chosen.len() as u32 + 3;
        match shape {
            Pixel::Folded => words.extend([
                0x7C02_04F0,              // v_cmp_lt_f32 vcc_lo, 0.5, v2
                0xBE94_3C6A,              // s_and_saveexec_b32 s20, vcc_lo
                0xBF88_0000 | (skip + 2), // s_cbranch_execz -> past the sample
                0x8A18_6A14,              // s_andn2_b32 s24, s20, vcc_lo
                SCC1 | skip,              // s_cbranch_scc1 -> past the sample
            ]),
            Pixel::Joined => words.extend([
                0x7C02_04F0, // v_cmp_lt_f32 vcc_lo, 0.5, v2
                0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
                0xBF88_0001, // s_cbranch_execz -> the other lanes
                0xBF82_0001, // s_branch -> the choice
                0x8A7E_6A14, // the other lanes: s_andn2_b32 exec_lo, s20, vcc_lo
            ]),
            _ => words.extend([
                0xBE94_037E,                     // s_mov_b32 s20, exec_lo
                0x7C22_04F0,                     // v_cmpx_lt_f32 exec_lo, 0.5, v2
                0xBF88_0000 | (skip + computed), // s_cbranch_execz -> past the sample
            ]),
        }
        if shape == Pixel::Computed {
            words.push(0x0604_0502); // v_add_f32 v2, v2, v2
        }
        words.extend(chosen);
        words.extend([
            0xF080_0F08,
            // image_sample v[4:7], v[2:3] (v[0:1]: the computed pair), s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
            if outside { 0x0082_0400 } else { 0x0082_0402 },
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0x877E_167E, // s_and_b32 exec_lo, exec_lo, s22
        ]);
    } else {
        if shape == Pixel::Plain {
            words.push(0xBEFE_097E); // s_wqm_b32 exec_lo, exec_lo
        }
        words.extend(chosen);
        let sample = match shape {
            Pixel::Other(sample) => sample,
            _ => [0xF080_0F08, 0x0082_0400], // image_sample v[4:7], v[0:1], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
        };
        words.extend([
            0x7E00_02FF,
            1.25f32.to_bits(), // v_mov_b32 v0, 1.25
            0x7E02_02F0,       // v_mov_b32 v1, 0.5
        ]);
        words.extend(sample);
        words.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    }
    words.extend([
        0xF800_180F,
        0x0706_0504, // exp mrt0 v4, v5, v6, v7 done vm
        0xBF81_0000, // s_endpgm
    ]);
    let input = PsInput {
        usage: Usage::Generic,
        row: 0,
        mask: 0b1111,
        attr: 0,
        flat: shape == Pixel::Flat,
        fp16: false,
        pack2: false,
        row2: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![table(0, 1)],
        used: vec![views(ResourceType::Srv, 0, 8, 1), views(ResourceType::Cbv, 8, 4, 1)],
        io: StageIo::Ps {
            inputs: if section { vec![input] } else { Vec::new() },
            input_addr: if section { 1 << 1 } else { 0 }, // PERSP_CENTER: i in v0, j in v1
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let texcoord = [float("TEXCOORD", 0, 0, 0, 4)];
    let isg1 = super::container::signature(if section { &texcoord } else { &[] }, true);
    let osg1 = super::container::signature(&[super::image_lod_tests::target()], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    super::emit(&c, &native, &p, &st, &io, &res)
}

/// POSITION to SV_Position and TEXCOORD0 to the input of `ps`.
fn vertex_shader(ps: &HostShader) -> HostShader {
    let mut outputs: Vec<SigElement> =
        emitted_inputs(&ps.dxil).into_iter().filter(|e| e.kind == 0).collect();
    assert_eq!(outputs.len(), 1, "TEXCOORD0 alone: {outputs:?}");
    let element = |name: &str, kind, interp, row| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    outputs[0].usage = (1 << outputs[0].cols) - 1;
    let cols = outputs[0].cols;
    outputs.push(element("SV_Position", 3, 4, outputs[0].row.map_or(0, |r| r + 1)));
    let mut s = Shader::new(false);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    // Vertex input 0 is POSITION (output 1), input 1 TEXCOORD0 (output 0).
    for (input, output, cols) in [(0, 1, 4), (1, 0, cols)] {
        for col in 0..cols {
            let args = [s.m.i32(input), s.m.i32(0), s.m.i8(col), undef];
            let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
            let out = [s.m.i32(output), s.m.i32(0), s.m.i8(col), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
        }
    }
    s.inputs.push(element("POSITION", 0, 0, 0));
    s.inputs.push(element("TEXCOORD", 0, 0, 1));
    s.outputs = outputs;
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

/// How a compute program's lanes come to their sampler.
#[derive(Clone, Copy, PartialEq)]
enum Chooser {
    /// A uniform condition the test generates: a word of a constant buffer from the table
    /// against 1.0, and a branch on the comparison. Which word and which entry are this
    /// file's own layout: word 4 of the buffer at entry 1 here.
    Flag,
    /// A root constant in s2 against 1.0: a scalar the hardware sets up, no word of a
    /// constant buffer.
    RootConstant,
    /// The flag's word compared the other way round (`s_cmp_eq_u32`, `s_cbranch_scc0`).
    Equal,
    /// The flag's word compared with another word of the buffer.
    TwoWords,
    /// A branch on a lane mask instead (`s_cbranch_vccz`, lanes below 16 set): the
    /// console wave takes it as one, with one sampler for all its lanes.
    Lanes,
    /// A scalar an earlier branch on that lane mask leaves (1.0 where the wave has a lane
    /// below 16, else 0), against 1.0: the console wave takes the first branch as one, so
    /// all 32 lanes have 1.0 and clamp; each lane by its own bit would have its own
    /// scalar, and the lanes from 16 on would wrap.
    LaneFlag,
    /// [`Chooser::Flag`], and the sample in a block after the join's.
    Later,
    /// No choice: the first sampler of the pair, as any program names an immediate one.
    One,
}

/// What stands around a compute program's choice and its sample.
#[derive(Clone, Copy, PartialEq)]
enum Around {
    Nothing,
    /// Both under a lane mask no branch tests (`s_mov_b32 exec_lo, 2`): the sample runs
    /// for lane 1 alone.
    Mask,
    /// A comparison sample (`image_sample_c_lz`) after it, with these immediate sampler
    /// words.
    Compared([u32; 4]),
    /// Both behind a branch on a lane mask only the lanes below 16 are in
    /// (`v_cmp_gt_u32` and `s_cbranch_vccz`), EXEC left whole: the console wave has such
    /// a lane, does not branch, and samples in all 32 lanes; the lanes from 16 on would
    /// keep [`SENTINEL`] if each went by its own bit.
    Branch,
    /// Both in a section only the lanes from 16 on run (`v_cmpx_le_u32` and
    /// `s_cbranch_execz`), a divergent section: the console samples in exactly
    /// the lanes that enter, and the lanes below 16 keep [`SENTINEL`].
    Section,
    /// [`Around::Section`], with the table's address parked in lane 0 of a VGPR that has
    /// another use (`v_writelane_b32`) and read back inside the section
    /// (`v_readlane_b32`), where lane 0 does not run: the constant buffer is loaded from
    /// the table at the address read back.
    Spilled,
    /// Both behind a scalar branch whose scalar a branch on that lane mask left (1.0 where
    /// the wave has a lane below 16, as it has): the console samples in all 32 lanes, and
    /// the lanes from 16 on would keep [`SENTINEL`] if each had its own scalar.
    LaneScalar,
    /// Both behind a scalar branch on word 6 of the constant buffer against 1.0: every
    /// lane samples or every lane keeps [`SENTINEL`].
    Word,
    /// [`Around::Section`] with EXEC put back to the whole wave inside the section, before
    /// the choice: the console's wave enters the section for its lanes from 16 on and
    /// then samples in all 32; a lane below 16 that went by its own bit is not there.
    Widened,
    /// [`Around::Section`] entered on `EXEC & ~VCC` with VCC the lanes below 16, and
    /// `s_cbranch_vccnz` inside it round the choice and the sample: VCC is not zero, the
    /// console's wave takes that branch and no lane samples. A lane's own `VCC & EXEC` is
    /// clear in every lane.
    Vcc,
    /// Both in a loop the lanes leave on different passes (`v_cmpx_le_i32` on a count of
    /// 0 in the even lanes and 1 in the odd, and `s_cbranch_execnz` back).
    Loop,
    /// Both in a section of the lanes below 16 (`v_cmp_gt_u32`, `s_and_saveexec_b32`,
    /// `s_cbranch_execz`), behind `s_andn2_b32` of the saved EXEC and VCC and
    /// `s_cbranch_scc1` past them: the console's wave has lanes from 16 on, branches, and
    /// no lane samples; the lanes' condition folds to a constant (Neon's third hole).
    Folded,
    /// Both under EXEC set to every lane (`s_mov_b32 exec_lo, -1`), as mesh programs
    /// start.
    Every,
    /// A wave64 program of 64 threads: both in a section of the lanes from 16 on
    /// (`v_cmp_le_u32`, `s_and_saveexec_b64`, a copy of the section's EXEC), skipped by
    /// `s_cbranch_execz` after the high dword of EXEC is cleared and entered with the copy put
    /// back. The console tests the low dword alone (lanes 16 to 31), enters, and samples in
    /// lanes 16 to 63; each lane by its own bit of the tested EXEC, lanes 32 to 63 would skip.
    HighCleared,
    /// A wave64 program of 64 threads: both in a section `s_and_saveexec_b64` makes of VCC
    /// whose high dword is no compare's (`s_mov_b32 vcc_hi, -1` after the `v_cmp_le_u32`).
    HighMask,
    /// Both, and the store, in a loop with no way out (`s_branch` back to the choice),
    /// behind a branch on a lane mask only the lanes below 16 are in (`v_cmp_gt_u32` and
    /// `s_cbranch_vccz` past the loop): the console's wave has such a lane and goes round
    /// in all 32; the lanes from 16 on would skip the loop, each by its own bit (Neon's
    /// seventh review).
    Endless,
    /// Both after a loop closed by a plain jump (`s_branch` back) that is left by a branch
    /// on a lane mask only the lanes below 16 are in (`v_cmp_gt_u32` and `s_cbranch_vccz`
    /// out): the console's wave has such a lane and never leaves; as lowered, the lanes
    /// below 16 stop at the iteration limit and those from 16 on sample (Neon's seventh
    /// review).
    Spin,
}

/// A wave32 compute program of 32 threads: the texture at entry 0 of the table in s0,
/// sampled by `sample` (to v12) at (1.25, 0.5) with the sampler of `pair` chosen as
/// `chooser` says, each lane's red to dword `lane` of the raw UAV at entry 2.
fn compute(
    chooser: Chooser,
    pair: [[u32; 4]; 2],
    sample: [u32; 2],
    around: Around,
) -> Result<HostShader, ShaderError> {
    lowered(&compute_program(chooser, pair, sample, around))
}

/// The program of [`compute`].
fn compute_program(
    chooser: Chooser,
    pair: [[u32; 4]; 2],
    sample: [u32; 2],
    around: Around,
) -> NativeProgram {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_02FF,
        1.25f32.to_bits(), // v_mov_b32 v2, 1.25
        0x7E06_02F0,       // v_mov_b32 v3, 0.5
        0x3420_0082,       // v_lshlrev_b32 v16, 2, v0
    ];
    // The constant buffer, from the table at s[4:5] (read back from a lane: at s[6:7]).
    let buffer = [
        0xF408_0202 | u32::from(around == Around::Spilled),
        0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20
        0xF420_0084,
        0xFA00_0010, // s_buffer_load_dword s2, s[8:11], 0x10
    ];
    // The choice and its sample.
    let mut inner = Vec::new();
    if around == Around::Spilled {
        inner.extend([
            0xD760_0006,
            0x0001_0114, // v_readlane_b32 s6, v20, 0
            0xBE87_0380, // s_mov_b32 s7, 0
        ]);
    }
    let branch = match chooser {
        Chooser::Flag | Chooser::Later => {
            inner.extend(buffer);
            inner.push(0xBF07_F202); // s_cmp_lg_u32 s2, 1.0
            Some(SCC1)
        }
        Chooser::RootConstant => {
            inner.push(0xBF07_F202); // s_cmp_lg_u32 s2, 1.0
            Some(SCC1)
        }
        Chooser::Equal => {
            inner.extend(buffer);
            inner.push(0xBF06_F202); // s_cmp_eq_u32 s2, 1.0
            Some(0xBF84_0000) // s_cbranch_scc0
        }
        Chooser::TwoWords => {
            inner.extend(buffer);
            inner.extend([
                0xF420_00C4,
                0xFA00_0014, // s_buffer_load_dword s3, s[8:11], 0x14
                0xBF07_0302, // s_cmp_lg_u32 s2, s3
            ]);
            Some(SCC1)
        }
        Chooser::Lanes => {
            inner.push(0x7D88_0090); // v_cmp_gt_u32 vcc_lo, 16, v0
            Some(VCCZ)
        }
        Chooser::LaneFlag => {
            inner.extend([
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBF86_0002, // s_cbranch_vccz -> no lane
                0xBE82_03F2, // s_mov_b32 s2, 1.0
                0xBF82_0001, // s_branch -> the compare
                0xBE82_0380, // no lane: s_mov_b32 s2, 0
                0xBF07_F202, // s_cmp_lg_u32 s2, 1.0
            ]);
            Some(SCC1)
        }
        Chooser::One => None,
    };
    match branch {
        Some(branch) => inner.extend(choice(branch, pair)),
        None => {
            for (k, word) in pair[0].into_iter().enumerate() {
                inner.extend(mov(16 + k as u32, word));
            }
        }
    }
    if chooser == Chooser::Later {
        inner.push(0xBF82_0000); // s_branch -> the next instruction
    }
    inner.extend(sample);
    inner.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    let past = inner.len() as u32;
    let sentinel = [0x7E18_02FF, SENTINEL[0].to_bits()]; // v_mov_b32 v12, 7.0
    match around {
        Around::Nothing => words.extend(inner),
        Around::Mask => {
            words.push(0xBEFE_0382); // s_mov_b32 exec_lo, 2
            words.extend(inner);
            words.push(0xBEFE_03C1); // s_mov_b32 exec_lo, -1
        }
        Around::Compared(sampler) => {
            words.extend(inner);
            for (k, word) in sampler.into_iter().enumerate() {
                words.extend(mov(20 + k as u32, word));
            }
            words.extend([
                0xF0BC_0108,
                0x00A6_1401, // image_sample_c_lz v20, v[1:3], s[24:31], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
            ]);
        }
        Around::Branch => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                VCCZ | past, // s_cbranch_vccz -> past the sample
            ]);
            words.extend(inner);
        }
        Around::Section | Around::Spilled => {
            if around == Around::Spilled {
                words.extend([
                    0xD761_0014,
                    0x0001_0000, // v_writelane_b32 v20, s0, 0
                    0x7E2A_0314, // v_mov_b32 v21, v20
                ]);
            }
            words.extend(sentinel);
            words.extend([
                0xBEA4_037E,        // s_mov_b32 s36, exec_lo
                0x7DA6_0090,        // v_cmpx_le_u32 exec_lo, 16, v0
                0xBF88_0000 | past, // s_cbranch_execz -> past the sample
            ]);
            words.extend(inner);
            words.push(0xBEFE_0324); // s_mov_b32 exec_lo, s36
        }
        Around::Widened => {
            words.extend(sentinel);
            words.extend([
                0xBEA4_037E,              // s_mov_b32 s36, exec_lo
                0x7DA6_0090,              // v_cmpx_le_u32 exec_lo, 16, v0
                0xBF88_0000 | (past + 1), // s_cbranch_execz -> past the sample
                0xBEFE_0324,              // s_mov_b32 exec_lo, s36
            ]);
            words.extend(inner);
        }
        Around::Vcc => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090,              // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBEA4_037E,              // s_mov_b32 s36, exec_lo
                0x8A7E_6A7E,              // s_andn2_b32 exec_lo, exec_lo, vcc_lo
                0xBF88_0000 | (past + 1), // s_cbranch_execz -> past the sample
                0xBF87_0000 | past,       // s_cbranch_vccnz -> past the sample
            ]);
            words.extend(inner);
            words.push(0xBEFE_0324); // s_mov_b32 exec_lo, s36
        }
        Around::LaneScalar => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBF86_0002, // s_cbranch_vccz -> no lane
                0xBE83_03F2, // s_mov_b32 s3, 1.0
                0xBF82_0001, // s_branch -> the compare
                0xBE83_0380, // no lane: s_mov_b32 s3, 0
                0xBF07_F203, // s_cmp_lg_u32 s3, 1.0
                SCC1 | past, // s_cbranch_scc1 -> past the sample
            ]);
            words.extend(inner);
        }
        Around::Loop => {
            words.extend(sentinel);
            words.extend([
                0x3628_0081, // v_and_b32 v20, 1, v0
                0xBEA4_037E, // s_mov_b32 s36, exec_lo
            ]);
            words.extend(inner);
            words.extend([
                0x4A28_28C1,                               // v_add_nc_u32 v20, -1, v20
                0x7D26_2880,                               // v_cmpx_le_i32 exec_lo, 0, v20
                0xBF89_0000 | (0x1_0000 - (past + 3)), // s_cbranch_execnz -> the choice
                0xBEFE_0324,                               // s_mov_b32 exec_lo, s36
            ]);
        }
        Around::Every => {
            words.push(0xBEFE_03C1); // s_mov_b32 exec_lo, -1
            words.extend(inner);
        }
        Around::Folded => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090,              // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBEA4_3C6A,              // s_and_saveexec_b32 s36, vcc_lo
                0xBF88_0000 | (past + 2), // s_cbranch_execz -> past the sample
                0x8A26_6A24,              // s_andn2_b32 s38, s36, vcc_lo
                SCC1 | past,              // s_cbranch_scc1 -> past the sample
            ]);
            words.extend(inner);
            words.push(0xBEFE_0324); // s_mov_b32 exec_lo, s36
        }
        Around::HighCleared => {
            words.extend(sentinel);
            words.extend([
                0x7D86_0090,              // v_cmp_le_u32 vcc, 16, v0
                0xBEA4_246A,              // s_and_saveexec_b64 s[36:37], vcc
                0xBEA6_047E,              // s_mov_b64 s[38:39], exec
                0xBEFF_0380,              // s_mov_b32 exec_hi, 0
                0xBF88_0000 | (past + 1), // s_cbranch_execz -> past the sample
                0xBEFE_0426,              // s_mov_b64 exec, s[38:39]
            ]);
            words.extend(inner);
            words.push(0xBEFE_0424); // s_mov_b64 exec, s[36:37]
        }
        Around::HighMask => {
            words.extend(sentinel);
            words.extend([
                0x7D86_0090,        // v_cmp_le_u32 vcc, 16, v0
                0xBEEB_03C1,        // s_mov_b32 vcc_hi, -1
                0xBEA4_246A,        // s_and_saveexec_b64 s[36:37], vcc
                0xBF88_0000 | past, // s_cbranch_execz -> past the sample
            ]);
            words.extend(inner);
            words.push(0xBEFE_0424); // s_mov_b64 exec, s[36:37]
        }
        Around::Endless => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090,              // v_cmp_gt_u32 vcc_lo, 16, v0
                VCCZ | (past + 3),        // s_cbranch_vccz -> past the loop
            ]);
            words.extend(inner);
            words.extend([
                0xE070_1000,
                0x800A_0C10,                           // buffer_store_dword v12, v16, s[40:43], 0 offen
                0xBF82_0000 | (0x1_0000 - (past + 3)), // s_branch -> the choice
            ]);
        }
        Around::Spin => {
            words.extend(sentinel);
            words.extend([
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                VCCZ | 1,    // s_cbranch_vccz -> out of the loop
                0xBF82_FFFD, // s_branch -> the compare
            ]);
            words.extend(inner);
        }
        Around::Word => {
            words.extend(sentinel);
            words.extend([
                0xF408_0202,
                0xFA00_0020, // s_load_dwordx4 s[8:11], s[4:5], 0x20
                0xF420_00C4,
                0xFA00_0018, // s_buffer_load_dword s3, s[8:11], 0x18
                0xBF07_F203, // s_cmp_lg_u32 s3, 1.0
                SCC1 | past, // s_cbranch_scc1 -> past the sample
            ]);
            words.extend(inner);
        }
    }
    words.extend([
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let mut layout = vec![table(0, 0)];
    if chooser == Chooser::RootConstant {
        layout.push(root_constant());
    }
    let wave64 = matches!(around, Around::HighCleared | Around::HighMask);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout,
        used: vec![
            views(ResourceType::Srv, 0, 8, 0),
            views(ResourceType::Cbv, 8, 4, 0),
            views(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [if wave64 { 64 } else { 32 }, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The compute program `native` lowered to a host shader.
fn lowered(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
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
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered.samplers.iter().map(|&w| StaticSampler::from_words(w)).collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    })
}

/// Both immediate samplers, each once, and a module `dxil.dll` takes.
fn checked(shader: &HostShader) {
    let mut modes: Vec<[u32; 3]> = shader.static_samplers.iter().map(|s| s.address).collect();
    modes.sort_unstable();
    assert_eq!(modes, [[1; 3], [3; 3]], "wrap and clamp, one static sampler each");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
}

/// The user data of `shader` with its table at heap index 0.
fn user_data(shader: &HostShader) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Table => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    ud
}

fn refusal(r: Result<HostShader, ShaderError>) -> Option<String> {
    match r {
        Err(ShaderError::Unsupported { what, .. }) => Some(what.to_string()),
        Err(e) => Some(e.to_string()),
        Ok(_) => None,
    }
}

/// The flags a run is made with: the sampler each chooses, by what the fixture's sample
/// reads with it. Only the bits of 1.0 clamp (the integer 1 is another word).
const FLAGS: [(u32, f32); 4] = [(SET, CLAMPED), (0, WRAPPED), (0x4000_0000, WRAPPED), (1, WRAPPED)];

#[test]
fn a_pixel_samples_with_the_immediate_sampler_its_branch_chose() {
    let ps = pixel_shader(Pixel::Plain, POINT).expect("emit");
    checked(&ps);
    let vs = super::image_lod_tests::vertex_shader();
    // A triangle over the whole target: POSITION.
    let vertices = [-1.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: graphics pipeline failed: {e}"));
        for (flag, want) in FLAGS {
            let texels = host
                .draw_with(
                    &root,
                    &pso,
                    &Draw {
                        roots: vec![
                            (0, Root::Constants(vec![0])),
                            (1, Root::Constants(user_data(&ps))),
                        ],
                        textures: &[(2, 1, &TEXELS)],
                        samplers: &[],
                        vertices: &vertices,
                        stride: 16,
                        size: (4, 4),
                        mesh_groups: 0,
                    },
                    &[&[0, 0, 0, 0, SET ^ 1, flag, 0, 0]],
                )
                .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
            assert_eq!(texels.len(), 16);
            assert!(
                texels.iter().all(|t| *t == [want, 0.0, 0.0, 1.0]),
                "{label}, flag {flag:#x}: {texels:?}"
            );
        }
        eprintln!("{label}: the pixel sampled with the sampler its branch chose");
    }
}

/// A masked pixel case: the choice and its sample in a section some lanes of a quad
/// sit out, the level taken from the differences of a coordinate that changes by two texels
/// a pixel, made where the coordinate is interpolated (every lane of the quad there). The
/// texture says which level and texel each lane read: level 0 is 10 to 13 across, level 1
/// is 20 and 21, level 2 is 30.
///
/// Pixel x has u = 0.25 + x / 2. Pixel 0 (u below 0.5) sits out and keeps the sentinel,
/// with pixel 1 of its quad sampling beside it. Half a level-0 width a pixel is level 1,
/// whose two texels a pixel reads by turns when wrapped (21 at odd x, 20 at even) and
/// whose last texel every pixel past the edge reads when clamped.
#[test]
fn a_section_of_a_quad_samples_the_level_its_derivatives_choose() {
    section_levels(Pixel::Section);
}

/// Coordinates the program computes before the section, where every lane of the quad runs
/// under whole-quad mode, by per-lane arithmetic of an interpolated attribute and a
/// constant: their differences are made there, and the sample in the section takes them as
/// its gradients. Twice U is level 2.
#[test]
fn coordinates_computed_before_a_section_of_a_quad_sample_the_level_their_differences_choose() {
    section_levels(Pixel::Outside);
}

/// The same with the factor a word of the constant buffer, loaded where every lane runs.
#[test]
fn a_coordinate_scaled_by_a_buffer_word_samples_the_level_its_differences_choose() {
    section_levels(Pixel::Scaled);
}

/// A coordinate each lane selects by its own compare (`v_cmp_gt_f32`, `v_cndmask_b32`)
/// where every lane of the quad runs: doubled U in the quads of pixels 0 to 3 (level 2),
/// U itself in the others (level 1).
#[test]
fn a_coordinate_selected_by_each_lanes_own_compare_samples_the_level_its_differences_choose() {
    section_levels(Pixel::Selected);
}

/// Each admitted implicit sample is made with explicit gradients (`SampleGrad`): its two
/// coordinates differenced where they are written, not derivatives where it stands.
#[test]
fn an_admitted_implicit_sample_takes_the_differences_of_its_coordinates() {
    for (shape, pair) in [
        (Pixel::Plain, POINT),
        (Pixel::Section, POINT_MIP),
        (Pixel::Outside, POINT_MIP),
        (Pixel::Scaled, POINT_MIP),
        (Pixel::Ended, POINT_MIP),
    ] {
        lower::CHOICE_MADE.with(|m| m.set((0, 0)));
        pixel_shader(shape, pair).expect("emit");
        let made = lower::CHOICE_MADE.with(std::cell::Cell::get);
        assert_eq!(made, (2, 1), "coordinates differenced, gradient samples");
    }
    // A named level takes none.
    lower::CHOICE_MADE.with(|m| m.set((0, 0)));
    compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Nothing).expect("lower");
    assert_eq!(lower::CHOICE_MADE.with(std::cell::Cell::get), (0, 0));
}

/// The reasons of the coordinate rule (the text of `sampler_choice.rs`).
const UNLISTED: &str = "an immediate sampler chosen by control flow for an implicit sample at coordinates made of other than interpolated attributes, constants, constant buffer words and per-lane arithmetic";
const JOINED: &str = "an immediate sampler chosen by control flow for an implicit sample at coordinates joined from different paths";
const UNINTERPOLATED: &str = "an immediate sampler chosen by control flow for an implicit sample at an attribute flat, of halves, or not interpolated by one pair";

/// Nor is a coordinate scaled by a scalar read from the first lane: a lane read is no
/// value of the list, though whole-quad mode is on and no branch splits the quad.
#[test]
fn a_coordinate_that_depends_on_a_first_lane_read_is_refused() {
    let why = refusal(pixel_shader(Pixel::FirstLane, POINT_MIP));
    assert_eq!(why.as_deref(), Some(UNLISTED));
}

/// Nor is a coordinate a branch on such a read chooses: a join is no value of the list,
/// whatever branch it joins after.
#[test]
fn a_coordinate_a_branch_on_a_first_lane_read_chooses_is_refused() {
    let why = refusal(pixel_shader(Pixel::Diamond, POINT_MIP));
    assert_eq!(why.as_deref(), Some(JOINED));
}

/// Nor a coordinate selected by a mask that is no compare's (a constant VCC).
#[test]
fn a_coordinate_selected_by_a_mask_no_compare_wrote_is_refused() {
    let why = refusal(pixel_shader(Pixel::SelectedConstant, POINT_MIP));
    assert_eq!(why.as_deref(), Some(UNLISTED));
}

/// Nor a coordinate a quad swizzle reads from the neighbour.
#[test]
fn a_coordinate_a_quad_swizzle_reads_is_refused() {
    let why = refusal(pixel_shader(Pixel::Swizzle, POINT_MIP));
    assert_eq!(why.as_deref(), Some(UNLISTED));
}

/// Nor a coordinate a branch on the SCC of an EXEC write chooses: the lanes go each by
/// its own bit as on the console, but a join is no value of the list. (Round five admitted
/// this shape; the rule of explicit gradients supersedes it.)
#[test]
fn a_coordinate_a_lane_mask_branch_chooses_is_refused() {
    let why = refusal(pixel_shader(Pixel::LaneArm, POINT_MIP));
    assert_eq!(why.as_deref(), Some(JOINED));
}

/// An attribute the stage data marks flat is no interpolated attribute of the list,
/// though its interpolation pair is the perspective centre's: the console's value is the
/// provoking vertex's.
#[test]
fn a_flat_attribute_sampled_in_a_section_of_a_quad_is_refused() {
    let why = refusal(pixel_shader(Pixel::Flat, POINT_MIP));
    assert_eq!(why.as_deref(), Some(UNINTERPOLATED));
}

/// Neon's first hole: a branch on VCC that the lanes' `VCC & EXEC` folds away chooses U.
/// The console skips the arm, the lowering runs it in every lane; U is a join either way.
#[test]
fn a_coordinate_a_branch_on_vcc_outside_exec_chooses_is_refused() {
    let why = refusal(pixel_shader(Pixel::VccCancel, POINT_MIP));
    assert_eq!(why.as_deref(), Some(JOINED));
}

/// Neon's second hole: U copied from a scalar an arm of a lane-mask branch wrote. The
/// console's scalar is the wave's, the lowering's each lane's; the scalar is a join.
#[test]
fn a_coordinate_copied_from_a_scalar_a_lane_mask_arm_wrote_is_refused() {
    let why = refusal(pixel_shader(Pixel::ArmScalar, POINT_MIP));
    assert_eq!(why.as_deref(), Some(JOINED));
}

/// The lane-set rule: every branch the sample's block hangs on is a flag of the proved
/// form, or the `s_cbranch_execz` of a section whose EXEC the sample's is.
const BEHIND: &str = "an immediate sampler chosen by control flow behind a branch neither a proved flag nor the section of its sample's EXEC";
/// The sample's EXEC is not the program's, its whole quads', or a section of them.
const LANE_SET: &str = "an immediate sampler chosen by control flow for a sample under an EXEC not proved the program's, its whole quads' or a section of them";

/// Neon's third hole: inside the section, a scalar branch on `s_andn2_b32` of the saved
/// EXEC and VCC past the choice and the sample. The console skips them where some lane is
/// outside VCC; the lanes' condition folds to a constant. No such branch is of the list,
/// in a pixel program or a compute one.
#[test]
fn a_chosen_sampler_behind_a_branch_on_a_folded_lane_mask_is_refused() {
    let why = refusal(pixel_shader(Pixel::Folded, POINT_MIP));
    assert_eq!(why.as_deref(), Some(BEHIND));
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Folded));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// Neon's diamond: `s_cbranch_execz` to a block that writes EXEC, the other way jumping
/// over it, the sample after both. Its EXEC is a join of the two, no EXEC of the list.
#[test]
fn a_chosen_sampler_under_an_exec_joined_from_two_ways_is_refused() {
    let why = refusal(pixel_shader(Pixel::Joined, POINT_MIP));
    assert_eq!(why.as_deref(), Some(LANE_SET));
}

/// After an if both of whose ways put EXEC back to the one saved mask: the join's EXEC is a
/// phi of two copies of that mask, which is that mask, and the sample takes the levels of
/// [`Pixel::Ended`].
#[test]
fn a_sample_under_a_join_of_copies_of_one_mask_takes_the_level_its_differences_choose() {
    section_levels(Pixel::Rejoined);
}

/// But a join of two masks is none of the list: here the lowering guards the sample (the
/// lanes' EXEC differs by way), and where it does not, the list refuses the phi
/// ([`Pixel::Joined`]). Nor a join with one way's EXEC another value that equals the mask
/// as it runs.
#[test]
fn a_sample_under_a_join_of_two_masks_or_of_another_copy_is_refused() {
    let why = refusal(pixel_shader(Pixel::TwoMasks, POINT_MIP));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for a sample under a lane mask")
    );
    let why = refusal(pixel_shader(Pixel::Joined, POINT_MIP));
    assert_eq!(why.as_deref(), Some(LANE_SET));
    let why = refusal(pixel_shader(Pixel::OtherCopy, POINT_MIP));
    assert_eq!(why.as_deref(), Some(LANE_SET));
}

/// The same levels where whole-quad mode ends before the sample, as compiled code ends it
/// once the coordinates are computed: the helper lanes hold the coordinates and the
/// console differences their registers whether EXEC holds them or not. No lane sits out:
/// pixel 0 reads the first texel of level 1.
#[test]
fn a_sample_after_whole_quad_mode_ends_takes_the_level_its_derivatives_choose() {
    section_levels(Pixel::Ended);
}

fn texel(red: f32) -> [f32; 4] {
    [red, 0.0, 0.0, 1.0]
}

/// What the pixel program `ps` writes to its 8 by 2 target on `host` with `flag` in word 5
/// of the constant buffer: TEXCOORD0.x from 0 at the left edge to 4 at the right, sampling a
/// texture of three levels whose texels say which (level 0 is 10 to 13 across, level 1 is
/// 20 and 21, level 2 is 30).
fn levels_drawn(host: &Host, label: &str, ps: &HostShader, flag: u32) -> Vec<[f32; 4]> {
    let vs = vertex_shader(ps);
    // A triangle over the whole target: POSITION, then TEXCOORD0.
    #[rustfmt::skip]
    let vertices = [
        -1.0, -1.0, 0.5, 1.0, 0.0, 0.25, 0.0, 0.0,
        -1.0, 3.0, 0.5, 1.0, 0.0, 0.25, 0.0, 0.0,
        3.0, -1.0, 0.5, 1.0, 8.0, 0.25, 0.0, 0.0,
    ];
    let mut texels = Vec::new();
    for _row in 0..4 {
        texels.extend([10.0, 11.0, 12.0, 13.0].map(texel));
    }
    texels.extend([20.0, 21.0, 20.0, 21.0].map(texel));
    texels.push(texel(30.0));
    let (root, pso) = host
        .pipeline(&vs, ps)
        .unwrap_or_else(|e| panic!("{label}: graphics pipeline failed: {e}"));
    host.draw_with(
        &root,
        &pso,
        &Draw {
            roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(user_data(ps)))],
            textures: &[(4, 4, &texels)],
            samplers: &[],
            vertices: &vertices,
            stride: 32,
            size: (8, 2),
            mesh_groups: 0,
        },
        &[&[0, 0, 0, 0, SET ^ 1, flag, 2.0f32.to_bits(), 0]],
    )
    .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"))
}

/// The draw of the tests above that run, of a pixel program of `shape`.
fn section_levels(shape: Pixel) {
    let ps = pixel_shader(shape, POINT_MIP).expect("emit");
    checked(&ps);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (flag, clamped) in [(SET, true), (0, false)] {
            let out = levels_drawn(&host, label, &ps, flag);
            // Twice the coordinate (`Pixel::Outside`, `Pixel::Scaled`) changes by four
            // level-0 texels a pixel: level 2, one texel, whichever the sampler.
            let doubled = matches!(shape, Pixel::Outside | Pixel::Scaled);
            let row: Vec<[f32; 4]> = (0..8)
                .map(|x| match x {
                    0 if matches!(shape, Pixel::Ended | Pixel::Rejoined) => texel(20.0),
                    0 => SENTINEL,
                    1..=3 if shape == Pixel::Selected => texel(30.0),
                    _ if doubled => texel(30.0),
                    1 => texel(21.0),
                    _ if clamped || x % 2 == 1 => texel(21.0),
                    _ => texel(20.0),
                })
                .collect();
            assert_eq!(out, [row.clone(), row].concat(), "{label}, flag {flag:#x}");
        }
        eprintln!("{label}: a section of a quad sampled its level with the sampler its branch chose");
    }
}

/// A chosen sampler's level bias and level clamps act on the differences it samples with
/// (`SampleGrad`) as on the implicit derivatives of a sample with that sampler alone (Neon's
/// sixth review, its section 2): the sample of [`Pixel::Ended`] after the choice reads what
/// [`Pixel::Single`] reads with the sampler the flag chooses, on the software adapter. Biased
/// by one level that is level 2; biased so but held at most at level 1, level 1; held at
/// least at level 2, level 2.
#[test]
fn a_chosen_samplers_bias_and_level_clamps_act_on_its_differences_as_on_derivatives() {
    // Word 1: the least and the most level (unsigned 4.8); word 2: the bias (signed 5.8) and
    // a point mip filter.
    let cases = [
        ("biased by one level", 0x00FF_F000, 0x0400_0100, true),
        ("biased by one level, at most level 1", 0x0010_0000, 0x0400_0100, false),
        ("at least level 2", 0x00FF_F200, 0x0400_0000, true),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no SM 6.6 warp device; skipped");
        return;
    };
    for (what, lod, bias, level_2) in cases {
        let pair = [[CLAMP, lod, bias, 0], [0, lod, bias, 0]];
        lower::CHOICE_MADE.with(|m| m.set((0, 0)));
        let chosen = pixel_shader(Pixel::Ended, pair).expect("emit");
        let made = lower::CHOICE_MADE.with(std::cell::Cell::get);
        assert_eq!(made, (2, 1), "{what}: coordinates differenced, gradient samples");
        checked(&chosen);
        // The flag 1.0 chooses the clamped sampler, any other the wrapped one.
        for (flag, k) in [(SET, 0), (0, 1)] {
            let alone = pixel_shader(Pixel::Single, [pair[k], pair[1 - k]]).expect("emit");
            let want = levels_drawn(&host, "warp", &alone, flag);
            let got = levels_drawn(&host, "warp", &chosen, flag);
            assert_eq!(got, want, "{what}, flag {flag:#x}");
            let level = |t: &[f32; 4]| if level_2 { t[0] == 30.0 } else { [20.0, 21.0].contains(&t[0]) };
            assert!(got.iter().all(level), "{what}, flag {flag:#x}: {got:?}");
        }
    }
    eprintln!("warp: a chosen sampler's bias and level clamps acted as on implicit derivatives");
}

#[test]
fn lanes_sample_at_level_zero_with_the_immediate_sampler_their_branch_chose() {
    let cs = compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Nothing).expect("lower");
    checked(&cs);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&cs)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (flag, want) in FLAGS {
            let buffer = [0, 0, 0, 0, flag, SET ^ 1, 0, 0];
            let views = [View::Texture(2, 1, &TEXELS), View::Cbv(&buffer), View::RawUav(32)];
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user_data(&cs)))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let red: Vec<f32> = words[0].iter().copied().map(f32::from_bits).collect();
            assert_eq!(red, [want; 32], "{label}, flag {flag:#x}");
        }
        eprintln!("{label}: each lane sampled with the sampler its branch chose");
    }
}

/// One group of `cs` dispatched on WARP (and on the host GPU without `VICTORY_WARP=1`)
/// with `buffer` as its constant buffer: `check` is given each lane's red.
fn dispatched(cs: &HostShader, buffers: &[[u32; 8]], check: impl Fn(&str, &[u32; 8], Vec<f32>)) {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(cs)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for buffer in buffers {
            let views = [View::Texture(2, 1, &TEXELS), View::Cbv(buffer), View::RawUav(32)];
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user_data(cs)))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, buffer, words[0].iter().copied().map(f32::from_bits).collect());
        }
    }
}

/// `low` in the lanes below 16 and `high` in the lanes from 16 on.
fn halves(low: f32, high: f32) -> Vec<f32> {
    [[low; 16], [high; 16]].concat()
}

/// A section of a compute wave: the choice and its sample behind
/// `s_cbranch_execz` on the EXEC a compare narrowed. The console samples in the lanes of
/// that EXEC and each lane goes by its own bit of it: the same lanes.
#[test]
fn a_section_of_a_wave_samples_with_the_sampler_its_branch_chose() {
    let cs = compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Section).expect("lower");
    checked(&cs);
    let buffers = FLAGS.map(|(flag, _)| [0, 0, 0, 0, flag, SET ^ 1, 0, 0]);
    dispatched(&cs, &buffers, |label, buffer, red| {
        let want = FLAGS.iter().find(|f| f.0 == buffer[4]).expect("a flag").1;
        assert_eq!(red, halves(SENTINEL[0], want), "{label}, flag {:#x}", buffer[4]);
    });
}

/// Under EXEC set to every lane (`s_mov` of -1, as the test sets it up): every lane samples
/// with the sampler its branch chose.
#[test]
fn lanes_under_every_lane_of_exec_sample_with_the_sampler_their_branch_chose() {
    let cs = compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Every).expect("lower");
    checked(&cs);
    let buffers = FLAGS.map(|(flag, _)| [0, 0, 0, 0, flag, SET ^ 1, 0, 0]);
    dispatched(&cs, &buffers, |label, buffer, red| {
        let want = FLAGS.iter().find(|f| f.0 == buffer[4]).expect("a flag").1;
        assert_eq!(red, [want; 32], "{label}, flag {:#x}", buffer[4]);
    });
}

/// A scalar branch on a word of the constant buffer around the choice and its sample:
/// every lane computes that word's compare alike, so all sample or none does.
#[test]
fn a_sample_behind_a_branch_on_a_buffer_word_runs_in_every_lane_or_in_none() {
    let cs = compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Word).expect("lower");
    checked(&cs);
    let buffers = [
        [0, 0, 0, 0, SET, SET ^ 1, SET, 0],
        [0, 0, 0, 0, 0, SET ^ 1, SET, 0],
        [0, 0, 0, 0, SET, SET ^ 1, 0, 0],
        [0, 0, 0, 0, 0, SET ^ 1, 1, 0],
    ];
    dispatched(&cs, &buffers, |label, buffer, red| {
        let want = match (buffer[6] == SET, buffer[4] == SET) {
            (false, _) => SENTINEL[0],
            (true, true) => CLAMPED,
            (true, false) => WRAPPED,
        };
        assert_eq!(red, [want; 32], "{label}, words {buffer:x?}");
    });
}

/// A branch on a lane mask around the choice and its sample, EXEC left whole: the console
/// wave has a lane in the mask, does not branch, and samples in all 32 lanes. Each lane
/// going by its own bit, the lanes from 16 on would skip the sample. A branch on VCC is
/// no branch of the list.
#[test]
fn a_chosen_sampler_behind_a_branch_on_a_mask_some_of_its_lanes_are_out_of_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Branch));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// A branch on VCC is the whole register's on the console and each lane's `VCC & EXEC` as
/// lowered. With VCC the lanes below 16 and EXEC the lanes from 16 on, the console's wave
/// takes `s_cbranch_vccnz` and no lane samples; every lane's own condition is clear, a
/// constant, and the lanes from 16 on would sample. The section's EXEC, `s_andn2_b32` of
/// EXEC and VCC, is no section of the list.
#[test]
fn a_chosen_sampler_behind_a_branch_on_vcc_bits_outside_exec_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Vcc));
    assert_eq!(why.as_deref(), Some(LANE_SET));
}

/// A section whose EXEC is put back to the whole wave before the sample: the console
/// samples in all 32 lanes once its wave has entered, and the lanes below 16 are not in
/// the section as lowered.
#[test]
fn a_chosen_sampler_in_a_section_whose_exec_is_widened_before_it_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Widened));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// A branch on a lane mask around a plain sample (one immediate sampler, no choice), EXEC
/// left whole: the console wave has a lane in the mask, does not branch, and samples in
/// all 32 lanes; as translated, the lanes from 16 on skip the sample. That is how such a
/// branch is lowered for every instruction behind it, whichever sampler a sample has.
///
/// The test states what the translation does today (structurisation lowers a branch on a
/// lane mask as each lane's own, on the word that compiled code skips only what the lanes
/// outside the mask would not change); the console has 2.0 in all 32 lanes, and a
/// lowering that takes such a branch as the wave does turns this test over.
#[test]
fn a_plain_sample_behind_a_branch_on_a_mixed_lane_mask_is_passed_over_today_by_the_lanes_out_of_it() {
    let cs = compute(Chooser::One, POINT, SAMPLE_LZ, Around::Branch).expect("lower");
    dispatched(&cs, &[[0; 8]], |label, _, red| {
        assert_eq!(red, halves(CLAMPED, SENTINEL[0]), "{label}");
    });
}

/// Nor is a scalar branch around them every lane's alike on its word: the scalar a branch
/// on a lane mask left is 1.0 for the console's whole wave, which then samples in all 32
/// lanes, and each lane's own as lowered.
#[test]
fn a_chosen_sampler_behind_a_scalar_branch_a_lane_branch_decided_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::LaneScalar));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// In a loop the lanes leave on different passes, a lane's own bit of the mask the loop
/// goes round on is another on each pass: that every lane the console samples in takes
/// the branch back by its own bit is not proved there. The EXEC at the sample is the
/// loop's join of passes, no EXEC of the list.
#[test]
fn a_chosen_sampler_in_a_loop_the_lanes_leave_apart_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Loop));
    assert_eq!(why.as_deref(), Some(LANE_SET));
}

/// The descriptor tracker follows a table address through a lane of a VGPR and back and
/// finds the root parameter; the heap index the lowering makes is of the value read back,
/// by a lane read (`WaveReadLaneAt`) of lane 0 in a section lane 0 does not run. The
/// constant buffer of the flag is then not proved the draw's one buffer.
#[test]
fn a_flag_of_a_buffer_whose_table_address_is_read_back_from_a_lane_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Spilled));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by a flag of a constant buffer whose table address is not a root parameter's own value")
    );
}

/// The console takes the derivatives of an implicit sample from the coordinate registers
/// of the quad's four lanes. A coordinate computed inside a section is not computed by the
/// lanes that sit the section out: the console differences against what those lanes held
/// before, and the host against a value those lanes never defined.
#[test]
fn coordinates_computed_in_a_section_of_a_quad_are_refused() {
    let why = refusal(pixel_shader(Pixel::Computed, POINT_MIP));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for an implicit sample at coordinates not every lane of the quad computes")
    );
}

/// Nor is a coordinate the console's in its helper lanes for being written under whole-quad
/// mode, where it is computed from a value written before the mode was turned on: the
/// helper lanes did not run that write, and multiply by what the register held.
#[test]
fn a_coordinate_computed_from_a_value_written_before_whole_quad_mode_is_refused() {
    let why = refusal(pixel_shader(Pixel::Before, POINT_MIP));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for an implicit sample at coordinates computed outside whole-quad mode")
    );
}

/// Nor where that value is the accumulator `v_fmac_f32` adds to (Neon's sixth review): the
/// destination's old value is an operand of the sum, which `ir::effects` records as read.
#[test]
fn a_coordinate_accumulated_into_a_value_written_before_whole_quad_mode_is_refused() {
    let why = refusal(pixel_shader(Pixel::Accumulated, POINT_MIP));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for an implicit sample at coordinates computed outside whole-quad mode")
    );
}

/// A branch one way of which is a loop with no way out (Neon's sixth review): the lanes that
/// go round it leave by the lowering's iteration limit and reach nothing after the branch,
/// so the sample hangs on it, a branch neither a proved flag nor a section. (As lowered the
/// loop's plain jump back leaves the program too: nothing is certain to follow it.)
#[test]
fn a_chosen_sampler_beside_a_loop_with_no_way_out_is_refused() {
    let why = refusal(pixel_shader(Pixel::Endless, POINT_MIP));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// The sample inside a loop with no way out, behind a branch on a lane mask (Neon's seventh
/// review): the loop's plain jump back leaves the program at the iteration limit as
/// lowered, so the branch decides whether a lane samples.
#[test]
fn a_chosen_sampler_in_a_loop_with_no_way_out_behind_a_lane_branch_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Endless));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// The sample after a loop closed by a plain jump and left by a branch on a lane mask
/// (Neon's seventh review): as lowered, a lane still in the loop at the iteration limit
/// leaves the program there, and the branch out of the loop decides whether a lane
/// samples.
#[test]
fn a_chosen_sampler_after_a_loop_closed_by_a_plain_jump_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Spin));
    assert_eq!(why.as_deref(), Some(BEHIND));
}

/// EXEC is never a value of the list, as an operand of the arithmetic or of the compare of a
/// select: here `exec_hi` of a wave32 program, holding a first-lane read (Neon's seventh
/// review).
#[test]
fn a_coordinate_computed_from_exec_as_a_number_is_refused() {
    let why = [Pixel::ExecHigh, Pixel::SelectedExecHigh]
        .map(|shape| refusal(pixel_shader(shape, POINT_MIP)));
    assert_eq!(why.each_ref().map(Option::as_deref), [Some(UNLISTED); 2]);
}

/// A select whose source is its own mask (`v_cndmask_b32 v0, vcc_lo, v0, vcc_lo`): the lanes
/// outside the compare take the wave's mask as a number, no value of the list, though the
/// mask is the select's mask too (Neon's sixth review).
#[test]
fn a_coordinate_selected_from_its_own_mask_as_a_number_is_refused() {
    let why = refusal(pixel_shader(Pixel::SelectedMask, POINT_MIP));
    assert_eq!(why.as_deref(), Some(UNLISTED));
}

/// What the lowering stops a mask's dword written alone with (`lower::LANE_MASK`).
const MASK_DATA: &str = "a lane mask read as data";

/// A wave64 section skipped by `s_cbranch_execz` on an EXEC whose high dword is cleared
/// (Neon's sixth review): the branch tests a mask the sample's EXEC is not made of, though
/// its low dword is. The console's wave enters on its low dword and samples in lanes 16 to
/// 63; lanes 32 to 63 would skip, each by its own bit.
///
/// The lowering refuses the write of EXEC's high dword alone before it looks at the choice.
/// Without that write the shape is admitted: `Lower::lanes_sampled` and `run_of` follow the
/// low dword of a wave64 mask, so a lowering that lets such a write through must make them
/// follow both, or this test turns over.
#[test]
fn a_chosen_sampler_behind_an_execz_on_another_high_dword_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::HighCleared));
    assert_eq!(why.as_deref(), Some(MASK_DATA));
}

/// A wave64 section of `s_and_saveexec_b64` whose mask has a high dword no compare wrote
/// (Neon's sixth review): no section of the list. Refused, likewise, at the write of VCC's
/// high dword alone; `run_of` asks only that the low dword be the compare's.
#[test]
fn a_chosen_sampler_in_a_section_of_a_mask_half_a_compare_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::HighMask));
    assert_eq!(why.as_deref(), Some(MASK_DATA));
}

/// Without whole-quad mode the console's helper lanes run nothing and hold no coordinates,
/// where the host's run the whole program.
#[test]
fn an_implicit_sample_outside_whole_quad_mode_is_refused() {
    let why = refusal(pixel_shader(Pixel::Unwhole, POINT));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for an implicit sample at coordinates computed outside whole-quad mode")
    );
}

const LANES: &str = "an immediate sampler chosen by a per-lane branch";
const UNPROVED: &str =
    "an immediate sampler chosen by a scalar condition not proved the same in every lane";

/// The console takes a branch on a lane mask as one wave: with a lane below 16 every lane
/// of the wave samples clamped, the lanes from 16 on too, whose own bit is clear. A lane's
/// own path through the branch does not say which sampler its wave has.
#[test]
fn an_immediate_sampler_a_lane_branch_chooses_is_refused() {
    let why = refusal(compute(Chooser::Lanes, POINT, SAMPLE_LZ, Around::Nothing));
    assert_eq!(why.as_deref(), Some(LANES));
}

/// Nor does a scalar say it that an earlier branch on a lane mask left: the scalar
/// compare is the same instruction in every lane, its operand is not the same number.
#[test]
fn an_immediate_sampler_chosen_by_a_scalar_a_lane_branch_left_is_refused() {
    let why = refusal(compute(Chooser::LaneFlag, POINT, SAMPLE_LZ, Around::Nothing));
    assert_eq!(why.as_deref(), Some(UNPROVED));
}

/// Only one compare is taken as proved: a word of a constant buffer at
/// a fixed place against a constant, by `s_cmp_lg_u32` and `s_cbranch_scc1`.
#[test]
fn an_immediate_sampler_chosen_by_another_scalar_compare_is_refused() {
    for (label, chooser) in [
        ("a root constant", Chooser::RootConstant),
        ("s_cmp_eq_u32 and s_cbranch_scc0", Chooser::Equal),
        ("two words of the buffer", Chooser::TwoWords),
    ] {
        let why = refusal(compute(chooser, POINT, SAMPLE_LZ, Around::Nothing));
        assert_eq!(why.as_deref(), Some(UNPROVED), "{label}");
    }
}

#[test]
fn an_immediate_sampler_chosen_before_the_block_of_its_sample_is_refused() {
    let why = refusal(compute(Chooser::Later, POINT, SAMPLE_LZ, Around::Nothing));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen in another block than its sample")
    );
}

/// A gather at a named level reads its four texels without the sampler (clamped, whatever
/// the sampler says), and beyond a plain 2D view the host's gather reads level 0: neither
/// honours the sampler the branch chose. No gather takes one.
#[test]
fn a_gather_with_a_chosen_immediate_sampler_is_refused() {
    for (label, sample) in [
        // image_gather4_l v12, v[2:4], s[24:31], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
        ("image_gather4_l", [0xF110_0108, 0x0086_0C02]),
        // image_gather4_l_o v12, v[1:4], the same descriptors
        ("image_gather4_l_o", [0xF150_0108, 0x0086_0C01]),
        // image_gather4_l v12, v[2:5], dim:SQ_RSRC_IMG_2D_ARRAY
        ("image_gather4_l of an array", [0xF110_0128, 0x0086_0C02]),
        // image_gather4_lz v12, v[2:3]
        ("image_gather4_lz", [0xF11C_0108, 0x0086_0C02]),
    ] {
        let why = refusal(compute(Chooser::Flag, POINT, sample, Around::Nothing));
        assert_eq!(
            why.as_deref(),
            Some("a gather with an immediate sampler chosen by control flow"),
            "{label}"
        );
    }
}

/// Of the samples only `image_sample` of a pixel program and `image_sample_lz` take a
/// chosen sampler: outside a pixel program `image_sample` reads level 0 by a shortcut,
/// and every other operation with such a sampler is refused.
#[test]
fn another_operation_with_a_chosen_immediate_sampler_is_refused() {
    const WHY: &str = "an immediate sampler chosen by control flow for other than image_sample of a pixel program or image_sample_lz";
    for (label, sample) in [
        ("image_sample outside a pixel program", SAMPLE),
        ("image_sample_l", [0xF090_0F08, 0x0086_0C02]),
        ("image_sample_b", [0xF094_0F08, 0x0086_0C01]),
        ("image_sample_d", [0xF088_0F08, 0x0086_0C02]),
        ("image_sample_c_lz", [0xF0BC_0108, 0x0086_0C01]),
        ("image_sample_lz_o", [0xF0DC_0F08, 0x0086_0C01]),
    ] {
        let why = refusal(compute(Chooser::Flag, POINT, sample, Around::Nothing));
        assert_eq!(why.as_deref(), Some(WHY), "{label}");
    }
    for (label, sample) in [
        ("image_sample_b of a pixel program", [0xF094_0F08, 0x0082_0400]),
        ("image_sample_c of a pixel program", [0xF0A0_0108, 0x0082_0400]),
    ] {
        let why = refusal(pixel_shader(Pixel::Other(sample), POINT));
        assert_eq!(why.as_deref(), Some(WHY), "{label}");
    }
    // image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D
    let why = refusal(pixel_shader(Pixel::Other([0xF180_0308, 0x0082_0400]), POINT));
    assert_eq!(
        why.as_deref(),
        Some("a level query with an immediate sampler chosen by control flow")
    );
}

/// A chosen sampler is lowered only for a 2D view with 32-bit normalised coordinates and
/// 32-bit results: an array, a 3D view, texel coordinates, or 16-bit coordinates or results
/// are refused.
#[test]
fn a_view_or_form_other_than_2d_normalised_32_bit_is_refused() {
    for (label, sample) in [
        ("a 2D array", [0xF09C_0F28, 0x0086_0C02]),
        ("a 3D view", [0xF09C_0F10, 0x0086_0C02]),
        ("texel coordinates", [0xF09C_1F08, 0x0086_0C02]),
        ("16-bit coordinates", [0xF09C_0F08, 0x4086_0C02]),
        ("16-bit results", [0xF09C_0F08, 0x8086_0C02]),
    ] {
        let why = refusal(compute(Chooser::Flag, POINT, sample, Around::Nothing));
        assert_eq!(
            why.as_deref(),
            Some("an immediate sampler chosen by control flow for other than a 2D view with 32-bit normalised coordinates and results"),
            "{label}"
        );
    }
}

/// Both samples would run for every lane that reaches them and the mask would only keep
/// the old register, so a chosen sampler under a lane mask is refused.
#[test]
fn a_chosen_immediate_sampler_under_a_lane_mask_is_refused() {
    let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Mask));
    assert_eq!(
        why.as_deref(),
        Some("an immediate sampler chosen by control flow for a sample under a lane mask")
    );
}

/// The two samplers may differ in clamping against wrapping and in nothing else: each
/// sample is made with both, so the one a lane does not take must be as legal for the
/// view as its own, and take the same level.
#[test]
fn immediate_samplers_that_differ_in_more_than_clamp_against_wrap_are_refused() {
    let wrap = |word: usize, bits: u32| {
        let mut other = [0; 4];
        other[word] = bits;
        [[CLAMP, 0, 0, 0], other]
    };
    for (label, pair) in [
        ("the filters", wrap(2, 0x0050_0000)),
        ("the mip filter", wrap(2, 0x0400_0000)),
        ("the comparison function", wrap(0, 3 << 12)),
        ("the anisotropy", wrap(0, 2 << 9)),
        ("the level bias", wrap(2, 0x80)),
        ("the level clamps", wrap(1, 0x00FF_F000)),
        ("the border colour", wrap(3, 0x4000_0000)),
        ("clamp against mirror", wrap(0, 0x49)),
        ("clamp in x alone", [[2, 0, 0, 0], [0; 4]]),
        ("clamp against clamp to the border", [[CLAMP, 0, 0, 0], [0x1B6, 0, 0, 0]]),
    ] {
        let why = refusal(compute(Chooser::Flag, pair, SAMPLE_LZ, Around::Nothing));
        assert_eq!(
            why.as_deref(),
            Some("immediate samplers chosen by control flow that are not one clamped and one wrapped and alike otherwise"),
            "{label}"
        );
    }
}

/// A sampler that forces unnormalised coordinates reads the coordinates as texels; the
/// host's static sampler has no such bit and the coordinates are made once, for both.
#[test]
fn an_immediate_sampler_that_forces_unnormalised_coordinates_is_refused() {
    const BIT: u32 = 1 << 15;
    for (label, pair) in [
        ("both", [[CLAMP | BIT, 0, 0, 0], [BIT, 0, 0, 0]]),
        ("the clamped", [[CLAMP | BIT, 0, 0, 0], [0; 4]]),
        ("the wrapped", [[CLAMP, 0, 0, 0], [BIT, 0, 0, 0]]),
    ] {
        let why = refusal(compute(Chooser::Flag, pair, SAMPLE_LZ, Around::Nothing));
        assert_eq!(
            why.as_deref(),
            Some("an immediate sampler chosen by control flow that forces unnormalised coordinates"),
            "{label}"
        );
    }
}

/// A static sampler is one binding by its four words, and a comparison sample makes it a
/// comparison sampler for every use: the plain sample the choice makes with it would then
/// be made with a comparison sampler.
#[test]
fn a_chosen_immediate_sampler_a_comparison_sample_uses_too_is_refused() {
    for (label, sampler) in [("the clamped", POINT[0]), ("the wrapped", POINT[1])] {
        let why = refusal(compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Compared(sampler)));
        assert_eq!(
            why.as_deref(),
            Some("an immediate sampler chosen by control flow that a comparison sample uses too"),
            "{label}"
        );
    }
    // Other words are another binding.
    let other = [CLAMP, 0, 0x0050_0000, 0];
    compute(Chooser::Flag, POINT, SAMPLE_LZ, Around::Compared(other)).expect("lower");
}
