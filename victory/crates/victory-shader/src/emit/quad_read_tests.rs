//! A pixel program's quad reads at a triangle's edge: GFX10 quad operations
//! (`quad_perm` DPP, `ds_swizzle_b32` quad mode) read the helper lanes of a
//! partial quad as any other lane, with the value the helper computed. A
//! wave read does not see helper lanes (SM 6.0 to 6.6); a quad read does.

use super::bitcode::{Attr, Term};
use super::container::{self, SigElement};
use super::host::{Draw, Host};
use super::shader::Shader;
use super::{HostShader, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::{io, ir, resource, structure};

/// v2 = TEXCOORD0.x; v3 = v2 of the lane across x in the quad (`quad_perm`
/// DPP), v4 = v2 of the lane across y (quad-mode `ds_swizzle_b32`); target 0
/// is (v2, v3, v4, v2).
const WORDS: &[u32] = &[
    0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
    0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
    0x7E06_02FA, // v_mov_b32_dpp v3, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
    0xFF00_B102,
    0xD8D4_804E, // ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,2,3,0,1)
    0x0400_0002,
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E0A_0302, // v_mov_b32 v5, v2
    0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
    0x0504_0302,
    0xBF81_0000, // s_endpgm
];

fn element(name: &str, kind: u8, interp: u8, row: u32) -> SigElement {
    SigElement {
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
    }
}

fn host_shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
    HostShader {
        stage,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    }
}

/// [`WORDS`] as [`pixel_program`] translates it.
fn pixel_shader() -> Result<HostShader, crate::ShaderError> {
    pixel_program(WORDS)
}

/// A wave32 pixel program interpolating TEXCOORD0 (attr 0, row 0) at the
/// pixel centre.
fn pixel_program(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b1111,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            // PERSP_CENTER: I and J in v0, v1
            input_addr: 1 << 1,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg1 = container::signature(&[element("TEXCOORD", 0, 0, 0)], true);
    let osg1 = container::signature(&[element("SV_Target", 16, 0, 0)], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    super::emit(&c, &native, &p, &st, &io, &res)
}

/// POSITION (row 0) to SV_Position (row 1), TEXCOORD0 (row 1) to TEXCOORD0
/// (row 0, the pixel program's row).
fn vertex_shader() -> HostShader {
    let mut s = Shader::new(false);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    for (from, to) in [(0u32, 1u32), (1, 0)] {
        for col in 0..4u8 {
            let args = [s.m.i32(from), s.m.i32(0), s.m.i8(col), undef];
            let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
            let out = [s.m.i32(to), s.m.i32(0), s.m.i8(col), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
        }
    }
    s.inputs.push(element("POSITION", 0, 0, 0));
    s.inputs.push(element("TEXCOORD", 0, 0, 1));
    s.outputs.push(element("TEXCOORD", 0, 2, 0));
    s.outputs.push(element("SV_Position", 3, 4, 1));
    s.m.terminate(b, Term::Ret);
    host_shader(HwStage::Vs, s.finish())
}

/// The interpolant, linear over the target in clip space (w is 1).
fn f(x: f32, y: f32) -> f32 {
    x + 4.0 * y + 8.0
}

/// The interpolant at the centre of texel (`px`, `py`) of the 8x8 target.
fn at(px: usize, py: usize) -> f32 {
    f((px as f32 + 0.5) / 4.0 - 1.0, 1.0 - (py as f32 + 0.5) / 4.0)
}

/// Both reads of the pixel program are quad reads: a wave read of a helper
/// lane is undefined in DXIL. (WARP's wave reads happen to see helper lanes,
/// so the draw below passes there either way; this is what fails on WARP.)
#[test]
fn pixel_quad_reads_are_quad_operations() {
    super::shader::take_ops();
    pixel_shader().expect("emit");
    let ops = super::shader::take_ops();
    let lane_reads: Vec<&String> = ops.iter().filter(|o| o.contains("ReadLane")).collect();
    assert!(
        !lane_reads.is_empty() && lane_reads.iter().all(|o| o.starts_with("quadReadLaneAt")),
        "{lane_reads:?}"
    );
}

/// The same patterns in compute stay wave reads: compute has no helper
/// lanes, and its lanes are the console's wave (`cross_lane`).
#[test]
fn compute_quad_patterns_stay_wave_reads() {
    use crate::native::ComputeIo;
    let words = [
        0x7E06_02FAu32, // v_mov_b32_dpp v3, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
        0xFF00_B100,
        0xD8D4_804E, // ds_swizzle_b32 v4, v0 offset:swizzle(QUAD_PERM,2,3,0,1)
        0x0400_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD834_0000, // ds_write_b32 v0, v3
        0x0000_0300,
        0xD834_0000, // ds_write_b32 v0, v4
        0x0000_0400,
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            // two user SGPRs, 512 bytes of LDS
            rsrc2: 1 << 15 | 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(&p, &native).expect("io");
    super::shader::take_ops();
    super::emit(&c, &native, &p, &st, &io, &res).expect("emit");
    let ops = super::shader::take_ops();
    let lane_reads: Vec<&String> = ops.iter().filter(|o| o.contains("ReadLane")).collect();
    assert!(
        lane_reads.iter().filter(|o| o.starts_with("waveReadLaneAt")).count() == 2
            && lane_reads.iter().all(|o| !o.starts_with("quad")),
        "{lane_reads:?}"
    );
}

/// A triangle whose long edge runs along the target's diagonal covers the
/// texels with x <= y: each quad on the diagonal has its top-right texel
/// uncovered, a helper lane. Its top-left texel reads it across x, its
/// bottom-right texel across y. Every covered texel must hold its quad
/// neighbours' interpolants, the helpers' included, as on the console.
#[test]
fn pixel_quad_reads_see_helper_lanes() {
    let ps = pixel_shader().expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut ps.dxil.clone()).expect("DXIL validation failed");
    }
    let vs = vertex_shader();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let vertex = |(x, y): (f32, f32)| [x, y, 0.5, 1.0, f(x, y), 0.0, 0.0, 1.0];
    let vertices = [(-1.0, 1.125), (-1.0, -1.0), (1.125, -1.0)].map(vertex).concat();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: graphics pipeline: {e}"));
        let d = Draw {
            roots: Vec::new(),
            textures: &[],
            samplers: &[],
            vertices: &vertices,
            stride: 32,
            size: (8, 8),
            mesh_groups: 0,
        };
        let texels = host.draw(&root, &pso, &d).unwrap_or_else(|e| panic!("{label}: draw: {e}"));
        let mut wrong = Vec::new();
        let (mut covered, mut from_helpers) = (0, 0);
        for (i, t) in texels.iter().enumerate() {
            let (px, py) = (i % 8, i / 8);
            if px > py {
                assert_eq!(t[3], 0.0, "{label}: texel ({px}, {py}) is outside the triangle");
                continue;
            }
            covered += 1;
            let want = [at(px, py), at(px ^ 1, py), at(px, py ^ 1), at(px, py)];
            from_helpers += usize::from(px ^ 1 > py) + usize::from(px > py ^ 1);
            if t.iter().zip(want).any(|(a, b)| (a - b).abs() > 1e-4) {
                wrong.push(format!("({px}, {py}): {t:?}, not {want:?}"));
            }
        }
        assert_eq!((covered, from_helpers), (36, 8), "{label}: the fixture's coverage");
        assert!(wrong.is_empty(), "{label}: {} texels wrong:\n{}", wrong.len(), wrong.join("\n"));
        eprintln!("{label}: quad reads saw the helper lanes");
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device; skipped");
    }
}

// ---- round two: source EXEC, swizzle modes, masks, lanes that are not there ----

const S_ENDPGM: u32 = 0xBF81_0000;
/// v2 = TEXCOORD0.x.
const INTERP: [u32; 2] = [
    0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
    0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
];
/// v7 = host lane & 1 (the quad's x), v3 = v4 = 4.0, VCC = lanes with v7 = 0.
const EVEN_LANES: [u32; 7] = [
    0xD765_0006, // v_mbcnt_lo_u32_b32 v6, -1, 0
    0x0001_00C1,
    0x360E_0C81, // v_and_b32 v7, 1, v6
    0x7E06_02F6, // v_mov_b32 v3, 4.0
    0x7E08_02F6, // v_mov_b32 v4, 4.0
    0x7D84_0E80, // v_cmp_eq_u32 vcc, 0, v7
    0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
];
/// ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,1,0,3,2), then its wait.
const SWIZZLE_ACROSS_X: [u32; 3] = [0xD8D4_80B1, 0x0400_0002, 0xBF8C_C07F];
/// exp mrt0 v2, v3, v4, v5 done vm (v5 = v2), s_endpgm.
const EXPORT: [u32; 4] = [0x7E0A_0302, 0xF800_180F, 0x0504_0302, S_ENDPGM];

/// The odd lanes leave EXEC (`s_and_saveexec`, no branch); the even ones read
/// their odd neighbour with `v_mov_b32_dpp v3, v2 quad_perm:[1,0,3,2]` (DPP
/// word `dpp`) and the quad-mode swizzle into v4; EXEC comes back.
fn masked_source(dpp: u32) -> Vec<u32> {
    let mut w = INTERP.to_vec();
    w.extend(EVEN_LANES);
    w.extend([0x7E06_02FA, dpp]);
    w.extend(SWIZZLE_ACROSS_X);
    w.push(0x887E_027E); // s_or_b32 exec_lo, exec_lo, s2
    w.extend(EXPORT);
    w
}

/// DPP words: quad_perm:[1,0,3,2] of v2, all rows and banks; `bound_ctrl` (bit
/// 19), `fi` (bit 18), neither.
const DPP_ZERO: u32 = 0xFF08_B102;
const DPP_FETCH: u32 = 0xFF04_B102;
const DPP_PLAIN: u32 = 0xFF00_B102;

/// Draws `ps` over the whole 8x8 target (no helper lanes) on each device.
fn draw_whole(ps: &HostShader, mut check: impl FnMut(&str, &[[f32; 4]])) {
    let vs = vertex_shader();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let vertex = |(x, y): (f32, f32)| [x, y, 0.5, 1.0, f(x, y), 0.0, 0.0, 1.0];
    let vertices = [(-1.0, -1.0), (-1.0, 3.0), (3.0, -1.0)].map(vertex).concat();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.pipeline(&vs, ps).unwrap_or_else(|e| panic!("{label}: pipeline: {e}"));
        let d = Draw {
            roots: Vec::new(),
            textures: &[],
            samplers: &[],
            vertices: &vertices,
            stride: 32,
            size: (8, 8),
            mesh_groups: 0,
        };
        check(label, &host.draw(&root, &pso, &d).unwrap_or_else(|e| panic!("{label}: draw: {e}")));
    }
}

/// A source lane outside EXEC: the swizzle reads 0, DPP with `bound_ctrl` 0,
/// DPP with `fi` the lane's value (GFX10, RDNA2 ISA 12.13.1 and the DPP field
/// table). The lanes outside EXEC keep 4.0. Shows on WARP.
#[test]
fn reads_of_a_lane_outside_exec_follow_the_console() {
    for (dpp, fetch) in [(DPP_ZERO, false), (DPP_FETCH, true)] {
        let ps = pixel_program(&masked_source(dpp)).expect("emit");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut ps.dxil.clone()).expect("DXIL validation failed");
        }
        draw_whole(&ps, |label, texels| {
            let mut wrong = Vec::new();
            for (i, t) in texels.iter().enumerate() {
                let (px, py) = (i % 8, i / 8);
                let want = if px & 1 == 1 {
                    [at(px, py), 4.0, 4.0, at(px, py)]
                } else {
                    let d = if fetch { at(px ^ 1, py) } else { 0.0 };
                    [at(px, py), d, 0.0, at(px, py)]
                };
                if t.iter().zip(want).any(|(a, b)| (a - b).abs() > 1e-4) {
                    wrong.push(format!("({px}, {py}): {t:?}, not {want:?}"));
                }
            }
            assert!(wrong.is_empty(), "{label}, fi {fetch}: {} texels wrong:\n{}", wrong.len(), wrong.join("\n"));
        });
    }
}

/// With neither `fi` nor `bound_ctrl` the readings differ (LLVM's GFX10 rules
/// keep the destination; RDNA2's field table reads 0): refused where a guard
/// may leave the source lane out, not guessed.
#[test]
fn a_plain_dpp_read_of_a_lane_outside_exec_is_refused() {
    let e = pixel_program(&masked_source(DPP_PLAIN)).err().map(|e| e.to_string());
    assert!(e.as_deref().is_some_and(|e| e.contains("without fi or bound_ctrl")), "{e:?}");
}

/// `ds_swizzle_b32` with bit 15 set is the quad mode only as `0x80xx`; rotate
/// (`0xC000`, lane 1 reads itself) and FFT (`0xE000`, lane 1 reads lane 16)
/// are refused, in pixel and compute alike.
#[test]
fn swizzle_rotate_and_fft_modes_are_refused() {
    use crate::native::ComputeIo;
    for offset in [0xC000u32, 0xE000] {
        let mut ps = INTERP.to_vec();
        ps.extend([0xD8D4_0000 | offset, 0x0400_0002, 0xBF8C_C07F]);
        ps.extend(EXPORT);
        let e = pixel_program(&ps).err().map(|e| e.to_string());
        assert!(e.as_deref().is_some_and(|e| e.contains("rotate or FFT")), "pixel {offset:#x}: {e:?}");
        let words = [
            0xD8D4_0000 | offset, // ds_swizzle_b32 v4, v0
            0x0400_0000,
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xD834_0000, // ds_write_b32 v0, v4
            0x0000_0400,
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 2 << 1,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let bytes = crate::dxbc::build(&[]);
        let c = Container::parse(&bytes).expect("container");
        let p = ir::build(&native.code, false, 8).expect("ir");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
        let io = io::raise(&p, &native).expect("io");
        let e = super::emit(&c, &native, &p, &st, &io, &res).err().map(|e| e.to_string());
        assert!(e.as_deref().is_some_and(|e| e.contains("rotate or FFT")), "compute {offset:#x}: {e:?}");
    }
}

/// DPP row and bank masks name console rows and banks (a bank is a quad); a
/// host quad sits elsewhere in its wave, so a pixel DPP with masks is refused.
#[test]
fn a_pixel_dpp_with_a_bank_mask_is_refused() {
    let mut w = INTERP.to_vec();
    w.extend([0x7E06_02FA, 0xF100_B102]); // v_mov_b32_dpp v3, v2 quad_perm:[1,0,3,2] bank_mask:0x1
    w.push(0x7E08_0302); // v_mov_b32 v4, v2
    w.extend(EXPORT);
    let e = pixel_program(&w).err().map(|e| e.to_string());
    assert!(e.as_deref().is_some_and(|e| e.contains("row or bank mask")), "{e:?}");
}

/// The lane reads `ops` names.
fn lane_reads(ops: &[String]) -> Vec<&str> {
    ops.iter().filter(|o| o.contains("ReadLane")).map(String::as_str).collect()
}

/// Inside a per-lane branch or a loop lanes leave at different times, the
/// quad's other lanes may not be there, and a quad read of an absent lane is
/// undefined: the read stays the wave read it was before quad reads (a
/// shortcut, the same as before for those sites), with no quad read.
#[test]
fn pixel_quad_reads_where_not_every_lane_runs_stay_wave_reads() {
    let mut branch = INTERP.to_vec();
    branch.extend(EVEN_LANES);
    branch.push(0xBF88_0003); // s_cbranch_execz over the swizzle
    branch.extend(SWIZZLE_ACROSS_X);
    branch.push(0x887E_027E); // s_or_b32 exec_lo, exec_lo, s2
    branch.extend(EXPORT);
    let mut lp = INTERP.to_vec();
    lp.extend(&EVEN_LANES[..5]);
    lp.extend([
        0x4A10_0E81, // v_add_nc_u32 v8, 1, v7 (one or two rounds)
        0xBE84_037E, // s_mov_b32 s4, exec_lo
    ]);
    lp.extend(SWIZZLE_ACROSS_X); // the loop
    lp.extend([
        0x4A10_10C1, // v_add_nc_u32 v8, -1, v8
        0x7D8A_1080, // v_cmp_ne_u32 vcc, 0, v8
        0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
        0xBF89_FFF9, // s_cbranch_execnz to the loop
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
    ]);
    lp.extend(EXPORT);
    for (name, words) in [("branch", branch), ("loop", lp)] {
        super::shader::take_ops();
        let ps = pixel_program(&words).unwrap_or_else(|e| panic!("{name}: {e}"));
        let ops = super::shader::take_ops();
        let reads = lane_reads(&ops);
        assert!(
            !reads.is_empty() && reads.iter().all(|o| o.starts_with("waveReadLaneAt")),
            "{name}: {reads:?}"
        );
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut ps.dxil.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
