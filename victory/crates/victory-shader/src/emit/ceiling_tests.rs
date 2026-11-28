//! The console compiler's unsigned 32-bit division, as it sizes work by a count: a group or
//! tile count `ceil(n / d)` written `(n + d - 1) / d`. The compiler expands the division into
//! a float reciprocal of `d` (`v_rcp_iflag_f32`, which the translation lowers as `1.0 / x`)
//! scaled to 2^32 and refined in integers, then two fix-up compares. Off by one at a multiple
//! of `d`, the last row of groups or tiles is never dispatched or walked, which is the shape of
//! the bottom strip (90 probe rows = 11 x 8 + 2). Each lane computes one count; every lane's
//! arithmetic is its own, so the program runs in any wave size, on the software adapter too.

use super::host::{Dispatch, Host, Root, View};
use super::wave_model::asm::{Asm, I, Lit, S, V};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Lane t: n = s1 + t, d = s0; stores (n + d - 1) / d by the console compiler's expansion of
/// an unsigned division (the LLVM AMDGPU form: reciprocal, one Newton step in integers, the
/// quotient estimate, two corrections).
fn ceiling() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, S(1), 0) // v_add_nc_u32 v10, s1, v0: n
        .vop2(0x25, 10, S(0), 10) // n + d
        .vop2(0x25, 10, I(-1), 10) // n + d - 1
        .vop1(1, 11, S(0)) // v11 = d
        .vop1(6, 1, V(11)) // v_cvt_f32_u32 v1, v11
        .vop1(0x2B, 1, V(1)) // v_rcp_iflag_f32 v1, v1
        .vop2(8, 1, Lit(0x4F7F_FFFE), 1) // v_mul_f32 v1, 0x4f7ffffe, v1
        .vop1(7, 1, V(1)) // v_cvt_u32_f32 v1, v1: about 2^32 / d
        .vop2(0x26, 2, I(0), 11) // v_sub_nc_u32 v2, 0, v11: -d
        .vop3(0x169, 2, V(2), V(1), I(0)) // v_mul_lo_u32 v2, v2, v1
        .vop3(0x16A, 2, V(1), V(2), I(0)) // v_mul_hi_u32 v2, v1, v2
        .vop2(0x25, 1, V(1), 2) // v_add_nc_u32 v1, v1, v2: the refined reciprocal
        .vop3(0x16A, 1, V(10), V(1), I(0)) // v_mul_hi_u32 v1, v10, v1: the quotient's estimate
        .vop3(0x169, 2, V(1), V(11), I(0)) // v_mul_lo_u32 v2, v1, v11
        .vop2(0x26, 2, V(10), 2) // v_sub_nc_u32 v2, v10, v2: the remainder
        .vop2(0x25, 4, I(1), 1) // v_add_nc_u32 v4, 1, v1
        .vopc(0xC3, V(11), 2) // v_cmp_le_u32 vcc, v11, v2
        .vop2(0x27, 5, V(11), 2) // v_subrev_nc_u32 v5, v11, v2
        .raw(&[1 << 25 | 1 << 17 | 4 << 9 | (256 + 1)]) // v_cndmask_b32 v1, v1, v4, vcc
        .raw(&[1 << 25 | 2 << 17 | 5 << 9 | (256 + 2)]) // v_cndmask_b32 v2, v2, v5, vcc
        .vop2(0x25, 4, I(1), 1) // v_add_nc_u32 v4, 1, v1
        .vopc(0xC3, V(11), 2) // v_cmp_le_u32 vcc, v11, v2
        .raw(&[1 << 25 | 1 << 17 | 4 << 9 | (256 + 1)]) // v_cndmask_b32 v1, v1, v4, vcc: the count
        .store(1, 3, 0)
        .endpgm();
    a.finish()
}

/// A compute program of 64 threads with root constants s0..s3 and a raw UAV at s4
/// (`wave64_tests`' layout).
fn native(words: &[u32]) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
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
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn translated(native: &NativeProgram, host_wave: u32) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = crate::dxbc::Container::parse(&bytes)?;
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let shader = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    shader
}

/// `ceil(n / d)` for 64 consecutive n from every start, at d = 8 and 16 (and 3, 12, 24, whose
/// reciprocals are not exact in binary), through windows that hold the probe grid's 88 and 90
/// rows, 1080 and 1440 lines: either side of each multiple. On the adapter the tests are given
/// (the card, or the software one with `VICTORY_WARP=1`).
#[test]
fn a_count_rounded_up_by_the_compilers_division_is_the_exact_ceiling() {
    let words = ceiling();
    let native = native(&words);
    let names: Vec<&str> = crate::gfx10::decode(&native.code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect();
    for want in ["v_cvt_f32_u32", "v_rcp_iflag_f32", "v_mul_f32", "v_cvt_u32_f32", "v_mul_lo_u32", "v_mul_hi_u32", "v_subrev_nc_u32", "v_cndmask_b32"] {
        assert!(names.contains(&want), "{want} is not among {names:?}");
    }
    let shader = translated(&native, 32).expect("translates");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the adapter does not take this program: {e}; not run");
            return;
        }
    };
    let mut wrong = Vec::new();
    for d in [8u32, 16, 3, 12, 24] {
        for start in [1u32, 60, 1050, 1400, 2530] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if let (RootType::Constants, 0, k @ 0..=1) = (u.root, u.slot, u.dword) {
                    ud[u.index as usize] = [d, start][k as usize];
                }
            }
            let out = host
                .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &[View::RawUav(64)], samplers: &[], groups: [1, 1, 1] })
                .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
            for (t, &got) in out[0].iter().enumerate() {
                let n = start + t as u32;
                let want = n.div_ceil(d);
                if got != want {
                    wrong.push(format!("ceil({n} / {d}): console {want}, ours {got}"));
                }
            }
        }
    }
    for w in wrong.iter().take(20) {
        eprintln!("{w}");
    }
    assert!(wrong.is_empty(), "{} counts differ (above, the first 20)", wrong.len());
    eprintln!("every count exact: d = 8, 16, 3, 12, 24, n in five windows of 64");
}
