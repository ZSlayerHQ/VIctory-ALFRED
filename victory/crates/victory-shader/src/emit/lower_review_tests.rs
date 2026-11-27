//! The counterexamples of the review of job/land-2b (the shader picks report), each as
//! the smallest program that computes otherwise than the console: a refusal is asserted
//! where the lowering cannot match it. A child of `lower`'s tests, whose helpers it uses.
//!
//! Every instruction word below is assembled for its test in the console's shape; none is a
//! dump of a title program.

use super::*;

/// The reason a translation stopped, or `None` where it went through.
fn refusal(r: Result<Lowered, crate::ShaderError>) -> Option<String> {
    r.err().map(|e| e.to_string())
}

/// `v_lshrrev_b64 v[4:5], 32, 1.0`: an inline float constant as a 64-bit operand is the
/// 64-bit double (1.0 is 0x3ff00000_00000000), so the console's low word is 0x3ff00000.
/// The two-dword source took the 32-bit float's bits widened (0x00000000_3f800000) and
/// gave 0. Refused (#726).
#[test]
fn an_inline_float_in_a_64_bit_shift_is_refused() {
    let words = [
        0x7E00_0280, // v_mov_b32 v0, 0
        0xD700_0004, // v_lshrrev_b64 v[4:5], 32, 1.0
        160 | 242 << 9,
        0xD834_0000,
        0x0000_0400, // ds_write_b32 v0, v4
        S_ENDPGM,
    ];
    let why = refusal(lower_with(&words, vec![root(RootType::Constants, 0, 0)], vec![]));
    assert!(
        why.as_deref().is_some_and(|w| w.contains("an inline float constant as a 64-bit operand")),
        "{why:?}"
    );
}

/// A compute program of `threads` lanes in one wave32 group (`lower_wave`, wider).
fn lower_group(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> Result<Lowered, crate::ShaderError> {
    lower_group_wave(words, layout, used, threads, false)
}

/// [`lower_group`] in console waves of 64 when `wave64`, on a host whose waves are 32 lanes.
fn lower_group64(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> Result<Lowered, crate::ShaderError> {
    let was = super::super::TEST_HOST_WAVE.replace(Some(32));
    let r = lower_group_wave(words, layout, used, threads, true);
    super::super::TEST_HOST_WAVE.set(was);
    r
}

fn lower_group_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    lower_group_lds(words, layout, used, threads, wave64, true)
}

/// [`lower_group_wave`] with 512 bytes of LDS or none.
fn lower_group_lds(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
    wave64: bool,
    lds: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(lds) << 15 | sgprs.max(2) << 1,
            threads: [threads, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// The root SRV at s[0:1] (user data 0), as the review's programs read it.
fn root_srv() -> (Vec<RootParam>, Vec<UsedResource>) {
    let used = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 2,
        user_data: 0,
        slot: 0,
    };
    (vec![root(RootType::Descriptor, 0, 0)], vec![used])
}

/// Lanes 0 and 1 hold byte offsets 0 and 4; EXEC is lane 1 alone (no branch) at the
/// `v_readfirstlane_b32`, and whole again at the load through the root SRV built from it.
/// With words 11 and 22 there the console's lanes all read 22 (lane 1's offset). A first
/// lane taken at the load (#714) read 11; each lane's own offset (before #714) gave lane 0
/// 11 and lanes past 1 nothing. The `v_readfirstlane_b32`'s own result is used: 22.
const FIRST_LANE_UNDER_EXEC: [u32; 17] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBEFE_0382, // s_mov_b32 exec_lo, 2
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0x8004_0400, // s_add_u32 s4, s0, s4
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xBE8C_0302, // s_mov_b32 s12, s2 (the output's table)
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

/// The root SRV at s[0:1], and the output UAV's table at s2.
fn root_srv_and_output() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (mut layout, mut used) = root_srv();
    layout.push(root(RootType::Table, 2, 1));
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 2,
        slot: 1,
    });
    (layout, used)
}

#[test]
fn a_first_lane_offset_read_under_another_exec_takes_the_reads_own_result() {
    let (layout, used) = root_srv_and_output();
    let out = lower_group(&FIRST_LANE_UNDER_EXEC, layout, used, 32).expect("translates");
    assert_eq!(out.scalar_wave_first, 0, "no first lane taken at the load");
}

/// [`a_first_lane_offset_read_under_another_exec_takes_the_reads_own_result`] run: every
/// lane reads 22. The program asks for waves of 32, which WARP does not make (as for the
/// wave tests of `valu_tests`): there it is reported skipped.
#[cfg(windows)]
#[test]
fn a_first_lane_offset_read_under_another_exec_reads_what_the_console_reads() {
    use super::super::super::host::{Dispatch, Root, View};
    let (layout, used) = root_srv_and_output();
    let out = lower_group(&FIRST_LANE_UNDER_EXEC, layout, used, 32).expect("translates");
    let shader = super::super::super::HostShader {
        stage: HwStage::Cs,
        dxil: out.dxil,
        user_data: out.user_data,
        samplers: out.samplers,
        static_samplers: Vec::new(),
        root_cbvs: out.root_cbvs,
        root_srvs: out.root_srvs,
        root_uavs: out.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = super::super::super::validator::Validator::load(
        super::super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = super::super::super::valu_tests::hardware_host("first-lane offset under another EXEC") else {
        return;
    };
    let (root_sig, pso) = match host.compute_pipeline(&shader) {
        Ok(p) => p,
        Err(e) if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") => {
            eprintln!("first-lane offset under another EXEC: skipped on WARP ({e})");
            return;
        }
        Err(e) => panic!("compute_pipeline: {e}"),
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        // the output's table: heap index 0
        ud[u.index as usize] = 0;
    }
    let mut roots = vec![(0, Root::Constants(ud))];
    for (n, _) in shader.root_srvs.iter().enumerate() {
        roots.push((1 + n as u32, Root::View { srv: true, data: vec![11, 22, 0, 0] }));
    }
    let got = host
        .dispatch(&root_sig, &pso, &Dispatch { roots, views: &[View::RawUav(32)], samplers: &[], groups: [1, 1, 1] })
        .expect("dispatch");
    assert_eq!(got[0], vec![22; 32], "every lane reads the word at lane 1's offset");
}

/// The same program with an offset every lane holds alike (a root constant times 4, in a
/// VGPR): it still translates, through the read's own result.
#[test]
fn a_first_lane_offset_every_lane_holds_alike_translates() {
    let words = [
        0x7E06_0202, // v_mov_b32 v3, s2
        0x1602_0684, // v_mul_u32_u24 v1, 4, v3
        0xBEFE_0382, // s_mov_b32 exec_lo, 2
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0x8004_0400, // s_add_u32 s4, s0, s4
        0x8205_8001, // s_addc_u32 s5, s1, 0
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xE030_0000,
        0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xD834_0000,
        0x0000_0201, // ds_write_b32 v1, v2
        S_ENDPGM,
    ];
    let (mut layout, used) = root_srv();
    layout.push(root(RootType::Constants, 2, 1));
    let out = lower_group(&words, layout, used, 32).expect("a uniform first-lane offset");
    assert_eq!(out.scalar_wave_first, 0);
}

/// A writable buffer of 4-byte records in a table, read by `s_buffer_load_dword`. The
/// host's view of a table slot is raw only where its bytes start on 16
/// (`victory-gpu`'s `raw_window`): at first element 1 it stays structured, and a raw read
/// through it can lose the value (NVIDIA reads zeros). Nothing here can tell which view
/// the slot will hold, so the read is refused (#714). A root UAV has no view and stays.
#[test]
fn a_table_uav_read_by_scalar_loads_is_refused() {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
        0x7E02_020C, // v_mov_b32 v1, s12
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let why = refusal(lower_with(&words, vec![root(RootType::Table, 0, 0)], used));
    assert!(
        why.as_deref().is_some_and(|w| w.contains("a writable buffer in a table read by scalar loads")),
        "{why:?}"
    );
}

/// `s_round_mode 4` (doubles round up), then 1.0 + 2^-53 in doubles: the console's low word
/// is 1 (the sum rounds up to 1 + 2^-52), DXIL's round-to-nearest gives 0. Without the mode
/// write the same happens where the program's FLOAT_MODE starts the wave rounding up, and
/// that does not reach the translator: refused either way (the sibling fork's doubles
/// outside wave sections; the second review).
#[test]
fn a_double_whose_float_mode_is_not_known_is_refused() {
    let words = [
        0xBFA4_0004, // s_round_mode 4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_02FF, // v_mov_b32 v1, 0x3ca00000 (v[0:1] = 2^-53)
        0x3CA0_0000,
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7E0A_02FF, // v_mov_b32 v5, 0x3ff00000 (v[4:5] = 1.0)
        0x3FF0_0000,
        0xD564_0002, // v_add_f64 v[2:3], v[4:5], v[0:1]
        0x0002_0104,
        0x7E0C_0280, // v_mov_b32 v6, 0
        0xD834_0000,
        0x0000_0206, // ds_write_b32 v6, v2
        S_ENDPGM,
    ];
    let why = refusal(lower_with(&words, vec![root(RootType::Constants, 0, 0)], vec![]));
    assert!(why.as_deref().is_some_and(|w| w.contains("float mode is not known")), "{why:?}");
    let plain = refusal(lower_with(&words[1..], vec![root(RootType::Constants, 0, 0)], vec![]));
    assert!(plain.as_deref().is_some_and(|w| w.contains("float mode is not known")), "{plain:?}");
}

/// A table root at user data 0 and a root constant (an index) at user data 1.
fn table_and_index(stride: u32, dwords: u8) -> (Vec<RootParam>, Vec<UsedResource>) {
    let used = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    (vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)], vec![used])
}

/// Two image descriptors of one table at computed addresses: A (s[8:15]) is read only for
/// its height (dword 2), B (s[16:23], 32 bytes on) is loaded as a 1D image. With A a 2D
/// image of 8 x 4 and B a 1D image of 8 the console's height is 4; taking A's kind from
/// B gave a 1D view and a height of 1 (the sibling fork's size-only image lookup).
const ANOTHER_SLOTS_KIND: [u32; 17] = [
        0x8F02_8501, // s_lshl_b32 s2, s1, 5
        0x8004_0200, // s_add_u32 s4, s0, s2
        0xBE85_0380, // s_mov_b32 s5, 0
        0x8006_A004, // s_add_u32 s6, s4, 32
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], 0
        0xFA00_0000,
        0xF40C_0403, // s_load_dwordx8 s[16:23], s[6:7], 0
        0xFA00_0000,
        0x7E02_020A, // v_mov_b32 v1, s10 (A's dword 2)
        0xF000_0100, // image_load v2, v0, s[16:23] dmask:0x1 dim:1D
        0x0004_0200,
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
];

/// With `VICTORY_TWIN_STRICT=1` only a load of the same address tells a descriptor's kind,
/// so A is refused.
#[test]
fn a_descriptor_does_not_take_another_slots_kind() {
    let (layout, used) = table_and_index(8, 8);
    let was = super::super::TEST_TWIN_STRICT.replace(Some(true));
    let why = refusal(lower_with(&ANOTHER_SLOTS_KIND, layout, used));
    super::super::TEST_TWIN_STRICT.set(was);
    assert!(why.as_deref().is_some_and(|w| w.contains("a descriptor read only as data")), "{why:?}");
}

/// A buffer descriptor of records of 16 bytes, read only for its record count (dword 2).
/// The host's view of the slot is raw where its bytes start on 16 (`raw_window`) and
/// structured elsewhere (first element 1, stride 4, two records: the console's count is 2,
/// a raw query divided by the stride gives 0).
const A_RECORD_COUNT_ALONE: [u32; 8] = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
        0x7E02_020A, // v_mov_b32 v1, s10 (the record count)
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
];

/// With `VICTORY_TWIN_STRICT=1`, until the bound view's kind is known the count is refused
/// (the second review).
#[test]
fn a_record_count_is_refused_until_the_view_is_known() {
    let (mut layout, used) = table_and_index(16, 4);
    layout.truncate(1);
    let was = super::super::TEST_TWIN_STRICT.replace(Some(true));
    let why = refusal(lower_with(&A_RECORD_COUNT_ALONE, layout, used));
    super::super::TEST_TWIN_STRICT.set(was);
    assert!(why.as_deref().is_some_and(|w| w.contains("read as data")), "{why:?}");
}

/// A program whose only cross-lane instruction is a lane permute, every lane running: it
/// needs the console's lane layout (the wave size, lanes in group order), which the
/// detection did not see for `v_permlane*` (a debug assertion in `cross_lane`).
#[test]
fn a_permute_alone_asks_for_console_lanes() {
    let words = [
        0x7E00_0291, // v_mov_b32 v0, 17
        0xD778_0005, // v_permlanex16_b32 v5, v0, 0, 0
        0x0201_0100,
        0x7E02_0280, // v_mov_b32 v1, 0
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        S_ENDPGM,
    ];
    let out = lower_group(&words, vec![root(RootType::Constants, 0, 0)], vec![], 32).expect("a permute");
    assert_eq!(out.wave_size, Some(32));
}

/// v0 = 17 and v5 = 99 in every lane, then EXEC narrowed to lane 0 (no branch) and
/// `v_permlanex16_b32 v5, v0, 0, 0` with FI and BOUND_CTRL clear: lane 0's source, lane 16,
/// is outside EXEC, so the console keeps 99 in lane 0. The host still runs lane 16 and
/// lane 0 wrote 17. Refused until the controls are lowered.
#[test]
fn a_permute_with_a_source_lane_outside_exec_is_refused() {
    let words = [
        0x7E00_0291, // v_mov_b32 v0, 17
        0x7E0A_02FF, // v_mov_b32 v5, 99
        99,
        0xBEFE_0381, // s_mov_b32 exec_lo, 1
        0xD778_0005, // v_permlanex16_b32 v5, v0, 0, 0
        0x0201_0100,
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0x7E02_0280, // v_mov_b32 v1, 0
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        S_ENDPGM,
    ];
    let why = refusal(lower_group(&words, vec![root(RootType::Constants, 0, 0)], vec![], 32));
    assert!(why.as_deref().is_some_and(|w| w.contains("a lane permute where a source lane may not run")), "{why:?}");
}

/// The permute inside a lane branch (lane 0 alone enters): its source, lane 16, does not
/// run there. Refused as above.
#[test]
fn a_permute_in_a_lane_branch_is_refused() {
    let words = [
        0x7E02_0291, // v_mov_b32 v1, 17
        0xD4C2_0002, // v_cmp_eq_u32 s2, 0, v0
        0x0002_0080,
        0xBE83_3C02, // s_and_saveexec_b32 s3, s2
        0xBF88_0002, // s_cbranch_execz +2
        0xD778_0005, // v_permlanex16_b32 v5, v1, 0, 0
        0x0201_0101,
        0x887E_037E, // s_or_b32 exec_lo, exec_lo, s3
        0x7E0C_0280, // v_mov_b32 v6, 0
        0xD834_0000,
        0x0000_0506, // ds_write_b32 v6, v5
        S_ENDPGM,
    ];
    let why = refusal(lower_group(&words, vec![root(RootType::Constants, 0, 0)], vec![], 32));
    assert!(why.as_deref().is_some_and(|w| w.contains("a lane permute where a source lane may not run")), "{why:?}");
}

/// `v1 = 4 * lane`, then `v_fma_mixhi_f16 v1, 0, 0, 0` with every lane in EXEC: the high
/// half is 0 in every lane, the low half still the lane's own. `lane_uniform` took the
/// constants for the whole register; the kept half is an input too (the second review).
const MIXHI_OVER_A_LANE_VALUE: [u32; 19] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xCC22_0001, // v_fma_mixhi_f16 v1, 0, 0, 0
    0x0201_0080,
    0xBEFE_0382, // s_mov_b32 exec_lo, 2
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0x8004_0400, // s_add_u32 s4, s0, s4
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

#[test]
fn a_half_write_over_a_lane_value_is_not_uniform() {
    let code: Vec<u8> = MIXHI_OVER_A_LANE_VALUE.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let all = super::super::all_lane_blocks(&st).unwrap();
    let mixhi = &p.blocks[0].insts[1];
    assert_eq!(mixhi.name(), "v_fma_mixhi_f16");
    let v1 = mixhi.write(crate::ir::Reg::V(1)).unwrap();
    assert!(!super::super::lane_uniform(&p, &st, &all).contains(&v1), "v1 holds each lane's low half");
}

/// The same through the root SRV: the offset is the `v_readfirstlane_b32`'s own result
/// (lane 1's, under its EXEC), not a first lane taken at the load.
#[test]
fn a_half_write_over_a_lane_value_takes_the_reads_own_result() {
    let (layout, used) = root_srv_and_output();
    let out = lower_group(&MIXHI_OVER_A_LANE_VALUE, layout, used, 32).expect("translates");
    assert_eq!(out.scalar_wave_first, 0, "no first lane taken at the load");
}

/// Lanes 0 and 1 hold offsets 0 and 32 in one VGPR, read into s2 with EXEC lane 0 and into
/// s3 with EXEC lane 1: two slots of one table. A (s2) is read only for its height, B (s3)
/// loaded as a 1D image. Both addresses are the table plus that VGPR, but the first-lane
/// reads differ, so B tells nothing of A's kind (the second review).
const TWO_SLOTS_UNDER_TWO_EXECS: [u32; 22] = [
    0x3402_0085, // v_lshlrev_b32 v1, 5, v0
    0xBEFE_0381, // s_mov_b32 exec_lo, 1
    0x7E04_0501, // v_readfirstlane_b32 s2, v1
    0xBEFE_0382, // s_mov_b32 exec_lo, 2
    0x7E06_0501, // v_readfirstlane_b32 s3, v1
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0x8004_0200, // s_add_u32 s4, s0, s2
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8006_0300, // s_add_u32 s6, s0, s3
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], 0
    0xFA00_0000,
    0xF40C_0403, // s_load_dwordx8 s[16:23], s[6:7], 0
    0xFA00_0000,
    0x7E04_020A, // v_mov_b32 v2, s10 (A's dword 2)
    0xF000_0100, // image_load v3, v0, s[16:23] dmask:0x1 dim:1D
    0x0004_0300,
    0xD834_0000,
    0x0000_0201, // ds_write_b32 v1, v2
    0xD834_0000,
    0x0000_0301, // ds_write_b32 v1, v3
    S_ENDPGM,
];

/// [`TWO_SLOTS_UNDER_TWO_EXECS`] with `VICTORY_TWIN_STRICT=1`: refused.
#[test]
fn two_first_lane_reads_of_one_vgpr_are_two_slots() {
    let (mut layout, used) = table_and_index(8, 8);
    layout.truncate(1);
    let was = super::super::TEST_TWIN_STRICT.replace(Some(true));
    let why = refusal(lower_group(&TWO_SLOTS_UNDER_TWO_EXECS, layout, used, 32));
    super::super::TEST_TWIN_STRICT.set(was);
    assert!(why.as_deref().is_some_and(|w| w.contains("a descriptor read only as data")), "{why:?}");
}

/// The size-only descriptors that still take `f45cd882`'s shortcuts by default (counted, as
/// `Lowered::twin_hazards`), each refused with `VICTORY_TWIN_STRICT=1`: a twin by another
/// first-lane read, another slot's kind, a declared record count. Any fixture that takes
/// them translated before the landing, including three test-generated cases: two pixel
/// programs and one record count. This list may only shrink:
/// a fixture leaves it when its shortcut is proven or refused by default (the reviews of
/// job/land-2c-shader).
#[test]
fn size_only_shortcuts_only_shrink() {
    const PINNED: &[&str] = &[
        "two first-lane reads of one VGPR under two EXECs",
        "another slot's kind",
        "a record count alone",
    ];
    let was = super::super::TEST_TWIN_STRICT.replace(Some(false));
    let (mut slots, used8) = table_and_index(8, 8);
    let wide = slots.clone();
    slots.truncate(1);
    let (mut records, used4) = table_and_index(16, 4);
    records.truncate(1);
    let fixtures = [
        ("two first-lane reads of one VGPR under two EXECs", lower_group(&TWO_SLOTS_UNDER_TWO_EXECS, slots, used8.clone(), 32)),
        ("another slot's kind", lower_with(&ANOTHER_SLOTS_KIND, wide, used8)),
        ("a record count alone", lower_with(&A_RECORD_COUNT_ALONE, records, used4)),
    ];
    super::super::TEST_TWIN_STRICT.set(was);
    let hazards: Vec<&str> =
        fixtures.iter().filter(|(_, r)| r.as_ref().is_ok_and(|o| o.twin_hazards > 0)).map(|(n, _)| *n).collect();
    assert_eq!(hazards, PINNED, "the pinned list may only shrink");
}

/// A console wave of 64 in a group of 64 with no LDS, on a host of 32-lane waves:
/// `v1 = thread >> 5`, read into s4 by `v_readfirstlane_b32`, then
/// `v_permlane16_b32 v2, v0, s4, 0`. The console's selector is lane 0's, 0, for all 64
/// lanes (lane 32 writes 32); the upper host half reads its own first lane, 1 (lane 32
/// wrote 33). Refused with `VICTORY_FIRST_LANE_STRICT=1` (the second review; by default it
/// keeps the split it had before the landing,
/// [`a_permute_selector_read_from_the_first_lane_keeps_the_split_by_default`]).
#[test]
fn a_permute_selector_read_from_the_first_lane_of_a_split_wave_is_refused() {
    let words = [
        0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0xD777_0002, // v_permlane16_b32 v2, v0, s4, 0
        0x0200_0900,
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        S_ENDPGM,
    ];
    let (layout, used) = root_srv_and_output();
    let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(true));
    let why = refusal(lower_group64(&words, layout, used, 64));
    super::super::TEST_FIRST_LANE_STRICT.set(on);
    assert!(why.as_deref().is_some_and(|w| w.contains("selector a first-lane read gives")), "{why:?}");
}

/// A first-lane offset (`lane * 4`, every lane in EXEC) in a console wave of 64 split over
/// two host waves (a constant permute makes it run in console lanes, no exchange): the
/// console's offset is lane 0's, 0, for all 64 lanes; the upper host half would read lane
/// 32's, 128. Refused with `VICTORY_FIRST_LANE_STRICT=1`.
#[test]
fn a_first_lane_offset_in_a_split_wave_is_refused() {
    let words = [
        0xD777_0005, // v_permlane16_b32 v5, v0, 0, 0
        0x0201_0100,
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0x8004_0400, // s_add_u32 s4, s0, s4
        0x8205_8001, // s_addc_u32 s5, s1, 0
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xE030_0000,
        0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
        S_ENDPGM,
    ];
    // strict: refused; by default each host half reads its own first lane, as before the
    // landing (`f45cd882`), counted
    let (layout, used) = root_srv_and_output();
    let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(true));
    let why = refusal(lower_group64(&words, layout.clone(), used.clone(), 64));
    super::super::TEST_FIRST_LANE_STRICT.set(Some(false));
    let before = lower_group64(&words, layout, used, 64);
    super::super::TEST_FIRST_LANE_STRICT.set(on);
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
    assert!(before.expect("as before").first_lane_shortcuts > 0, "counted");
}

/// The same two reads under one EXEC (every lane): both scalars are lane 0's offset, one
/// slot, and B tells A's kind; it translates (the two offsets are provably equal under the
/// same execution mask).
#[test]
fn two_first_lane_reads_under_one_exec_are_one_slot() {
    let words = [
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0x7E04_0501, // v_readfirstlane_b32 s2, v1
        0x7E06_0501, // v_readfirstlane_b32 s3, v1
        0x8004_0200, // s_add_u32 s4, s0, s2
        0xBE85_0380, // s_mov_b32 s5, 0
        0x8006_0300, // s_add_u32 s6, s0, s3
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], 0
        0xFA00_0000,
        0xF40C_0403, // s_load_dwordx8 s[16:23], s[6:7], 0
        0xFA00_0000,
        0x7E04_020A, // v_mov_b32 v2, s10 (A's dword 2)
        0xF000_0100, // image_load v3, v0, s[16:23] dmask:0x1 dim:1D
        0x0004_0300,
        0xD834_0000,
        0x0000_0201, // ds_write_b32 v1, v2
        0xD834_0000,
        0x0000_0301, // ds_write_b32 v1, v3
        S_ENDPGM,
    ];
    let (mut layout, used) = table_and_index(8, 8);
    layout.truncate(1);
    let out = lower_group(&words, layout, used, 32).expect("one slot");
    assert_eq!(out.twin_hazards, 0, "the reads are alike: no shortcut");
}

/// The third review: a console wave of 64 in a group of 64 with no LDS reads `v1 = 4 *
/// thread` with `v_readfirstlane_b32` under full EXEC and uses it as a root SRV offset, with
/// no other lane-sensitive instruction. The console's offset is lane 0's, 0, for all 64
/// threads; host waves of 32 would give the upper one lane 32's, 128.
const FIRST_LANE_OFFSET_IN_WAVE64: [u32; 15] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0x8004_0400, // s_add_u32 s4, s0, s4
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

/// With `VICTORY_FIRST_LANE_STRICT=1` it asks for the console's waves of 64 (where the
/// host's are not known to be narrower), and on a host of 32-lane waves, where the console
/// wave would be split with no exchange, it is refused.
#[test]
fn a_first_lane_offset_asks_for_the_console_wave() {
    let (wave, why) = on_both_hosts(&FIRST_LANE_OFFSET_IN_WAVE64);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}

/// By default, as before the landing: no wave size asked for on either host, nothing
/// refused, counted.
#[test]
fn a_first_lane_offset_keeps_the_host_waves_by_default() {
    for host in [0, 32] {
        let (layout, used) = root_srv_and_output();
        let was = super::super::TEST_HOST_WAVE.replace(Some(host));
        let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(false));
        let out = lower_group_lds(&FIRST_LANE_OFFSET_IN_WAVE64, layout, used, 64, true, false);
        super::super::TEST_FIRST_LANE_STRICT.set(on);
        super::super::TEST_HOST_WAVE.set(was);
        let out = out.expect("as before");
        assert_eq!((out.wave_size, out.first_lane_shortcuts), (None, 1), "host {host}");
    }
}

/// The fourth review, first case: [`FIRST_LANE_OFFSET_IN_WAVE64`] with `s_and_b32 s4, s4, -4`
/// after the read. The offsets (0 and 128) are unchanged, but the address's term is now the
/// AND's scalar, with no first-lane tag.
const FIRST_LANE_OFFSET_BEHIND_AN_AND: [u32; 16] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0x8704_C404, // s_and_b32 s4, s4, -4
    0x8004_0400, // s_add_u32 s4, s0, s4
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

/// The dependence two scalar instructions and a phi away from the read: `s8 = (s4 & -4) + 0`,
/// or 0 where s2 (the output's table, independent) is 0; the root SRV at `s0 + s8`.
const FIRST_LANE_OFFSET_THROUGH_A_PHI: [u32; 20] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0x8704_C404, // s_and_b32 s4, s4, -4
    0x8008_8004, // s_add_u32 s8, s4, 0
    0xBF06_8002, // s_cmp_eq_u32 s2, 0
    0xBF85_0001, // s_cbranch_scc1 +1
    0xBE88_0380, // s_mov_b32 s8, 0
    0x8004_0800, // s_add_u32 s4, s0, s8
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

/// `words` in a console wave of 64, a group of 64 with no LDS, on a host of 32-lane waves,
/// with `VICTORY_FIRST_LANE_STRICT` as `strict` says.
fn on_32_lanes(words: &[u32], strict: bool) -> Result<Lowered, crate::ShaderError> {
    let (layout, used) = root_srv_and_output();
    let was = super::super::TEST_HOST_WAVE.replace(Some(32));
    let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(strict));
    let r = lower_group_lds(words, layout, used, 64, true, false);
    super::super::TEST_FIRST_LANE_STRICT.set(on);
    super::super::TEST_HOST_WAVE.set(was);
    r
}

/// `words` in a console wave of 64, a group of 64 with no LDS, strict: on a host whose waves
/// are not known (the wave size it declares) and on one of 32-lane waves (the refusal).
fn on_both_hosts(words: &[u32]) -> (Option<u32>, Option<String>) {
    let (layout, used) = root_srv_and_output();
    let was = super::super::TEST_HOST_WAVE.replace(Some(0));
    let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(true));
    let r = lower_group_lds(words, layout, used, 64, true, false);
    super::super::TEST_FIRST_LANE_STRICT.set(on);
    super::super::TEST_HOST_WAVE.set(was);
    (r.ok().and_then(|o| o.wave_size), refusal(on_32_lanes(words, true)))
}

/// The first-lane offsets that keep `f45cd882`'s lowering by default (counted,
/// `Lowered::first_lane_shortcuts`), each run in the console's lanes with
/// `VICTORY_FIRST_LANE_STRICT=1`: every compute program whose offsets depend on a first-lane
/// read keeps the host's waves and thread ids, the console's lanes not being shown to
/// compute what the host's do off the card (a console wave of 32), and on a host of
/// 32-lane waves a console wave of 64 that cannot be split would ask for 64-lane waves,
/// which that host refuses.
/// This list may only shrink: a fixture leaves it when its lowering is shown right or refused
/// by default (the fourth and fifth reviews of job/land-2c-shader).
#[test]
fn first_lane_offsets_kept_as_before_only_shrink() {
    const PINNED: &[&str] = &[
        "unwrapped by the tracker",
        "behind scalar arithmetic",
        "two scalar instructions and a phi away",
        "carried through memory",
        "a console wave of 32, unwrapped",
        "a console wave of 32, behind scalar arithmetic",
    ];
    let fixtures = [
        ("unwrapped by the tracker", on_32_lanes(&FIRST_LANE_OFFSET_IN_WAVE64, false)),
        ("behind scalar arithmetic", on_32_lanes(&FIRST_LANE_OFFSET_BEHIND_AN_AND, false)),
        ("two scalar instructions and a phi away", on_32_lanes(&FIRST_LANE_OFFSET_THROUGH_A_PHI, false)),
        ("carried through memory", on_32_lanes(&FIRST_LANE_OFFSET_THROUGH_MEMORY, false)),
        ("a console wave of 32, unwrapped", in_a_console_wave_of_32(&FIRST_LANE_OFFSET_IN_WAVE64, 32, false)),
        ("a console wave of 32, behind scalar arithmetic", in_a_console_wave_of_32(&FIRST_LANE_OFFSET_BEHIND_AN_AND, 32, false)),
    ];
    let taken: Vec<&str> =
        fixtures.iter().filter(|(_, r)| r.as_ref().is_ok_and(|o| o.first_lane_shortcuts > 0)).map(|(n, _)| *n).collect();
    assert_eq!(taken, PINNED, "the pinned list may only shrink");
}

#[test]
fn a_first_lane_offset_behind_scalar_arithmetic_asks_for_the_console_wave() {
    let (wave, why) = on_both_hosts(&FIRST_LANE_OFFSET_BEHIND_AN_AND);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}

#[test]
fn a_first_lane_offset_two_instructions_and_a_phi_away_asks_for_the_console_wave() {
    let (wave, why) = on_both_hosts(&FIRST_LANE_OFFSET_THROUGH_A_PHI);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}

/// The fourth review, second case: `s4 = readfirstlane(thread >> 5)`, `s5 = 0`, and a branch
/// on `s4 == 0` past `s5 = 1`; `v_permlane16_b32 v2, v0, s5, 0` at the join. The console's
/// s4 is 0 for all 64 lanes (selector 0: lane 32 reads 32); the upper host half's is 1
/// (selector 1: lane 32 would read 33). Refused in a split wave.
#[test]
fn a_permute_selector_a_first_lane_branch_chooses_is_refused_in_a_split_wave() {
    let words = [
        0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0xBE85_0381, // s_mov_b32 s5, 1
        0xD777_0002, // v_permlane16_b32 v2, v0, s5, 0
        0x0200_0B00,
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        S_ENDPGM,
    ];
    let (_, why) = on_both_hosts(&words);
    assert!(why.as_deref().is_some_and(|w| w.contains("selector a first-lane read gives")), "{why:?}");
}

/// The same selector chosen by a branch on s2 (the output's table: one value for every
/// thread, from no first-lane read), with a first-lane read elsewhere in the program: the
/// halves choose alike, and the split wave translates.
#[test]
fn a_permute_selector_an_independent_branch_chooses_translates_in_a_split_wave() {
    let words = [
        0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0xBE85_0381, // s_mov_b32 s5, 1
        0xD777_0002, // v_permlane16_b32 v2, v0, s5, 0
        0x0200_0B00,
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x7E08_0204, // v_mov_b32 v4, s4
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        S_ENDPGM,
    ];
    let (layout, used) = root_srv_and_output();
    let was = super::super::TEST_HOST_WAVE.replace(Some(32));
    let out = lower_group_lds(&words, layout, used, 64, true, false);
    super::super::TEST_HOST_WAVE.set(was);
    assert_eq!(out.expect("an independent selector").wave_size, Some(32));
}

/// `words` in a console wave of 32, a group of 70 threads (a count of this test's own: three
/// waves, the last one partial) with no LDS, on a host of `host`-lane waves (0: not known),
/// with `VICTORY_FIRST_LANE_STRICT` as `strict` says.
fn in_a_console_wave_of_32(words: &[u32], host: u32, strict: bool) -> Result<Lowered, crate::ShaderError> {
    let (layout, used) = root_srv_and_output();
    let was = super::super::TEST_HOST_WAVE.replace(Some(host));
    let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(strict));
    let r = lower_group_lds(words, layout, used, 70, false, false);
    super::super::TEST_FIRST_LANE_STRICT.set(on);
    super::super::TEST_HOST_WAVE.set(was);
    r
}

/// A console wave of 32 with a first-lane offset:
/// by default the lowering `f45cd882` gave it (no wave size, the host's thread ids), counted;
/// with the strict switch the console's lanes, in waves of its own 32 lanes on either host:
/// never a 64-lane wave, never a split, nothing refused (the fifth review).
#[test]
fn a_first_lane_offset_in_a_console_wave_of_32_keeps_its_own_wave() {
    for words in [&FIRST_LANE_OFFSET_IN_WAVE64[..], &FIRST_LANE_OFFSET_BEHIND_AN_AND[..]] {
        for host in [0, 32] {
            let before = in_a_console_wave_of_32(words, host, false).expect("as before");
            assert_eq!((before.wave_size, before.first_lane_shortcuts), (None, 1), "host {host}");
            let strict = in_a_console_wave_of_32(words, host, true).expect("its own wave");
            assert_eq!((strict.wave_size, strict.first_lane_shortcuts), (Some(32), 0), "host {host}");
        }
    }
}

/// The fifth review: a first-lane value carried through memory. `s4 = readfirstlane(4 *
/// thread)` is stored to the output UAV at byte 0, read back by `s_buffer_load_dword s20`
/// through the root view (whose memory may be the same), and used as the root SRV's offset.
/// The scalar load's result carried no mark, so the offset was lowered as any scalar: on a
/// host of 32-lane waves the upper half stores 128 where the console stores 0, and may read
/// it.
const FIRST_LANE_OFFSET_THROUGH_MEMORY: [u32; 25] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0x7E06_0204, // v_mov_b32 v3, s4
    0x7E08_0280, // v_mov_b32 v4, 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0304, // buffer_store_dword v3, v4, s[16:19], 0 offen
    0xBE88_0300, // s_mov_b32 s8, s0
    0xBE89_0301, // s_mov_b32 s9, s1
    0xBE8A_03C1, // s_mov_b32 s10, -1
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF420_0504,
    0xFA00_0000, // s_buffer_load_dword s20, s[8:11], 0
    0x8004_1400, // s_add_u32 s4, s0, s20
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    S_ENDPGM,
];

#[test]
fn a_first_lane_offset_carried_through_memory_asks_for_the_console_wave() {
    let (wave, why) = on_both_hosts(&FIRST_LANE_OFFSET_THROUGH_MEMORY);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}


/// The sixth review, 1.2: [`a_permute_selector_read_from_the_first_lane_of_a_split_wave_is_refused`]'s
/// program with an identity `ds_swizzle_b32` feeding the permute's data, so that the program
/// is in the console's lanes and split at `f45cd882` too. With every switch unset it keeps
/// that split: each host half takes the selector from its own first lane, as before the
/// landing, counted. `VICTORY_FIRST_LANE_STRICT=1` refuses it.
#[test]
fn a_permute_selector_read_from_the_first_lane_keeps_the_split_by_default() {
    let words = [
        0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0xD8D4_001F, // ds_swizzle_b32 v6, v0 (and 0x1f: each lane its own)
        0x0600_0000,
        0xD777_0002, // v_permlane16_b32 v2, v6, s4, 0
        0x0200_0906,
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        S_ENDPGM,
    ];
    let why = refusal(on_32_lanes(&words, true));
    assert!(why.as_deref().is_some_and(|w| w.contains("selector a first-lane read gives")), "{why:?}");
    let out = on_32_lanes(&words, false).expect("as before the landing");
    assert_eq!(out.wave_size, Some(32), "two host waves to a console wave");
    assert!(out.first_lane_shortcuts > 0, "counted");
}

/// The sixth review, 3.2: a first-lane value stored to the output UAV, then a selector loaded
/// by `s_buffer_load_dword` from the root SRV (other memory, holding no lane's value) and
/// given to `v_permlane16_b32`. What memory may carry marks every load of that class
/// ([`first_lane::dependents`]), so the selector counts as a first-lane read's; that must not
/// refuse by default a split the program had before the memory rule. Strict refuses it
/// (which store reaches which load is not bounded).
#[test]
fn a_permute_selector_loaded_from_other_memory_keeps_the_split_by_default() {
    let words = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0x7E06_0204, // v_mov_b32 v3, s4
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0xE070_1000,
        0x8004_0301, // buffer_store_dword v3, v1, s[16:19], 0 offen
        0xBE88_0300, // s_mov_b32 s8, s0
        0xBE89_0301, // s_mov_b32 s9, s1
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF420_0504,
        0xFA00_0000, // s_buffer_load_dword s20, s[8:11], 0
        0xD777_0002, // v_permlane16_b32 v2, v0, s20, 0
        0x0200_2900,
        0xE070_1000,
        0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
        S_ENDPGM,
    ];
    let why = refusal(on_32_lanes(&words, true));
    assert!(why.as_deref().is_some_and(|w| w.contains("selector a first-lane read gives")), "{why:?}");
    let out = on_32_lanes(&words, false).expect("as before the memory rule");
    assert_eq!(out.wave_size, Some(32), "two host waves to a console wave");
    assert!(out.first_lane_shortcuts > 0, "counted");
}

/// The sixth review, 3.1: a store that a first-lane branch chooses. `s4 =
/// readfirstlane(thread >> 5)`; where s4 is not 0, every thread stores the constant 1 to its
/// own word of the output UAV; then word 32 is read back by `s_buffer_load_dword s20` through
/// the root view (whose memory may be the same) and used as the root SRV's offset. The
/// console's s4 is lane 0's, 0: no store, and the word read is the 0 it held. The upper host
/// half's s4 is 1: it stores, and may read 1. The store's address, value and EXEC depend on
/// no first-lane read; the branch around it does.
const A_STORE_A_FIRST_LANE_BRANCH_CHOOSES: [u32; 26] = [
    0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
    0x7E08_0501, // v_readfirstlane_b32 s4, v1
    0x7E06_0281, // v_mov_b32 v3, 1
    0x3408_0082, // v_lshlrev_b32 v4, 2, v0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xBF06_8004, // s_cmp_eq_u32 s4, 0
    0xBF85_0002, // s_cbranch_scc1 +2
    0xE070_1000,
    0x8004_0304, // buffer_store_dword v3, v4, s[16:19], 0 offen
    0xBE88_0300, // s_mov_b32 s8, s0
    0xBE89_0301, // s_mov_b32 s9, s1
    0xBE8A_03C1, // s_mov_b32 s10, -1
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xF420_0504,
    0xFA00_0080, // s_buffer_load_dword s20, s[8:11], 0x80
    0x8004_1400, // s_add_u32 s4, s0, s20
    0x8205_8001, // s_addc_u32 s5, s1, 0
    0xBE86_03C1, // s_mov_b32 s6, -1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xE030_0000,
    0x8001_0200, // buffer_load_dword v2, off, s[4:7], 0
    0xE070_1000,
    0x8004_0204, // buffer_store_dword v2, v4, s[16:19], 0 offen
];

#[test]
fn a_store_a_first_lane_branch_chooses_marks_what_memory_carries() {
    let mut words = A_STORE_A_FIRST_LANE_BRANCH_CHOOSES.to_vec();
    words.push(S_ENDPGM);
    let (wave, why) = on_both_hosts(&words);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}

/// The recheck of job/land-2e, Question 1: the body of a loop closed by a plain jump, with no
/// way out of it but the translator's iteration limit (`after` is `s_branch` back to `body`'s
/// first word). `s4 = readfirstlane(thread >> 5)`, `v3 = 1`, `v4 = 4 * thread` and the output
/// UAV in s[16:19] are set before the loop; in it word 32 of the root view is read back by
/// `s_buffer_load_dword s20` and used as the root SRV's offset (s[24:27], s4 being the loop's
/// condition); the word there, s21, is the scalar offset of a last store of `v3`. No VGPR is
/// written in the loop: one written there is joined at the loop's head, where every branch
/// of the loop counts as choosing it, so a store of it is marked whatever the deciders say
/// (which hid the fault from the first form of this fixture).
fn in_a_closed_loop(body: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0x2C02_0085, // v_lshrrev_b32 v1, 5, v0
        0x7E08_0501, // v_readfirstlane_b32 s4, v1
        0x7E06_0281, // v_mov_b32 v3, 1
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    ];
    let head = words.len();
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xBE88_0300, // s_mov_b32 s8, s0
        0xBE89_0301, // s_mov_b32 s9, s1
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xF420_0504,
        0xFA00_0080, // s_buffer_load_dword s20, s[8:11], 0x80
        0x8018_1400, // s_add_u32 s24, s0, s20
        0x8219_8001, // s_addc_u32 s25, s1, 0
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_0380, // s_mov_b32 s27, 0
        0xF420_054C,
        0xFA00_0000, // s_buffer_load_dword s21, s[24:27], 0
        0xE070_1000,
        0x1504_0304, // buffer_store_dword v3, v4, s[16:19], s21 offen
    ]);
    let back = head as i32 - (words.len() as i32 + 1);
    words.push(0xBF82_0000 | (back as u32 & 0xFFFF)); // s_branch to the loop's first word
    words.push(S_ENDPGM);
    words
}

/// [`A_STORE_A_FIRST_LANE_BRANCH_CHOOSES`]'s conditional store and reload as the body of a
/// loop with no exit. In such a region no block reaches the end of the program, so no block
/// post-dominated another for [`first_lane::dependents`]' deciders (every set stayed whole)
/// and the walk from the marked branch stopped at both its successors: the store was not one
/// the branch chooses, the word read back carried no mark, and strict lowered the offset as
/// any scalar. The iteration limit is such a loop's way out.
#[test]
fn a_store_a_first_lane_branch_chooses_in_a_loop_without_an_exit_marks_what_memory_carries() {
    let words = in_a_closed_loop(&[
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0xBF85_0002, // s_cbranch_scc1 +2
        0xE070_1000,
        0x8004_0304, // buffer_store_dword v3, v4, s[16:19], 0 offen
    ]);
    let (wave, why) = on_both_hosts(&words);
    assert_eq!(wave, Some(64));
    assert!(why.as_deref().is_some_and(|w| w.contains("split over two host waves")), "{why:?}");
}

/// The control: the branch on the first-lane scalar skips one move (`v5`, read by nothing)
/// and the lanes are together again before the store, whose data (1), address (`4 * thread`)
/// and EXEC depend on no first-lane read. The branch does not choose that store, so the word
/// read back carries no mark and strict asks for no wave and refuses nothing: straight
/// through, and as the body of a loop with no exit (where the limit's way out must not make
/// every block of the loop one the branch chooses).
#[test]
fn a_store_after_the_lanes_of_a_first_lane_branch_rejoin_stays_unmarked() {
    let body = [
        0xBF06_8004, // s_cmp_eq_u32 s4, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xE070_1000,
        0x8004_0304, // buffer_store_dword v3, v4, s[16:19], 0 offen
    ];
    let mut straight = A_STORE_A_FIRST_LANE_BRANCH_CHOOSES[..8].to_vec();
    straight.extend_from_slice(&body);
    straight.extend_from_slice(&A_STORE_A_FIRST_LANE_BRANCH_CHOOSES[12..]);
    straight.push(S_ENDPGM);
    for (name, words) in [("straight through", straight), ("in a loop without an exit", in_a_closed_loop(&body))] {
        let (layout, used) = root_srv_and_output();
        let was = super::super::TEST_HOST_WAVE.replace(Some(0));
        let on = super::super::TEST_FIRST_LANE_STRICT.replace(Some(true));
        let unknown = lower_group_lds(&words, layout, used, 64, true, false);
        super::super::TEST_FIRST_LANE_STRICT.set(on);
        super::super::TEST_HOST_WAVE.set(was);
        let unknown = unknown.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((unknown.wave_size, unknown.first_lane_shortcuts), (None, 0), "{name}");
        let split = on_32_lanes(&words, true).unwrap_or_else(|e| panic!("{name}, a host of 32-lane waves: {e}"));
        assert_eq!((split.wave_size, split.first_lane_shortcuts), (None, 0), "{name}");
    }
}

/// The sixth review, 3.3: three loads of one table address (the table plus `32 * lane`). The
/// first, D1 (s[8:15]), behind a `v_readfirstlane_b32` under EXEC lane 0, is loaded as a 1D
/// image; the second, S (s[16:23]), behind a read under full EXEC, is read only for its
/// height; the third, D2 (s[24:31]), by S's own scalar, is loaded as a 2D image. D2 is proved
/// S's slot, D1 is not. Before the landing S took the kind of the first twin, D1's.
const A_SIZE_ONLY_DESCRIPTOR_WITH_TWO_TWINS: [u32; 27] = [
    0x3402_0085, // v_lshlrev_b32 v1, 5, v0
    0xBEFE_0381, // s_mov_b32 exec_lo, 1
    0x7E04_0501, // v_readfirstlane_b32 s2, v1
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0x7E06_0501, // v_readfirstlane_b32 s3, v1
    0x8004_0200, // s_add_u32 s4, s0, s2
    0xBE85_0380, // s_mov_b32 s5, 0
    0x8006_0300, // s_add_u32 s6, s0, s3
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], 0
    0xFA00_0000,
    0xF40C_0403, // s_load_dwordx8 s[16:23], s[6:7], 0
    0xFA00_0000,
    0xF40C_0603, // s_load_dwordx8 s[24:31], s[6:7], 0
    0xFA00_0000,
    0x7E04_0212, // v_mov_b32 v2, s18 (S's dword 2)
    0xF000_0100, // image_load v3, v0, s[8:15] dmask:0x1 dim:1D
    0x0002_0300,
    0xF000_0108, // image_load v4, v[0:1], s[24:31] dmask:0x1 dim:2D
    0x0006_0400,
    0xD834_0000,
    0x0000_0201, // ds_write_b32 v1, v2
    0xD834_0000,
    0x0000_0301, // ds_write_b32 v1, v3
    0xD834_0000,
    0x0000_0401, // ds_write_b32 v1, v4
    S_ENDPGM,
];

/// By default the size-only descriptor keeps the donor it had before the landing, the first
/// twin, whatever is proved of the others (a proof that prefers a later twin changes the
/// donor, and the view kind with it, whenever a mark changes what is proved): counted as a
/// twin by another first-lane read. With `VICTORY_TWIN_STRICT=1` the proved twin is taken.
#[test]
fn a_size_only_descriptor_keeps_its_first_twin_by_default() {
    let lowered = |strict: bool| {
        let (mut layout, used) = table_and_index(8, 8);
        layout.truncate(1);
        let was = super::super::TEST_TWIN_STRICT.replace(Some(strict));
        let out = lower_group(&A_SIZE_ONLY_DESCRIPTOR_WITH_TWO_TWINS, layout, used, 32);
        super::super::TEST_TWIN_STRICT.set(was);
        out
    };
    assert_eq!(lowered(false).expect("as before the landing").twin_hazards, 1, "the first twin, counted");
    assert_eq!(lowered(true).expect("the proved twin").twin_hazards, 0, "the proved twin");
}

/// The sixth review's own program for 3.3, which
/// [`A_SIZE_ONLY_DESCRIPTOR_WITH_TWO_TWINS`] left out: the index the three loads share is
/// made from a scalar reloaded after a store. `s4 = readfirstlane(4 * thread)`; a word is
/// stored at byte 0 of the output UAV (`stored`: that first-lane value, v7, or the constant
/// 0, v8); s48 is read back through the root view by `s_buffer_load_dword`; `v1 = s48 << 5`,
/// one value in every lane. Then the three loads of the image table plus v1, as there: D1
/// (s[8:15]) behind a `v_readfirstlane_b32` under EXEC lane 0, used as a 1D image; S
/// (s[16:23]) behind one under full EXEC, read only for its height; D2 (s[24:31]) by S's own
/// scalar, used as a 2D image.
///
/// With 0 stored nothing marks v1, a value every lane holds alike: D1 is proved S's slot.
/// With the first-lane value stored, what memory may carry marks s48 and v1 with it, D1's
/// proof is gone and only D2 is proved.
fn a_size_only_descriptor_behind_a_reload(marked: bool) -> Vec<u32> {
    let stored = if marked { 0x800A_0708 } else { 0x800A_0808 };
    let mut words = vec![
        0x340C_0082, // v_lshlrev_b32 v6, 2, v0
        0x7E08_0506, // v_readfirstlane_b32 s4, v6
        0x7E0E_0204, // v_mov_b32 v7, s4
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE8C_0303, // s_mov_b32 s12, s3 (the output's table)
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0A06,
        0xFA00_0000, // s_load_dwordx4 s[40:43], s[12:13], 0
        0xE070_1000,
        stored, // buffer_store_dword v7 (or v8), v8, s[40:43], 0 offen
        0xBEAC_0301, // s_mov_b32 s44, s1
        0xBEAD_0302, // s_mov_b32 s45, s2
        0xBEAE_03C1, // s_mov_b32 s46, -1
        0xBEAF_0380, // s_mov_b32 s47, 0
        0xF420_0C16,
        0xFA00_0000, // s_buffer_load_dword s48, s[44:47], 0
        0x7E12_0230, // v_mov_b32 v9, s48
        0x3402_1285, // v_lshlrev_b32 v1, 5, v9
    ];
    // (from `s_mov_b32 exec_lo, 1` on: the two first-lane reads and the three loads)
    words.extend_from_slice(&A_SIZE_ONLY_DESCRIPTOR_WITH_TWO_TWINS[1..]);
    words
}

/// The image table at s0, the root SRV at s[1:2] and the output UAV's table at s3.
fn image_table_root_srv_and_output() -> (Vec<RootParam>, Vec<UsedResource>) {
    let used = |kind, root, stride, dwords, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Descriptor, 1, 1), root(RootType::Table, 3, 2)],
        vec![
            used(ResourceType::Srv, RootType::Table, 8, 8, 0, 0),
            used(ResourceType::Srv, RootType::Descriptor, 0, 2, 1, 1),
            used(ResourceType::Uav, RootType::Table, 0, 4, 3, 2),
        ],
    )
}

/// The recheck of job/land-2e, Question 1 (3.3): the dimensions query of the size-only
/// descriptor, with the mark absent and present. By default it is the first twin's, a 1D
/// view, either way: the mark changes what is counted (the twin is then one by another
/// first-lane read) and not what is emitted. Before `e2c8abe0b` the default followed the
/// proof, and the mark turned the query into the 2D view's. With `VICTORY_TWIN_STRICT=1`
/// the proved twin is taken: D1 while nothing marks the index, D2 once the store does.
#[test]
fn a_mark_from_memory_does_not_change_a_size_only_descriptors_view_by_default() {
    const ONE_D: u32 = 1;
    const TWO_D: u32 = 2;
    let lowered = |marked: bool, strict: bool| {
        let (layout, used) = image_table_root_srv_and_output();
        let was = super::super::TEST_TWIN_STRICT.replace(Some(strict));
        let out = lower_group(&a_size_only_descriptor_behind_a_reload(marked), layout, used, 32);
        super::super::TEST_TWIN_STRICT.set(was);
        let out = out.unwrap_or_else(|e| panic!("marked {marked}, strict {strict}: {e}"));
        // (the three loads in program order: D1 by its own use, S by its twin, D2 by its own)
        assert_eq!(out.size_only_kinds.len(), 3, "marked {marked}, strict {strict}");
        assert_eq!((out.size_only_kinds[0], out.size_only_kinds[2]), (ONE_D, TWO_D));
        (out.size_only_kinds[1], out.twin_hazards)
    };
    assert_eq!(lowered(false, false), (ONE_D, 0), "no mark, default: the first twin, proved");
    assert_eq!(lowered(true, false), (ONE_D, 1), "marked, default: the first twin still, counted");
    assert_eq!(lowered(false, true), (ONE_D, 0), "no mark, strict: the first twin, proved");
    assert_eq!(lowered(true, true), (TWO_D, 0), "marked, strict: the twin by the same scalar");
}
