//! Tests that came with picked pull requests: fxceless/victory-fork#714
//! (mesh barriers, SMEM UAVs, same-target edges, first-lane offsets) and
//! #726 (64-bit shifts without INT64). A child of `lower`'s tests, whose
//! helpers they use.

use super::*;

#[test]
fn shifts_64_bit_need_no_int64() {
    let sfi0 = |words: &[u32]| -> u64 {
        let constants = RootParam {
            sgprs: 4,
            ..root(RootType::Constants, 0, 0)
        };
        let emitted = lower_with(words, vec![constants], vec![]).unwrap();
        let container = crate::Container::parse(&emitted.dxil).unwrap();
        let features = container.part(crate::FourCC::SFI0).unwrap();
        u64::from_le_bytes(features.data.try_into().unwrap())
    };
    for (name, op) in [
        ("s_lshl_b64", 0x1Fu32),
        ("s_lshr_b64", 0x21),
        ("s_ashr_i64", 0x23),
    ] {
        let words = [
            0x8000_0000 | (op << 23) | (4 << 16) | (2 << 8),
            0x8506_8081, // s_cselect_b32 s6, 1, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0204, // v_mov_b32 v1, s4
            0x7E04_0205, // v_mov_b32 v2, s5
            0x7E06_0206, // v_mov_b32 v3, s6
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xD834_0000,
            0x0000_0200, // ds_write_b32 v0, v2
            0xD834_0000,
            0x0000_0300, // ds_write_b32 v0, v3
            S_ENDPGM,
        ];
        assert_eq!(sfi0(&words) & 0x8000, 0, "{name} must not require INT64");
    }
    for (name, op) in [
        ("v_lshlrev_b64", 0x2FFu32),
        ("v_lshrrev_b64", 0x300),
        ("v_ashrrev_i64", 0x301),
    ] {
        let words = [
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0200, // v_mov_b32 v1, s0
            0x7E04_0201, // v_mov_b32 v2, s1
            0x7E06_0202, // v_mov_b32 v3, s2
            0xD400_0004 | (op << 16),
            259 | (257 << 9), // v_<op> v[4:5], v3, v[1:2]
            0xD834_0000,
            0x0000_0400, // ds_write_b32 v0, v4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        assert_eq!(sfi0(&words) & 0x8000, 0, "{name} must not require INT64");
    }
}

#[test]
fn all_lane_mesh_gs_alloc_req_preserves_lds_barrier() {
    let lower_mesh = |words: &[u32]| -> Lowered {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 256,
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
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
        .unwrap()
    };

    // All-lane GS_ALLOC_REQ in entry block 0: no block is added to no_barrier.
    let all_lane = lower_mesh(&[
        0xBEFC_0380, // s_mov_b32 m0, 0
        0xBF90_0009, // s_sendmsg GS_ALLOC_REQ (id = 9)
        0xBF8A_0000, // s_barrier
        S_ENDPGM,
    ]);
    assert_eq!(
        all_lane.no_barrier_blocks, 0,
        "all-lane GS_ALLOC_REQ must not strip mesh LDS barriers"
    );

    // Divergent (single-lane) GS_ALLOC_REQ inside a per-lane branch: its
    // block (and anything it dominates) MUST still be in no_barrier.
    let divergent = lower_mesh(&[
        0xD4C2_0002,
        0x0002_0080, // v_cmp_eq_u32 s2, 0, v0
        0xBE83_3C02, // s_and_saveexec_b32 s3, s2
        0xBF88_0003, // s_cbranch_execz +3 (-> s_endpgm)
        0xBEFC_0380, // s_mov_b32 m0, 0
        0xBF90_0009, // s_sendmsg GS_ALLOC_REQ
        0xBF8A_0000, // s_barrier (single-lane, dominated by divergent alloc)
        S_ENDPGM,
    ]);
    assert_eq!(
        divergent.no_barrier_blocks, 1,
        "divergent single-lane GS_ALLOC_REQ must still suppress dominated barriers"
    );
}

#[test]
fn smem_uav_uses_raw_buffer_load_not_cbuffer() {
    // Root UAV at s[0:1] built into buffer descriptor s[4:7], read by
    // s_buffer_load_dword (sbase = 4/2 = 2) and stored to LDS.
    let root_words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0301, // s_mov_b32 s5, s1
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF420_0202,
        0xFA00_0000, // s_buffer_load_dword s8, s[4:7], 0x0
        0x7E02_0208, // v_mov_b32 v1, s8
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let root_layout = vec![root(RootType::Descriptor, 0, 0)];
    let root_used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 2,
        user_data: 0,
        slot: 0,
    }];
    let root_out = lower_with(&root_words, root_layout, root_used)
        .expect("root UAV read via s_buffer_load_dword must lower as raw UAV load");
    assert_eq!(root_out.root_uavs, [0]);
    assert!(root_out.root_cbvs.is_empty());
    assert_eq!(root_out.smem_raw_uavs, 1);

    // Table UAV at s0 loaded via s_load_dwordx4 into s[8:11], read via
    // s_buffer_load_dword (sbase = 8/2 = 4), and also queried for its
    // size dword (s10, exercising `descriptor_data` for a UAV read only by SMEM).
    let table_words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
        0x800D_0A0C, // s_add_u32 s13, s12, s10 (reads UAV descriptor dword 2 as data)
        0x7E02_020D, // v_mov_b32 v1, s13
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let table_layout = vec![root(RootType::Table, 0, 0)];
    let table_used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    // The review of job/land-2b: no raw host view is guaranteed for a table slot, so the
    // table UAV is refused (`review::a_table_uav_read_by_scalar_loads_is_refused`).
    let table_out = lower_with(&table_words, table_layout, table_used);
    assert!(
        table_out.as_ref().is_err_and(|e| e.to_string().contains("a writable buffer in a table read by scalar loads")),
        "{:?}",
        table_out.err()
    );
}

#[test]
fn same_target_conditional_branch_keeps_both_edge_blocks() {
    // Block 0 branches on s0 to block 2 (offset 20) or falls through to
    // block 1. Block 1 sets s2 = 1 and executes a conditional branch whose
    // taken and not-taken targets are BOTH block 2 (offset 20). Block 2
    // joins s2 with a PHI and writes it to LDS. Both same-target edges
    // from block 1 to block 2 must retain distinct entries in `edges`.
    let words = [
        0xBE82_0380, // 0:  s_mov_b32 s2, 0
        0xBF06_8000, // 4:  s_cmp_eq_u32 s0, 0
        0xBF85_0003, // 8:  s_cbranch_scc1 +3 (-> 24)
        0xBE82_0381, // 12: s_mov_b32 s2, 1
        0xBF06_8001, // 16: s_cmp_eq_u32 s1, 0
        0xBF85_0000, // 20: s_cbranch_scc1 +0 (taken == not_taken == 24)
        0x7E02_0202, // 24: v_mov_b32 v1, s2
        0xD834_0000,
        0x0000_0100, // 28: ds_write_b32 v0, v1
        S_ENDPGM,    // 36: s_endpgm
    ];
    let layout = vec![
        root(RootType::Constants, 0, 0),
        root(RootType::Constants, 1, 1),
    ];
    let out = lower_with(&words, layout, vec![]).unwrap();
    // Block 0 -> block 2 (1 edge) + block 1 -> block 2 (2 distinct edges) = 3 edge blocks.
    // Previously the second edge from block 1 overwrote the first (leaving only 2).
    assert_eq!(out.edge_blocks, 3);
}

#[test]
fn scalar_dynamic_offset_wraps_vgpr_input_and_phi_in_wave_read_lane_first() {
    // 1. Source::Built (root SRV descriptor) with dynamic byte offset taken
    //    from input VGPR v0 via v_readfirstlane_b32: built_offset must emit
    //    waveReadLaneFirst.i32 and set WAVE_OPS.
    let built_words = [
        0x7E08_0500, // v_readfirstlane_b32 s4, v0 (program input VGPR!)
        0x8004_0400, // s_add_u32 s4, s0, s4
        0x8205_8001, // s_addc_u32 s5, s1, 0
        0xBE86_03C1, // s_mov_b32 s6, -1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xE030_0000,
        0x8001_0100, // buffer_load_dword v1, off, s[4:7], 0
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Descriptor, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 2,
        user_data: 0,
        slot: 0,
    }];
    let built_out = lower_with(&built_words, layout, used).unwrap();
    assert!(built_out.wave_ops);
    // The review of job/land-2c: the offset is the `v_readfirstlane_b32`'s own result
    // (its first lane of EXEC), not a first lane taken again at the load.
    assert_eq!(
        built_out.scalar_wave_first, 0,
        "built_offset must use the first-lane read's own result"
    );

    // 2. code_smem (s_load_dword from PC-relative code table) with dynamic
    //    offset from a VGPR phi (v2 joined across a scalar branch) via
    //    v_readfirstlane_b32.
    let code_words = [
        0x7E04_0280, // 0:  v_mov_b32 v2, 0
        0xBF06_8000, // 4:  s_cmp_eq_u32 s0, 0
        0xBF85_0001, // 8:  s_cbranch_scc1 +1 (-> 16)
        0x7E04_0300, // 12: v_mov_b32 v2, v0
        0xBE84_1F00, // 16: s_getpc_b64 s[4:5] (PC = 20)
        0x7E0C_0502, // 20: v_readfirstlane_b32 s6, v2 (VGPR phi!)
        0x8004_0604, // 24: s_add_u32 s4, s4, s6
        0x8205_8005, // 28: s_addc_u32 s5, s5, 0
        0xF400_0202,
        0xFA00_0020, // 32: s_load_dword s8, s[4:5], 0x20 (code table at offset 52)
        0x7E02_0208, // 40: v_mov_b32 v1, s8
        0xD834_0000,
        0x0000_0100, // 44: ds_write_b32 v0, v1
        S_ENDPGM,    // 52: s_endpgm
        0x1234_5678, // 56: constant table entry
    ];
    let code_out =
        lower_with(&code_words, vec![root(RootType::Constants, 0, 0)], vec![]).unwrap();
    assert!(code_out.wave_ops);
    assert_eq!(
        code_out.scalar_wave_first, 0,
        "code_smem must use the first-lane read's own result"
    );
}
