//! A stage at the root signature's limit that gains the slot table
//! (`VICTORY_D16_INTEGER_VIEWS=1`, plan K5 step a; Sol's RED review of the
//! accuracy ports, Q4). With the switch a table-bound d16 typed load asks
//! for the slot table's root SRV (`SLOT_STRIDES`, 2 DWORDs); a stage whose
//! signature had 63 or 64 DWORDs without it would reach 65 or 66, past
//! D3D12's 64, and lose its pipeline. The builder binds the slot table as a
//! one-view table instead (1 DWORD) where the root descriptor does not fit;
//! where even that does not fit, the pipeline is refused by name and
//! counted ([`d16_tables_refused`]).
//!
//! The program is synthetic GFX10 (the d16 fixtures' of `victory-shader`):
//! its table at s0, a d16 load of entry 1's typed view stored through entry
//! 0's raw one. Its stage is brought to the size under test with title root
//! SRVs it does not read, each a one-view table of 1 DWORD as the title's
//! root SRVs are by default (`bounded_root_srvs`): the builder counts
//! DWORDs, not what the code reads. The translator reads the switch at each
//! translation, but the host once per process, so the switch-on cases run
//! as a child test in a process of their own; the parent runs the
//! switch-off case itself (the same sizes get their pipelines, as on
//! `world`).

use super::*;
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

/// The program: s[8:11] the raw view, s[12:15] the typed one; v1 the d16
/// load of element v0, stored at dword v0.
fn code() -> Vec<u32> {
    vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 0x20
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xE200_2000,
        0x8003_0100, // buffer_load_format_d16_x v1, v0, s[12:15], 0 idxen
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]
}

/// The program translated (with the switch as this process has it), and
/// what the console's list says it reads.
fn translated() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
    crate::host::test_host().expect("a host device (VICTORY_D3D12_WARP=1 for WARP)");
    crate::descriptor::host::probe_view_limit();
    let used: Vec<UsedResource> = [(ResourceType::Uav, 0), (ResourceType::Srv, 8)]
        .map(|(kind, table_offset_dwords)| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        })
        .to_vec();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 0, slot: 0 }],
        used: used.clone(),
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 1, threads: [32, 1, 1] }),
        code: code().iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let c = victory_shader::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let p = victory_shader::spill::build(&native).expect("the program decodes");
    let st = victory_shader::structure::structure(&p).expect("the program structures");
    let res = victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("traced");
    let io = victory_shader::io::raise(&p, &native).expect("its inputs");
    let host = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).expect("the program translates");
    (host, used)
}

/// Root-signature DWORDs of the stage as the builder lays it out by default:
/// its user data, 2 per root CBV and per one of our root SRVs, 1 per title
/// root SRV (a one-view table).
fn dwords(host: &victory_shader::emit::HostShader) -> u32 {
    let ours = |s: u8| s == victory_shader::emit::SLOT_STRIDES;
    host.user_data_dwords()
        + 2 * host.root_cbvs.len() as u32
        + host.root_srvs.iter().map(|&s| if ours(s) { 2 } else { 1 }).sum::<u32>()
}

/// The program's stage brought to `total` DWORDs without the slot table
/// with title root SRVs at slots 100 on, and its host pipeline.
fn at(total: u32) -> (u32, Option<HostPipeline>) {
    let (mut host, used) = translated();
    let strides = host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES);
    let without = dwords(&host) - if strides { 2 } else { 0 };
    assert!(without <= total, "the program alone has {without} DWORDs");
    host.root_srvs.extend((0..total - without).map(|k| 100 + k as u8));
    let with = dwords(&host);
    assert!(bounded_root_srvs(), "title root SRVs are one-view tables (VICTORY_ROOT_SRV_DIRECT unset)");
    let shader = crate::pipeline::test_translated_program(host, used);
    (with, HostPipeline::new(&[shader], None))
}

/// Whether stage 0 of `p` binds the slot table as a one-view table.
fn tabled(p: &HostPipeline) -> bool {
    p.stages[0].strides_table
}

/// Off: the program asks for no slot table, and a stage of 62, 63 or 64
/// DWORDs gets its pipeline, as on `world`. On: the child below.
#[test]
fn a_d16_stage_at_the_root_signature_limit_keeps_its_pipeline() {
    if std::env::var_os("VICTORY_D16_INTEGER_VIEWS").is_none() {
        for total in [62, 63, 64] {
            let (with, p) = at(total);
            assert_eq!(with, total, "switch off: no slot table");
            assert!(p.is_some(), "switch off: a stage of {total} DWORDs has its pipeline");
        }
    }
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "pipeline::host::d16_root_tests::a_d16_stage_at_the_root_signature_limit_with_the_switch_child",
            "--ignored",
            "--test-threads",
            "1",
            "--nocapture",
        ])
        .env("VICTORY_D16_INTEGER_VIEWS", "1")
        .env_remove("VICTORY_SLOT_MAP")
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "the child: {text}");
}

#[test]
#[ignore = "a child process of the test above (VICTORY_D16_INTEGER_VIEWS=1)"]
fn a_d16_stage_at_the_root_signature_limit_with_the_switch_child() {
    assert!(crate::descriptor::host::view_kind::on(), "run by its parent, with VICTORY_D16_INTEGER_VIEWS=1");
    // 62 without the table: the root descriptor fits (64).
    let (with, p) = at(62);
    assert_eq!(with, 64, "62 and the slot table's root descriptor");
    let p = p.expect("a stage of 62 DWORDs and the slot table (64) has its pipeline");
    assert!(!tabled(&p), "at 64 the slot table stays a root descriptor");
    // 63: the root descriptor (65) does not fit, the one-view table (64) does.
    let (with, p) = at(63);
    assert_eq!(with, 65);
    let p = p.expect("a stage of 63 DWORDs and the slot table has its pipeline (the slot table a one-view table)");
    assert!(tabled(&p), "at 63 the slot table is a one-view table");
    assert!(p.stages[0].root_srvs.iter().any(|&(s, _)| s == victory_shader::emit::SLOT_STRIDES));
    // 64: not even the one-view table fits (65): refused by name, counted.
    let before = d16_tables_refused();
    let (with, p) = at(64);
    assert_eq!(with, 66);
    assert!(p.is_none(), "a stage of 64 DWORDs cannot take the slot table at all");
    assert_eq!(d16_tables_refused(), before + 1, "the refusal is counted");
}
