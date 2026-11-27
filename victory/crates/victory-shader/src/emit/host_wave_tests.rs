//! What a host of 32-lane waves is asked for. A compute program translated for such a host
//! must come out, and with a wave size the host can give: `dxil.dll` passes `WaveSize(64)`
//! and the host refuses the pipeline only when it is made, so neither the validator nor a
//! corpus list shows that loss. The size is read back from the emitted container (`PSV0`),
//! the bytes the host gets, after the whole of [`super::emit`] with its other forms of the
//! program. Every instruction word below is assembled for its test in the console's shape;
//! none is a dump of a title program.
//!
//! Translation only: no device is made, so nothing here is skipped on the software adapter.

use super::{HostCaps, HostWaves, WaveCheck, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// The most lanes a wave of the host in these tests has (`WaveLaneCountMax`).
const HOST_LANES: u32 = 32;

/// A wave64 compute program of `threads` threads (the thread index in v0), with or without
/// 512 bytes of LDS: a root SRV at s[0:1] (user data 0) and the table of the output UAV at
/// s2 (user data 2).
pub(super) fn native(words: &[u32], threads: u32, lds: bool) -> NativeProgram {
    native_wave(words, threads, lds, true)
}

/// [`native`] in console waves of 64 (`wave64`) or of 32.
fn native_wave(words: &[u32], threads: u32, lds: bool, wave64: bool) -> NativeProgram {
    let param = |root, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, root, dwords, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout: vec![param(RootType::Descriptor, 0, 0), param(RootType::Table, 2, 1)],
        used: vec![
            used(ResourceType::Srv, RootType::Descriptor, 2, 0, 0),
            used(ResourceType::Uav, RootType::Table, 4, 2, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: u32::from(lds) << 15 | 3 << 1, // 3 user SGPRs: s0..s2
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The wave size an emitted container requires of the host: `PSV0`'s fewest and most lanes
/// (runtime info bytes 16 and 20), the same number twice where the program fixes one, 0 and
/// all ones where it does not.
pub(super) fn required_wave(dxil: &[u8]) -> Option<u32> {
    let c = Container::parse(dxil).expect("the emitted container");
    let psv = c.part(FourCC::PSV0).expect("a PSV0 part").data;
    let u32_at = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().expect("4 bytes"));
    assert!(u32_at(0) >= 24, "runtime info with the wave's lane counts");
    let (min, max) = (u32_at(4 + 16), u32_at(4 + 20));
    match (min, max) {
        (0, u32::MAX) => None,
        (min, max) => {
            assert_eq!(min, max, "one wave size");
            Some(min)
        }
    }
}

/// `program` as the runtime translates it on a host whose waves have at most `host_lanes`
/// lanes (0: not known), by [`super::emit`] with every other form of the program it tries,
/// and the wave size the emitted container requires.
fn translated_for(host_lanes: u32, program: &NativeProgram) -> Result<Option<u32>, ShaderError> {
    Ok(required_wave(&emitted_for(host_lanes, program)?.dxil))
}

/// [`translated_for`]'s shader itself.
fn emitted_for(host_lanes: u32, program: &NativeProgram) -> Result<super::HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&program.code, program.wave64, program.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &program.layout, program.stage, Some(&st))?;
    let io = io::raise(&p, program)?;
    let was = lower::TEST_HOST_WAVE.replace(Some(host_lanes));
    let shader = super::emit(&c, program, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    shader
}

/// A console wave of 64 in a group of 64 with no LDS: `v1 = 4 * thread`, read into s4 by
/// `v_readfirstlane_b32` with every lane in EXEC and used as the root SRV's byte offset; the
/// word read goes to the output at the thread's own offset. Nothing else in it is sensitive
/// to lanes. (The two average-luminance downsample passes are first-lane offsets declared
/// this way: no wave size.)
pub(super) const FIRST_LANE_OFFSET: [u32; 15] = [
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

/// [`FIRST_LANE_OFFSET`] behind a lane permute with a constant selector, which stays inside
/// a half of 32 lanes: the program runs in the console's lanes, each console wave as two
/// host waves of 32 with no exchange between them. (The two probe integrations are declared
/// this way: waves of 32.)
const FIRST_LANE_OFFSET_IN_CONSOLE_LANES: [u32; 17] = [
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

/// A first-lane offset in a console wave of 64 that a host of 32-lane waves runs as two
/// waves: the program is translated, and asks for no wave wider than the host's. Alone it
/// fixes no wave size; in the console's lanes it asks for waves of 32, two to a console
/// wave. (Each host wave then reads its own first lane, the shortcut
/// `VICTORY_FIRST_LANE_STRICT=1` refuses.)
#[test]
fn a_first_lane_offset_in_a_console_wave_over_two_host_waves_asks_for_no_wider_wave() {
    let shapes: [(&str, &[u32], Option<u32>); 2] = [
        ("alone", &FIRST_LANE_OFFSET, None),
        ("in the console's lanes", &FIRST_LANE_OFFSET_IN_CONSOLE_LANES, Some(32)),
    ];
    for (name, words, declared) in shapes {
        let wave = translated_for(HOST_LANES, &native(words, 64, false))
            .unwrap_or_else(|e| panic!("{name}: not translated for a host of {HOST_LANES}-lane waves: {e}"));
        assert!(
            wave.is_none_or(|n| n <= HOST_LANES),
            "{name}: emitted with WaveSize({}) for a host of {HOST_LANES}-lane waves",
            wave.unwrap_or(0)
        );
        assert_eq!(wave, declared, "{name}: the wave size it declares");
    }
}

/// 64 threads in one console wave with LDS. Inside `if (thread < 8) { ... }` the threads
/// write their own LDS word and read it back, and s6 is changed; s6 is read after the join.
/// Two host waves cannot keep the console's order of an LDS write and a read inside one lane
/// branch (`lower::lds_meetings`), and no other form of the program is without the branch
/// (the code after it sees s6, so [`crate::ir::without_skips`] keeps it): the lowering keeps
/// the console's wave of 64.
const LDS_WRITTEN_AND_READ_IN_A_LANE_BRANCH: [u32; 19] = [
    0xBE86_0380, // s_mov_b32 s6, 0
    0x7D88_0088, // v_cmp_gt_u32 vcc, 8, v0
    0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    0xBF88_0005, // s_cbranch_execz +5
    0xD834_0000,
    0x0000_0000, // ds_write_b32 v0, v0
    0xD8D8_0000,
    0x0100_0000, // ds_read_b32 v1, v0
    0xBE86_0385, // s_mov_b32 s6, 5
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    0x7E04_0206, // v_mov_b32 v2, s6
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xBE8E_0302, // s_mov_b32 s14, s2
    0xBE8F_0380, // s_mov_b32 s15, 0
    0xF408_0407,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[14:15], 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    S_ENDPGM,
];

/// The form [`super::emit`] ends on is checked against the host: a program that still needs
/// the console's wave of 64 after every other form was tried is refused for a host of
/// 32-lane waves, with a reason of its own, and is not handed over as `WaveSize(64)` for the
/// host to refuse when the pipeline is made. Where the host's waves reach 64 lanes, or are
/// not known, it is emitted with the console's.
#[test]
fn a_wave_the_host_cannot_give_is_refused() {
    let program = native(&LDS_WRITTEN_AND_READ_IN_A_LANE_BRANCH, 64, true);
    for host in [0, 64] {
        let wave = translated_for(host, &program).unwrap_or_else(|e| panic!("host {host}: {e}"));
        assert_eq!(wave, Some(64), "host {host}: the console's wave");
    }
    match translated_for(HOST_LANES, &program) {
        Ok(wave) => panic!("emitted with wave size {wave:?} for a host of {HOST_LANES}-lane waves"),
        Err(e) => assert!(e.to_string().contains("a wave wider than the host's widest"), "{e}"),
    }
}

/// Console waves of 64 with LDS. s[8:9] holds a number, and one arm of a uniform branch puts
/// a lane mask there and sets the flag v1; behind the join only the lanes whose flag is set
/// narrow EXEC by s[8:9]. As written the join gives that mask a number on one edge, a
/// lane-constant mask, which asks for the console's lanes; with the edge that brings the
/// number threaded past the join ([`super::threading`]) the mask is the arm's alone and
/// nothing in the program asks for them. The LDS written and read in a lane branch, as in
/// [`LDS_WRITTEN_AND_READ_IN_A_LANE_BRANCH`], keeps every other form of the program at the
/// console's wave of 64 wherever its lanes or its one wave are asked for.
const A_LANE_CONSTANT_MASK_ON_AN_EDGE_THREADING_REMOVES: [u32; 37] = [
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBE88_0387, // s_mov_b32 s8, 7
    0xBE89_0380, // s_mov_b32 s9, 0
    0xBE86_0380, // s_mov_b32 s6, 0
    0xBF06_8002, // s_cmp_eq_u32 s2, 0
    0xBF85_0003, // s_cbranch_scc1 +3 (to the join)
    0x7D88_0088, // v_cmp_gt_u32 vcc, 8, v0
    0xBE88_046A, // s_mov_b64 s[8:9], vcc
    0x7E02_0281, // v_mov_b32 v1, 1
    0x7D84_0280, // the join: v_cmp_eq_u32 vcc, 0, v1
    0xBF87_0005, // s_cbranch_vccnz +5
    0x7E06_0281, // v_mov_b32 v3, 1
    0xBE8A_2408, // s_and_saveexec_b64 s[10:11], s[8:9]
    0x7E06_0285, // v_mov_b32 v3, 5
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBF82_0001, // s_branch +1
    0x7E06_0293, // v_mov_b32 v3, 19
    0x7D88_0088, // v_cmp_gt_u32 vcc, 8, v0
    0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    0xBF88_0005, // s_cbranch_execz +5
    0xD834_0000,
    0x0000_0000, // ds_write_b32 v0, v0
    0xD8D8_0000,
    0x0500_0000, // ds_read_b32 v5, v0
    0xBE86_0385, // s_mov_b32 s6, 5
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    0x7E04_0206, // v_mov_b32 v2, s6
    0x3408_0083, // v_lshlrev_b32 v4, 3, v0
    0xBE8E_0302, // s_mov_b32 s14, s2
    0xBE8F_0380, // s_mov_b32 s15, 0
    0xF408_0407,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[14:15], 0
    0xE070_1000,
    0x8004_0304, // buffer_store_dword v3, v4, s[16:19], 0 offen
    0xE070_1004,
    0x8004_0204, // buffer_store_dword v2, v4, s[16:19], 4 offen
    S_ENDPGM,
];

/// The recheck of job/land-2e, Question 2: a form that translates but asks for a wave wider
/// than the host's widest does not end the selection. Such a program used to be handed to
/// [`super::emit`]'s check as it was, the form with its constant edges threaded never tried
/// (it was tried for refused programs only), and refused for its width.
///
/// In a group of two console waves the threaded form asks for no wave size (LDS in a group
/// of several waves asks for none): it is the one a host of 32-lane waves gets, and a host
/// of 4-lane waves. Where the host's waves are not known the form as written stays, with the
/// console's 64. In a group of one console wave the LDS asks for that wave with the edge
/// threaded too: no form fits, and the refusal for the width stays.
#[test]
fn a_form_too_wide_for_the_host_gives_way_to_the_one_with_its_edges_threaded() {
    let two_waves = native(&A_LANE_CONSTANT_MASK_ON_AN_EDGE_THREADING_REMOVES, 128, true);
    let wave = translated_for(0, &two_waves).unwrap_or_else(|e| panic!("host unknown: {e}"));
    assert_eq!(wave, Some(64), "host unknown: as written, the console's wave");
    for host in [HOST_LANES, 4] {
        let wave = translated_for(host, &two_waves).unwrap_or_else(|e| panic!("host {host}: {e}"));
        assert_eq!(wave, None, "host {host}: the threaded form");
    }
    let one_wave = native(&A_LANE_CONSTANT_MASK_ON_AN_EDGE_THREADING_REMOVES, 64, true);
    for host in [HOST_LANES, 4] {
        match translated_for(host, &one_wave) {
            Ok(wave) => panic!("host {host}: emitted with wave size {wave:?}"),
            Err(e) => assert!(e.to_string().contains("a wave wider than the host's widest"), "host {host}: {e}"),
        }
    }
}

/// `program` translated for `host`, a snapshot of a device's answer passed to
/// [`super::emit_for`] (no host of the process or of the test's thread is read): the wave
/// size the emitted container requires and what the check found.
fn translated_as(host: HostCaps, program: &NativeProgram) -> Result<(Option<u32>, WaveCheck), ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&program.code, program.wave64, program.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &program.layout, program.stage, Some(&st))?;
    let io = io::raise(&p, program)?;
    let (shader, check) = super::emit_for(&c, program, &p, &st, &io, &res, host)?;
    Ok((required_wave(&shader.dxil), check))
}

/// A console wave of 32 in a group of 32 with a lane permute, which asks for the console's
/// lanes: waves of 32, whatever the host's are. The thread's own index goes to the output.
const A_LANE_PERMUTE_IN_A_CONSOLE_WAVE_OF_32: [u32; 10] = [
    0xD777_0005, // v_permlane16_b32 v5, v0, 0, 0
    0x0201_0100,
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0xE070_1000,
    0x8004_0501, // buffer_store_dword v5, v1, s[16:19], 0 offen
    S_ENDPGM,
];

/// The recheck of job/land-2e, Question 2: the device's answer is an input of the
/// translation, with both ends of its range and its wave operations. A program that fixes
/// waves of 32 is handed over where 32 is among the device's sizes, and refused where the
/// device's waves are all narrower (4 only, as the software adapter's here), all wider (64
/// only: the most lanes alone passed it), or it has no wave operations. Where no device has
/// answered it is handed over and said to be unchecked, not passed as one that fits.
#[test]
fn a_fixed_wave_is_checked_against_the_devices_whole_answer() {
    let program = native_wave(&A_LANE_PERMUTE_IN_A_CONSOLE_WAVE_OF_32, 32, false, false);
    let device = |min, max, wave_ops| HostCaps::Known(HostWaves { min, max, wave_ops });
    let refused = |host: HostCaps, why: &str| match translated_as(host, &program) {
        Ok((wave, check)) => panic!("{host:?}: emitted with wave size {wave:?} ({check:?})"),
        Err(e) => assert!(e.to_string().contains(why), "{host:?}: {e}"),
    };
    refused(device(4, 4, true), "a wave wider than the host's widest wave");
    refused(device(64, 64, true), "a wave narrower than the host's narrowest wave");
    refused(device(32, 32, false), "a wave size on a host without wave operations");
    for host in [device(32, 64, true), device(32, 32, true), device(8, 32, true)] {
        let got = translated_as(host, &program).unwrap_or_else(|e| panic!("{host:?}: {e}"));
        assert_eq!(got, (Some(32), WaveCheck::HostHas(32)), "{host:?}");
    }
    let got = translated_as(HostCaps::Unknown, &program).unwrap_or_else(|e| panic!("host unknown: {e}"));
    assert_eq!(got, (Some(32), WaveCheck::Unchecked(32)), "host unknown: said to be unchecked");
}

/// The snapshot passed in is the host of the whole emission, the lowering included: a
/// console wave of 64 with a first-lane offset in its lanes is split into waves of 32 for a
/// device whose widest wave has 32 lanes and keeps its 64 for one of 32 to 64, whatever host
/// the thread or the process holds meanwhile (here one of 4-lane waves, which would refuse
/// both).
#[test]
fn the_snapshot_passed_in_is_the_host_of_the_whole_emission() {
    let program = native(&FIRST_LANE_OFFSET_IN_CONSOLE_LANES, 64, false);
    let device = |min, max| HostCaps::Known(HostWaves { min, max, wave_ops: true });
    let was = lower::TEST_HOST_WAVE.replace(Some(4));
    let split = translated_as(device(32, 32), &program);
    let whole = translated_as(device(32, 64), &program);
    lower::TEST_HOST_WAVE.set(was);
    assert_eq!(split.unwrap_or_else(|e| panic!("32 only: {e}")), (Some(32), WaveCheck::HostHas(32)));
    assert_eq!(whole.unwrap_or_else(|e| panic!("32 to 64: {e}")), (Some(64), WaveCheck::HostHas(64)));
}

/// The sixth review's 1.2 (`lower_review_tests`'
/// `a_permute_selector_read_from_the_first_lane_keeps_the_split_by_default`): 64 threads in
/// one console wave. `s4 = readfirstlane(thread >> 5)` is the selector of
/// `v_permlane16_b32` over the thread indices (through an identity `ds_swizzle_b32`, which
/// puts the program in the console's lanes); each thread stores what it read at its own
/// word. Lanes 32 and 33 hold 32 and 33.
const A_PERMUTE_SELECTOR_FROM_THE_FIRST_LANE: [u32; 14] = [
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

/// The sixth review's 3.2 (`a_permute_selector_loaded_from_other_memory_keeps_the_split_by_
/// default`): a first-lane value is stored to the output, and the permute's selector is
/// word 0 of the root SRV, other memory, which holds no lane's value. The last store, of
/// what each thread read, is to the same word as the first.
const A_PERMUTE_SELECTOR_FROM_OTHER_MEMORY: [u32; 20] = [
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

/// `words` (a console wave of 64 in a group of 64, no LDS) as [`super::emit`] hands it to a
/// host of 32-lane waves, where it must ask for waves of 32, run once on the test host with
/// `srv` behind the root SRV: the 64 words of the output, one per thread. `None` on the
/// software adapter, whose waves have 4 lanes and which makes no pipeline of `WaveSize(32)`:
/// reported skipped there, after the emission and the validator have run.
#[cfg(windows)]
fn run_on_32_lane_waves(name: &str, words: &[u32], srv: [u32; 4]) -> Option<Vec<u32>> {
    use super::host::{Dispatch, Root, View};
    let program = native(words, 64, false);
    let shader = emitted_for(HOST_LANES, &program).unwrap_or_else(|e| panic!("{name}: not translated: {e}"));
    assert_eq!(required_wave(&shader.dxil), Some(HOST_LANES), "{name}: two host waves to a console wave");
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = super::valu_tests::hardware_host(name)?;
    let (root_sig, pso) = match host.compute_pipeline(&shader) {
        Ok(p) => p,
        Err(e) if std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") => {
            eprintln!("{name}: skipped on WARP ({e})");
            return None;
        }
        Err(e) => panic!("{name}: compute_pipeline: {e}"),
    };
    // (every user-data dword 0: the output's table is heap index 0)
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let mut roots = vec![(0, Root::Constants(ud))];
    for n in 0..shader.root_srvs.len() {
        roots.push((1 + n as u32, Root::View { srv: true, data: srv.to_vec() }));
    }
    let got = host
        .dispatch(&root_sig, &pso, &Dispatch { roots, views: &[View::RawUav(64)], samplers: &[], groups: [1, 1, 1] })
        .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
    Some(got[0].clone())
}

/// What `v_permlane16_b32` gives thread `t` of thread indices with the selector dword `s`
/// (and 0 above it): the index of lane `(s >> 4 * (t & 15)) & 15` of its row of 16.
#[cfg(windows)]
fn permuted(t: u32, s: u32) -> u32 {
    let pick = if t & 15 < 8 { s >> (4 * (t & 15)) & 15 } else { 0 };
    (t & !15) | pick
}

/// The recheck of job/land-2e, Question 1 (1.2), through [`super::emit`] and run. THIS
/// ASSERTS THE SHORTCUT'S ANSWER, NOT THE CONSOLE'S. The console's selector is its lane 0's,
/// `0 >> 5 = 0`, for all 64 lanes: lane 32 reads lane 32, the value 32. Each host half takes
/// the selector from its own first lane, so the upper half's is `32 >> 5 = 1` and lane 32
/// reads lane 33, the value 33 (lane 48 likewise 49). The exact lowering, the console wave's
/// first lane broadcast to both halves, is another job: when it lands this test fails, and
/// its expectation becomes `permuted(t, 0)` for every thread.
#[cfg(windows)]
#[test]
fn a_first_lane_permute_selector_gives_lane_32_lane_33s_value_by_the_shortcut_not_the_consoles_lane_32() {
    let name = "a permute selector from the first lane";
    let Some(got) = run_on_32_lane_waves(name, &A_PERMUTE_SELECTOR_FROM_THE_FIRST_LANE, [0; 4]) else {
        return;
    };
    let console: Vec<u32> = (0..64).map(|t| permuted(t, 0)).collect();
    let shortcut: Vec<u32> = (0..64).map(|t| permuted(t, t >> 5)).collect();
    assert_eq!((console[32], shortcut[32]), (32, 33), "the two answers for lane 32");
    assert_eq!(got[32], 33, "lane 32 reads lane 33's value (the console: lane 32's, 32)");
    assert_eq!(got, shortcut, "each host half by its own first lane's selector");
}

/// The same for 3.2, where the shortcut is exact: the selector comes from other memory and
/// is one value for both halves, here 1, so lane 32 reads lane 33 (and lane 0 lane 1) on the
/// console too. Only the first-lane analysis takes the selector for a first-lane value (a
/// store of one stands before the load).
#[cfg(windows)]
#[test]
fn a_permute_selector_loaded_from_other_memory_selects_the_consoles_lane() {
    let name = "a permute selector from other memory";
    let Some(got) = run_on_32_lane_waves(name, &A_PERMUTE_SELECTOR_FROM_OTHER_MEMORY, [1, 0, 0, 0]) else {
        return;
    };
    let console: Vec<u32> = (0..64).map(|t| permuted(t, 1)).collect();
    assert_eq!((console[0], console[32], console[33]), (1, 33, 32));
    assert_eq!(got, console, "one selector for both halves, as on the console");
}

/// [`LDS_WRITTEN_AND_READ_IN_A_LANE_BRANCH`] with what the recheck of job/land-2e found
/// wanting in it put right (Question 2): that fixture gives the thread's index, unscaled, to
/// byte-addressed LDS operations, so its eight lanes share dwords, and nothing reads what
/// its `ds_read_b32` loads. Here each thread of the branch (`lanes`: a compare of the
/// thread's index, or of its index in its half of 32) writes its index to its own dword,
/// byte `4 * thread`, and reads a dword back (`read`: the address register) into v1, which
/// is stored to the output beside s6 after the join.
fn lds_in_a_lane_branch(lanes: &[u32], read: u32) -> Vec<u32> {
    let mut words = vec![
        0x7E02_0280, // v_mov_b32 v1, 0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x3A0E_00A0, // v_xor_b32 v7, 32, v0 (the same lane of the other half)
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0xBE86_0380, // s_mov_b32 s6, 0
    ];
    words.extend_from_slice(lanes);
    words.extend_from_slice(&[
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0005, // s_cbranch_execz +5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0100_0000 | read, // ds_read_b32 v1, v4 (or v8)
        0xBE86_0385, // s_mov_b32 s6, 5
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E04_0206, // v_mov_b32 v2, s6
        0x3406_0083, // v_lshlrev_b32 v3, 3, v0
        0xBE8E_0302, // s_mov_b32 s14, s2
        0xBE8F_0380, // s_mov_b32 s15, 0
        0xF408_0407,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[14:15], 0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        0xE070_1004,
        0x8004_0103, // buffer_store_dword v1, v3, s[16:19], 4 offen
        S_ENDPGM,
    ]);
    words
}

/// What [`super::emit`] makes of an LDS program for a host whose waves are not known and
/// for one of 32-lane waves: the console's wave of 64, and the refusal for its width.
fn keeps_the_console_wave(name: &str, words: &[u32]) {
    let program = native(words, 64, true);
    let wave = translated_for(0, &program).unwrap_or_else(|e| panic!("{name}, host unknown: {e}"));
    assert_eq!(wave, Some(64), "{name}, host unknown: the console's wave");
    match translated_for(HOST_LANES, &program) {
        Ok(wave) => panic!("{name}: emitted with wave size {wave:?} for a host of {HOST_LANES}-lane waves"),
        Err(e) => assert!(e.to_string().contains("a wave wider than the host's widest"), "{name}: {e}"),
    }
}

/// The control. Threads 0 to 7 each write their own dword and read it back: no lane reads
/// what another wrote, so two host waves of 32 need not meet between the write and the read
/// and would compute what the console does. THE REFUSAL ASSERTED HERE IS THE RULE BEING
/// CONSERVATIVE, not a need of the program: `lower::lds_meetings` gives up on any lane
/// branch with both an LDS read and a write, whatever their addresses. A lowering that
/// proves the addresses lane-private (each `4 * thread`, from one register) may translate
/// this for a host of 32-lane waves; this test then changes with it.
#[test]
fn lane_private_lds_in_a_lane_branch_is_refused_for_its_width_though_its_halves_need_not_meet() {
    let words = lds_in_a_lane_branch(
        &[0x7D88_0088], // v_cmp_gt_u32 vcc, 8, v0
        4,
    );
    keeps_the_console_wave("each lane its own dword", &words);
}

/// The case the rule is for. Threads 0 to 7 and 32 to 39 (index in the half below 8) each
/// write their own dword, then read the dword of the same lane in the other half: on the
/// console every write of the wave is done before any read, and lane 0 reads the 32 lane 32
/// wrote, lane 32 the 0. That value is live (stored to the output). Two host waves that do
/// not meet between the write and the read may read before the other half has written, and
/// they cannot meet inside a lane branch: the console's wave of 64 is kept, and a host of
/// 32-lane waves refuses it. This one must not come to translate without a meeting.
#[test]
fn lds_read_from_the_other_half_in_a_lane_branch_keeps_the_console_wave() {
    let words = lds_in_a_lane_branch(
        &[
            0x360A_009F, // v_and_b32 v5, 31, v0
            0x7D88_0A88, // v_cmp_gt_u32 vcc, 8, v5
        ],
        8,
    );
    keeps_the_console_wave("a dword of the other half", &words);
}
