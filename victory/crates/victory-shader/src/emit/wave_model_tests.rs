//! The translation of one console wave of 64 lanes into two host waves of 32, checked lane by
//! lane against the CPU model of the console's wave ([`super::wave_model`]).
//!
//! Each fixture is one program, run by the model and, translated for a host of 32-lane waves
//! by [`super::emit`] (every form the runtime tries), on the host's adapter. Each program
//! holds every kind of cross-lane read the console waves of the ray-lighting and water passes
//! use, under lanes masked off or gone in one half only: the first-lane read
//! (`v_readfirstlane_b32`), lane reads by number (`v_readlane_b32` of lanes 0, 31, 32, 63),
//! a ballot read as a whole-wave count, first lane and "some lane" test (`s_bcnt1_i32_b64`,
//! `s_ff1_i32_b64`, SCC of `s_and_b64` with EXEC), and a reduction (the console compiler's
//! `WaveActiveBitOr`), with a value written under the narrowed EXEC (the lanes left out keep
//! their old one). Each lane stores what it got; a refusal is reported as one, not as a
//! disagreement.
//!
//! The software adapter has no 32-lane waves (G-127): there the fixtures translate, validate
//! and stop before the dispatch. Their verdict comes from the card.

use super::host::{Dispatch, Host, Root, View};
use super::wave_model::asm::{Asm, EXEC, I, Lit, Op, S, V, VCC};
use super::wave_model::{LANES, console};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// What each lane stores, a row of 64 dwords each (row k at byte 256 k).
const SLOTS: [&str; 10] = [
    "the value written under the narrowed EXEC (v10)",
    "first-lane read of v10",
    "lane read of lane 0",
    "lane read of lane 31",
    "lane read of lane 32",
    "lane read of lane 63",
    "ballot count (s_bcnt1_i32_b64)",
    "ballot's first lane (s_ff1_i32_b64)",
    "some lane of the ballot (SCC of s_and_b64 with EXEC)",
    "reduction: WaveActiveBitOr of 1 << (t >> 1)",
];

/// The scalar each slot from 1 on stores.
const SCALARS: [u8; 9] = [20, 21, 22, 23, 24, 25, 26, 27, 7];

/// `WaveActiveBitOr(v7)` into s7 as the console compiler expands it (the words of
/// `wave.rs`' `WAVE_OR_V7_TO_S7`, written out so the fixture does not lean on that file):
/// every lane takes part, those outside the saved EXEC (s[4:5]) with 0; v1 and v2 are used.
const WAVE_OR_V7_TO_S7: [u32; 20] = [
    0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
    0xD501_0001,
    0x0012_0E80, // v_cndmask_b32 v1, 0, v7, s[4:5]
    0x3802_02E9,
    0xDE54_C101, // v_or_b32 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
    0x3802_02E9,
    0x9360_1201, // v_or_b32 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
    0x3802_02E9,
    0x0009_2401, // v_or_b32 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
    0x3802_02FA,
    0xFF09_4001, // v_or_b32 v1, v1, v1 row_mirror bound_ctrl:1
    0xD778_0002,
    0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
    0x3802_0302, // v_or_b32 v1, v2, v1
    0xD760_0007,
    0x0001_4101, // v_readlane_b32 s7, v1, 32
    0x3802_0207, // v_or_b32 v1, s7, v1
    0xD760_0007,
    0x0001_0101, // v_readlane_b32 s7, v1, 0
    0xBEFE_0404, // s_mov_b64 exec, s[4:5]
];

/// The program's start: the UAV's descriptor into s[8:11] (as `wave64_tests`), v3 = 4 t,
/// v10 = 100 + t, v7 = 1 << (t >> 1), and every slot's scalar set to a value no kind gives.
fn entry_code(a: &mut Asm) {
    a.sop1(3, 5, I(0)) // s_mov_b32 s5, 0
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0) // v_add_nc_u32 v10, 100, v0
        .vop1(1, 5, I(1)) // v_mov_b32 v5, 1
        .vop2(0x16, 6, I(1), 0) // v_lshrrev_b32 v6, 1, v0
        .vop2(0x1A, 7, V(6), 5); // v_lshlrev_b32 v7, v6, v5
    for s in SCALARS {
        a.sop1(3, s, Lit(0xDEAD_0000 | u32::from(s)));
    }
}

/// Under the EXEC the caller narrowed: the masked write, then every kind of cross-lane read.
/// The ballot is of the lanes below `ballot`.
fn reads(a: &mut Asm, ballot: Op) {
    a.vop2(0x25, 10, Lit(1000), 0) // v_add_nc_u32 v10, 1000, v0: the lanes of EXEC only
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .vop3(0x360, 21, V(10), I(0), I(0)) // v_readlane_b32 s21, v10, 0
        .vop3(0x360, 22, V(10), I(31), I(0))
        .vop3(0x360, 23, V(10), I(32), I(0))
        .vop3(0x360, 24, V(10), I(63), I(0))
        .vop3(0xC4, 14, ballot, V(0), I(0)) // v_cmp_gt_u32 s[14:15], ballot, v0
        .sop1(0x10, 25, S(14)) // s_bcnt1_i32_b64 s25, s[14:15]
        .sop1(0x14, 26, S(14)) // s_ff1_i32_b64 s26, s[14:15]
        .sop2(0xF, 16, S(14), S(EXEC)) // s_and_b64 s[16:17], s[14:15], exec
        .sop2(0xA, 27, I(1), I(0)) // s_cselect_b32 s27, 1, 0
        .raw(&WAVE_OR_V7_TO_S7);
}

/// Every lane of EXEC stores v10 and the scalars, then the program ends.
fn epilogue(a: &mut Asm) {
    a.store(10, 3, 0);
    for (k, s) in SCALARS.into_iter().enumerate() {
        a.vop1(1, 2, S(s)).store(2, 3, 256 * (k as u32 + 1));
    }
    a.endpgm();
}

/// EXEC narrowed to the lanes s0 <= t < s1 with no branch (`v_cmpx`), then restored.
fn masked() -> Vec<u32> {
    let mut a = Asm::default();
    entry_code(&mut a);
    a.sop1(4, 12, S(EXEC)) // s_mov_b64 s[12:13], exec
        .vopc(0xD3, S(0), 0) // v_cmpx_le_u32 exec, s0, v0
        .vopc(0xD4, S(1), 0); // v_cmpx_gt_u32 exec, s1, v0
    reads(&mut a, S(2));
    a.sop1(4, EXEC, S(12));
    epilogue(&mut a);
    a.finish()
}

/// The same inside `if (s0 <= t) { ... }` (`s_and_saveexec_b64`, skipped with
/// `s_cbranch_execz` when no lane is in), narrowed further to t < s1 inside.
fn branched() -> Vec<u32> {
    let mut a = Asm::default();
    entry_code(&mut a);
    a.vopc(0xC3, S(0), 0) // v_cmp_le_u32 vcc, s0, v0
        .sop1(0x24, 12, S(VCC)) // s_and_saveexec_b64 s[12:13], vcc
        .branch(8, "join") // s_cbranch_execz
        .vopc(0xD4, S(1), 0); // v_cmpx_gt_u32 exec, s1, v0
    reads(&mut a, S(2));
    a.label("join").sop1(4, EXEC, S(12));
    epilogue(&mut a);
    a.finish()
}

/// The lanes outside s0 <= t < s1 leave the program at its start (`v_cmpx`, then
/// `s_cbranch_execz` to the end); of the lanes left, those from s3 on are in EXEC for the
/// reads; the lanes left store.
fn left() -> Vec<u32> {
    let mut a = Asm::default();
    entry_code(&mut a);
    a.vopc(0xD3, S(0), 0) // v_cmpx_le_u32 exec, s0, v0
        .vopc(0xD4, S(1), 0) // v_cmpx_gt_u32 exec, s1, v0
        .branch(8, "end")
        .sop1(4, 12, S(EXEC))
        .vopc(0xD3, S(3), 0); // v_cmpx_le_u32 exec, s3, v0
    reads(&mut a, S(2));
    a.sop1(4, EXEC, S(12));
    epilogue(&mut a);
    a.label("end").endpgm();
    a.finish()
}

/// A wave64 compute program of 64 threads (one console wave) with 4 root constants in s0..s3
/// and the table of a raw UAV at s4 (`wave64_tests`' layout).
fn native(words: &[u32]) -> NativeProgram {
    native_of(words, 64)
}

/// [`native`] with groups of `threads` threads (several console waves past 64).
fn native_of(words: &[u32], threads: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` as the runtime translates it for a host of 32-lane waves.
fn translated(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = crate::dxbc::Container::parse(&bytes)?;
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let shader = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    shader
}

/// The UAV (`dwords` long) after `shader` ran once (one group) with root constants
/// `constants`, on the host's adapter or, with `card` false, on the software adapter only (a
/// loop that could hang a card at a group barrier: G-106, G-127); None (and a line) where
/// the adapter does not take it.
fn dispatched(shader: &HostShader, constants: [u32; 4], dwords: usize, card: bool) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(!card) else {
        eprintln!("no SM 6.6 device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the adapter does not take this program (no waves of 32?): {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords as u32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.into_iter().next().expect("the UAV"))
}

/// The wave size an emitted container fixes (`PSV0`'s fewest and most lanes, runtime info
/// bytes 16 and 20), None where it fixes none.
fn required_wave(dxil: &[u8]) -> Option<u32> {
    let c = crate::dxbc::Container::parse(dxil).expect("the emitted container");
    let psv = c.part(crate::dxbc::FourCC::PSV0).expect("a PSV0 part").data;
    let u32_at = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().expect("4 bytes"));
    assert!(u32_at(0) >= 24, "runtime info with the wave's lane counts");
    match (u32_at(4 + 16), u32_at(4 + 20)) {
        (0, u32::MAX) => None,
        (min, max) => {
            assert_eq!(min, max, "one wave size");
            Some(min)
        }
    }
}

/// One program of the family: its words, the kinds its rows of 64 lanes hold, the
/// instructions it must decode to (the model's encodings are the translator's reading of
/// them too), and whether it may run on the card.
struct Fixture<'a> {
    name: &'a str,
    words: Vec<u32>,
    slots: &'a [&'a str],
    has: &'a [&'a str],
    card: bool,
}

/// Runs a fixture on the model and the device for each set of constants and gives one line
/// per disagreeing row (its kind, how many lanes differ, the first, the console's value and
/// ours). A refusal is printed, not counted: it is no wrong value.
fn family(f: &Fixture, cases: &[[u32; 4]]) -> Vec<String> {
    let native = native(&f.words);
    let names: Vec<&str> = crate::gfx10::decode(&native.code)
        .expect("decode")
        .iter()
        .map(|i| i.name().unwrap_or("?"))
        .collect();
    for want in f.has {
        assert!(names.contains(want), "{}: {want} is not among {names:?}", f.name);
    }
    let dwords = f.slots.len() * LANES;
    let shader = match translated(&native) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: refused: {e}", f.name);
            return Vec::new();
        }
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).unwrap_or_else(|e| panic!("{}: DXIL validation failed: {e}", f.name));
    }
    // The hang rule (G-106): a group barrier on a cycle of the emitted control flow may wait
    // for a host wave that left the loop; such a program never goes to the card from here.
    let looped = super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote");
    if looped > 0 && f.card {
        eprintln!("{}: {looped} group barriers on a loop of the emitted program; not run on the card", f.name);
        return Vec::new();
    }
    // Only a run in host waves of 32 says what a 32-lane card does: a program that fixes
    // that size runs nowhere else; one that fixes none runs in whatever waves the adapter
    // has, which on the software adapter are not 32 lanes (G-127).
    let wave = required_wave(&shader.dxil);
    let warp = !f.card || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    eprintln!("{}: translated, wave size {wave:?}", f.name);
    if wave.is_none() && warp {
        eprintln!(
            "{}: fixes no wave size, so its cross-lane reads are the host wave's own; not compared on the software adapter",
            f.name
        );
        return Vec::new();
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut found = Vec::new();
    let mut ran = 0;
    for &k in cases {
        let want = console(&f.words, &k, dwords);
        let Some(got) = dispatched(&shader, k, dwords, f.card) else {
            break;
        };
        ran += 1;
        for (slot, what) in f.slots.iter().enumerate() {
            let at = slot * LANES;
            let bad: Vec<usize> = (0..LANES).filter(|&l| want[at + l] != got[at + l]).collect();
            if let Some(&l) = bad.first() {
                found.push(format!(
                    "{} s0..s3 = {k:?}: {what}: {} lanes differ, lane {l}: console {:#x}, ours {:#x}",
                    f.name,
                    bad.len(),
                    want[at + l],
                    got[at + l]
                ));
            }
        }
    }
    eprintln!(
        "{}: translated; {ran} of {} cases run on the device, {} disagreements",
        f.name,
        cases.len(),
        found.len()
    );
    found
}

/// The instructions of [`reads`].
const READS: [&str; 7] = [
    "v_readfirstlane_b32",
    "v_readlane_b32",
    "s_bcnt1_i32_b64",
    "s_ff1_i32_b64",
    "s_cselect_b32",
    "v_cmp_gt_u32",
    "v_permlanex16_b32",
];

/// Windows s0 <= t < s1 in one half only, across the halves, of one lane at either side of
/// the boundary, all and none; ballot of t < s2.
const WINDOWS: [[u32; 4]; 10] = [
    [0, 64, 50, 0],
    [40, 64, 50, 0],
    [32, 64, 40, 0],
    [3, 20, 10, 0],
    [0, 32, 64, 0],
    [31, 33, 64, 0],
    [32, 33, 64, 0],
    [63, 64, 64, 0],
    [10, 10, 64, 0],
    [20, 45, 30, 0],
];

fn verdict(found: Vec<String>) {
    for f in &found {
        eprintln!("{f}");
    }
    assert!(found.is_empty(), "{} disagreements with the console's wave (above)", found.len());
}

fn reads_fixture(name: &'static str, words: Vec<u32>) -> Fixture<'static> {
    Fixture { name, words, slots: &SLOTS, has: &READS, card: true }
}

#[test]
fn a_console_wave_with_lanes_masked_in_one_half_reads_across_the_halves_as_the_console() {
    verdict(family(&reads_fixture("masked", masked()), &WINDOWS));
}

#[test]
fn a_console_wave_in_a_lane_branch_reads_across_the_halves_as_the_console() {
    verdict(family(&reads_fixture("branched", branched()), &WINDOWS));
}

#[test]
fn a_console_wave_with_lanes_gone_from_one_half_reads_across_the_halves_as_the_console() {
    // (lanes left: s0..s1; of those, from s3 on in EXEC)
    let cases = [
        [40, 64, 50, 0],
        [40, 64, 50, 50],
        [3, 20, 10, 0],
        [3, 20, 10, 12],
        [3, 20, 10, 45],
        [32, 33, 64, 0],
        [0, 64, 50, 40],
        [20, 45, 30, 33],
    ];
    verdict(family(&reads_fixture("left", left()), &cases));
}

/// The first-lane read on its own: no other instruction of the program reads across lanes.
/// EXEC narrowed to s0 <= t < s1, the masked write of 1000 + t, the first lane of it into
/// s20; every lane stores v10 and s20.
fn first_lane_alone() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0) // v_add_nc_u32 v10, 100, v0
        .sop1(4, 12, S(EXEC))
        .vopc(0xD3, S(0), 0) // v_cmpx_le_u32 exec, s0, v0
        .vopc(0xD4, S(1), 0) // v_cmpx_gt_u32 exec, s1, v0
        .vop2(0x25, 10, Lit(1000), 0) // v_add_nc_u32 v10, 1000, v0
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .sop1(4, EXEC, S(12))
        .store(10, 3, 0)
        .vop1(1, 2, S(20))
        .store(2, 3, 256)
        .endpgm();
    a.finish()
}

/// A console wave64 program whose one cross-lane instruction is a first-lane read gets the
/// console's first lane in all 64 lanes, whichever half holds it. Straight-line code with no
/// loop and no group barrier of the title's: safe on the card.
#[test]
fn a_first_lane_read_alone_gives_every_lane_the_consoles_first_lane() {
    let f = Fixture {
        name: "first-lane alone",
        words: first_lane_alone(),
        slots: &["the value written under the narrowed EXEC (v10)", "first-lane read of v10"],
        has: &["v_readfirstlane_b32", "v_cmpx_le_u32"],
        card: true,
    };
    verdict(family(&f, &[[0, 64, 0, 0], [40, 64, 0, 0], [3, 20, 0, 0], [32, 33, 0, 0], [10, 10, 0, 0]]));
}

/// A loop on a scalar count made from the console wave's first thread index, so the two
/// console waves of a group of 128 go round 1 and 2 times; inside, a first-lane read of a
/// value that differs between lanes (`v10 = 100 + t`). A group barrier for that read's
/// exchange would be met by one console wave alone on the second trip.
fn scalar_loop_of_the_wave() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0) // v10 = 100 + t
        .vop1(2, 21, V(0)) // v_readfirstlane_b32 s21, v0: 0 or 64
        .sop2(0x20, 21, S(21), I(6)) // s_lshr_b32 s21, s21, 6
        .sop2(0, 21, S(21), I(1)) // trips: (first thread >> 6) + 1
        .sop1(3, 28, I(0))
        .label("loop")
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .sop2(0, 28, S(28), S(20))
        .sop2(0, 21, S(21), I(-1))
        .sopc(7, S(21), I(0)) // s_cmp_lg_u32 s21, 0
        .branch(5, "loop") // s_cbranch_scc1
        .vop1(1, 2, S(28))
        .store(2, 3, 0)
        .endpgm();
    a.finish()
}

/// In a group of two console waves, a varying first-lane read inside a loop the two console
/// waves may go round different numbers of times (its count is a scalar of each wave's own)
/// emits no group barrier there: it is refused (Astra's review of the strip tip, finding 1).
#[test]
fn a_first_lane_read_in_a_scalar_loop_of_each_console_wave_has_no_group_barrier() {
    let words = scalar_loop_of_the_wave();
    let names: Vec<&str> = crate::gfx10::decode(&native(&words).code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect();
    for want in ["s_lshr_b32", "s_cmp_lg_u32", "s_cbranch_scc1", "v_readfirstlane_b32"] {
        assert!(names.contains(&want), "{want} is not among {names:?}");
    }
    match translated(&native_of(&words, 128)) {
        Err(e) => eprintln!("128 threads: refused: {e}"),
        Ok(shader) => {
            let n = super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote");
            assert_eq!(n, 0, "{n} group barriers on a loop the two console waves may go round unequally");
        }
    }
    // one console wave: both halves go round alike, so the exchange stays inside the loop
    let one = translated(&native_of(&words, 64)).expect("64 threads translate");
    assert!(super::bitcode::census::take_looped(&one.dxil).expect("the module") > 0);
}

/// [`scalar_loop_of_the_wave`] with its trip count s21 set by `trips` in place of the first
/// thread index's: the same loop, the same varying first-lane read inside.
fn first_lane_loop(trips: impl FnOnce(&mut Asm)) -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0); // v10 = 100 + t
    trips(&mut a);
    a.sop1(3, 28, I(0))
        .label("loop")
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .sop2(0, 28, S(28), S(20))
        .sop2(0, 21, S(21), I(-1))
        .sopc(7, S(21), I(0)) // s_cmp_lg_u32 s21, 0
        .branch(5, "loop") // s_cbranch_scc1
        .vop1(1, 2, S(28))
        .store(2, 3, 0)
        .endpgm();
    a.finish()
}

/// The refusal of a varying first-lane read where the group's console waves may part.
fn refused_for_the_group(words: &[u32], threads: u32) {
    match translated(&native_of(words, threads)) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::GROUP_UNEVEN),
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("translated: an exchange the group's console waves may meet unequally"),
    }
}

/// The scalar loop of each console wave's own count stays refused in a group of two console
/// waves, census or not (its count comes from `v_readfirstlane_b32` of the thread index).
#[test]
fn a_first_lane_read_in_a_scalar_loop_of_each_console_wave_is_refused_in_a_group() {
    refused_for_the_group(&scalar_loop_of_the_wave(), 128);
}

/// A loop on a root constant: every console wave of the group goes round it as often (the
/// census, `group_uniform`), so in a group of 128 threads the varying first-lane read inside
/// keeps its exchange, group barriers in the loop; one console wave keeps it too.
#[test]
fn a_first_lane_read_in_a_root_constant_loop_keeps_its_exchange_in_a_group() {
    let words = first_lane_loop(|a| {
        a.sop1(3, 21, S(2)); // s_mov_b32 s21, s2: a root constant
    });
    for threads in [128, 64] {
        let shader = translated(&native_of(&words, threads)).unwrap_or_else(|e| panic!("{threads} threads: {e}"));
        let n = super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote");
        assert!(n > 0, "{threads} threads: no group barrier in the loop, so no exchange there");
    }
}

/// A loop on a scalar load from the UAV: two console waves may read different counts (the
/// census shows no scalar load from writable memory alike), so it stays refused in a group
/// of 128 threads, and keeps its exchange in one console wave.
#[test]
fn a_first_lane_read_in_a_uav_scalar_loop_is_refused_in_a_group() {
    let words = first_lane_loop(|a| {
        a.raw(&[0x3D << 26 | 0x08 << 18 | 21 << 6 | 4, 125 << 25]); // s_buffer_load_dword s21, s[8:11], 0
    });
    let names: Vec<&str> = crate::gfx10::decode(&native(&words).code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect();
    assert!(names.contains(&"s_buffer_load_dword"), "s_buffer_load_dword is not among {names:?}");
    refused_for_the_group(&words, 128);
    let one = translated(&native_of(&words, 64)).expect("64 threads translate");
    assert!(super::bitcode::census::take_looped(&one.dxil).expect("the module") > 0);
}

/// A loop closed by a plain back jump (`s_branch`), left at its head on a scalar count of
/// each console wave's own (1 or 2 trips in a group of 128); after it, a varying first-lane
/// read. At the iteration limit the translation returns a thread from such a loop
/// (`LOOP_LIMIT`), so one console wave can be gone when the other reaches the exchange's
/// group barriers after the loop.
fn plain_jump_loop_then_first_lane() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0) // v10 = 100 + t
        .vop1(2, 21, V(0)) // v_readfirstlane_b32 s21, v0: 0 or 64
        .sop2(0x20, 21, S(21), I(6)) // s_lshr_b32 s21, s21, 6
        .sop2(0, 21, S(21), I(1)) // trips: (first thread >> 6) + 1
        .label("head")
        .sopc(6, S(21), I(0)) // s_cmp_eq_u32 s21, 0
        .branch(5, "done") // s_cbranch_scc1
        .sop2(0, 21, S(21), I(-1))
        .branch(2, "head") // s_branch: the plain back jump
        .label("done")
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .vop1(1, 2, S(20))
        .store(2, 3, 0)
        .endpgm();
    a.finish()
}

/// After a loop the group's console waves may go round unequally and the iteration limit
/// may return threads from (a plain back jump), a varying first-lane read is refused in a
/// group of 128 (Astra's review of the census follow-up, 1); in one console wave both
/// halves go round alike and keep the exchange.
#[test]
fn a_first_lane_read_after_a_loop_the_limit_may_return_some_waves_from_is_refused() {
    let words = plain_jump_loop_then_first_lane();
    let names: Vec<&str> = crate::gfx10::decode(&native(&words).code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect();
    assert!(names.contains(&"s_branch"), "s_branch is not among {names:?}");
    refused_for_the_group(&words, 128);
    translated(&native_of(&words, 64)).expect("64 threads translate");
}

/// A root constant chooses between the end and a loop with no way out (only the iteration
/// limit returns from it); inside, a branch on each console wave's first thread index (0 or
/// 64) runs a varying first-lane read in one console wave only, then both jump back from one
/// latch (Astra's review of the census follow-up, 2).
fn closed_loop_one_wave_reads() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop2(0x25, 10, Lit(100), 0) // v10 = 100 + t
        .vop1(2, 21, V(0)) // v_readfirstlane_b32 s21, v0: 0 or 64
        .sop1(3, 28, I(0))
        .sopc(6, S(2), I(0)) // s_cmp_eq_u32 s2, 0: a root constant
        .branch(5, "end") // s_cbranch_scc1
        .label("loop")
        .sopc(6, S(21), I(0)) // s_cmp_eq_u32 s21, 0
        .branch(4, "latch") // s_cbranch_scc0: the second console wave skips the read
        .vop1(2, 20, V(10)) // v_readfirstlane_b32 s20, v10
        .sop2(0, 28, S(28), S(20))
        .vop1(1, 2, S(28))
        .store(2, 3, 0)
        .label("latch")
        .branch(2, "loop") // s_branch
        .label("end")
        .endpgm();
    a.finish()
}

/// The census shows no block of a loop with no path to the program's end group-uniform, so
/// the read only one console wave runs there is refused in a group of 128.
#[test]
fn a_first_lane_read_in_a_closed_loop_one_console_wave_runs_is_refused() {
    refused_for_the_group(&closed_loop_one_wave_reads(), 128);
}

/// With `VICTORY_GROUP_SYNC=0` the exchange's group barriers keep their fence and lose the
/// wait, so the other host wave's first lane may not be written yet: a varying first-lane
/// read across the halves is refused, in one console wave (straight-line code) and in a
/// group of two (a root-constant loop), as W35's meetings are (Astra's review of the census
/// follow-up, 4).
#[test]
fn a_first_lane_read_across_the_halves_is_refused_with_group_barriers_off() {
    let looped = first_lane_loop(|a| {
        a.sop1(3, 21, S(2));
    });
    let was = super::TEST_GROUP_SYNC.replace(Some(0));
    let results = [translated(&native(&first_lane_alone())), translated(&native_of(&looped, 128))];
    super::TEST_GROUP_SYNC.set(was);
    for r in results {
        match r {
            Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::FIRST_LANE_NO_WAIT),
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("translated: a first-lane exchange whose barriers do not wait"),
        }
    }
}

/// [`native_of`] whose table holds, past the UAV at dword 0, a shader-resource view at dword
/// 4 and another UAV at dword 8.
fn native_with_srv_then_uav(words: &[u32], threads: u32) -> NativeProgram {
    let mut n = native_of(words, threads);
    let uav = n.used[0];
    n.used.push(UsedResource { kind: ResourceType::Srv, table_offset_dwords: 4, ..uav });
    n.used.push(UsedResource { table_offset_dwords: 8, ..uav });
    n
}

/// [`first_lane_loop`] counting from a scalar load through the descriptor at byte 16 of the
/// table (the shader-resource view), plus `soffset` (125: none) to the descriptor's address.
fn descriptor_count_loop(soffset: u32) -> Vec<u32> {
    first_lane_loop(|a| {
        a.raw(&[0x3D << 26 | 0x02 << 18 | 12 << 6 | 2, soffset << 25 | 16]) // s_load_dwordx4 s[12:15], s[4:5], 16
            .raw(&[0x3D << 26 | 0x08 << 18 | 21 << 6 | 6, 125 << 25]); // s_buffer_load_dword s21, s[12:15], 0
    })
}

/// A count loaded through a descriptor the table gives at a fixed place, a read-only view, is
/// alike in every console wave; one at that place plus a root constant (s2) may be the UAV
/// past it, which two console waves can read differently: the varying first-lane read in its
/// loop is refused in a group of 128 (Astra's review of the census follow-up, 3).
#[test]
fn a_first_lane_read_in_a_loop_counted_through_a_dynamically_offset_descriptor_is_refused() {
    let fixed = translated(&native_with_srv_then_uav(&descriptor_count_loop(125), 128)).expect("the fixed view translates");
    assert!(super::bitcode::census::take_looped(&fixed.dxil).expect("the module") > 0, "no exchange in the loop");
    match translated(&native_with_srv_then_uav(&descriptor_count_loop(2), 128)) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::GROUP_UNEVEN),
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("translated: an exchange in a loop counted from a view the root constant may pick"),
    }
}

/// `VICTORY_STRICT_HALVES` off (the default outside tests): the lowering before the strip job,
/// which takes each host wave's own first lane, so the 128-thread program the strict form
/// refuses translates again, with no group barrier on its loop (the program that draws the
/// player's body was refused this way on a 32-lane host).
#[test]
fn with_strict_halves_off_the_scalar_loop_translates_as_before() {
    use super::lower::TEST_STRICT_HALVES;
    let words = scalar_loop_of_the_wave();
    TEST_STRICT_HALVES.set(Some(false));
    let r = translated(&native_of(&words, 128));
    TEST_STRICT_HALVES.set(None);
    let shader = r.expect("128 threads translate with the strict forms off");
    assert_eq!(super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote"), 0);
}

/// A per-lane loop inside a loop on a scalar count, a first-lane read in the inner one: the
/// inner loop's body is the inner loop alone, so its per-lane exit makes it one the two
/// halves may go round unequally, and the local-wave form refuses the read there (Astra's
/// finding 3) rather than give each half its own first lane.
fn nested_loops() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0)
        .vop2(0x1A, 9, I(8), 0) // v9 = t << 8
        .vop1(1, 14, S(0))
        .vopc(0xC4, I(32), 0) // v_cmp_gt_u32 vcc, 32, v0
        .raw(&[1 << 25 | 13 << 17 | 14 << 9 | 1]) // v_cndmask_b32 v13, s1, v14, vcc: inner trips
        .sop1(3, 28, I(0))
        .sop1(3, 30, S(2)) // outer trips
        .sop1(4, 12, S(EXEC))
        .label("outer")
        .vop1(1, 11, I(0))
        .sop1(4, 14, S(EXEC))
        .label("inner")
        .vop2(0x25, 11, I(1), 11)
        .vop2(0x25, 15, V(9), 11)
        .vop1(2, 20, V(15)) // v_readfirstlane_b32 s20, v15
        .sop2(0, 28, S(28), S(20))
        .vopc(0xC4, V(13), 11) // v_cmp_gt_u32 vcc, v13, v11
        .sop2(0xF, EXEC, S(VCC), S(EXEC)) // s_and_b64 exec, vcc, exec
        .branch(9, "inner") // s_cbranch_execnz: per lane
        .sop1(4, EXEC, S(14))
        .sop2(0, 30, S(30), I(-1))
        .sopc(7, S(30), I(0)) // s_cmp_lg_u32 s30, 0
        .branch(5, "outer") // s_cbranch_scc1: scalar
        .sop1(4, EXEC, S(12))
        .vop1(1, 2, S(28))
        .store(2, 3, 0)
        .endpgm();
    a.finish()
}

#[test]
fn a_first_lane_read_in_a_per_lane_loop_inside_a_scalar_loop_is_refused() {
    let words = nested_loops();
    match translated(&native(&words)) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::UNEQUAL_LOOP),
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("translated: each half its own first lane in the inner loop"),
    }
    // the model: lanes 0-31 go round the inner loop 3 times, 32-63 once, twice over
    let mem = console(&words, &[3, 1, 2, 0], LANES);
    assert_eq!(mem[0], 2 * (1 + 2 + 3), "the first lane's trips, summed over both outer trips");
}

/// What a lane of [`allocation`] stores.
const ALLOCATION_SLOTS: [&str; 4] = [
    "slot: the first lane's base plus the lanes of the mask below (v_mbcnt)",
    "slot by the wave's exclusive prefix sum (row_shr, permlanex16 fi, lane 31)",
    "the mask's count in its first lane (s_bcnt1_i32_b64)",
    "the base read back by the first-lane read (written in the first lane alone)",
];

/// The probe allocation pass's pattern (`indirect_light_probes:CS_Primary_RegisterAllocation
/// Requests`, as the console compiler writes the allocation idiom), with the lanes s0 <= t <
/// s1 asking: the first of them found by `s_ff1_i32_b64` against the lane number
/// (`v_mbcnt` of -1) and run alone (its atomic's result here the root constant s2), the
/// count of asking lanes in it, the base read back with `v_readfirstlane_b32` under the
/// asking lanes (the others hold 0: a masked write keeping the old value), each lane's slot
/// by `v_mbcnt` of the mask, and the same slot by the compiler's exclusive prefix sum (DPP
/// row shifts, `v_permlanex16_b32` with `fi` under EXEC 0xffff0000 per half, lane 31 added
/// to the upper half).
fn allocation() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .sop1(3, 20, Lit(0xDEAD_0014))
        .sop1(3, 21, Lit(0xDEAD_0015))
        .vop1(1, 10, I(0))
        .vop1(1, 11, I(0))
        .vop1(1, 8, I(0))
        .sop1(4, 12, S(EXEC))
        .vopc(0xD3, S(0), 0) // v_cmpx_le_u32 exec, s0, v0
        .vopc(0xD4, S(1), 0) // v_cmpx_gt_u32 exec, s1, v0
        .branch(8, "rest")
        .sop1(4, 14, S(EXEC)) // the asking lanes
        .vop3(0x365, 2, S(14), I(0), I(0)) // v_mbcnt_lo_u32_b32 v2, s14, 0
        .vop3(0x366, 2, S(15), V(2), I(0)) // v_mbcnt_hi_u32_b32 v2, s15, v2
        .sop1(0x14, 16, S(EXEC)) // s_ff1_i32_b64 s16, exec
        .vop3(0x365, 9, I(-1), I(0), I(0))
        .vop3(0x366, 9, I(-1), V(9), I(0)) // v9 = the lane's number
        .sop1(4, 18, S(EXEC))
        .vopc(0xD2, S(16), 9) // v_cmpx_eq_u32 exec, s16, v9: the first asking lane alone
        .branch(8, "joined")
        .sop1(0x10, 20, S(14)) // s_bcnt1_i32_b64 s20, s[14:15]
        .vop1(1, 8, S(2)) // its atomic's result: the base
        .label("joined")
        .sop1(4, EXEC, S(18))
        .vop1(2, 21, V(8)) // v_readfirstlane_b32 s21, v8
        .vop2(0x25, 10, S(21), 2) // slot = base + lanes below
        .vop1(1, 12, I(1))
        .sop1(0x25, 22, I(-1)) // s_or_saveexec_b64 s[22:23], -1
        .vop3(0x101, 13, I(0), V(12), S(22)) // v_cndmask_b32 v13, 0, v12, s[22:23]
        .vop1(1, 4, V(13))
        .vop2_row_shr(0x25, 4, 4, 4, 1, true)
        .vop2_row_shr(0x25, 4, 4, 4, 2, false)
        .vop2_row_shr(0x25, 4, 4, 4, 4, false)
        .vop2_row_shr(0x25, 4, 4, 4, 8, false)
        .sop1(3, EXEC, Lit(0xFFFF_0000))
        .sop1(3, EXEC + 1, Lit(0xFFFF_0000))
        .permlanex16_fi(5, 4, I(-1), I(-1))
        .vop2(0x25, 4, V(5), 4)
        .vop3(0x360, 24, V(4), I(31), I(0)) // v_readlane_b32 s24, v4, 31
        .sop1(3, EXEC, I(0))
        .sop1(3, EXEC + 1, I(-1))
        .vop2(0x25, 4, S(24), 4) // the upper half adds the lower half's total
        .sop1(3, EXEC, I(-1))
        .sop1(3, EXEC + 1, I(-1))
        .vop2(0x26, 4, V(4), 13) // exclusive
        .sop1(4, EXEC, S(22))
        .vop1(1, 11, V(4))
        .label("rest")
        .sop1(4, EXEC, S(12))
        .store(10, 3, 0)
        .store(11, 3, 256)
        .vop1(1, 2, S(20))
        .store(2, 3, 512)
        .vop1(1, 2, S(21))
        .store(2, 3, 768)
        .endpgm();
    a.finish()
}

/// The allocation pattern with the asking lanes in one half only, across the halves, one
/// lane either side of the boundary, all and none (base 1000 + s0). Here the base is not
/// reserved by each half's own atomic, so the approximate local-wave form this program ends
/// in gives the upper half's lanes the lower half's slots (the card: lanes 32-63 from 0 with
/// every lane asking, lane 32 with lanes 20-44): two requests on one slot.
/// Ignored: it records that fault of the approximate local-wave form (no title program
/// reaches that form on a 32-lane host: the whole-game translate with `VICTORY_WAVE_LOCAL=idiom`
/// emitted the same programs) and fails on the card until that form refuses or exchanges.
#[test]
#[ignore = "fails on a 32-lane card: the approximate local-wave form reads each half's own first lane (a finding for the GPU lead)"]
fn the_probe_allocation_pattern_gives_each_asking_lane_the_consoles_slot() {
    let cases: Vec<[u32; 4]> = [(0, 64), (40, 64), (32, 64), (3, 20), (0, 32), (31, 33), (32, 33), (63, 64), (10, 10), (20, 45)]
        .iter()
        .map(|&(lo, hi)| [lo, hi, 1000 + lo, 0])
        .collect();
    let f = Fixture {
        name: "allocation",
        words: allocation(),
        slots: &ALLOCATION_SLOTS,
        has: &["v_mbcnt_hi_u32_b32", "s_ff1_i32_b64", "s_bcnt1_i32_b64", "v_permlanex16_b32", "v_readfirstlane_b32", "v_readlane_b32"],
        card: true,
    };
    verdict(family(&f, &cases));
}

/// The model's reading of the allocation pattern: every asking lane's two slots agree.
#[test]
fn the_model_gives_the_allocation_patterns_two_slots_alike() {
    let words = allocation();
    for (lo, hi) in [(40u32, 64u32), (3, 20), (20, 45), (0, 64)] {
        let mem = console(&words, &[lo, hi, 1000, 0], ALLOCATION_SLOTS.len() * LANES);
        for t in lo..hi {
            let t = t as usize;
            assert_eq!(mem[t], 1000 + (t as u32 - lo), "lane {t}: slot by v_mbcnt");
            assert_eq!(mem[LANES + t], t as u32 - lo, "lane {t}: slot by the prefix sum");
            assert_eq!(mem[2 * LANES + t], hi - lo, "lane {t}: the count");
        }
    }
}

/// What [`allocation_with_atomic`] leaves, as the pass does: each asking lane's slot, and
/// the counter.
const ATOMIC_SLOTS: [&str; 2] = [
    "slot: the base read back plus the lanes of the mask below (v_mbcnt)",
    "the counter after the first lane's atomic add (dword 0 of row 1)",
];

/// The allocation pass's own order (`CS_Primary_RegisterAllocationRequests`):
/// inside the asking lanes (s0 <= t < s1), the first of them alone counts them, sums that
/// count over the wave (the console compiler's `WaveActiveSum`), finds its first lane again
/// and adds the sum to a counter with a returning atomic, reads the old value back with
/// `v_readfirstlane_b32` there and adds the root constant s2; back under the asking lanes,
/// the base is read back with `v_readfirstlane_b32` (the other lanes hold 0) and each lane's
/// slot is base + `v_mbcnt` of the mask. The counter is dword 0 of row 1. Only the slots and
/// the counter are stored, as the pass stores only the request at its slot.
fn allocation_with_atomic() -> Vec<u32> {
    // `WaveActiveSum(v7)` into s7: the wave OR's words with v_add_nc_u32 for v_or_b32
    let wave_sum: Vec<u32> = WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 0x1C { w & 0x01FF_FFFF | 0x25 << 25 } else { w }).collect();
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop1(1, 14, Lit(256)) // the counter's byte
        .sop1(3, 20, Lit(0xDEAD_0014))
        .sop1(3, 7, Lit(0xDEAD_0007))
        .sop1(3, 27, Lit(0xDEAD_001B))
        .vop1(1, 10, I(0))
        .vop1(1, 8, I(0))
        .sop1(4, 12, S(EXEC))
        .vopc(0xD3, S(0), 0) // v_cmpx_le_u32 exec, s0, v0
        .vopc(0xD4, S(1), 0) // v_cmpx_gt_u32 exec, s1, v0
        .branch(8, "rest")
        .sop1(4, 14, S(EXEC)) // the asking lanes
        .vop3(0x365, 15, S(14), I(0), I(0)) // v_mbcnt_lo_u32_b32 v15, s14, 0
        .vop3(0x366, 15, S(15), V(15), I(0)) // v_mbcnt_hi_u32_b32 v15, s15, v15 (the sum uses v1, v2)
        .sop1(0x14, 16, S(EXEC)) // s_ff1_i32_b64 s16, exec
        .sop1(4, 18, S(EXEC))
        .vop3(0x365, 9, I(-1), I(0), I(0))
        .vop3(0x366, 9, I(-1), V(9), I(0)) // v9 = the lane's number
        .vop1(1, 8, I(0))
        .vopc(0xD2, S(16), 9) // v_cmpx_eq_u32 exec, s16, v9: the first asking lane alone
        .branch(8, "joined")
        .sop1(0x10, 20, S(14)) // s_bcnt1_i32_b64 s20, s[14:15]
        .vop1(1, 7, S(20))
        .raw(&wave_sum) // s7 = the sum of v7 over the lanes in EXEC: the count
        .sop1(0x14, 17, S(EXEC)) // s_ff1_i32_b64 s17, exec
        .sop1(4, 24, S(EXEC))
        .vop1(1, 8, I(0))
        .vopc(0xD2, S(17), 9) // v_cmpx_eq_u32 exec, s17, v9
        .branch(8, "added")
        .vop1(1, 8, S(7))
        .atomic_add(8, 14) // buffer_atomic_add v8, v14, s[8:11], 0 offen glc
        .label("added")
        .sop1(4, EXEC, S(24))
        .vop1(2, 26, V(8)) // v_readfirstlane_b32 s26, v8: the old counter
        .vop1(1, 8, S(26)) // (the lanes of EXEC: the first alone)
        .vop2(0x25, 8, S(2), 8) // + s2: the base
        .label("joined")
        .sop1(4, EXEC, S(18))
        .vop1(2, 27, V(8)) // v_readfirstlane_b32 s27, v8 (the other asking lanes hold 0)
        .vop2(0x25, 10, S(27), 15) // slot = base + lanes below
        .label("rest")
        .sop1(4, EXEC, S(12))
        .store(10, 3, 0)
        .endpgm();
    a.finish()
}

/// [`allocation_with_atomic`] with the asking lanes in one half only, across the halves,
/// one lane either side of the boundary, all and none (base 1000 + s0).
#[test]
fn the_probe_allocation_pass_order_gives_each_asking_lane_the_consoles_slot() {
    let cases: Vec<[u32; 4]> = [(0, 64), (40, 64), (32, 64), (3, 20), (0, 32), (31, 33), (32, 33), (63, 64), (10, 10), (20, 45)]
        .iter()
        .map(|&(lo, hi)| [lo, hi, 1000 + lo, 0])
        .collect();
    let f = Fixture {
        name: "allocation with atomic",
        words: allocation_with_atomic(),
        slots: &ATOMIC_SLOTS,
        has: &["v_mbcnt_hi_u32_b32", "s_ff1_i32_b64", "s_bcnt1_i32_b64", "buffer_atomic_add", "v_readfirstlane_b32", "v_permlanex16_b32"],
        card: true,
    };
    verdict(family(&f, &cases));
}

/// The model's reading of the pass's order.
#[test]
fn the_model_reads_the_allocation_pass_order() {
    let words = allocation_with_atomic();
    let mem = console(&words, &[20, 45, 1020, 0], ATOMIC_SLOTS.len() * LANES);
    for t in 20..45usize {
        assert_eq!(mem[t], 1020 + (t as u32 - 20), "lane {t}");
    }
    assert_eq!(mem[LANES], 25, "the counter");
}

/// What a lane of [`per_lane_loop`] stores.
const LOOP_SLOTS: [&str; 4] = [
    "the lane's own trips (v11)",
    "sum over the wave's trips of the first-lane read",
    "sum over the wave's trips of WaveActiveBitOr",
    "the last trip's first-lane read",
];

/// A single-block loop whose back edge each lane takes on its own, as the console compiler
/// writes `do { ... } while (++n < trips)` with `trips` per lane: lanes 0-31 go round s0
/// times, lanes 32-63 s1 times; the wave goes round while some lane does (`s_and_b64 exec,
/// vcc, exec; s_cbranch_execnz`). Inside, under the lanes still in, a kind that reads across
/// the halves: the first-lane read of (t << 8) + n (`first`), or the wave OR of
/// 1 << (t >> 1), each summed over the wave's trips in a scalar.
fn per_lane_loop(first: bool) -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0) // v_lshlrev_b32 v3, 2, v0
        .vop1(1, 5, I(1))
        .vop2(0x16, 6, I(1), 0)
        .vop2(0x1A, 7, V(6), 5) // v7 = 1 << (t >> 1)
        .vop2(0x1A, 9, I(8), 0) // v9 = t << 8
        .vop1(1, 14, S(0)) // v14 = s0
        .vopc(0xC4, I(32), 0) // v_cmp_gt_u32 vcc, 32, v0
        .raw(&[1 << 25 | 13 << 17 | 14 << 9 | 1]) // v_cndmask_b32 v13, s1, v14, vcc: the lane's trips
        .vop1(1, 11, I(0)) // v11 = 0
        .sop1(3, 20, Lit(0xDEAD_0014))
        .sop1(3, 28, I(0))
        .sop1(3, 29, I(0))
        .sop1(4, 12, S(EXEC))
        .label("loop")
        .vop2(0x25, 11, I(1), 11); // v_add_nc_u32 v11, 1, v11
    if first {
        a.vop2(0x25, 15, V(9), 11) // v15 = (t << 8) + n
            .vop1(2, 20, V(15)) // v_readfirstlane_b32 s20, v15
            .sop2(0, 28, S(28), S(20)); // s_add_u32 s28, s28, s20
    } else {
        a.raw(&WAVE_OR_V7_TO_S7).sop2(0, 29, S(29), S(7)); // s_add_u32 s29, s29, s7
    }
    a.vopc(0xC4, V(13), 11) // v_cmp_gt_u32 vcc, v13, v11: another trip
        .sop2(0xF, EXEC, S(VCC), S(EXEC)) // s_and_b64 exec, vcc, exec
        .branch(9, "loop") // s_cbranch_execnz
        .sop1(4, EXEC, S(12))
        .store(11, 3, 0);
    for (k, s) in [28u8, 29, 20].into_iter().enumerate() {
        a.vop1(1, 2, S(s)).store(2, 3, 256 * (k as u32 + 1));
    }
    a.endpgm();
    a.finish()
}

/// Trips (lanes 0-31, lanes 32-63): one half goes round again alone, either half; unequal
/// trips both above one; equal ones.
const TRIPS: [[u32; 4]; 6] = [[3, 1, 0, 0], [1, 3, 0, 0], [2, 5, 0, 0], [5, 2, 0, 0], [4, 4, 0, 0], [1, 1, 0, 0]];

/// A loop whose trips are one scalar count (the same for the whole wave, so for both
/// halves): the wave OR of [`per_lane_loop`] s0 times, summed in s29.
fn uniform_loop() -> Vec<u32> {
    let mut a = Asm::default();
    a.sop1(3, 5, I(0))
        .raw(&[0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]) // s_load_dwordx4 s[8:11], s[4:5], 0
        .vop2(0x1A, 3, I(2), 0)
        .vop1(1, 5, I(1))
        .vop2(0x16, 6, I(1), 0)
        .vop2(0x1A, 7, V(6), 5) // v7 = 1 << (t >> 1)
        .sop1(3, 29, I(0))
        .sop1(3, 30, S(0)) // s30 = trips
        .label("loop")
        .raw(&WAVE_OR_V7_TO_S7)
        .sop2(0, 29, S(29), S(7))
        .sop2(0, 30, S(30), I(-1)) // s_add_u32 s30, s30, -1
        .sopc(7, S(30), I(0)) // s_cmp_lg_u32 s30, 0
        .branch(5, "loop") // s_cbranch_scc1
        .vop1(1, 2, S(29))
        .store(2, 3, 0)
        .endpgm();
    a.finish()
}

/// The emitted program's group barriers on a loop of its control flow, or the refusal.
fn looped_barriers(words: &[u32]) -> Result<u32, ShaderError> {
    let shader = translated(&native(words))?;
    Ok(super::bitcode::census::take_looped(&shader.dxil).expect("the module this thread wrote"))
}

/// From the emitted control flow (no run): no group barrier stands in a loop whose back edge
/// each lane takes on its own, where the two host waves of a console wave may go round
/// different numbers of times and pair one trip's barrier with another's (or wait at it for
/// a half that left: G-106). The exchange goes outside the loop, or the program is refused.
#[test]
fn no_group_barrier_stands_in_a_loop_the_two_halves_may_go_round_unequally() {
    for (name, words) in [("loop first-lane", per_lane_loop(true)), ("loop wave OR", per_lane_loop(false))] {
        match looped_barriers(&words) {
            Err(e) => eprintln!("{name}: refused: {e}"),
            Ok(n) => assert_eq!(n, 0, "{name}: {n} group barriers on a loop the halves may go round unequally"),
        }
    }
}

/// The contrast: a loop the whole wave goes round together keeps its exchange inside, so the
/// check above is not empty.
#[test]
fn a_loop_both_halves_go_round_together_keeps_its_exchange_inside() {
    let n = looped_barriers(&uniform_loop()).expect("translates");
    assert!(n > 0, "the uniform loop's wave OR has no group barrier on its loop");
}

/// After the loop fix, on the card: a program with a group barrier on a loop never goes
/// there from [`family`] (G-106). The console's answer is the model's.
#[test]
#[ignore = "fails by design until the exact wave-wide loop form lands (board W34)"]
fn a_first_lane_read_in_a_loop_one_half_goes_round_again_alone_is_the_consoles() {
    let f = Fixture {
        name: "loop first-lane",
        words: per_lane_loop(true),
        slots: &LOOP_SLOTS,
        has: &["v_readfirstlane_b32", "s_cbranch_execnz"],
        card: true,
    };
    verdict(family(&f, &TRIPS));
}

/// [`a_first_lane_read_in_a_loop_one_half_goes_round_again_alone_is_the_consoles`] for the
/// wave OR.
#[test]
#[ignore = "fails by design until the exact wave-wide loop form lands (board W34)"]
fn a_wave_or_in_a_loop_one_half_goes_round_again_alone_is_the_consoles() {
    let f = Fixture {
        name: "loop wave OR",
        words: per_lane_loop(false),
        slots: &LOOP_SLOTS,
        has: &["v_permlanex16_b32", "s_cbranch_execnz"],
        card: true,
    };
    verdict(family(&f, &TRIPS));
}

/// The model's answers for the loop: one half going round again alone still sums every
/// trip of the wave in the scalar every lane stores.
#[test]
fn the_model_sums_every_trip_of_the_wave_whichever_half_goes_round() {
    let words = per_lane_loop(true);
    let mem = console(&words, &[1, 3, 0, 0], LOOP_SLOTS.len() * LANES);
    // trip 1: lane 0 first, (0 << 8) + 1; trips 2, 3: lane 32 first, (32 << 8) + 2, + 3
    let sum = 1 + (32 << 8) + 2 + (32 << 8) + 3;
    assert!(mem[LANES..2 * LANES].iter().all(|&x| x == sum), "{:#x}", mem[LANES]);
    assert_eq!(mem[3 * LANES], (32 << 8) + 3);
    assert_eq!((mem[0], mem[40]), (1, 3));
    let words = per_lane_loop(false);
    let mem = console(&words, &[3, 1, 0, 0], LOOP_SLOTS.len() * LANES);
    // trip 1: every lane, all 32 bits; trips 2, 3: lanes 0-31, the low 16
    assert!(mem[2 * LANES..3 * LANES].iter().all(|&x| x == u32::MAX.wrapping_add(2 * 0xFFFF)));
}
