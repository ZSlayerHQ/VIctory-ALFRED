//! A console wave of 64 on a host of narrower waves, where the wave is asked as one: the
//! cases the review of `job/land-2e` names for programs that reached 32-lane
//! waves only through the last approximate form. The group here is the tests' own: 64
//! threads, one console wave. Every case
//! stands behind a branch the whole wave takes alike but which the compiler makes from a
//! compare in the vector unit (a scalar copied into a VGPR), so the per-lane structure
//! sees the code as code only some lanes run.
//!
//! Each program goes through the whole of [`super::emit`] for a host of 32-lane waves and
//! must come out in an exact form that asks no wave size of the host; it is then run on the
//! device there is. With `VICTORY_WARP=1` that is the software adapter, whose waves have 4
//! lanes: a console wave is sixteen host waves there, and the answers must still be the
//! console's. Nothing here is skipped on it.

use super::host::{Adapter, Dispatch, Host, Root, View};
use super::{HostShader, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_cbranch_vccnz` over the next `n` words: `S_CBRANCH_VCCNZ | n`.
const S_CBRANCH_VCCNZ: u32 = 0xBF87_0000;
/// `s_and_saveexec_b64 s[12:13], vcc` and the restore `s_mov_b64 exec, s[12:13]`.
const SAVEEXEC_S12_VCC: u32 = 0xBE8C_246A;
const RESTORE_EXEC_S12: u32 = 0xBEFE_040C;

pub(super) const V_MOV_B32: u16 = 0x01;
const V_READFIRSTLANE_B32: u16 = 0x02;
const V_CVT_F32_U32: u16 = 0x06;
const V_RCP_F32: u16 = 0x2A;
const V_MUL_F32: u8 = 8;
const V_LSHRREV_B32: u8 = 22;
const V_LSHLREV_B32: u8 = 26;
const V_ADD_NC_U32: u8 = 37;
const V_SUB_NC_U32: u8 = 38;
const V_CMP_EQ_U32: u16 = 0xC2;
const V_CMP_LE_U32: u16 = 0xC3;
const V_READLANE_B32: u16 = 0x360;
const DS_WRITE_B32: u8 = 0x0D;
const DS_READ_B32: u8 = 0x36;

pub(super) const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// An inline constant 0..=64 as a source operand.
const fn k(n: u16) -> u16 {
    128 + n
}

/// VOP1 `op vdst, src0` (a scalar destination by its number too).
pub(super) const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1` (`src0` 255: a literal dword follows).
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32, (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9]
}

/// `buffer_store_dword vdata, off, s[8:11], 0 offset:byte`.
pub(super) const fn store(vdata: u8, byte: u16) -> [u32; 2] {
    [0x38 << 26 | 0x1C << 18 | (byte as u32 & 0xFFF), 0x80 << 24 | 2 << 16 | (vdata as u32) << 8]
}

/// `buffer_store_dword vdata, vaddr, s[8:11], 0 offen offset:byte`.
const fn store_at(vdata: u8, vaddr: u8, byte: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte as u32 & 0xFFF),
        0x80 << 24 | 2 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [0x36 << 26 | (op as u32) << 18, (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32]
}

/// `WaveActiveSum(v7)` into s7 in the steps of the wave OR (its words with `v_add_nc_u32`
/// for `v_or_b32`): every lane takes part, those outside EXEC with 0.
fn wave_sum_v7_to_s7() -> Vec<u32> {
    super::wave::WAVE_OR_V7_TO_S7
        .iter()
        .map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | u32::from(V_ADD_NC_U32) << 25 } else { w })
        .collect()
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11]. `body` stands behind
/// `if (s1 != 0)`, which the program asks in the vector unit: s1 is copied into v20, every
/// lane compares its copy with 0, and the wave branches on "some lane's is 0"
/// (`s_cbranch_vccnz`). With or without 512 bytes of LDS.
pub(super) fn native(body: &[u32], lds: bool) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        vop1(V_MOV_B32, 20, sgpr(1)),
        vopc(V_CMP_EQ_U32, k(0), 20),
        S_CBRANCH_VCCNZ | body.len() as u32,
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
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
            rsrc2: u32::from(lds) << 15 | 5 << 1, // 5 user SGPRs: s0..s4
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The wave size an emitted container requires of the host (`PSV0`): none where its fewest
/// and most lanes are 0 and all ones.
pub(super) fn required_wave(dxil: &[u8]) -> Option<u32> {
    let c = Container::parse(dxil).expect("the emitted container");
    let psv = c.part(FourCC::PSV0).expect("a PSV0 part").data;
    let u32_at = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().expect("4 bytes"));
    match (u32_at(4 + 16), u32_at(4 + 20)) {
        (0, u32::MAX) => None,
        (min, _) => Some(min),
    }
}

/// `program` as the runtime translates it on a host of 32-lane waves, by [`super::emit`] with
/// every form it tries: it must come out exact, in a form that asks no wave size.
/// `names`: instructions the program must decode to, with how many of each.
fn exact(program: &NativeProgram, names: &[(&str, usize)]) -> HostShader {
    exact_in(program, names).0
}

/// [`exact`], with the form taken.
fn exact_in(program: &NativeProgram, names: &[(&str, usize)]) -> (HostShader, String) {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("a container");
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    for &(name, n) in names {
        let found = p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).count();
        assert_eq!(found, n, "{name} in the fixture's code");
    }
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &program.layout, program.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, program).expect("io::raise");
    let was = (lower::TEST_HOST_WAVE.replace(Some(32)), super::TEST_WAVE_FORMS.replace(true));
    let shader = super::emit(&c, program, &p, &st, &io, &res);
    let forms = super::take_wave_forms();
    lower::TEST_HOST_WAVE.set(was.0);
    super::TEST_WAVE_FORMS.set(was.1);
    // the form handed over: the first that translated for waves the host has
    let taken = forms
        .iter()
        .filter_map(|f| f.rsplit_once(": wave "))
        .find(|(_, lanes)| *lanes != "64")
        .map(|(form, _)| form.to_owned());
    let tried = forms.join("\n  ");
    let shader = shader.unwrap_or_else(|e| panic!("not translated for a host of 32-lane waves: {e}\n  {tried}"));
    remember(&shader, program);
    let taken = taken.unwrap_or_else(|| panic!("translated in no form of the ladder\n  {tried}"));
    assert!(
        !taken.contains("APPROXIMATE") && !taken.contains("local waves"),
        "translated per host wave, not for the console's wave: {taken}\n  {tried}"
    );
    assert!(taken.starts_with("meeting"), "the form taken: {taken}\n  {tried}");
    assert_eq!(required_wave(&shader.dxil), None, "a form for any host wave asks no wave size");
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    (shader, taken)
}

/// Runs `shader` once (one group of 64 threads) with the root constants `s0`, `s1` and gives
/// the first `dwords` dwords of the UAV; None, and a line, with no device at all. A device
/// that does not take the program is a failure: it asks for no wave size.
fn run(shader: &HostShader, constants: [u32; 2], dwords: u32) -> Option<Vec<u32>> {
    run_with(shader, &constants, dwords)
}

/// [`run`] with as many of the root constants as `constants` has, from `s0` on (up to `s3`).
fn run_with(shader: &HostShader, constants: &[u32], dwords: u32) -> Option<Vec<u32>> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = device()?;
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("the device does not take a program that asks no wave size: {e}"));
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, Some(&c)) = (u.root, u.slot, constants.get(u.dword as usize)) {
            ud[u.index as usize] = c;
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    dispatched(&host);
    let views = vec![crate::oracle::fixture::View::Uav(dwords)];
    oracle(shader, &crate::oracle::fixture::Setup { constants, views, root_views: Vec::new() }, &out);
    Some(out[0].clone())
}

thread_local! {
    /// The programs of the shaders [`exact_in`] handed over on this thread, by their DXIL: what
    /// [`oracle`] runs on the console's side; and the shaders a fixture runs on purpose without
    /// the console's answer ([`without_oracle`]).
    static TRANSLATED: std::cell::RefCell<Vec<(Vec<u8>, NativeProgram)>> = const { std::cell::RefCell::new(Vec::new()) };
    static UNCHECKED: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Notes `program` as the console program `shader` was translated from, for [`oracle`].
fn remember(shader: &HostShader, program: &NativeProgram) {
    TRANSLATED.with_borrow_mut(|t| t.push((shader.dxil.clone(), program.clone())));
}

/// The next dispatch, of `shader`, is run on purpose without the console's answer (a form the
/// switches turn on or off): [`oracle`] does not check that one dispatch, and fails if the
/// next is of another shader or one with a remembered program. Every other dispatch must
/// have its program.
fn without_oracle(shader: &HostShader) {
    UNCHECKED.with_borrow_mut(|u| {
        assert!(u.is_empty(), "one dispatch without the oracle at a time");
        u.push(shader.dxil.clone());
    });
}

/// The dispatches the oracle refuses, by fixture and dispatch (from 1, in the fixture's order),
/// with a part of the reason. A refusal not here, or one here that does not come, fails the
/// fixture.
const REFUSED: &[(&str, u32, &str)] = &[
    // loops the console never leaves (the hand values are the translator's loop cap)
    ("a_meeting_beside_a_loop_some_lanes_only_leave_at_the_limit_is_refused", 2, "without ending"),
    ("a_meeting_between_two_closed_loops_the_lanes_part_into_is_refused", 1, "without ending"),
    ("a_meeting_between_two_closed_loops_the_lanes_part_into_is_refused", 2, "without ending"),
    ("a_meeting_in_a_loop_one_half_goes_round_by_a_counted_way_back_is_refused", 1, "without ending"),
    ("a_meeting_in_a_loop_one_half_goes_round_by_a_counted_way_back_is_refused", 3, "without ending"),
    // a descriptor whose last dword is 0: the INVALID format, unbound by 8.1.4
    ("a_first_lane_offset_is_the_console_waves_first_lane_in_every_host_wave", 1, "INVALID format"),
    // lane selects in a VGPR: 12.12 allows an SGPR or M0
    ("a_lane_read_at_a_lane_number_in_a_vgpr_is_each_threads_own", 1, "a lane select of Vector"),
    ("a_lane_read_at_a_lane_number_in_a_vgpr_is_each_threads_own", 2, "a lane select of Vector"),
    ("a_lane_read_whose_number_is_the_register_it_reads_is_each_threads_own", 1, "a lane select of Vector"),
    ("a_lane_read_whose_number_is_the_register_it_reads_is_each_threads_own", 2, "a lane select of Vector"),
];

/// The second reference of a fixture's dispatch: the console program `shader` was translated
/// from, run on the oracle ([`crate::oracle`]) with the same inputs, and every dword of every
/// UAV compared with `out`, the host's read-back: bit for bit where the reference pins the
/// result, within the tolerance the oracle carries where it does not, among the values that
/// could land where lanes raced (the classes are printed apart, `ORACLE <fixture>: ...`). A
/// disagreement fails the fixture with the dword and both values; a refusal must be one
/// [`REFUSED`] names for this dispatch, and one it names must come. A shader with no program
/// (not handed over by [`exact_in`] or [`remember`], nor marked [`without_oracle`]) fails, and
/// so does one whose DXIL two different programs translate to.
fn oracle(shader: &HostShader, setup: &crate::oracle::fixture::Setup, out: &[Vec<u32>]) {
    let programs: Vec<NativeProgram> =
        TRANSLATED.with_borrow(|t| t.iter().filter(|(d, _)| *d == shader.dxil).map(|(_, p)| p.clone()).collect());
    if let Some(exempt) = UNCHECKED.with_borrow_mut(std::mem::take).pop() {
        assert!(exempt == shader.dxil, "the dispatch marked without_oracle is not the next one");
        assert!(programs.is_empty(), "a dispatch marked without_oracle has a remembered program");
        return;
    }
    let Some(program) = programs.first() else {
        panic!("a dispatch with no console program for the oracle: remember it, or mark it without_oracle");
    };
    assert!(
        programs.iter().all(|p| p == program),
        "two console programs translate to this one DXIL: the oracle cannot tell which ran"
    );
    let (fixture, n) = RAN.with_borrow_mut(|r| {
        let r = r.as_mut().expect("an open gate");
        r.oracle += 1;
        (r.fixture, r.oracle)
    });
    let name = fixture.rsplit("::").nth(1).unwrap_or(fixture);
    let expected = REFUSED.iter().find(|(f, k, _)| *f == name && *k == n).map(|(_, _, why)| *why);
    match (crate::oracle::fixture::check(name, program, setup, out), expected) {
        (Ok(_), None) => {}
        (Err(r), Some(why)) if r.to_string().contains(why) => eprintln!("ORACLE {name}: REFUSED (expected) at {r}"),
        (Err(r), _) => panic!("ORACLE {name}: dispatch {n} REFUSED at {r}, which REFUSED does not name for it"),
        (Ok(_), Some(why)) => panic!("ORACLE {name}: dispatch {n} agrees where REFUSED names a refusal ({why})"),
    }
}

/// The device a fixture runs on; None, and a line, with no device at all. The adapter is the
/// shared rule's (`crate::adapter`, through [`Host::new`]): `VICTORY_ADAPTER` picks one, and
/// with `VICTORY_REQUIRE_GPU` set a fixture that comes here with no hardware device, the
/// software adapter or another adapter than the one asked for fails there instead of
/// passing without having run.
fn device() -> Option<Host> {
    let host = Host::new(false);
    if let Err(why) = &host {
        eprintln!("no SM 6.6 device ({why}); not run");
    }
    host.ok()
}

/// What the fixture on this thread ran on, for its one line ([`Gate`]).
struct Ran {
    fixture: &'static str,
    /// Per dispatch that returned, in order: the adapter of the device it ran on (None
    /// where DXGI does not say).
    dispatches: Vec<Option<Adapter>>,
    /// Why a read-back was not asserted, where it was not.
    unchecked: Option<&'static str>,
    /// Dispatches the oracle was asked of.
    oracle: u32,
}

thread_local! {
    static RAN: std::cell::RefCell<Option<Ran>> = const { std::cell::RefCell::new(None) };
}

/// A dispatch on `host` came back: [`Host::dispatch`] returns its read-back only once the
/// fence it signals after the command list reports a completed value at or above the signal
/// (not a removed device's `UINT64_MAX`) and the device is still there after that. The
/// adapter is asked of the device of this dispatch, by its LUID.
fn dispatched(host: &Host) {
    RAN.with_borrow_mut(|ran| {
        let ran = ran.as_mut().expect("a fixture that dispatches opens its gate first (`gate!`)");
        ran.dispatches.push(host.adapter());
    });
}

/// `dispatches` by adapter, in the order each first ran: the adapter and how many.
fn by_adapter(dispatches: &[Option<Adapter>]) -> Vec<(Option<Adapter>, u32)> {
    let mut out: Vec<(Option<Adapter>, u32)> = Vec::new();
    for a in dispatches {
        match out.iter_mut().find(|(b, _)| b == a) {
            Some((_, n)) => *n += 1,
            None => out.push((a.clone(), 1)),
        }
    }
    out
}

fn adapter_line(a: &Option<Adapter>) -> String {
    a.as_ref().map_or("an adapter DXGI does not describe (UNKNOWN)".to_owned(), Adapter::to_string)
}

/// What the gates of this run of the tests add up to ([`Tally::line`]).
#[derive(Default)]
struct Tally {
    /// Fixtures that opened a gate, and of them: those with a dispatch that came back,
    /// those without (`NOT RUN`), those that failed.
    fixtures: u32,
    ran: u32,
    not_run: u32,
    failed: u32,
    /// Of `ran`: every assertion on what was read back held; a read-back was not asserted.
    matched: u32,
    unasserted: u32,
    dispatches: Vec<Option<Adapter>>,
}

static TALLY: std::sync::Mutex<Option<Tally>> = std::sync::Mutex::new(None);

unsafe extern "C" {
    /// The C runtime's: `f` runs when the process exits.
    fn atexit(f: extern "C" fn()) -> i32;
}

/// Prints [`Tally::line`] when the test process exits, after the last fixture: from the
/// main thread, so it is not held back with a passing test's output. The test harness
/// leaves a run with a failed test by `ExitProcess`, which runs nothing registered here: a
/// run without this line has not passed, and its failures are the harness's own lines.
extern "C" fn summary() {
    use std::io::Write;
    let tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = tally.as_ref() {
        let env = |name: &str| std::env::var(name).map_or(format!("{name} unset"), |v| format!("{name}={v}"));
        let settings = ["VICTORY_REQUIRE_GPU", "VICTORY_ADAPTER", "VICTORY_WARP", "VICTORY_WAVE_MEET", "VICTORY_WAVE_MEET_WIDE"];
        let _ = writeln!(std::io::stderr(), "{} [{}]", t.line(), settings.map(env).join(", "));
    }
}

impl Tally {
    /// The one line to check a gate's run by, `MEET-GATE SUMMARY: ...`. The whole of the
    /// `meet_tests` at this commit, translated for a host of 32-lane waves, on a card: 25
    /// fixtures with a gate; 24 with completed dispatches (37 dispatches), all on the one
    /// adapter asked for; 1 `NOT RUN` (the minimum plus zero, which no form translates: a
    /// translation check); 0 failed; read-back matched in 23 and not asserted in 1 (the
    /// switches fixture's run with the meeting off, asserted on the software adapter's waves
    /// of 4 only). On the software adapter: matched in 24, not asserted in 0. Any other
    /// figure, a second adapter or an unknown one leaves the gate open.
    fn line(&self) -> String {
        let on = by_adapter(&self.dispatches);
        let adapters = match on.as_slice() {
            [] => "no dispatch".to_owned(),
            [(a, n)] => format!("all {n} dispatches on adapter {}", adapter_line(a)),
            several => format!(
                "dispatches on {} ADAPTERS: {}",
                several.len(),
                several.iter().map(|(a, n)| format!("{n} on {}", adapter_line(a))).collect::<Vec<_>>().join("; ")
            ),
        };
        format!(
            "MEET-GATE SUMMARY: {} fixtures with a gate, translated for 32-lane host waves: {} with completed dispatches, \
             {} NOT RUN, {} failed; read-back matched in {}, not asserted in {}; {adapters}",
            self.fixtures, self.ran, self.not_run, self.failed, self.matched, self.unasserted
        )
    }

    /// Adds a fixture's gate, closed as `ran` says (`failed`: by a failed assertion).
    fn add(ran: &Ran, failed: bool) {
        let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
        let t = tally.get_or_insert_with(|| {
            // SAFETY: registers a function that takes nothing and only reads `TALLY`.
            unsafe { atexit(summary) };
            Tally::default()
        });
        t.fixtures += 1;
        t.dispatches.extend(ran.dispatches.iter().cloned());
        match (failed, ran.dispatches.is_empty(), ran.unchecked) {
            (true, ..) => t.failed += 1,
            (false, true, _) => t.not_run += 1,
            (false, false, None) => (t.ran, t.matched) = (t.ran + 1, t.matched + 1),
            (false, false, Some(_)) => (t.ran, t.unasserted) = (t.ran + 1, t.unasserted + 1),
        }
    }
}

/// A fixture's proof that it ran: opened first thing in a fixture that dispatches
/// (`gate!`), and when the fixture ends without a failed assertion it prints one line,
/// `MEET-GATE <fixture>: ...`, with the adapter of each of its dispatches (description,
/// vendor, device, LUID), how many came back fenced to completion on it, and that what was
/// read back matched (every assertion on it held). A fixture that ended without a dispatch
/// says so in that line instead: a pass alone does not show a program ran.
/// `VICTORY_REQUIRE_GPU` makes a fixture that comes to dispatch on another device than the
/// one asked for a failure ([`device`]). The gates of a run add up to one last line
/// ([`Tally::line`]).
struct Gate;

impl Gate {
    fn open(fixture: &'static str) -> Gate {
        RAN.set(Some(Ran { fixture, dispatches: Vec::new(), unchecked: None, oracle: 0 }));
        Gate
    }

    /// A read-back of this fixture is not asserted on this device, and why.
    fn unchecked(&self, why: &'static str) {
        RAN.with_borrow_mut(|ran| ran.as_mut().expect("an open gate").unchecked = Some(why));
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        let Some(ran) = RAN.take() else { return };
        Tally::add(&ran, std::thread::panicking());
        if std::thread::panicking() {
            return; // (the failure is the fixture's line)
        }
        let name = ran.fixture.rsplit("::").nth(1).unwrap_or(ran.fixture);
        let on = by_adapter(&ran.dispatches);
        if on.is_empty() {
            eprintln!("MEET-GATE {name}: NOT RUN (no dispatch: no device, or nothing translated to run)");
            return;
        }
        let on: Vec<String> = on
            .iter()
            .map(|(a, n)| format!("adapter {}: {n} dispatch(es) fenced to completion", adapter_line(a)))
            .collect();
        eprintln!(
            "MEET-GATE {name}: {}; {}",
            on.join("; "),
            ran.unchecked.map_or("read-back matched".to_owned(), |why| format!("read-back not asserted: {why}")),
        );
    }
}

/// Opens the [`Gate`] of the fixture it stands in, named by its function.
macro_rules! gate {
    () => {{
        fn here() {}
        Gate::open(std::any::type_name_of_val(&here))
    }};
}

/// A fixture's dispatches are recorded one by one, each with the adapter of its own device,
/// and the run's last line adds the fixtures up: one that ran on two adapters, or on one
/// DXGI does not describe, shows as such and not as its last dispatch's.
#[test]
fn the_gates_record_is_per_dispatch_and_adds_up() {
    let adapter = |description: &str, vendor: u32, luid: u32| Adapter {
        description: description.to_owned(),
        vendor,
        device: 7,
        luid: (0, luid),
        software: false,
    };
    let (intel, nvidia) = (adapter("Intel(R) Arc(TM) Graphics", 0x8086, 0x11), adapter("NVIDIA GeForce RTX 5080 Laptop GPU", 0x10de, 0x22));
    let mixed = [Some(intel.clone()), Some(nvidia.clone()), Some(nvidia.clone()), None];
    assert_eq!(by_adapter(&mixed), [(Some(intel.clone()), 1), (Some(nvidia.clone()), 2), (None, 1)]);
    assert_eq!(
        nvidia.to_string(),
        "\"NVIDIA GeForce RTX 5080 Laptop GPU\" (vendor 0x10de, device 0x0007, LUID 00000000:00000022; hardware)"
    );
    let one = Tally { fixtures: 19, ran: 18, not_run: 1, failed: 0, matched: 17, unasserted: 1, dispatches: vec![Some(nvidia.clone()); 25] };
    assert_eq!(
        one.line(),
        format!(
            "MEET-GATE SUMMARY: 19 fixtures with a gate, translated for 32-lane host waves: 18 with completed dispatches, \
             1 NOT RUN, 0 failed; read-back matched in 17, not asserted in 1; all 25 dispatches on adapter {nvidia}"
        )
    );
    let two = Tally { dispatches: mixed.to_vec(), ..Tally::default() }.line();
    assert!(two.contains("dispatches on 3 ADAPTERS: 1 on \"Intel(R) Arc(TM) Graphics\""), "{two}");
    assert!(two.contains("2 on \"NVIDIA GeForce RTX 5080 Laptop GPU\"") && two.contains("1 on an adapter DXGI does not describe (UNKNOWN)"), "{two}");
}

/// 32 terms of 1 and 32 of 3: the sum over the wave is 128, the count of its lanes 64, the
/// mean 2, and one lane (the wave's first) writes the mark of the writer. Half a console
/// wave on its own gives 32 or 96, 32, and a writer in each half.
#[test]
fn a_sum_and_a_count_are_the_console_waves_and_one_lane_writes() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_LSHRREV_B32, 6, k(5), 0), // v6 = thread >> 5
        vop2(V_LSHLREV_B32, 6, k(1), 6), // v6 = 2 v6
        vop2(V_ADD_NC_U32, 7, k(1), 6),  // v7 = 1 + v6: 1 below lane 32, 3 from it on
    ];
    body.extend(wave_sum_v7_to_s7());
    body.extend([
        vop1(V_MOV_B32, 10, sgpr(7)), // v10 = the sum
        vop1(V_MOV_B32, 7, k(1)),     // v7 = 1
    ]);
    body.extend(wave_sum_v7_to_s7());
    body.extend([
        vop1(V_MOV_B32, 11, sgpr(7)),       // v11 = the count
        vop1(V_CVT_F32_U32, 12, vgpr(10)),  // v12 = float(sum)
        vop1(V_CVT_F32_U32, 13, vgpr(11)),  // v13 = float(count)
        vop1(V_RCP_F32, 13, vgpr(13)),      // v13 = 1 / count
        vop2(V_MUL_F32, 12, vgpr(12), 13),  // v12 = the mean
    ]);
    body.extend(store(10, 0));
    body.extend(store(11, 4));
    body.extend(store(12, 8));
    body.extend([
        0xBE86_147E,                     // s_ff1_i32_b64 s6, exec: the wave's first lane
        vopc(V_CMP_EQ_U32, sgpr(6), 0),  // vcc = (s6 == thread)
        SAVEEXEC_S12_VCC,
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 14, k(1)),       // v14 = 1
    ]);
    body.extend(store_at(14, 9, 16)); // the writer's mark, at dword 4 + thread
    body.push(RESTORE_EXEC_S12);
    let program = native(&body, false);
    let shader = exact(
        &program,
        &[("v_add_nc_u32", 13), ("s_ff1_i32_b64", 1), ("v_rcp_f32", 1), ("v_mul_f32", 1), ("buffer_store_dword", 4), ("s_cbranch_vccnz", 1)],
    );
    let Some(out) = run(&shader, [0, 1], 68) else { return };
    assert_eq!(out[0], 128, "the sum of 32 ones and 32 threes over the console's wave");
    assert_eq!(out[1], 64, "the lanes of the console's wave");
    assert_eq!(f32::from_bits(out[2]), 2.0, "the mean");
    let writers: Vec<usize> = (0..64).filter(|&t| out[4 + t] != 0).collect();
    assert_eq!(writers, [0], "one lane writes, the console wave's first");
    // (and with the branch taken by the whole wave, nothing is written)
    let out = run(&shader, [0, 0], 68).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "nothing is written when the wave skips the body");
}

/// Every thread holds a value of its own (0x100 + thread). With EXEC kept to the threads
/// from s0 (40) on, the first lane of EXEC is console lane 40 and every lane must see its
/// 0x128, the lower half too, which has no lane in EXEC. Lane 37's value is 0x125 for every
/// lane; lane 37 of 64 is not lane 5 of a half (0x105).
#[test]
fn a_masked_first_lane_and_lane_37_are_the_console_waves() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                             // v7 = 0x100 + thread
        vopc(V_CMP_LE_U32, sgpr(0), 0),    // vcc = (s0 <= thread)
        SAVEEXEC_S12_VCC,
        vop1(V_READFIRSTLANE_B32, 6, vgpr(7)), // s6 = v7 of the first lane of EXEC
        RESTORE_EXEC_S12,
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.extend([
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 10, sgpr(6)),
        vop1(V_MOV_B32, 11, sgpr(7)),
    ]);
    body.extend(store_at(10, 9, 0)); // dword thread: the first lane's
    body.extend(store_at(11, 9, 256)); // dword 64 + thread: lane 37's
    let program = native(&body, false);
    let shader = exact(&program, &[("v_readfirstlane_b32", 1), ("v_readlane_b32", 1), ("v_cmp_le_u32", 1), ("buffer_store_dword", 2)]);
    let Some(out) = run(&shader, [40, 1], 128) else { return };
    for t in 0..64 {
        assert_eq!(out[t], 0x128, "thread {t}: the first lane of EXEC is console lane 40");
        assert_eq!(out[64 + t], 0x125, "thread {t}: lane 37 of the console's wave");
    }
}

/// Every thread writes 0x200 + thread to its own dword of LDS and then reads the dword of
/// thread 63 - thread, which the other half of the wave wrote: no barrier stands between,
/// because the console runs the wave's lanes in step. On narrower host waves the reads must
/// still come after every write.
#[test]
fn what_the_other_half_wrote_to_lds_is_read() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x200, // v7 = 0x200 + thread
    ];
    body.extend(ds(DS_WRITE_B32, 0, 9, 7)); // LDS[4 thread] = v7
    body.extend([
        vop2(V_SUB_NC_U32, 8, k(63), 0), // v8 = 63 - thread
        vop2(V_LSHLREV_B32, 8, k(2), 8), // v8 = 4 v8
    ]);
    body.extend(ds(DS_READ_B32, 10, 8, 0)); // v10 = LDS[v8]
    body.extend(store_at(10, 9, 0)); // dword thread
    let program = native(&body, true);
    let shader = exact(&program, &[("ds_write_b32", 1), ("ds_read_b32", 1), ("v_sub_nc_u32", 1), ("buffer_store_dword", 1)]);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    for t in 0..64u32 {
        assert_eq!(out[t as usize], 0x200 + 63 - t, "thread {t} reads what thread {} wrote", 63 - t);
    }
}

/// The threads below s0 (40) OR their bits (`1 << (thread >> 1)`) over the wave inside
/// `if (thread < s0)` and store the result. `skip` is the branch past the body when no
/// thread is below s0.
fn a_wave_or_inside_a_lane_branch(skip: u32) -> NativeProgram {
    let mut body = vec![
        vop1(V_MOV_B32, 5, k(1)),         // v5 = 1
        vop2(V_LSHRREV_B32, 6, k(1), 0),  // v6 = thread >> 1
        vop2(V_LSHLREV_B32, 7, vgpr(6), 5), // v7 = v5 << v6
        0x7D88_0000,                      // v_cmp_gt_u32 vcc, s0, v0
        SAVEEXEC_S12_VCC,
        skip | 23, // to the restore of EXEC
    ];
    body.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    body.push(vop1(V_MOV_B32, 0, sgpr(7))); // v0 = the OR
    body.extend(store(0, 0));
    body.push(RESTORE_EXEC_S12);
    native(&body, false)
}

/// A wave OR inside a per-lane branch. Where the branch only skips work no lane is left for
/// (`s_cbranch_execz`), the skip-free form brings every lane to the meeting. Where it is
/// taken on SCC (`s_cbranch_scc0` on "some lane is left"), the lanes must take it as the
/// console does, by the whole wave at once: they vote at a meeting, and the OR's meeting
/// stands inside the branch. That form is off unless asked for (`VICTORY_WAVE_MEET_WIDE=1`):
/// without it the program is not exact, with it the console wave's OR comes out, on the
/// software adapter's sixteen host waves too.
#[test]
fn a_wave_or_inside_a_lane_branch_meets_once_the_wave_takes_the_branch_together() {
    let _gate = gate!();
    let names = [("v_or_b32", 6), ("v_cmp_gt_u32", 1), ("buffer_store_dword", 1)];
    let (shader, taken) = exact_in(&a_wave_or_inside_a_lane_branch(0xBF88_0000), &names);
    assert_eq!(taken, "meeting, without skips", "an execz skip needs no vote");
    let Some(out) = run(&shader, [40, 1], 1) else { return };
    assert_eq!(out[0], 0x000F_FFFF, "the OR of the bits of threads 0..39");
    let was = super::TEST_MEET_WIDE.replace(true);
    let wide = exact_in(&a_wave_or_inside_a_lane_branch(0xBF84_0000), &names);
    super::TEST_MEET_WIDE.set(was);
    assert_eq!(wide.1, "meeting, wave-wide branches");
    let out = run(&wide.0, [40, 1], 1).expect("the device that just ran");
    assert_eq!(out[0], 0x000F_FFFF, "the OR of the bits of threads 0..39, the branch taken by the wave");
    let out = run(&wide.0, [0, 1], 1).expect("the device that just ran");
    assert_eq!(out[0], 0, "no thread below s0: the wave skips the body");
}

/// The review's smallest case of a first-lane byte offset (4.2; the two average-luminance
/// downsample passes read theirs this way):
/// `v1 = 4 * thread`, `s4 = readfirstlane(v1)` with every lane in EXEC, one dword loaded
/// from the root buffer at byte `s4` and stored at the thread's own offset. The console
/// reads byte 0 for the whole wave of 64. The words at bytes 0, 4, ... are all different
/// (bytes 0 and 128 among them, the first lanes of two host waves of 32), and every thread,
/// thread 32 too, must store the word at byte 0. The program asks for no wave size, so it
/// runs on any device: a host wave that reads its own first lane shows there.
#[test]
fn a_first_lane_offset_is_the_console_waves_first_lane_in_every_host_wave() {
    let _gate = gate!();
    let program = super::host_wave_tests::native(&super::host_wave_tests::FIRST_LANE_OFFSET, 64, false);
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("a container");
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &program.layout, program.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &program).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let shader = super::emit(&c, &program, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.expect("translated for a host of 32-lane waves");
    remember(&shader, &program);
    assert_eq!(required_wave(&shader.dxil), None, "no wave size, as before");
    if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let Some((out, _)) = run_first_lane_offset(&shader) else { return };
    for (t, &word) in out.iter().enumerate() {
        assert_eq!(word, 0x5000, "thread {t} reads the word at byte 0, the console wave's first lane's offset");
    }
}

/// Runs the first-lane offset program's `shader` once over a root buffer of the words
/// 0x5000, 0x5001, ...: what each of the 64 threads stored, and whether the device is the
/// software adapter. None, and a line, with no device at all.
fn run_first_lane_offset(shader: &HostShader) -> Option<(Vec<u32>, bool)> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = device()?;
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("the device does not take a program that asks no wave size: {e}"));
    // (user data: the output's table at heap index 0)
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let mut roots = vec![(0, Root::Constants(ud))];
    let words: Vec<u32> = (0..64).map(|n| 0x5000 + n).collect();
    for (n, _) in shader.root_srvs.iter().enumerate() {
        roots.push((1 + n as u32, Root::View { srv: true, data: words.clone() }));
    }
    let out = host
        .dispatch(&root, &pso, &Dispatch { roots, views: &[View::RawUav(64)], samplers: &[], groups: [1, 1, 1] })
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    dispatched(&host);
    let setup = crate::oracle::fixture::Setup {
        constants: &[],
        views: vec![crate::oracle::fixture::View::Uav(64)],
        root_views: vec![&words],
    };
    oracle(shader, &setup, &out);
    Some((out[0][..64].to_vec(), host.is_software()))
}

/// The two switches of the form. Unset, the meeting is tried and its wave-wide form is not
/// (`VICTORY_WAVE_MEET_WIDE=1` asks for that one). `VICTORY_WAVE_MEET=0` turns the meeting
/// off whole: no meeting form is tried, for a program of the ladder or for a first-lane
/// offset, and the first-lane offset comes out as before the form, each host wave reading
/// its own first lane's word (on the software adapter's waves of 4 lanes, thread 4 reads
/// word 4). Both are part of the shader cache's key, so one build and one cache serve a
/// run each way.
#[test]
fn the_meeting_is_off_with_its_switch_and_its_wide_form_off_without_its_own() {
    let gate = gate!();
    let meetings = |forms: &[String]| forms.iter().filter(|f| f.starts_with("meeting") || f.starts_with("met:")).count();
    // unset: the meeting is tried, the wave-wide form of it is not
    let votes = a_wave_or_inside_a_lane_branch(0xBF84_0000);
    let (_, forms) = tried(&votes);
    assert!(meetings(&forms) > 0, "the meeting form is tried by default\n  {}", forms.join("\n  "));
    assert!(
        !forms.iter().any(|f| f.contains("wave-wide branches")),
        "a wave-wide form is tried without its switch\n  {}",
        forms.join("\n  ")
    );
    let first_lane = super::host_wave_tests::native(&super::host_wave_tests::FIRST_LANE_OFFSET, 64, false);
    let (on, forms) = tried(&first_lane);
    assert!(forms.iter().any(|f| f == "meeting: wave any"), "a first-lane offset meets by default\n  {}", forms.join("\n  "));
    // off: nothing of the form, in either place it is offered
    let was = super::TEST_MEET_OFF.replace(true);
    let ladder = tried(&votes);
    let wide = super::TEST_MEET_WIDE.replace(true);
    let asked_wide = tried(&votes);
    super::TEST_MEET_WIDE.set(wide);
    let off = tried(&first_lane);
    super::TEST_MEET_OFF.set(was);
    for (what, forms) in [("the ladder", &ladder.1), ("the ladder, wave-wide asked", &asked_wide.1), ("a first-lane offset", &off.1)] {
        assert_eq!(meetings(forms), 0, "{what}: a meeting with VICTORY_WAVE_MEET=0\n  {}", forms.join("\n  "));
    }
    let (on, off) = (on.expect("translated with the meeting"), off.0.expect("translated without the meeting"));
    assert_ne!(on.dxil, off.dxil, "the switch changes the program handed over");
    assert_eq!(required_wave(&off.dxil), None, "no wave size, as before the form");
    // (the meeting off: each host wave reads its own first lane, not the console's answer)
    without_oracle(&off);
    if let Some((out, software)) = run_first_lane_offset(&off) {
        if software {
            assert_eq!((out[0], out[4]), (0x5000, 0x5004), "without the meeting each host wave reads its own first lane's word");
        } else {
            gate.unchecked("each host wave's own first lane is asserted on the software adapter's waves of 4 only");
        }
    }
    // the cache's key holds both switches
    let key = |switch: &'static str, value: Option<&'static str>| {
        super::translation_fingerprint_with(|name| value.filter(|_| name == switch).map(Into::into))
    };
    assert_ne!(key("VICTORY_WAVE_MEET", None), key("VICTORY_WAVE_MEET", Some("0")));
    assert_ne!(key("VICTORY_WAVE_MEET_WIDE", None), key("VICTORY_WAVE_MEET_WIDE", Some("1")));
}

const V_CNDMASK_B32: u8 = 1;
const V_ADD_F32: u8 = 3;

/// v7 = `terms[n]` in lane n and 0.0 in the lanes after them (the thread index in v0).
pub(super) fn lane_terms(terms: &[(u16, f32)]) -> Vec<u32> {
    let mut w = vec![vop1(V_MOV_B32, 7, k(0))]; // v7 = 0
    for &(lane, term) in terms {
        w.extend([
            vop1(V_MOV_B32, 8, 255),
            term.to_bits(),                     // v8 = the term
            vopc(V_CMP_EQ_U32, k(lane), 0),     // vcc = (lane == thread)
            vop2(V_CNDMASK_B32, 7, vgpr(7), 8), // v7 = vcc ? v8 : v7
        ]);
    }
    w
}

/// `WaveActiveSum(v7)` of floats into s7: the wave OR's words with `v_add_f32` for
/// `v_or_b32`. `two_apart_first`: the first two steps the other way round, each lane adding
/// the lane two away (dpp8:[2,3,0,1,6,7,4,5]) before its neighbour (dpp8:[1,0,3,2,5,4,7,6]),
/// so that lanes 0..3 hold (x0 + x2) + (x1 + x3) where the compiler's own expansion holds
/// (x0 + x1) + (x2 + x3). Every lane is counted once either way.
pub(super) fn wave_sum_f32_v7_to_s7(two_apart_first: bool) -> Vec<u32> {
    let mut w: Vec<u32> = super::wave::WAVE_OR_V7_TO_S7
        .iter()
        .map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | u32::from(V_ADD_F32) << 25 } else { w })
        .collect();
    if two_apart_first {
        use super::wave::reduction::{NEIGHBOURS, TWO_APART, WORK, dpp8};
        (w[4], w[6]) = (dpp8(TWO_APART, WORK), dpp8(NEIGHBOURS, WORK));
    }
    w
}

/// The review's case of a sum of floats (Question 1): 2^25, 1, -2^25 and 0 in lanes 0..3,
/// 0 in the others. Each addition rounds to 24 bits, so the order of the additions is part
/// of the answer: neighbours first, (2^25 + 1) + (-2^25 + 0) = 0; lanes two apart first,
/// (2^25 - 2^25) + (1 + 0) = 1. The console's answer is the one its instructions add up,
/// and the meeting must give that, not the sum in an order of its own.
#[test]
fn a_float_sum_over_the_wave_is_added_up_in_the_programs_order() {
    let _gate = gate!();
    let terms = [(0, 33_554_432.0f32), (1, 1.0), (2, -33_554_432.0)];
    for (two_apart_first, want) in [(false, 0.0f32), (true, 1.0)] {
        let mut body = lane_terms(&terms);
        body.extend(wave_sum_f32_v7_to_s7(two_apart_first));
        body.push(vop1(V_MOV_B32, 10, sgpr(7))); // v10 = the sum
        body.extend(store(10, 0));
        let program = native(&body, false);
        let shader = exact(&program, &[("v_add_f32", 6), ("v_cndmask_b32", 4), ("buffer_store_dword", 1)]);
        let Some(out) = run(&shader, [0, 1], 1) else { return };
        assert_eq!(
            f32::from_bits(out[0]),
            want,
            "the sum as the program adds it up (lanes two apart first: {two_apart_first})"
        );
    }
}

/// `WavePrefixSum(v7)` of floats into v2, exclusive, as the compiler expands a prefix: each
/// row of 16 lanes adds the lanes 1, 2, 4 and 8 below (`row_shr`), the rows above take the
/// last lane of the row below, and every lane then takes the result of the lane below it
/// (lane 0: 0).
fn exclusive_prefix_sum_f32_v7_to_v2() -> Vec<u32> {
    const EXEC: u8 = 126;
    let sop1 = |sdst: u8, op: u8, src: u8| 0xBE80_0000 | u32::from(sdst) << 16 | u32::from(op) << 8 | u32::from(src);
    // `s_mov_b32 exec_lo, lo` and `s_mov_b32 exec_hi, hi`
    let exec = |lo: u32, hi: u32| [sop1(EXEC, 0x03, 255), lo, sop1(EXEC + 1, 0x03, 255), hi];
    let row_shr = |n: u32| 0xF << 28 | 0xF << 24 | (0x110 + n) << 8 | 1; // of v1
    let mut w = vec![
        sop1(12, 0x25, 193), // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x80 | 263 << 9 | 12 << 18, // v_cndmask_b32 v1, 0, v7, s[12:13]
    ];
    for n in [1, 2, 4, 8] {
        w.extend([vop2(V_ADD_F32, 1, 0xFA, 1), row_shr(n)]); // v_add_f32 v1, v1, v1 row_shr:n
    }
    for (last, lo, hi) in [(15, 0xFFFF_0000, 0), (31, 0, 0x0000_FFFF), (47, 0, 0xFFFF_0000)] {
        w.extend(vop3(V_READLANE_B32, 7, vgpr(1), k(last))); // s7 = v1 of the row's last lane
        w.extend(exec(lo, hi)); // the 16 lanes of the next row
        w.push(vop2(V_ADD_F32, 1, sgpr(7), 1)); // v1 = s7 + v1
    }
    w.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1
    w.push(vop1(V_MOV_B32, 3, vgpr(1))); // v3 = v1
    w.extend([vop1(V_MOV_B32, 3, 0xFA), row_shr(1)]); // v_mov_b32 v3, v1 row_shr:1
    for (last, lo, hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
        w.extend(vop3(V_READLANE_B32, 7, vgpr(1), k(last)));
        w.extend(exec(lo, hi)); // the first lane of the next row
        w.push(vop1(V_MOV_B32, 3, sgpr(7))); // v3 = s7
    }
    w.extend([
        sop1(EXEC, 0x04, 193),        // s_mov_b64 exec, -1
        vop1(V_MOV_B32, 2, k(0)),     // v2 = 0
        sop1(EXEC, 0x04, 194),        // s_mov_b64 exec, -2: every lane but lane 0
        vop1(V_MOV_B32, 2, vgpr(3)),  // v2 = v3
        sop1(EXEC, 0x04, 12),         // s_mov_b64 exec, s[12:13]
    ]);
    w
}

/// What [`exclusive_prefix_sum_f32_v7_to_v2`] leaves in each lane on the console, its
/// additions one by one.
fn console_exclusive_prefix(x: &[f32; 64]) -> [f32; 64] {
    let mut v = *x;
    for n in [1, 2, 4, 8] {
        let old = v;
        for l in (0..64).filter(|l| l % 16 >= n) {
            v[l] = old[l - n] + old[l];
        }
    }
    for row in 1..4 {
        let below = v[16 * row - 1];
        for x in &mut v[16 * row..16 * row + 16] {
            *x += below; // (below + x on the console: the same sum)
        }
    }
    let mut out = [0.0; 64];
    out[1..].copy_from_slice(&v[..63]);
    out
}

/// The review's case of a prefix of floats (Question 1): 2^25, 0, -2^25 and 1 in lanes
/// 0..3. Lane 4's exclusive prefix on the console is (1 + -2^25) + (0 + 2^25) = 0, the row's
/// lanes added in pairs; added one after the other from lane 0 it is 1. Terms in the second
/// row and in the upper half go through the steps that carry a row into the next. Every
/// lane must hold what the console's additions give it.
#[test]
fn a_float_prefix_is_added_up_in_the_programs_order() {
    let _gate = gate!();
    let terms = [(0, 33_554_432.0f32), (2, -33_554_432.0), (3, 1.0), (17, 3.0), (30, 16_777_216.0), (37, 1.0), (50, 0.5)];
    let mut x = [0.0f32; 64];
    for &(lane, term) in &terms {
        x[usize::from(lane)] = term;
    }
    let want = console_exclusive_prefix(&x);
    assert_eq!((want[4], want[3]), (0.0, 0.0), "the console's lanes 4 and 3, by its own additions");
    let mut body = lane_terms(&terms);
    body.extend(exclusive_prefix_sum_f32_v7_to_v2());
    body.push(vop2(V_LSHLREV_B32, 9, k(2), 0)); // v9 = 4 thread
    body.extend(store_at(2, 9, 0)); // dword thread: the lane's prefix
    let program = native(&body, false);
    let shader = exact(&program, &[("v_add_f32", 7), ("v_readlane_b32", 6), ("v_cndmask_b32", 8), ("buffer_store_dword", 1)]);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    assert_eq!(f32::from_bits(out[4]), 0.0, "lane 4: the four lanes below, added in pairs as the program adds them");
    for lane in 0..64 {
        assert_eq!(
            out[lane],
            want[lane].to_bits(),
            "lane {lane}: {} where the console's additions give {}",
            f32::from_bits(out[lane]),
            want[lane]
        );
    }
}

/// `program` as the runtime translates it on a host of 32-lane waves ([`super::emit`]): what
/// comes out, and the forms tried.
fn tried(program: &NativeProgram) -> (Result<HostShader, crate::ShaderError>, Vec<String>) {
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("a container");
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &program.layout, program.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, program).expect("io::raise");
    let was = (lower::TEST_HOST_WAVE.replace(Some(32)), super::TEST_WAVE_FORMS.replace(true));
    let shader = super::emit(&c, program, &p, &st, &io, &res);
    let forms = super::take_wave_forms();
    lower::TEST_HOST_WAVE.set(was.0);
    super::TEST_WAVE_FORMS.set(was.1);
    (shader, forms)
}

/// The meeting form must not take `program`: every meeting form tried is refused, one of
/// them for `reason` (a group barrier would stand where some thread does not come, or the
/// answer would not be the console's). Nothing is run: such a program can hang a device.
fn refused_by_the_meeting(program: &NativeProgram, reason: &str) {
    let (_, forms) = tried(program);
    let tried = forms.join("\n  ");
    let meetings: Vec<&String> = forms.iter().filter(|f| f.starts_with("meeting")).collect();
    assert!(!meetings.is_empty(), "the meeting form was not tried\n  {tried}");
    for m in &meetings {
        assert!(!m.contains(": wave "), "taken in the meeting form: {m}\n  {tried}");
    }
    assert!(forms.iter().any(|f| f.contains(reason)), "not refused for \"{reason}\"\n  {tried}");
}

const S_BARRIER: u32 = 0xBF8A_0000;
/// `s_cbranch_scc0` over the next `n` words: a skip the skip-free forms keep.
const S_CBRANCH_SCC0: u32 = 0xBF84_0000;
const V_CMP_GT_U32: u16 = 0xC4;

/// The second review's Question A, a loop the lanes leave one by one: the loop's one block
/// reads lane 37 (the wave must meet) and ends in the branch back, which each lane takes
/// while its own count is short: threads 0..31 go round twice, threads 32..63 once. Every
/// thread runs the block the first time; the second time half of them have left, and a
/// group barrier there waits for threads that never come. The meeting form must refuse it.
#[test]
fn a_meeting_in_a_loop_the_lanes_leave_one_by_one_is_refused() {
    refused_by_the_meeting(&a_loop_the_lanes_leave_one_by_one(), "in a loop the lanes leave one by one");
}

/// The program of the test above.
fn a_loop_the_lanes_leave_one_by_one() -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop1(V_MOV_B32, 10, k(0)),       // v10 = 0
        vop1(V_MOV_B32, 6, k(0)),        // v6 = 0: the rounds done
        vop2(V_LSHRREV_B32, 5, k(5), 0), // v5 = thread >> 5
        vop2(V_SUB_NC_U32, 5, k(2), 5),  // v5 = 2 - v5: the rounds to do
        0xBE8E_0480,                     // s_mov_b64 s[14:15], 0
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // loop: s7 = v7 of lane 37
    body.extend([
        vop2(V_ADD_NC_U32, 10, sgpr(7), 10), // v10 += s7
        vop2(V_ADD_NC_U32, 6, k(1), 6),      // v6 += 1
        vopc(V_CMP_EQ_U32, vgpr(6), 5),      // vcc = (v6 == v5): the lane is done
        0x888E_0E6A,                         // s_or_b64 s[14:15], vcc, s[14:15]
        0x8AFE_0E7E,                         // s_andn2_b64 exec, exec, s[14:15]
        0xBF89_FFF8,                         // s_cbranch_execnz loop
        0x88FE_0E7E,                         // s_or_b64 exec, exec, s[14:15]
        vop2(V_LSHLREV_B32, 9, k(2), 0),     // v9 = 4 thread
    ]);
    body.extend(store_at(10, 9, 0));
    native(&body, false)
}

/// A program of [`native`] that reads lane 37 (the wave must meet) and has a barrier of its
/// own, inside `if (thread < 32)` (`inside`) or before it.
fn a_barrier_of_the_programs_own(inside: bool) -> NativeProgram {
    let mut body = vec![vop2(V_ADD_NC_U32, 7, 255, 0), 0x100]; // v7 = 0x100 + thread
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    if !inside {
        body.push(S_BARRIER);
    }
    body.extend([
        vopc(V_CMP_GT_U32, k(32), 0), // vcc = (32 > thread)
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | (1 + u32::from(inside)),
    ]);
    if inside {
        body.push(S_BARRIER);
    }
    body.extend([
        vop1(V_MOV_B32, 12, k(1)), // v12 = 1 in the threads below 32
        RESTORE_EXEC_S12,
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
    ]);
    body.extend(store_at(11, 9, 0));
    native(&body, false)
}

/// The second review's Question A, the program's own barrier: `s_barrier` waits for the
/// group like a barrier of a meeting, so in the meeting form it may only stand where every
/// thread comes. Inside `if (thread < 32)` 32 threads would wait for 32 that never come:
/// refused. Before the branch every thread reaches it: taken, and run.
#[test]
fn a_barrier_of_the_programs_own_where_not_every_thread_comes_is_refused() {
    let _gate = gate!();
    refused_by_the_meeting(&a_barrier_of_the_programs_own(true), "a barrier of the program's own");
    let program = a_barrier_of_the_programs_own(false);
    let shader = exact(&program, &[("s_barrier", 1), ("v_readlane_b32", 1), ("s_cbranch_scc0", 1)]);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x125, "thread {t}: lane 37 of the console's wave, past the program's barrier");
    }
}

/// A program of [`native`] with LDS: the threads below s0 write 1 to the first dword of
/// LDS inside `if (thread < s0)`, and with `twice` then 2; every thread reads the dword
/// after the branch.
fn lds_writes_inside_a_lane_branch(twice: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 8, k(0)),        // v8 = 0: the first dword of LDS
        vop1(V_MOV_B32, 14, k(1)),
        vop1(V_MOV_B32, 15, k(2)),
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | if twice { 4 } else { 2 },
    ];
    body.extend(ds(DS_WRITE_B32, 0, 8, 14)); // LDS[0] = 1
    if twice {
        body.extend(ds(DS_WRITE_B32, 0, 8, 15)); // LDS[0] = 2
    }
    body.push(RESTORE_EXEC_S12);
    body.extend(ds(DS_READ_B32, 10, 8, 0)); // v10 = LDS[0]
    body.extend(store_at(10, 9, 0));
    native(&body, true)
}

/// The second review's Question B, LDS writes inside a branch only some lanes take: the
/// waves meet where the branch begins and ends, not between the instructions in it, so two
/// writes there can land in the other order (a thread of one host wave at the second
/// before a thread of another at the first). One write keeps the console's result: taken,
/// and run. Two are refused.
#[test]
fn two_lds_writes_inside_a_lane_branch_are_refused() {
    let _gate = gate!();
    refused_by_the_meeting(&lds_writes_inside_a_lane_branch(true), "LDS writes where not every lane runs");
    let program = lds_writes_inside_a_lane_branch(false);
    let shader = exact(&program, &[("ds_write_b32", 1), ("ds_read_b32", 1), ("s_cbranch_scc0", 1)]);
    let Some(out) = run(&shader, [40, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 1, "thread {t} reads what the threads below 40 wrote");
    }
}

/// The second review's Question B, the program's LDS addresses and the meeting's dwords:
/// the meeting keeps its dwords after the program's 128. Thread 5 writes 1 at byte 768,
/// dword 192, which is past the program's LDS (the console drops the write) and is the
/// dword that holds the bits of lanes 0..31 at a meeting. The first lane of EXEC kept to
/// the threads from 37 on must still be lane 37 (0x125), not lane 0; and a read past the
/// program's LDS (byte 512, where the meeting keeps thread 0's value) gives 0.
#[test]
fn an_lds_address_past_the_programs_own_does_not_reach_the_meetings_dwords() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 14, k(1)),       // v14 = 1
        vop1(V_MOV_B32, 15, 255),
        768,                             // v15 = 768
        vop1(V_MOV_B32, 16, 255),
        512,                             // v16 = 512
        vopc(V_CMP_EQ_U32, k(5), 0),     // vcc = (5 == thread)
        SAVEEXEC_S12_VCC,
    ];
    body.extend(ds(DS_WRITE_B32, 0, 15, 14)); // LDS[768] = 1, in thread 5
    body.extend([
        RESTORE_EXEC_S12,
        vopc(V_CMP_LE_U32, sgpr(0), 0), // vcc = (s0 <= thread)
        SAVEEXEC_S12_VCC,
        vop1(V_READFIRSTLANE_B32, 6, vgpr(7)), // s6 = v7 of the first lane of EXEC
        RESTORE_EXEC_S12,
        vop1(V_MOV_B32, 10, sgpr(6)),
    ]);
    body.extend(ds(DS_READ_B32, 11, 16, 0)); // v11 = LDS[512]
    body.extend(store_at(10, 9, 0)); // dword thread: the first lane's
    body.extend(store_at(11, 9, 256)); // dword 64 + thread: the read past the end
    let program = native(&body, true);
    let shader = exact(&program, &[("ds_write_b32", 1), ("ds_read_b32", 1), ("v_readfirstlane_b32", 1)]);
    let Some(out) = run(&shader, [37, 1], 128) else { return };
    for t in 0..64 {
        assert_eq!(out[t], 0x125, "thread {t}: the first lane of EXEC is console lane 37");
        assert_eq!(out[64 + t], 0, "thread {t}: a read past the program's LDS");
    }
}

/// The second review's Question A (second finding, which names no fault and asks for
/// cover): a group that is not whole console waves. 65 threads are a wave of 64 and a wave
/// of one thread. Until the eighth review the form took it, each wave answering for itself
/// (the first lane's value thread 0's and thread 64's, the count of EXEC 64 and 1), and ran
/// it so. Every review from the sixth on found another value one for each wave that the
/// proof took for one of the group, so the form takes groups of one console wave only now
/// (`meet::SEVERAL_WAVES`): this program and the same in a group of 128 are refused. The
/// same in a group of 64: taken, and run.
#[test]
fn a_group_of_several_console_waves_is_refused() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                                 // v7 = 0x100 + thread
        vop1(V_READFIRSTLANE_B32, 6, vgpr(7)), // s6 = v7 of the wave's first lane
        vop1(V_MOV_B32, 10, sgpr(6)),
        0xBE87_107E,                           // s_bcnt1_i32_b64 s7, exec
        vop1(V_MOV_B32, 11, sgpr(7)),
        vop2(V_LSHLREV_B32, 9, k(2), 0),       // v9 = 4 thread
    ];
    body.extend(store_at(10, 9, 0)); // dword thread: the first lane's
    body.extend(store_at(11, 9, 260)); // dword 65 + thread: the count
    let names = [("v_readfirstlane_b32", 1), ("s_bcnt1_i32_b64", 1), ("buffer_store_dword", 2)];
    for threads in [65, 128] {
        let program = in_a_group_of(native(&body, false), threads);
        has(&program, &names);
        refused_by_the_meeting(&program, SEVERAL);
    }
    let shader = exact(&native(&body, false), &names);
    let Some(out) = run(&shader, [0, 1], 130) else { return };
    for t in 0..64 {
        assert_eq!(out[t], 0x100, "thread {t}: the wave's first lane");
        assert_eq!(out[65 + t], 64, "thread {t}: the lanes of the wave");
    }
}

/// What the form says of a group of several console waves.
const SEVERAL: &str = "a group of several console waves";

/// The same finding's other case: a lane that returns. The threads from 32 on leave the
/// program at a branch (to an end of their own) and the others read lane 37: a meeting
/// there would wait for threads that have returned. Refused, as before the review.
#[test]
fn a_meeting_past_a_branch_some_lanes_return_at_is_refused() {
    refused_by_the_meeting(&a_branch_some_lanes_return_at(), "where not every lane runs");
}

/// The program of the test above.
fn a_branch_some_lanes_return_at() -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                        // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vopc(V_CMP_GT_U32, k(32), 0), // vcc = (32 > thread)
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | 6, // no thread below 32: to the end after this one
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    body.push(S_ENDPGM);
    native(&body, false)
}

/// `s_branch` over the next `n` words (back with a negative `n`).
const fn s_branch(n: i16) -> u32 {
    0xBF82_0000 | n as u16 as u32
}

/// A program of [`native`] with a loop closed by a plain jump and nothing else, beside a
/// meeting: the threads whose compare holds go into the loop, the others past it to a read
/// of lane 37 and the end. `per_lane`: the compare is of the thread index (`32 > thread`:
/// the threads below 32 loop); else of a copy of s0 (`32 > s0`: every thread goes the same
/// way).
fn a_closed_loop_beside_a_meeting(per_lane: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 6, k(0)),        // v6 = 0
        vop1(V_MOV_B32, 21, sgpr(0)),    // v21 = s0
        vopc(V_CMP_GT_U32, k(32), if per_lane { 0 } else { 21 }), // vcc = (32 > thread), or (32 > s0)
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | 2,              // the threads not in it: past the loop
        vop2(V_ADD_NC_U32, 6, k(1), 6),  // loop: v6 += 1
        s_branch(-2),                    // and round again, with no way out
        RESTORE_EXEC_S12,
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    native(&body, false)
}

/// The third review's Question 1, the iteration limit's own return: a loop closed by a plain
/// jump is left at the limit by a return the lowering adds (`Lower::exit`), which the
/// program does not have. Threads 0..31 go into such a loop and return there; threads
/// 32..63 go past it to a meeting, whose barrier would wait for 32 threads that have
/// returned. Without the return the loop has no end, so the meeting's block seemed to come
/// after the branch whichever way a thread went. Refused. With every thread going the same
/// way (all into the loop, or all to the meeting) no thread is missed: taken, and run both
/// ways.
#[test]
fn a_meeting_beside_a_loop_some_lanes_only_leave_at_the_limit_is_refused() {
    let _gate = gate!();
    refused_by_the_meeting(&a_closed_loop_beside_a_meeting(true), "where not every lane runs");
    let program = a_closed_loop_beside_a_meeting(false);
    let shader = exact(&program, &[("s_branch", 1), ("v_readlane_b32", 1), ("s_cbranch_scc0", 1)]);
    let Some(out) = run(&shader, [40, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x125, "thread {t}: lane 37 of the console's wave, every thread past the loop");
    }
    let out = run(&shader, [0, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread leaves at the limit together: nothing is written");
}

/// A program of [`native`] with no end on either side of a branch: the threads whose
/// compare holds go into a loop closed by a plain jump, the others past it to a read of lane
/// 37, a store of it, and a second such loop. Neither loop has a way out, so past the branch
/// no thread reaches the end of the program: each leaves at the iteration limit of the loop
/// it is in. `per_lane`: the compare is of the thread index (`32 > thread`: the threads
/// below 32 take the first loop); else of a copy of s0 (`32 > s0`: every thread goes the
/// same way).
fn two_closed_loops_and_a_meeting_before_the_second(per_lane: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 6, k(0)),        // v6 = 0
        vop1(V_MOV_B32, 21, sgpr(0)),    // v21 = s0
        vopc(V_CMP_GT_U32, k(32), if per_lane { 0 } else { 21 }), // vcc = (32 > thread), or (32 > s0)
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | 2,              // the threads not in it: past the first loop
        vop2(V_ADD_NC_U32, 6, k(1), 6),  // first loop: v6 += 1
        s_branch(-2),                    // and round again, with no way out
        RESTORE_EXEC_S12,
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    body.extend([
        vop2(V_ADD_NC_U32, 6, k(1), 6), // second loop: v6 += 1
        s_branch(-2),                   // and round again, with no way out
    ]);
    native(&body, false)
}

/// The fourth review's Question 2: a branch each lane takes for itself with a closed loop on
/// both sides. Threads 0..31 go into the first loop and return at its limit; threads 32..63
/// read lane 37 at a meeting and then go into the second. With no end of the program past
/// the branch, "every way from the branch to an end passes the meeting" was true of no way at
/// all: nothing called the loops' limits ends (that took a block not every thread runs,
/// which took an end), the meeting's block counted as one every thread runs, and its barrier
/// would wait for 32 threads that have returned. A loop no way leads out of is left at its
/// limit whoever goes round it. Refused, by the selection the runtime uses; nothing is run.
/// With every thread going the same way no one is missed, closed loops or not: taken, and
/// run both ways (all past the first loop: the meeting, the store, then the second loop's
/// limit together; all into the first: its limit together, nothing written).
#[test]
fn a_meeting_between_two_closed_loops_the_lanes_part_into_is_refused() {
    let _gate = gate!();
    let names = [("s_branch", 2), ("v_readlane_b32", 1), ("s_cbranch_scc0", 1)];
    let program = two_closed_loops_and_a_meeting_before_the_second(true);
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    for (name, n) in names {
        let found = p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).count();
        assert_eq!(found, n, "{name} in the fixture's code");
    }
    refused_by_the_meeting(&program, "where not every lane runs");
    let program = two_closed_loops_and_a_meeting_before_the_second(false);
    let shader = exact(&program, &names);
    let Some(out) = run(&shader, [40, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x125, "thread {t}: lane 37 of the console's wave, stored before the second loop");
    }
    let out = run(&shader, [0, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread leaves at the first loop's limit together: nothing is written");
}

/// The control of a loop every thread goes round together: three rounds counted in a
/// scalar, closed by a plain jump (the limit's return stands at that jump, and no thread
/// takes it before the others), then a read of lane 37. Taken, and run: 0x125 and the
/// three rounds.
#[test]
fn a_meeting_past_a_loop_the_threads_go_round_together_is_taken() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 10, k(0)),       // v10 = 0
        0xBE86_0380,                     // s_mov_b32 s6, 0
        0x8006_8106,                     // loop: s_add_u32 s6, s6, 1
        vop2(V_ADD_NC_U32, 10, k(1), 10), // v10 += 1
        0xBF06_8306,                     // s_cmp_eq_u32 s6, 3
        0xBF85_0001,                     // s_cbranch_scc1 past the jump back
        s_branch(-5),                    // round again
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop2(V_ADD_NC_U32, 11, sgpr(7), 10)); // v11 = s7 + v10
    body.extend(store_at(11, 9, 0));
    let program = native(&body, false);
    let shader = exact(&program, &[("s_branch", 1), ("v_readlane_b32", 1), ("s_cmp_eq_u32", 1), ("s_cbranch_scc1", 1)]);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x128, "thread {t}: lane 37 of the console's wave and three rounds");
    }
}

/// `s_cbranch_vccz` over the next `n` words.
const S_CBRANCH_VCCZ: u32 = 0xBF86_0000;
/// `s_cbranch_scc1` over the next `n` words (back with a 16-bit negative `n`).
const S_CBRANCH_SCC1: u32 = 0xBF85_0000;

/// `s_cmp_eq_u32 s<s>, <n>` with an inline constant 0..=64.
const fn s_cmp_eq(s: u8, n: u8) -> u32 {
    0xBF06_0000 | (0x80 + n as u32) << 8 | s as u32
}

/// The instructions `names` counts, in `program`'s code.
fn has(program: &NativeProgram, names: &[(&str, usize)]) {
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    for &(name, n) in names {
        let found = p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).count();
        assert_eq!(found, n, "{name} in the fixture's code");
    }
}

/// Whether the rule of the loops ([`super::lower`]'s `meet::walked`) speaks against a group
/// barrier of `program` in some meeting form tried, whatever else refuses it: the form's
/// record names the loop and the clause.
fn the_loops_rule_refuses(program: &NativeProgram) -> bool {
    let (_, forms) = tried(program);
    assert!(forms.iter().any(|f| f.starts_with("meeting")), "the meeting form was not tried\n  {}", forms.join("\n  "));
    forms.iter().any(|f| f.starts_with("not walked together: "))
}

/// A program of [`native`] with one loop (header H) that has two ways round. At H the
/// threads whose compare holds go to X, the others straight to M. X goes back to H while
/// `s3 != 0` (a branch back: the lowering counts it, and at the iteration limit the thread
/// returns, its other arm, M, being in the loop); M reads lane 37 (the wave must meet), adds it up, and leaves the loop
/// when `s2 == 0`; else L jumps back to H (a plain jump: at the limit the thread returns).
/// `per_lane`: H's compare is of the thread index (`32 > thread`: the threads below 32 take
/// X); else of a copy of s0 (`32 > s0`: every thread goes the same way).
fn a_counted_way_back_beside_a_meeting_in_one_loop(per_lane: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 10, k(0)),       // v10 = 0
        vop1(V_MOV_B32, 21, sgpr(0)),    // v21 = s0
        vopc(V_CMP_GT_U32, k(32), if per_lane { 0 } else { 21 }), // H: vcc = (32 > thread), or (32 > s0)
        S_CBRANCH_VCCZ | 2,              // those whose compare fails: to M
        s_cmp_eq(3, 0),                  // X: s3 == 0
        S_CBRANCH_SCC0 | 0xFFFC,         // while s3 != 0: back to H
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // M: s7 = v7 of lane 37
    body.extend([
        vop2(V_ADD_NC_U32, 10, sgpr(7), 10), // v10 += s7
        s_cmp_eq(2, 0),                      // s2 == 0
        S_CBRANCH_SCC1 | 1,                  // then out of the loop
        s_branch(-10),                       // L: round again, to H
    ]);
    body.extend(store_at(10, 9, 0));
    native(&body, false)
}

/// The fifth review's Question 1, a loop one half goes round by a counted way back while the
/// other meets. Threads 0..31 went round H and X (X's branch back is counted, and no meeting
/// stands on that way) until the iteration limit sent them on to M, their counter one short
/// of the limit; threads 32..63 had met at M alone and gone round by L, their counter at 1.
/// At L the first returned (the limit) and the others went round to a second meeting, whose
/// barrier waited for 32 threads that had returned. The form took it. Refused by the rule of
/// the loops (H's branch is each lane's own and a way leads back to it, and X, which goes
/// back, stands where only some lanes run), and, now that the limit returns a thread at X
/// (`crate::bounds::cut_at_limit`: M is in the loop), as a meeting where not every thread
/// comes: X's return is an end where only some lanes run. Nothing is run.
/// With every thread going the same way at H the counters move alike and the limit is
/// everyone's or no one's: taken, and run three ways (all round X to the limit and its
/// return there together, nothing written; all straight to M and out; all straight to M
/// and round by L to its limit, the return at L's jump together, nothing written).
#[test]
fn a_meeting_in_a_loop_one_half_goes_round_by_a_counted_way_back_is_refused() {
    let _gate = gate!();
    let names = [
        ("s_cbranch_vccz", 1),
        ("s_cbranch_scc0", 1),
        ("s_cbranch_scc1", 1),
        ("s_branch", 1),
        ("s_cmp_eq_u32", 2),
        ("v_readlane_b32", 1),
    ];
    let program = a_counted_way_back_beside_a_meeting_in_one_loop(true);
    has(&program, &names);
    refused_by_the_meeting(&program, "a meeting of the wave's lanes where not every lane runs");
    assert!(the_loops_rule_refuses(&program));
    let program = a_counted_way_back_beside_a_meeting_in_one_loop(false);
    let shader = exact(&program, &names);
    let Some(out) = run_with(&shader, &[0, 1, 0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0), "every thread returns at X's limit together: nothing is written");
    let out = run_with(&shader, &[40, 1, 0, 0], 64).expect("the device that just ran");
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x125, "thread {t}: lane 37 of the console's wave, every thread straight to the meeting");
    }
    let out = run_with(&shader, &[40, 1, 1, 0], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread straight to M and round by L to its limit, returning there together: nothing is written");
}

/// A program of [`native`] with a loop in a loop, every branch on scalars the whole group
/// shares. The outer loop's first block (O) counts its rounds in v11 and jumps to the inner
/// loop's (I), which reads lane 37 (the wave must meet), adds it up in v10, counts its
/// rounds in s6 and leaves both loops when `s6 == s0`. Else a first block goes back to I in
/// an odd round, and a second block goes back to O in round 2 and to I in the other even
/// rounds. `one_block`: both ways back of that second block are the two arms of its branch
/// (to O taken, to I by falling into it); else the fall passes a block of its own (one
/// instruction) that goes on to I, and every block has one way back.
fn two_ways_back_from_one_block(one_block: bool) -> NativeProgram {
    let j = u32::from(!one_block);
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                             // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0),   // v9 = 4 thread
        vop1(V_MOV_B32, 10, k(0)),         // v10 = 0
        vop1(V_MOV_B32, 11, k(0)),         // v11 = 0
        0xBE86_0380,                       // s_mov_b32 s6, 0
        vop2(V_ADD_NC_U32, 11, k(1), 11),  // O: v11 += 1
        s_branch(4 + j as i16),            // to I, over the blocks that go back
        0x870F_8106,                       // s_and_b32 s15, s6, 1: SCC = (s6 is odd)
        S_CBRANCH_SCC1 | (2 + j),          // an odd round: back to I
        s_cmp_eq(6, 2),                    // s6 == 2
        S_CBRANCH_SCC1 | 0xFFFA,           // round 2: back to O; else on, to I
    ];
    if !one_block {
        body.push(vop1(V_MOV_B32, 12, k(1))); // a block of its own on the way to I
    }
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // I: s7 = v7 of lane 37
    body.extend([
        vop2(V_ADD_NC_U32, 10, sgpr(7), 10), // v10 += s7
        0x8006_8106,                         // s_add_u32 s6, s6, 1
        0xBF06_0006,                         // s_cmp_eq_u32 s6, s0
        S_CBRANCH_SCC1 | 1,                  // s6 == s0: out of both loops
        s_branch(-(11 + j as i16)),          // else to the blocks that go back
    ]);
    body.extend(store_at(10, 9, 0)); // dword thread: the lane reads added up
    body.extend(store_at(11, 9, 256)); // dword 64 + thread: the outer loop's rounds
    native(&body, false)
}

/// The fifth review's Question 3, a block both of whose ways go back (to the outer loop's
/// header by its branch, to the inner loop's by falling into it): the lowering counts the
/// first and gives the second no count, so the inner loop's counter has no value on that way
/// and nothing says the threads' counters move alike, or that the limit ever ends the loop.
/// Every branch here is one the whole group takes alike and every block one every thread
/// runs, so the meeting in the inner loop was taken. Each arm is counted now
/// (`back_edges`: the outer arm, whose other arm is in its loop, returns the thread at its
/// limit), the counters move alike and the limit is everyone's or no one's: taken, and run
/// as the form whose fall through is a block of its own (five rounds of the inner loop, by
/// each of the three ways back, and two of the outer).
#[test]
fn a_meeting_in_a_loop_with_a_block_both_of_whose_ways_go_back_is_taken() {
    let _gate = gate!();
    let names = [
        ("s_branch", 2),
        ("s_cbranch_scc1", 3),
        ("s_and_b32", 1),
        ("s_cmp_eq_u32", 2),
        ("v_readlane_b32", 1),
    ];
    for one_block in [true, false] {
        let program = two_ways_back_from_one_block(one_block);
        if one_block {
            has(&program, &names);
        }
        let shader = exact(&program, &names);
        let Some(out) = run(&shader, [5, 1], 128) else { return };
        for t in 0..64 {
            assert_eq!(out[t], 5 * 0x125, "thread {t}, one block {one_block}: lane 37 of the console's wave in each of five rounds");
            assert_eq!(out[64 + t], 2, "thread {t}, one block {one_block}: the outer loop's two rounds");
        }
    }
}

/// What the rule of the loops must leave standing: a branch each lane takes for itself
/// inside a loop the threads walk together, both of whose arms join again before the block
/// that goes back. The loop's first block ends in `if (thread < 32)` (the threads below 32
/// add 1 to v12), every thread then reads lane 37 and the block that reads it goes round
/// three times, counted in a scalar. The loop leaves only at its end, so a way leads from
/// the lane branch back to the loop's first block: that block still counts as one every
/// thread runs, the way passing the block where the arms join. Taken, and run: three reads
/// of lane 37 in every thread, three additions in the threads below 32 and none above.
#[test]
fn a_lane_branch_inside_a_loop_the_threads_walk_together_is_taken() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                            // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0),  // v9 = 4 thread
        vop1(V_MOV_B32, 10, k(0)),        // v10 = 0
        vop1(V_MOV_B32, 12, k(0)),        // v12 = 0
        0xBE86_0380,                      // s_mov_b32 s6, 0
        vopc(V_CMP_GT_U32, k(32), 0),     // loop: vcc = (32 > thread)
        SAVEEXEC_S12_VCC,
        S_CBRANCH_SCC0 | 1,               // no thread below 32: over the addition
        vop2(V_ADD_NC_U32, 12, k(1), 12), // v12 += 1 in the threads below 32
        RESTORE_EXEC_S12,
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.extend([
        vop2(V_ADD_NC_U32, 10, sgpr(7), 10), // v10 += s7
        0x8006_8106,                         // s_add_u32 s6, s6, 1
        s_cmp_eq(6, 3),                      // s6 == 3
        S_CBRANCH_SCC0 | 0xFFF5,             // else round again
    ]);
    body.extend(store_at(10, 9, 0)); // dword thread: the lane reads added up
    body.extend(store_at(12, 9, 256)); // dword 64 + thread: the additions of the threads below 32
    let program = native(&body, false);
    assert!(!the_loops_rule_refuses(&program));
    let shader = exact(&program, &[("s_cbranch_scc0", 2), ("v_readlane_b32", 1), ("s_cmp_eq_u32", 1), ("s_add_u32", 1)]);
    let Some(out) = run(&shader, [0, 1], 128) else { return };
    for t in 0..64 {
        assert_eq!(out[t], 3 * 0x125, "thread {t}: lane 37 of the console's wave in each of three rounds");
        assert_eq!(out[64 + t], if t < 32 { 3 } else { 0 }, "thread {t}: the additions inside the lane branch");
    }
}

/// A program of [`native`] whose read of lane 37 (the wave must meet) stands behind a branch
/// on `v10 == 1`, where `v10` is 1 or 0 by a select (`v_cndmask_b32`). `pair`: the select is
/// on the scalar pair s[14:15] = 0xFFFF_FFFF, 0, which no mask is known for: the lanes below
/// 32 get 1 and the others 0. Else on the compare `32 > s0` of a copy of s0, one for the wave.
fn a_select_before_a_branch_around_a_meeting(pair: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
        vop1(V_MOV_B32, 21, sgpr(0)),    // v21 = s0
        0xBE8E_03C1,                     // s_mov_b32 s14, -1
        0xBE8F_0380,                     // s_mov_b32 s15, 0
        vopc(V_CMP_GT_U32, k(32), 21),   // vcc = (32 > s0)
        0xD501_000A,
        0x80 | 0x81 << 9 | (if pair { 14 } else { 106 }) << 18, // v_cndmask_b32 v10, 0, 1, s[14:15] (or vcc)
        vopc(V_CMP_EQ_U32, k(1), 10),    // vcc = (1 == v10)
        S_CBRANCH_VCCZ | 3,              // the lanes whose v10 is not 1: past the read
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    native(&body, false)
}

/// The sixth review's first claim: a select on a scalar pair that is no mask the structure
/// follows gives each lane the bit of its own number, so what it writes differs between the
/// lanes though both scalars are one for the wave. The proof took the result for one value,
/// the branch on it for one every thread takes alike, and the meeting behind it was taken
/// with threads 32..63 never coming. Refused now: such a select's result is each lane's
/// own. A select on a mask the proof follows (a compare of a value the wave shares) stays
/// one value: taken, and run both ways.
#[test]
fn a_select_on_a_scalar_pair_that_is_no_known_mask_is_each_lanes_own() {
    let _gate = gate!();
    let names = [("v_cndmask_b32", 1), ("s_cbranch_vccz", 1), ("v_readlane_b32", 1)];
    let program = a_select_before_a_branch_around_a_meeting(true);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_select_before_a_branch_around_a_meeting(false), &names);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "every thread reads lane 37: {out:x?}");
    let out = run(&shader, [40, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread goes past the read: {out:x?}");
}

/// A program of [`native`] that reads the lane of v7 whose number is in s6 (a first
/// meeting), and a second time lane 37 behind `if (s7 == s3)` on what the first read gave.
/// s6 is 1 where the compare holds and 0 elsewhere. `per_lane`: the compare is of the thread
/// index (`32 > thread`: the halves ask for different lanes and get different answers); else
/// of a copy of s0 (`32 > s0`: one lane number for the wave).
fn a_lane_read_at_a_number_in_a_scalar(per_lane: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
        vop1(V_MOV_B32, 21, sgpr(0)),    // v21 = s0
        0xBE86_0380,                     // s_mov_b32 s6, 0
        vopc(V_CMP_GT_U32, k(32), if per_lane { 0 } else { 21 }), // vcc = (32 > thread), or (32 > s0)
        S_CBRANCH_VCCZ | 1,              // those whose compare fails: over the next
        0xBE86_0381,                     // s_mov_b32 s6, 1
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), sgpr(6))); // s7 = v7 of lane s6
    body.extend([
        0xBF06_0307,        // s_cmp_eq_u32 s7, s3
        S_CBRANCH_SCC0 | 3, // not what s3 names: past the second read
    ]);
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    native(&body, false)
}

/// The sixth review's second claim: a lane read answers for the lane each thread names. The
/// proof took its result for one value of the wave whatever the lane number was, so with
/// the halves naming lanes 1 and 0 the branch on the result counted as taken alike and the
/// second meeting behind it was taken, where only one half comes. Refused now: the result
/// is one value only where the lane number is. With one lane number for the wave: taken,
/// and run three ways (lane 1 read and matched, read and not matched, lane 0 read and
/// matched).
#[test]
fn a_lane_read_at_a_lane_number_the_halves_hold_differently_is_not_one_value() {
    let _gate = gate!();
    let names = [("v_readlane_b32", 2), ("s_cbranch_vccz", 1), ("s_cbranch_scc0", 1), ("s_cmp_eq_u32", 1)];
    let program = a_lane_read_at_a_number_in_a_scalar(true);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_lane_read_at_a_number_in_a_scalar(false), &names);
    let Some(out) = run_with(&shader, &[0, 1, 0, 0x101], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "lane 1 holds 0x101: every thread reads lane 37: {out:x?}");
    let out = run_with(&shader, &[0, 1, 0, 0x100], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "lane 1 does not hold 0x100: every thread goes past: {out:x?}");
    let out = run_with(&shader, &[40, 1, 0, 0x100], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0x125), "lane 0 holds 0x100: every thread reads lane 37: {out:x?}");
}

/// `v_swap_b32 vdst, vsrc`.
const V_SWAP_B32: u16 = 0x65;

/// A program of [`native`] whose read of lane 37 stands behind a branch on `32 > v10`,
/// where v10 is 0 and then takes v5's value by `v_swap_b32 v5, v10`. `per_lane`: v5 is the
/// thread index; else a copy of s0.
fn a_swap_before_a_branch_around_a_meeting(per_lane: bool) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
        vop1(V_MOV_B32, 5, if per_lane { vgpr(0) } else { sgpr(0) }), // v5 = thread, or s0
        vop1(V_MOV_B32, 10, k(0)),       // v10 = 0
        vop1(V_SWAP_B32, 5, vgpr(10)),   // v5 and v10 change places
        vopc(V_CMP_GT_U32, k(32), 10),   // vcc = (32 > v10)
        S_CBRANCH_VCCZ | 3,              // those whose compare fails: past the read
    ];
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(store_at(11, 9, 0));
    native(&body, false)
}

/// The sixth review's third claim: a swap writes each of its two registers with what the
/// other held, so both old values are operands. The proof left the first register's old
/// value out as one the instruction only writes over, and with the thread index there and 0
/// in the second it took the second, after the swap, for one value of the wave: the branch
/// on it counted as taken alike and the meeting behind it was taken, where threads 32..63
/// never come. Refused now. With a value the wave shares in the first register: taken, and
/// run both ways.
#[test]
fn a_swap_gives_each_register_what_the_other_held() {
    let _gate = gate!();
    let names = [("v_swap_b32", 1), ("s_cbranch_vccz", 1), ("v_readlane_b32", 1)];
    let program = a_swap_before_a_branch_around_a_meeting(true);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_swap_before_a_branch_around_a_meeting(false), &names);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "every thread reads lane 37: {out:x?}");
    let out = run(&shader, [40, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread goes past the read: {out:x?}");
}

/// What the program of [`a_scalar_load_from_memory_before_a_branch_around_a_meeting`] stores
/// before its load.
#[derive(Clone, Copy)]
enum Stores {
    Nothing,
    /// every thread its v7 at the address the load reads (`global_store_dword`)
    ByAddress,
    /// every thread its v7 through the UAV's descriptor (`buffer_store_dword`): a piece of
    /// memory the proof knows, which is title memory too and may be where the load reads
    ThroughTheUav,
}

/// A program of [`native`] whose read of lane 37 stands behind `if (s6 == 0)`, where s6 is
/// a dword loaded from memory at the address the root constants s[2:3] hold
/// (`s_load_dword`), after what `stores` says. What the read gives goes to LDS
/// (`ds_write_b32`), which is no title memory: `stores` is all the program stores there.
fn a_scalar_load_from_memory_before_a_branch_around_a_meeting(stores: Stores) -> NativeProgram {
    let mut body = vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
        vop1(V_MOV_B32, 2, sgpr(2)),     // v[2:3] = s[2:3]
        vop1(V_MOV_B32, 3, sgpr(3)),
    ];
    match stores {
        Stores::Nothing => {}
        Stores::ByAddress => body.extend([0xDC70_8000, 0x007D_0702]), // global_store_dword v[2:3], v7
        Stores::ThroughTheUav => body.extend(store(7, 0x400)),        // buffer_store_dword v7, off, s[8:11]
    }
    body.extend([
        0xF400_0181,
        0xFA00_0000,        // s_load_dword s6, s[2:3], 0
        s_cmp_eq(6, 0),     // s6 == 0
        S_CBRANCH_SCC0 | 3, // else past the read
    ]);
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), k(37))); // s7 = v7 of lane 37
    body.push(vop1(V_MOV_B32, 11, sgpr(7)));
    body.extend(ds(DS_WRITE_B32, 0, 9, 11)); // LDS dword thread = v11
    native(&body, true)
}

/// The sixth review's fourth claim, its second half: a scalar load from memory by address
/// counted as a constant of the program whatever the program stores. Where the program also
/// stores to memory by address, a host wave that loads before another has stored reads
/// another dword than one that loads after, and the branch on it is not one the group takes
/// alike: the meeting behind it must not be taken. Refused now. Without the store the load
/// is a constant of the run and the meeting is taken (translated and validated; not run:
/// the fixtures' device has no copy of title memory to load from). (Until the seventh
/// review the control kept the result in the UAV; a store through a descriptor is one to
/// title memory too, so the program writes its result to LDS now, in all three cases.)
#[test]
fn a_scalar_load_from_memory_the_program_stores_to_is_no_constant() {
    let names = [("s_load_dword", 1), ("s_cbranch_scc0", 1), ("v_readlane_b32", 1)];
    let program = a_scalar_load_from_memory_before_a_branch_around_a_meeting(Stores::ByAddress);
    has(&program, &names);
    has(&program, &[("global_store_dword", 1)]);
    refused_by_the_meeting(&program, "where not every lane runs");
    exact(&a_scalar_load_from_memory_before_a_branch_around_a_meeting(Stores::Nothing), &names);
}

/// The seventh review's first claim, the same with the store through a descriptor the proof
/// knows the piece of memory of (a UAV of a table): that piece is title memory too, and the
/// load by address may read the very dwords the program writes, so the load is no constant
/// of the run there either. Refused now. The control is the program without the store above
/// (translated and validated, not run: the fixtures' device has no title memory).
#[test]
fn a_scalar_load_from_memory_beside_a_store_through_a_descriptor_is_no_constant() {
    let names = [("s_load_dword", 1), ("s_cbranch_scc0", 1), ("v_readlane_b32", 1), ("buffer_store_dword", 1)];
    let program = a_scalar_load_from_memory_before_a_branch_around_a_meeting(Stores::ThroughTheUav);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
}

/// `s_branch` over the next `n` words.
const S_BRANCH: u32 = 0xBF82_0000;
/// `v_cmpx_gt_u32 exec, src0, vsrc1` (VOPC).
const V_CMPX_GT_U32: u16 = 0xD4;

/// `program` with a group of `threads` threads.
fn in_a_group_of(mut program: NativeProgram, threads: u32) -> NativeProgram {
    let StageIo::Cs(cs) = &mut program.io else { unreachable!() };
    cs.threads = [threads, 1, 1];
    program
}

/// The words of a read of lane `lane` of v7 into v11 (`v_readlane_b32 s7, v7, lane`: the
/// wave must meet) and the store of v11 at dword `thread`.
fn read_lane_and_store(lane: u16) -> Vec<u32> {
    let mut words = vop3(V_READLANE_B32, 7, vgpr(7), k(lane)).to_vec();
    words.push(vop1(V_MOV_B32, 11, sgpr(7)));
    words.extend(store_at(11, 9, 0));
    words
}

/// v7 = 0x100 + thread, v9 = 4 thread, v11 = 0: what [`read_lane_and_store`] reads and
/// where it stores.
fn lanes_and_places() -> Vec<u32> {
    vec![
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x100,                           // v7 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
    ]
}

/// A program of [`native`] in a group of `threads` whose read of lane 37 stands behind
/// `if (s6 == 0)`, where s6 is joined from the two arms of `if (s0 == 0)`: 0, or the low
/// dword of the lane mask `lanes > thread` read as a number (`v_cmp_gt_u32 s[6:7]`): the
/// ballot of each console wave.
fn a_mask_joined_with_a_scalar_before_a_branch_around_a_meeting(threads: u32, lanes: u16) -> NativeProgram {
    let mut body = lanes_and_places();
    body.extend([
        s_cmp_eq(0, 0),     // s0 == 0
        S_CBRANCH_SCC0 | 2, // else: to the compare
        0xBE86_0380,        // s_mov_b32 s6, 0
        S_BRANCH | 2,       // to the join
    ]);
    body.extend(vop3(V_CMP_GT_U32, 6, k(lanes), vgpr(0))); // s[6:7] = the mask (lanes > thread)
    body.extend([
        s_cmp_eq(6, 0),     // the join: s6 == 0
        S_CBRANCH_SCC0 | 3, // else past the read
    ]);
    body.extend(read_lane_and_store(37));
    in_a_group_of(native(&body, false), threads)
}

/// The seventh review's second claim: with several console waves to a group, a lane mask
/// read as a number is each wave's own ballot (`Lower::mask_dword`), and a join of it with a
/// scalar took it for one value of the group because masks stay in the proof by their lane
/// bits. In a group of 128 with the mask `64 > thread`, the first wave gets 0xFFFF_FFFF and
/// skips, the second 0 and meets alone. Refused (since the eighth review as a group of
/// several console waves). With one console wave to the group the ballot is the wave's one
/// answer: taken, and run three ways (the scalar's arm, and the mask's with a ballot of
/// 0xFFFF_FFFF and of 0).
#[test]
fn a_mask_joined_with_a_scalar_is_each_console_waves_own() {
    let _gate = gate!();
    let names = [("v_cmp_gt_u32", 1), ("s_cbranch_scc0", 2), ("v_readlane_b32", 1)];
    let program = a_mask_joined_with_a_scalar_before_a_branch_around_a_meeting(128, 64);
    has(&program, &names);
    refused_by_the_meeting(&program, SEVERAL);
    let half = exact(&a_mask_joined_with_a_scalar_before_a_branch_around_a_meeting(64, 32), &names);
    let none = exact(&a_mask_joined_with_a_scalar_before_a_branch_around_a_meeting(64, 0), &names);
    let Some(out) = run(&half, [0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "s6 = 0: every thread reads lane 37: {out:x?}");
    let out = run(&half, [1, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "s6 = the ballot 0xFFFF_FFFF: every thread goes past: {out:x?}");
    let out = run(&none, [1, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0x125), "s6 = the ballot 0: every thread reads lane 37: {out:x?}");
}

/// A program of [`native`] in a group of `threads` whose read of lane 0 stands behind
/// `if (v10 == 1)`, where v10 is a select of `src1` where the mask `0 == v11` holds (every
/// lane): `v_cndmask_b32 v10, 0, src1, vcc`.
fn a_select_of_a_number_before_a_branch_around_a_meeting(threads: u32, src1: u16) -> NativeProgram {
    let mut body = lanes_and_places();
    body.extend([
        vopc(V_CMP_EQ_U32, k(0), 11), // vcc = (0 == v11): every lane
        0xD501_000A,
        0x80 | u32::from(src1) << 9 | 106 << 18, // v_cndmask_b32 v10, 0, src1, vcc
        vopc(V_CMP_EQ_U32, k(1), 10),            // vcc = (1 == v10)
        S_CBRANCH_VCCZ | 3,                      // the lanes whose v10 is not 1: past the read
    ]);
    body.extend(read_lane_and_store(0));
    in_a_group_of(native(&body, false), threads)
}

/// The seventh review's third claim: a select (`v_cndmask_b32`) reads its first two
/// operands as numbers, a lane mask among them as each console wave's ballot, but the proof
/// asked only of other instructions whether they read a mask as a number. With EXEC there in
/// a group of 65, the first wave selects 0xFFFF_FFFF and skips, the second 1 and meets
/// alone. Refused (since the eighth review as a group of several console waves; a select of
/// a root constant there too). With one console wave EXEC as a number is the wave's one:
/// taken, and run both ways (EXEC's high dword is 1 in a group of 33, 0xFFFF_FFFF in one of
/// 64).
#[test]
fn a_select_of_a_mask_as_a_number_is_each_console_waves_own() {
    let _gate = gate!();
    let names = [("v_cndmask_b32", 1), ("s_cbranch_vccz", 1), ("v_readlane_b32", 1)];
    for src1 in [126, sgpr(0)] {
        // exec_lo, s0
        let program = a_select_of_a_number_before_a_branch_around_a_meeting(65, src1);
        has(&program, &names);
        refused_by_the_meeting(&program, SEVERAL);
    }
    let one = exact(&a_select_of_a_number_before_a_branch_around_a_meeting(33, 127), &names); // exec_hi
    let all = exact(&a_select_of_a_number_before_a_branch_around_a_meeting(64, 127), &names);
    let Some(out) = run(&one, [0, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, if t < 33 { 0x100 } else { 0 }, "thread {t}: lane 0 (a group of 33)");
    }
    let out = run(&all, [0, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "every thread goes past the read: {out:x?}");
}

/// A program of [`native`] in a group of `threads` whose read of lane 37 of v7 stands
/// behind `if (s7 == 1)`, where s7 is lane 37 of v12 = 1 (`v_readlane_b32 s7, v12, 37`).
fn a_lane_read_of_a_lane_past_the_group(threads: u32) -> NativeProgram {
    let mut body = lanes_and_places();
    body.push(vop1(V_MOV_B32, 12, k(1))); // v12 = 1
    body.extend(vop3(V_READLANE_B32, 7, vgpr(12), k(37))); // s7 = v12 of lane 37
    body.extend([
        s_cmp_eq(7, 1),     // s7 == 1
        S_CBRANCH_SCC0 | 3, // else past the read
    ]);
    body.extend(read_lane_and_store(37));
    in_a_group_of(native(&body, false), threads)
}

/// The seventh review's fourth claim, its first half: with several console waves to a
/// group whose last wave is not full, a lane read of a lane that wave does not have reads
/// what no thread wrote (`Lower::meet_get`'s absent), so the result is not one value of the
/// group even where the register read is. In a group of 65 the first wave reads 1 and meets,
/// the second 0 and skips. Refused (since the eighth review as a group of several console
/// waves, full ones too: 128). In a group of 64: taken, and run.
#[test]
fn a_lane_read_past_a_last_wave_that_is_not_full_is_not_one_value() {
    let _gate = gate!();
    let names = [("v_readlane_b32", 2), ("s_cbranch_scc0", 1)];
    for threads in [65, 128] {
        let program = a_lane_read_of_a_lane_past_the_group(threads);
        has(&program, &names);
        refused_by_the_meeting(&program, SEVERAL);
    }
    let shader = exact(&a_lane_read_of_a_lane_past_the_group(64), &names);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "every thread reads lane 37: {out:x?}");
}

/// A program of [`native`] that reads the lane of v7 whose number is in the VGPR `lane`
/// (`v_readlane_b32 s7, v7, lane`, a lane number the decoder takes from a VGPR), and lane 37
/// behind `if (s7 == s3)` on what that gave. `lane`: the thread index (each thread names its
/// own lane), or v21, a copy of s0 (one lane number for the wave).
fn a_lane_read_at_a_number_in_a_vgpr(lane: u8) -> NativeProgram {
    let mut body = lanes_and_places();
    body.push(vop1(V_MOV_B32, 21, sgpr(0))); // v21 = s0
    body.extend(vop3(V_READLANE_B32, 7, vgpr(7), vgpr(lane))); // s7 = v7 of lane v[lane]
    body.extend([
        0xBF06_0307,        // s_cmp_eq_u32 s7, s3
        S_CBRANCH_SCC0 | 3, // not what s3 names: past the second read
    ]);
    body.extend(read_lane_and_store(37));
    native(&body, false)
}

/// The seventh review's fourth claim, its second half: a lane read whose lane number is in
/// a VGPR is lowered per thread (each reads the lane its own register names), and the proof
/// left every VGPR out where only the register read is the wave's. With the thread index
/// there, thread 0 matches s3 and meets alone. Refused now. With one lane number in the VGPR
/// (a copy of s0): taken, and run both ways.
#[test]
fn a_lane_read_at_a_lane_number_in_a_vgpr_is_each_threads_own() {
    let _gate = gate!();
    let names = [("v_readlane_b32", 2), ("s_cbranch_scc0", 1), ("s_cmp_eq_u32", 1)];
    let program = a_lane_read_at_a_number_in_a_vgpr(0);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_lane_read_at_a_number_in_a_vgpr(21), &names);
    let Some(out) = run_with(&shader, &[1, 1, 0, 0x101], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "lane 1 holds 0x101: every thread reads lane 37: {out:x?}");
    let out = run_with(&shader, &[1, 1, 0, 0x100], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "lane 1 does not hold 0x100: every thread goes past: {out:x?}");
}

/// A program of [`native`] that reads the lane of `reg` whose number is in `reg` itself
/// (`v_readlane_b32 s7, reg, reg`), and lane 37 of v7 behind `if (s7 == s3)` on what that
/// gave. `reg`: v0, the thread index (each thread reads its own lane: its own number); or
/// v21, a copy of s0 (one lane number, and one value, for the wave).
fn a_lane_read_of_the_lane_number_itself(reg: u8) -> NativeProgram {
    let mut body = lanes_and_places();
    body.push(vop1(V_MOV_B32, 21, sgpr(0))); // v21 = s0
    body.extend(vop3(V_READLANE_B32, 7, vgpr(reg), vgpr(reg))); // s7 = v[reg] of lane v[reg]
    body.extend([
        0xBF06_0307,        // s_cmp_eq_u32 s7, s3
        S_CBRANCH_SCC0 | 3, // not what s3 names: past the second read
    ]);
    body.extend(read_lane_and_store(37));
    native(&body, false)
}

/// The eighth review's claim on the fourth item: where the lane number is in the very VGPR
/// the lane read reads, the proof left that register out as the one read, and with it the
/// number: thread t reads lane t of the thread index, t, and only thread 0 matches s3 = 0
/// and meets. Refused now: the register read is left out only where it is not the number
/// too. With a copy of s0 there: taken, and run both ways.
#[test]
fn a_lane_read_whose_number_is_the_register_it_reads_is_each_threads_own() {
    let _gate = gate!();
    let names = [("v_readlane_b32", 2), ("s_cbranch_scc0", 1), ("s_cmp_eq_u32", 1)];
    let program = a_lane_read_of_the_lane_number_itself(0);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_lane_read_of_the_lane_number_itself(21), &names);
    let Some(out) = run_with(&shader, &[1, 1, 0, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "lane 1 of v21 is 1: every thread reads lane 37: {out:x?}");
    let out = run_with(&shader, &[1, 1, 0, 0], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "lane 1 of v21 is not 0: every thread goes past: {out:x?}");
}

/// A program of [`native`] in a group of `threads` whose read of lane 37 stands behind
/// `if (SCC)`, where SCC is joined from the two arms of `if (s0 == 0)`: `s2 == 0`, or the SCC
/// of `s_and_b64 s[14:15], exec, exec` ("some lane of the wave") with EXEC narrowed to
/// `lanes > thread` (`v_cmpx_gt_u32`) and opened again after.
fn some_lane_joined_with_a_scalar_test_before_a_branch_around_a_meeting(threads: u32, lanes: u16) -> NativeProgram {
    let mut body = lanes_and_places();
    body.extend([
        s_cmp_eq(0, 0),                         // s0 == 0
        S_CBRANCH_SCC0 | 2,                     // else: to the narrowing
        s_cmp_eq(2, 0),                         // SCC = (s2 == 0)
        S_BRANCH | 3,                           // to the join
        vopc(V_CMPX_GT_U32, k(lanes), 0),       // exec = (lanes > thread)
        0x878E_7E7E,                            // s_and_b64 s[14:15], exec, exec: SCC = some lane
        0xBEFE_04C1,                            // s_mov_b64 exec, -1
        S_CBRANCH_SCC0 | 3,                     // the join: SCC clear, past the read
    ]);
    body.extend(read_lane_and_store(37));
    in_a_group_of(native(&body, false), threads)
}

/// The seventh review's fifth claim: with several console waves to a group, SCC of a mask
/// operation as data ("some lane", `Lower::scc_some`) is each console wave's own, and a join
/// of it with a scalar test took it for one value of the group. In a group of 128 with EXEC
/// narrowed to `64 > thread`, the first wave has a lane and meets, the second none and
/// skips. Refused (since the eighth review as a group of several console waves). With one
/// console wave to the group it is the wave's one answer: taken, and run both ways.
#[test]
fn some_lane_of_a_mask_operation_joined_with_a_scalar_test_is_each_console_waves_own() {
    let _gate = gate!();
    let names = [("v_cmpx_gt_u32", 1), ("s_and_b64", 1), ("s_cbranch_scc0", 2), ("v_readlane_b32", 1)];
    let program = some_lane_joined_with_a_scalar_test_before_a_branch_around_a_meeting(128, 64);
    has(&program, &names);
    refused_by_the_meeting(&program, SEVERAL);
    let shader = exact(&some_lane_joined_with_a_scalar_test_before_a_branch_around_a_meeting(64, 32), &names);
    let Some(out) = run(&shader, [1, 1], 64) else { return };
    assert!(out.iter().all(|&x| x == 0x125), "some lane is left: every thread reads lane 37: {out:x?}");
    let out = run_with(&shader, &[0, 1, 1], 64).expect("the device that just ran");
    assert!(out.iter().all(|&x| x == 0), "s2 is not 0: every thread goes past: {out:x?}");
}

/// What the rule of the loops alone says of the earlier reviews' refusals (each still
/// refused by its own test above): it speaks of loops on a way to a group barrier, so it
/// refuses the meeting inside a loop the lanes leave one by one, and has nothing against a
/// barrier with no loop before it (a branch some lanes return at, a barrier of the program's
/// own or LDS writes inside a lane branch, a branch on a float sum the halves add up
/// differently) or beside a loop no way leads from to the barrier (the closed loops: those
/// meetings stand where not every thread comes, which the ends of such loops show).
#[test]
fn what_the_rule_of_the_loops_alone_refuses_of_the_earlier_cases() {
    assert!(the_loops_rule_refuses(&a_loop_the_lanes_leave_one_by_one()), "a meeting in a loop the lanes leave one by one");
    for (what, program) in [
        ("a meeting past a branch some lanes return at", a_branch_some_lanes_return_at()),
        ("a barrier of the program's own inside a lane branch", a_barrier_of_the_programs_own(true)),
        ("two LDS writes inside a lane branch", lds_writes_inside_a_lane_branch(true)),
        ("a meeting beside a closed loop", a_closed_loop_beside_a_meeting(true)),
        ("a meeting between two closed loops", two_closed_loops_and_a_meeting_before_the_second(true)),
        ("a branch on a float sum the halves add up differently", a_float_sum_each_half_adds_up_its_own_way(true, true)),
    ] {
        assert!(!the_loops_rule_refuses(&program), "{what}");
    }
}

/// `v_cndmask_b32 v, 0, v7, saved`: a whole-wave section's terms, 0.0 outside the saved EXEC
/// (the pair of [`super::wave::reduction`]).
const fn section_terms(v: u8) -> [u32; 2] {
    super::wave::reduction::terms(v, k(0) as u32, 7)
}

/// The additions of [`wave_sum_f32_v7_to_s7`] between its terms and its restore of EXEC, on
/// registers of the caller's: the terms in `v` are summed over the wave into scalar `s`
/// (`tmp`: the other row's lanes).
fn float_sum_steps(v: u8, tmp: u8, s: u8, two_apart_first: bool) -> Vec<u32> {
    let add = |src0: u16| vop2(V_ADD_F32, v, src0, v);
    let dpp8 = |sel: [u32; 8]| sel.iter().enumerate().fold(u32::from(v), |w, (i, &l)| w | l << (8 + 3 * i));
    let near = dpp8([1, 0, 3, 2, 5, 4, 7, 6]);
    let (first, second) =
        if two_apart_first { (dpp8([2, 3, 0, 1, 6, 7, 4, 5]), near) } else { (near, dpp8([2, 2, 0, 0, 6, 6, 4, 4])) };
    let mut w = vec![
        add(0xE9),
        first,
        add(0xE9),
        second,
        add(0xE9),
        dpp8([4, 4, 4, 4, 0, 0, 0, 0]),
        add(0xFA),
        0xFF09_4000 | u32::from(v), // row_mirror bound_ctrl:1
    ];
    let other_row = vop3(0x378, tmp, vgpr(v), k(0)); // v_permlanex16_b32 tmp, v, 0, 0
    w.extend([other_row[0], other_row[1] | (k(0) as u32) << 18, add(vgpr(tmp))]);
    w.extend(vop3(V_READLANE_B32, s, vgpr(v), k(32)));
    w.push(add(sgpr(s)));
    w.extend(vop3(V_READLANE_B32, s, vgpr(v), k(0)));
    w
}

/// A program of [`native`] whose one whole-wave section sums 2^25, 1, -2^25 and 0 (lanes
/// 0..3) twice, neighbours first into s7 (0.0) and lanes two apart first into s6 (1.0), and
/// leaves v5 = s7 in every lane; with `two_orders` the lanes from 32 on take s6 instead
/// (EXEC set to them for that move). `branch`: the threads whose v5 is 0.0 then read lane
/// 37 inside `if (v5 == 0)` and every thread stores what it has of that; else every thread
/// stores its v5.
fn a_float_sum_each_half_adds_up_its_own_way(two_orders: bool, branch: bool) -> NativeProgram {
    let mut body = lane_terms(&[(0, 33_554_432.0f32), (1, 1.0), (2, -33_554_432.0)]);
    use super::wave::reduction::{RESTORE_EXEC, SAVE_EXEC, TMP, WORK};
    assert_eq!(float_sum_steps(WORK, TMP, 7, false), wave_sum_f32_v7_to_s7(false)[3..19], "the compiler's steps");
    assert_eq!(float_sum_steps(WORK, TMP, 7, true), wave_sum_f32_v7_to_s7(true)[3..19], "the steps, two apart first");
    body.push(SAVE_EXEC); // s_or_saveexec_b64 saved, -1
    body.extend(section_terms(WORK));
    body.extend(float_sum_steps(WORK, TMP, 7, false)); // s7: neighbours first
    body.extend(section_terms(3));
    body.extend(float_sum_steps(3, TMP, 6, true)); // s6: lanes two apart first
    body.extend([
        vop1(V_MOV_B32, 5, sgpr(7)), // v5 = s7
        0xBEFE_03FF,
        0, // s_mov_b32 exec_lo, 0
        0xBEFF_03FF,
        0xFFFF_FFFF, // s_mov_b32 exec_hi, -1: the lanes from 32 on
        vop1(V_MOV_B32, 5, sgpr(if two_orders { 6 } else { 7 })), // v5 = s6 there (or s7 again)
        RESTORE_EXEC, // s_mov_b64 exec, saved
        vop2(V_ADD_NC_U32, 8, 255, 0),
        0x100,                           // v8 = 0x100 + thread
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop1(V_MOV_B32, 11, k(0)),       // v11 = 0
    ]);
    if branch {
        body.extend([
            vopc(V_CMP_EQ_U32, k(0), 5), // vcc = (0 == v5)
            SAVEEXEC_S12_VCC,
            S_CBRANCH_SCC0 | 3,
        ]);
        body.extend(vop3(V_READLANE_B32, 7, vgpr(8), k(37))); // s7 = v8 of lane 37
        body.extend([vop1(V_MOV_B32, 11, sgpr(7)), RESTORE_EXEC_S12]);
        body.extend(store_at(11, 9, 0));
    } else {
        body.extend(store_at(5, 9, 0));
    }
    native(&body, false)
}

/// The third review's Question 2, item 3: a sum of floats the lanes add up in different
/// orders is not one value. Lanes 0..31 hold the sum added neighbours first (0.0), lanes
/// 32..63 the same terms with lanes two apart first (1.0), and the program branches on it
/// around a read of lane 37: the threads from 32 on never come to that meeting, and a
/// barrier there would wait for them. The section's simulation calls both "the sum over the
/// wave", which they are; the meeting form must not take that for "the same in every lane".
/// Refused. As data the sum is exact, each lane's as its own additions give it: taken, and
/// run. And with every lane taking the same sum the branch is one the wave takes alike:
/// taken, and run.
#[test]
fn a_float_sum_the_halves_add_up_differently_does_not_bring_every_thread_to_a_meeting() {
    let _gate = gate!();
    refused_by_the_meeting(&a_float_sum_each_half_adds_up_its_own_way(true, true), "where not every lane runs");
    let names = [("v_add_f32", 12), ("v_permlanex16_b32", 2), ("v_readlane_b32", 4)];
    let shader = exact(&a_float_sum_each_half_adds_up_its_own_way(true, false), &names);
    let Some(out) = run(&shader, [0, 1], 64) else { return };
    for (t, &x) in out.iter().enumerate() {
        let want = if t < 32 { 0.0f32 } else { 1.0 };
        assert_eq!(x, want.to_bits(), "thread {t}: the sum as its half adds it up");
    }
    let names = [("v_add_f32", 12), ("v_permlanex16_b32", 2), ("v_readlane_b32", 5), ("s_cbranch_scc0", 1)];
    let shader = exact(&a_float_sum_each_half_adds_up_its_own_way(false, true), &names);
    let out = run(&shader, [0, 1], 64).expect("the device that just ran");
    for (t, &x) in out.iter().enumerate() {
        assert_eq!(x, 0x125, "thread {t}: lane 37 of the console's wave, the sum 0.0 in every lane");
    }
}

/// `WaveActiveSum(v7)` of floats into s7 ([`wave_sum_f32_v7_to_s7`]), or the minimum, with
/// one more addition before the lane read that takes the result: of +0.0, which the
/// program makes as +0.0 + -0.0 (two constants, `computed`) or names outright.
fn a_wave_result_plus_zero(min: bool, computed: bool) -> Vec<u32> {
    use super::wave::reduction::WORK;
    let mut w = wave_sum_f32_v7_to_s7(false);
    if min {
        // v_min_f32 for v_add_f32, and +inf outside the saved EXEC
        for x in &mut w {
            if *x >> 25 == u32::from(V_ADD_F32) {
                *x = *x & 0x01FF_FFFF | 15 << 25;
            }
        }
        // v_cndmask_b32 work, +inf, v7, saved
        let terms = super::wave::reduction::terms(WORK, 0xFF, 7);
        w.splice(1..3, [terms[0], terms[1], 0x7F80_0000]);
    }
    let at = w.len() - 3; // before `v_readlane_b32 s7, work, 0`
    let zero = if computed {
        vec![
            vop1(V_MOV_B32, 3, k(0)), // v3 = +0.0
            vop1(V_MOV_B32, 4, 255),
            0x8000_0000,                    // v4 = -0.0
            vop2(V_ADD_F32, 3, vgpr(3), 4), // v3 = v3 + v4: +0.0
            vop2(V_ADD_F32, WORK, vgpr(3), WORK), // work = v3 + work
        ]
    } else {
        vec![vop2(V_ADD_F32, WORK, k(0), WORK)] // work = +0.0 + work
    };
    w.splice(at..at, zero);
    w
}

/// The third review's Question 2, item 3, the addition the order's record dropped: every
/// lane holds -0.0, so the sum over the wave is -0.0, and the program then adds
/// +0.0 + -0.0 to it, which is +0.0: the result is +0.0, bits 0. The record took
/// +0.0 + -0.0 for -0.0 (the first constant "leaves the other unchanged") and the addition
/// of that for none, and the meeting gave -0.0.
#[test]
fn a_zero_the_program_computes_is_added_to_a_float_sum() {
    let _gate = gate!();
    let mut body = vec![vop1(V_MOV_B32, 7, 255), 0x8000_0000]; // v7 = -0.0
    body.extend(a_wave_result_plus_zero(false, true));
    body.push(vop1(V_MOV_B32, 10, sgpr(7))); // v10 = the result
    body.extend(store(10, 0));
    let program = native(&body, false);
    let shader = exact(&program, &[("v_add_f32", 8), ("v_readlane_b32", 2), ("buffer_store_dword", 1)]);
    let Some(out) = run(&shader, [0, 1], 1) else { return };
    assert_eq!(out[0], 0, "-0.0 sixty-four times, and +0.0: +0.0 ({:#010x})", out[0]);
}

/// The same record, where the +0.0 is added to something that is not a sum of floats: the
/// minimum over the wave of -0.0 in every lane is -0.0, and +0.0 added to it gives +0.0.
/// The record kept "the minimum over the wave" for the result, with the addition nowhere.
/// What the section leaves is then not a reduction the simulation knows: no form may hand
/// it over as the minimum.
#[test]
fn a_zero_added_to_a_wave_minimum_is_not_dropped() {
    let _gate = gate!();
    let mut body = vec![vop1(V_MOV_B32, 7, 255), 0x8000_0000]; // v7 = -0.0
    body.extend(a_wave_result_plus_zero(true, false));
    body.push(vop1(V_MOV_B32, 10, sgpr(7))); // v10 = the result
    body.extend(store(10, 0));
    let program = native(&body, false);
    let p = ir::build(&program.code, program.wave64, program.vgprs).expect("ir::build");
    let count = |name: &str| p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).count();
    assert_eq!((count("v_min_f32"), count("v_add_f32"), count("v_readlane_b32")), (6, 1, 2), "the fixture's code");
    let (shader, forms) = tried(&program);
    assert!(
        !forms.iter().any(|f| f.starts_with("met:") && f.contains("reduction MinF")),
        "met as the minimum over the wave, the addition dropped\n  {}",
        forms.join("\n  ")
    );
    let Ok(shader) = shader else { return };
    remember(&shader, &program);
    // (translated: then it must be the console's +0.0, which only a run shows)
    let Some(out) = run(&shader, [0, 1], 1) else { return };
    assert_eq!(out[0], 0, "the minimum -0.0, and +0.0: +0.0 ({:#010x})\n  {}", out[0], forms.join("\n  "));
}

const DS_ADD_RTN_U32: u8 = 0x20;
const DS_WRXCHG_RTN_B32: u8 = 0x2D;

/// The third review's Question 2, item 4: an atomic that returns the old value, at an
/// address past the program's LDS. The console reads 0 there and drops the write. The
/// meeting form sends every write past the end to one dword no one reads, and an atomic
/// read that dword back: what another lane had just exchanged into it, or the add before.
/// The threads below s0 (40) exchange 0x55 at byte 768 and then add 0x55 there, both
/// returning: each must get 0 twice, and the threads from 40 on, which run neither, keep
/// the 0x777 their registers held. The same add at the thread's own dword, inside the
/// program's LDS, returns what was written there and adds to it.
#[test]
fn a_returning_atomic_past_the_programs_lds_returns_zero_and_writes_nothing() {
    let _gate = gate!();
    let mut body = vec![
        vop2(V_LSHLREV_B32, 9, k(2), 0), // v9 = 4 thread
        vop2(V_ADD_NC_U32, 7, 255, 0),
        0x200,                           // v7 = 0x200 + thread
        vop1(V_MOV_B32, 14, 255),
        0x55,                            // v14 = 0x55
        vop1(V_MOV_B32, 15, 255),
        768,                             // v15 = 768
        vop1(V_MOV_B32, 10, 255),
        0x777,                           // v10 = 0x777
        vop1(V_MOV_B32, 11, 255),
        0x777,                           // v11 = 0x777
        0x7D88_0000,                     // v_cmp_gt_u32 vcc, s0, v0
        SAVEEXEC_S12_VCC,
    ];
    body.extend(ds(DS_WRXCHG_RTN_B32, 10, 15, 14)); // v10 = LDS[768], and LDS[768] = 0x55
    body.extend(ds(DS_ADD_RTN_U32, 11, 15, 14)); // v11 = LDS[768], and LDS[768] += 0x55
    body.push(RESTORE_EXEC_S12);
    body.extend(ds(DS_WRITE_B32, 0, 9, 7)); // LDS[4 thread] = v7
    body.extend(ds(DS_ADD_RTN_U32, 12, 9, 14)); // v12 = LDS[4 thread], and LDS[4 thread] += 0x55
    body.extend(ds(DS_READ_B32, 13, 9, 0)); // v13 = LDS[4 thread]
    body.extend(store_at(10, 9, 0)); // dword thread: the exchange's
    body.extend(store_at(11, 9, 256)); // dword 64 + thread: the add's
    body.extend(store_at(12, 9, 512)); // dword 128 + thread: the add's inside the LDS
    body.extend(store_at(13, 9, 768)); // dword 192 + thread: what it left
    let program = native(&body, true);
    let names = [("ds_wrxchg_rtn_b32", 1), ("ds_add_rtn_u32", 2), ("ds_write_b32", 1), ("ds_read_b32", 1)];
    let shader = exact(&program, &names);
    let Some(out) = run(&shader, [40, 1], 256) else { return };
    for t in 0..64 {
        let past = if t < 40 { 0 } else { 0x777 };
        assert_eq!(out[t], past, "thread {t}: an exchange past the program's LDS ({:#x})", out[t]);
        assert_eq!(out[64 + t], past, "thread {t}: a returning add past the program's LDS ({:#x})", out[64 + t]);
        assert_eq!(out[128 + t], 0x200 + t as u32, "thread {t}: a returning add inside it");
        assert_eq!(out[192 + t], 0x255 + t as u32, "thread {t}: what that add left");
    }
}

/// [`run_with`] with the views `views` in the heap from index 0 on (the UAV read back
/// first).
fn run_views(shader: &HostShader, constants: &[u32], views: &[View], dwords: u32) -> Option<Vec<u32>> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = device()?;
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("the device does not take a program that asks no wave size: {e}"));
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, Some(&c)) = (u.root, u.slot, constants.get(u.dword as usize)) {
            ud[u.index as usize] = c;
        }
    }
    let dispatch = Dispatch { roots: vec![(0, Root::Constants(ud))], views, samplers: &[], groups: [1, 1, 1] };
    let out = host.dispatch(&root, &pso, &dispatch).unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    dispatched(&host);
    let setup = crate::oracle::fixture::Setup {
        constants,
        views: views
            .iter()
            .map(|v| match v {
                View::RawUav(n) => crate::oracle::fixture::View::Uav(*n),
                View::Buffer { data, stride: super::host::Stride::Raw } => crate::oracle::fixture::View::Srv(data),
                _ => panic!("a view the oracle's fixtures do not lay out"),
            })
            .collect(),
        root_views: Vec::new(),
    };
    oracle(shader, &setup, &out);
    assert!(out[0].len() >= dwords as usize, "the UAV read back");
    Some(out[0].clone())
}

/// A program of [`native`] with a second descriptor in its table, at byte 32 (the heap's
/// next entry), of a range of `kind`: every thread stores its v7 through the UAV (at byte
/// 1024), then loads dword 0 through the second descriptor into v12, and the read of lane
/// 37 stands behind `if (v12 == 0)`, a compare of a value the wave shares if nothing the
/// program stores is where it loads.
fn a_load_through_a_second_view_beside_a_store(kind: ResourceType) -> NativeProgram {
    let mut body = vec![
        0x3D << 26 | 0x02 << 18 | 12 << 6 | 2,
        125 << 25 | 0x20, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    ];
    body.extend(lanes_and_places());
    body.extend(store(7, 0x400)); // buffer_store_dword v7, off, s[8:11] offset:1024
    body.extend([
        0x38 << 26 | 0x0C << 18,        // buffer_load_dword v12, off, s[12:15], 0
        0x80 << 24 | 3 << 16 | 12 << 8,
        vopc(V_CMP_EQ_U32, k(0), 12),   // vcc = (0 == v12)
        S_CBRANCH_VCCZ | 3,             // the lanes whose v12 is not 0: past the read
    ]);
    body.extend(read_lane_and_store(37));
    let mut program = native(&body, false);
    program.used.push(UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 1,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    });
    program
}

/// The ninth review's claim: the proof counted a load through a descriptor the program
/// stores nothing through as one value of the wave, but two UAV descriptors in other places
/// of the tables (or a root UAV and a table's) may view one buffer: a host wave that loads
/// after another has stored reads what it stored, and the branch on it is not one the wave
/// takes alike. Refused now: where the program stores anything, only a load through a CBV
/// or an SRV is (a resource is not a UAV and an SRV of one dispatch). Through an SRV: taken,
/// and run both ways.
#[test]
fn a_load_through_a_uav_beside_a_store_through_another_is_no_constant() {
    let _gate = gate!();
    let names = [("buffer_load_dword", 1), ("buffer_store_dword", 2), ("s_cbranch_vccz", 1), ("v_readlane_b32", 1)];
    let program = a_load_through_a_second_view_beside_a_store(ResourceType::Uav);
    has(&program, &names);
    refused_by_the_meeting(&program, "where not every lane runs");
    let shader = exact(&a_load_through_a_second_view_beside_a_store(ResourceType::Srv), &names);
    let raw = super::host::Stride::Raw;
    let Some(out) = run_views(&shader, &[0, 1], &[View::RawUav(260), View::Buffer { data: &[0; 4], stride: raw }], 64)
    else {
        return;
    };
    assert!(out[..64].iter().all(|&x| x == 0x125), "dword 0 of the SRV is 0: every thread reads lane 37: {:x?}", &out[..64]);
    let views = [View::RawUav(260), View::Buffer { data: &[1; 4], stride: raw }];
    let out = run_views(&shader, &[0, 1], &views, 64).expect("the device that just ran");
    assert!(out[..64].iter().all(|&x| x == 0), "dword 0 of the SRV is 1: every thread goes past: {:x?}", &out[..64]);
}

/// The ninth review's follow-up: a program that allocates no LDS but has an LDS access
/// (`ds_write_b32`) is refused for it ("LDS the program does not allocate"). The meeting
/// form adds its own dwords to the group-shared memory, and the check asked of the sum: the
/// access went to the form's spare dword. Refused in the meeting form now too.
#[test]
fn an_lds_access_in_a_program_without_lds_is_refused_in_the_meeting_form() {
    let mut body = lanes_and_places();
    body.extend(read_lane_and_store(37));
    body.extend(ds(DS_WRITE_B32, 0, 9, 11)); // LDS dword thread = v11
    let program = native(&body, false);
    has(&program, &[("ds_write_b32", 1), ("v_readlane_b32", 1)]);
    refused_by_the_meeting(&program, "LDS the program does not allocate");
}
