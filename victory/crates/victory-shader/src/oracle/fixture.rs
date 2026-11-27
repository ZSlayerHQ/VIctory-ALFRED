//! The oracle for the translator's fixtures: a fixture's program (a [`NativeProgram`]) run with
//! the fixture's inputs on the interpreter, and every dword of its output compared with what the
//! translated program left on the host ([`check`]).
//!
//! The console side of a fixture's inputs, as the oracle lays them out: root constants in their
//! user SGPRs; a descriptor table at a 32-bit address in its user SGPR, its entries 32 bytes
//! apart in the order of the host's views (the host's heap); each view a raw buffer behind a
//! buffer resource of its size (stride 0, format 32_UINT, OOB_SELECT 3, dst_sel XYZW, 8.1.8);
//! a root descriptor a 64-bit address in its two user SGPRs. One group, group id 0.
//!
//! MODE is an assumption: 0x00F0, round to nearest even, single and double denormals kept,
//! IEEE and DX10_CLAMP clear. The console's value comes from `SPI_SHADER_PGM_RSRC1`, which a
//! program record does not keep, so a fixture whose answer depends on the denormal, IEEE or
//! clamp modes is checked against this choice, not against the console's register.
//!
//! [`check`] compares every UAV in full: as many UAVs and dwords on both sides, or it fails.

use super::compare::{Diff, compare_choices};
use super::{Group, Memory, Mode, Refusal, Tol, Wave64Permute};
use crate::native::{NativeProgram, RootType, StageIo};

const HEAP: u64 = 0x0010_0000;
const VIEWS: u64 = 0x1000_0000;
const ROOT_VIEWS: u64 = 0x4000_0000;
const SPACING: u64 = 0x0100_0000;

/// A heap entry of a fixture's dispatch, in the host's order.
#[derive(Debug, Clone)]
pub(crate) enum View<'a> {
    /// A raw buffer of this many zeroed dwords the program may write: compared afterwards.
    Uav(u32),
    /// A raw buffer of these dwords the program reads.
    Srv(&'a [u32]),
}

/// A fixture's inputs.
#[derive(Debug, Clone, Default)]
pub(crate) struct Setup<'a> {
    /// Root constants, dword 0 on.
    pub constants: &'a [u32],
    pub views: Vec<View<'a>>,
    /// The buffers behind root descriptors, in the order of the program's layout.
    pub root_views: Vec<&'a [u32]>,
}

/// A raw buffer resource over `bytes` bytes at `base`.
fn raw_resource(base: u64, bytes: u32) -> [u32; 4] {
    // dst_sel X Y Z W (4 5 6 7), format 20 (32_UINT), resource level 1, OOB_SELECT 3
    let w3 = 4 | 5 << 3 | 6 << 6 | 7 << 9 | 20 << 12 | 1 << 24 | 3 << 28;
    [base as u32, (base >> 32) as u32 & 0xFFFF, bytes, w3]
}

/// What the oracle left in each UAV (dwords, their trust, and where several lanes stored to one
/// dword the values one of which landed), in the order of the views.
pub(crate) type Outputs = Vec<(Vec<u32>, Vec<Tol>, Vec<Vec<u32>>)>;

/// Runs `program` on the interpreter with `setup`.
pub(crate) fn run(program: &NativeProgram, setup: &Setup) -> Result<Outputs, Refusal> {
    let StageIo::Cs(io) = &program.io else {
        return Err(Refusal { offset: 0, instruction: "(dispatch)".into(), reason: "not a compute program".into() });
    };
    let mut memory = Memory::new();
    let mut heap = Vec::new();
    for (i, view) in setup.views.iter().enumerate() {
        let base = VIEWS + i as u64 * SPACING;
        let dwords: Vec<u32> = match view {
            View::Uav(n) => vec![0; *n as usize],
            View::Srv(data) => data.to_vec(),
        };
        heap.extend(raw_resource(base, 4 * dwords.len() as u32));
        heap.extend([0; 4]);
        memory.map_dwords(base, &dwords);
    }
    memory.map_dwords(HEAP, &heap);
    let mut sgprs = vec![0u32; usize::from(io.user_sgprs())];
    let mut roots = setup.root_views.iter().enumerate();
    for p in &program.layout {
        let first = usize::from(program.stage.user_sgpr(p.user_data));
        let mut put = |k: usize, v: u32| {
            if let Some(s) = sgprs.get_mut(first + k) {
                *s = v;
            }
        };
        match p.root {
            RootType::Constants => {
                for k in 0..p.sgprs as usize {
                    put(k, setup.constants.get(k).copied().unwrap_or(0));
                }
            }
            RootType::Table => put(0, HEAP as u32),
            RootType::Descriptor => {
                let Some((n, data)) = roots.next() else { continue };
                let base = ROOT_VIEWS + n as u64 * SPACING;
                memory.map_dwords(base, data);
                put(0, base as u32);
                put(1, (base >> 32) as u32);
            }
            _ => {}
        }
    }
    let group = Group {
        code: program.code.clone(),
        wave64: program.wave64,
        threads: io.threads,
        group_id: [0; 3],
        rsrc2: io.rsrc2,
        user_sgprs: sgprs,
        vgprs: program.vgprs,
        mode: Mode(0x00F0),
        permute: Wave64Permute::Unchosen,
        step_limit: 1 << 20,
    };
    super::run(&group, &mut memory)?;
    Ok(setup
        .views
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v, View::Uav(_)))
        .map(|(i, _)| {
            let base = VIEWS + i as u64 * SPACING;
            let (words, tol) = memory.dwords(base).expect("the view's region");
            (words, tol, memory.choices(base).expect("the view's region"))
        })
        .collect())
}

/// Runs `program` with `setup` and compares each UAV with the host's read-back of it (`host`,
/// in the order of the UAVs). A disagreement in any compared class, or a read-back of another
/// number of UAVs or dwords, panics with the dword, the lane it belongs to where the fixture
/// writes one dword per thread, and both values; a refusal is given back with its reason, for
/// the fixture to hold against what it expects.
pub(crate) fn check(what: &str, program: &NativeProgram, setup: &Setup, host: &[Vec<u32>]) -> Result<Diff, Refusal> {
    let outputs = run(program, setup)?;
    assert_eq!(
        outputs.len(),
        host.len(),
        "ORACLE {what}: {} UAVs on the oracle's side, {} read back",
        outputs.len(),
        host.len()
    );
    let mut total = Diff::default();
    for (n, ((dwords, tol, choices), other)) in outputs.iter().zip(host).enumerate() {
        let d = compare_choices(dwords, tol, choices, other);
        if !d.agrees() {
            let show = |m: &super::compare::Mismatch| {
                format!(
                    "UAV {n} dword {} (byte {:#x}): oracle {:#010x} ({}), host {:#010x} ({}){}",
                    m.dword,
                    4 * m.dword,
                    m.oracle,
                    f32::from_bits(m.oracle),
                    m.other,
                    f32::from_bits(m.other),
                    if m.tol.unordered {
                        format!(", not one of the {} values the lanes stored", choices[m.dword].len())
                    } else if m.tol.exact() {
                        String::new()
                    } else {
                        format!(", allowed {} ULPs", m.tol.ulps)
                    }
                )
            };
            let lines: Vec<String> =
                d.exact_differ.iter().chain(&d.approximate_differ).chain(&d.unordered_differ).take(8).map(show).collect();
            panic!("ORACLE {what}: the translated program disagrees with the console's instructions: {}
  {}", d.line(), lines.join("
  "));
        }
        total.add(&d);
    }
    eprintln!("ORACLE {what}: agrees; {}", total.line());
    Ok(total)
}
