//! A mesh program and the pixel program drawn with it, translated for one host pipeline
//! ([`link_mesh`]). Ported after `26600c61e` in the darkworldhm fork.
//!
//! D3D12 caps a mesh shader's outputs at 32 KiB: 4 bytes per component of every vertex
//! element times the declared vertex maximum rounded up to 32, plus the same for primitive
//! elements ([`super::container::mesh_output_bytes`], as the validator measures it). A
//! console mesh program can pass it as the console declares its outputs: more components a
//! vertex than the cap holds at its vertex maximum. The made-up mesh of `link_tests` is one
//! (72 components for 128 vertices, 36,864 bytes). NVIDIA's driver builds such pipelines
//! anyway, but the programs are invalid. D3D12 links the two stages by semantic, and a pipeline needs only the
//! outputs its pixel program reads, which fit under the cap. So the pair is translated
//! again for its
//! pipeline: the mesh program declares and writes only the outputs the pixel program reads
//! (and its system values), and the pixel program leaves out the inputs it never reads,
//! which D3D12 would otherwise refuse as inputs the mesh program does not write. The
//! elements kept keep their rows and columns, each program its code and bindings.
//!
//! Only a pair whose interface is whole is linked ([`mesh_interface`]): pruning by name and
//! index says nothing of whether the mesh program writes what the pixel program reads.

use super::container::{MESH_OUTPUT_LIMIT, SigElement, mesh_output_bytes};
use super::{HostShader, translate};
use crate::HwStage;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC, u32_at};

thread_local! {
    /// See [`with_linked`].
    static LINKED: std::cell::RefCell<Option<Linked>> = const { std::cell::RefCell::new(None) };
    /// See [`per_vertex_fetches`].
    static FETCHED: std::cell::RefCell<Option<Vec<(String, u32)>>> =
        const { std::cell::RefCell::new(None) };
}

/// What a stage leaves out of its signatures for its partner in one host pipeline
/// ([`link_mesh`]): elements of arbitrary semantics, by name and index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Linked {
    pub inputs: Vec<(String, u32)>,
    pub outputs: Vec<(String, u32)>,
}

/// Runs `f` with programs lowered for a linked pipeline: the elements `linked` names leave
/// the signatures; stores to such an output are not made, and the other elements keep their
/// rows and modes (ids count only what is kept). An input left out must be one the program
/// never reads: a lowering that reads one fails.
pub(super) fn with_linked<R>(linked: Linked, f: impl FnOnce() -> R) -> R {
    let was = LINKED.replace(Some(linked));
    let r = f();
    LINKED.set(was);
    r
}

/// Per element of `inputs` and of `outputs` (the signatures as the console declares them),
/// whether the stage being lowered leaves it out ([`with_linked`]); empty when not linking.
pub(super) fn left_out(inputs: &[SigElement], outputs: &[SigElement]) -> (Vec<bool>, Vec<bool>) {
    LINKED.with_borrow(|l| match l {
        Some(l) => {
            let of = |els: &[SigElement], names: &[(String, u32)]| {
                els.iter()
                    .map(|e| {
                        e.kind == 0
                            && names
                                .iter()
                                .any(|(n, i)| n.eq_ignore_ascii_case(&e.name) && *i == e.index)
                    })
                    .collect::<Vec<bool>>()
            };
            (of(inputs, &l.inputs), of(outputs, &l.outputs))
        }
        None => (Vec::new(), Vec::new()),
    })
}

/// The translations of the mesh program `ms` (container `ms_container`) and the pixel program
/// `ps` (container `ps_container`) for one host pipeline, when `ms`'s outputs pass D3D12's 32
/// KiB as translated alone: the mesh program without the outputs the pixel program does not
/// read, the pixel program without the inputs it does not read (see the module notes).
///
/// `None` (make the pipeline of `ms` and `ps` as they are) when `ms` fits, when the
/// interface between the two is not whole ([`mesh_interface`], asked before anything is
/// pruned), when the pixel program reads every output `ms` has, when the linked mesh program
/// still does not fit, or when a linked translation binds anything differently from its
/// shared one (the pipeline's root signature is made from the shared ones).
pub fn link_mesh(
    ms_container: &[u8],
    ms: &HostShader,
    ps_container: &[u8],
    ps: &HostShader,
) -> Result<Option<(HostShader, HostShader)>, ShaderError> {
    if ms.stage != HwStage::Ms || ps.stage != HwStage::Ps || mesh_fits(&ms.dxil)? {
        return Ok(None);
    }
    match mesh_interface(ms, ps_container, ps) {
        Ok(()) => {}
        Err(ShaderError::Unsupported { .. }) => return Ok(None),
        Err(e) => return Err(e),
    }
    let named = |e: &Element| (e.name.clone(), e.index);
    let inputs = elements(&ps.dxil, FourCC::ISG1)?;
    let read = |e: &Element| {
        inputs.iter().any(|i| {
            i.arbitrary() && i.used != 0 && i.name.eq_ignore_ascii_case(&e.name) && i.index == e.index
        })
    };
    let outputs: Vec<(String, u32)> = elements(&ms.dxil, FourCC::OSG1)?
        .iter()
        .filter(|e| e.arbitrary() && !read(e))
        .map(named)
        .collect();
    if outputs.is_empty() {
        return Ok(None);
    }
    let unread: Vec<(String, u32)> = inputs
        .iter()
        .filter(|i| i.arbitrary() && i.used == 0)
        .map(named)
        .collect();
    let linked_ms = with_linked(
        Linked {
            inputs: Vec::new(),
            outputs,
        },
        || translate(ms_container),
    )?;
    let linked_ps = if unread.is_empty() {
        ps.clone()
    } else {
        with_linked(
            Linked {
                inputs: unread,
                outputs: Vec::new(),
            },
            || translate(ps_container),
        )?
    };
    let same = |a: &HostShader, b: &HostShader| {
        HostShader { dxil: Vec::new(), ..a.clone() } == HostShader { dxil: Vec::new(), ..b.clone() }
    };
    if !mesh_fits(&linked_ms.dxil)? || !same(ms, &linked_ms) || !same(ps, &linked_ps) {
        return Ok(None);
    }
    Ok(Some((linked_ms, linked_ps)))
}

/// Whether the mesh program `ms` writes what the pixel program `ps` (container
/// `ps_container`) reads of it, both as translated alone: `Unsupported` with what is wrong
/// when it does not, and [`link_mesh`] then links nothing.
///
/// D3D12 matches a pixel input to a mesh output by semantic name and index, not by register
/// (the mesh shader specification, "Signature Linkage"), and nothing in a pipeline's creation
/// is promised to refuse a pair whose elements of one name disagree. So, of every input of an
/// arbitrary semantic the pixel program reads:
///
/// - one mesh output that the linked mesh program keeps has its name and index: the outputs
///   kept are the per-vertex ones a read input names and every per-primitive one, and no two
///   of those share a name and index (which one a read would get is not defined). Two
///   per-vertex outputs of a name nothing reads are both pruned, and are not asked about;
/// - that output has the input's component type;
/// - the mesh program stores every component read, counted from the element's first
///   (packing may differ between the stages, the components of a semantic may not): the
///   components some store of the program names, which is what the emitted part records, not
///   the ones the element declares. That a store runs for every vertex or primitive drawn is
///   not something a signature says;
/// - it is not per primitive while the pixel program fetches the input per vertex (weights it
///   computes, or P10 and P20: the three vertices' values, which a primitive does not have).
///   What counts is the element `attributeAtVertex` is emitted for
///   ([`per_vertex_fetches`]), not the attribute row the console read names: another element
///   of that row read flat is no fetch, and the half of a packed parameter can be an element
///   of another row.
///
/// How the read interpolates is not asked: a per-primitive output read with an interpolating
/// mode is read flat by Direct3D whatever the pixel input declares (the specification's
/// overview), and between a per-vertex output and its read the pixel program's mode decides,
/// as between a vertex and a pixel program.
///
/// Of the system values the pixel program reads, those a stage before it must supply
/// ([`STAGE_SUPPLIED`]) need a mesh output of their kind and index that stores every
/// component read, counted from the element's first as above (a primitive id declared and
/// never stored supplies nothing; a clip distance stored in its first component does not
/// supply its second). The others the rasterizer makes (the position, the front face, the
/// sample index, the coverage, barycentrics), and no mesh output is asked for them.
///
/// An element whose row of its signature part is not found ([`Port::known`]) has components
/// nobody can name. It is still matched by kind and index, and nothing is asked of its
/// components, read or stored: asking would refuse a pair for a read the pixel program may
/// not make, or for a store the mesh program may make, and the pair refused is then drawn
/// from the same two programs unlinked, the mesh one over 32 KiB. No element of a kind in
/// [`STAGE_SUPPLIED`] is in that state as the translation emits it: the two parts are written
/// from one list of elements, and each of those kinds has its system value in the signature.
///
/// Inputs the pixel program declares and never reads are not asked about: the linked pixel
/// program leaves them out.
pub fn mesh_interface(
    ms: &HostShader,
    ps_container: &[u8],
    ps: &HostShader,
) -> Result<(), ShaderError> {
    let outputs = ports(&ms.dxil)?.outputs;
    let inputs = ports(&ps.dxil)?.inputs;
    let fetched = per_vertex_fetches(ps_container)?;
    match interface(&outputs, &inputs, &fetched) {
        Some(what) => Err(ShaderError::Unsupported { offset: 0, what }),
        None => Ok(()),
    }
}

/// What is wrong between the mesh outputs and the pixel inputs ([`mesh_interface`]), the
/// pixel program fetching the inputs `fetched` (semantic and index) per vertex.
pub(super) fn interface(
    outputs: &[Port],
    inputs: &[Port],
    fetched: &[(String, u32)],
) -> Option<&'static str> {
    let same = |a: &Port, b: &Port| a.name.eq_ignore_ascii_case(&b.name) && a.index == b.index;
    let read: Vec<&Port> = inputs.iter().filter(|i| i.used != 0).collect();
    // What the linked mesh program keeps of its arbitrary semantics ([`link_mesh`]).
    let kept: Vec<&Port> = outputs
        .iter()
        .filter(|o| o.kind == 0)
        .filter(|o| o.primitive || read.iter().any(|i| i.kind == 0 && same(i, o)))
        .collect();
    if (0..kept.len()).any(|n| kept[..n].iter().any(|o| same(o, kept[n]))) {
        return Some(TWICE);
    }
    for i in read {
        if i.kind != 0 {
            if !STAGE_SUPPLIED.contains(&i.kind) {
                continue;
            }
            let mut supplied = outputs
                .iter()
                .filter(|o| o.kind == i.kind && o.index == i.index)
                .peekable();
            if supplied.peek().is_none() {
                return Some(SYSTEM_VALUE);
            }
            if !supplied.any(|o| !i.known || !o.known || i.used & !o.used == 0) {
                return Some(COMPONENT);
            }
            continue;
        }
        let Some(o) = kept.iter().find(|o| same(o, i)) else {
            return Some(UNWRITTEN);
        };
        if o.comp != i.comp {
            return Some(TYPE);
        }
        if i.used & !o.used != 0 {
            return Some(COMPONENT);
        }
        let fetch = |(name, index): &(String, u32)| {
            name.eq_ignore_ascii_case(&i.name) && *index == i.index
        };
        if o.primitive && fetched.iter().any(fetch) {
            return Some(PER_VERTEX);
        }
    }
    None
}

/// The DXIL semantic kinds of the system values a pixel program gets from the stage before
/// it and not from the rasterizer: the render-target and viewport array indices, the clip
/// and cull distances, and the primitive id, which a mesh pipeline has only when the mesh
/// program writes it.
const STAGE_SUPPLIED: [u8; 5] = [4, 5, 6, 7, 10];

/// What [`mesh_interface`] refuses, as its error names it.
pub(super) const TWICE: &str = "linking a mesh program that writes one semantic twice";
pub(super) const UNWRITTEN: &str = "linking a pixel input the mesh program does not write";
pub(super) const TYPE: &str = "linking a mesh output read as another type";
pub(super) const COMPONENT: &str = "linking a read of a component the mesh program does not write";
pub(super) const PER_VERTEX: &str = "linking a per-primitive mesh output read per vertex";
pub(super) const SYSTEM_VALUE: &str =
    "linking a read of a system value the mesh program does not write";

/// The inputs (semantic and index) the pixel program of `container` fetches per vertex: the
/// elements its lowering emits `attributeAtVertex` for. Nothing in the emitted signature
/// tells such an element from one read flat (both are declared constant), and the console's
/// read does not name it either: it names an attribute row, of which the element is the one
/// holding the channel read, or for a parameter of packed 16-bit values the one holding the
/// half read, which can be in another row. So the program is lowered once more with every
/// fetch written down ([`fetched`]).
///
/// The lowering starts over at times and tries a program in more than one form; every start
/// writes here. They agree: which elements are fetched follows from the program's
/// interpolations and its signature, which no start changes, so a start that was given up
/// adds nothing the one handed over lacks.
///
/// Costs one translation of the pixel program per pair asked about (a pair is asked about
/// once, and once more for the log when it does not link).
fn per_vertex_fetches(container: &[u8]) -> Result<Vec<(String, u32)>, ShaderError> {
    let was = FETCHED.replace(Some(Vec::new()));
    let translated = translate(container);
    let fetched = FETCHED.replace(was).unwrap_or_default();
    translated.map(|_| fetched)
}

/// The lowering's note of an `attributeAtVertex` it emits for input element `e`
/// ([`per_vertex_fetches`]; nothing is kept when nobody asked).
pub(super) fn fetched(e: &SigElement) {
    FETCHED.with_borrow_mut(|f| {
        if let Some(f) = f
            && !f.iter().any(|(n, i)| *n == e.name && *i == e.index)
        {
            f.push((e.name.clone(), e.index));
        }
    });
}

/// One element of an emitted program's interface: `PSV0`'s row for it, and what its row of
/// the signature part says of its components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Port {
    pub name: String,
    pub index: u32,
    /// DXIL semantic kind (0: an arbitrary semantic).
    pub kind: u8,
    /// `PSV0`'s component type.
    pub comp: u8,
    /// DXIL interpolation mode.
    pub interp: u8,
    pub row: Option<u32>,
    pub cols: u8,
    /// From the element's first component: of an input, the ones the program reads; of an
    /// output, the ones some store of the program writes.
    pub used: u8,
    /// Whether `used` is what the element's row of the signature part says. Not for a system
    /// value whose row was not found there: `used` is then every component it declares, so
    /// that it counts as read, or as supplied, and its components are asked nothing
    /// ([`mesh_interface`]).
    pub known: bool,
    /// A mesh program's per-primitive output.
    pub primitive: bool,
}

/// An emitted program's inputs, and its outputs (per vertex, then per primitive).
struct Ports {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

/// The interface of an emitted program, out of its `PSV0` part (`container::psv` lays it
/// out) and its signature parts (`container::signature`).
fn ports(dxil: &[u8]) -> Result<Ports, ShaderError> {
    const TRUNCATED: ShaderError = ShaderError::Malformed("truncated PSV0");
    let c = Container::parse(dxil)?;
    let psv = c
        .part(FourCC::PSV0)
        .ok_or(ShaderError::Malformed("no PSV0 part"))?
        .data;
    let skip = |at: usize, count: usize, size: usize| {
        count
            .checked_mul(size)
            .and_then(|n| at.checked_add(n))
            .ok_or(TRUNCATED)
    };
    // runtime info (its size first): shader kind at byte 24, element counts at 28 to 30
    let info_size = u32_at(psv, 0)? as usize;
    let info = psv.get(4..skip(4, info_size, 1)?).ok_or(TRUNCATED)?;
    let byte = |at: usize| info.get(at).copied().ok_or(TRUNCATED);
    let mesh = byte(24)? == 13;
    let counts = [
        usize::from(byte(28)?),
        usize::from(byte(29)?),
        if mesh { usize::from(byte(30)?) } else { 0 },
    ];
    let mut at = 4 + info_size;
    let bindings = u32_at(psv, at)? as usize;
    at += 4;
    if bindings > 0 {
        at = skip(at + 4, bindings, u32_at(psv, at)? as usize)?;
    }
    let strings_len = u32_at(psv, at)? as usize;
    let strings = psv
        .get(at + 4..skip(at + 4, strings_len, 1)?)
        .ok_or(TRUNCATED)?;
    at += 4 + strings_len;
    let indices = at + 4;
    at = skip(indices, u32_at(psv, at)? as usize, 4)?;
    let mut out = Ports {
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    if counts.iter().sum::<usize>() == 0 {
        return Ok(out);
    }
    let size = u32_at(psv, at)? as usize;
    at += 4;
    if size < 16 {
        return Err(TRUNCATED);
    }
    // The rows of the three signature parts, and which of them an element was found in.
    let parts = [
        elements(dxil, FourCC::ISG1)?,
        elements(dxil, FourCC::OSG1)?,
        elements(dxil, FourCC::PSG1)?,
    ];
    let mut taken = parts.each_ref().map(|rows| vec![false; rows.len()]);
    for (class, count) in counts.into_iter().enumerate() {
        for _ in 0..count {
            let e = psv.get(at..skip(at, 1, size)?).ok_or(TRUNCATED)?;
            at += size;
            let name = strings
                .get(u32_at(e, 0)? as usize..)
                .and_then(|s| s.split(|&b| b == 0).next())
                .ok_or(TRUNCATED)?;
            let name = String::from_utf8_lossy(name).into_owned();
            let index = u32_at(psv, skip(indices, u32_at(e, 4)? as usize, 4)?)?;
            // rows, start row, columns (their count, the first one, "allocated"), kind,
            // component type, interpolation mode
            let (cols, col) = (e[10] & 0xf, e[10] >> 4 & 3);
            let row = (e[10] & 0x40 != 0).then_some(u32::from(e[9]));
            let kind = e[11];
            if cols > 4 {
                return Err(ShaderError::Malformed("a PSV0 element of over four columns"));
            }
            // Its row of the signature part: the one of its semantic in its register and
            // columns. An element without a register has no column in `PSV0`; its row of the
            // part has the mask, and is found by the count of its components.
            let found = parts[class].iter().enumerate().position(|(n, r)| {
                !taken[class][n]
                    && r.index == index
                    && r.arbitrary() == (kind == 0)
                    && (kind != 0 || r.name == name)
                    && match row {
                        Some(row) => r.row == row && r.mask == ((1u8 << cols) - 1) << col,
                        None => r.row == u32::MAX && r.mask.count_ones() == u32::from(cols),
                    }
            });
            let (used, known) = match found {
                Some(n) => {
                    taken[class][n] = true;
                    let r = &parts[class][n];
                    let first = r.mask.trailing_zeros().min(7);
                    // (an output's byte is the components never written)
                    let used = if class == 0 { r.used } else { !r.used };
                    ((used & r.mask) >> first, true)
                }
                // A system value the part lists some other way: which of its components are
                // read, or written, is not known ([`Port::known`]).
                None if kind != 0 => ((1u8 << cols) - 1, false),
                None => {
                    return Err(ShaderError::Malformed(
                        "a PSV0 element its signature part lacks",
                    ));
                }
            };
            let port = Port {
                name,
                index,
                kind,
                comp: e[12],
                interp: e[13],
                row,
                cols,
                used,
                known,
                primitive: class == 2,
            };
            if class == 0 {
                out.inputs.push(port);
            } else {
                out.outputs.push(port);
            }
        }
    }
    Ok(out)
}

/// Whether an emitted mesh program's outputs fit D3D12's 32 KiB (a program of another stage
/// fits).
pub fn mesh_fits(dxil: &[u8]) -> Result<bool, ShaderError> {
    let c = Container::parse(dxil)?;
    let Some(psv) = c.part(FourCC::PSV0) else {
        return Ok(true);
    };
    // runtime info (its size first): shader kind at byte 24, mesh maxima at bytes 12 and 14
    let info = psv.data.get(4..).ok_or(ShaderError::Malformed("truncated"))?;
    if info.get(24) != Some(&13) {
        return Ok(true);
    }
    let half = |at: usize| {
        info.get(at..at + 2)
            .map_or(0, |b| u32::from(u16::from_le_bytes([b[0], b[1]])))
    };
    let declared = |fourcc| -> Result<Vec<SigElement>, ShaderError> {
        Ok(elements(dxil, fourcc)?
            .into_iter()
            .map(|e| SigElement {
                name: e.name,
                index: e.index,
                kind: 0,
                comp: 0,
                interp: 0,
                row: None,
                col: 0,
                cols: e.mask.count_ones() as u8,
                usage: 0,
                half: false,
            })
            .collect())
    };
    let bytes = mesh_output_bytes(
        &declared(FourCC::OSG1)?,
        &declared(FourCC::PSG1)?,
        half(12),
        half(14),
    );
    Ok(bytes <= MESH_OUTPUT_LIMIT)
}

/// A row of an emitted signature part (`container::signature`).
struct Element {
    name: String,
    index: u32,
    /// The system value (0: an arbitrary semantic).
    sv: u32,
    /// Its register.
    row: u32,
    mask: u8,
    /// Inputs: the components read. Outputs: the components never written.
    used: u8,
}

impl Element {
    fn arbitrary(&self) -> bool {
        self.sv == 0
    }
}

fn elements(dxil: &[u8], fourcc: FourCC) -> Result<Vec<Element>, ShaderError> {
    let c = Container::parse(dxil)?;
    let Some(part) = c.part(fourcc) else {
        return Ok(Vec::new());
    };
    let part = part.data;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    (0..count)
        .map(|i| {
            let at = base
                .checked_add(
                    32usize
                        .checked_mul(i)
                        .ok_or(ShaderError::Malformed("signature table overflow"))?,
                )
                .ok_or(ShaderError::Malformed("signature table overflow"))?;
            let name_at = u32_at(part, at + 4)? as usize;
            let name = part
                .get(name_at..)
                .and_then(|s| s.split(|&b| b == 0).next())
                .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
            let bytes = part
                .get(at + 24..at + 26)
                .ok_or(ShaderError::Malformed("truncated"))?;
            Ok(Element {
                name: String::from_utf8_lossy(name).into_owned(),
                index: u32_at(part, at + 8)?,
                sv: u32_at(part, at + 12)?,
                row: u32_at(part, at + 20)?,
                mask: bytes[0],
                used: bytes[1],
            })
        })
        .collect()
}
