//! Fixtures for a mesh program whose outputs pass D3D12's 32 KiB and the pixel program drawn
//! with it ([`super::link`]): the mesh and pixel programs are translated again for their
//! pipeline, the mesh writing only what the pixel reads. Each console program is synthetic,
//! built here as a DXBC container around a native program ([`native_part`]).

use super::container::{self, SigElement};
use super::link::{self, Linked};
use super::{Container, FourCC, HostShader, translate};
use crate::HwStage;

const TRIANGLE: u32 = 1 << 10 | 2 << 20;

/// One parameter export of the fixtures' mesh program: `exp param<n>` of 1.0 to `channels`
/// (a bit each, x first) of row `row` of the vertex signature, or of the primitive signature.
#[derive(Debug, Clone, Copy)]
struct Write {
    row: u8,
    primitive: bool,
    channels: u32,
}

/// The fixtures' mesh program: a position per lane, a triangle per lane, and the parameter
/// exports `writes` (the stage data lists them: [`mesh_stage_data`]).
fn mesh_code(writes: &[Write]) -> Vec<u32> {
    let mut code = vec![
        0xD765_0001, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001_00C1,
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E12_02FF, // v_mov_b32 v9, prim
        TRIANGLE,
    ];
    for (n, w) in writes.iter().enumerate() {
        // exp param<n> v7, v7, v7, v7 (the channels of `w` enabled)
        code.extend([0xF800_0200 | (n as u32) << 4 | w.channels, 0x0707_0707]);
    }
    code.extend([
        0xF800_0941, // exp prim v9, off, off, off done
        0x0000_0009,
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ]);
    code
}

/// The mesh program's stage data: its maxima, and `exp param<n>` as `writes` has it (bit 7
/// of the parameter: a row of the primitive signature).
fn mesh_stage_data(writes: &[Write], max_vertices: u16) -> Vec<u8> {
    const VS_OUTPUTS: usize = 0xc8;
    const VS_OUTPUT_COUNT: usize = 0x188;
    let mut data = stage_data(&[], Some((128, max_vertices, 32)));
    data[VS_OUTPUT_COUNT..VS_OUTPUT_COUNT + 4].copy_from_slice(&(writes.len() as u32).to_le_bytes());
    for (n, w) in writes.iter().enumerate() {
        let at = VS_OUTPUTS + 6 * n;
        data[at] = 6; // generic
        data[at + 1] = w.row;
        data[at + 2] = n as u8 | if w.primitive { 0x80 } else { 0 };
    }
    data
}

/// `v_interp_mov_f32 v2, p0, attr2.z` and `exp mrt0, v2, v2, v2, v2 done vm`.
fn pixel_code() -> Vec<u32> {
    vec![0xC80A_0A02, 0xF800_180F, 2 * 0x0101_0101, 0xBF81_0000]
}

/// A vertex program writing TEXCOORD0 and TEXCOORD1, then its position.
fn vertex_code() -> Vec<u32> {
    vec![
        0x7E02_02F2, // v_mov_b32 v1, 1.0
        0x7E04_0287, // v_mov_b32 v2, 7
        0xF800_020F,
        0x0202_0101, // exp param0, v1, v1, v2, v2
        0xF800_021F,
        0x0101_0101, // exp param1, v1, v1, v1, v1
        0xF800_08CF,
        0x0101_0101, // exp pos0, v1, v1, v1, v1 done
        0xBF81_0000, // s_endpgm
    ]
}

fn element(name: &str, index: u32, kind: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

/// The `DXIL` part of a container around one native program: its program header, the `XDAT`
/// index and the `XBCH`-packed `FSSM` blob (`native.rs` lays them out).
fn native_part(stage: HwStage, code: &[u32], data: &[u8]) -> Vec<u8> {
    const SLOTS: [(HwStage, usize); 8] = [
        (HwStage::Ls, 0x08),
        (HwStage::Hs, 0x10),
        (HwStage::Es, 0x18),
        (HwStage::Gs, 0x20),
        (HwStage::Vs, 0x28),
        (HwStage::Ps, 0x30),
        (HwStage::Ms, 0x48),
        (HwStage::Cs, 0x60),
    ];
    const RECORD: usize = 0x54;
    const REGS: usize = 0x50;
    const FLAG_WAVE32: u32 = 1 << 5;
    let mut payload = 6u64.to_le_bytes().to_vec();
    payload.resize(0x68, 0);
    for (_, at) in SLOTS {
        payload[at..at + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEF_u64.to_le_bytes());
    }
    let at = SLOTS.iter().find(|s| s.0 == stage).unwrap().1;
    let record = payload.len() as u64;
    payload[at..at + 8].copy_from_slice(&record.to_le_bytes());
    let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let code_at = record + RECORD as u64 + data.len() as u64;
    payload.extend_from_slice(&code_at.to_le_bytes());
    payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
    payload.extend_from_slice(&[0; 4]);
    payload.extend_from_slice(&FLAG_WAVE32.to_le_bytes());
    payload.resize(record as usize + REGS, 0);
    payload.extend_from_slice(&(31u32 | 31 << 20).to_le_bytes());
    payload.extend_from_slice(data);
    payload.extend_from_slice(&code);
    let mut blob = b"FSSM".to_vec();
    blob.extend_from_slice(&0x101u32.to_le_bytes());
    blob.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    blob.extend_from_slice(&payload);
    let mut buf = vec![0u8; zlib_rs::compress_bound(blob.len())];
    let (stream, rc) = zlib_rs::compress_slice(&mut buf, &blob, zlib_rs::DeflateConfig::new(6));
    assert_eq!(rc, zlib_rs::ReturnCode::Ok);
    let mut packed = b"XBCH".to_vec();
    packed.extend_from_slice(&3u32.to_le_bytes());
    packed.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    packed.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    packed.extend_from_slice(stream);
    let mut part = vec![0u8; 0x18];
    part.extend_from_slice(b"XDAT");
    part.extend_from_slice(&1u32.to_le_bytes());
    part.extend_from_slice(&0x24u32.to_le_bytes());
    let first = part.len() + 12;
    for v in [0x0du32, first as u32, packed.len() as u32] {
        part.extend_from_slice(&v.to_le_bytes());
    }
    part.extend_from_slice(&packed);
    part
}

/// Stage data (`native.rs` reads the I/O out of it): no export patches, and either pixel
/// inputs or the mesh's thread and vertex maxima.
fn stage_data(ps_inputs: &[(u8, u8)], mesh: Option<(u16, u16, u16)>) -> Vec<u8> {
    const PS_INPUTS: usize = 0x24;
    const PS_INPUT_COUNT: usize = 0x124;
    const PS_PATCH_SIZE: usize = 0x148;
    const PS_PATCHES: usize = 0x14c;
    const PATCH_BYTES: usize = 4 + 10 * 10 * 4;
    const MS_INFO: usize = 0x448;
    let mut data = vec![0u8; 0x1000];
    data[PS_PATCH_SIZE..PS_PATCH_SIZE + 4].copy_from_slice(&0u32.to_le_bytes());
    for mrt in 0..8 {
        let at = PS_PATCHES + mrt * PATCH_BYTES;
        data[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    }
    data[PS_INPUT_COUNT..PS_INPUT_COUNT + 4]
        .copy_from_slice(&(ps_inputs.len() as u32).to_le_bytes());
    for (n, &(row, attr)) in ps_inputs.iter().enumerate() {
        let at = PS_INPUTS + 8 * n;
        // usage 6 (generic), row, mask 0xf, attribute, second row, flat
        data[at] = 6;
        data[at + 1] = row;
        data[at + 2] = 0xf;
        data[at + 3] = attr;
        data[at + 5] = row;
        data[at + 7] = 1;
    }
    if let Some((threads, max_vertices, max_primitives)) = mesh {
        let mut info = [0u8; 8];
        info[0..2].copy_from_slice(&threads.to_le_bytes());
        info[2..4].copy_from_slice(&max_vertices.to_le_bytes());
        info[4..6].copy_from_slice(&max_primitives.to_le_bytes());
        info[7] = 2; // triangles
        data[MS_INFO..MS_INFO + 8].copy_from_slice(&info);
    }
    data
}

/// A mesh container: `outputs` float4 TEXCOORDs after the position, 128 lanes and
/// `max_vertices` vertices.
fn mesh_container(outputs: u32, max_vertices: u16) -> Vec<u8> {
    mesh_of(&texcoords(outputs, 1), &[], max_vertices)
}

/// `count` float4 TEXCOORDs, TEXCOORD n in row `first_row` + n.
fn texcoords(count: u32, first_row: u32) -> Vec<SigElement> {
    (0..count)
        .map(|i| element("TEXCOORD", i, 0, first_row + i, 4))
        .collect()
}

/// A mesh container of 128 lanes and `max_vertices` vertices: the position, then the
/// `vertex` outputs, and the per-primitive outputs `prims`. Its program writes the four
/// channels of the row of its first TEXCOORD2 (what the fixtures' pixel programs read), as
/// far as the element there has them.
fn mesh_of(vertex: &[SigElement], prims: &[SigElement], max_vertices: u16) -> Vec<u8> {
    let texcoord2 = |els: &[SigElement], primitive: bool| {
        els.iter()
            .find(|e| e.name == "TEXCOORD" && e.index == 2)
            .and_then(|e| e.row)
            .map(|row| Write {
                row: row as u8,
                primitive,
                channels: 0xF,
            })
    };
    let write = texcoord2(vertex, false).or(texcoord2(prims, true));
    mesh_writing(vertex, prims, max_vertices, write.as_slice())
}

/// [`mesh_of`], its program writing `writes` and nothing else of the parameters.
fn mesh_writing(
    vertex: &[SigElement],
    prims: &[SigElement],
    max_vertices: u16,
    writes: &[Write],
) -> Vec<u8> {
    let mut osg1 = vec![element("SV_Position", 0, 3, 0, 4)];
    osg1.extend_from_slice(vertex);
    let osg1 = container::signature(&osg1, false);
    let psg1 = container::signature(prims, false);
    let data = mesh_stage_data(writes, max_vertices);
    let part = native_part(HwStage::Ms, &mesh_code(writes), &data);
    let mut parts: Vec<(FourCC, &[u8])> = vec![(FourCC::OSG1, &osg1)];
    if !prims.is_empty() {
        parts.push((FourCC::PSG1, &psg1));
    }
    parts.push((FourCC::DXIL, &part));
    crate::dxbc::build(&parts)
}

/// A pixel container declaring `inputs` float4 TEXCOORDs (row n for TEXCOORD n) and reading
/// TEXCOORD2's z.
fn pixel_container(inputs: u32) -> Vec<u8> {
    pixel_of(&texcoords(inputs, 0), &pixel_code(), 0)
}

/// A pixel container declaring `inputs` and running `code`, its hardware attributes 0 to 2
/// rows 0 to 2 of the signature; `input_addr` (`SPI_PS_INPUT_ADDR`) says which barycentrics
/// its first registers hold.
fn pixel_of(inputs: &[SigElement], code: &[u32], input_addr: u32) -> Vec<u8> {
    const PS_INPUT_ADDR: usize = 0x134;
    let isg1 = container::signature(inputs, true);
    let mut data = stage_data(&[(0, 0), (1, 1), (2, 2)], None);
    data[PS_INPUT_ADDR..PS_INPUT_ADDR + 4].copy_from_slice(&input_addr.to_le_bytes());
    pixel_with(&isg1, code, &data)
}

/// A pixel container of the input signature part `isg1`, running `code` with the stage data
/// `data`.
fn pixel_with(isg1: &[u8], code: &[u32], data: &[u8]) -> Vec<u8> {
    let mut target = element("SV_Target", 0, 16, 0, 4);
    target.usage = 15;
    let osg1 = container::signature(&[target], false);
    let part = native_part(HwStage::Ps, code, data);
    crate::dxbc::build(&[
        (FourCC::ISG1, isg1),
        (FourCC::OSG1, &osg1),
        (FourCC::DXIL, &part),
    ])
}

/// A vertex container writing TEXCOORD0 (row 1) and TEXCOORD1 (row 2).
fn vertex_container() -> Vec<u8> {
    const VS_OUTPUTS: usize = 0xc8;
    const VS_OUTPUT_COUNT: usize = 0x188;
    let outputs = [(1u8, 0u8), (2, 1)];
    let mut data = vec![0u8; 0x1000];
    data[VS_OUTPUT_COUNT..VS_OUTPUT_COUNT + 4].copy_from_slice(&2u32.to_le_bytes());
    for (n, &(row, param)) in outputs.iter().enumerate() {
        let at = VS_OUTPUTS + 6 * n;
        data[at] = 6; // generic
        data[at + 1] = row;
        data[at + 2] = param;
    }
    let osg1 = container::signature(
        &[
            element("SV_Position", 0, 3, 0, 4),
            element("TEXCOORD", 0, 0, 1, 4),
            element("TEXCOORD", 1, 0, 2, 4),
        ],
        false,
    );
    let part = native_part(HwStage::Vs, &vertex_code(), &data);
    crate::dxbc::build(&[(FourCC::OSG1, &osg1), (FourCC::DXIL, &part)])
}

/// The rows of an emitted signature part, by semantic, index and register.
fn rows(shader: &HostShader, fourcc: FourCC) -> Vec<(String, u32, u32)> {
    let c = Container::parse(&shader.dxil).unwrap();
    let Some(part) = c.part(fourcc) else {
        return Vec::new();
    };
    let part = part.data;
    let u32_at = |at: usize| u32::from_le_bytes(part[at..at + 4].try_into().unwrap());
    let count = u32_at(0) as usize;
    let base = u32_at(4) as usize;
    (0..count)
        .map(|i| {
            let at = base + 32 * i;
            let name_at = u32_at(at + 4) as usize;
            let name = part[name_at..].split(|&b| b == 0).next().unwrap();
            (
                String::from_utf8_lossy(name).into_owned(),
                u32_at(at + 8),
                u32_at(at + 20),
            )
        })
        .collect()
}

fn validate(shader: &HostShader) {
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut shader.dxil.clone()).unwrap();
    }
    let _ = shader;
}

/// The 17 float4 TEXCOORDs over the position are 72 components: 128 vertices take 36,864
/// bytes, past D3D12's 32 KiB, and the validator refuses the translation made alone.
#[test]
fn a_mesh_over_the_limit_is_invalid_dxil_as_translated_alone() {
    let ms = translate(&mesh_container(17, 128)).unwrap();
    assert!(!super::mesh_fits(&ms.dxil).unwrap(), "the mesh fits 32 KiB");
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        assert!(
            v.validate(&mut ms.dxil.clone()).is_err(),
            "the validator took a mesh output over 32 KiB"
        );
    }
}

/// The mesh above is linked with the pixel program that reads TEXCOORD2: the mesh keeps
/// only the outputs the pixel reads (position and TEXCOORD2), the pixel only the input it
/// reads, and both validate.
#[test]
fn a_mesh_over_the_limit_links_with_its_pixel_program() {
    let ms_container = mesh_container(17, 128);
    let ps_container = pixel_container(17);
    let ms = translate(&ms_container).unwrap();
    let ps = translate(&ps_container).unwrap();
    let (lms, lps) = super::link_mesh(&ms_container, &ms, &ps_container, &ps)
        .unwrap()
        .expect("the pair is linked");
    assert!(super::mesh_fits(&lms.dxil).unwrap(), "the linked mesh does not fit");
    assert_eq!(
        rows(&lms, FourCC::OSG1),
        [
            ("SV_Position".to_owned(), 0, 0),
            ("TEXCOORD".to_owned(), 2, 3),
        ]
    );
    assert_eq!(rows(&lps, FourCC::ISG1), [("TEXCOORD".to_owned(), 2, 2)]);
    validate(&ps);
    validate(&lms);
    validate(&lps);
}

/// 64 components at 128 vertices are exactly 32,768 bytes: the mesh fits, and no linked pair
/// is made for it.
#[test]
fn a_mesh_at_the_limit_is_not_linked() {
    let ms_container = mesh_container(15, 128);
    let ps_container = pixel_container(15);
    let ms = translate(&ms_container).unwrap();
    let ps = translate(&ps_container).unwrap();
    assert!(super::mesh_fits(&ms.dxil).unwrap(), "the mesh does not fit");
    assert!(super::link_mesh(&ms_container, &ms, &ps_container, &ps)
        .unwrap()
        .is_none());
    validate(&ms);
}

/// The pixel side: the inputs the program never reads leave its signature, the one it reads
/// keeps its row and is read as the signature's first element now; leaving out an input it
/// reads is refused.
#[test]
fn a_linked_pixel_program_declares_only_the_inputs_it_reads() {
    let container = pixel_container(3);
    let whole = translate(&container).unwrap();
    assert_eq!(
        rows(&whole, FourCC::ISG1),
        [
            ("TEXCOORD".to_owned(), 0, 0),
            ("TEXCOORD".to_owned(), 1, 1),
            ("TEXCOORD".to_owned(), 2, 2),
        ]
    );
    let unread = vec![("TEXCOORD".to_owned(), 0), ("texcoord".to_owned(), 1)];
    let linked = link::with_linked(
        Linked {
            inputs: unread,
            outputs: Vec::new(),
        },
        || translate(&container),
    )
    .unwrap();
    assert_eq!(rows(&linked, FourCC::ISG1), [("TEXCOORD".to_owned(), 2, 2)]);
    validate(&whole);
    validate(&linked);
    let read = link::with_linked(
        Linked {
            inputs: vec![("TEXCOORD".to_owned(), 2)],
            outputs: Vec::new(),
        },
        || translate(&container),
    );
    assert!(
        matches!(&read, Err(crate::ShaderError::Unsupported { what, .. }) if *what == "an input left out of a linked pipeline read"),
        "{:?}",
        read.err()
    );
}

/// The other side, for a stage whose outputs a pixel program reads: an output left out is
/// neither stored nor declared, the others keep their rows and are stored to their element
/// as the signature numbers it now (SV_Position 0, TEXCOORD1 1).
#[test]
fn a_linked_stage_stores_only_the_outputs_it_keeps() {
    let container = vertex_container();
    let linked = link::with_linked(
        Linked {
            inputs: Vec::new(),
            outputs: vec![("TEXCOORD".to_owned(), 0)],
        },
        || translate(&container),
    )
    .unwrap();
    assert_eq!(
        rows(&linked, FourCC::OSG1),
        [
            ("SV_Position".to_owned(), 0, 0),
            ("TEXCOORD".to_owned(), 1, 2),
        ]
    );
    validate(&linked);
}

/// The pair of `ms_container` and `ps_container` as [`super::link_mesh`] links it; the mesh
/// program must be one that passes 32 KiB alone (else nothing is asked of the interface).
fn link(
    ms_container: &[u8],
    ps_container: &[u8],
) -> Result<Option<(HostShader, HostShader)>, crate::ShaderError> {
    let ms = translate(ms_container).unwrap();
    let ps = translate(ps_container).unwrap();
    assert!(!super::mesh_fits(&ms.dxil).unwrap(), "the mesh fits 32 KiB");
    super::link_mesh(ms_container, &ms, ps_container, &ps)
}

/// What [`super::mesh_interface`] finds wrong between the two (none: the interface is whole).
fn wrong(ms_container: &[u8], ps_container: &[u8]) -> Option<&'static str> {
    let ms = translate(ms_container).unwrap();
    let ps = translate(ps_container).unwrap();
    match super::mesh_interface(&ms, ps_container, &ps) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(e) => panic!("{e}"),
    }
}

/// Finding 2 (a) of the review of the linked pair: the pixel program reads TEXCOORD2's z, and
/// the mesh program exports TEXCOORD2 as x alone. The read is of a component nothing writes:
/// the pair is not linked (its pipeline is made of the shared pair, as with linking off).
#[test]
fn a_read_of_a_component_the_mesh_does_not_export_is_not_linked() {
    let mut vertex = texcoords(17, 1);
    vertex[2].cols = 1;
    let linked = link(&mesh_of(&vertex, &[], 128), &pixel_container(17)).unwrap();
    assert!(
        linked.is_none(),
        "linked a pair whose pixel program reads TEXCOORD2.z of a mesh exporting TEXCOORD2.x only"
    );
    assert_eq!(
        wrong(&mesh_of(&vertex, &[], 128), &pixel_container(17)),
        Some(link::COMPONENT)
    );
    // x, y and z are enough for a read of z.
    vertex[2].cols = 3;
    assert_eq!(wrong(&mesh_of(&vertex, &[], 128), &pixel_container(17)), None);
    assert!(link(&mesh_of(&vertex, &[], 128), &pixel_container(17)).unwrap().is_some());
}

/// Finding 2 (b): two mesh outputs are TEXCOORD2 (rows 3 and 18). Which of them the pixel
/// program's TEXCOORD2 is, is not defined: the pair is not linked.
#[test]
fn two_mesh_outputs_of_one_semantic_are_not_linked() {
    let mut vertex = texcoords(17, 1);
    vertex.push(element("TEXCOORD", 2, 0, 18, 4));
    let linked = link(&mesh_of(&vertex, &[], 128), &pixel_container(17)).unwrap();
    assert!(
        linked.is_none(),
        "linked a pair whose mesh program declares TEXCOORD2 twice"
    );
    assert_eq!(
        wrong(&mesh_of(&vertex, &[], 128), &pixel_container(17)),
        Some(link::TWICE)
    );
    // Per vertex and per primitive are one name too.
    vertex.pop();
    let ms = mesh_of(&vertex, &[element("TEXCOORD", 2, 0, 0, 4)], 128);
    assert_eq!(wrong(&ms, &pixel_container(17)), Some(link::TWICE));
    assert!(link(&ms, &pixel_container(17)).unwrap().is_none());
    // Per-primitive outputs are not pruned: two of a name nothing reads are both kept.
    let twice = [element("TEXCOORD", 9, 0, 0, 4), element("TEXCOORD", 9, 0, 1, 4)];
    let ms = mesh_of(&texcoords(17, 1), &twice, 128);
    assert_eq!(wrong(&ms, &pixel_container(17)), Some(link::TWICE));
}

/// Finding 2 (c), the type: the mesh program exports TEXCOORD2 as unsigned integers, the
/// pixel program reads it as floats. The pair is not linked.
#[test]
fn an_output_read_as_another_type_is_not_linked() {
    let mut vertex = texcoords(17, 1);
    vertex[2].comp = 5;
    let linked = link(&mesh_of(&vertex, &[], 128), &pixel_container(17)).unwrap();
    assert!(
        linked.is_none(),
        "linked a pair whose pixel program reads as float a TEXCOORD2 the mesh exports as uint"
    );
    assert_eq!(
        wrong(&mesh_of(&vertex, &[], 128), &pixel_container(17)),
        Some(link::TYPE)
    );
}

/// The vertex TEXCOORDs but TEXCOORD2, which is per primitive (row 0 of the primitive
/// signature): 68 vertex components at 128 vertices and 4 at 32 primitives, 35,328 bytes.
fn mesh_with_a_primitive_texcoord2() -> Vec<u8> {
    let mut vertex = texcoords(17, 1);
    vertex.remove(2);
    mesh_of(&vertex, &[element("TEXCOORD", 2, 0, 0, 4)], 128)
}

/// `v_interp_p1_f32 v2, v<i>, attr2.z` and `v_interp_p2_f32 v2, v<j>, attr2.z`, then
/// `exp mrt0, v2, v2, v2, v2 done vm`.
fn interpolating_pixel_code(i: u32, j: u32) -> Vec<u32> {
    let at = 2 << 18 | 2 << 10 | 2 << 8;
    vec![
        0xC800_0000 | at | i,
        0xC801_0000 | at | j,
        0xF800_180F,
        2 * 0x0101_0101,
        0xBF81_0000,
    ]
}

/// A per-primitive output the pixel program interpolates in the ordinary way (i in v0, j in
/// v1, at the pixel's centre) links: Direct3D reads a per-primitive attribute flat whatever
/// mode the pixel input declares (the mesh shader specification, its overview: "the driver
/// will still force the attribute to nointerpolate"), so the read is defined. The mesh
/// program writes the primitive's TEXCOORD2 and keeps it linked; the pixel program keeps its
/// read.
///
/// The second review of the linked pair: this replaces
/// `a_constant_output_read_interpolated_is_not_linked`, which asserted the opposite.
#[test]
fn a_per_primitive_output_read_interpolated_links() {
    const PERSP_CENTER: u32 = 1 << 1;
    let ms = mesh_with_a_primitive_texcoord2();
    let ps = pixel_of(&texcoords(17, 0), &interpolating_pixel_code(0, 1), PERSP_CENTER);
    assert_eq!(wrong(&ms, &ps), None);
    let (lms, lps) = link(&ms, &ps)
        .unwrap()
        .expect("a pair whose pixel program interpolates a per-primitive TEXCOORD2 is linked");
    assert!(stores(&lms, FourCC::OSG1).is_empty());
    assert_eq!(stores(&lms, FourCC::PSG1), [(2, 0xF, 0xF)]);
    assert_eq!(reads(&lps), [(2, 0xF, 0b0100)]);
    validate(&lms);
    validate(&lps);
}

/// Finding 2 (d): TEXCOORD2 is a per-primitive output, and the pixel program reads it per
/// vertex (an interpolation with weights it computes, `v3` here: P0 + i P10 + j P20 from the
/// three vertices' values). A primitive has no vertex values: the pair is not linked.
#[test]
fn a_per_primitive_output_read_per_vertex_is_not_linked() {
    let mut code = vec![0x7E06_02F0]; // v_mov_b32 v3, 0.5
    code.extend(interpolating_pixel_code(3, 3));
    let ps = pixel_of(&texcoords(17, 0), &code, 0);
    let linked = link(&mesh_with_a_primitive_texcoord2(), &ps).unwrap();
    assert!(
        linked.is_none(),
        "linked a pair whose pixel program reads per vertex a TEXCOORD2 the mesh exports per primitive"
    );
    assert_eq!(
        wrong(&mesh_with_a_primitive_texcoord2(), &ps),
        Some(link::PER_VERTEX)
    );
    // The same read of a per-vertex TEXCOORD2 is whole, and links.
    assert_eq!(wrong(&mesh_container(17, 128), &ps), None);
    assert!(link(&mesh_container(17, 128), &ps).unwrap().is_some());
}

/// What the check leaves alone. A per-primitive output read flat (the provoking vertex's
/// value, which is the primitive's) links: the mesh keeps its position and the primitive's
/// TEXCOORD2, and both validate. An input the pixel program declares and never reads is not
/// asked about, though the mesh program does not write it (the linked pixel program leaves
/// it out); one it reads is.
#[test]
fn a_whole_interface_links() {
    let ms_container = mesh_with_a_primitive_texcoord2();
    let ps_container = pixel_container(17);
    assert_eq!(wrong(&ms_container, &ps_container), None);
    let (lms, lps) = link(&ms_container, &ps_container)
        .unwrap()
        .expect("the pair is linked");
    assert_eq!(rows(&lms, FourCC::OSG1), [("SV_Position".to_owned(), 0, 0)]);
    assert_eq!(rows(&lms, FourCC::PSG1), [("TEXCOORD".to_owned(), 2, 0)]);
    assert_eq!(rows(&lps, FourCC::ISG1), [("TEXCOORD".to_owned(), 2, 2)]);
    validate(&lms);
    validate(&lps);

    let mut vertex = texcoords(18, 1);
    vertex.remove(5);
    assert_eq!(wrong(&mesh_of(&vertex, &[], 128), &pixel_container(18)), None);
    assert!(link(&mesh_of(&vertex, &[], 128), &pixel_container(18)).unwrap().is_some());
    let mut vertex = texcoords(18, 1);
    vertex.remove(2);
    assert_eq!(
        wrong(&mesh_of(&vertex, &[], 128), &pixel_container(18)),
        Some(link::UNWRITTEN)
    );
    assert!(link(&mesh_of(&vertex, &[], 128), &pixel_container(18)).unwrap().is_none());
}

/// The arbitrary semantics of an emitted output signature part, by index, with the
/// components each declares and the ones the program stores (both from the element's first).
fn stores(shader: &HostShader, fourcc: FourCC) -> Vec<(u32, u8, u8)> {
    masks(shader, fourcc)
        .into_iter()
        .map(|(index, declared, never)| (index, declared, declared & !never))
        .collect()
}

/// The arbitrary semantics of an emitted pixel program's inputs, by index, with the
/// components each declares and the ones the program reads.
fn reads(shader: &HostShader) -> Vec<(u32, u8, u8)> {
    masks(shader, FourCC::ISG1)
}

/// The rows of arbitrary semantics of an emitted signature part: index, mask and the byte
/// after it (inputs: read; outputs: never written), both moved to the element's first column.
fn masks(shader: &HostShader, fourcc: FourCC) -> Vec<(u32, u8, u8)> {
    let c = Container::parse(&shader.dxil).unwrap();
    let Some(part) = c.part(fourcc) else {
        return Vec::new();
    };
    let part = part.data;
    let u32_at = |at: usize| u32::from_le_bytes(part[at..at + 4].try_into().unwrap());
    (0..u32_at(0) as usize)
        .map(|i| u32_at(4) as usize + 32 * i)
        .filter(|&at| u32_at(at + 12) == 0)
        .map(|at| {
            let col = part[at + 24].trailing_zeros().min(7);
            (u32_at(at + 8), part[at + 24] >> col, (part[at + 25] & part[at + 24]) >> col)
        })
        .collect()
}

/// The written case (the second review, its tests): the fixtures' mesh program stores the
/// TEXCOORD2 the pixel program reads, and the linked pair keeps exactly that: the mesh
/// declares and stores TEXCOORD2's four components and nothing else of the parameters, the
/// pixel declares it and reads z.
#[test]
fn the_linked_mesh_stores_what_the_pixel_reads() {
    let ms_container = mesh_container(17, 128);
    let ps_container = pixel_container(17);
    let ms = translate(&ms_container).unwrap();
    let stored: Vec<(u32, u8, u8)> = (0..17)
        .map(|i| (i, 0xF, if i == 2 { 0xF } else { 0 }))
        .collect();
    assert_eq!(stores(&ms, FourCC::OSG1), stored);
    assert_eq!(wrong(&ms_container, &ps_container), None);
    let (lms, lps) = link(&ms_container, &ps_container)
        .unwrap()
        .expect("the pair is linked");
    assert_eq!(stores(&lms, FourCC::OSG1), [(2, 0xF, 0xF)]);
    assert_eq!(reads(&lps), [(2, 0xF, 0b0100)]);
    validate(&lms);
    validate(&lps);
    // Per primitive too.
    let ms = translate(&mesh_with_a_primitive_texcoord2()).unwrap();
    assert_eq!(stores(&ms, FourCC::PSG1), [(2, 0xF, 0xF)]);
}

/// The two stages pack TEXCOORD2 differently: the mesh program has it as x and y of row 3,
/// the pixel program as z and w of row 2 (after TEXCOORD17's two), and reads its first
/// component (the register's z). Direct3D matches the two by semantic, component for
/// component of the element: the pair links, each side keeping its own packing. A mesh
/// element of one component too few for the read does not.
#[test]
fn stages_that_pack_a_semantic_differently_link() {
    let mut vertex = texcoords(17, 1);
    vertex[2].cols = 2;
    let ms = mesh_of(&vertex, &[], 128);
    let mut inputs = texcoords(17, 0);
    inputs[2].col = 2;
    inputs[2].cols = 2;
    inputs.push(element("TEXCOORD", 17, 0, 2, 2));
    let ps = pixel_of(&inputs, &pixel_code(), 0);
    assert_eq!(wrong(&ms, &ps), None);
    let (lms, lps) = link(&ms, &ps).unwrap().expect("the pair is linked");
    assert_eq!(
        rows(&lms, FourCC::OSG1),
        [
            ("SV_Position".to_owned(), 0, 0),
            ("TEXCOORD".to_owned(), 2, 3),
        ]
    );
    assert_eq!(stores(&lms, FourCC::OSG1), [(2, 0b11, 0b11)]);
    assert_eq!(rows(&lps, FourCC::ISG1), [("TEXCOORD".to_owned(), 2, 2)]);
    assert_eq!(reads(&lps), [(2, 0b11, 0b01)]);
    validate(&lms);
    validate(&lps);

    // The pixel program's TEXCOORD2 is y, z and w of its row now, and the read of the
    // register's z its second component, which a mesh TEXCOORD2 of one component lacks.
    let mut inputs = texcoords(17, 0);
    inputs[2].col = 1;
    inputs[2].cols = 3;
    let ps = pixel_of(&inputs, &pixel_code(), 0);
    let mut vertex = texcoords(17, 1);
    vertex[2].cols = 1;
    let ms = mesh_of(&vertex, &[], 128);
    assert_eq!(wrong(&ms, &ps), Some(link::COMPONENT));
    assert!(link(&ms, &ps).unwrap().is_none());
}

/// The second review, its interface check: a declaration is not a write. The mesh program
/// declares TEXCOORD2 as four floats and stores x alone, or nothing of it; the pixel program
/// reads z. The pair is not linked.
#[test]
fn a_read_of_a_component_the_mesh_never_stores_is_not_linked() {
    let vertex = texcoords(17, 1);
    let x_alone = [Write {
        row: 3,
        primitive: false,
        channels: 0b0001,
    }];
    for writes in [&x_alone[..], &[]] {
        let ms = mesh_writing(&vertex, &[], 128, writes);
        let stored = stores(&translate(&ms).unwrap(), FourCC::OSG1)[2];
        assert_eq!(stored, (2, 0xF, writes.len() as u8), "the fixture's stores");
        assert!(
            link(&ms, &pixel_container(17)).unwrap().is_none(),
            "linked a pair whose pixel program reads TEXCOORD2.z of a mesh storing {stored:?}"
        );
        assert_eq!(wrong(&ms, &pixel_container(17)), Some(link::COMPONENT));
    }
    // x, y and z stored are enough for the read of z.
    let xyz = [Write {
        row: 3,
        primitive: false,
        channels: 0b0111,
    }];
    let ms = mesh_writing(&vertex, &[], 128, &xyz);
    assert_eq!(wrong(&ms, &pixel_container(17)), None);
    assert!(link(&ms, &pixel_container(17)).unwrap().is_some());
}

/// The second review: two mesh outputs of one semantic that no pixel input reads are both
/// pruned, so nothing of them reaches the host and the pair links (TEXCOORD9 twice here,
/// the pixel program reading TEXCOORD2).
#[test]
fn two_outputs_of_one_semantic_nothing_reads_are_pruned() {
    let mut vertex = texcoords(17, 1);
    vertex.push(element("TEXCOORD", 9, 0, 18, 4));
    let ms = mesh_of(&vertex, &[], 128);
    assert_eq!(wrong(&ms, &pixel_container(17)), None);
    let (lms, _) = link(&ms, &pixel_container(17))
        .unwrap()
        .expect("a pair whose mesh declares an unread TEXCOORD9 twice is linked");
    assert_eq!(
        rows(&lms, FourCC::OSG1),
        [
            ("SV_Position".to_owned(), 0, 0),
            ("TEXCOORD".to_owned(), 2, 3),
        ]
    );
    validate(&lms);
}

/// An element of the check, for the cases no native program of these fixtures reaches.
fn port(name: &str, index: u32, kind: u8, comp: u8, used: u8, primitive: bool) -> link::Port {
    link::Port {
        name: name.to_owned(),
        index,
        kind,
        comp,
        interp: 1,
        row: Some(index),
        cols: 4,
        used,
        known: true,
        primitive,
    }
}

/// The second review: system values a stage must supply. A pixel program that reads
/// `SV_PrimitiveID` (kind 10) with no mesh output of that kind and index does not link
/// (Direct3D wants the mesh program to write it); with one it does. What the rasterizer
/// makes itself (the front face, kind 13; the position, the sample index, the coverage,
/// barycentrics) needs no mesh output; an input never read is not asked about.
#[test]
fn a_system_value_the_mesh_must_supply_and_does_not_is_not_linked() {
    let position = port("", 0, 3, 3, 0xF, false);
    let texcoord = port("TEXCOORD", 2, 0, 3, 0xF, false);
    let outputs = [position.clone(), texcoord.clone()];
    let reads_id = [texcoord.clone(), port("", 0, 10, 1, 1, false)];
    assert_eq!(
        link::interface(&outputs, &reads_id, &[]),
        Some(link::SYSTEM_VALUE),
        "a read of SV_PrimitiveID no mesh output supplies passed"
    );
    let with_id = [position.clone(), texcoord.clone(), port("", 0, 10, 1, 1, true)];
    assert_eq!(link::interface(&with_id, &reads_id, &[]), None);
    // another index is another value
    let other = [position.clone(), texcoord.clone(), port("", 1, 10, 1, 1, true)];
    assert_eq!(
        link::interface(&other, &reads_id, &[]),
        Some(link::SYSTEM_VALUE)
    );
    for kind in [4, 5, 6, 7] {
        let reads = [texcoord.clone(), port("", 0, kind, 1, 1, false)];
        assert_eq!(
            link::interface(&outputs, &reads, &[]),
            Some(link::SYSTEM_VALUE),
            "kind {kind}"
        );
    }
    for kind in [3, 12, 13, 14, 15, 28] {
        let reads = [texcoord.clone(), port("", 0, kind, 1, 1, false)];
        assert_eq!(link::interface(&outputs, &reads, &[]), None, "kind {kind}");
    }
    let unread = [texcoord.clone(), port("", 0, 10, 1, 0, false)];
    assert_eq!(link::interface(&outputs, &unread, &[]), None);
}

/// The third review: a declaration is not a write for a system value either. Of a value the
/// stage before must supply, the mesh output of its kind and index stores every component
/// the pixel program reads, counted from the element's first as for an arbitrary semantic. A
/// `SV_PrimitiveID` (kind 10) declared and never stored does not link; a `SV_ClipDistance0`
/// (kind 6) stored in its first component and read in its second does not; stored in both it
/// does. The kind and the index are still what matches the two.
#[test]
fn a_system_value_the_mesh_does_not_store_is_not_linked() {
    let position = port("", 0, 3, 3, 0xF, false);
    let texcoord = port("TEXCOORD", 2, 0, 3, 0xF, false);
    let with = |supplied: link::Port| [position.clone(), texcoord.clone(), supplied];
    let reads_id = [texcoord.clone(), port("", 0, 10, 1, 0b1, false)];
    assert_eq!(
        link::interface(&with(port("", 0, 10, 1, 0, true)), &reads_id, &[]),
        Some(link::COMPONENT),
        "a read of an SV_PrimitiveID the mesh program declares and never stores passed"
    );
    assert_eq!(link::interface(&with(port("", 0, 10, 1, 0b1, true)), &reads_id, &[]), None);

    let reads_second = [texcoord.clone(), port("", 0, 6, 3, 0b10, false)];
    assert_eq!(
        link::interface(&with(port("", 0, 6, 3, 0b01, false)), &reads_second, &[]),
        Some(link::COMPONENT),
        "a read of SV_ClipDistance0's second component, of which the mesh program stores the first, passed"
    );
    assert_eq!(link::interface(&with(port("", 0, 6, 3, 0b11, false)), &reads_second, &[]), None);
    // The first component read of that output is whole.
    let reads_first = [texcoord.clone(), port("", 0, 6, 3, 0b01, false)];
    assert_eq!(link::interface(&with(port("", 0, 6, 3, 0b01, false)), &reads_first, &[]), None);
    // Another index is another value, however much of it is stored.
    assert_eq!(
        link::interface(&with(port("", 1, 6, 3, 0b11, false)), &reads_second, &[]),
        Some(link::SYSTEM_VALUE)
    );
    // What the rasterizer makes is asked of no mesh output, stored or not.
    let reads_face = [texcoord.clone(), port("", 0, 13, 1, 0b1, false)];
    assert_eq!(link::interface(&with(port("", 0, 13, 1, 0, false)), &reads_face, &[]), None);

    // An element whose row of the signature part was not found: its components are not
    // known, and are taken as all of them. It is matched by kind and index and asked nothing
    // more: not every component of the mesh output for an input that may read one, and not a
    // store the part would have named for an output.
    let unknown = |used: u8| link::Port { known: false, ..port("", 0, 6, 3, used, false) };
    let reads_some = [texcoord.clone(), unknown(0b1111)];
    assert_eq!(link::interface(&with(port("", 0, 6, 3, 0b01, false)), &reads_some, &[]), None);
    assert_eq!(link::interface(&with(unknown(0b1111)), &reads_second, &[]), None);
    assert_eq!(
        link::interface(&with(port("", 1, 6, 3, 0b11, false)), &reads_some, &[]),
        Some(link::SYSTEM_VALUE)
    );
}

/// The second review, its other concerns: a pixel input without a register. `PSV0` keeps no
/// column for it and `ISG1` keeps its mask (the last component here, as the translation
/// declares an element whose console mask is empty). Never read, it is a dead declaration:
/// the interface is whole and the linked pixel program leaves it out.
#[test]
fn a_dead_input_without_a_register_is_pruned() {
    let mut inputs = texcoords(17, 0);
    let mut dead = element("TEXCOORD", 20, 0, 0, 1);
    dead.row = None;
    dead.col = 3;
    inputs.push(dead);
    let ps = pixel_of(&inputs, &pixel_code(), 0);
    let shared = translate(&ps).unwrap();
    assert_eq!(
        (rows(&shared, FourCC::ISG1)[17].clone(), reads(&shared)[17]),
        (("TEXCOORD".to_owned(), 20, u32::MAX), (20, 0b1, 0)),
        "the fixture's dead input"
    );
    let ms = mesh_container(17, 128);
    assert_eq!(wrong(&ms, &ps), None);
    let (_, lps) = link(&ms, &ps).unwrap().expect("the pair is linked");
    assert_eq!(rows(&lps, FourCC::ISG1), [("TEXCOORD".to_owned(), 2, 2)]);
    validate(&lps);
}

/// Sixteen float4 TEXCOORDs no fixture's pixel program reads (TEXCOORD4 to TEXCOORD19, rows
/// 1 to 16): what takes a mesh program past 32 KiB at 128 vertices.
fn unread_texcoords() -> Vec<SigElement> {
    (0..16).map(|i| element("TEXCOORD", 4 + i, 0, 1 + i, 4)).collect()
}

/// The third review, its counterexample: a per-vertex scalar and a per-primitive scalar in one
/// pixel row. TEXCOORD3 is x of row 2 and TEXCOORD2 its z; the pixel program fetches
/// TEXCOORD3 per vertex (weights it computes) and reads TEXCOORD2 flat (P0), and both reach
/// its target. The mesh program writes TEXCOORD3 per vertex and TEXCOORD2 per primitive
/// (35,456 bytes with its sixteen unread float4 outputs; 2,688 linked). Nothing fetches the
/// primitive's element per vertex: the pair links, as it does without the interface check.
#[test]
fn a_flat_read_beside_a_per_vertex_fetch_in_one_row_links() {
    let code = [
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0xC808_0803, // v_interp_p1_f32 v2, v3, attr2.x
        0xC809_0803, // v_interp_p2_f32 v2, v3, attr2.x
        0xC812_0A02, // v_interp_mov_f32 v4, p0, attr2.z
        0xF800_180F, // exp mrt0, v2, v4, v2, v4 done vm
        0x0402_0402,
        0xBF81_0000, // s_endpgm
    ];
    let mut flat = element("TEXCOORD", 2, 0, 2, 1);
    flat.col = 2;
    let ps = pixel_of(&[element("TEXCOORD", 3, 0, 2, 1), flat], &code, 0);
    let mut vertex = unread_texcoords();
    vertex.push(element("TEXCOORD", 3, 0, 17, 1));
    let writes = [
        Write { row: 17, primitive: false, channels: 0b0001 },
        Write { row: 0, primitive: true, channels: 0b0001 },
    ];
    let ms = mesh_writing(&vertex, &[element("TEXCOORD", 2, 0, 0, 1)], 128, &writes);
    assert_eq!(
        wrong(&ms, &ps),
        None,
        "a flat read of a per-primitive TEXCOORD2 was refused for the per-vertex fetch of the TEXCOORD3 beside it"
    );
    let (lms, lps) = link(&ms, &ps).unwrap().expect("the pair is linked");
    assert_eq!(stores(&lms, FourCC::OSG1), [(3, 0b1, 0b1)]);
    assert_eq!(stores(&lms, FourCC::PSG1), [(2, 0b1, 0b1)]);
    assert_eq!(reads(&lps), [(3, 0b1, 0b1), (2, 0b1, 0b1)]);
    validate(&lms);
    validate(&lps);

    // The two the other way round: the element fetched per vertex is the primitive's.
    let mut vertex = unread_texcoords();
    vertex.push(element("TEXCOORD", 2, 0, 17, 1));
    let ms = mesh_writing(&vertex, &[element("TEXCOORD", 3, 0, 0, 1)], 128, &writes);
    assert_eq!(wrong(&ms, &ps), Some(link::PER_VERTEX));
    assert!(link(&ms, &ps).unwrap().is_none());
}

/// A pixel container whose attribute 2 is a parameter of 16-bit values packed two per dword:
/// TEXCOORD2 (row 2, two components) in its first dword, TEXCOORD3 (row 3, two components)
/// in its second. Its program reads the low half of the parameter's second dword, which is
/// TEXCOORD3's x, with weights it computes (`v3`).
fn packed_half_pixel() -> Vec<u8> {
    const PS_INPUTS: usize = 0x24;
    const FLAT: u8 = 1;
    const FP16: u8 = 1 << 3;
    const PACK2: u8 = 1 << 4;
    let code = [
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0xD742_0002, // v_interp_p1ll_f16 v2, v3, attr2.y
        0x0002_0642,
        0xD75A_0002, // v_interp_p2_f16 v2, v3, v2, attr2.y
        0x040A_0642,
        0xF800_180F, // exp mrt0, v2, v2, v2, v2 done vm
        2 * 0x0101_0101,
        0xBF81_0000, // s_endpgm
    ];
    let inputs = [element("TEXCOORD", 2, 0, 2, 2), element("TEXCOORD", 3, 0, 3, 2)];
    let mut isg1 = container::signature(&inputs, true);
    for n in 0..inputs.len() {
        // The component type of each row: a 16-bit float.
        isg1[8 + 32 * n + 16..][..4].copy_from_slice(&6u32.to_le_bytes());
    }
    let mut data = stage_data(&[(0, 0), (1, 1), (2, 2)], None);
    // Attribute 2: rows 2 and 3, two 16-bit values per dword.
    data[PS_INPUTS + 8 * 2 + 5] = 3;
    data[PS_INPUTS + 8 * 2 + 7] = FLAT | FP16 | PACK2;
    pixel_with(&isg1, &code, &data)
}

/// The third review: the per-vertex fetch of a packed half. The read is of attribute 2
/// (signature row 2), and the half it names is in the parameter's second row, row 3:
/// TEXCOORD3, which is what `attributeAtVertex` fetches. The mesh program writes TEXCOORD3
/// per primitive, and a primitive has no vertex values: the pair is not linked. With
/// TEXCOORD3 per vertex it is.
#[test]
fn a_packed_half_fetched_per_vertex_from_a_primitive_output_is_not_linked() {
    let ps = packed_half_pixel();
    let shared = translate(&ps).unwrap();
    assert_eq!(
        reads(&shared),
        [(2, 0b11, 0), (3, 0b11, 0b01)],
        "the fixture reads TEXCOORD3.x through attribute 2"
    );
    let mut vertex = unread_texcoords();
    vertex.push(element("TEXCOORD", 2, 0, 17, 2));
    let primitive = Write { row: 0, primitive: true, channels: 0b0011 };
    let ms = mesh_writing(&vertex, &[element("TEXCOORD", 3, 0, 0, 2)], 128, &[primitive]);
    assert_eq!(
        wrong(&ms, &ps),
        Some(link::PER_VERTEX),
        "a per-vertex fetch of a per-primitive TEXCOORD3 through the packed half of another row passed"
    );
    assert!(
        link(&ms, &ps).unwrap().is_none(),
        "linked a pair whose pixel program fetches per vertex a TEXCOORD3 the mesh exports per primitive"
    );
    // The same read of a per-vertex TEXCOORD3 is whole, and links.
    vertex.push(element("TEXCOORD", 3, 0, 18, 2));
    let per_vertex = Write { row: 18, primitive: false, channels: 0b0011 };
    let ms = mesh_writing(&vertex, &[], 128, &[per_vertex]);
    assert_eq!(wrong(&ms, &ps), None);
    let (lms, lps) = link(&ms, &ps).unwrap().expect("the pair is linked");
    assert_eq!(stores(&lms, FourCC::OSG1), [(3, 0b11, 0b11)]);
    assert_eq!(reads(&lps), [(3, 0b11, 0b01)]);
    validate(&lms);
    validate(&lps);
}
