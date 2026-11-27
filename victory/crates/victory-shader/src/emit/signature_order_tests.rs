//! The order of a signature part's rows, against the validator: it rebuilds
//! the part from the module's metadata, rows sorted by register and then by
//! the offset of the semantic's name, and refuses a part that differs
//! ("Container part 'Program Input Signature' does not match expected").
//!
//! Synthetic GFX10 pixel programs whose console signature lists its
//! elements in one of two independently chosen orders, translated and
//! handed to `dxil.dll`; each
//! test also reads the order back out of the emitted part, so it says
//! something where there is no validator.

use super::bitcode::{Attr, Term};
use super::container::SigElement;
use super::shader::Shader;
use super::{HostShader, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::{io, ir, resource, structure};

/// `SPI_PS_INPUT_ADDR`: PERSP_CENTER (i in v0, j in v1).
const PERSP_CENTER: u32 = 1 << 1;
const S_ENDPGM: u32 = 0xBF81_0000;

/// `v_interp_p1_f32 v<dst>, v0, attr<attr>.<chan>` and
/// `v_interp_p2_f32 v<dst>, v1, attr<attr>.<chan>`: the attribute's channel
/// interpolated at the pixel's centre.
pub(super) fn interp(dst: u32, attr: u32, chan: u32) -> [u32; 2] {
    let at = dst << 18 | attr << 10 | chan << 8;
    [0xC800_0000 | at, 0xC801_0001 | at]
}

/// `exp mrt0, v<r>, v<g>, v<b>, v<a> done vm`.
pub(super) fn exp_mrt0(r: u32, g: u32, b: u32, a: u32) -> [u32; 2] {
    [0xF800_180F, r | g << 8 | b << 16 | a << 24]
}

/// A float element of an arbitrary semantic, as the console's part lists it.
pub(super) fn float(name: &str, index: u32, row: u32, col: u8, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: false,
    }
}

/// A console `ISG1` part with its rows in the order given, each with its
/// own name: written here, not by `container::signature`, so the order the
/// program under test is handed does not depend on the writer under test.
pub(super) fn console_part(elements: &[SigElement]) -> Vec<u8> {
    let mut rows = Vec::new();
    let mut names = Vec::new();
    let table_end = 8 + 32 * elements.len() as u32;
    for e in elements {
        // System value and component type as the part encodes them: an
        // arbitrary semantic or the front face; float or unsigned.
        let system_value = if e.kind == 13 { 9 } else { 0 };
        let comp = if e.comp == 9 { 3 } else { 1 };
        let name_at = table_end + names.len() as u32;
        names.extend(e.name.bytes().chain([0]));
        for v in [0, name_at, e.index, system_value, comp, e.row.unwrap_or(u32::MAX)] {
            rows.extend(v.to_le_bytes());
        }
        rows.extend([((1u8 << e.cols) - 1) << e.col, 0, 0, 0]);
        rows.extend(0u32.to_le_bytes());
    }
    let mut part = Vec::new();
    part.extend((elements.len() as u32).to_le_bytes());
    part.extend(8u32.to_le_bytes());
    part.extend(rows);
    part.extend(names);
    part
}

/// Translates a pixel program whose console signature lists `inputs` in
/// that order, with hardware attribute `n` holding row `n` of the
/// signature, and one float4 target.
pub(super) fn pixel(words: &[u32], inputs: &[SigElement], flat: bool) -> HostShader {
    let rows = inputs.iter().filter_map(|e| e.row).max().map_or(0, |r| r + 1);
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: (0..rows as u8)
                .map(|row| PsInput {
                    usage: Usage::Generic,
                    row,
                    mask: 0b1111,
                    attr: row,
                    flat,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                })
                .collect(),
            input_addr: PERSP_CENTER,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let target = SigElement {
        name: "SV_Target".into(),
        kind: 16,
        cols: 4,
        usage: 0xF,
        ..float("", 0, 0, 0, 4)
    };
    let isg1 = console_part(inputs);
    let osg1 = super::container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    super::emit(&c, &native, &p, &st, &io, &res).unwrap()
}

/// The rows of the translation's `ISG1` part, in the part's order.
pub(super) fn emitted_inputs(dxil: &[u8]) -> Vec<SigElement> {
    let c = Container::parse(dxil).unwrap();
    super::signature(c.part(FourCC::ISG1).unwrap().data).unwrap()
}

/// What `dxil.dll` says of the translation; `None` where there is no
/// validator (the test's own look at the part still stands).
pub(super) fn validated(dxil: &[u8]) -> Option<Result<(), String>> {
    match validator::Validator::load(validator::Validator::default_path().as_deref()) {
        Ok(mut v) => Some(v.validate(&mut dxil.to_vec())),
        Err(_) => {
            eprintln!("no validator; the emitted part checked alone");
            None
        }
    }
}

/// The semantic and index of each row of the emitted part.
fn listed(shader: &HostShader) -> Vec<(String, u32)> {
    emitted_inputs(&shader.dxil).into_iter().map(|e| (e.name, e.index)).collect()
}

/// Elements of one register listed with the later column first (a made-up
/// signature: COLOR5 in w before NORMAL0 in xyz): the part keeps that
/// order, where sorting the register's elements by column was refused.
#[test]
fn a_register_listed_with_its_later_column_first_validates() {
    let inputs = [float("COLOR", 5, 0, 3, 1), float("NORMAL", 0, 0, 0, 3)];
    let [p1, p2] = interp(2, 0, 0); // v2 = NORMAL0.x
    let [q1, q2] = interp(3, 0, 3); // v3 = COLOR5 (attr0.w)
    let [e0, e1] = exp_mrt0(2, 3, 2, 3);
    let shader = pixel(&[p1, p2, q1, q2, e0, e1, S_ENDPGM], &inputs, false);
    if let Some(r) = validated(&shader.dxil) {
        r.unwrap();
    }
    assert_eq!(
        listed(&shader),
        [("COLOR".to_owned(), 5), ("NORMAL".to_owned(), 0)],
        "a register's elements in the order of their names, which is the console's"
    );
}

/// Elements listed with the later register first: the part has them by
/// register (keeping the console's order across registers was refused).
#[test]
fn registers_listed_with_the_later_one_first_validate() {
    let inputs = [float("TEXCOORD", 1, 1, 0, 2), float("TEXCOORD", 0, 0, 0, 2)];
    let [p1, p2] = interp(2, 0, 0); // v2 = TEXCOORD0.x
    let [q1, q2] = interp(3, 1, 0); // v3 = TEXCOORD1.x
    let [e0, e1] = exp_mrt0(2, 3, 2, 3);
    let shader = pixel(&[p1, p2, q1, q2, e0, e1, S_ENDPGM], &inputs, false);
    if let Some(r) = validated(&shader.dxil) {
        r.unwrap();
    }
    assert_eq!(
        listed(&shader),
        [("TEXCOORD".to_owned(), 0), ("TEXCOORD".to_owned(), 1)],
        "by register"
    );
}

/// Both shapes at once, and a third register between them, built straight
/// on the emitter (no console program): elements read in element order
/// into the channels of target 0.
#[test]
fn signature_rows_take_the_validators_order() {
    for (label, inputs) in [
        ("a later row first", vec![float("TEXCOORD", 1, 1, 0, 2), float("TEXCOORD", 0, 0, 0, 2)]),
        ("a later column first", vec![float("TEXCOORD", 16, 0, 3, 1), float("TANGENT", 0, 0, 0, 3)]),
        (
            "both",
            vec![
                float("TEXCOORD", 3, 2, 2, 1),
                float("TEXCOORD", 1, 1, 0, 1),
                float("TEXCOORD", 2, 2, 0, 1),
                float("TEXCOORD", 0, 0, 0, 1),
            ],
        ),
    ] {
        let mut s = Shader::new(true);
        let b = s.entry;
        let undef = s.m.undef(s.t.i32);
        let mut channel = 0u8;
        let mut elements = Vec::new();
        for (id, e) in inputs.into_iter().enumerate() {
            for c in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(c), undef];
                let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
                let out = [s.m.i32(0), s.m.i32(0), s.m.i8(channel), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
                channel += 1;
            }
            elements.push(SigElement { interp: 2, usage: (1 << e.cols) - 1, ..e });
        }
        s.inputs = elements;
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            kind: 16,
            usage: 0xF,
            ..float("", 0, 0, 0, 4)
        });
        s.m.terminate(b, Term::Ret);
        let dxil = s.finish();
        let rows: Vec<u32> = emitted_inputs(&dxil).iter().filter_map(|e| e.row).collect();
        assert!(rows.is_sorted(), "{label}: registers {rows:?}");
        if let Some(r) = validated(&dxil) {
            r.unwrap_or_else(|e| panic!("{label}: {e}"));
        }
    }
}
