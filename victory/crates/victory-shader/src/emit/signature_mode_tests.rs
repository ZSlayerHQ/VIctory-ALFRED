//! One interpolation mode per signature register, against the validator
//! ("signature element ... has interpolation mode that differs from another
//! element packed into the same row"): synthetic GFX10 pixel programs
//! whose console signature packs an element the program does not read
//! beside one it reads another way, translated and handed to `dxil.dll`.
//! Each test also reads the modes back out of the emitted `PSV0`, so it
//! says something where there is no validator: the element that is not
//! read changes, an element that is read never does.

use super::container::SigElement;
use super::signature_order_tests::{exp_mrt0, float, interp, pixel, validated};
use crate::dxbc::{Container, FourCC};

const S_ENDPGM: u32 = 0xBF81_0000;
/// DXIL interpolation modes.
const CONSTANT: u8 = 1;
const LINEAR: u8 = 2;

/// `v_interp_mov_f32 v<dst>, p0, attr<attr>.<chan>`: the provoking vertex's
/// value, not interpolated.
fn flat(dst: u32, attr: u32, chan: u32) -> u32 {
    0xC802_0002 | dst << 18 | attr << 10 | chan << 8
}

/// `SV_IsFrontFace` in `row`, column `col`.
fn front_face(row: u32, col: u8) -> SigElement {
    SigElement {
        name: "SV_IsFrontFace".into(),
        kind: 13,
        comp: 5,
        ..float("", 0, row, col, 1)
    }
}

/// An unsigned element of an arbitrary semantic.
fn unsigned(name: &str, index: u32, row: u32, col: u8, cols: u8) -> SigElement {
    SigElement { comp: 5, ..float(name, index, row, col, cols) }
}

/// The interpolation mode of each input element, in element order, as the
/// translation's `PSV0` declares them (runtime info version 2: the info,
/// the bindings, the string table, the index table, then 16-byte elements
/// with the mode in byte 13).
fn input_modes(dxil: &[u8]) -> Vec<u8> {
    let c = Container::parse(dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let u32_at = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().unwrap()) as usize;
    let info = 4;
    let mut at = info + u32_at(0);
    let inputs = usize::from(psv[info + 28]);
    let bindings = u32_at(at);
    at += 4;
    if bindings > 0 {
        at += 4 + bindings * u32_at(at);
    }
    at += 4 + u32_at(at); // the strings
    at += 4 + 4 * u32_at(at); // the semantic indices
    assert_eq!(u32_at(at), 16, "the size of a PSV0 signature element");
    at += 4;
    (0..inputs).map(|i| psv[at + 16 * i + 13]).collect()
}

/// A float the program does not read (TEXCOORD3 here) shares a register
/// with the front face, which is constant. The float was declared linear by
/// its type and the register refused.
#[test]
fn an_unread_input_beside_the_front_face_is_constant() {
    let inputs = [float("TEXCOORD", 0, 0, 0, 4), float("TEXCOORD", 3, 1, 0, 1), front_face(1, 1)];
    let [p1, p2] = interp(2, 0, 0); // v2 = TEXCOORD0.x
    let [e0, e1] = exp_mrt0(2, 2, 2, 2);
    let shader = pixel(&[p1, p2, e0, e1, S_ENDPGM], &inputs, false);
    if let Some(r) = validated(&shader.dxil) {
        r.unwrap();
    }
    assert_eq!(input_modes(&shader.dxil), [LINEAR, CONSTANT, CONSTANT]);
}

/// An element the program does not read beside one it reads at the
/// provoking vertex: the register is constant.
#[test]
fn an_unread_input_beside_one_read_flat_is_constant() {
    let inputs = [float("TEXCOORD", 0, 0, 0, 2), float("TEXCOORD", 1, 0, 2, 2)];
    let [e0, e1] = exp_mrt0(2, 2, 2, 2);
    let shader = pixel(&[flat(2, 0, 0), e0, e1, S_ENDPGM], &inputs, false);
    if let Some(r) = validated(&shader.dxil) {
        r.unwrap();
    }
    assert_eq!(input_modes(&shader.dxil), [CONSTANT, CONSTANT]);
}

/// Two elements of one register read two ways keep the modes they are
/// read with: nothing a program reads is redeclared. (The validator still
/// refuses this register; it needs the elements in registers of their
/// own.)
#[test]
fn inputs_read_two_ways_in_one_register_keep_their_modes() {
    let inputs = [float("TEXCOORD", 0, 0, 0, 2), float("TEXCOORD", 1, 0, 2, 2)];
    let [p1, p2] = interp(2, 0, 0); // v2 = TEXCOORD0.x, interpolated
    let [e0, e1] = exp_mrt0(2, 3, 2, 3);
    let words = [p1, p2, flat(3, 0, 2), e0, e1, S_ENDPGM]; // v3 = TEXCOORD1.x, flat
    let shader = pixel(&words, &inputs, false);
    assert_eq!(input_modes(&shader.dxil), [LINEAR, CONSTANT]);
}

/// An integer may only be constant, so an unread one beside an
/// interpolated float cannot take the float's mode, and the float is read:
/// both keep theirs. (Refused still, for the same reason as above.)
#[test]
fn an_unread_integer_beside_an_interpolated_float_is_left() {
    let inputs = [float("TEXCOORD", 0, 0, 0, 1), unsigned("COLOR", 0, 0, 1, 1)];
    let [p1, p2] = interp(2, 0, 0);
    let [e0, e1] = exp_mrt0(2, 2, 2, 2);
    let shader = pixel(&[p1, p2, e0, e1, S_ENDPGM], &inputs, false);
    assert_eq!(input_modes(&shader.dxil), [LINEAR, CONSTANT]);
}
