//! EXTRQ and INSERTQ against a bit-by-bit reference written from APM4's
//! text, the manual's own examples, and the CPU where it has SSE4A.

use super::super::bitfield::{
    BitRange, Xmm, extract, extract_range_held_in, insert, insert_range_held_in,
};
use super::Values;

/// The field's length and index as APM4 gives them: six bits each, "other
/// bits of the field are ignored", a length of 0 meaning 64 (pp. 172, 195).
fn length_and_index(length: u8, index: u8) -> (u32, u32) {
    let length = u32::from(length & 0x3f);
    (if length == 0 { 64 } else { length }, u32::from(index & 0x3f))
}

/// EXTRQ's low quadword, one bit at a time: bit `i` of the result is bit
/// `index + i` of `value`, for `i` below the length, where that bit exists
/// (past bit 63 there is none: the choice for what APM4 leaves undefined).
fn bitwise_extract(value: u64, length: u8, index: u8) -> u64 {
    let (length, index) = length_and_index(length, index);
    (0..length)
        .filter(|i| index + i < 64)
        .fold(0, |out, i| out | (value >> (index + i) & 1) << i)
}

/// INSERTQ's low quadword, one bit at a time: bit `index + i` of `dest`
/// becomes bit `i` of `source`, for `i` below the length, where it exists.
fn bitwise_insert(dest: u64, source: u64, length: u8, index: u8) -> u64 {
    let (length, index) = length_and_index(length, index);
    (0..length).filter(|i| index + i < 64).fold(dest, |out, i| {
        let at = index + i;
        out & !(1 << at) | (source >> i & 1) << at
    })
}

#[test]
fn every_length_and_index_byte_against_the_bitwise_reference() {
    let mut values = Values::new(0xb17f);
    for _ in 0..6 {
        let (dest, source, upper) = (values.next(), values.next(), values.next());
        for length in 0..=255u8 {
            for index in 0..=255u8 {
                let range = BitRange::of(length, index);
                let want = [bitwise_extract(dest, length, index), 0];
                assert_eq!(extract([dest, upper], range), want, "extract {length} {index}");
                let want = [bitwise_insert(dest, source, length, index), 0];
                assert_eq!(insert([dest, upper], [source, upper], range), want, "insert {length} {index}");
            }
        }
    }
}

#[test]
fn the_manuals_examples() {
    // APM4 p. 172: "if the bit index is 32 (20h) and the field length is
    // 16 (10h), then the result in the destination register will be source
    // [47:32] in bits 15:0, with zeros in bits 63:16". APM4 p. 195: "the
    // result in the destination register will be source operand[15:0] in
    // bits 47:32. Bits 63:48 and bits 31:0 are not modified."
    let mut values = Values::new(0xe8a3);
    let range = BitRange::of(16, 32);
    for _ in 0..64 {
        let (a, b) = (values.next(), values.next());
        let [low, _] = extract([a, b], range);
        assert_eq!((low & 0xffff, low >> 16), (a >> 32 & 0xffff, 0));
        let [low, _] = insert([a, 0], [b, 0], range);
        assert_eq!(low >> 32 & 0xffff, b & 0xffff);
        assert_eq!((low >> 48, low as u32), (a >> 48, a as u32));
    }
    // "If the length field is 0 and the bit index is 0, bits 63:0 of the
    // source are extracted" (p. 172) / "inserted" (p. 195).
    let (a, b) = (values.next(), values.next());
    assert_eq!(extract([a, b], BitRange::of(0, 0))[0], a);
    assert_eq!(insert([a, 0], [b, 0], BitRange::of(0, 0))[0], b);
}

#[test]
fn ranges_held_in_registers() {
    // EXTRQ's register form: length in source bits 5:0, index in 13:8; INSERTQ's:
    // bits 69:64 and 77:72 (APM4 pp. 172, 195). The other bits are ignored.
    let mut values = Values::new(0x77aa);
    for _ in 0..256 {
        let (low, high) = (values.next(), values.next());
        let (length, index) = (low as u8, (low >> 8) as u8);
        assert_eq!(extract_range_held_in([low, high]), BitRange::of(length, index));
        let (length, index) = (high as u8, (high >> 8) as u8);
        assert_eq!(insert_range_held_in([low, high]), BitRange::of(length, index));
    }
}

/// The CPU's own `extrq xmm0, xmm1` (66 0F 79, ModRM 11 000 001) or
/// `insertq xmm0, xmm1` (F2 0F 79, the same ModRM); returns xmm0.
pub(super) fn cpu(insert: bool, xmm0: Xmm, xmm1: Xmm) -> Xmm {
    use std::arch::x86_64::{__m128i, _mm_set_epi64x};
    assert!(std::arch::is_x86_feature_detected!("sse4a"));
    // SAFETY: SSE2 is part of x86-64 and SSE4A is present (asserted); the
    // instruction touches the two registers named.
    unsafe {
        let mut a = _mm_set_epi64x(xmm0[1] as i64, xmm0[0] as i64);
        let b = _mm_set_epi64x(xmm1[1] as i64, xmm1[0] as i64);
        if insert {
            std::arch::asm!(".byte 0xf2, 0x0f, 0x79, 0xc1", inout("xmm0") a, in("xmm1") b,
                options(pure, nomem, nostack));
        } else {
            std::arch::asm!(".byte 0x66, 0x0f, 0x79, 0xc1", inout("xmm0") a, in("xmm1") b,
                options(pure, nomem, nostack));
        }
        std::mem::transmute::<__m128i, Xmm>(a)
    }
}

/// Whether APM4 defines the result for this length and index.
fn defined(length: u8, index: u8) -> bool {
    let (width, index) = length_and_index(length, index);
    if length & 0x3f == 0 { index == 0 } else { width + index <= 64 }
}

/// Every length and index APM4 does (or does not) define, against the CPU
/// where it has SSE4A, else against the bitwise reference.
fn against_the_cpu_or_the_reference(where_defined: bool) {
    let sse4a = std::arch::is_x86_feature_detected!("sse4a");
    let mut values = Values::new(0x5a5a);
    for _ in 0..4 {
        let (dest, source, upper) = (values.next(), values.next(), values.next());
        for length in 0..64u8 {
            for index in (0..64u8).filter(|&i| defined(length, i) == where_defined) {
                let fields = u64::from(length) | u64::from(index) << 8;
                let range = BitRange::of(length, index);
                let ours = (extract([dest, upper], range), insert([dest, upper], [source, upper], range));
                let theirs = if sse4a {
                    (cpu(false, [dest, upper], [fields, upper]), cpu(true, [dest, upper], [source, fields]))
                } else {
                    let e = [bitwise_extract(dest, length, index), 0];
                    (e, [bitwise_insert(dest, source, length, index), 0])
                };
                if where_defined {
                    // The upper quadword is undefined everywhere: low only.
                    assert_eq!((ours.0[0], ours.1[0]), (theirs.0[0], theirs.1[0]), "{length} {index}");
                } else {
                    assert_eq!(ours, theirs, "{length} {index}");
                }
            }
        }
    }
}

/// On a CPU with SSE4A, the CPU's results; elsewhere the reference's.
#[test]
fn defined_results_match_the_cpu_or_the_reference() {
    against_the_cpu_or_the_reference(true);
}

/// The results APM4 leaves undefined and the upper quadword, against the
/// CPU at hand: they matched the AMD Family 1Ah Model 44h CPU this module
/// was written on; another family may differ, so this is run by hand.
#[test]
#[ignore = "compares results the manual leaves undefined with this CPU; run by hand on an AMD CPU"]
fn undefined_results_match_the_cpu_or_the_reference() {
    against_the_cpu_or_the_reference(false);
}
