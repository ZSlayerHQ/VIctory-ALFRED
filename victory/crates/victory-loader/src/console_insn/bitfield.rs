//! SSE4A's EXTRQ and INSERTQ on an XMM register (APM4 pp. 172, 195),
//! written around the range of bits a field covers.
//!
//! APM4 gives a field by its length and the index of its lowest bit, "each
//! six bits in length; other bits of the field are ignored", a length of 0
//! meaning 64. It leaves undefined a field that runs past bit 63 (index +
//! length over 64, so a length of 0 with any index but 0), and the upper
//! quadword of the destination. Here a [`BitRange`] is clipped at bit 64:
//! EXTRQ takes only the bits that exist, INSERTQ places only the source
//! bits that land below bit 64; and the upper quadword is zeroed.
//!
//! Evidence for those choices: on an AMD Family 1Ah Model 44h CPU (not the
//! console's Family 17h), every length and index, run with the register
//! forms against generated values, gave exactly these results, including
//! the upper quadword zeroed by both instructions (the test
//! `undefined_results_match_the_cpu_or_the_reference`, run by hand there).

/// The bits a field covers in a quadword: `low..high`, half-open, never
/// empty, `high` at most 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BitRange {
    pub(super) low: u32,
    pub(super) high: u32,
}

impl BitRange {
    /// The range an encoded length and index select, clipped at bit 64.
    pub(super) fn of(length: u8, index: u8) -> Self {
        let low = u32::from(index & 0x3f);
        let width = match length & 0x3f {
            0 => 64,
            n => u32::from(n),
        };
        Self { low, high: (low + width).min(64) }
    }

    /// Ones in the range's bits.
    fn mask(self) -> u64 {
        let width = self.high - self.low;
        (u64::MAX >> (64 - width)) << self.low
    }

    /// The range's bits of `value`, moved down to bit 0.
    fn take(self, value: u64) -> u64 {
        (value & self.mask()) >> self.low
    }

    /// `into` with the range's bits replaced by the low bits of `bits`.
    fn put(self, into: u64, bits: u64) -> u64 {
        into & !self.mask() | (bits << self.low) & self.mask()
    }
}

/// An XMM register: its low and high quadwords.
pub(super) type Xmm = [u64; 2];

/// EXTRQ's result in its destination `dest`: the range's bits of the low
/// quadword moved down to bit 0, the rest of the low quadword cleared.
pub(super) fn extract(dest: Xmm, range: BitRange) -> Xmm {
    [range.take(dest[0]), 0]
}

/// INSERTQ's result in its destination `dest`: the low bits of `source`'s
/// low quadword in the range, the rest of the low quadword kept.
pub(super) fn insert(dest: Xmm, source: Xmm, range: BitRange) -> Xmm {
    [range.put(dest[0], source[0]), 0]
}

/// The range EXTRQ's register form reads from its source: length in bits
/// 5:0, index in bits 13:8.
pub(super) fn extract_range_held_in(source: Xmm) -> BitRange {
    BitRange::of(source[0] as u8, (source[0] >> 8) as u8)
}

/// The range INSERTQ's register form reads from its source: length in
/// bits 69:64, index in bits 77:72.
pub(super) fn insert_range_held_in(source: Xmm) -> BitRange {
    BitRange::of(source[1] as u8, (source[1] >> 8) as u8)
}
