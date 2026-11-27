//! The title-slot to host-slot map (design `jobs/2026-11-25-slot-lookup`,
//! behind `VICTORY_SLOT_MAP=1`): its layout, which the translated shader
//! reads and the host (`victory-gpu`'s `descriptor::slot_map`) writes.
//!
//! A translated shader forms a view index with the console's own byte
//! arithmetic: the **title slot**, a position in the title's heap. It never
//! names a host heap slot by it. It reads the map for one:
//!
//! - the root SRV `t<MAP_DIR>` in `ROOT_CBV_SPACE` is the directory, one
//!   dword per [`CHUNK_SLOTS`] title slots: the number of the chunk of the
//!   pool that holds their entries ([`DIR_ENTRIES`] dwords);
//! - the root SRV `t<MAP_POOL>` there is the pool, [`CHUNK_SLOTS`] dwords a
//!   chunk: the entry of title slot `t` is dword `t % CHUNK_SLOTS` of chunk
//!   `directory[t / CHUNK_SLOTS]`;
//! - an entry holds the host slot in its low [`HOST_BITS`] bits and above
//!   them the [`class`] of the descriptor made there (0: no descriptor);
//! - an entry whose class is not the one the access declares (a miss, or a
//!   view of another kind) reads the typed null of the declared class, host
//!   slot [`NULL_BASE`] plus the class.
//!
//! [`lookup`] is that arithmetic on the CPU, for the host's tests and the
//! translator's fixtures; [`Shader::title_handle`] emits it. Samplers have no
//! map: the host mirrors the title's sampler heap slot for slot. Views the
//! host makes itself (the memory copies) are host slots, and take
//! `Shader::heap_handle` directly.

use super::bitcode::{Attr, Bin, Cast, Tag, V, pred};
use super::shader::Shader;

/// `VICTORY_SLOT_MAP=1`: every title view index a shader forms reads the
/// map for its host slot ([`Shader::title_handle`]). Off by default; the
/// translation differs, so the shader cache keys on it like every switch the
/// translator reads.
pub fn on() -> bool {
    #[cfg(test)]
    if TEST_ON.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SLOT_MAP").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's `VICTORY_SLOT_MAP=1`, for its thread only.
    pub(super) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: the map lookups [`Shader::title_handle`] emitted on this thread.
    pub(super) static TEST_LOOKUPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The root SRV `t<MAP_DIR>` in `ROOT_CBV_SPACE`: the directory.
pub const MAP_DIR: u8 = 252;
/// The root SRV `t<MAP_POOL>` in `ROOT_CBV_SPACE`: the chunk pool.
pub const MAP_POOL: u8 = 251;
/// Title slots the map can name. A title slot is clamped to `MAP_SLOTS - 1`,
/// and that last entry is never mapped: an index that wraps below the heap's
/// start (a huge unsigned number) reads a miss.
pub const MAP_SLOTS: u32 = 1 << 21;
/// Title slots per chunk, as a shift.
pub const CHUNK_SHIFT: u32 = 11;
/// Title slots per chunk: entries per chunk of the pool.
pub const CHUNK_SLOTS: u32 = 1 << CHUNK_SHIFT;
/// Dwords of the directory.
pub const DIR_ENTRIES: u32 = MAP_SLOTS >> CHUNK_SHIFT;
/// Bits of an entry that hold the host slot (a host heap holds at most a
/// million views).
pub const HOST_BITS: u32 = 20;
/// Bits of an entry above [`HOST_BITS`] that hold the class.
pub const CLASS_BITS: u32 = 6;
/// The first typed null: the null of class `c` is host slot `NULL_BASE + c`.
pub const NULL_BASE: u32 = 0;
/// Classes (with 0, the miss), and so typed nulls.
pub const CLASSES: u32 = 1 << CLASS_BITS;
/// The UAV bit of a handle's first property word (`annotateHandle`).
pub const UAV: u32 = 0x1000;

/// DXIL resource kinds the class keeps apart (the low byte of a handle's
/// first property word).
const RAW_BUFFER: u32 = 11;
const STRUCTURED_BUFFER: u32 = 12;

/// The class of a view a handle with first property word `props0` may read:
/// the DXIL resource kind, plus 32 for unordered access. Raw and structured
/// buffers are one class: the host makes buffer views in the heap raw, and a
/// structured access reads its element by the stride it is told.
pub const fn class(props0: u32) -> u32 {
    let kind = match props0 & 0xff {
        STRUCTURED_BUFFER => RAW_BUFFER,
        kind => kind,
    };
    let unordered = if props0 & UAV != 0 { 32 } else { 0 };
    (kind & 31) | unordered
}

/// The entry naming host slot `host`, made a view of class `class`.
pub const fn entry(host: u32, class: u32) -> u32 {
    (host & ((1 << HOST_BITS) - 1)) | (class << HOST_BITS)
}

/// The host slot an access of class `want` reads for title slot `t`, given
/// the directory and the pool as the shader sees them: the shader's own
/// arithmetic. Reads outside either buffer read 0, as raw loads past a
/// view's end do on the card.
pub fn lookup(directory: &[u32], pool: &[u32], t: u32, want: u32) -> u32 {
    let t = t.min(MAP_SLOTS - 1);
    let chunk = directory.get((t >> CHUNK_SHIFT) as usize).copied().unwrap_or(0);
    let at = (chunk << CHUNK_SHIFT | (t & (CHUNK_SLOTS - 1))) as usize;
    let e = pool.get(at).copied().unwrap_or(0);
    if e >> HOST_BITS == want {
        e & ((1 << HOST_BITS) - 1)
    } else {
        NULL_BASE + want
    }
}

impl Shader {
    /// A handle for the descriptor at title slot `t` (a view or sampler
    /// index the console's arithmetic formed), annotated with `props`;
    /// `nonuniform` when `t` may differ between lanes. With the map on, a
    /// view reads the map for its host slot ([`lookup`]: two dependent raw
    /// loads, uniform where `t` is): a miss or a view of another class reads
    /// the typed null of the access's class. Samplers, and views with the map
    /// off, index the heap at `t` itself.
    pub fn title_handle(&mut self, block: usize, t: V, sampler: bool, nonuniform: bool, props: (u32, u32)) -> V {
        if sampler || !on() {
            return self.heap_handle(block, t, sampler, nonuniform, props);
        }
        #[cfg(test)]
        TEST_LOOKUPS.set(TEST_LOOKUPS.get() + 1);
        let want = class(props.0);
        let e = self.map_entry(block, t);
        let bits = self.m.i32(HOST_BITS);
        let got = self.m.bin(block, Bin::LShr, e, bits);
        let w = self.m.i32(want);
        let ok = self.m.cmp(block, pred::EQ, got, w);
        let mask = self.m.i32((1 << HOST_BITS) - 1);
        let host = self.m.bin(block, Bin::And, e, mask);
        let null = self.m.i32(NULL_BASE + want);
        let h = self.m.select(block, ok, host, null);
        // (`heap_handle` keeps the host heap's limit on it, as before.)
        self.heap_handle(block, h, false, nonuniform, props)
    }

    /// The slot of the host's table of strides (`SLOT_STRIDES`) for the view
    /// at title slot `t`: with the map on, the host slot its entry names
    /// whatever the class (a miss: null slot 0, stride 0). The access then
    /// takes its handle through [`Shader::title_handle`] from the same map.
    pub fn stride_slot(&mut self, block: usize, t: V) -> V {
        if !on() {
            return t;
        }
        let e = self.map_entry(block, t);
        let mask = self.m.i32((1 << HOST_BITS) - 1);
        self.m.bin(block, Bin::And, e, mask)
    }

    /// The map's entry for title slot `t`: `t` clamped to the last entry
    /// (a miss), its chunk from the directory, the entry from that chunk.
    fn map_entry(&mut self, block: usize, t: V) -> V {
        let i32t = self.t.i32;
        let last = self.m.i32(MAP_SLOTS - 1);
        // (as `dx.op.binary` UMin, like the heap's own clamp)
        let t = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[t, last]);
        self.m.tag(t, Tag::Clamp);
        let shift = self.m.i32(CHUNK_SHIFT);
        let two = self.m.i32(2);
        let of = self.m.bin(block, Bin::LShr, t, shift);
        let at = self.m.bin(block, Bin::Shl, of, two);
        let directory = self.root_srv_handle(MAP_DIR);
        let chunk = self.raw_dword(block, directory, at);
        let low = self.m.i32(CHUNK_SLOTS - 1);
        let within = self.m.bin(block, Bin::And, t, low);
        let base = self.m.bin(block, Bin::Shl, chunk, shift);
        let slot = self.m.bin(block, Bin::Or, base, within);
        let at = self.m.bin(block, Bin::Shl, slot, two);
        let pool = self.root_srv_handle(MAP_POOL);
        self.raw_dword(block, pool, at)
    }

    /// The dword at byte `at` of the raw buffer `h`.
    pub(crate) fn raw_dword(&mut self, block: usize, h: V, at: V) -> V {
        let iu = self.m.undef(self.t.i32);
        let mask = self.m.i8(1);
        let align = self.m.i32(4);
        let ret = self.t.res_ret;
        let r = self.op(block, "rawBufferLoad.f32", Attr::ReadOnly, ret, 139, &[h, at, iu, mask, align]);
        let f = self.m.extract(block, r, 0);
        self.m.cast(block, Cast::Bitcast, f, self.t.i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_and_structured_are_one_class_and_unordered_access_another() {
        // Texture2D SRV, raw SRV, structured SRV, raw UAV, typed buffer UAV.
        assert_eq!(class(2), 2);
        assert_eq!(class(RAW_BUFFER), class(STRUCTURED_BUFFER | 4 << 8));
        assert_eq!(class(RAW_BUFFER | UAV), RAW_BUFFER | 32);
        assert_ne!(class(10 | UAV), class(10));
        // Every class leaves room for its typed null below the shared range.
        for kind in 0..=18 {
            assert!(class(kind | UAV) < CLASSES);
        }
    }

    #[test]
    fn a_lookup_reads_the_entry_of_its_class_and_the_typed_null_otherwise() {
        let mut directory = vec![0u32; DIR_ENTRIES as usize];
        // Chunk 0 of the pool is all misses; title chunk 1 is pool chunk 2.
        let mut pool = vec![0u32; 3 * CHUNK_SLOTS as usize];
        directory[1] = 2;
        let t = CHUNK_SLOTS + 5;
        pool[(2 * CHUNK_SLOTS + 5) as usize] = entry(70_000, 2);
        assert_eq!(lookup(&directory, &pool, t, 2), 70_000);
        // Another kind of access to the same slot, its neighbour (never
        // made), a slot in an unmapped chunk: typed nulls.
        assert_eq!(lookup(&directory, &pool, t, 34), NULL_BASE + 34);
        assert_eq!(lookup(&directory, &pool, t + 1, 2), NULL_BASE + 2);
        assert_eq!(lookup(&directory, &pool, 7, 2), NULL_BASE + 2);
        // An index that wrapped below the heap's start: the last entry.
        assert_eq!(lookup(&directory, &pool, (-1i32) as u32, 2), NULL_BASE + 2);
    }
}
