//! The host's slot table by view-heap slot: the element strides programs
//! read for buffers they pick from the heap at an index they compute. A
//! child of `lower` so it shares the emitter's state, apart to keep
//! `lower.rs` under the policy's file-size cap.

use super::super::bitcode::{Attr, Bin, Tag, V, pred};
use super::{Lower, Why};
use crate::native::RootType;
use crate::resource::Source;

impl Lower<'_> {
    /// The bytes per element of the buffer view `src` names, if the program
    /// picks it from the view heap at an index it computes (root type
    /// `ResourceHeap`): read from the host's table by heap slot in the
    /// current block.
    pub(super) fn slot_stride(&mut self, src: &Source) -> Result<Option<V>, Why> {
        let Source::Loaded { load, first } = src else {
            return Ok(None);
        };
        let heap = self
            .scalar
            .get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap);
        if !heap {
            return Ok(None);
        }
        let key = (self.cur, *load, *first);
        if let Some(&v) = self.strides.get(&key) {
            return Ok(Some(v));
        }
        let c = self.cur;
        let slot = self.heap_index(*load, *first, false)?;
        let slot = self.s.stride_slot(c, slot);
        // Inside the table whatever the index (lanes that do not run
        // compute one too): a root view has no bounds of its own.
        let last = self.s.m.i32(super::super::shader::SLOT_STRIDES_COUNT - 1);
        let i32t = self.s.t.i32;
        let slot = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        self.s.m.tag(slot, Tag::Clamp);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, slot, two);
        let h = self.s.root_srv_handle(super::super::shader::SLOT_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, byte, iu, mask, align],
        );
        let f = self.s.m.extract(c, r, 0);
        let mut v = self.as_int(f);
        if super::super::d16_integer_views_on() {
            // The host writes a typed view's format kind beside the stride
            // (`VIEW_INTEGER`, `VIEW_SIGNED`) only with the switch on.
            let bits = self.s.m.i32(super::super::shader::VIEW_STRIDE_BITS);
            v = self.s.m.bin(c, Bin::And, v, bits);
        }
        self.strides.insert(key, v);
        Ok(Some(v))
    }

    /// With `VICTORY_D16_INTEGER_VIEWS=1`: whether the typed buffer view
    /// behind the heap handle `h` has an integer format, and whether a
    /// signed one, from the slot table's kind bits
    /// ([`super::super::shader::VIEW_INTEGER`], `VIEW_SIGNED`), read at the
    /// heap index the handle was made from after the host heap's clamp
    /// (`Shader::view_slots`; under the slot map the host slot): whatever
    /// picked the descriptor (a load, an `s_cselect`, control flow, each lane
    /// its own), the access reads the kind of the view it reads. A handle
    /// that is no heap handle is refused.
    pub(super) fn slot_integer(&mut self, h: V) -> Result<(V, V), Why> {
        const WHY: Why = "a 16-bit typed access of a view outside the heap: its format kind is not known";
        // Read at each access, in its block: a handle made in one block
        // may be used in others it dominates, a word read in one of those
        // would not dominate the next.
        let &slot = self.s.view_slots.get(&h).ok_or(WHY)?;
        let word = self.slot_read(slot);
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let [int, signed] = [super::super::shader::VIEW_INTEGER, super::super::shader::VIEW_SIGNED].map(|bit| {
            let bit = self.s.m.i32(bit);
            let x = self.s.m.bin(c, Bin::And, word, bit);
            self.s.m.cmp(c, pred::NE, x, zero)
        });
        Ok((int, signed))
    }

    /// Dword `slot` of the slot table, read in the current block: the index
    /// clamped inside the table (a root view has no bounds of its own), then
    /// a raw read.
    fn slot_read(&mut self, slot: V) -> V {
        let c = self.cur;
        let last = self.s.m.i32(super::super::shader::SLOT_STRIDES_COUNT - 1);
        let i32t = self.s.t.i32;
        let slot = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        self.s.m.tag(slot, Tag::Clamp);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, slot, two);
        let h = self.s.root_srv_handle(super::super::shader::SLOT_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, byte, iu, mask, align]);
        let f = self.s.m.extract(c, r, 0);
        self.as_int(f)
    }
}
