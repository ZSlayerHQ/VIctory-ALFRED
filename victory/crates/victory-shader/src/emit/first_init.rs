//! The thread group's set-up of its shared memory by its first thread
//! ([`Lower::first_thread_init`]), a child module of the lowering so it shares the emitter's
//! state (moved out of `lower.rs` unchanged, which had grown past the repository's file-size
//! limit).

use super::{Attr, EXEC, Fields, Lower, Operand, Reg, StageIo, V, Value, flags, pred};

impl Lower<'_> {
    /// An LDS write (instruction `i` of block `b`) that is the thread group's set-up of its
    /// shared memory: constants to a constant address, under `v_cmpx_eq_u32 exec, 0, <the
    /// thread's place in its group>`, so by the group's first thread alone. Gives that
    /// thread's test and (byte address, constant) per dword written.
    ///
    /// The console runs it only if the first thread's lane still runs there: a group whose
    /// first thread has left (its pixel is not water, say) starts from what the group before
    /// it left in LDS, which on the console is the neighbouring tile's sane values and on a
    /// PC card is anything. Seen in `water_shading:CS_TransmissionVolumetrics_Tiled`
    /// (on an RTX 4070 SUPER): it keeps the tile's least and
    /// greatest depth in LDS, set to "none yet" by thread (0,0) after the lanes off the water
    /// have left; of 124 tiles whose first pixel was not water 71 wrote nothing (dark slabs
    /// on the water along the horizon and beside rocks, plants and hulls), of 3,250 others
    /// none. Here the first thread always sets it up (`VICTORY_LDS_FIRST_INIT=0`: as the
    /// console, only if its lane runs): the tile then works from its own depths alone.
    pub(super) fn first_thread_init(
        &mut self,
        b: usize,
        i: usize,
        addr: Option<&Operand>,
        data: &[Operand],
        elements: &[(u32, u32)],
    ) -> Option<(V, Vec<(u32, u32)>)> {
        if std::env::var("VICTORY_LDS_FIRST_INIT").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        let StageIo::Cs(cs) = self.x.native.io else { return None };
        let p = self.x.p;
        let insts = &p.blocks[b].insts;
        // the compare that left EXEC as it is at the write
        let j = (0..i).rev().find(|&j| insts[j].write(Reg::S(EXEC)).is_some())?;
        let Fields::Valu(cmp) = &insts[j].fields else { return None };
        if insts[j].name() != "v_cmpx_eq_u32" {
            return None;
        }
        let place = match cmp.src.as_slice() {
            [Operand::Const(128), Operand::Vector { reg, .. }] | [Operand::Vector { reg, .. }, Operand::Const(128)] => {
                insts[j].read(Reg::V(*reg))?
            }
            _ => return None,
        };
        // ... of the thread's place: its x, or x | y (| z), as the program entered with them
        let input = |v: Value| p.inputs.iter().find(|i| i.1 == v).and_then(|i| match i.0 {
            Reg::V(c @ 0..=2) => Some(c),
            _ => None,
        });
        let mut parts: Vec<u8> = Vec::new();
        let mut open = vec![place];
        while let Some(v) = open.pop() {
            if let Some(c) = input(v) {
                parts.push(c);
                continue;
            }
            let made = p.blocks.iter().flat_map(|blk| &blk.insts).find(|k| k.writes.iter().any(|w| w.1 == v))?;
            let Fields::Valu(or) = &made.fields else { return None };
            if made.name() != "v_or_b32" || or.sdwa.is_some() || or.dpp.is_some() {
                return None;
            }
            for op in &or.src {
                let Operand::Vector { reg, .. } = op else { return None };
                open.push(made.read(Reg::V(*reg))?);
            }
            if parts.len() + open.len() > 3 {
                return None;
            }
        }
        // (every dimension of the group that has more than one thread is tested)
        if (0..3u8).any(|c| cs.threads[usize::from(c)] > 1 && !parts.contains(&c)) {
            return None;
        }
        // the address and what is written: constants moved in since the compare
        let constant = |op: &Operand| -> Option<u32> {
            let Operand::Vector { reg, .. } = *op else { return None };
            let k = (j + 1..i).rev().find(|&k| insts[k].write(Reg::V(reg)).is_some())?;
            let Fields::Valu(mov) = &insts[k].fields else { return None };
            if insts[k].name() != "v_mov_b32" || mov.sdwa.is_some() || mov.dpp.is_some() {
                return None;
            }
            match mov.src.as_slice() {
                [Operand::Const(c)] => crate::resource::inline(*c),
                [Operand::Literal(x)] => Some(*x),
                _ => None,
            }
        };
        let at = constant(addr?)?;
        let mut out = Vec::new();
        for (e, &(off, w)) in elements.iter().enumerate() {
            if w != 1 {
                return None;
            }
            let bytes = at.checked_add(off)?;
            if bytes % 4 != 0 || bytes / 4 >= cs.lds_bytes() / 4 {
                return None;
            }
            out.push((bytes, constant(data.get(e)?)?));
        }
        // the first thread: every coordinate is 0, so its index in the group is
        let flat = if self.console_lanes {
            self.s.flags |= flags::WAVE_OPS;
            self.console_thread()
        } else {
            let (e, i32t) = (self.s.entry, self.s.t.i32);
            self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[])
        };
        let (c, zero) = (self.cur, self.s.m.i32(0));
        Some((self.s.m.cmp(c, pred::EQ, flat, zero), out))
    }
}
