//! Two more shapes of the branches [`super::without_early_outs`] and [`super::without_skips`]
//! leave out, two general control-flow patterns of compiled programs, and the switch that
//! lets those two forms take them (`VICTORY_WAVE_JOINS=1`; off by default).
//!
//! - An early-out spelled as a far jump: a program too long for a branch to reach its end
//!   has `s_cbranch_execnz <body>` over `s_getpc_b64`, an offset added, `s_setpc_b64` (which
//!   `build` resolves to a jump) to a block that only ends the program, where a shorter one
//!   has `s_cbranch_execz <end>`. Left out, the lanes that would have left walk the body with
//!   EXEC clear, as for the short spelling.
//! - A skip whose scalars come back through a join inside it: a scalar phi all of whose
//!   arguments are, through copies and such phis, one value (a join of branches that left it
//!   alone) is that value, so the skip's join sees no change (`deep` only).
//!
//! Off by default because the forms' premise is not proved for these shapes either: a body
//! or region walked with EXEC clear could put EXEC back from a mask saved before the branch
//! and store (the review of job/shadow-mask, finding 1, which holds for the short spellings
//! too), and the programs these shapes bring to the exchanged form take group barriers in
//! sequences one console wave of a group can leave early (finding 2).

use std::collections::HashMap;

use super::{Exit, Phi, Program, Reg, Value};

#[cfg(test)]
thread_local! {
    /// A test's [`on`], for its thread only.
    pub(crate) static TEST_JOINS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `VICTORY_WAVE_JOINS=1`: the two forms take the shapes of this module.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_JOINS.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAVE_JOINS").is_some_and(|v| v == "1"))
}

/// Whether block `b` only ends the program: program-control words alone. With
/// `VICTORY_WAVE_GUARDS=1` ([`crate::emit::wave_guards_on`]) none of them may be one that
/// another wave or the hardware sees on the way out (a barrier the leaving wave waits at, a
/// message it sends, a trap); without, any program-control word passes, as on `world`.
pub(super) fn ends_program(p: &Program, b: usize) -> bool {
    let seen = |name: &str| matches!(name, "s_barrier" | "s_sendmsg" | "s_sendmsghalt" | "s_trap");
    let guards = crate::emit::wave_guards_on();
    p.blocks[b].exit == Exit::End
        && p.blocks[b]
            .insts
            .iter()
            .all(|i| matches!(i.fields, crate::gfx10::Fields::Sopp { .. }) && !(guards && seen(i.name())))
}

/// Whether block `b` is the far jump of a console branch whose target is too far for the
/// branch's 16-bit offset (`s_getpc_b64`, an address added to it, `s_setpc_b64`) to a block
/// that only ends the program. Code that does any work on the way stays what it is.
fn far_jump_to_end(p: &Program, b: usize) -> bool {
    let block = &p.blocks[b];
    matches!(block.exit, Exit::Jump(target) if ends_program(p, target))
        && block
            .insts
            .iter()
            .all(|i| matches!(i.name(), "s_getpc_b64" | "s_add_u32" | "s_addc_u32" | "s_setpc_b64"))
}

/// Where the lanes that stay go when block `b` ends in an early-out: `s_cbranch_execz` to a
/// block that only ends the program, or (with [`on`]) `s_cbranch_execnz` whose other way is a
/// far jump to such a block. The branch is then left out and the code goes on there.
pub(super) fn early_out_continuation(p: &Program, b: usize) -> Option<usize> {
    let Exit::Branch { taken, not_taken } = p.blocks[b].exit else {
        return None;
    };
    match p.blocks[b].insts.last()?.name() {
        "s_cbranch_execz" if ends_program(p, taken) => Some(not_taken),
        "s_cbranch_execnz" if on() && far_jump_to_end(p, not_taken) => Some(taken),
        _ => None,
    }
}

/// The value `v` is a copy of, followed to the end of `copy`'s chains.
pub(super) fn follow(copy: &HashMap<Value, Value>, mut v: Value) -> Value {
    while let Some(&c) = copy.get(&v) {
        v = c;
    }
    v
}

/// The one value every argument of the scalar `phi` comes to (the phi's own value left
/// aside, as a loop's back edge brings it), if there is one.
fn single_source(copy: &HashMap<Value, Value>, phi: &Phi) -> Option<Value> {
    let mut one = None;
    for x in phi.args.iter().map(|&a| follow(copy, a)).filter(|&x| x != phi.value) {
        if *one.get_or_insert(x) != x {
            return None;
        }
    }
    one
}

/// With [`on`], adds to `copy` the scalar phis that pass one value through (a join of
/// branches none of which changed it, one after the other, as the console's compiler leaves
/// them): such a phi is a copy of that value. A phi that merges a real change is not.
/// Repeats until no phi is added, for a phi may pass through the ones before it.
pub(super) fn copy_through_phis(p: &Program, copy: &mut HashMap<Value, Value>) {
    if !on() {
        return;
    }
    let mut added = true;
    while added {
        added = false;
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            if copy.contains_key(&phi.value) || !matches!(phi.reg, Reg::S(_) | Reg::Scc) {
                continue;
            }
            if let Some(one) = single_source(copy, phi) {
                copy.insert(phi.value, one);
                added = true;
            }
        }
    }
}
