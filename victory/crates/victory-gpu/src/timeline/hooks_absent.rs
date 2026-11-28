//! The timeline's wrappers of the functions threads wait with: absent from
//! this tree.
//!
//! The private build has a module here that, with `VICTORY_TIMELINE` asked
//! for, binds the title's waits and wake-ups (and those of our own import
//! table) to wrappers that record each wait, who ended it and where. The
//! public tree leaves it out. What is here only gives the callers what they
//! need, and does nothing: every import stays bound to the host's function,
//! so a timeline of this build records no waits or wake-ups.

/// The address an import of the host's `_name` is bound to: `address`, the
/// host's own function, unchanged.
pub fn import(_name: &str, address: usize) -> usize {
    address
}
