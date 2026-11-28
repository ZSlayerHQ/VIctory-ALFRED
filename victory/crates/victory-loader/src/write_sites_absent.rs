//! Changes to the title's code asked for at the start: absent from this tree.
//!
//! The private build has a module here that, before the title runs, changes
//! some places of its code as asked. Those places are the title's own, so
//! the public tree leaves it out. What is here only gives the caller what it
//! needs, and does nothing: the mapped image is run as it was loaded.

/// Changes asked for in the image `_name` mapped at `_base`, `_size` bytes
/// long: none here.
///
/// # Safety
///
/// None needed here; the caller keeps the private build's contract (the image
/// is mapped, and nothing of it has run yet).
pub unsafe fn apply_asked(_name: &str, _base: usize, _size: usize) {}
