//! The title's own writes beside the guards: absent from this tree.
//!
//! The private build has a module here that binds some of the title's
//! imports (its copies of memory) to wrappers of ours and takes some of its
//! stores that fault on a guarded block, so that the block can stay guarded
//! (`VICTORY_COPY_BESIDE`, `VICTORY_STORE_BESIDE`). It is made from what the
//! title's code is, so the public tree leaves it out. What is here only gives
//! the callers what they need, and does nothing: an import stays bound to
//! the host's function, a copy of descriptors is a plain copy, a faulting
//! store is left to the guard's own handling, and every count is zero.

use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

/// How much may be put into a block beside its guard before the block is
/// opened instead (`crate::watch::put_beside`). Nothing in this tree puts
/// anything beside a guard outside the tests; the value is our own choice.
#[allow(dead_code, reason = "read by `crate::watch::put_beside`, which only the tests call in this tree")]
pub(crate) const MOST: u32 = 16;

/// The counts the frame log reads (`crate::copystat`): always zero here.
#[cfg(feature = "diagnostics")]
pub(crate) static DONE: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// The address an import of the host's `_name` is bound to: `address`, the
/// host's own function, unchanged.
pub fn import(_name: &str, address: usize) -> usize {
    address
}

/// A copy of `len` descriptor bytes from `src` to `dst`: a plain copy.
///
/// # Safety
///
/// Both ranges are `len` bytes the caller may read and write.
pub(crate) unsafe fn put(dst: *mut u8, src: *const u8, len: usize) {
    // SAFETY: by the caller.
    unsafe { std::ptr::copy(src, dst, len) };
}

/// Whether a store of the title's that faulted at `_at` was carried out here
/// (the guard then stays): never; the guard's own handler opens the block.
pub(crate) fn store(_context: &mut CONTEXT, _at: usize) -> bool {
    false
}

/// The busiest stores, for the frame log: none.
#[cfg(feature = "diagnostics")]
pub(crate) fn busiest() -> &'static str {
    "none (not in this build)"
}
