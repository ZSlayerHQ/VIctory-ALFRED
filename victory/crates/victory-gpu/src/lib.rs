//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod bind_trace;
/// The frame budget: one scope type for the phase a thread is in.
mod budget;
#[cfg(windows)]
mod build_label;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
/// `VICTORY_AVOID_CPUS`: title threads' affinity and ideal processors.
pub mod cpu;
#[cfg(all(windows, test))]
mod crash_report;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod digest;
#[cfg(windows)]
mod draw;
#[cfg(all(windows, test))]
mod draw_map_tests;
pub mod dxgi;
/// Debug events on our own lists, for outside GPU tools.
#[cfg(windows)]
mod events;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
mod kernelstat;
#[cfg(windows)]
mod ledger;
mod live;
pub mod list;
#[cfg(all(windows, feature = "diagnostics"))]
mod null_trace;
pub mod object;
#[cfg(all(windows, feature = "diagnostics"))]
mod passes;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
#[cfg(windows)]
mod receipts;
pub mod resource;
pub mod root;
#[cfg(all(windows, feature = "diagnostics"))]
mod runs;
/// The user-mode copy of title memory and its fault redirect (S0: its fixtures alone call it).
#[cfg(windows)]
mod safecopy;
#[cfg(windows)]
mod sampler;
/// The timed key sequence parser shared by `VICTORY_KEYS` and the live
/// `LIVE_KEYS` switch.
pub mod scripted_keys;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memclass;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod storestat;
#[cfg(all(windows, feature = "diagnostics"))]
pub mod stalls;
#[cfg(all(windows, feature = "diagnostics"))]
mod stamps;
#[cfg(windows)]
mod tiles;
#[cfg(all(test, windows))]
mod tiled_probe;
pub mod signature;
#[cfg(windows)]
mod switches;
#[cfg(windows)]
mod texback;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod texture_pack;
#[cfg(windows)]
mod watch;

/// A call of the title's into `victory-hle` is under way on this thread while this lives
/// ([`hle_call`]): the frame budget's `hle_calls`.
pub use budget::Scope as CallScope;

/// For `victory-hle`, at the top of every export and object method the title calls: the
/// call's time on this thread is the frame budget's `hle_calls`, not the title's own. Nothing
/// in a build without diagnostics, and nothing when the thread is already inside a call of
/// ours.
#[inline]
pub fn hle_call() -> CallScope {
    budget::entry(budget::Phase::HleCalls)
}

/// The live switches' (`VICTORY_LIVE`) final segment, at the title's
/// shutdown: one `live segment` line for the stretch since the last change.
/// Does nothing when the switch is unset.
pub fn live_finish() {
    live::finish();
}

/// The live key sequence playing now (`LIVE_KEYS`), if one is; the input
/// side evaluates it through the `VICTORY_KEYS` path.
pub use live::keys as live_keys;
/// Logs a live key sequence's one done line and stops it; `id` names the
/// run the input side evaluated.
pub use live::keys_done as live_keys_done;
pub use live::KeysRun;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
/// The title's `VirtualQueryEx`, counted with the runtime's other memory calls.
#[cfg(windows)]
pub use kernelstat::title_virtual_query_ex;
#[cfg(all(windows, feature = "diagnostics"))]
pub use memclass::{
    TitleAllocation, memory_census_allocated, memory_census_exit, memory_census_physical,
    memory_census_storage,
};
#[cfg(windows)]
pub use watch::{
    KernelWrite, guard_ledger, kernel_writing, title_memory_reading, title_memory_remapped,
    title_memory_remapping, title_memory_written, title_unmap_view, title_virtual_alloc,
    title_virtual_free, title_virtual_protect, title_virtual_query, writing as kernel_will_write,
};
#[cfg(all(windows, feature = "diagnostics"))]
pub use resource::made::storage_destination;

/// Whether the texture upload recheck (`VICTORY_TEXTURE_RECHECK`) is on: for
/// `victory-hle`, which reports where a DirectStorage request's bytes are
/// being written while it runs.
#[cfg(windows)]
pub fn texture_recheck_on() -> bool {
    host::recheck::on()
}

/// A DirectStorage request's write into title memory `[start, start + size)`
/// is about to run: what the texture recheck compares a stale upload's
/// writers with. Inert with `VICTORY_TEXTURE_RECHECK` off.
#[cfg(windows)]
pub fn storage_write_begin(start: usize, size: usize) {
    host::recheck::storage_begin(start, size);
}

/// The write [`storage_write_begin`] announced has ended, its bytes there.
#[cfg(windows)]
pub fn storage_write_end(start: usize, size: usize) {
    host::recheck::storage_end(start, size);
}
