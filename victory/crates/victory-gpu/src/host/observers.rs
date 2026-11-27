//! The start-up line naming every diagnostic that costs frame time and is on (finding 5 of
//! the logging audit), so a log says what it measured under. No default changes here.

use super::{dred_on, timing};

/// Says the observers line once, at device creation: the debug layer, GPU-based validation,
/// DRED, GPU timing, the captures, the memory map walk, the counting allocator (in every
/// diagnostics build).
pub(crate) fn say(debug: bool, gpu_validation: bool) {
    let mut on: Vec<&str> = Vec::new();
    if debug {
        on.push("the debug layer");
    }
    if gpu_validation {
        on.push("GPU-based validation");
    }
    if dred_on() {
        on.push("DRED");
    }
    if timing() {
        on.push("GPU timing");
    }
    if crate::capture::world_on() {
        on.push("world captures");
    }
    if crate::capture::dump_on() {
        on.push("target dumps");
    }
    if crate::copystat::memory_map() {
        on.push("the memory map walk");
    }
    on.push("the counting allocator");
    tracing::info!("observers on: {}", on.join(", "));
}
