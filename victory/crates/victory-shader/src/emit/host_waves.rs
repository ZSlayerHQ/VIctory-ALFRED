//! What the host's device says of its waves, as an input of the translation.
//!
//! A program that fixes a wave size (`WaveSize(n)` in the emitted container) makes a
//! pipeline only on a device with waves of n lanes: n within the device's fewest and most
//! lanes, on a device with wave operations. `dxil.dll` passes any size and the host refuses
//! the pipeline when it is made, so the translation is told the device and refuses first.
//!
//! One snapshot ([`HostCaps`]) goes through a whole emission (`emit_for`): the selection of
//! the program's form, the lowering, the check of the form selected, and the shader cache's
//! key. The process keeps the device's answer ([`set_host_caps`]) for callers that pass
//! none; that answer is replaced whole, a failed query included, so that a second device
//! never translates for the first one's waves (the recheck of job/land-2e, Question 2).

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// A program whose selected form asks for a wave wider than the host's widest: the host
/// refuses that pipeline when it is made, which neither `dxil.dll` nor a corpus list shows.
pub(crate) const WAVE_TOO_WIDE: &str = "a wave wider than the host's widest wave";
/// Likewise for a wave narrower than the host's narrowest (waves of 32 on a device whose
/// waves all have 64 lanes).
pub(crate) const WAVE_TOO_NARROW: &str = "a wave narrower than the host's narrowest wave";
/// A fixed wave size on a device that reports no wave operations: its lane counts say
/// nothing of what it runs.
pub(crate) const NO_WAVE_OPS: &str = "a wave size on a host without wave operations";

/// A device's answer (`D3D12_FEATURE_DATA_D3D12_OPTIONS1`): the fewest and the most lanes
/// its waves have, and whether it has wave operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostWaves {
    /// `WaveLaneCountMin`.
    pub min: u32,
    /// `WaveLaneCountMax`.
    pub max: u32,
    /// `WaveOps`.
    pub wave_ops: bool,
}

/// The host a program is translated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HostCaps {
    /// No device has answered (an offline translation), or its query failed: no
    /// host-specific form is taken and no wave size can be checked.
    #[default]
    Unknown,
    /// The device's answer.
    Known(HostWaves),
}

/// What the check of a selected form's wave size found, where it did not refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveCheck {
    /// The program fixes no wave size: any device's waves do.
    AnyWave,
    /// It fixes this one, which the host has.
    HostHas(u32),
    /// It fixes this one and the host's waves are not known: handed over unchecked, and the
    /// host may still refuse the pipeline.
    Unchecked(u32),
}

impl HostCaps {
    /// A host with wave operations whose widest wave has `lanes` lanes and whose narrowest
    /// is not told (taken as the fewest D3D12 allows, 4, so nothing is refused for it); 0:
    /// not known. What `VICTORY_HOST_MAX_WAVE` says, and the tests' hosts.
    pub fn at_most(lanes: u32) -> Self {
        match lanes {
            0 => Self::Unknown,
            max => Self::Known(HostWaves { min: max.min(4), max, wave_ops: true }),
        }
    }

    /// The most lanes a wave of the host has (0: not known).
    pub fn max_wave(self) -> u32 {
        match self {
            Self::Unknown => 0,
            Self::Known(h) => h.max,
        }
    }

    /// Whether a program that fixes the wave size `wave` (`None`: none) makes a pipeline on
    /// this host: the reason where it is known not to.
    pub fn check(self, wave: Option<u32>) -> Result<WaveCheck, &'static str> {
        let Some(lanes) = wave else {
            return Ok(WaveCheck::AnyWave);
        };
        match self {
            Self::Unknown => Ok(WaveCheck::Unchecked(lanes)),
            Self::Known(h) if !h.wave_ops => Err(NO_WAVE_OPS),
            Self::Known(h) if lanes > h.max => Err(WAVE_TOO_WIDE),
            Self::Known(h) if lanes < h.min => Err(WAVE_TOO_NARROW),
            Self::Known(_) => Ok(WaveCheck::HostHas(lanes)),
        }
    }

    /// The host as part of the shader cache's key.
    pub(super) fn key(self, out: &mut Vec<u8>) {
        match self {
            Self::Unknown => out.push(0),
            Self::Known(h) => {
                out.push(1);
                out.extend_from_slice(&h.min.to_le_bytes());
                out.extend_from_slice(&h.max.to_le_bytes());
                out.push(u8::from(h.wave_ops));
            }
        }
    }
}

/// Where a device's answer is kept: replaced whole each time, never updated in part.
#[derive(Debug, Default)]
pub struct HostCell(Mutex<HostCaps>);

impl HostCell {
    /// Nothing known yet.
    pub const fn new() -> Self {
        Self(Mutex::new(HostCaps::Unknown))
    }

    /// The device's answer, or [`HostCaps::Unknown`] where it gave none: what was known of
    /// an earlier device is gone either way.
    pub fn set(&self, caps: HostCaps) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = caps;
    }

    pub fn get(&self) -> HostCaps {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The process's host: the device `victory-gpu` made last.
static PROCESS: HostCell = HostCell::new();

/// The process's host, for the code that asks a device ([`HostCell::set`]).
pub fn process_host() -> &'static HostCell {
    &PROCESS
}

/// Tells the translation the host: every later [`host_caps`] snapshot is `caps`.
pub fn set_host_caps(caps: HostCaps) {
    PROCESS.set(caps);
}

/// A snapshot of the process's host ([`set_host_caps`]; unknown until then).
pub fn host_caps() -> HostCaps {
    #[cfg(test)]
    if let Some(lanes) = super::lower::TEST_HOST_WAVE.get() {
        return HostCaps::at_most(lanes);
    }
    PROCESS.get()
}

thread_local! {
    /// The host of the emission under way on this thread ([`with_host`]).
    static EMITTING: Cell<Option<HostCaps>> = const { Cell::new(None) };
}

/// Runs `f`, an emission, with `host` as what the lowering sees ([`emitting_for`]): the
/// snapshot the caller took, whatever the process is told meanwhile.
pub(super) fn with_host<R>(host: HostCaps, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<HostCaps>);
    impl Drop for Restore {
        fn drop(&mut self) {
            EMITTING.set(self.0);
        }
    }
    let _restore = Restore(EMITTING.replace(Some(host)));
    f()
}

/// The host the lowering translates for: the emission's snapshot, else (a lowering run on
/// its own, as the tests do) the process's.
pub(super) fn emitting_for() -> HostCaps {
    EMITTING.get().unwrap_or_else(host_caps)
}

/// Programs handed over with a fixed wave size and the host's waves unknown.
static UNCHECKED: AtomicU64 = AtomicU64::new(0);

/// How many programs [`super::emit`] and [`super::translate_for`] handed over with a fixed
/// wave size while the host's waves were not known ([`WaveCheck::Unchecked`]): each may
/// still be refused when its pipeline is made. `VICTORY_WAVE_WHY=1` prints each.
pub fn unchecked_waves() -> u64 {
    UNCHECKED.load(Ordering::Relaxed)
}

/// Counts an unchecked wave size (and prints it with `VICTORY_WAVE_WHY`).
pub(super) fn note(check: WaveCheck) {
    if let WaveCheck::Unchecked(lanes) = check {
        UNCHECKED.fetch_add(1, Ordering::Relaxed);
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            eprintln!("WAVE-UNCHECKED WaveSize({lanes}) with the host's waves unknown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn host(min: u32, max: u32) -> HostCaps {
        HostCaps::Known(HostWaves { min, max, wave_ops: true })
    }

    /// The device's range is its fewest and most lanes, both ends in it.
    #[test]
    fn a_wave_size_is_checked_against_both_ends_of_the_hosts_range() {
        for caps in [HostCaps::Unknown, host(4, 4), host(32, 64), host(64, 64)] {
            assert_eq!(caps.check(None), Ok(WaveCheck::AnyWave), "{caps:?}");
        }
        assert_eq!(host(4, 4).check(Some(32)), Err(WAVE_TOO_WIDE));
        assert_eq!(host(32, 32).check(Some(64)), Err(WAVE_TOO_WIDE));
        assert_eq!(host(32, 64).check(Some(32)), Ok(WaveCheck::HostHas(32)));
        assert_eq!(host(32, 64).check(Some(64)), Ok(WaveCheck::HostHas(64)));
        assert_eq!(host(8, 32).check(Some(32)), Ok(WaveCheck::HostHas(32)));
        assert_eq!(host(64, 64).check(Some(32)), Err(WAVE_TOO_NARROW));
        assert_eq!(host(64, 64).check(Some(64)), Ok(WaveCheck::HostHas(64)));
    }

    /// Not known is said, not passed as a size that fits; and lane counts without wave
    /// operations are no range.
    #[test]
    fn an_unknown_host_and_one_without_wave_operations_are_told_apart_from_a_fit() {
        assert_eq!(HostCaps::Unknown.check(Some(64)), Ok(WaveCheck::Unchecked(64)));
        assert_eq!(HostCaps::at_most(0), HostCaps::Unknown);
        let no_ops = HostCaps::Known(HostWaves { min: 32, max: 32, wave_ops: false });
        assert_eq!(no_ops.check(Some(32)), Err(NO_WAVE_OPS));
        assert_eq!(no_ops.check(None), Ok(WaveCheck::AnyWave));
    }

    /// A device that does not answer leaves nothing of the device before it.
    #[test]
    fn an_answer_replaces_the_one_before_it_whole() {
        let cell = HostCell::new();
        assert_eq!(cell.get(), HostCaps::Unknown);
        cell.set(host(32, 32));
        assert_eq!(cell.get(), host(32, 32));
        cell.set(HostCaps::Unknown);
        assert_eq!(cell.get(), HostCaps::Unknown);
    }

    /// The emission's snapshot is what the lowering sees, and is gone after it.
    #[test]
    fn the_snapshot_of_an_emission_ends_with_it() {
        let was = super::super::lower::TEST_HOST_WAVE.replace(Some(32));
        assert_eq!(emitting_for(), HostCaps::at_most(32));
        with_host(host(64, 64), || assert_eq!(emitting_for(), host(64, 64)));
        assert_eq!(emitting_for(), HostCaps::at_most(32));
        super::super::lower::TEST_HOST_WAVE.set(was);
    }
}
