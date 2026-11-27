//! `VICTORY_D3D12_BACKGROUND`: the display driver's background processing,
//! and the counts of pipeline states made.
//!
//! The CPU trace of 2026-11-26 (`jobs/2026-11-26-cbj-review/trace-findings.md`)
//! found four Direct3D 12 background-scheduler threads inside the display
//! driver's compiler using most of a second of CPU a frame in a still scene.
//! Direct3D 12 lets a device say what background work it allows
//! (`ID3D12Device6::SetBackgroundProcessingMode`); the switch picks one of
//! its modes. Unset, no call is made, as before.
//!
//! Once a scene has settled, a pipeline state made is one that could not be
//! loaded from the pipeline library (a miss by name), so the count made is
//! also the count of re-creations: `pipeline::library::MADE`. No count per
//! minute existed (the library's save line and "host pipelines all made"
//! line carry the running totals only), so one is logged every
//! [`SUMMARY_EVERY`] frames and kept in the live stats file.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// A mode `VICTORY_D3D12_BACKGROUND` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Allowed,
    Intrusive,
    NoBackground,
    NoProfiling,
}

impl Mode {
    fn d3d(self) -> D3D12_BACKGROUND_PROCESSING_MODE {
        match self {
            Mode::Allowed => D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
            Mode::Intrusive => D3D12_BACKGROUND_PROCESSING_MODE_ALLOW_INTRUSIVE_MEASUREMENTS,
            Mode::NoBackground => D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
            Mode::NoProfiling => D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_PROFILING_BY_SYSTEM,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Allowed => "allowed",
            Mode::Intrusive => "intrusive",
            Mode::NoBackground => "no-background",
            Mode::NoProfiling => "no-profiling",
        }
    }
}

/// The mode `text` names; `None` for anything else. `0` is the old spelling
/// of `no-background`, kept so an existing run's arguments still work.
pub(crate) fn parse(text: &str) -> Option<Mode> {
    match text.trim().to_ascii_lowercase().as_str() {
        "allowed" => Some(Mode::Allowed),
        "intrusive" => Some(Mode::Intrusive),
        "no-background" | "0" => Some(Mode::NoBackground),
        "no-profiling" => Some(Mode::NoProfiling),
        _ => None,
    }
}

/// The mode `VICTORY_D3D12_BACKGROUND` names, read once; `None` when it is
/// unset or names nothing.
pub(crate) fn configured() -> Option<Mode> {
    static MODE: OnceLock<Option<Mode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(text) = std::env::var("VICTORY_D3D12_BACKGROUND") else {
            return None;
        };
        let mode = parse(&text);
        if mode.is_none() && !text.trim().is_empty() {
            tracing::warn!(
                value = %text,
                "VICTORY_D3D12_BACKGROUND: no such mode (allowed, intrusive, no-background, no-profiling); no call is made"
            );
        }
        mode
    })
}

/// The device has been made: set the configured background mode, if any.
pub(crate) fn apply(device: &ID3D12Device) {
    apply_mode(device, configured());
}

/// Calls made, for the test that unset makes none.
static CALLS: AtomicU64 = AtomicU64::new(0);
static NO_DEVICE6_SAID: AtomicBool = AtomicBool::new(false);
static FAILED_SAID: AtomicBool = AtomicBool::new(false);

/// [`apply`] with `mode` given: the call, its one line, and the test hooks.
fn apply_mode(device: &ID3D12Device, mode: Option<Mode>) {
    let Some(mode) = mode else { return };
    let Ok(device6) = device.cast::<ID3D12Device6>() else {
        if !NO_DEVICE6_SAID.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                mode = mode.name(),
                "ID3D12Device6 is not on this device: VICTORY_D3D12_BACKGROUND is not applied"
            );
        }
        return;
    };
    CALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: a call on the live device's interface, with `None` for both
    // optional out pointers.
    let set = unsafe {
        device6.SetBackgroundProcessingMode(
            mode.d3d(),
            D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
            None,
            None,
        )
    };
    match &set {
        Ok(()) => tracing::info!(
            mode = mode.name(),
            ok = true,
            "D3D12 background processing mode set"
        ),
        Err(e) => {
            // The disabling modes can be refused with Windows' developer mode
            // off; say so once and carry on with the device as it was.
            if !FAILED_SAID.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    mode = mode.name(),
                    error = %e,
                    "D3D12 background processing mode refused; the run carries on"
                );
            }
        }
    }
}

/// One pipeline-states line every this many frames.
pub(crate) const SUMMARY_EVERY: u64 = 600;

/// The last summary's frame, time, and running counts.
static LAST: Mutex<Option<(u64, u64, u64, u64, u64)>> = Mutex::new(None);
/// The last stats write's time and running count made, for its rate.
static STATS_LAST: Mutex<Option<(u64, u64)>> = Mutex::new(None);

/// A presented frame: every [`SUMMARY_EVERY`] frames, one line with the
/// pipeline states made (and loaded, and shaders translated) since the one
/// before, and their rate a minute.
pub(crate) fn frame(frame: u64) {
    if frame == 0 || !frame.is_multiple_of(SUMMARY_EVERY) {
        return;
    }
    let (made, loaded) = crate::pipeline::library_counts();
    let translated = crate::pipeline::kept()[2];
    let now = super::now_ms();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let (frames, before_made, before_loaded, before_translated, before_ms) =
        last.unwrap_or((0, made, loaded, translated, now));
    *last = Some((frame, made, loaded, translated, now));
    let minutes = now.saturating_sub(before_ms) as f64 / 60_000.0;
    let per_minute = |count: u64| {
        if minutes > 0.0 {
            count as f64 / minutes
        } else {
            0.0
        }
    };
    let made_since = made.saturating_sub(before_made);
    let loaded_since = loaded.saturating_sub(before_loaded);
    let translated_since = translated.saturating_sub(before_translated);
    tracing::info!(
        frames = frame.saturating_sub(frames),
        created = made_since,
        created_per_minute = format_args!("{:.1}", per_minute(made_since)),
        loaded = loaded_since,
        shaders_translated = translated_since,
        total_created = made,
        "pipeline states made since the last such line (every 600 frames)"
    );
}

/// The live stats file's pipeline figures: created in all, and per minute
/// since the previous write.
pub(crate) fn pipeline_figures() -> (u64, f64) {
    let made = crate::pipeline::library_counts().0;
    let now = super::now_ms();
    let mut last = STATS_LAST.lock().unwrap_or_else(|e| e.into_inner());
    let rate = match *last {
        Some((was, before)) if now > was => {
            (made.saturating_sub(before)) as f64 * 60_000.0 / now.saturating_sub(was) as f64
        }
        _ => 0.0,
    };
    *last = Some((now, made));
    (made, rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two tests below share the device's background mode and the call
    /// counter: one at a time.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Every mode is parsed, case and spaces around it ignored; the old
    /// `0` spelling still names no-background; anything else is no call.
    #[test]
    fn the_setting_parses() {
        assert_eq!(parse("allowed"), Some(Mode::Allowed));
        assert_eq!(parse(" INTRUSIVE "), Some(Mode::Intrusive));
        assert_eq!(parse("no-background"), Some(Mode::NoBackground));
        assert_eq!(parse("no-profiling"), Some(Mode::NoProfiling));
        assert_eq!(parse("0"), Some(Mode::NoBackground));
        assert_eq!(parse(""), None);
        assert_eq!(parse("everything"), None);
        assert_eq!(
            Mode::NoProfiling.d3d(),
            D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_PROFILING_BY_SYSTEM
        );
    }

    /// Unset makes no call (the counter the tests read shows it).
    #[test]
    fn unset_makes_no_call() {
        let _one = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return;
        };
        let before = CALLS.load(Ordering::Relaxed);
        apply_mode(&host.device, None);
        assert_eq!(CALLS.load(Ordering::Relaxed), before);
    }

    /// On the software adapter the call is made, its result is the device's
    /// to refuse, and the device still works afterwards.
    #[test]
    fn the_software_adapter_takes_the_call() {
        let _one = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::test_host() else {
            return;
        };
        let before = CALLS.load(Ordering::Relaxed);
        apply_mode(&host.device, Some(Mode::NoBackground));
        assert!(CALLS.load(Ordering::Relaxed) > before, "the call was made");
        // SAFETY: a plain query of the live device.
        assert!(
            unsafe { host.device.GetDeviceRemovedReason() }.is_ok(),
            "the device still lives"
        );
    }
}
