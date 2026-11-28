//! Whether the title asks for a guard page: the first
//! `PAGE_GUARD` (0x100) protection the title passes through each of our
//! calls that sees one gets one line, so a single run says whether this
//! title ever sets one. Diagnostics builds, with `VICTORY_LOG_GUARD_ASKS=1`
//! only ([`on`]): every run is a diagnostics build, and the switch also
//! installs pass-throughs of the title's `VirtualAlloc` and `VirtualProtect`
//! ([`super::module`]). Unset, the imports bind as without it and the
//! `XMemVirtualAlloc` hook reads one cached flag.
//!
//! Carried from the Z-Slayer build's `victory_gpu::guard_asked`, where the
//! guard ledger also sees the title's `VirtualAlloc` and `VirtualProtect`;
//! here xmem does.

use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the guard-page line is asked for (`VICTORY_LOG_GUARD_ASKS=1`;
/// off unset or `0`), read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LOG_GUARD_ASKS").is_some_and(|v| v != "0"))
}

/// The host's `PAGE_GUARD`.
const PAGE_GUARD: u32 = 0x100;

/// The calls a title's protection value reaches us through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuardPlace {
    /// `XMemVirtualAlloc`, before the console bits are masked off.
    XMemVirtualAlloc,
    /// The title's own `VirtualAlloc` (a pass-through the switch installs).
    VirtualAlloc,
    /// The title's own `VirtualProtect` (with `VICTORY_SOFT_VIEWS`, or a
    /// pass-through the switch installs).
    VirtualProtect,
}

impl GuardPlace {
    /// How many places there are.
    const COUNT: usize = 3;

    /// This place's flag in an array of them.
    fn index(self) -> usize {
        match self {
            Self::XMemVirtualAlloc => 0,
            Self::VirtualAlloc => 1,
            Self::VirtualProtect => 2,
        }
    }
}

impl fmt::Display for GuardPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::XMemVirtualAlloc => "XMemVirtualAlloc",
            Self::VirtualAlloc => "its VirtualAlloc",
            Self::VirtualProtect => "its VirtualProtect",
        })
    }
}

/// Which places have said their line, this run.
static ASKED: [AtomicBool; GuardPlace::COUNT] = [const { AtomicBool::new(false) }; GuardPlace::COUNT];

/// [`title_asked_guard`] over the flags it is given and saying its line to
/// `say`, so a test can supply both.
fn ask(
    asked: &[AtomicBool; GuardPlace::COUNT],
    place: GuardPlace,
    address: usize,
    bytes: usize,
    protect: u32,
    say: impl FnOnce(fmt::Arguments<'_>),
) -> bool {
    if protect & PAGE_GUARD == 0 || asked[place.index()].swap(true, Ordering::Relaxed) {
        return false;
    }
    say(format_args!(
        "the title asked for a guard page (PAGE_GUARD) through {place}, first at {address:#x}, {bytes} bytes, \
         protection {protect:#x}"
    ));
    true
}

/// Logs one line the first time the title asks for a guard page (`0x100` in
/// `protect`) through `place`, at `address` and `bytes` long; true when it
/// logged. Nothing for any other protection, nothing after the first, and
/// nothing at all without the switch ([`on`]).
pub(crate) fn title_asked_guard(place: GuardPlace, address: usize, bytes: usize, protect: u32) -> bool {
    on() && ask(&ASKED, place, address, bytes, protect, |line| tracing::info!("{line}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lines a run of `asks` says, over flags of its own (no parallel
    /// test shares them).
    fn lines(asks: &[(GuardPlace, usize, usize, u32)]) -> Vec<String> {
        let asked = [const { AtomicBool::new(false) }; GuardPlace::COUNT];
        let mut said = Vec::new();
        for &(place, address, bytes, protect) in asks {
            let before = said.len();
            let logged = ask(&asked, place, address, bytes, protect, |line| said.push(line.to_string()));
            assert_eq!(logged, said.len() > before, "true exactly when it said a line");
        }
        said
    }

    #[test]
    fn a_guard_page_ask_logs_once_per_place() {
        use GuardPlace::*;
        let said = lines(&[
            (XMemVirtualAlloc, 0x1_0000_0000, 0x1_0000, 0x104),
            (XMemVirtualAlloc, 0x1_0001_0000, 0x2000, 0x104),
            (XMemVirtualAlloc, 0x1_0002_0000, 0x1000, 0x04),
            (VirtualProtect, 0x7000_0000, 0x1000, 0x102),
            (VirtualProtect, 0x7000_1000, 0x1000, 0x104),
            (VirtualAlloc, 0x8000_0000, 0x3000, 0x40),
            (VirtualAlloc, 0x8000_4000, 0x1000, 0x140),
        ]);
        assert_eq!(
            said,
            [
                "the title asked for a guard page (PAGE_GUARD) through XMemVirtualAlloc, first at 0x100000000, \
                 65536 bytes, protection 0x104",
                "the title asked for a guard page (PAGE_GUARD) through its VirtualProtect, first at 0x70000000, \
                 4096 bytes, protection 0x102",
                "the title asked for a guard page (PAGE_GUARD) through its VirtualAlloc, first at 0x80004000, \
                 4096 bytes, protection 0x140",
            ]
        );
    }

    #[test]
    fn a_run_without_a_guard_page_logs_nothing() {
        use GuardPlace::*;
        // Every host protection but the guard, and the console's GPU and
        // write-combining bits above it, through every place.
        let mut asks = Vec::new();
        for place in [XMemVirtualAlloc, VirtualAlloc, VirtualProtect] {
            for protect in [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x204, 0x404, 0x804, 0x4_0004] {
                asks.push((place, 0x1_0000_0000, 0x1_0000, protect));
            }
        }
        assert_eq!(lines(&asks), Vec::<String>::new());
    }

    #[test]
    fn the_logging_call_says_its_line_once_and_only_when_asked() {
        // The call itself, over the run's flags: the only test that passes
        // this place a guard page. Without the switch, not even once.
        assert_eq!(title_asked_guard(GuardPlace::VirtualAlloc, 0x1000, 4096, 0x104), on());
        assert!(!title_asked_guard(GuardPlace::VirtualAlloc, 0x2000, 8192, 0x104));
        assert!(!title_asked_guard(GuardPlace::VirtualAlloc, 0x3000, 4096, 0x04));
    }
}
