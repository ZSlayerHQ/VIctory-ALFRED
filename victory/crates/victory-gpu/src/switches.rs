//! The runtime's boolean environment switches: `VICTORY_...` names whose set state turns a
//! diagnostic or an experiment on.

use std::ffi::OsStr;
use std::sync::Mutex;

/// Whether a switch's value says on: anything but `0`, `false`, `off` or `no` (any case),
/// and an empty value off; surrounding space is ignored. A switch whose value is a path, a
/// number or a mode (`VICTORY_TEXTURE_HIGH=2`, `VICTORY_D3D12_GBV=states`) parses its own
/// value and does not use this.
pub(crate) fn value_on(value: &OsStr) -> bool {
    let text = value.to_string_lossy();
    let text = text.trim();
    !(text.is_empty()
        || text.eq_ignore_ascii_case("0")
        || text.eq_ignore_ascii_case("false")
        || text.eq_ignore_ascii_case("off")
        || text.eq_ignore_ascii_case("no"))
}

/// Whether the boolean switch `name` is on: set, and not to an off value ([`value_on`]).
/// Says each switch set to a non-default state once, with its value and state, so a log
/// says what it measured under (before, `VICTORY_D3D12_DRED=0` turned DRED on).
pub(crate) fn switch_on(name: &'static str) -> bool {
    let set = std::env::var_os(name);
    let on = set.as_deref().is_some_and(value_on);
    if set.is_some() {
        static SAID: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if !said.contains(&name) {
            said.push(name);
            let value = set.as_deref().map(|v| v.to_string_lossy()).unwrap_or_default();
            tracing::info!(switch = name, value = %value, on, "diagnostic switch set");
        }
    }
    on
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    /// Every spelling the rule names is read as it says.
    #[test]
    fn the_off_spellings_are_off_and_the_rest_on() {
        for off in ["", "0", "false", "FALSE", "off", "Off", " 0 ", "no", "   "] {
            assert!(!value_on(OsStr::new(off)), "{off:?} is off");
        }
        for on in ["1", "true", "yes", "on", "states", "2", " 1 "] {
            assert!(value_on(OsStr::new(on)), "{on:?} is on");
        }
        assert!(!switch_on("VICTORY_SWITCH_THAT_CANNOT_EXIST"), "an unset switch is off");
    }
}
