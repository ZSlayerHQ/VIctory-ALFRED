//! The timed key sequences `VICTORY_KEYS` plays at launch and the live
//! `LIVE_KEYS` switch plays mid-run: one parser for both.
//!
//! An entry is `<seconds>:<key>[:<hold>]`: the key goes down that many
//! seconds after the sequence's clock starts, and stays down for its hold
//! (0.2 s unless given, seconds or `<N>f` for N of the title's presents).
//! A key is a letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`,
//! `Down`, `Left` or `Right`. Entries are comma-separated; a leading `+`
//! marks an entry that counts from `VICTORY_KEYS_AFTER`'s marker file
//! instead of input start. [`key_script`] keeps the launch script's old
//! tolerance and skips entries that do not parse; [`live_keys_script`] is
//! the live switch's stricter reading, which refuses a sequence whole and
//! enforces the agent-facing limits.

use std::time::Duration;

/// How long a scripted key stays down when its entry gives no hold.
pub const KEY_PRESS: Duration = Duration::from_millis(200);

/// The most presses one live sequence may name.
pub const MAX_LIVE_PRESSES: usize = 64;

/// The longest stretch one live sequence may cover: no press starts after
/// it, and no timed hold ends after it.
pub const MAX_LIVE_DURATION: Duration = Duration::from_secs(120);

/// How long a scripted key stays down: a stretch of time, or a number of
/// the title's presents counted from the poll that pressed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    Time(Duration),
    Frames(u64),
}

/// One scripted key: when it goes down, how long it stays down, and its
/// virtual key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptedKey {
    pub at: Duration,
    pub hold: Hold,
    pub key: u8,
}

/// A launch key list split by its clock: an entry without a leading `+`
/// counts from input start, a `+` entry from the marker file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyScript {
    pub absolute: Vec<ScriptedKey>,
    pub relative: Vec<ScriptedKey>,
}

/// `text` as a count of presented frames (`<N>f`, N at least 1); `None`
/// for anything else.
fn frames(text: &str) -> Option<u64> {
    let count = text.trim().strip_suffix('f')?.parse::<u64>().ok()?;
    (count >= 1).then_some(count)
}

/// The virtual key `name` stands for: a letter, a digit, or one of the
/// named keys.
fn virtual_key(name: &str) -> Option<u8> {
    Some(match name {
        "Enter" => 0x0D,
        "Esc" => 0x1B,
        "Tab" => 0x09,
        "Back" => 0x08,
        "Up" => 0x26,
        "Down" => 0x28,
        "Left" => 0x25,
        "Right" => 0x27,
        name if name.len() == 1 && name.as_bytes()[0].is_ascii_alphanumeric() => {
            name.as_bytes()[0].to_ascii_uppercase()
        }
        _ => return None,
    })
}

/// `text` as a duration; `None` for anything but a non-negative number of
/// seconds.
fn seconds(text: &str) -> Option<Duration> {
    Duration::try_from_secs_f64(text.trim().parse().ok()?).ok()
}

/// One entry of a key list: `<seconds>:<key>[:<hold>]`, the hold seconds or
/// `<N>f` for N of the title's presents. An entry without a hold is held
/// for [`KEY_PRESS`]; a hold over 600 s, or an entry that does not parse,
/// is skipped.
fn one_key(entry: &str) -> Option<ScriptedKey> {
    let mut fields = entry.split(':');
    let at = seconds(fields.next()?)?;
    let key = virtual_key(fields.next()?.trim())?;
    let hold = match fields.next() {
        None => Hold::Time(KEY_PRESS),
        Some(text) => match frames(text) {
            Some(frames) => Hold::Frames(frames),
            None => {
                let hold = seconds(text)?;
                (hold <= Duration::from_secs(600)).then_some(Hold::Time(hold))?
            }
        },
    };
    fields
        .next()
        .is_none()
        .then_some(ScriptedKey { at, hold, key })
}

/// A `VICTORY_KEYS` list: entries that do not parse are skipped and the
/// rest kept, as the launch script has always read them.
pub fn key_script(list: &str) -> KeyScript {
    let mut script = KeyScript::default();
    for entry in list.split(',') {
        let entry = entry.trim();
        if let Some(rest) = entry.strip_prefix('+') {
            script.relative.extend(one_key(rest));
        } else {
            script.absolute.extend(one_key(entry));
        }
    }
    script
}

/// Why a live key sequence was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveKeysError {
    /// An entry is empty or does not parse, or names a key the parser does
    /// not know.
    Entry,
    /// More than [`MAX_LIVE_PRESSES`] presses.
    TooManyPresses,
    /// A press starts, or a timed hold ends, after [`MAX_LIVE_DURATION`].
    TooLong,
}

impl LiveKeysError {
    /// The reason, for the one warning a refused sequence logs.
    pub fn reason(self) -> &'static str {
        match self {
            LiveKeysError::Entry => {
                "an entry does not parse or names a key the sequence parser does not accept"
            }
            LiveKeysError::TooManyPresses => "more than 64 presses",
            LiveKeysError::TooLong => "a press starts or a hold ends after 120 seconds",
        }
    }
}

/// The end of `key`'s timed stretch; a frame hold has no end in time, so it
/// covers up to its start.
fn end(key: &ScriptedKey) -> Duration {
    match key.hold {
        Hold::Time(hold) => key.at.saturating_add(hold),
        Hold::Frames(_) => key.at,
    }
}

/// A live `LIVE_KEYS` sequence, read whole: every entry must parse, there
/// are at most [`MAX_LIVE_PRESSES`] presses, and the sequence covers at most
/// [`MAX_LIVE_DURATION`]. A leading `+` has no marker clock here: every
/// entry counts from the frame the change was applied, so the two clocks
/// are one.
pub fn live_keys_script(list: &str) -> Result<Vec<ScriptedKey>, LiveKeysError> {
    let mut keys = Vec::new();
    for entry in list.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            return Err(LiveKeysError::Entry);
        }
        let entry = entry.strip_prefix('+').unwrap_or(entry);
        keys.push(one_key(entry).ok_or(LiveKeysError::Entry)?);
    }
    if keys.len() > MAX_LIVE_PRESSES {
        return Err(LiveKeysError::TooManyPresses);
    }
    if keys.iter().any(|key| end(key) > MAX_LIVE_DURATION) {
        return Err(LiveKeysError::TooLong);
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The launch reading keeps skipping what it cannot parse, as before.
    #[test]
    fn the_launch_script_skips_malformed_entries() {
        let keys =
            key_script("300:W:20,nope,310:Q:-1,320:A:notanumber,330:B:601,340:C:0,350:!:5,,355:D")
                .absolute;
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[1].hold, Hold::Time(Duration::ZERO));
        assert_eq!(keys[2].hold, Hold::Time(KEY_PRESS));
    }

    /// A live sequence is read whole: one bad entry refuses it, including a
    /// key the parser does not know and a malformed hold.
    #[test]
    fn a_live_sequence_refuses_one_bad_entry() {
        assert!(live_keys_script("0:W,2:W").is_ok());
        assert_eq!(live_keys_script("0:W,,2:W"), Err(LiveKeysError::Entry));
        assert_eq!(live_keys_script("0:Space"), Err(LiveKeysError::Entry));
        assert_eq!(live_keys_script("0:W:wat"), Err(LiveKeysError::Entry));
        assert_eq!(live_keys_script("0:W:0f"), Err(LiveKeysError::Entry));
        assert_eq!(live_keys_script("0:W:1:2"), Err(LiveKeysError::Entry));
        // A leading `+` is one clock with the rest for a live sequence.
        let keys = live_keys_script("+2:W").expect("it parses");
        assert_eq!((keys[0].at, keys[0].key), (Duration::from_secs(2), 0x57));
    }

    /// Sixty-four presses are accepted; sixty-five are refused.
    #[test]
    fn a_live_sequence_refuses_more_than_sixty_four_presses() {
        let many = vec!["0:W"; MAX_LIVE_PRESSES].join(",");
        assert_eq!(
            live_keys_script(&many).map(|keys| keys.len()),
            Ok(MAX_LIVE_PRESSES)
        );
        let too_many = vec!["0:W"; MAX_LIVE_PRESSES + 1].join(",");
        assert_eq!(
            live_keys_script(&too_many),
            Err(LiveKeysError::TooManyPresses)
        );
    }

    /// A hundred and twenty seconds exactly is accepted, a second past it
    /// refused: the whole sequence's end, hold included.
    #[test]
    fn a_live_sequence_refuses_a_stretch_past_a_hundred_and_twenty_seconds() {
        assert!(live_keys_script("120:W:0").is_ok());
        assert!(live_keys_script("0:W:120").is_ok());
        assert_eq!(live_keys_script("121:W:0"), Err(LiveKeysError::TooLong));
        assert_eq!(live_keys_script("0:W:121"), Err(LiveKeysError::TooLong));
    }
}
