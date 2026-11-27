//! Which text the window's caption may show as the build's name (`host.rs`, for
//! `VICTORY_BUILD_NAME`, read when it runs, the only name it shows): a short public label,
//! never a path, an address or a login, as the caption can appear in a shared recording
//! (Sol's review of the caption's first landing, finding 1).

/// What stands in for a label that does not pass.
pub const FALLBACK: &str = "build";

/// The longest label that passes.
const LONGEST: usize = 40;

/// Segments that name a place rather than a build: URL schemes, a host's first word, home
/// folders.
const PLACES: &[&str] = &[
    "http", "https", "ftp", "file", "git", "ssh", "www", "users", "home",
];

/// Whether `text` may be shown: 1 to 40 of `A-Z a-z 0-9 . _ / -`, beginning with a letter or
/// digit, not ending with `/`, without `..` or `//`, and no segment (between `/`s) a place
/// ([`PLACES`]) or of a host's or a dotted login's shape (letters only after its last `.`:
/// `example.org`, `first.last`). A backslash, a drive's colon, `@`, `:`, a space and anything
/// not ASCII fail on the characters alone.
pub fn public(text: &str) -> bool {
    let bytes = text.as_bytes();
    let allowed = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-');
    if bytes.is_empty()
        || bytes.len() > LONGEST
        || !bytes.iter().all(allowed)
        || !bytes[0].is_ascii_alphanumeric()
        || text.ends_with('/')
        || text.contains("..")
        || text.contains("//")
    {
        return false;
    }
    text.split('/').all(|segment| {
        let place = PLACES
            .iter()
            .any(|place| segment.eq_ignore_ascii_case(place));
        let host = segment.rsplit_once('.').is_some_and(|(_, last)| {
            !last.is_empty() && last.bytes().all(|b| b.is_ascii_alphabetic())
        });
        !place && !host
    })
}
