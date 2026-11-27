//! What the debug layer said, kept for the tests: a message's ID, severity, text and the
//! thread that drained it, in the store tests read, or in the start-up bucket the device's
//! own making fills (one host device is shared by tests running in parallel, so a message
//! may be another test's).

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING,
};

/// One message the debug layer kept.
#[derive(Debug, Clone)]
pub(crate) struct DebugMessage {
    pub(crate) id: i32,
    pub(crate) severity: i32,
    pub(crate) text: String,
    pub(crate) thread: String,
}

impl std::fmt::Display for DebugMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (id {}, severity {}, thread {})", self.text, self.id, self.severity, self.thread)
    }
}

/// The debug layer's errors since the layer was last drained, as
/// [`Host::log_messages`](super::Host::log_messages) took them (submissions drain the layer
/// too, so a test after one would see none).
pub(crate) static DEBUG_ERRORS: Mutex<Vec<DebugMessage>> = Mutex::new(Vec::new());

/// What the layer said while the host device was being made: the device's start-up noise,
/// not a test's fault, so only the log carries it.
static STARTUP_ERRORS: Mutex<Vec<DebugMessage>> = Mutex::new(Vec::new());

/// The message kept for what the layer said at `severity`: an error's, with its ID, text and
/// the draining thread's name; `None` for a warning or a note.
fn message(id: i32, severity: i32, text: &str) -> Option<DebugMessage> {
    (severity <= D3D12_MESSAGE_SEVERITY_ERROR.0).then(|| DebugMessage {
        id,
        severity,
        text: text.to_owned(),
        thread: std::thread::current().name().unwrap_or("unnamed").to_owned(),
    })
}

/// Keeps a message of error severity in the tests' store, or (`startup`) in the start-up
/// bucket the tests do not read.
pub(crate) fn keep_error(startup: bool, id: i32, severity: i32, text: &str) {
    let Some(message) = message(id, severity, text) else {
        return;
    };
    let mut bucket = if startup {
        STARTUP_ERRORS.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        DEBUG_ERRORS.lock().unwrap_or_else(|e| e.into_inner())
    };
    bucket.push(message);
}

/// A kept message says what the layer said of it, not only its text; warnings are left out.
#[test]
fn a_kept_message_keeps_its_id_severity_and_thread() {
    let m = message(1234, D3D12_MESSAGE_SEVERITY_ERROR.0, "made up").expect("an error is kept");
    assert_eq!(
        (m.id, m.severity, m.text.as_str()),
        (1234, D3D12_MESSAGE_SEVERITY_ERROR.0, "made up")
    );
    assert!(!m.thread.is_empty(), "the thread that drained it is named");
    assert!(
        message(5, D3D12_MESSAGE_SEVERITY_WARNING.0, "a warning").is_none(),
        "warnings are not kept"
    );
}
