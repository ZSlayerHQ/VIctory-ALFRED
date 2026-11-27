//! The format kind of a typed buffer view in the slot table, with
//! `VICTORY_D16_INTEGER_VIEWS=1` only (plan K5; our 6514f1b89 and a9d7ffabb
//! on job/fork-accuracy-r2, after the base fork's 4243a96a7 and pull request
//! 15 on the runtime fork, by secure-silicon). Translated shaders read the
//! table's dword at the slot a d16 typed load reads (victory_shader's
//! `lower::slots`): bit 31 an integer format, bit 30 a signed one, beside
//! the element stride. Through such a view the load keeps each channel's
//! low 16 bits instead of narrowing a float.
//!
//! Off (the default) the host writes what `world` writes: the stride alone,
//! at title slots (or host slots under the slot map), and nothing at the
//! dynamic slots of tables made afresh. `world`'s shaders read the whole
//! dword as the stride, so a kind bit written with the switch off would
//! corrupt the strides of programs that pick buffers from the heap.

use windows::Win32::Graphics::Direct3D12::*;

use super::{HostHeap, View, srv_desc, uav_desc};

/// `VICTORY_D16_INTEGER_VIEWS=1`, the translator's switch of the same name
/// (`victory_shader::emit`), read once per process; a test's thread may
/// set it for itself ([`TEST_ON`]).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if TEST_ON.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_D16_INTEGER_VIEWS").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's `VICTORY_D16_INTEGER_VIEWS=1`, for its thread only.
    pub(crate) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The slot table's dword for a slot holding `view` whose element stride is
/// `stride` ([`super::view_stride`]): with the switch, the stride inside
/// `VIEW_STRIDE_BITS` (a stride reaching past them would read as a kind)
/// and the view's kind; without, the stride as `world` writes it.
pub(super) fn word(view: Option<&View>, stride: Option<u32>) -> u32 {
    let stride = stride.unwrap_or(0);
    if !on() {
        return stride;
    }
    stride & victory_shader::emit::VIEW_STRIDE_BITS | view.map_or(0, kind)
}

/// The kind bits of a buffer view: a typed view (no structure stride, not
/// raw) of an integer format gets `VIEW_INTEGER`, a signed one also
/// `VIEW_SIGNED`; anything else none.
pub(super) fn kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => match d[..10].try_into() {
            Ok(words) => (words, *unordered),
            Err(_) => return 0,
        },
        _ => return 0,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    if stride > 0 || raw {
        return 0;
    }
    match integer_format(format.0 as u32) {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    }
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

impl HostHeap {
    /// With the switch: the slot table's dword at dynamic slot `index`, which
    /// a table made afresh owns until it is released. Its shader is handed
    /// the dynamic slots' offset and indexes the slot table by them, so the
    /// word is the new view's (`None`: a null view or a constant buffer, 0),
    /// not what the last table there left. Without the switch nothing is
    /// written, as on `world`. Title slots keep `note_buffer`'s word.
    pub(super) fn dynamic_word(&self, index: u32, view: Option<&View>) {
        if !on() || !(self.ring..self.ring.saturating_add(self.dynamic)).contains(&index) {
            return;
        }
        let word = view.map_or(0, |v| word(Some(v), super::view_stride(v)));
        self.write_word(index, word);
    }

    /// Writes dword `index` of the slot table.
    pub(super) fn write_word(&self, index: u32, word: u32) {
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword, either value is one the slot had).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(word) };
        }
    }
}

#[cfg(test)]
#[path = "view_kind_tests.rs"]
mod tests;
