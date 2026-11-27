//! The slot table's kind bits (`VICTORY_D16_INTEGER_VIEWS`), off and on.

use super::super::view_stride;
use super::*;
use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};

/// The switch set for this thread until dropped.
struct Switched(bool);

impl Drop for Switched {
    fn drop(&mut self) {
        TEST_ON.set(self.0);
    }
}

fn switched_on() -> Switched {
    Switched(TEST_ON.replace(true))
}

/// A typed (or with `flags` 1 raw, with `stride` structured) buffer UAV of
/// DXGI format `format`. Words: format, buffer dimension, first element
/// (two), count, stride, counter (two), flags.
fn uav(format: u32, stride: u32, flags: u32) -> View {
    View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]))
}

const R16G16B16A16_FLOAT: u32 = 10;
const R16G16B16A16_UINT: u32 = 12;
const R16_FLOAT: u32 = 54;
const R16_UINT: u32 = 57;
const R16_SNORM: u32 = 58;
const R16_SINT: u32 = 59;

/// Typed buffer views of integer formats have kind bits; float, normalized,
/// raw and structured ones none (our 6514f1b89).
#[test]
fn integer_typed_buffer_views_are_marked_in_the_slot_table() {
    assert_eq!(kind(&uav(R16G16B16A16_UINT, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
    assert_eq!(kind(&uav(R16_UINT, 0, 0)), VIEW_INTEGER, "R16_UINT");
    assert_eq!(kind(&uav(R16_SINT, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
    assert_eq!(kind(&uav(64, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R8_SINT");
    assert_eq!(kind(&uav(R16G16B16A16_FLOAT, 0, 0)), 0, "R16G16B16A16_FLOAT");
    assert_eq!(kind(&uav(R16_SNORM, 0, 0)), 0, "R16_SNORM");
    assert_eq!(kind(&uav(R16G16B16A16_UINT, 0, 1)), 0, "raw");
    assert_eq!(kind(&uav(R16G16B16A16_UINT, 16, 0)), 0, "structured");
    let srv = View::Shader(0, Some([62, 1, 0, 0, 0, 0, 0, 0, 0, 0]));
    assert_eq!(kind(&srv), VIEW_INTEGER, "R8_UINT");
}

/// The dword of slot `index` of `heap`'s slot table.
fn word_at(heap: &HostHeap, index: u32) -> u32 {
    heap.stride_at(index)
}

/// Off is off: without the switch the host writes `world`'s words, the
/// stride alone, for every format (the word under the slot map is the same
/// function's, [`word`]), and nothing at a dynamic slot.
#[test]
fn off_the_host_writes_no_kind_bits() {
    if on() {
        eprintln!("NOT RUN: VICTORY_D16_INTEGER_VIEWS=1 is set for this process");
        return;
    }
    for format in [R16G16B16A16_UINT, R16_UINT, R16_SINT, 64, R16_FLOAT] {
        let v = uav(format, 0, 0);
        assert_eq!(word(Some(&v), view_stride(&v)), view_stride(&v).unwrap_or(0), "format {format}");
    }
    // A stride past VIEW_STRIDE_BITS is written as world writes it.
    let big = uav(R16G16B16A16_FLOAT, 0x8000_0010, 0);
    assert_eq!(word(Some(&big), view_stride(&big)), 0x8000_0010);
    let Some(heap) = HostHeap::new(false, 8) else {
        eprintln!("NOT RUN (no host device): the heap's words");
        return;
    };
    let _ = heap.note_buffer(2, Some(&uav(R16_UINT, 0, 0)), false);
    assert_eq!(word_at(&heap, 2), 2, "R16_UINT at a title slot: its stride, no kind");
    let _ = heap.note_buffer(3, Some(&uav(R16_SINT, 0, 0)), false);
    assert_eq!(word_at(&heap, 3), 2, "R16_SINT at a title slot: its stride, no kind");
    let first = heap.dynamic(1).expect("a free dynamic slot");
    let before = word_at(&heap, first);
    heap.make_at(first, &uav(R16G16B16A16_UINT, 0, 0));
    assert_eq!(word_at(&heap, first), before, "a dynamic slot's word is not written");
    heap.release(first, 1);
}

/// With the switch, a title slot's word carries its view's kind, and a slot
/// rewritten from R16_UINT to R16_FLOAT holds the new kind (the plan's
/// silent failure: a reused slot whose dword still holds the old one).
#[test]
fn a_title_slot_rewritten_from_an_integer_view_to_a_float_one_loses_its_kind() {
    let _on = switched_on();
    let Some(heap) = HostHeap::new(false, 8) else {
        eprintln!("NOT RUN (no host device)");
        return;
    };
    let _ = heap.note_buffer(2, Some(&uav(R16_UINT, 0, 0)), false);
    assert_eq!(word_at(&heap, 2), 2 | VIEW_INTEGER, "R16_UINT");
    let _ = heap.note_buffer(2, Some(&uav(R16_FLOAT, 0, 0)), false);
    assert_eq!(word_at(&heap, 2), 2, "R16_FLOAT, written over it");
    let _ = heap.note_buffer(2, Some(&uav(R16_SINT, 0, 0)), false);
    assert_eq!(word_at(&heap, 2), 2 | VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
    let _ = heap.note_buffer(2, None, false);
    assert_eq!(word_at(&heap, 2), 0, "nothing");
    // A stride that reaches the kind bits (none D3D12 takes does) is no kind.
    let _ = heap.note_buffer(2, Some(&uav(R16G16B16A16_FLOAT, 0x8000_0010, 0)), false);
    assert_eq!(word_at(&heap, 2) & (VIEW_INTEGER | VIEW_SIGNED), 0, "a stride is not a kind");
}

/// With the switch, a table made afresh lives in dynamic slots and its
/// shader indexes the slot table by them: such a slot carries its view's
/// stride and kind, and nothing of the table before (our a9d7ffabb).
#[test]
fn a_dynamic_slot_carries_its_views_stride_and_kind() {
    let _on = switched_on();
    let Some(heap) = HostHeap::new(false, 8) else {
        eprintln!("NOT RUN (no host device)");
        return;
    };
    let first = heap.dynamic(2).expect("free dynamic slots");
    assert!(first >= heap.count(), "a dynamic slot lies past the title's");
    heap.make_at(first, &uav(R16G16B16A16_UINT, 0, 0));
    assert_eq!(word_at(&heap, first), 8 | VIEW_INTEGER, "R16G16B16A16_UINT");
    heap.make_at(first + 1, &uav(R16_SINT, 0, 0));
    assert_eq!(word_at(&heap, first + 1), 2 | VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
    // The next table to take the slots: a float view, a constant buffer,
    // nothing.
    heap.make_at(first, &uav(R16G16B16A16_FLOAT, 0, 0));
    assert_eq!(word_at(&heap, first), 8, "R16G16B16A16_FLOAT");
    heap.constant_at(first + 1, 0, 256);
    assert_eq!(word_at(&heap, first + 1), 0, "a constant buffer");
    heap.make_null(first);
    assert_eq!(word_at(&heap, first), 0, "a null view");
    heap.release(first, 2);
}
