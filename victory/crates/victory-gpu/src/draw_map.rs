//! Binding a draw or dispatch whose shaders read the slot map (`VICTORY_SLOT_MAP=1`;
//! design `jobs/2026-11-25-slot-lookup`): the view heap under the map, its pool and the
//! list's own directory as the root SRVs `MAP_POOL` and `MAP_DIR`, and the title's tables
//! at their own offsets.
//!
//! - **Versions.** A list's directory is 4 KiB of its scratch memory, taken where it first
//!   binds a heap's map. It is filled when the list is submitted, with the map as of then
//!   (`host_map::Mapped::fill`), as the title's table data is taken then, and the list
//!   holds that version until it is reset: the chunks and host slots it names stay as they
//!   are while it may run.
//! - **Tables.** Under the map no table is copied for a draw: a shader forms the title
//!   slot from the title's own offset and reads the map for its host slot. The tables'
//!   entries are resolved instead (`HostHeap::mirror`), with their buffers and textures
//!   made ready as before, and a constant buffer's record (a raw view of its host copy
//!   under the map) has its range brought up to date.
//! - **Refusals.** A shader that reads the map with no view heap under the map bound, or
//!   whose directory cannot be had, is not run: its draw or dispatch is skipped, logged
//!   once per pipeline. Never bound to a heap without the map, whose slots it would read
//!   as wrong views.

use std::sync::Arc;

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use super::{Root, State, View, readable};
use crate::descriptor::host::{HostHeap, host_map::Mapped};
use crate::host::HostList;
use crate::object::This;

/// Entries of a table resolved whole for a draw at most (a table that holds a constant
/// buffer: the copy that it replaces made every entry up to the last named or picked).
/// Past it only the named and picked entries are.
const WHOLE_TABLE_MOST: u32 = 4096;

/// Pipelines whose map-reading draws were skipped, logged once each (at most 64 named).
static SKIPPED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Logs, once per pipeline, a draw or dispatch skipped because its shaders read the map
/// and `why`.
pub(super) fn skipped(pipeline: &str, why: &'static str) {
    let mut seen = SKIPPED.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 64 && !seen.iter().any(|p| p == pipeline) {
        seen.push(pipeline.to_owned());
        tracing::warn!(pipeline, "VICTORY_SLOT_MAP=1: a draw or dispatch whose shaders read the slot map is skipped: {why}");
    }
}

impl State {
    /// The bound view heap's host heap, if it is under the map.
    pub(super) fn mapped_heap(&self) -> Option<(&'static crate::descriptor::Heap, &'static HostHeap, &'static Arc<Mapped>)> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let mapped = host.mapped.as_ref()?;
        Some((heap, host, mapped))
    }

    /// The GPU address of `list`'s directory of `mapped`'s map: taken from its scratch
    /// memory where the list first binds it, and filled when the list is submitted.
    pub(super) fn map_directory(&mut self, list: &HostList, mapped: &Arc<Mapped>) -> Option<u64> {
        use windows::core::Interface;
        // Odd: no title resource's address, which `first_use` keys textures by.
        let key = Arc::as_ptr(mapped) as usize | 1;
        let at = (list.list.as_raw() as usize, key);
        if !list.first_use(key)
            && let Some(&gpu) = self.map_dirs.get(&at)
        {
            return Some(gpu);
        }
        let bytes = u64::from(victory_shader::emit::slot_map::DIR_ENTRIES) * 4;
        let (gpu, data) = list.scratch(bytes)?;
        list.keep(mapped.pool_buffer());
        let (mapped, data, held) = (Arc::clone(mapped), data as usize, std::sync::Mutex::new(None));
        list.defer_host(move || mapped.fill(data, &held));
        self.map_dirs.insert(at, gpu);
        Some(gpu)
    }

    /// [`State::tables`] under the map: no table is made afresh. Each table's named
    /// entries, the entries root constants pick, and every entry up to them of a table
    /// that holds a constant buffer, are resolved and made ready for the draw.
    pub(super) fn mapped_tables(&mut self, list: &HostList, used: &[UsedResource], indexed: &[RootIndexedView], compute: bool) {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut slots: Vec<u8> = used.iter().filter(tables).map(|u| u.slot).collect();
        slots.extend(indexed.iter().map(|i| i.table_slot));
        slots.sort_unstable();
        slots.dedup();
        for slot in slots {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let entries: Vec<&UsedResource> = used.iter().filter(tables).filter(|u| u.slot == slot).collect();
            for u in &entries {
                if u.kind == ResourceType::Cbv {
                    self.mapped_entry(list, heap, host, table + u64::from(u.table_offset_dwords) * 4, None);
                } else {
                    self.table_entry(list, u, compute);
                }
            }
            let picked: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(compute, index.constant_slot, RootType::Constants, index.constant_dword);
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            for &k in &picked {
                if !entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                    self.mapped_entry(list, heap, host, table + u64::from(k) * 32, Some(None));
                }
            }
            if heap.samplers() || !entries.iter().any(|u| u.kind == ResourceType::Cbv) {
                continue;
            }
            let n = entries.iter().map(|u| u.table_offset_dwords / 8).chain(picked.iter().copied()).max().unwrap_or(0) + 1;
            if n > WHOLE_TABLE_MOST {
                once!(slot, n, "VICTORY_SLOT_MAP=1: a table too long to resolve whole for a draw: its named and picked entries are");
                continue;
            }
            for k in 0..n {
                let named = entries.iter().any(|u| u.table_offset_dwords / 8 == k) || picked.contains(&k);
                if !named {
                    let handle = table + u64::from(k) * 32;
                    if let Some(index) = heap.index(handle) {
                        host.mirror(index, handle, true);
                    }
                }
            }
        }
    }

    /// The table entry at `handle` resolved for a draw: a constant buffer's range brought
    /// up to date (`kind` `None`), or a view made ready as `Some(kind)` says (`Some(None)`:
    /// as its own kind).
    fn mapped_entry(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &HostHeap,
        handle: u64,
        kind: Option<Option<ResourceType>>,
    ) {
        let Some(index) = heap.index(handle) else {
            return;
        };
        let Some(view) = host.mirror(index, handle, true) else {
            return;
        };
        if let View::Constant(address, size) = view {
            if let Some(shadow) = crate::shadow::find(address, u64::from(size)) {
                shadow.refresh(list, address, u64::from(size));
            }
            return;
        }
        let Some(kind) = kind.and_then(|k| k.or_else(|| super::view_kind(&view))) else {
            return;
        };
        readable(list, kind, &view);
        if crate::descriptor::host::is_texture_view(&view) {
            host.texture_slot_tabled(index, true);
        }
    }
}
