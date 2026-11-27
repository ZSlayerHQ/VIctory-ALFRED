//! Hardware probe: a reserved resource backed by tiles of a CPU-mapped
//! custom L0 write-back heap.
//!
//! Ported from Paradoxo321/rodrion-vi (GPL-3.0), its
//! `crates/victory-gpu/src/inplace.rs` `hw_adapter_tiled_uma_probe`: test 3a
//! (a reserved resource over a CPU-mapped custom L0 write-back heap, where
//! aliases read each other's bytes, a NULL hole reads zero, and a GPU write
//! lands in the CPU's mapping), the feature line (tiled-resources tier, UMA,
//! cache-coherent UMA, GPU upload heaps) and test 5's bandwidth comparison
//! (a 16 MiB read from L0 write-back memory against VRAM), with its
//! 1,000-call `UpdateTileMappings` timing. Run it by name with `--ignored`
//! on the machine being asked; it prints its lines and returns without
//! failing where the driver refuses the heap or the device.

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

const K64: usize = 64 << 10;

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn custom_l0_heap(bytes: u64, page: D3D12_CPU_PAGE_PROPERTY) -> D3D12_HEAP_DESC {
    D3D12_HEAP_DESC {
        SizeInBytes: bytes,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: page,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        },
        Alignment: K64 as u64,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    }
}

fn fill(at: *mut u8, len: usize, seed: u8) {
    for i in 0..len {
        // SAFETY: the probe's own mapped memory of `len` bytes.
        unsafe { at.add(i).write((i as u8).wrapping_mul(31).wrapping_add(seed)) };
    }
}

fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the probe's own memory of at least that length.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

/// A queue, a list and a fence of the probe's own on `device`: nothing of
/// the host's submissions is touched.
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: the probe's own event handle.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of the probe's own.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                done: 0,
                event,
            })
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
        // SAFETY: the probe's own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            let cast = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cast)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000) != windows::Win32::Foundation::WAIT_OBJECT_0 {
                return Err(windows::core::Error::from_hresult(
                    windows::Win32::Foundation::E_FAIL,
                ));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    fn committed(
        &self,
        kind: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE),
                state,
                None,
                &mut buffer,
            )?
        };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// What the GPU reads of `size` bytes of `resource` from `offset`.
    fn fetch(
        &mut self,
        resource: &ID3D12Resource,
        offset: u64,
        size: usize,
    ) -> windows::core::Result<Vec<u8>> {
        let back = self.committed(D3D12_HEAP_TYPE_READBACK, size as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
            back.Unmap(0, None);
            Ok(bytes)
        }
    }

    /// The GPU writes `bytes` at `offset` of `resource` (a copy).
    fn store(
        &mut self,
        resource: &ID3D12Resource,
        offset: u64,
        bytes: &[u8],
    ) -> windows::core::Result<()> {
        let up = self.committed(
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            up.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64) })
    }
}

/// The feature line: the four facts that decide whether the tiled path and
/// direct-to-VRAM uploads can be planned on this machine.
fn feature_line(device: &ID3D12Device, name: &str) {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a plain query into a struct of its own size.
    let tier = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            ptr::from_mut(&mut options).cast(),
            size_of_val(&options) as u32,
        )
    };
    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1 {
        NodeIndex: 0,
        ..Default::default()
    };
    // SAFETY: as above.
    let uma = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1,
            ptr::from_mut(&mut arch).cast(),
            size_of_val(&arch) as u32,
        )
    };
    let mut options16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: as above; the query fails on an OS or driver that has no such options block.
    let uploads = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            ptr::from_mut(&mut options16).cast(),
            size_of_val(&options16) as u32,
        )
    };
    eprintln!(
        "[probe {name}] tiled-resources tier: {} | UMA: {} | cache-coherent UMA: {} | GPU upload heaps: {}",
        match tier {
            Ok(()) => format!("{:?}", options.TiledResourcesTier),
            Err(_) => "query failed".to_owned(),
        },
        match uma {
            Ok(()) => arch.UMA.as_bool().to_string(),
            Err(_) => "query failed".to_owned(),
        },
        match uma {
            Ok(()) => arch.CacheCoherentUMA.as_bool().to_string(),
            Err(_) => "query failed".to_owned(),
        },
        match uploads {
            Ok(()) => options16.GPUUploadHeapSupported.as_bool().to_string(),
            Err(_) => "query failed".to_owned(),
        }
    );
}

/// Test 3a: a reserved resource whose four tiles are a heap tile, a NULL
/// hole, another heap tile, and an alias of the first; then 1,000
/// `UpdateTileMappings` calls, timed.
fn custom_heap_tiles(device: &ID3D12Device, name: &str) {
    const TILES: u64 = 16;
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain heap creation.
    if let Err(e) = unsafe {
        device.CreateHeap(
            &custom_l0_heap(TILES * K64 as u64, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK),
            &mut heap,
        )
    } {
        eprintln!(
            "[3a {name}] CreateHeap CUSTOM L0 WRITE_BACK refused ({e:?}): tiles cannot be backed by CPU-mapped memory here"
        );
        return;
    }
    let heap = heap.expect("a heap when CreateHeap succeeds");
    // A placed buffer over the whole heap, persistently mapped.
    let mut placed: Option<ID3D12Resource> = None;
    // SAFETY: a live heap of the probe's own.
    let made = unsafe {
        device.CreatePlacedResource(
            &heap,
            0,
            &buffer_desc(TILES * K64 as u64, D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut placed,
        )
    };
    let mut base: *mut c_void = std::ptr::null_mut();
    // SAFETY: the placed buffer, mapped once for the probe's life.
    let mapped = match (&made, &placed) {
        (Ok(()), Some(buffer)) => unsafe { buffer.Map(0, None, Some(&mut base)) }.is_ok() && !base.is_null(),
        _ => false,
    };
    eprintln!("[3a {name}] placed buffer: {made:?}; CPU mapping: {mapped}");
    let at = base.cast::<u8>();
    // Four tiles: heap[7], a NULL hole, heap[2], and heap[7] again (an alias).
    let mut reserved: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer without memory.
    if let Err(e) = unsafe {
        device.CreateReservedResource(
            &buffer_desc(4 * K64 as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut reserved,
        )
    } {
        eprintln!("[3a {name}] CreateReservedResource refused ({e:?})");
        return;
    }
    let resource = reserved.expect("a reserved buffer when CreateReservedResource succeeds");
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            eprintln!("[3a {name}] no queue of the probe's own ({e:?})");
            return;
        }
    };
    map_four(&rig.queue, &resource, &heap);
    // SAFETY: a plain query of a live device.
    if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
        eprintln!("[3a {name}] the device was removed by UpdateTileMappings ({e:?})");
        return;
    }
    if mapped {
        fill(at.wrapping_add(7 * K64), K64, 0x11);
        fill(at.wrapping_add(2 * K64), K64, 0x22);
    }
    let tile = |rig: &mut Rig, n: u64| rig.fetch(&resource, n * K64 as u64, K64);
    let t0 = tile(&mut rig, 0).is_ok_and(|got| mapped && same(at.wrapping_add(7 * K64), &got));
    let t1 = tile(&mut rig, 1).is_ok_and(|got| got.iter().all(|&b| b == 0));
    let t2 = tile(&mut rig, 2).is_ok_and(|got| mapped && same(at.wrapping_add(2 * K64), &got));
    let t3 = tile(&mut rig, 3).is_ok_and(|got| mapped && same(at.wrapping_add(7 * K64), &got));
    // The GPU writes tile 0; the alias (tile 3) and the CPU's mapping must see it.
    let wrote = rig.store(&resource, 32, &[0xAB; 64]).is_ok();
    let alias = rig
        .fetch(&resource, 3 * K64 as u64 + 32, 64)
        .is_ok_and(|got| got == [0xAB; 64]);
    let cpu_sees = mapped && same(at.wrapping_add(7 * K64 + 32), &[0xAB; 64]);
    // SAFETY: a plain query of a live device.
    let removed = unsafe { device.GetDeviceRemovedReason() };
    eprintln!(
        "[3a {name}] tile0=heap[7]: {t0} | hole reads zero: {t1} | tile2=heap[2]: {t2} | tile3 aliases tile0: {t3} | GPU write: {wrote} | alias reads it: {alias} | CPU mapping sees it: {cpu_sees} | device: {removed:?}"
    );
    if removed.is_err() {
        return;
    }
    let began = Instant::now();
    for _ in 0..1000 {
        map_four(&rig.queue, &resource, &heap);
    }
    let micros = began.elapsed().as_micros();
    // SAFETY: a plain query of a live device.
    let removed = unsafe { device.GetDeviceRemovedReason() };
    eprintln!(
        "[3a {name}] 1000 x UpdateTileMappings (4 regions): {micros} us, {:.2} us a call | device: {removed:?}",
        micros as f64 / 1000.0,
    );
    if let Some(buffer) = placed {
        // SAFETY: still mapped from above and unmapped once.
        unsafe { buffer.Unmap(0, None) };
    }
}

/// `UpdateTileMappings` for test 3a's four regions: heap[7], a NULL hole,
/// heap[2], heap[7] again.
fn map_four(queue: &ID3D12CommandQueue, resource: &ID3D12Resource, heap: &ID3D12Heap) {
    let coords = [
        D3D12_TILED_RESOURCE_COORDINATE { X: 0, Y: 0, Z: 0, Subresource: 0 },
        D3D12_TILED_RESOURCE_COORDINATE { X: 1, Y: 0, Z: 0, Subresource: 0 },
        D3D12_TILED_RESOURCE_COORDINATE { X: 2, Y: 0, Z: 0, Subresource: 0 },
        D3D12_TILED_RESOURCE_COORDINATE { X: 3, Y: 0, Z: 0, Subresource: 0 },
    ];
    let sizes = [D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }; 4];
    let flags = [
        D3D12_TILE_RANGE_FLAG_NONE,
        D3D12_TILE_RANGE_FLAG_NULL,
        D3D12_TILE_RANGE_FLAG_NONE,
        D3D12_TILE_RANGE_FLAG_NONE,
    ];
    let offsets = [7u32, 0, 2, 7];
    let counts = [1u32; 4];
    // SAFETY: a live reserved resource and heap; four regions and four
    // ranges whose arrays live through the call.
    unsafe {
        queue.UpdateTileMappings(
            resource,
            4,
            Some(coords.as_ptr()),
            Some(sizes.as_ptr()),
            heap,
            4,
            Some(flags.as_ptr()),
            Some(offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
}

/// Test 5: a 16 MiB reserved resource whose tiles come from two 64 MiB
/// L0 write-back heaps — CPU writes seen by the GPU, GPU writes seen by
/// the CPU — then the bandwidth of reading it against a plain VRAM buffer.
fn two_heap_bandwidth(device: &ID3D12Device, name: &str) {
    const HEAP: u64 = 64 << 20;
    const TILES: usize = 256;
    let mut heaps = Vec::new();
    for n in 0..2 {
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain heap creation.
        if let Err(e) = unsafe {
            device.CreateHeap(
                &custom_l0_heap(HEAP, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK),
                &mut heap,
            )
        } {
            eprintln!("[5 {name}] heap {n} refused ({e:?}); stopping after {n} heaps");
            return;
        }
        let heap = heap.expect("a heap when CreateHeap succeeds");
        let mut placed: Option<ID3D12Resource> = None;
        // SAFETY: a live heap of the probe's own.
        if let Err(e) = unsafe {
            device.CreatePlacedResource(
                &heap,
                0,
                &buffer_desc(HEAP, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut placed,
            )
        } {
            eprintln!("[5 {name}] heap {n}: no placed buffer ({e:?})");
            return;
        }
        let placed = placed.expect("a placed buffer when creation succeeds");
        let mut base: *mut c_void = std::ptr::null_mut();
        // SAFETY: the probe's own placed buffer, mapped once.
        if unsafe { placed.Map(0, None, Some(&mut base)) }.is_err() || base.is_null() {
            eprintln!("[5 {name}] heap {n}: no CPU mapping");
            return;
        }
        heaps.push((heap, placed, base.cast::<u8>()));
    }
    let mut reserved: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer without memory.
    if let Err(e) = unsafe {
        device.CreateReservedResource(
            &buffer_desc((TILES * K64) as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut reserved,
        )
    } {
        eprintln!("[5 {name}] CreateReservedResource refused ({e:?})");
        return;
    }
    let resource = reserved.expect("a reserved buffer when creation succeeds");
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            eprintln!("[5 {name}] no queue of the probe's own ({e:?})");
            return;
        }
    };
    // Even tiles from heap 0 (offset +5), odd tiles from heap 1 (offset +17).
    let mut evens = Vec::new();
    let mut odds = Vec::new();
    for i in 0..TILES {
        let coordinate = D3D12_TILED_RESOURCE_COORDINATE {
            X: i as u32,
            Y: 0,
            Z: 0,
            Subresource: 0,
        };
        if i % 2 == 0 {
            evens.push((coordinate, (i / 2 + 5) as u32));
        } else {
            odds.push((coordinate, (i / 2 + 17) as u32));
        }
    }
    let began = Instant::now();
    map_ranges(&rig.queue, &resource, &heaps[0].0, &evens);
    map_ranges(&rig.queue, &resource, &heaps[1].0, &odds);
    let mapping_us = began.elapsed().as_micros();
    // A CBV, an SRV and a UAV over the reserved resource can be made.
    let made_views = make_views(device, &resource, (TILES * K64) as u64);
    // SAFETY: a plain query of a live device.
    if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
        eprintln!("[5 {name}] the device was removed by the tile mappings ({e:?})");
        return;
    }
    // CPU writes through the mapped heaps; the GPU reads them through the resource.
    fill(heaps[0].2.wrapping_add(5 * K64), K64, 0x33);
    fill(heaps[1].2.wrapping_add(17 * K64), K64, 0x44);
    let r0 = rig
        .fetch(&resource, 0, K64)
        .is_ok_and(|got| same(heaps[0].2.wrapping_add(5 * K64), &got));
    let r1 = rig
        .fetch(&resource, K64 as u64, K64)
        .is_ok_and(|got| same(heaps[1].2.wrapping_add(17 * K64), &got));
    // GPU writes through the resource; the CPU reads the mapped heaps.
    let w0 = rig.store(&resource, 16, &[0xE1; 64]).is_ok()
        && same(heaps[0].2.wrapping_add(5 * K64 + 16), &[0xE1; 64]);
    let w1 = rig.store(&resource, K64 as u64 + 16, &[0xE2; 64]).is_ok()
        && same(heaps[1].2.wrapping_add(17 * K64 + 16), &[0xE2; 64]);
    // Bandwidth: a 16 MiB read of the tiled L0 write-back memory against
    // the same read of a plain VRAM buffer.
    let bytes = TILES * K64;
    let began = Instant::now();
    let tiled = rig.fetch(&resource, 0, bytes).is_ok();
    let tiled_ms = began.elapsed().as_secs_f64() * 1000.0;
    let (vram_ok, vram_ms) = match rig.committed(D3D12_HEAP_TYPE_DEFAULT, bytes as u64, D3D12_RESOURCE_STATE_COMMON) {
        Ok(plain) => {
            let began = Instant::now();
            let read = rig.fetch(&plain, 0, bytes).is_ok();
            (read, began.elapsed().as_secs_f64() * 1000.0)
        }
        Err(e) => {
            eprintln!("[5 {name}] no VRAM buffer for the comparison ({e:?})");
            (false, 0.0)
        }
    };
    let gbs = |ms: f64| if ms > 0.0 { (bytes / (1 << 20)) as f64 / 1024.0 / ms * 1000.0 } else { 0.0 };
    // SAFETY: a plain query of a live device.
    let removed = unsafe { device.GetDeviceRemovedReason() };
    eprintln!(
        "[5 {name}] mapping 256 tiles across 2 heaps: {mapping_us} us | views (CBV/SRV/UAV): {made_views} | CPU write seen by GPU: {r0} / {r1} | GPU write seen by CPU: {w0} / {w1} | 16 MiB GPU read: L0 write-back {tiled_ms:.2} ms ({:.1} GB/s, ok={tiled}), VRAM {vram_ms:.2} ms ({:.1} GB/s, ok={vram_ok}) | device: {removed:?}",
        gbs(tiled_ms),
        gbs(vram_ms),
    );
    for (_, placed, base) in &heaps {
        let _ = (placed, base);
        // SAFETY: still mapped from above and unmapped once.
        unsafe { placed.Unmap(0, None) };
    }
}

/// One `UpdateTileMappings` call: each coordinate takes one tile from
/// `heap` at its offset.
fn map_ranges(
    queue: &ID3D12CommandQueue,
    resource: &ID3D12Resource,
    heap: &ID3D12Heap,
    ranges: &[(D3D12_TILED_RESOURCE_COORDINATE, u32)],
) {
    let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = ranges.iter().map(|r| r.0).collect();
    let sizes = vec![D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() }; ranges.len()];
    let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; ranges.len()];
    let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
    let counts = vec![1u32; ranges.len()];
    // SAFETY: a live reserved resource and heap; as many regions and ranges
    // as the arrays hold, which live through the call.
    unsafe {
        queue.UpdateTileMappings(
            resource,
            ranges.len() as u32,
            Some(coords.as_ptr()),
            Some(sizes.as_ptr()),
            heap,
            ranges.len() as u32,
            Some(flags.as_ptr()),
            Some(offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
}

/// A CBV, an SRV and a UAV over the reserved resource, as a tiled pool
/// would make: true if the descriptor heap was made and the views were
/// asked for without the device falling over.
fn make_views(device: &ID3D12Device, resource: &ID3D12Resource, bytes: u64) -> bool {
    let heap: windows::core::Result<ID3D12DescriptorHeap> = {
        // SAFETY: plain heap creation.
        unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
        }
    };
    let Ok(heap) = heap else {
        return false;
    };
    // SAFETY: four descriptors of the probe's own heap; raw buffer views over
    // the resource's whole length.
    unsafe {
        let first = heap.GetCPUDescriptorHandleForHeapStart();
        let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let at = |n: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: first.ptr + n * step,
        };
        device.CreateConstantBufferView(
            Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: resource.GetGPUVirtualAddress(),
                SizeInBytes: 256,
            }),
            at(0),
        );
        device.CreateShaderResourceView(
            resource,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: (bytes / 4) as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            }),
            at(1),
        );
        device.CreateUnorderedAccessView(
            resource,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: (bytes / 4) as u32,
                        StructureByteStride: 0,
                        CounterOffsetInBytes: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    },
                },
            }),
            at(2),
        );
    }
    true
}

/// The probe. Run it by name on the machine being asked:
/// `cargo test -p victory-gpu hw_adapter_tiled_uma_probe -- --ignored --nocapture`
#[test]
#[ignore = "hardware probe: run by name on a machine with a card (see the file header)"]
fn hw_adapter_tiled_uma_probe() {
    let made = match victory_shader::adapter::test_device(
        victory_shader::adapter::Asked::Hardware,
        None,
    ) {
        Ok(made) => made,
        Err(why) => {
            eprintln!("tiled probe skipped: {why}");
            return;
        }
    };
    let name = made
        .adapter
        .as_ref()
        .map(|adapter| adapter.description.clone())
        .unwrap_or_else(|| "the device's adapter".to_owned());
    feature_line(&made.device, &name);
    custom_heap_tiles(&made.device, &name);
    // SAFETY: a plain query of a live device.
    if let Err(e) = unsafe { made.device.GetDeviceRemovedReason() } {
        eprintln!("[probe {name}] the device was removed; no further tests ({e:?})");
        return;
    }
    two_heap_bandwidth(&made.device, &name);
    eprintln!("[probe {name}] done");
}
