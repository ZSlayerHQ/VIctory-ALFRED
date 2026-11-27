//! The host side of the title-slot to host-slot map (`VICTORY_SLOT_MAP=1`; design
//! `jobs/2026-11-25-slot-lookup`): what a host view heap keeps for it beside the CPU map
//! (`super::super::slot_map`).
//!
//! - **Classes.** The map's entry says the class of the descriptor the host made in its
//!   host slot (`victory_shader::emit::slot_map::class`: the DXIL resource kind, plus 32
//!   for unordered access), and a shader reads it only through a handle of that class.
//!   The class comes from the description the host actually made the view with
//!   ([`srv_class`], [`uav_class`]), after its own changes (a structured view made raw,
//!   a depth texture's colour twin), not from the title's. A structured buffer view the
//!   host could not make raw keeps the bare structured kind, which no access's class is
//!   (raw and structured accesses are one class, and read raw): it reads its null.
//! - **Typed nulls.** Host slot `c` of a view heap under the map is the null of class `c`
//!   ([`typed_null`]), which an access of class `c` reads for a miss or a view of
//!   another class.
//! - **Constant buffers.** A constant buffer record is a raw view of its memory's host
//!   copy ([`constant_raw`]), of the raw buffer's class, which the translator reads it as.
#![cfg_attr(not(test), allow(dead_code))]

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// DXIL resource kinds.
const TYPED_BUFFER: u32 = 10;
const RAW_BUFFER: u32 = 11;
const STRUCTURED_BUFFER: u32 = 12;
const CBUFFER: u32 = 13;
const ACCELERATION_STRUCTURE: u32 = 16;

/// The DXIL kind of each SRV dimension (`D3D12_SRV_DIMENSION`, 0 unknown to 11 the
/// ray-tracing structure; 1, a buffer, is told apart by its description). UAV dimensions
/// are numbered alike up to 8.
const KIND_OF_DIMENSION: [u32; 12] = [0, RAW_BUFFER, 1, 6, 2, 7, 3, 8, 4, 5, 9, ACCELERATION_STRUCTURE];

/// The SRV (and UAV) dimension of each texture kind, by kind (0: none).
fn dimension_of_kind(kind: u32) -> u32 {
    KIND_OF_DIMENSION
        .iter()
        .position(|&k| k == kind && (1..=9).contains(&k))
        .map_or(0, |d| d as u32)
}

/// The kind of a texture's own view: its dimension, array size and samples.
fn own_kind(texture: &D3D12_RESOURCE_DESC) -> u32 {
    let arrayed = texture.DepthOrArraySize > 1;
    match texture.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            if arrayed {
                6
            } else {
                1
            }
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => match (texture.SampleDesc.Count > 1, arrayed) {
            (true, true) => 8,
            (true, false) => 3,
            (false, true) => 7,
            (false, false) => 2,
        },
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 4,
        _ => 0,
    }
}

/// The kind of a buffer view: raw, typed, or structured (the bare kind, no access's).
fn buffer_kind(format: DXGI_FORMAT, stride: u32, raw: bool) -> u32 {
    if raw {
        RAW_BUFFER
    } else if format == DXGI_FORMAT_UNKNOWN && stride > 0 {
        STRUCTURED_BUFFER
    } else {
        TYPED_BUFFER
    }
}

/// The class of a view of `kind`, `unordered` or not. Unlike an access's class, a
/// structured kind stays apart (see the module).
fn of(kind: u32, unordered: bool) -> u32 {
    (kind & 31) | if unordered { 32 } else { 0 }
}

/// The map's class (`victory_shader::emit::slot_map::class`) of a shader resource view made
/// with `desc`, or with no description of `texture`'s own view.
pub(crate) fn srv_class(desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>, texture: Option<&D3D12_RESOURCE_DESC>) -> u32 {
    let kind = match desc {
        Some(d) if d.ViewDimension == D3D12_SRV_DIMENSION_BUFFER => {
            // SAFETY: a buffer view's union member.
            let b = unsafe { d.Anonymous.Buffer };
            buffer_kind(d.Format, b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0)
        }
        Some(d) => KIND_OF_DIMENSION.get(d.ViewDimension.0 as usize).copied().unwrap_or(0),
        None => texture.map_or(0, own_kind),
    };
    of(kind, false)
}

/// The map's class of an unordered access view made with `desc`, or with no description
/// of `texture`'s own view.
pub(crate) fn uav_class(desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>, texture: Option<&D3D12_RESOURCE_DESC>) -> u32 {
    let kind = match desc {
        Some(d) if d.ViewDimension == D3D12_UAV_DIMENSION_BUFFER => {
            // SAFETY: a buffer view's union member.
            let b = unsafe { d.Anonymous.Buffer };
            buffer_kind(d.Format, b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0)
        }
        Some(d) if (2..=8).contains(&d.ViewDimension.0) => KIND_OF_DIMENSION[d.ViewDimension.0 as usize],
        Some(_) => 0,
        None => texture.map_or(0, own_kind),
    };
    of(kind, true)
}

/// The class of a constant buffer view (DXIL `CBuffer`).
pub(crate) const CBV_CLASS: u32 = CBUFFER;

thread_local! {
    /// The class of the last view [`super::make`] made on this thread ([`note`]), 0 when
    /// none since [`Mapped::resolve`] cleared it.
    static MADE_CLASS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// [`super::make`] made a view of class `class`.
pub(crate) fn note(class: u32) {
    MADE_CLASS.with(|c| c.set(class));
}

/// The host object the descriptor of `view` would view (as [`super::make`] picks it: a
/// texture's host copy, a buffer's shadow copy), as a number for the map's key; 0 for
/// none. A texture made again, or a buffer given a new host copy, is another backing:
/// the view's next resolve takes a new host slot.
fn backing_of(view: &super::View) -> u64 {
    use super::View;
    use windows::core::Interface;
    let shadow = |address: u64, len: u64| crate::shadow::find(address, len).map_or(0, |s| s.buffer.as_raw() as u64);
    match view {
        View::Shader(r, _) | View::Unordered(r, _, _) => {
            let r = *r as crate::object::This;
            match crate::resource::host_resource(r) {
                Some(texture) => texture.as_raw() as u64,
                None => crate::resource::buffer_bytes(r).map_or(0, |(data, len)| shadow(data as u64, len)),
            }
        }
        &View::Constant(address, size) => shadow(address, u64::from(size)),
        View::Placed(_, resource, desc) => match super::placed_texture(view) {
            Some(texture) => backing_of(&texture),
            None => shadow(
                u64::from(desc[10]) | u64::from(desc[11]) << 32,
                u64::from(resource[4]) | u64::from(resource[5]) << 32,
            ),
        },
        View::Sampler(_) | View::ConsoleSampler(_) => 0,
    }
}

/// What a host view heap keeps under the map: the CPU map, the card's pool, and the host
/// resource each pool slot's descriptor views (held until the map frees the slot: a
/// version a list holds may still name it).
pub(crate) struct Mapped {
    map: std::sync::Mutex<super::super::slot_map::SlotMap>,
    /// The pool the shaders read (`MAP_POOL`): `chunks * CHUNK_SLOTS` dwords of upload
    /// memory, its GPU address and its mapping. A chunk is written only while no
    /// version names it.
    pool: (ID3D12Resource, u64, usize),
    /// Chunks of the pool, chunk 0 included.
    chunks: u32,
    kept: std::sync::Mutex<Vec<Option<ID3D12Resource>>>,
}

impl Mapped {
    /// The map of a title heap of `title_slots`, with `pool_slots` host slots from
    /// `POOL_BASE` and a card pool of `chunks` chunks (chunk 0 all misses).
    pub(crate) fn new(title_slots: u32, pool_slots: u32, chunks: u32) -> Option<Self> {
        use victory_shader::emit::slot_map::CHUNK_SLOTS;
        let host = crate::host::host()?;
        let bytes = u64::from(chunks) * u64::from(CHUNK_SLOTS) * 4;
        let (buffer, data) = host.upload_buffer(bytes)?;
        // SAFETY: the buffer's whole mapping; a live buffer.
        let gpu = unsafe {
            std::ptr::write_bytes(data, 0, bytes as usize);
            buffer.GetGPUVirtualAddress()
        };
        crate::host::name(&buffer, || "slot map pool".to_owned());
        Some(Mapped {
            map: std::sync::Mutex::new(super::super::slot_map::SlotMap::new(title_slots, pool_slots, chunks)),
            pool: (buffer, gpu, data as usize),
            chunks,
            kept: std::sync::Mutex::new(vec![None; pool_slots as usize]),
        })
    }

    /// Title slot `t` holds `view`, whose record is `record`: it maps to the view's host
    /// slot, made at `at(host slot)` by [`super::make`] when the view has none yet, and
    /// then told to `made(host slot)` (the slot's stride), all before any version can
    /// name the slot.
    pub(crate) fn resolve(
        &self,
        t: u32,
        view: &super::View,
        record: u64,
        at: impl Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE,
        made: impl FnOnce(u32),
    ) -> super::super::slot_map::Resolved {
        use super::super::slot_map::{POOL_BASE, Resolved};
        let key = (record, backing_of(view));
        let mut kept_now = None;
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let r = map.resolve_with(t, key, |h| {
            note(0);
            let kept = match *view {
                super::View::Constant(address, size) => constant_raw(address, size, at(h))?,
                _ => super::make(view, at(h))?,
            };
            kept_now = Some(kept);
            let class = MADE_CLASS.with(std::cell::Cell::get);
            if class != 0 {
                made(h);
            }
            Some(class)
        });
        if let (Resolved::Mapped { host, fresh: true }, Some(kept)) = (r, kept_now) {
            let mut slots = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            slots[(host - POOL_BASE) as usize] = kept;
        }
        r
    }

    /// The view at title slot `t` (record `record`) is made again: every title slot that
    /// shares its host slot moves to one fresh slot, made as [`Mapped::resolve`] makes it
    /// (the old slot, which a held version may still name, is never rewritten). What `t`
    /// reads now.
    pub(crate) fn remake(
        &self,
        t: u32,
        view: &super::View,
        record: u64,
        at: impl Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE,
        made: impl Fn(u32),
    ) -> super::super::slot_map::Resolved {
        let titles = self.map.lock().unwrap_or_else(|e| e.into_inner()).unkey(t);
        for &other in titles.iter().filter(|&&o| o != t) {
            self.resolve(other, view, record, &at, &made);
        }
        self.resolve(t, view, record, &at, &made)
    }

    /// Title slot `t` holds no view the map can name: it reads a miss.
    pub(crate) fn unmap(&self, t: u32) {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).unmap(t);
    }

    /// The host slot title slot `t` maps to now; `None` for a miss.
    pub(crate) fn host_of(&self, t: u32) -> Option<u32> {
        let e = self.map.lock().unwrap_or_else(|e| e.into_inner()).entry(t);
        (e != 0).then_some(e & ((1 << victory_shader::emit::slot_map::HOST_BITS) - 1))
    }

    /// Whether host slot `host` shows `buffer`; `None` if it holds no host resource
    /// (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, host: u32, buffer: &ID3D12Resource) -> Option<bool> {
        use windows::core::Interface;
        let at = host.checked_sub(super::super::slot_map::POOL_BASE)? as usize;
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let shown = kept.get(at)?.as_ref()?;
        Some(shown.as_raw() == buffer.as_raw())
    }

    /// The version a list is filled with ([`super::super::slot_map::SlotMap::publish`]),
    /// its chunks written into the card's pool; the resources of host slots freed since
    /// are let go.
    pub(crate) fn publish(
        &self,
    ) -> Result<std::sync::Arc<super::super::slot_map::Version>, super::super::slot_map::Full> {
        use super::super::slot_map::POOL_BASE;
        use victory_shader::emit::slot_map::CHUNK_SLOTS;
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let base = self.pool.2 as *mut u32;
        let v = map.publish(&mut |chunk, entries| {
            // SAFETY: inside the mapping (`chunk` is below the pool's chunks); no held
            // version names a chunk the map gives to be written.
            unsafe {
                std::ptr::copy_nonoverlapping(entries.as_ptr(), base.add((chunk * CHUNK_SLOTS) as usize), entries.len());
            }
        });
        let freed = map.take_freed();
        drop(map);
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        for h in freed {
            kept[(h - POOL_BASE) as usize] = None;
        }
        v
    }

    /// The pool's GPU address, for the root SRV `MAP_POOL`.
    pub(crate) fn pool_gpu(&self) -> u64 {
        self.pool.1
    }

    /// The pool's buffer, for a list to hold.
    pub(crate) fn pool_buffer(&self) -> &ID3D12Resource {
        &self.pool.0
    }

    /// Fills a list's directory (`DIR_ENTRIES` dwords of its upload memory at `at`) with
    /// the map as it is now, when the list is submitted, and holds that version in `held`
    /// until the list is reset (a list sent again waits for its earlier run, so the version
    /// before it goes then). With no free chunk for the version (counted), the directory
    /// names chunk 0 alone: every title slot reads its typed null, never a chunk another
    /// held version may still be given.
    pub(crate) fn fill(&self, at: usize, held: &std::sync::Mutex<Option<std::sync::Arc<super::super::slot_map::Version>>>) {
        use victory_shader::emit::slot_map::DIR_ENTRIES;
        let v = self.publish();
        let dst = at as *mut u32;
        match &v {
            // SAFETY: the list's scratch of `DIR_ENTRIES` dwords, alive until it is reset;
            // the list does not run while it is submitted.
            Ok(v) => unsafe { std::ptr::copy_nonoverlapping(v.directory().as_ptr(), dst, DIR_ENTRIES as usize) },
            Err(_) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        chunks = self.chunks,
                        "VICTORY_SLOT_MAP=1: no free chunk of the slot map's pool for a list's version: its shaders read typed nulls"
                    );
                });
                // SAFETY: as above.
                unsafe { std::ptr::write_bytes(dst, 0, DIR_ENTRIES as usize) };
            }
        }
        *held.lock().unwrap_or_else(|e| e.into_inner()) = v.ok();
    }

    pub(crate) fn counts(&self) -> super::super::slot_map::Counts {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).counts()
    }

    /// Whether host slot `host` holds a host resource (tests, diagnostics).
    pub(crate) fn keeps(&self, host: u32) -> bool {
        let at = host.wrapping_sub(super::super::slot_map::POOL_BASE) as usize;
        self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(at).is_some_and(Option::is_some)
    }

    /// The pool's dwords as the card reads them (tests).
    #[cfg(test)]
    fn pool_words(&self, chunks: u32) -> Vec<u32> {
        use victory_shader::emit::slot_map::CHUNK_SLOTS;
        let n = (chunks * CHUNK_SLOTS) as usize;
        // SAFETY: the mapping holds `chunks` chunks; no list writes it.
        unsafe { std::slice::from_raw_parts(self.pool.2 as *const u32, n).to_vec() }
    }
}

/// Under the map, a constant buffer record (`size` bytes at title `address`) is a raw
/// view of its memory's host copy, made at `at` (design § 3; the translator reads such
/// a record by byte, `victory_shader`'s `emit::cbv_raw`). The console starts records on
/// 16 bytes, D3D12 constant buffer views on 256: a raw view starts on 16, and reads past
/// the record's size read zeros, as the console's bound does. `None` (a miss: zeros, counted
/// with the views the map could not make) without a host copy, or off 16 bytes in it.
fn constant_raw(address: u64, size: u32, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<super::Kept> {
    let host = crate::host::host()?;
    let shadow = crate::shadow::find(address, u64::from(size))?;
    // SAFETY: a live buffer.
    let base = unsafe { shadow.buffer.GetGPUVirtualAddress() };
    let offset = shadow.gpu(address) - base;
    if offset % 16 != 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(offset, "VICTORY_SLOT_MAP=1: a constant buffer record off 16 bytes in its host copy reads zeros"));
        return None;
    }
    let len = u64::from(size).min(shadow.left(address));
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: offset / 4,
                NumElements: (len / 4).clamp(1, u64::from(u32::MAX)) as u32,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            },
        },
    };
    // SAFETY: a slot of a live heap; the range lies in the host copy.
    unsafe { host.device.CreateShaderResourceView(&shadow.buffer, Some(&desc), at) };
    note(srv_class(Some(&desc), None));
    Some(Some(shadow.buffer.clone()))
}

/// Makes the typed null of class `class` at `at`: what an access of that class reads
/// for a title slot the map names no view of its class at.
pub(crate) fn typed_null(class: u32, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    #[cfg(test)]
    MADE.with(|m| m.set(m.get() + 1));
    let Some(host) = crate::host::host() else { return };
    let device = &host.device;
    let kind = class & 31;
    // SAFETY (each call): a slot of a live heap; a null resource is allowed with a desc.
    if class & 32 != 0 {
        let desc = match kind {
            TYPED_BUFFER | RAW_BUFFER | STRUCTURED_BUFFER => {
                let raw = kind != TYPED_BUFFER;
                D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: if raw { DXGI_FORMAT_R32_TYPELESS } else { DXGI_FORMAT_R32_UINT },
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: 1,
                            Flags: if raw { D3D12_BUFFER_UAV_FLAG_RAW } else { D3D12_BUFFER_UAV_FLAG_NONE },
                            ..Default::default()
                        },
                    },
                }
            }
            _ => super::null_uav_desc(Some(dimension_of_kind(kind)).filter(|&d| d != 0)),
        };
        unsafe { device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&desc), at) };
        return;
    }
    match kind {
        CBUFFER => unsafe { device.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at) },
        RAW_BUFFER | STRUCTURED_BUFFER => super::null_buffer_view(at),
        TYPED_BUFFER => {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV { NumElements: 1, ..Default::default() },
                },
            };
            unsafe { device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at) };
        }
        ACCELERATION_STRUCTURE => {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: 0 },
                },
            };
            unsafe { device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at) };
        }
        _ => {
            let mut words = [0u32; 10];
            words[1] = dimension_of_kind(kind);
            super::null_like(&words, at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::emit::slot_map::{CLASSES, UAV, class};

    /// DXIL resource kinds the translator declares.
    const TEXTURE2D: u32 = 2;
    const TEXTURE2D_ARRAY: u32 = 7;
    const TEXTURE3D: u32 = 4;
    const TEXTURE_CUBE: u32 = 5;
    const TYPED_BUFFER: u32 = 10;
    const RAW_BUFFER: u32 = 11;
    const STRUCTURED_BUFFER: u32 = 12;

    fn srv(dimension: D3D12_SRV_DIMENSION, format: DXGI_FORMAT, flags: D3D12_BUFFER_SRV_FLAGS, stride: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: dimension,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV { NumElements: 1, StructureByteStride: stride, Flags: flags, ..Default::default() },
            },
        }
    }

    fn uav(dimension: D3D12_UAV_DIMENSION, format: DXGI_FORMAT, flags: D3D12_BUFFER_UAV_FLAGS, stride: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: dimension,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: 1, StructureByteStride: stride, Flags: flags, ..Default::default() },
            },
        }
    }

    fn texture(dimension: D3D12_RESOURCE_DIMENSION, depth_or_array: u16, samples: u32) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: dimension,
            Width: 4,
            Height: 4,
            DepthOrArraySize: depth_or_array,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
            ..Default::default()
        }
    }

    /// The class of each view the host makes is the class of the access the translator
    /// declares for it: a raw buffer view a raw or structured access's, a texture view its
    /// dimension's, unordered views apart from shader views. A structured view the host
    /// could not make raw is a class no access declares: it reads its null, never a
    /// structured descriptor through a raw handle.
    #[test]
    fn a_made_view_has_the_class_of_the_access_that_may_read_it() {
        let none = D3D12_BUFFER_SRV_FLAG_NONE;
        let raw = srv(D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R32_TYPELESS, D3D12_BUFFER_SRV_FLAG_RAW, 0);
        assert_eq!(srv_class(Some(&raw), None), class(RAW_BUFFER));
        assert_eq!(srv_class(Some(&raw), None), class(STRUCTURED_BUFFER));
        let typed = srv(D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R32G32B32A32_FLOAT, none, 0);
        assert_eq!(srv_class(Some(&typed), None), class(TYPED_BUFFER));
        let structured = srv(D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_UNKNOWN, none, 16);
        let apart = srv_class(Some(&structured), None);
        assert!(apart != 0 && apart < CLASSES);
        assert!((0..=18).all(|kind| class(kind) != apart && class(kind | UAV) != apart));
        for (dimension, kind) in [
            (D3D12_SRV_DIMENSION_TEXTURE2D, TEXTURE2D),
            (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, TEXTURE2D_ARRAY),
            (D3D12_SRV_DIMENSION_TEXTURE3D, TEXTURE3D),
            (D3D12_SRV_DIMENSION_TEXTURECUBE, TEXTURE_CUBE),
        ] {
            let desc = srv(dimension, DXGI_FORMAT_R8G8B8A8_UNORM, none, 0);
            assert_eq!(srv_class(Some(&desc), None), class(kind), "{dimension:?}");
        }
        // A texture's own view: its dimension and array size.
        assert_eq!(srv_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1))), class(TEXTURE2D));
        assert_eq!(srv_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 6, 1))), class(TEXTURE2D_ARRAY));
        assert_eq!(srv_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 1))), class(TEXTURE3D));
        assert_eq!(srv_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 4))), class(3));
        let raw_uav = uav(D3D12_UAV_DIMENSION_BUFFER, DXGI_FORMAT_R32_TYPELESS, D3D12_BUFFER_UAV_FLAG_RAW, 0);
        assert_eq!(uav_class(Some(&raw_uav), None), class(RAW_BUFFER | UAV));
        let texture_uav = uav(D3D12_UAV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_UINT, D3D12_BUFFER_UAV_FLAG_NONE, 0);
        assert_eq!(uav_class(Some(&texture_uav), None), class(TEXTURE2D | UAV));
        assert_ne!(uav_class(Some(&texture_uav), None), srv_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1))));
        assert_eq!(uav_class(None, Some(&texture(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 1))), class(TEXTURE3D | UAV));
    }

    /// A placed view of 256 bytes of title buffer memory (64 dwords, dword `i` holding
    /// `tag + i`), structured of dwords as the title writes it, registered as buffer
    /// `id`; with `empty`, of no bytes (the host makes no view of it).
    fn placed(id: usize, tag: u32, empty: bool) -> crate::descriptor::View {
        let (resource, desc) = placed_words(id, tag, empty);
        crate::descriptor::View::Placed(false, resource, desc)
    }

    /// The title's words of [`placed`]'s view: the resource's and the view's.
    fn placed_words(id: usize, tag: u32, empty: bool) -> ([u32; 14], [u32; 16]) {
        let memory: &'static mut [u32] = Box::leak((0..64).map(|i| tag + i).collect::<Vec<u32>>().into_boxed_slice());
        let address = memory.as_ptr() as u64;
        crate::shadow::register(id, address, 256);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = if empty { 0 } else { 256 };
        let mut desc = [0u32; 16];
        desc[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[6] = 64;
        desc[7] = 4;
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        (resource, desc)
    }

    /// A view heap under the map, behind a title heap of 70,000 slots with 8 pool slots:
    /// the typed nulls are made (in slots 0..64), the dynamic ring lies in the host-own
    /// range past them, and title slots past the host heap's end and below it that hold
    /// one view read one pool slot, made once. The view's element stride (4: structured
    /// of dwords) is written at that host slot, never at the title slot. The title
    /// writing a slot to nothing leaves it a miss; the published map says so.
    #[test]
    fn a_mapped_view_heap_puts_nulls_the_ring_and_the_pool_where_the_map_says() {
        use super::super::super::slot_map::{POOL_BASE, TYPED_NULLS};
        use victory_shader::emit::slot_map::{NULL_BASE, lookup};
        let Some(_host) = crate::host::test_host() else {
            return;
        };
        const TITLE: u32 = 70_000;
        let before = MADE.with(std::cell::Cell::get);
        let h = super::super::HostHeap::with_pool(false, TITLE, Some(8)).expect("a view heap");
        assert!(MADE.with(std::cell::Cell::get) - before >= CLASSES, "the typed nulls are made");
        let ring = h.dynamic(1).expect("a dynamic slot");
        assert!((TYPED_NULLS..POOL_BASE).contains(&ring), "the ring is the host's own: slot {ring}");
        // The title's heap memory, and the view's record at two of its slots.
        let memory: &'static mut [u8] = Box::leak(vec![0u8; TITLE as usize * 32].into_boxed_slice());
        let at = |t: u32| memory.as_ptr() as u64 + u64::from(t) * 32;
        let (resource, desc) = placed_words(0x5B01, 0xE00, false);
        for t in [69_000, 5] {
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, at(t));
            assert!(h.mirror(t, at(t), true).is_some(), "title slot {t} holds the view");
        }
        let shared = h.host_slot(69_000).expect("the view's title slot is mapped");
        assert!((POOL_BASE..POOL_BASE + 8).contains(&shared), "a pool slot: {shared}");
        assert_eq!(h.host_slot(5), Some(shared), "one view, one host slot");
        assert_eq!(h.stride_at(shared), 4, "the stride by host slot");
        assert_eq!((h.stride_at(5), h.stride_at(69_000)), (0, 0), "nothing by title slot");
        // The title writes slot 5 to nothing.
        // SAFETY: the record's words, in the test's own memory.
        unsafe { std::ptr::write_bytes(at(5) as *mut u8, 0, 32) };
        assert!(h.mirror(5, at(5), true).is_none());
        assert_eq!(h.host_slot(5), None);
        let mapped = h.mapped.as_ref().expect("the heap's map");
        let v = mapped.publish().expect("a version");
        let pool = mapped.pool_words(mapped.chunks);
        let raw = class(RAW_BUFFER);
        assert_eq!(lookup(v.directory(), &pool, 69_000, raw), shared);
        assert_eq!(lookup(v.directory(), &pool, 5, raw), NULL_BASE + raw);
    }

    /// The host half of the map on the device: title slots past a capped pool (4 host
    /// slots behind 4,096 title slots) holding one view share one pool slot, made once
    /// as a raw view of the buffer's host copy (class 11, the class `make` reports) and
    /// holding that copy; another view takes another slot; a view the host cannot make
    /// takes none; a fifth distinct view finds the pool full. The published pool, as the
    /// card reads it, names those slots, and nothing else.
    #[test]
    fn views_resolve_to_shared_pool_slots_made_once_and_published() {
        use super::super::super::slot_map::{POOL_BASE, Resolved};
        use victory_shader::emit::slot_map::{NULL_BASE, lookup};
        let Some(host) = crate::host::test_host() else {
            return;
        };
        let (a, b, empty) = (placed(0x5A01, 0xA00, false), placed(0x5A02, 0xB00, false), placed(0x5A03, 0, true));
        // SAFETY: plain heap creation and queries.
        let (heap, start, step) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: POOL_BASE + 4,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .expect("a heap");
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            (heap, start, host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize)
        };
        let at = |h: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + h as usize * step };
        let mapped = Mapped::new(4096, 4, 4).expect("a pool on the device");
        let raw = class(RAW_BUFFER);
        let first = mapped.resolve(4000, &a, 0x1_0000_0001, at, |_| {});
        let Resolved::Mapped { host: ha, fresh: true } = first else { panic!("{first:?}") };
        for t in [10, 3000] {
            assert_eq!(mapped.resolve(t, &a, 0x1_0000_0001, at, |_| {}), Resolved::Mapped { host: ha, fresh: false });
        }
        let Resolved::Mapped { host: hb, fresh: true } = mapped.resolve(11, &b, 0x1_0000_0002, at, |_| {}) else { panic!() };
        assert_ne!(ha, hb);
        assert_eq!(mapped.resolve(12, &empty, 0x1_0000_0003, at, |_| {}), Resolved::Unmade);
        assert!(mapped.keeps(ha) && mapped.keeps(hb), "each slot holds its host copy");
        for (k, t) in [(4u64, 20), (5, 21)] {
            assert!(matches!(mapped.resolve(t, &placed(0x5A00 + k as usize, 0xC00, false), 0x1_0000_0000 + k, at, |_| {}), Resolved::Mapped { .. }));
        }
        assert_eq!(mapped.resolve(22, &placed(0x5A06, 0xD00, false), 0x1_0000_0006, at, |_| {}), Resolved::Refused);
        let counts = mapped.counts();
        assert_eq!((counts.live, counts.refused, counts.unmade), (4, 1, 1));
        let v = mapped.publish().expect("a version");
        let pool = mapped.pool_words(4);
        for t in [10, 3000, 4000] {
            assert_eq!(lookup(v.directory(), &pool, t, raw), ha, "title slot {t}");
        }
        assert_eq!(lookup(v.directory(), &pool, 11, raw), hb);
        for t in [12, 22, 9, 4095] {
            assert_eq!(lookup(v.directory(), &pool, t, raw), NULL_BASE + raw, "title slot {t}");
        }
        // Read as another class (a texture), the slot is that class's null.
        assert_eq!(lookup(v.directory(), &pool, 10, class(2)), NULL_BASE + 2);
        // The title rewrites slot 11 to nothing; the next version reads a miss there,
        // the one before still names b's slot (its chunk is not written again).
        mapped.unmap(11);
        let w = mapped.publish().expect("a version");
        let pool = mapped.pool_words(4);
        assert_eq!(lookup(w.directory(), &pool, 11, raw), NULL_BASE + raw);
        assert_eq!(lookup(v.directory(), &pool, 11, raw), hb);
        assert_ne!(mapped.pool_gpu(), 0);
        drop(heap);
        // SAFETY: a plain query.
        unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is not removed");
    }

    /// Every class has a typed null, made in a heap of the 64, and the device takes each
    /// (it is not removed). (What a shader reads through them is the translator's
    /// fixtures' and the acceptance fixture's to show.)
    #[test]
    fn the_typed_nulls_of_every_class_are_made() {
        let Some(host) = crate::host::test_host() else {
            return;
        };
        // SAFETY: plain heap creation; slots of it.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: CLASSES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .expect("a heap");
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            for c in 0..CLASSES {
                typed_null(c, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + c as usize * step });
            }
            assert!(MADE.with(std::cell::Cell::get) >= CLASSES, "a null of every class was made");
            host.device.GetDeviceRemovedReason().expect("the device took every null");
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: typed nulls made on this thread.
    static MADE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
