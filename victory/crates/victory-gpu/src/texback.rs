//! Textures our shaders write that a guest's own code may read on the CPU.
//!
//! On the console the CPU and the GPU share memory: what a compute pass
//! stores into a texture is in title memory at once, for the title's CPU
//! code to read. Here a texture lives in its host copy on the card, and a
//! shader-written one is never copied back, so the title reads whatever its
//! memory last held. The port follows the copy-back of the CBJ fork's commit
//! `23925a4aa` (itself from upstream work by other authors), written as our
//! own code; the mapping of that commit onto this tree is
//! `jobs/2026-11-26-texture-back/design.md` in the outer repository.
//!
//! This is its first half (J1), the plain data flow:
//! - the switch `VICTORY_TEXTURE_BACK` ([`kinds`]): unset, empty or `0`
//!   off; otherwise a comma-separated list of kinds `WIDTHxHEIGHTxFORMAT`
//!   (FORMAT a DXGI format number). There is no default list: `1` is not a
//!   kind, and the kinds a run wants are given to it from outside;
//! - selection ([`wanted`], [`wanted_shape`]): 2D, single-sample,
//!   single-plane textures of a listed kind;
//! - [`record`]: a host list at its close copies every subresource of such
//!   a texture into one read-back buffer of ours, leaving the texture's state
//!   in the list as it found it;
//! - [`retile`]: the read-back rows into the title's (console) layout for
//!   that texture, in a caller's span of bytes.
//!
//! Nothing lands, and nothing is wired: nothing calls [`record`] from
//! `HostList::close` and nothing stores into title memory, so with the
//! switch set the runtime does exactly what it does with it unset. The
//! landing into title memory (lock order, the memory watch's ordering, late
//! flights, staleness when the title releases or remaps the memory) is J2,
//! with a reviewed design of its own (the mapping note's sections 3 and 7).

// Nothing outside the tests calls this module until J2 wires it.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::OnceLock;

use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;
use crate::object::This;

/// The switch's name.
pub(crate) const SWITCH: &str = "VICTORY_TEXTURE_BACK";

/// A kind of texture copied back: width and height in texels, and the DXGI
/// format number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Kind {
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) format: u32,
}

/// What a switch value says: its kinds, in order and without repeats, and
/// the entries that are not a kind, as written.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Parsed {
    pub(crate) kinds: Vec<Kind>,
    pub(crate) malformed: Vec<String>,
}

/// The switch's value, read: empty (after trimming) or `0` is off, no
/// kinds; otherwise each comma-separated entry is a kind
/// `WIDTHxHEIGHTxFORMAT` of three decimal numbers above zero, or malformed.
/// Empty entries (a trailing comma) are passed over. `1` is malformed like
/// any other entry that is not a kind.
pub(crate) fn parse(value: &str) -> Parsed {
    let value = value.trim();
    let mut parsed = Parsed::default();
    if value.is_empty() || value == "0" {
        return parsed;
    }
    for entry in value.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        match kind(entry) {
            Some(k) if !parsed.kinds.contains(&k) => parsed.kinds.push(k),
            Some(_) => {}
            None => parsed.malformed.push(entry.to_owned()),
        }
    }
    parsed
}

/// One entry as a kind.
fn kind(entry: &str) -> Option<Kind> {
    fn number<T: std::str::FromStr + PartialEq + Default>(text: &str) -> Option<T> {
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        text.parse().ok().filter(|n| *n != T::default())
    }
    let mut parts = entry.split('x');
    let (w, h, f) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    Some(Kind {
        width: number(w)?,
        height: number(h)?,
        format: number(f)?,
    })
}

/// The kinds copied back, from the switch, read once. Each malformed entry
/// is said once and skipped; the kinds in force are said once when there
/// are any.
pub(crate) fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let Some(value) = std::env::var_os(SWITCH) else {
            return Vec::new();
        };
        let parsed = parse(&value.to_string_lossy());
        for entry in &parsed.malformed {
            let hint = if entry == "1" { " (there is no default list: name the kinds)" } else { "" };
            tracing::warn!(
                switch = SWITCH,
                entry = %entry,
                "not a texture kind WIDTHxHEIGHTxFORMAT{hint}: skipped"
            );
        }
        if !parsed.kinds.is_empty() {
            tracing::info!(
                switch = SWITCH,
                kinds = ?parsed.kinds,
                "texture kinds copied back when shaders write them (recorded only: nothing lands yet)"
            );
        }
        parsed.kinds
    })
}

/// Whether the switch names any kind.
pub(crate) fn on() -> bool {
    !kinds().is_empty()
}

/// What selection asks of a texture's description
/// ([`crate::resource::texture_kind`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    /// A 2D texture (not 1D, not 3D).
    pub(crate) two_d: bool,
    pub(crate) width: u64,
    pub(crate) height: u32,
    /// 1 for single-sample.
    pub(crate) samples: u32,
    /// Planes of its console layout (0: none).
    pub(crate) planes: usize,
    pub(crate) format: u32,
}

/// Whether a texture of `shape` is copied back with `kinds` in force: 2D,
/// single-sample, of one plane, and of a listed kind.
pub(crate) fn wanted_shape(kinds: &[Kind], shape: &Shape) -> bool {
    let kind = Kind {
        width: shape.width,
        height: shape.height,
        format: shape.format,
    };
    shape.two_d && shape.samples == 1 && shape.planes == 1 && kinds.contains(&kind)
}

/// Whether title texture `resource` is copied back. With the switch off
/// nothing of the resource is asked.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|shape| wanted_shape(kinds, &shape))
}

/// Where one subresource's rows lie in a read-back buffer (from its placed
/// footprint): the byte offset, the bytes from one row to the next, the rows
/// (of elements: of blocks for block compression) and the depth slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rows {
    pub(crate) offset: u64,
    pub(crate) row_pitch: u32,
    pub(crate) rows: u32,
    pub(crate) depth: u32,
}

impl Rows {
    /// The bytes of the subresource, every row pitched.
    fn bytes(&self) -> Option<u64> {
        u64::from(self.row_pitch)
            .checked_mul(u64::from(self.rows))?
            .checked_mul(u64::from(self.depth))
    }
}

/// Why rows could not be re-tiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Retile {
    /// Not one entry of rows for each mip of each array slice of the layout.
    Count { rows: usize, mips: u32, slices: u32 },
    /// The span is smaller than the layout.
    Span { span: usize, layout: u64 },
    /// Subresource `subresource`'s rows lie past the end of the read-back
    /// bytes.
    Short { subresource: usize },
    /// The tiler refused subresource `subresource`.
    Tile {
        subresource: usize,
        error: victory_formats::texture::Error,
    },
    /// The read-back buffer could not be mapped.
    Map,
}

/// The read-back bytes `linear`, laid out as `rows` says (one entry per
/// subresource in D3D12's order: mip `i % mips` of array slice
/// `i / mips`), tiled into `span` in `layout`: the title's layout of the
/// texture, in the bytes the title has for it. Bytes of `span` no
/// subresource covers are left as they are.
pub(crate) fn retile(
    layout: &Layout,
    rows: &[Rows],
    mips: u32,
    linear: &[u8],
    span: &mut [u8],
) -> Result<(), Retile> {
    let slices = layout.slices;
    if mips != layout.desc.mips || rows.len() as u64 != u64::from(mips) * u64::from(slices) {
        return Err(Retile::Count {
            rows: rows.len(),
            mips,
            slices,
        });
    }
    if (span.len() as u64) < layout.size {
        return Err(Retile::Span {
            span: span.len(),
            layout: layout.size,
        });
    }
    for (i, r) in rows.iter().enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let bytes = r
            .bytes()
            .and_then(|b| r.offset.checked_add(b))
            .filter(|&end| end <= linear.len() as u64)
            .ok_or(Retile::Short { subresource: i })?;
        let from = &linear[r.offset as usize..bytes as usize];
        let pitch = r.row_pitch as usize;
        layout
            .tile(span, mip, slice, from, pitch, pitch * r.rows as usize)
            .map_err(|error| Retile::Tile {
                subresource: i,
                error,
            })?;
    }
    Ok(())
}

/// A texture's subresources copied into a read-back buffer of ours by a
/// list ([`record`]), and what re-tiling them needs. It must live until the
/// list has run: the list writes the buffer.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    /// The buffer's bytes.
    bytes: u64,
    rows: Vec<Rows>,
    mips: u32,
    layout: Layout,
    /// The texture's title memory: address and bytes. Not written here.
    title: (usize, usize),
}

impl Back {
    /// The bytes of the read-back buffer (what the copy costs on the card).
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The texture's title memory when it was recorded: address and bytes.
    pub(crate) fn title_span(&self) -> (usize, usize) {
        self.title
    }

    /// Once the list that recorded it has run: its texels tiled into `span`
    /// ([`retile`]), a buffer of the caller's. Title memory is not touched.
    pub(crate) fn retile_into(&self, span: &mut [u8]) -> Result<(), Retile> {
        let mut data = std::ptr::null_mut();
        // SAFETY: our read-back buffer, written by a list that has run.
        if unsafe { self.buffer.Map(0, None, Some(&mut data)) }.is_err() || data.is_null() {
            return Err(Retile::Map);
        }
        // SAFETY: the buffer's mapping of `bytes` bytes, read while mapped.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), self.bytes as usize) };
        let done = retile(&self.layout, &self.rows, self.mips, linear, span);
        // SAFETY: mapped above; nothing was written.
        unsafe {
            self.buffer
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }))
        };
        done
    }
}

/// Says once why a texture was not recorded.
fn refused(why: &'static str) -> Option<Back> {
    static SAID: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if !said.contains(&why) {
        said.push(why);
        tracing::warn!(
            switch = SWITCH,
            "a texture to copy back was not recorded: {why}"
        );
    }
    None
}

/// In `list`, still open (at its close): copies every subresource (all
/// mips of all array slices) of title texture `resource`'s host copy into
/// one new read-back buffer, sized from the card's copyable footprints with
/// every row pitched. The texture is moved to the copy-source state for the
/// copies and back to the state the list had it in; a list that has not
/// used the texture before starts and ends with it in copy source, which
/// its prelude brings about at submission. `None`, with one line said, for
/// a texture that is not 2D and single-sample, has no single-plane console
/// layout, has a colour twin (its unordered access writes the twin) or no
/// host copy, or whose host copy's levels and slices differ from its
/// layout's. Selection is the caller's ([`wanted`]).
pub(crate) fn record(list: &HostList, resource: This) -> Option<Back> {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    const COPY_SOURCE: u32 = 0x800;
    let host = crate::host::host()?;
    let Some((layout, at, size)) = crate::resource::title_texture(resource) else {
        return refused("no console layout of one plane");
    };
    if crate::resource::twin(resource).is_some() {
        return refused("its unordered access writes a colour twin");
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return refused("no host copy");
    };
    let texture = &copy.texture;
    // SAFETY: a plain query of a live texture.
    let desc = unsafe { texture.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1 {
        return refused("not a 2D single-sample texture");
    }
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = u32::from(desc.DepthOrArraySize.max(1));
    if mips != layout.desc.mips || slices != layout.slices {
        return refused("its host copy's levels or slices are not its layout's");
    }
    let n = mips * slices;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut counts = vec![0u32; n as usize];
    // SAFETY: out arrays of `n` entries each.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            n,
            0,
            Some(footprints.as_mut_ptr()),
            Some(counts.as_mut_ptr()),
            None,
            None,
        );
    }
    let rows: Vec<Rows> = footprints
        .iter()
        .zip(&counts)
        .map(|(f, &rows)| Rows {
            offset: f.Offset,
            row_pitch: f.Footprint.RowPitch,
            rows,
            depth: f.Footprint.Depth,
        })
        .collect();
    // The total the card gives can leave out the last row's padding; the
    // tiler reads every row pitched.
    let Some(bytes) = rows
        .iter()
        .map(|r| r.bytes().and_then(|b| r.offset.checked_add(b)))
        .try_fold(0u64, |most, end| end.map(|end| most.max(end)))
    else {
        return refused("its footprints overflow");
    };
    let buffer = host.read_back_buffer(bytes)?;
    // To copy source for the copies, and back after them.
    let moved = list.transition(resource, COPY_SOURCE);
    if let Some((texture, before)) = moved {
        let after = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(texture, State(before as i32), State(after as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture is in copy source, the buffer
        // (made in copy dest) holds every footprint.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    if let Some((_, before)) = moved {
        crate::list::ensure_state(list, resource, before);
    }
    Some(Back {
        buffer,
        bytes,
        rows,
        mips,
        layout,
        title: (at, size),
    })
}

#[cfg(test)]
mod tests;
