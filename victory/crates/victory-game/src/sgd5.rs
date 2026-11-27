//! `SGD5`: the engine's shader file. Each record holds one shader container,
//! deflate-packed. The header tables are not decoded yet, so records are
//! found by their marker and cross-checked by their sizes:
//!
//! ```text
//! u8 name length (with NUL), name, NUL, u8 kind,
//! u32 n, u32 0x000DDCE7, u32 n, u32 unpacked size, raw deflate (n - 12 bytes)
//! ```

pub const MAGIC: &[u8; 4] = b"SGD5";
const MARKER: [u8; 4] = 0x000D_DCE7_u32.to_le_bytes();
/// Containers are at most a few MiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: usize = 64 << 20;
/// A string, block or table entry referring to one of the file's shared blocks.
const SHARED: u32 = 0x8000_0000;

#[cfg(test)]
thread_local! {
    /// Pass values the parser built on this thread; the tests' measure of how many a file's tables expand to.
    static PASSES_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Pass slots the parser looked at on this thread; a span shared by many techniques must be walked once.
    static SLOT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_pass() {
    PASSES_BUILT.with(|n| n.set(n.get() + 1));
}

#[cfg(not(test))]
fn note_pass() {}

#[cfg(test)]
fn note_visit() {
    SLOT_VISITS.with(|n| n.set(n.get() + 1));
}

#[cfg(not(test))]
fn note_visit() {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    /// Offset of the marker in the file.
    pub offset: usize,
    /// Like `PS_<name>_<hash>_Wrapped`; `None` if the bytes before the
    /// record do not read as one.
    pub name: Option<&'a str>,
    /// Record type: an opaque byte the file format carries, meaning unknown.
    pub kind: u8,
    packed: &'a [u8],
    unpacked_size: usize,
}

impl Record<'_> {
    /// The container, or `None` if the stream is corrupt.
    pub fn unpack(&self) -> Option<Vec<u8>> {
        let mut state = zlib_rs::Inflate::new(false, 15);
        let mut out = Vec::new();
        let (mut in_pos, mut out_pos) = (0usize, 0usize);
        loop {
            if out_pos == self.unpacked_size {
                let mut extra = [0u8; 1];
                let status = state
                    .decompress(
                        &self.packed[in_pos..],
                        &mut extra,
                        zlib_rs::InflateFlush::NoFlush,
                    )
                    .ok()?;
                return (status == zlib_rs::Status::StreamEnd
                    && state.total_out() as usize == self.unpacked_size)
                    .then_some(out);
            }
            if out_pos == out.len() {
                let next_len = if out.is_empty() {
                    self.unpacked_size
                        .min(self.packed.len().saturating_mul(4).max(4096))
                } else {
                    out.len().saturating_mul(2).min(self.unpacked_size)
                };
                out.resize(next_len, 0);
            }
            let status = state
                .decompress(
                    &self.packed[in_pos..],
                    &mut out[out_pos..],
                    zlib_rs::InflateFlush::NoFlush,
                )
                .ok()?;
            let next_in = state.total_in() as usize;
            let next_out = state.total_out() as usize;
            match status {
                zlib_rs::Status::StreamEnd => {
                    out.truncate(next_out);
                    return (next_out == self.unpacked_size).then_some(out);
                }
                zlib_rs::Status::Ok if next_in != in_pos || next_out != out_pos => {
                    in_pos = next_in;
                    out_pos = next_out;
                }
                zlib_rs::Status::Ok | zlib_rs::Status::BufError => return None,
            }
        }
    }
}

/// An SGD5 file, or a file holding one (an archive: records are found by
/// their marker, so no index is needed).
pub fn is_sgd5(file: &[u8]) -> bool {
    memchr::memmem::find(file, MAGIC).is_some()
}

/// Every record in an SGD5 file, in file order.
pub fn records(file: &[u8]) -> impl Iterator<Item = Record<'_>> {
    memchr::memmem::find_iter(file, &MARKER).filter_map(|at| record(file, at))
}

fn record(file: &[u8], at: usize) -> Option<Record<'_>> {
    let u32_at = |o: usize| {
        file.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    };
    let n = u32_at(at.checked_sub(4)?)?;
    let unpacked_size = u32_at(at + 8)?;
    if u32_at(at + 4)? != n || n < 12 || unpacked_size > MAX_UNPACKED {
        return None;
    }
    let nul = at.checked_sub(6)?;
    Some(Record {
        offset: at,
        name: (file[nul] == 0).then(|| name_before(file, nul)).flatten(),
        kind: file[at - 5],
        packed: file.get(at + 12..at + n)?,
        unpacked_size,
    })
}

/// The printable name that ends at `nul` and follows its length byte.
fn name_before(file: &[u8], nul: usize) -> Option<&str> {
    let run = file[..nul]
        .iter()
        .rposition(|b| !b.is_ascii_graphic())
        .map_or(0, |p| p + 1);
    // From 32 characters on, the length byte is printable too.
    (run.saturating_sub(1)..nul)
        .find(|&l| usize::from(file[l]) == nul - l)
        .and_then(|l| std::str::from_utf8(&file[l + 1..nul]).ok())
}

/// The stage slots, in file order. The vertex slot also holds mesh shaders.
/// Slots 3 and 4 (tessellation) are empty in every database seen.
pub const STAGES: [&str; 6] = ["vertex", "pixel", "geometry", "slot3", "slot4", "compute"];

/// A shader object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shader<'a> {
    pub record: Record<'a>,
    /// The engine's identity of the shader: programs (and the title's
    /// pipeline cache) are keyed by these.
    pub hash: u64,
}

/// A parsed database: its shaders by stage slot and its effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Database<'a> {
    /// The shader objects of each stage slot (`STAGES`), in file order.
    pub shaders: [Vec<Shader<'a>>; 6],
    pub effects: Vec<Effect<'a>>,
    /// File offsets of the magic and of the byte after the last effect.
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect<'a> {
    pub name: &'a str,
    pub techniques: Vec<Technique<'a>>,
    /// The passes the techniques name, parsed once, one slot per pass from
    /// the lowest first to the highest end (a table can name the same passes
    /// from every technique). A slot no technique names is `None`: such a
    /// pass is never validated, so it cannot reject the database.
    pub passes: Vec<Option<Pass<'a>>>,
    /// Every state description the effect holds (its passes pick from
    /// these), as compact descriptions.
    pub rasterizers: Vec<&'a [u8]>,
    pub depth_stencils: Vec<&'a [u8]>,
    pub blends: Vec<&'a [u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Technique<'a> {
    pub name: &'a str,
    /// Its passes are `Effect::passes[first..first + count]`.
    pub first: u32,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass<'a> {
    /// Per stage slot, an index into `Database::shaders` of that slot.
    pub shaders: [Option<usize>; 6],
    /// The state the pass sets, as the engine's compact descriptions
    /// (D3D12 enum values); `None`: the engine's current state.
    pub rasterizer: Option<&'a [u8]>,
    pub depth_stencil: Option<&'a [u8]>,
    pub blend: Option<&'a [u8]>,
    pub stencil_ref: Option<u8>,
}

/// Every database whose tables parse, in file order.
pub fn databases(file: &[u8]) -> Vec<Database<'_>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = memchr::memmem::find(&file[from..], MAGIC).map(|p| from + p) {
        match parse(file, at) {
            Some(db) => {
                from = db.end;
                out.push(db);
            }
            None => from = at + 1,
        }
    }
    out
}

struct Cursor<'a> {
    file: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.file.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(b)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<&'a str> {
        let n = self.u8()?;
        let (&nul, s) = self.bytes(n.into())?.split_last()?;
        (nul == 0).then_some(())?;
        std::str::from_utf8(s).ok()
    }
    /// `u32 n`, then `n` items of `size` bytes.
    fn items(&mut self, size: usize) -> Option<Vec<&'a [u8]>> {
        let n = self.u32()? as usize;
        let all = self.bytes(n.checked_mul(size)?)?;
        Some(all.chunks_exact(size).collect())
    }
    fn u32s(&mut self) -> Option<Vec<u32>> {
        let items = self.items(4)?;
        Some(
            items
                .iter()
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        )
    }
}

fn parse(file: &[u8], start: usize) -> Option<Database<'_>> {
    let mut c = Cursor { file, at: start };
    (c.bytes(4)? == MAGIC).then_some(())?;
    c.u64()?;
    for _ in 0..c.u32()? {
        let size = c.u32()? as usize;
        c.bytes(size)?;
    }
    let size = c.u32()? as usize;
    c.bytes(size)?;
    let mut shaders: [Vec<Shader<'_>>; 6] = Default::default();
    for list in &mut shaders {
        for _ in 0..c.u32()? {
            list.push(shader_entry(&mut c)?);
        }
    }
    let mut effects = Vec::new();
    for _ in 0..c.u32()? {
        effects.push(effect(&mut c, &shaders)?);
    }
    Some(Database {
        shaders,
        effects,
        start,
        end: c.at,
    })
}

fn shader_entry<'a>(c: &mut Cursor<'a>) -> Option<Shader<'a>> {
    let name = c.str()?;
    let kind = c.u8()?;
    let n = c.u32()? as usize;
    let offset = c.at;
    let mut r = Cursor {
        file: c.bytes(n)?,
        at: 0,
    };
    (r.bytes(4)? == MARKER && r.u32()? as usize == n).then_some(())?;
    let unpacked_size = r.u32()? as usize;
    (unpacked_size <= MAX_UNPACKED).then_some(())?;
    let hash = c.u64()?;
    c.bytes(0x90)?;
    let m = c.u32()? as usize;
    c.bytes(m)?;
    Some(Shader {
        record: Record {
            offset,
            name: Some(name),
            kind,
            packed: &r.file[12..],
            unpacked_size,
        },
        hash,
    })
}

fn effect<'a>(c: &mut Cursor<'a>, shaders: &[Vec<Shader<'a>>; 6]) -> Option<Effect<'a>> {
    let name = c.str()?;
    let size = c.u32()? as usize;
    let mut stages: [Vec<u32>; 6] = Default::default();
    for (list, all) in stages.iter_mut().zip(shaders) {
        *list = c.u32s()?;
        list.iter()
            .all(|&i| (i as usize) < all.len())
            .then_some(())?;
    }
    c.bytes(0x50)?;
    let rasterizers = c.items(16)?;
    let depth_stencils = c.items(24)?;
    let blends = c.items(0x44)?;
    c.items(0x24)?;
    let techniques = c.u32s()?;
    let passes = c.items(24)?;
    let extra = c.u32s()?;
    // `size` is what the loader allocates for all this: a check on the parse.
    let mut loaded = stages.iter().map(|s| s.len() + 1).sum::<usize>() * 8
        + techniques.len() * 8
        + passes.len() * 32
        + extra.len() * 4;
    let strings = c.u32()?;
    if strings != SHARED {
        c.bytes(strings as usize)?;
        loaded += strings as usize;
    }
    for _ in 0..2 {
        let n = c.u32()? as usize;
        c.bytes(n)?;
        loaded += n;
    }
    let blocks = c.u32()? & 0x00FF_FFFF;
    if blocks != 0 {
        loaded = loaded.next_multiple_of(8) + blocks as usize * 8;
    }
    for _ in 0..blocks {
        match c.u32()? {
            SHARED => {
                c.u32()?;
            }
            n => {
                c.bytes(n as usize)?;
                loaded += n as usize;
            }
        }
    }
    let n = c.u32()? as usize;
    let names = c.bytes(n)?;
    (loaded + names.len() == size).then_some(())?;

    let state = |list: &[&'a [u8]], i: usize| list.get(i).copied();
    let pass = |p: &[u8]| -> Option<Pass<'a>> {
        note_pass();
        let flags = p[10];
        let mut out = Pass {
            shaders: [None; 6],
            rasterizer: None,
            depth_stencil: None,
            blend: None,
            stencil_ref: (flags & 8 != 0).then_some(p[9]),
        };
        for (slot, list) in stages.iter().enumerate() {
            if p[slot] != 0 {
                out.shaders[slot] = Some(*list.get(usize::from(p[slot]) - 1)? as usize);
            }
        }
        if flags & 1 != 0 {
            out.rasterizer = Some(state(&rasterizers, p[6].into())?);
        }
        if flags & 2 != 0 {
            let i = usize::from(p[7]) + usize::from(flags >> 6 & 1) * 256;
            out.depth_stencil = Some(state(&depth_stencils, i)?);
        }
        if flags & 4 != 0 {
            let i = usize::from(p[8]) + usize::from(flags >> 7) * 256;
            out.blend = Some(state(&blends, i)?);
        }
        Some(out)
    };
    // A technique holds a range of the effect's passes, which are parsed once: a table can name the same passes
    // from every technique, and a pass list built per technique made a small file expand to tens of millions of
    // pass values.
    let descriptors = techniques
        .iter()
        .map(|&t| {
            let at = (t & 0x7FFF) as usize;
            let tail = names.get(at..)?;
            let name = std::str::from_utf8(&tail[..memchr::memchr(0, tail)?]).ok()?;
            let first = (t >> 15 & 0x1FF) as usize;
            let count = (t >> 24) as usize;
            (first + count <= passes.len()).then_some((name, first, count))
        })
        .collect::<Option<Vec<_>>>()?;
    let first = descriptors.iter().map(|&(_, first, _)| first).min().unwrap_or(0);
    let end = descriptors
        .iter()
        .map(|&(_, first, count)| first + count)
        .max()
        .unwrap_or(0);
    passes.get(first..end)?;
    let mut parsed: Vec<Option<Pass<'a>>> = Vec::with_capacity(end - first);
    parsed.resize_with(end - first, || None);
    // Mark the named slots with boundary deltas and walk the span once: techniques may name overlapping ranges,
    // and walking a range per technique visited a shared span once per technique.
    let mut delta = vec![0i32; end - first + 1];
    for &(_, at, count) in &descriptors {
        delta[at - first] += 1;
        delta[at - first + count] -= 1;
    }
    let mut covered = 0i32;
    for (offset, change) in delta.iter().take(end - first).enumerate() {
        covered += change;
        if covered > 0 {
            note_visit();
            parsed[offset] = Some(pass(passes[first + offset])?);
        }
    }
    let techniques = descriptors
        .into_iter()
        .map(|(name, at, count)| Technique {
            name,
            first: (at - first) as u32,
            count: count as u32,
        })
        .collect();
    Some(Effect {
        name,
        techniques,
        passes: parsed,
        rasterizers,
        depth_stencils,
        blends,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_bytes(name: &str, kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; zlib_rs::compress_bound(payload.len())];
        let config = zlib_rs::DeflateConfig {
            window_bits: -15,
            ..zlib_rs::DeflateConfig::new(6)
        };
        let (packed, rc) = zlib_rs::compress_slice(&mut buf, payload, config);
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![0xF4, 0x01, 0, 0, name.len() as u8 + 1];
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0, kind]);
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(packed);
        out
    }

    #[test]
    fn reads_records() {
        let long = format!("PS_{}_Wrapped", "x".repeat(40));
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&[1, 2, 3, 4]);
        file.extend(record_bytes(
            "VS_Blit_0011aabb_Wrapped",
            0x32,
            b"first container",
        ));
        // A marker whose sizes disagree is not a record.
        file.extend_from_slice(&[9, 0, 0, 0, 0xE7, 0xDC, 0x0D, 0, 8, 0, 0, 0]);
        file.extend(record_bytes(&long, 0x41, &[7; 300]));

        assert!(is_sgd5(&file));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, Some("VS_Blit_0011aabb_Wrapped"));
        assert_eq!(found[0].kind, 0x32);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"first container"[..]));
        assert_eq!(found[1].name, Some(long.as_str()));
        assert_eq!(found[1].unpack(), Some(vec![7; 300]));

        // The same file inside an archive.
        let mut archive = vec![0x5A; 0x1000];
        archive.extend_from_slice(&file);
        assert!(is_sgd5(&archive));
        assert_eq!(records(&archive).count(), 2);
        assert!(!is_sgd5(&archive[..0x1000]));
    }

    #[test]
    fn corrupt_stream_does_not_unpack() {
        let mut file = record_bytes("CS_x", 0x3c, &[1; 64]);
        file[27] = 0xFF; // first packed byte: reserved block type
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), None);
    }

    #[test]
    fn handles_empty_and_rejects_mismatched_or_truncated_streams() {
        let empty = record_bytes("CS_empty", 0x3c, &[]);
        assert_eq!(records(&empty).next().unwrap().unpack(), Some(Vec::new()));

        let payload: Vec<u8> = (0..16_384u32)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 5)) as u8)
            .collect();
        let good = record_bytes("PS_big", 0x32, &payload);
        let r = records(&good).next().unwrap();
        assert_eq!(r.unpack(), Some(payload.clone()));

        // Oversized stream (declared unpacked_size smaller than actual output).
        let mut smaller = good.clone();
        smaller[r.offset + 8..r.offset + 12]
            .copy_from_slice(&((payload.len() - 1) as u32).to_le_bytes());
        assert_eq!(records(&smaller).next().unwrap().unpack(), None);

        // Under-filled stream (declared unpacked_size larger than actual output).
        let mut larger = good.clone();
        larger[r.offset + 8..r.offset + 12]
            .copy_from_slice(&((payload.len() + 1) as u32).to_le_bytes());
        assert_eq!(records(&larger).next().unwrap().unpack(), None);

        // Truncated raw deflate stream.
        let mut truncated = good[..good.len() - 1].to_vec();
        let new_n = (truncated.len() - r.offset) as u32;
        truncated[r.offset - 4..r.offset].copy_from_slice(&new_n.to_le_bytes());
        truncated[r.offset + 4..r.offset + 8].copy_from_slice(&new_n.to_le_bytes());
        assert_eq!(records(&truncated).next().unwrap().unpack(), None);
    }

    fn deflated(payload: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; zlib_rs::compress_bound(payload.len())];
        let config = zlib_rs::DeflateConfig {
            window_bits: -15,
            ..zlib_rs::DeflateConfig::new(6)
        };
        let (out, rc) = zlib_rs::compress_slice(&mut buf, payload, config);
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        out.to_vec()
    }

    fn u32s(out: &mut Vec<u8>, values: &[u32]) {
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn str(out: &mut Vec<u8>, s: &str) {
        out.push(s.len() as u8 + 1);
        out.extend_from_slice(s.as_bytes());
        out.push(0);
    }

    /// A database: one vertex, two pixel and one compute shader; one effect
    /// with a graphics technique of two passes and a compute technique.
    fn database() -> Vec<u8> {
        database_with([2 << 24, 5 | 2 << 15 | 1 << 24])
    }

    /// [`database`] with its two technique descriptors given: name offset in the low 15 bits, first pass in the
    /// next 9, pass count in the top 8.
    fn database_with(techniques: [u32; 2]) -> Vec<u8> {
        let mut passes = [[0u8; 24]; 3];
        passes[0][..2].copy_from_slice(&[1, 1]);
        passes[1][..2].copy_from_slice(&[1, 2]);
        passes[1][6] = 0;
        passes[1][8] = 1;
        passes[1][9] = 0x83;
        passes[1][10] = 1 | 4 | 8;
        passes[2][5] = 1;
        database_with_passes(techniques, passes)
    }

    /// [`database_with`] with the three pass records given.
    fn database_with_passes(techniques: [u32; 2], passes: [[u8; 24]; 3]) -> Vec<u8> {
        let mut f = MAGIC.to_vec();
        f.extend_from_slice(&[0; 8]);
        u32s(&mut f, &[1, 4, 0xAA, 0]); // one shared block of 4 bytes, no strings
        let lists: [&[(&str, u8)]; 6] = [
            &[("VS_a", 0x32)],
            &[("PS_a", 0x32), ("PS_b", 0x40)],
            &[],
            &[],
            &[],
            &[("x:CS_c", 0x3c)],
        ];
        for (slot, list) in lists.iter().enumerate() {
            u32s(&mut f, &[list.len() as u32]);
            for (i, (name, kind)) in list.iter().enumerate() {
                let payload = [slot as u8 * 16 + i as u8; 40];
                let packed = deflated(&payload);
                str(&mut f, name);
                f.push(*kind);
                let n = packed.len() as u32 + 12;
                u32s(&mut f, &[n, 0x000D_DCE7, n, payload.len() as u32]);
                f.extend_from_slice(&packed);
                f.extend_from_slice(&(0x1000 + (slot * 16 + i) as u64).to_le_bytes());
                f.extend_from_slice(&[0; 0x90]);
                u32s(&mut f, &[4, 0x0101]);
            }
        }
        u32s(&mut f, &[1]);
        str(&mut f, "fx");
        let mut body = Vec::new();
        // Local lists: vertex [0], pixel [1, 0], compute [0].
        u32s(&mut body, &[1, 0, 2, 1, 0, 0, 0, 0, 1, 0]);
        body.extend_from_slice(&[0; 0x50]);
        u32s(&mut body, &[1]);
        body.extend_from_slice(&[3; 16]);
        u32s(&mut body, &[0, 2]);
        body.extend_from_slice(&[4; 0x44]);
        body.extend_from_slice(&[5; 0x44]);
        u32s(&mut body, &[0]);
        let names = b"draw\0dispatch\0";
        // draw: passes 0-1; dispatch: pass 2.
        u32s(&mut body, &[2, techniques[0], techniques[1]]);
        u32s(&mut body, &[3]);
        body.extend(passes.iter().flatten());
        u32s(&mut body, &[2, 7, 8]);
        u32s(&mut body, &[SHARED, 3]);
        body.extend_from_slice(b"abc");
        u32s(&mut body, &[1]);
        body.push(9);
        // Blocks: one of its own, one shared.
        u32s(&mut body, &[2 | 8 << 24, 6]);
        body.extend_from_slice(&[6; 6]);
        u32s(&mut body, &[SHARED, 0x1234]);
        u32s(&mut body, &[names.len() as u32]);
        body.extend_from_slice(names);
        let loaded: usize = (2 + 3 + 1 + 1 + 1 + 2) * 8 + 2 * 8 + 3 * 32 + 2 * 4 + 3 + 1;
        let loaded = loaded.next_multiple_of(8) + 2 * 8 + 6 + names.len();
        u32s(&mut f, &[loaded as u32]);
        f.extend(body);
        f.push(1);
        f
    }

    #[test]
    fn reads_the_database_tables() {
        let file = database();
        let dbs = databases(&file);
        assert_eq!(dbs.len(), 1);
        let db = &dbs[0];
        assert_eq!(db.end, file.len() - 1);
        assert_eq!(
            db.shaders.iter().map(Vec::len).collect::<Vec<_>>(),
            [1, 2, 0, 0, 0, 1]
        );
        assert_eq!(db.shaders[1][1].record.name, Some("PS_b"));
        assert_eq!(db.shaders[1][1].record.kind, 0x40);
        assert_eq!(db.shaders[1][1].hash, 0x1011);
        assert_eq!(db.shaders[1][1].record.unpack(), Some(vec![17; 40]));

        let fx = &db.effects[0];
        assert_eq!(fx.name, "fx");
        let names: Vec<_> = fx.techniques.iter().map(|t| t.name).collect();
        assert_eq!(names, ["draw", "dispatch"]);
        let passes = |technique: &Technique<'_>| -> Vec<&Pass<'_>> {
            fx.passes[technique.first as usize..technique.first as usize + technique.count as usize]
                .iter()
                .map(|pass| pass.as_ref().expect("a technique names only parsed passes"))
                .collect()
        };
        let draw = passes(&fx.techniques[0]);
        assert_eq!(draw[0].shaders, [Some(0), Some(1), None, None, None, None]);
        assert_eq!(draw[0].blend, None);
        assert_eq!(draw[0].stencil_ref, None);
        assert_eq!(draw[1].shaders[1], Some(0));
        assert_eq!(draw[1].rasterizer, Some(&[3; 16][..]));
        assert_eq!(draw[1].depth_stencil, None);
        assert_eq!(draw[1].blend, Some(&[5; 0x44][..]));
        assert_eq!(draw[1].stencil_ref, Some(0x83));
        let dispatch = passes(&fx.techniques[1]);
        assert_eq!(dispatch[0].shaders, [None, None, None, None, None, Some(0)]);
    }

    #[test]
    fn a_pass_list_shared_by_every_technique_is_parsed_once() {
        // Both techniques name the same three passes: a parser that builds a pass list per technique makes six
        // pass values out of a table that holds three.
        PASSES_BUILT.with(|n| n.set(0));
        let file = database_with([3 << 24, 3 << 24]);
        let dbs = databases(&file);
        assert_eq!(dbs.len(), 1);
        assert_eq!(PASSES_BUILT.with(|n| n.get()), 3);
        let fx = &dbs[0].effects[0];
        assert_eq!(fx.passes.len(), 3);
        assert_eq!(
            fx.techniques.iter().map(|t| (t.first, t.count)).collect::<Vec<_>>(),
            [(0, 3), (0, 3)]
        );
    }

    #[test]
    fn fully_overlapping_techniques_visit_each_pass_once() {
        // Two techniques name the same three passes. A parser that walks each technique's range visits three
        // slots twice; with fully overlapping descriptors that becomes techniques x passes (255 x 255, and tens of
        // millions on a crafted file). The span must be walked once.
        SLOT_VISITS.with(|n| n.set(0));
        let file = database_with([3 << 24, 3 << 24]);
        let dbs = databases(&file);
        assert_eq!(dbs.len(), 1);
        assert_eq!(SLOT_VISITS.with(|n| n.get()), 3);
    }

    #[test]
    fn an_unreferenced_pass_that_does_not_parse_does_not_reject_its_database() {
        // Pass 1 is named by no technique and names a rasterizer the effect does not hold; techniques use passes 0
        // and 2 only.
        let mut passes = [[0u8; 24]; 3];
        passes[0][..2].copy_from_slice(&[1, 1]);
        passes[1][6] = 9;
        passes[1][10] = 1;
        passes[2][5] = 1;
        let file = database_with_passes([1 << 24, 5 | 2 << 15 | 1 << 24], passes);
        let dbs = databases(&file);
        assert_eq!(dbs.len(), 1);
    }

    #[test]
    fn a_bad_table_still_leaves_the_marker_records() {
        let mut file = database();
        // The effect's size no longer matches what its tables hold.
        let at = memchr::memmem::find(&file, b"\x03fx\0").unwrap() + 4;
        file[at] ^= 1;
        assert!(databases(&file).is_empty());
        assert_eq!(records(&file).count(), 4);
    }
}
