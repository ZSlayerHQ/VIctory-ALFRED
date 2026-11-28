//! Pictures the host draws on the processor and places on the swap chain's
//! buffer (the settings menu, `menu.rs`; the boot video, `bootvideo.rs`): the
//! buffers on their way to the card, the copy or blend out of them, the writing of
//! a picture's pixels in the swap chain's format, and a box filter.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{Host, HostList, transition};

mod blend;
pub(super) use blend::Blend;

/// A picture's way to the swap chain's buffer: upload buffers written in
/// turn, and the card's buffer each is copied into, which the back buffer is
/// copied from at every present.
pub(super) struct Buffers {
    pub(super) format: DXGI_FORMAT,
    pub(super) size: (u32, u32),
    pub(super) pitch: u32,
    pub(super) uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    pub(super) turn: usize,
    pub(super) alpha: bool,
    fresh: bool,
    /// The card's buffer was copied into before (it is in the copy-source state;
    /// at first it is in the common state, promoted by the first copy).
    used: bool,
}

// SAFETY: free-threaded resources, and memory only written under the presenter's lock.
unsafe impl Send for Buffers {}

impl Buffers {
    /// `uploads` upload buffers of a picture `width` by `height` of `format`,
    /// written in turn: one is written again `uploads` pictures later, when the
    /// card has long read it (a picture is written at most once a present).
    pub(super) fn new(host: &Host, format: DXGI_FORMAT, (width, height): (u32, u32), uploads: usize) -> Option<Buffers> {
        Self::overlay(host, format, (width, height), uploads, false)
    }

    pub(super) fn overlay(host: &Host, format: DXGI_FORMAT, (width, height): (u32, u32), uploads: usize, alpha: bool) -> Option<Buffers> {
        let native = pixel_bytes(format)?;
        let bytes_per_pixel = if alpha && format != DXGI_FORMAT_R16G16B16A16_FLOAT { 4 } else { native };
        let pitch = (width * bytes_per_pixel).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let uploads = (0..uploads).map(|_| host.upload_buffer(bytes)).collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(Buffers { format, size: (width, height), pitch, uploads, card: card?, turn: 0, alpha, fresh: false, used: false })
    }

    /// The next upload buffer, to be written whole; copied to the card's at the next copy.
    pub(super) fn next(&mut self) -> &mut [u8] {
        self.turn = (self.turn + 1) % self.uploads.len();
        self.fresh = true;
        let len = self.pitch as usize * self.size.1 as usize;
        // SAFETY: a mapped upload buffer of this length, which the card read frames ago.
        unsafe { std::slice::from_raw_parts_mut(self.uploads[self.turn].1, len) }
    }

    /// The bytes they hold (on the card or for it).
    pub(super) fn bytes(&self) -> u64 {
        u64::from(self.pitch) * u64::from(self.size.1) * (self.uploads.len() as u64 + 1)
    }

    pub(super) fn write_canvas(&mut self, canvas: &[u8], size: (u32, u32)) {
        let (alpha, format, pitch) = (self.alpha, self.format, self.pitch as usize);
        let out = self.next();
        if alpha && format == DXGI_FORMAT_R16G16B16A16_FLOAT {
            for (from, to) in canvas.chunks_exact(size.0 as usize * 4).zip(out.chunks_mut(pitch)) {
                write_float_row(from, to, true);
            }
        } else {
            write_out(canvas, size, if alpha { DXGI_FORMAT_R8G8B8A8_UNORM } else { format }, out, pitch);
        }
    }

    pub(super) fn draw(&mut self, blend: &Blend, host: &Host, list: &HostList, back: &ID3D12Resource, part: (u32, u32), at: (u32, u32)) {
        if !self.alpha {
            self.copy_out(list, back, part, at);
        } else if let Some(pipeline) = blend.pipeline() {
            pipeline.draw(host, list, self, back, part, at);
        }
    }

    /// Records the copy of their top left `part` onto `back` at `at` (the back
    /// buffer in the copy-destination state), and before it that of a fresh picture to the card.
    pub(super) fn copy_out(&mut self, list: &HostList, back: &ID3D12Resource, part: (u32, u32), at: (u32, u32)) {
        // Our own copies are not part of a recorded frame (the recorder is in diagnostics builds).
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let list = &list.list;
        // SAFETY: an open list and live resources; the part lies within both.
        unsafe {
            if std::mem::take(&mut self.fresh) {
                if std::mem::replace(&mut self.used, true) {
                    list.ResourceBarrier(&[transition(&self.card, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
                }
                list.CopyResource(&self.card, &self.uploads[self.turn].0);
                list.ResourceBarrier(&[transition(&self.card, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: self.format,
                            Width: self.size.0,
                            Height: self.size.1,
                            Depth: 1,
                            RowPitch: self.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX { left: 0, top: 0, front: 0, right: part.0, bottom: part.1, back: 1 };
            list.CopyTextureRegion(&to, at.0, at.1, 0, &from, Some(&part));
        }
    }
}

/// Bytes a pixel of `format` takes, among those [`write_row`] writes.
pub(super) fn pixel_bytes(format: DXGI_FORMAT) -> Option<u32> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// A half-precision float's bits, for values from 0 to 1.
fn half(v: f32) -> u16 {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x7F_FFFF;
    if v <= 0.0 || exp < -10 {
        return 0;
    }
    if exp <= 0 {
        let m = mantissa | 0x80_0000;
        let shift = (14 - exp) as u32;
        return ((m >> shift) + ((m >> (shift - 1)) & 1)) as u16;
    }
    ((exp as u32) << 10 | mantissa >> 13) as u16 + ((mantissa >> 12) & 1) as u16
}

/// The light of an sRGB byte (for the floating-point swap chain, whose values are linear).
fn linear(v: u8) -> f32 {
    let v = f32::from(v) / 255.0;
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn write_float_row(from: &[u8], to: &mut [u8], alpha: bool) {
    static HALVES: std::sync::OnceLock<[[u8; 2]; 256]> = std::sync::OnceLock::new();
    let halves = HALVES.get_or_init(|| std::array::from_fn(|v| half(linear(v as u8)).to_le_bytes()));
    let one = half(1.0).to_le_bytes();
    for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(8)) {
        q[..2].copy_from_slice(&halves[usize::from(p[0])]);
        q[2..4].copy_from_slice(&halves[usize::from(p[1])]);
        q[4..6].copy_from_slice(&halves[usize::from(p[2])]);
        let coverage = if alpha { half(f32::from(p[3]) / 255.0).to_le_bytes() } else { one };
        q[6..].copy_from_slice(&coverage);
    }
}

/// Writes a row of pixels `from` (four bytes each, red first) into `to` as
/// pixels of `format`. False for a format it does not write.
pub(super) fn write_row(from: &[u8], format: DXGI_FORMAT, to: &mut [u8]) -> bool {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => to[..from.len()].copy_from_slice(from),
        DXGI_FORMAT_B8G8R8A8_UNORM => {
            for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(4)) {
                q.copy_from_slice(&[p[2], p[1], p[0], 0xFF]);
            }
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let ten = |v: u8| (u32::from(v) * 1023 + 127) / 255;
            for (p, q) in from.chunks_exact(4).zip(to.chunks_exact_mut(4)) {
                q.copy_from_slice(&(ten(p[0]) | ten(p[1]) << 10 | ten(p[2]) << 20 | 3 << 30).to_le_bytes());
            }
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => write_float_row(from, to, false),
        _ => return false,
    }
    true
}

/// Writes `canvas` (`size`, four bytes a pixel, red first) into `out`, rows
/// `pitch` bytes apart, as pixels of `format`. False for a format it does not write.
pub(super) fn write_out(canvas: &[u8], size: (u32, u32), format: DXGI_FORMAT, out: &mut [u8], pitch: usize) -> bool {
    let (w, h) = (size.0 as usize, size.1 as usize);
    canvas.chunks_exact(w * 4).take(h).zip(out.chunks_mut(pitch)).all(|(from, to)| write_row(from, format, to))
}

/// A filtered RGB or straight-alpha RGBA image as a four-byte canvas.
pub(super) fn rgba(pixels: Vec<u8>, channels: usize) -> Vec<u8> {
    if channels == 4 { pixels } else { pixels.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect() }
}

/// A box filter: `src` (`sw` by `sh`, `n` bytes a pixel) to `dw` by `dh`,
/// each pixel the mean of the area of the source it covers, `step` source
/// pixels to one on each axis (the area past the source's edge counts as
/// nothing, and is not counted). Four-channel images are straight RGBA;
/// the colour average is weighted by coverage.
pub(super) fn boxed(src: &[u8], (sw, sh): (usize, usize), n: usize, (dw, dh): (usize, usize), step: (f32, f32)) -> Vec<u8> {
    if n == 4 { box_pixels::<true>(src, (sw, sh), n, (dw, dh), step) } else { box_pixels::<false>(src, (sw, sh), n, (dw, dh), step) }
}

fn box_pixels<const ALPHA: bool>(src: &[u8], (sw, sh): (usize, usize), n: usize, (dw, dh): (usize, usize), step: (f32, f32)) -> Vec<u8> {
    // For each pixel on an axis: the first source pixel it covers, and how much of each.
    let spans = |d: usize, s: usize, step: f32| -> Vec<(usize, Vec<f32>)> {
        (0..d)
            .map(|i| {
                let (lo, hi) = (i as f32 * step, ((i + 1) as f32 * step).min(s as f32));
                if hi <= lo {
                    return (0, Vec::new());
                }
                let first = lo.floor() as usize;
                let weights: Vec<f32> =
                    (first..(hi.ceil() as usize).min(s)).map(|j| (hi.min(j as f32 + 1.0) - lo.max(j as f32)).max(0.0)).collect();
                let total: f32 = weights.iter().sum();
                (first, weights.into_iter().map(|w| w / total).collect())
            })
            .collect()
    };
    let (across, down) = (spans(dw, sw, step.0), spans(dh, sh, step.1));
    let mut wide = vec![0f32; sh * dw * n];
    for y in 0..sh {
        let row = &src[y * sw * n..(y + 1) * sw * n];
        let out = &mut wide[y * dw * n..(y + 1) * dw * n];
        for (x, (first, weights)) in across.iter().enumerate() {
            for (j, w) in weights.iter().enumerate() {
                let at = (first + j) * n;
                if ALPHA {
                    let a = f32::from(row[at + 3]);
                    for c in 0..3 {
                        out[x * n + c] += f32::from(row[at + c]) * (a / 255.0) * w;
                    }
                    out[x * n + 3] += a * w;
                } else {
                    for c in 0..n {
                        out[x * n + c] += f32::from(row[at + c]) * w;
                    }
                }
            }
        }
    }
    let mut out = vec![0u8; dh * dw * n];
    let mut sum = vec![0f32; dw * n];
    for (y, (first, weights)) in down.iter().enumerate() {
        sum.iter_mut().for_each(|s| *s = 0.0);
        for (j, w) in weights.iter().enumerate() {
            let row = &wide[(first + j) * dw * n..(first + j + 1) * dw * n];
            for (s, v) in sum.iter_mut().zip(row) {
                *s += v * w;
            }
        }
        if ALPHA {
            for px in sum.chunks_exact_mut(4) {
                if px[3] > 0.0 && px[3] < 255.0 - 0.0001 {
                    let scale = 255.0 / px[3];
                    for c in &mut px[..3] {
                        *c *= scale;
                    }
                }
            }
        }
        for (o, s) in out[y * dw * n..(y + 1) * dw * n].iter_mut().zip(&sum) {
            *o = (s + 0.5).min(255.0) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_of_what_is_written_are_exact() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(1.0), 0x3C00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(0.25), 0x3400);
        // A small one, below the normal range: 2^-20.
        assert_eq!(half(1.0 / 1_048_576.0), 0x0010);
        assert_eq!(half(linear(255)), 0x3C00);
    }

    #[test]
    fn every_format_is_written_as_its_pixels() {
        let canvas = [10u8, 128, 255, 255, 0, 0, 0, 255];
        let mut out = vec![0u8; 2 * 8 + 3];
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R8G8B8A8_UNORM, &mut out, 19));
        assert_eq!(&out[..8], &canvas);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_B8G8R8A8_UNORM, &mut out, 19));
        assert_eq!(&out[..4], &[255, 128, 10, 255]);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R10G10B10A2_UNORM, &mut out, 19));
        assert_eq!(u32::from_le_bytes([out[0], out[1], out[2], out[3]]), 40 | 514 << 10 | 1023 << 20 | 3 << 30);
        assert!(write_out(&canvas, (2, 1), DXGI_FORMAT_R16G16B16A16_FLOAT, &mut out, 19));
        assert_eq!(&out[4..8], &[0x00, 0x3C, 0x00, 0x3C]);
        assert_eq!(&out[8..16], &[0, 0, 0, 0, 0, 0, 0x00, 0x3C]);
        assert!(!write_out(&canvas, (2, 1), DXGI_FORMAT_R32G32B32A32_FLOAT, &mut out, 19));
    }

    #[test]
    fn the_box_filter_keeps_a_flat_colour_and_averages_an_edge() {
        let flat = vec![77u8; 10 * 6 * 3];
        assert!(boxed(&flat, (10, 6), 3, (4, 3), (2.5, 2.0)).iter().all(|v| *v == 77));
        // Black then white halves, halved: a grey column where they meet, at 1.5 pixels a pixel.
        let edge = [0u8, 0, 0, 255, 255, 255];
        assert_eq!(boxed(&edge, (6, 1), 1, (4, 1), (1.5, 1.0)), [0, 0, 255, 255]);
        let mid: Vec<u8> = [0, 0, 0, 0, 255, 255, 255, 255].to_vec();
        assert_eq!(boxed(&mid, (8, 1), 1, (3, 1), (8.0 / 3.0, 1.0)), [0, 128, 255]);
    }

    #[test]
    fn the_box_filter_weights_colour_by_coverage() {
        let pixels = [40, 30, 50, 0, 255, 128, 224, 255, 255, 128, 224, 128, 40, 30, 50, 0];
        assert_eq!(boxed(&pixels, (2, 2), 4, (1, 1), (2.0, 2.0)), [255, 128, 224, 96]);
    }
}
