//! QOI images ("Quite OK Image", qoiformat.org): a decoder of our own, for
//! the settings menu's art (`menu/art.rs`) and the boot video's frames (`bootvideo.rs`). Three or four bytes a pixel.

/// A decoded image: its size, and its pixels top row first, `channels` bytes each.
pub(super) struct Image {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channels: usize,
    pub(super) pixels: Vec<u8>,
}

/// The most pixels an image of ours has room for (the panel is 1600 by 1200).
const MOST: usize = 1 << 24;

fn slot(p: [u8; 4]) -> usize {
    (usize::from(p[0]) * 3 + usize::from(p[1]) * 5 + usize::from(p[2]) * 7 + usize::from(p[3]) * 11) % 64
}

/// The image in `data`; `None` for anything not a whole QOI image of at most [`MOST`] pixels.
pub(super) fn decode(data: &[u8]) -> Option<Image> {
    let header = data.get(..14)?;
    if &header[..4] != b"qoif" {
        return None;
    }
    let width = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let height = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let channels = usize::from(header[12]);
    let count = (width as usize).checked_mul(height as usize).filter(|n| *n <= MOST)?;
    if !matches!(channels, 3 | 4) {
        return None;
    }
    let mut pixels = Vec::with_capacity(count * channels);
    let mut seen = [[0u8; 4]; 64];
    let mut px = [0, 0, 0, 255u8];
    let mut at = 14;
    let mut left = count;
    while left > 0 {
        let tag = *data.get(at)?;
        at += 1;
        let mut run = 1;
        match tag {
            0xFE => {
                px[..3].copy_from_slice(data.get(at..at + 3)?);
                at += 3;
            }
            0xFF => {
                px.copy_from_slice(data.get(at..at + 4)?);
                at += 4;
            }
            _ => match tag >> 6 {
                0 => px = seen[usize::from(tag)],
                1 => {
                    px[0] = px[0].wrapping_add(((tag >> 4) & 3).wrapping_sub(2));
                    px[1] = px[1].wrapping_add(((tag >> 2) & 3).wrapping_sub(2));
                    px[2] = px[2].wrapping_add((tag & 3).wrapping_sub(2));
                }
                2 => {
                    let next = *data.get(at)?;
                    at += 1;
                    let dg = (tag & 63).wrapping_sub(32);
                    px[0] = px[0].wrapping_add(dg).wrapping_add(next >> 4).wrapping_sub(8);
                    px[1] = px[1].wrapping_add(dg);
                    px[2] = px[2].wrapping_add(dg).wrapping_add(next & 15).wrapping_sub(8);
                }
                _ => run = usize::from(tag & 63) + 1,
            },
        }
        seen[slot(px)] = px;
        let run = run.min(left);
        for _ in 0..run {
            pixels.extend_from_slice(&px[..channels]);
        }
        left -= run;
    }
    Some(Image { width, height, channels, pixels })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A QOI encoder (the format's every operation), to make images for tests.
    pub(in crate::host) fn encode(width: u32, height: u32, channels: usize, pixels: &[u8]) -> Vec<u8> {
        let mut out = b"qoif".to_vec();
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&[channels as u8, 0]);
        let mut seen = [[0u8; 4]; 64];
        let mut was = [0, 0, 0, 255u8];
        let mut run = 0;
        for p in pixels.chunks_exact(channels) {
            let px = [p[0], p[1], p[2], if channels == 4 { p[3] } else { 255 }];
            if px == was {
                run += 1;
                if run == 62 {
                    out.push(0xC0 | 61);
                    run = 0;
                }
                continue;
            }
            if run > 0 {
                out.push(0xC0 | (run - 1));
                run = 0;
            }
            let i = slot(px);
            if seen[i] == px {
                out.push(i as u8);
            } else {
                seen[i] = px;
                let d = |a: u8, b: u8| i16::from(a.wrapping_sub(b) as i8);
                let (dr, dg, db) = (d(px[0], was[0]), d(px[1], was[1]), d(px[2], was[2]));
                if px[3] != was[3] {
                    out.extend_from_slice(&[0xFF, px[0], px[1], px[2], px[3]]);
                } else if (-2..=1).contains(&dr) && (-2..=1).contains(&dg) && (-2..=1).contains(&db) {
                    out.push(0x40 | ((dr + 2) << 4 | (dg + 2) << 2 | (db + 2)) as u8);
                } else if (-32..=31).contains(&dg) && (-8..=7).contains(&(dr - dg)) && (-8..=7).contains(&(db - dg)) {
                    out.push(0x80 | (dg + 32) as u8);
                    out.push(((dr - dg + 8) << 4 | (db - dg + 8)) as u8);
                } else {
                    out.extend_from_slice(&[0xFE, px[0], px[1], px[2]]);
                }
            }
            was = px;
        }
        if run > 0 {
            out.push(0xC0 | (run - 1));
        }
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
        out
    }

    /// A small image that takes every operation: long runs, repeats, small and larger steps, jumps.
    fn made(width: u32, height: u32, channels: usize) -> Vec<u8> {
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let p: [u8; 4] = match (x / 8 + y) % 5 {
                    0 => [10, 20, 30, 255],
                    1 => [x as u8, (x + 1) as u8, (x / 2) as u8, 255],
                    2 => [(x * 9) as u8, (x * 13 + y) as u8, (x * 7) as u8, 255],
                    3 => [(x * 37 + y * 101) as u8, (x * 59) as u8, (y * 83 + x) as u8, (x * 3) as u8],
                    _ => [(x % 3) as u8 * 100, 0, 255, 128],
                };
                pixels.extend_from_slice(&p[..channels]);
            }
        }
        pixels
    }

    #[test]
    fn an_image_made_here_reads_back_exactly() {
        for channels in [3, 4] {
            let (width, height) = (97, 23);
            let pixels = made(width, height, channels);
            let image = decode(&encode(width, height, channels, &pixels)).expect("decodes");
            assert_eq!((image.width, image.height, image.channels), (width, height, channels));
            assert!(image.pixels == pixels, "{channels} channels");
        }
    }

    #[test]
    fn what_is_not_a_whole_image_is_refused() {
        let data = encode(16, 16, 3, &made(16, 16, 3));
        // Cut short anywhere before its last pixel: no image, and no panic.
        for cut in 0..data.len() - 8 {
            assert!(decode(&data[..cut]).is_none(), "{cut}");
        }
        assert!(decode(&data[..data.len() - 8]).is_some(), "the end marker is not needed");
        let mut wrong = data.clone();
        wrong[0] = b'Q';
        assert!(decode(&wrong).is_none());
        let mut huge = data.clone();
        huge[4..8].copy_from_slice(&65_536u32.to_be_bytes());
        assert!(decode(&huge).is_none());
        let mut two = data;
        two[12] = 2;
        assert!(decode(&two).is_none());
    }
}
