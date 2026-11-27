//! The texture pack's fixtures: synthetic textures and generated pictures
//! only (nothing of the title's).

use super::*;

/// The pack is one for the whole process: the tests that put one in place
/// run one at a time.
static ALONE: Mutex<()> = Mutex::new(());

/// Runs `f` with `pack` in place (none: the pack is off).
fn with_pack<R>(pack: Option<&'static Pack>, f: impl FnOnce() -> R) -> R {
    let _alone = ALONE.lock().unwrap_or_else(|e| e.into_inner());
    *TEST_PACK.lock().unwrap_or_else(|e| e.into_inner()) = pack;
    struct Off;
    impl Drop for Off {
        fn drop(&mut self) {
            *TEST_PACK.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    let _off = Off;
    f()
}

/// The pack's thread waits here before each preparation while a test holds
/// the gate shut (30 s at most).
static GATE: Mutex<bool> = Mutex::new(false);
static GATE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

pub(super) fn gate() {
    let shut = GATE.lock().unwrap_or_else(|e| e.into_inner());
    let _ = GATE_MOVED.wait_timeout_while(shut, std::time::Duration::from_secs(30), |shut| *shut);
}

fn shut_gate(shut: bool) {
    *GATE.lock().unwrap_or_else(|e| e.into_inner()) = shut;
    GATE_MOVED.notify_all();
}

/// Waits (a minute at most) until `id`'s preparation is over.
fn settle(pack: &Pack, id: &Identity) {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !pack.settled(id) {
        assert!(std::time::Instant::now() < until, "{id} still being prepared");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// A mapping entry.
fn entry(id: &str, image: &str) -> String {
    format!("[[texture]]\nid = \"{id}\"\nimage = '{image}'\n")
}

const ID_A: &str = "00112233445566ff00112233445566ff-1920x1080-f98-m11";
const ID_B: &str = "0011223344556600aabbccddeeff0011-512x512-f71-m10";
const ID_C: &str = "0011223344556600aabbccddeeff0011-64x64-f28-m1";

/// A picture of `width` x `height`, each texel what `f` says.
fn picture(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Image {
    let rgba = (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).flat_map(|(x, y)| f(x, y)).collect();
    Image { width, height, rgba }
}

const RED: [u8; 4] = [220, 30, 30, 255];
const BLUE: [u8; 4] = [30, 30, 220, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];

/// A square picture, red on the left half and blue on the right.
fn halves(side: u32) -> Image {
    picture(side, side, |x, _| if x < side / 2 { RED } else { BLUE })
}

fn near(a: [u8; 4], b: [u8; 4], within: u8) -> bool {
    a.iter().zip(&b).all(|(&x, &y)| x.abs_diff(y) <= within)
}

/// Writes `image` as a PNG or a JPEG file, with Windows' own encoders.
fn write_picture(path: &Path, image: &Image, jpeg: bool) {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::Graphics::Imaging::*;
    use windows::Win32::System::Com::*;
    let path: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let image = image.clone();
    std::thread::spawn(move || {
        // SAFETY: COM for this thread alone; every object is released before
        // the uninitialisation; the buffers outlive the calls that read them.
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok().unwrap();
            {
                let factory: IWICImagingFactory =
                    CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER).unwrap();
                let stream = factory.CreateStream().unwrap();
                stream
                    .InitializeFromFilename(windows::core::PCWSTR(path.as_ptr()), windows::Win32::Foundation::GENERIC_WRITE.0)
                    .unwrap();
                let container = if jpeg { GUID_ContainerFormatJpeg } else { GUID_ContainerFormatPng };
                let encoder = factory.CreateEncoder(&container, std::ptr::null()).unwrap();
                encoder.Initialize(&stream, WICBitmapEncoderNoCache).unwrap();
                let (mut frame, mut options) = (None, None);
                encoder.CreateNewFrame(&mut frame, &mut options).unwrap();
                let frame = frame.unwrap();
                frame.Initialize(options.as_ref()).unwrap();
                frame.SetSize(image.width, image.height).unwrap();
                let mut format = GUID_WICPixelFormat32bppRGBA;
                frame.SetPixelFormat(&mut format).unwrap();
                let bitmap = factory
                    .CreateBitmapFromMemory(image.width, image.height, &GUID_WICPixelFormat32bppRGBA, image.width * 4, &image.rgba)
                    .unwrap();
                frame.WriteSource(&bitmap, std::ptr::null()).unwrap();
                frame.Commit().unwrap();
                encoder.Commit().unwrap();
            }
            CoUninitialize();
        }
    })
    .join()
    .unwrap();
}

// Decoders written from the formats' descriptions, apart from the encoders.

fn widen_565(c: u16) -> [u8; 3] {
    let (r, g, b) = ((c >> 11) as u8 & 31, (c >> 5) as u8 & 63, c as u8 & 31);
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// A BC1 colour block; `four`: always four colours (BC3's colour half).
fn unpack_bc1(block: &[u8], four: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let (a, b) = (widen_565(c0), widen_565(c1));
    let colours: [[u8; 4]; 4] = if four || c0 > c1 {
        [
            [a[0], a[1], a[2], 255],
            [b[0], b[1], b[2], 255],
            std::array::from_fn(|c| if c == 3 { 255 } else { ((2 * u32::from(a[c]) + u32::from(b[c])) / 3) as u8 }),
            std::array::from_fn(|c| if c == 3 { 255 } else { ((u32::from(a[c]) + 2 * u32::from(b[c])) / 3) as u8 }),
        ]
    } else {
        [
            [a[0], a[1], a[2], 255],
            [b[0], b[1], b[2], 255],
            std::array::from_fn(|c| if c == 3 { 255 } else { ((u32::from(a[c]) + u32::from(b[c])) / 2) as u8 }),
            [0, 0, 0, 0],
        ]
    };
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    std::array::from_fn(|i| colours[(bits >> (2 * i) & 3) as usize])
}

fn unpack_bc4(block: &[u8]) -> [u8; 16] {
    let (a0, a1) = (u32::from(block[0]), u32::from(block[1]));
    let values: [u32; 8] = if a0 > a1 {
        [a0, a1, (6 * a0 + a1) / 7, (5 * a0 + 2 * a1) / 7, (4 * a0 + 3 * a1) / 7, (3 * a0 + 4 * a1) / 7, (2 * a0 + 5 * a1) / 7, (a0 + 6 * a1) / 7]
    } else {
        [a0, a1, (4 * a0 + a1) / 5, (3 * a0 + 2 * a1) / 5, (2 * a0 + 3 * a1) / 5, (a0 + 4 * a1) / 5, 0, 255]
    };
    let mut bits = [0u8; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(bits);
    std::array::from_fn(|i| values[(bits >> (3 * i) & 7) as usize] as u8)
}

/// A BC7 block of mode 6 (the only mode the encoder writes).
fn unpack_bc7(block: &[u8]) -> [[u8; 4]; 16] {
    let bits = u128::from_le_bytes(block.try_into().unwrap());
    let mut at = 0u32;
    let mut take = |n: u32| {
        let v = (bits >> at) as u32 & ((1u32 << n) - 1);
        at += n;
        v
    };
    assert_eq!(take(7), 0b100_0000, "mode 6");
    let mut ends = [[0u32; 4]; 2];
    for c in 0..4 {
        ends[0][c] = take(7);
        ends[1][c] = take(7);
    }
    let (p0, p1) = (take(1), take(1));
    let ends = [ends[0].map(|v| v << 1 | p0), ends[1].map(|v| v << 1 | p1)];
    let weights = [0u32, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    std::array::from_fn(|i| {
        let index = take(if i == 0 { 3 } else { 4 }) as usize;
        let w = weights[index];
        std::array::from_fn(|c| (((64 - w) * ends[0][c] + w * ends[1][c] + 32) >> 6) as u8)
    })
}

/// A level's blocks as a picture of `width` x `height`.
fn unpack(level: &[u8], width: u32, height: u32, encoding: Encoding) -> Image {
    let block_bytes = encoding.block_bytes().unwrap();
    let bw = width.div_ceil(4) as usize;
    picture(width, height, |x, y| {
        let at = ((y / 4) as usize * bw + (x / 4) as usize) * block_bytes;
        let block = &level[at..at + block_bytes];
        let i = (y % 4 * 4 + x % 4) as usize;
        match encoding {
            Encoding::Bc1 => unpack_bc1(block, false)[i],
            Encoding::Bc3 => {
                let mut t = unpack_bc1(&block[8..], true)[i];
                t[3] = unpack_bc4(block)[i];
                t
            }
            _ => unpack_bc7(block)[i],
        }
    })
}

/// Peak signal to noise, in dB, over the channels `channels` counts.
fn psnr(a: &Image, b: &Image, channels: usize) -> f64 {
    let mut sum = 0f64;
    let mut n = 0f64;
    for (x, y) in a.rgba.chunks_exact(4).zip(b.rgba.chunks_exact(4)) {
        for c in 0..channels {
            sum += (f64::from(x[c]) - f64::from(y[c])).powi(2);
            n += 1.0;
        }
    }
    let mse = sum / n;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

/// Smooth colour, and some detail on top: what a photograph is like.
fn photo(width: u32, height: u32) -> Image {
    picture(width, height, |x, y| {
        let wave = |v: u32, k: f32| ((v as f32 * k).sin() * 20.0) as i32;
        let r = (x * 255 / width) as i32 + wave(y, 0.3);
        let g = (y * 255 / height) as i32 + wave(x + y, 0.21);
        let b = ((x + y) * 255 / (width + height)) as i32 + wave(x, 0.17);
        [r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8, 255]
    })
}

#[test]
fn identities_print_and_parse_back_and_cover_every_level() {
    let shape = Shape { width: 1920, height: 1080, format: 98, mips: 11 };
    let id = Identity { hash: 0x5f1c_9a0e_7b2d_4c3a_0011_2233_4455_66ff, shape };
    let text = "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98-m11";
    assert_eq!(id.to_string(), text);
    assert_eq!(text.parse::<Identity>(), Ok(id));
    assert_eq!(format!(" {text} ").parse::<Identity>(), Ok(id), "spaces");
    for bad in [
        "",
        "5f1c9a0e7b2d4c3a-1920x1080-f98",
        "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98",
        "5f1c9a0e7b2d4c3a00112233445566f-1920x1080-f98-m11",
        "5f1c9a0e7b2d4c3a00112233445566ff-1920-f98-m11",
        "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-98-m11",
        "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98-11",
        "5f1c9a0e7b2d4c3a00112233445566ff-1920x1080-f98-m11-x",
    ] {
        assert!(bad.parse::<Identity>().is_err(), "{bad:?}");
    }
    // The hash is of the texels alone, rows without their padding.
    let small = Shape { width: 2, height: 2, format: 28, mips: 2 };
    let top = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut padded = vec![0xee; 2 * 256];
    padded[..8].copy_from_slice(&top[..8]);
    padded[256..264].copy_from_slice(&top[8..]);
    let (next, other) = ([9u8, 9, 9, 9], [9u8, 9, 9, 8]);
    let of = |top: &[u8], pitch, next: &[u8]| Identity::of(&small, [(top, pitch, 2, 8), (next, 4, 1, 4)]);
    assert_eq!(of(&top, 8, &next), of(&padded, 256, &next));
    // Every level counts: the same top level over another level 1 is
    // another texture; so is another shape.
    assert_ne!(of(&top, 8, &next), of(&top, 8, &other));
    let one = Shape { mips: 1, ..small };
    assert_ne!(Identity::of(&one, [(&top[..], 8, 2, 8)]), Identity::of(&small, [(&top[..], 8, 2, 8)]));
}

#[test]
fn the_mapping_file_maps_identities_with_their_fit() {
    let folder = Path::new("unused");
    let text = format!("fit = \"letterbox\"\n{}{}fit = \"cover\"\n", entry(ID_A, "a.png"), entry(ID_B, "b.jpg"));
    let pack = Pack::parse(&text, folder).unwrap();
    let (a, b): (Identity, Identity) = (ID_A.parse().unwrap(), ID_B.parse().unwrap());
    assert_eq!(pack.entries[&a].fit, Fit::Letterbox, "the file's fit");
    assert_eq!(pack.entries[&b].fit, Fit::Cover, "the entry's own");
    assert_eq!((pack.entries[&b].image.as_str(), pack.entries[&b].number), ("b.jpg", 2));
    let shape = Shape { width: 1920, height: 1080, format: 98, mips: 11 };
    assert!(pack.wants(&shape));
    assert!(!pack.wants(&Shape { format: 99, ..shape }), "another format");
    assert!(!pack.wants(&Shape { width: 1080, height: 1920, ..shape }), "another size");
    assert!(!pack.wants(&Shape { mips: 10, ..shape }), "other levels");
    let plain = Pack::parse(&entry(ID_C, "a.png"), folder).unwrap();
    assert_eq!(plain.entries.values().next().unwrap().fit, Fit::Cover, "cover by default");
    assert!(Pack::parse("", folder).unwrap().entries.is_empty());
    for bad in [
        entry("nonsense", "a.png"),
        format!("[[texture]]\nid = \"{ID_C}\"\n"),
        format!("{}scale = 2\n", entry(ID_C, "a.png")),
        "fit = \"stretch\"\n".to_string(),
        format!("{}{}", entry(ID_C, "a.png"), entry(ID_C, "b.png")),
    ] {
        assert!(Pack::parse(&bad, folder).is_err(), "{bad:?}");
    }
    let many: String = (0..=MOST_ENTRIES).map(|i| entry(&format!("{i:032x}-64x64-f28-m1"), "a.png")).collect();
    let why = Pack::parse(&many, folder).err().unwrap();
    assert!(why.contains("more than 256 entries"), "{why}");
    // A picture is named by a bare file name in the folder.
    for image in ["..\\a.png", "sub\\a.png", "sub/a.png", "C:\\a.png", "C:a.png", "\\\\server\\share\\a.png", "a.png:stream", ".", "..", ""] {
        let why = Pack::parse(&entry(ID_C, image), folder).err().unwrap_or_default();
        assert!(why.contains("the image of entry 1 is not the name of a file in the folder"), "{image:?}: {why}");
    }
}

/// What a broken mapping file is said to be carries nothing of it: a key, a
/// value or an id holding a path is not repeated.
#[test]
fn a_broken_mapping_is_said_in_fixed_words() {
    let folder = Path::new("unused");
    let secret = "C:\\Users\\someone\\pictures";
    for text in [
        format!("{}'{secret}' = 1\n", entry(ID_C, "a.png")),
        format!("fit = '{secret}'\n"),
        format!("[[texture]]\nid = '{secret}'\nimage = 'a.png'\n"),
        entry(ID_C, secret),
        format!("{secret}\n"),
        format!("{}{}", entry(ID_C, "a.png"), entry(ID_C, "b.png")),
    ] {
        let why = Pack::parse(&text, folder).err().unwrap_or_default();
        assert!(why.starts_with("overrides.toml") && !why.contains("someone") && !why.contains('\\'), "{why}");
    }
    let why = Pack::parse(&format!("\n\nfit = '{secret}'\n"), folder).err().unwrap_or_default();
    assert!(why.ends_with("(line 3)"), "{why}");
}

/// Only plain files in the folder itself are read: not one a name leads out
/// of the folder to, not a link (to a file inside or out), not a folder,
/// nor one larger than the pack takes.
#[test]
fn only_plain_files_in_the_folder_are_read() {
    const NOT_PLAIN: Result<Vec<u8>, &str> = Err("is not a plain file in the folder");
    let root = tempfile::tempdir().unwrap();
    let folder = root.path().join("pack");
    std::fs::create_dir(&folder).unwrap();
    let canonical = std::fs::canonicalize(&folder).unwrap();
    std::fs::write(folder.join("inside.png"), b"inside").unwrap();
    let outside = root.path().join("outside.png");
    std::fs::write(&outside, b"outside").unwrap();
    assert_eq!(read_in(&canonical, "inside.png", 100), Ok(b"inside".to_vec()));
    assert_eq!(read_in(&canonical, "..\\outside.png", 100), NOT_PLAIN);
    assert_eq!(read_in(&canonical, &outside.to_string_lossy(), 100), NOT_PLAIN);
    assert_eq!(read_in(&canonical, "missing.png", 100), Err("was not found"));
    std::fs::create_dir(folder.join("folder.png")).unwrap();
    assert_eq!(read_in(&canonical, "folder.png", 100), Err("could not be opened"), "a folder");
    // Links, where this account may make them (a file link needs the right).
    if std::os::windows::fs::symlink_file(&outside, folder.join("out.png")).is_ok() {
        assert_eq!(read_in(&canonical, "out.png", 100), NOT_PLAIN, "a link out of the folder");
    }
    if std::os::windows::fs::symlink_file(folder.join("inside.png"), folder.join("in.png")).is_ok() {
        assert_eq!(read_in(&canonical, "in.png", 100), NOT_PLAIN, "a link, even to a file inside");
    }
    // Where an opened file lies is Windows' answer, links resolved (what
    // refuses a link this account cannot make here).
    assert!(in_folder(&std::fs::File::open(folder.join("inside.png")).unwrap(), &canonical));
    assert!(!in_folder(&std::fs::File::open(&outside).unwrap(), &canonical));
    // The folder must be the one the file is in, once links are resolved.
    assert_eq!(read_in(&std::fs::canonicalize(root.path()).unwrap(), "inside.png", 100), Err("was not found"));
    assert_eq!(read_in(&canonical, "inside.png", 5), Err("is larger than the pack takes"));
    // The mapping file, likewise: too large; one naming a picture outside.
    std::fs::write(folder.join("overrides.toml"), vec![b'#'; MOST_MAPPING_BYTES as usize + 1]).unwrap();
    let why = Pack::load(&folder).err().unwrap_or_default();
    assert_eq!(why, "overrides.toml is larger than the pack takes");
    std::fs::write(folder.join("overrides.toml"), entry(ID_C, "..\\outside.png")).unwrap();
    assert!(Pack::load(&folder).is_err());
    std::fs::write(folder.join("overrides.toml"), entry(ID_C, "inside.png")).unwrap();
    assert!(Pack::load(&folder).is_ok());
    // A folder without the file: said without the folder.
    let empty = tempfile::tempdir().unwrap();
    let why = Pack::load(empty.path()).err().unwrap_or_default();
    assert_eq!(why, "overrides.toml was not found");
}

#[test]
fn pictures_decode_from_png_and_jpeg_within_bounds() {
    let folder = tempfile::tempdir().unwrap();
    let image = photo(40, 24);
    let png = folder.path().join("p.png");
    let jpeg = folder.path().join("p.jpg");
    write_picture(&png, &image, false);
    write_picture(&jpeg, &image, true);
    assert_eq!(decode(&std::fs::read(&png).unwrap()), Ok(image.clone()), "a PNG is exact");
    let from_jpeg = decode(&std::fs::read(&jpeg).unwrap()).unwrap();
    assert_eq!((from_jpeg.width, from_jpeg.height), (40, 24));
    assert!(psnr(&from_jpeg, &image, 3) > 25.0, "a JPEG is near: {:.1} dB", psnr(&from_jpeg, &image, 3));
    assert_eq!(decode(b"not a picture"), Err("is not a PNG or JPEG file"));
    assert_eq!(decode(b"GIF89a\x01\x00\x01\x00"), Err("is not a PNG or JPEG file"), "no other decoder is asked");
    assert_eq!(decode(b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR"), Err("could not be decoded"));
    // Over the texels the pack takes (8000 x 2200: 17.6 million), or a side
    // over it: refused before the texels are made.
    for (w, h) in [(8000, 2200), (8193, 1)] {
        let big = folder.path().join("big.png");
        write_picture(&big, &picture(w, h, |x, _| [x as u8, 0, 0, 255]), false);
        assert_eq!(decode(&std::fs::read(&big).unwrap()), Err("is larger than the pack takes"), "{w}x{h}");
    }
}

/// Prepared pictures stay within the pack's bytes: one that would take it
/// past them is refused before its picture is read; one that fails gives
/// its bytes back.
#[test]
fn prepared_pictures_stay_within_the_pack_s_memory() {
    let shape = Shape { width: 64, height: 64, format: 28, mips: 1 };
    let [a, b, c] = [1, 2, 3].map(|hash| Identity { hash, shape });
    let (pack, _folder) = pack_in(&[a, b], &entry(&c.to_string(), "missing.png"), 20 << 10);
    with_pack(Some(pack), || {
        for id in [c, a, b] {
            assert!(matches!(pack.replacement(&id), Replacement::Pending));
            settle(pack, &id);
        }
        assert!(matches!(pack.replacement(&c), Replacement::None), "no picture");
        assert!(matches!(pack.replacement(&a), Replacement::Ready(_)));
        assert!(matches!(pack.replacement(&b), Replacement::None), "16 KiB more would pass 20 KiB");
        assert_eq!(pack.retained.load(Ordering::Acquire), 16 << 10, "the missing picture's bytes given back");
    });
}

#[test]
fn cover_fills_the_texture_and_letterbox_shows_all_of_the_picture() {
    let square = halves(96);
    let cover = fitted(&square, 128, 64, Fit::Cover);
    assert_eq!((cover.width, cover.height), (128, 64));
    assert!(near(cover.texel(8, 32), RED, 2) && near(cover.texel(120, 32), BLUE, 2));
    assert!(near(cover.texel(8, 0), RED, 2) && near(cover.texel(120, 63), BLUE, 2), "no bars");
    assert!(near(cover.texel(62, 32), RED, 2) && near(cover.texel(65, 32), BLUE, 2), "centred");
    let letterbox = fitted(&square, 128, 64, Fit::Letterbox);
    assert_eq!(letterbox.texel(8, 32), BLACK, "a bar on the left");
    assert_eq!(letterbox.texel(120, 32), BLACK, "and on the right");
    assert!(near(letterbox.texel(34, 32), RED, 2) && near(letterbox.texel(93, 32), BLUE, 2), "64x64 in the middle");
    assert!(near(letterbox.texel(34, 0), RED, 2) && near(letterbox.texel(93, 63), BLUE, 2));
    // A tall picture covering a square texture: its middle quarter of rows
    // (48 to 80), twice as large.
    let green = [10, 200, 10, 255];
    let tall = picture(32, 128, |_, y| if y < 56 { RED } else if y < 72 { green } else { BLUE });
    let cut = fitted(&tall, 64, 64, Fit::Cover);
    assert!(near(cut.texel(5, 2), RED, 2) && near(cut.texel(5, 32), green, 2) && near(cut.texel(5, 61), BLUE, 2));
    assert!(near(cut.texel(5, 13), RED, 2) && near(cut.texel(5, 18), green, 2), "red to row 16");
    // The same size: the same texels.
    let image = photo(37, 23);
    assert_eq!(fitted(&image, 37, 23, Fit::Cover), image);
    assert_eq!(fitted(&image, 37, 23, Fit::Letterbox), image);
}

#[test]
fn every_level_is_made_and_has_the_texture_s_size() {
    let shape = Shape { width: 128, height: 64, format: 98, mips: 8 };
    let p = Prepared::new(&halves(96), &shape, Fit::Cover, Encoding::Bc7);
    let sizes: Vec<(usize, usize)> = p.levels.iter().map(|l| (l.row_bytes, l.rows)).collect();
    assert_eq!(sizes, [(32 * 16, 16), (16 * 16, 8), (8 * 16, 4), (4 * 16, 2), (2 * 16, 1), (16, 1), (16, 1), (16, 1)]);
    // The smallest level is the picture's average: half red, half blue.
    let one = unpack(&p.levels[7].bytes, 1, 1, Encoding::Bc7).texel(0, 0);
    assert!(near(one, [125, 30, 125, 255], 3), "{one:?}");
    // A part of another size is not written.
    let mut part = vec![0u8; 512 * 16];
    assert!(!p.write(0, &mut part, 512, 15));
    assert!(part.iter().all(|&b| b == 0));
    assert!(p.write(0, &mut part, 512, 16));
    let rgba = Prepared::new(&halves(96), &Shape { format: 87, ..shape }, Fit::Cover, Encoding::Bgra);
    assert_eq!(&rgba.levels[0].bytes[..4], &[RED[2], RED[1], RED[0], 255], "BGRA");
}

#[test]
fn bc7_keeps_a_photograph_and_single_colours() {
    let image = photo(64, 64);
    let back = unpack(&encode(&image, Encoding::Bc7).bytes, 64, 64, Encoding::Bc7);
    let db = psnr(&back, &image, 4);
    assert!(db > 40.0, "BC7: {db:.1} dB");
    // Single colours come back within one step, alpha too.
    let colours = picture(64, 64, |x, y| {
        let n = (x / 4 + 16 * (y / 4)).wrapping_mul(2_654_435_761).wrapping_add(12_345);
        let k = (n ^ n >> 15).wrapping_mul(2_246_822_519);
        [(k >> 24) as u8, (k >> 16) as u8, (k >> 8) as u8, (k >> 3) as u8 | 0x80]
    });
    let back = unpack(&encode(&colours, Encoding::Bc7).bytes, 64, 64, Encoding::Bc7);
    for (a, b) in back.rgba.chunks_exact(4).zip(colours.rgba.chunks_exact(4)) {
        assert!(near(a.try_into().unwrap(), b.try_into().unwrap(), 1), "{a:?} for {b:?}");
    }
    // Sizes that are not whole blocks: the edge texels fill the blocks.
    // (Colours along one line, which one mode-6 block holds.)
    let odd = picture(6, 3, |x, y| {
        let v = (x * 30 + y * 12) as u8;
        [v, v / 2, 255 - v, 255]
    });
    let back = unpack(&encode(&odd, Encoding::Bc7).bytes, 6, 3, Encoding::Bc7);
    let db = psnr(&back, &odd, 4);
    assert!(db > 40.0, "{db:.1} dB");
}

#[test]
fn bc1_and_bc3_keep_a_photograph() {
    let image = photo(64, 64);
    let bc1 = unpack(&encode(&image, Encoding::Bc1).bytes, 64, 64, Encoding::Bc1);
    let db = psnr(&bc1, &image, 3);
    assert!(db > 32.0, "BC1: {db:.1} dB");
    assert!(bc1.rgba.chunks_exact(4).all(|t| t[3] == 255), "four colours, none transparent");
    // BC3: the same colour, and alpha of its own.
    let faded = picture(64, 64, |x, y| {
        let mut t = image.texel(x, y);
        t[3] = (x * 3 + y / 8) as u8;
        t
    });
    let bc3 = unpack(&encode(&faded, Encoding::Bc3).bytes, 64, 64, Encoding::Bc3);
    assert!(psnr(&bc3, &faded, 3) > 32.0);
    for (a, b) in bc3.rgba.chunks_exact(4).zip(faded.rgba.chunks_exact(4)) {
        assert!(a[3].abs_diff(b[3]) <= 2, "alpha {} for {}", a[3], b[3]);
    }
    // One colour, one alpha: exact where 5:6:5 has it.
    let flat = picture(8, 8, |_, _| [255, 0, 255, 200]);
    let back = unpack(&encode(&flat, Encoding::Bc3).bytes, 8, 8, Encoding::Bc3);
    assert!(back.rgba.chunks_exact(4).all(|t| t == [255, 0, 255, 200]));
}

// Through the host: a title texture uploaded and read back.

use crate::object::This;
use windows::Win32::Graphics::Direct3D12::*;

/// Every level of a texture as the title wrote it: detiled from its
/// memory, rows packed.
fn title_levels(layout: &victory_formats::texture::Layout, memory: *const u8, mips: u32) -> Vec<Vec<u8>> {
    // SAFETY: the texture's own memory, its layout's size.
    let tiled = unsafe { std::slice::from_raw_parts(memory, layout.size as usize) };
    (0..mips)
        .map(|mip| {
            let m = &layout.mips[mip as usize];
            let row = m.width as usize * layout.desc.element_bytes as usize;
            let len = row * m.height as usize;
            let mut linear = vec![0u8; len];
            layout.detile(tiled, mip, 0, &mut linear, row, len).unwrap();
            linear
        })
        .collect()
}

/// Fills a title texture's memory with what `seed` makes.
fn fill(memory: *mut u8, size: usize, seed: u64) {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    // SAFETY: the texture's own memory, `size` bytes of it.
    let bytes = unsafe { std::slice::from_raw_parts_mut(memory, size) };
    for b in bytes {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = (x >> 32) as u8;
    }
}

/// Every level of the host copy of title texture `image`, rows packed.
fn read_back(host: &crate::host::Host, image: This, texture: &ID3D12Resource) -> Vec<Vec<u8>> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let n = usize::from(d.MipLevels);
    let mut placed = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let (mut rows, mut row_bytes, mut total) = (vec![0u32; n], vec![0u64; n], 0u64);
    // SAFETY: outputs for `n` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            n as u32,
            0,
            Some(placed.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_bytes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut readback: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a readback buffer.
    unsafe {
        host.device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)
            .unwrap();
    }
    let readback = readback.unwrap();
    let back = host.new_list().unwrap();
    crate::list::ensure_state(&back, image, 0x800);
    for (i, footprint) in placed.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list and live resources.
        unsafe { back.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    back.close();
    host.execute(&[&back]);
    let mut mapped = std::ptr::null_mut();
    // SAFETY: the copies finished (execute waits); the buffer is `total` bytes.
    let read = unsafe {
        readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: total as usize }), Some(&mut mapped)).unwrap();
        std::slice::from_raw_parts(mapped.cast::<u8>(), total as usize)
    };
    (0..n)
        .map(|i| {
            let pitch = placed[i].Footprint.RowPitch as usize;
            (0..rows[i] as usize)
                .flat_map(|r| &read[placed[i].Offset as usize + r * pitch..][..row_bytes[i] as usize])
                .copied()
                .collect()
        })
        .collect()
}

/// A title texture made for a test: the title's handle, its host copy,
/// its memory and its layout.
struct Title {
    this: This,
    texture: &'static ID3D12Resource,
    memory: *mut u8,
    layout: victory_formats::texture::Layout,
    shape: Shape,
}

impl Title {
    fn new(width: u32, height: u32, format: u32, mips: u32) -> Title {
        let (this, memory, layout) = crate::resource::test_texture(u64::from(width), height, format, mips as u16);
        let texture = crate::resource::host_resource(this).expect("a host copy");
        Title { this, texture, memory, layout, shape: Shape { width, height, format, mips } }
    }

    /// The title writes the texture anew.
    fn write(&self, seed: u64) -> Vec<Vec<u8>> {
        fill(self.memory, self.layout.size as usize, seed);
        title_levels(&self.layout, self.memory, self.shape.mips)
    }

    /// The identity of what the title wrote: every level's rows.
    fn identity(&self, levels: &[Vec<u8>]) -> Identity {
        Identity::of(
            &self.shape,
            levels.iter().enumerate().map(|(mip, level)| {
                let rows = self.layout.mips[mip].height as usize;
                let row = level.len() / rows;
                (&level[..], row, rows, row)
            }),
        )
    }

    /// One list samples the texture (it is uploaded if it changed), and
    /// one more after it (what the first list's check found changed is
    /// copied by the next list that samples it); then its host copy.
    fn sampled(&self, host: &crate::host::Host) -> Vec<Vec<u8>> {
        for _ in 0..2 {
            let list = host.new_list().unwrap();
            crate::texture::upload(&list, self.this, self.texture);
            list.close();
            host.execute(&[&list]);
        }
        read_back(host, self.this, self.texture)
    }
}

/// A folder of its own holding a 96x96 picture, red on the left and blue on
/// the right, and a mapping of `ids` to it, then `more`; and the pack it
/// makes, keeping `most` bytes of prepared pictures.
fn pack_in(ids: &[Identity], more: &str, most: usize) -> (&'static Pack, tempfile::TempDir) {
    let folder = tempfile::tempdir().unwrap();
    write_picture(&folder.path().join("halves.png"), &halves(96), false);
    let mut text: String = ids.iter().map(|id| entry(&id.to_string(), "halves.png")).collect();
    text.push_str(more);
    std::fs::write(folder.path().join("overrides.toml"), text).unwrap();
    let mut pack = Pack::load(folder.path()).unwrap();
    pack.most_retained = most;
    (Box::leak(Box::new(pack)), folder)
}

fn pack_for(ids: &[Identity]) -> (&'static Pack, tempfile::TempDir) {
    pack_in(ids, "", MOST_RETAINED)
}

/// The formats through the host, each a size of its own (no other test's
/// texture shares it).
const FORMATS: [(u32, u32, u32, u32); 4] = [(28, 136, 72, 3), (71, 132, 68, 4), (77, 140, 76, 3), (98, 144, 80, 8)];

/// A texture whose identity the pack maps shows the title's texels while
/// its picture is prepared (the submission does not wait for it), then the
/// picture, in every level, in its format; when the title writes something
/// else into it, that; when it writes the mapped texels again, the picture
/// again. Uncompressed and block-compressed.
#[test]
fn a_mapped_texture_is_replaced_on_first_upload_and_on_refresh() {
    let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    for (format, width, height, mips) in FORMATS {
        let title = Title::new(width, height, format, mips);
        let mapped = title.write(1);
        let id = title.identity(&mapped);
        let (pack, _folder) = pack_for(&[id]);
        with_pack(Some(pack), || {
            // The pack's thread is held: the submission goes on without it.
            shut_gate(true);
            let while_prepared = title.sampled(host);
            shut_gate(false);
            assert_eq!(while_prepared, mapped, "format {format}: the title's texels while the picture is prepared");
            settle(pack, &id);
            let Replacement::Ready(replacement) = pack.replacement(&id) else {
                panic!("format {format}: no picture prepared");
            };
            let shown = title.sampled(host);
            for (level, (shown, made)) in shown.iter().zip(&replacement.levels).enumerate() {
                assert_eq!(shown, &made.bytes, "format {format} level {level}: the picture's, once prepared");
                assert_ne!(shown, &mapped[level], "format {format} level {level}: not the title's");
            }
            // What the texture shows is the picture: red on the left, blue on
            // the right, in its top level.
            let top = match Encoding::of(format).unwrap() {
                Encoding::Rgba => Image { width, height, rgba: shown[0].clone() },
                encoding => unpack(&shown[0], width, height, encoding),
            };
            let (left, right) = (top.texel(4, height / 2), top.texel(width - 5, height / 2));
            assert!(near(left, RED, 12) && near(right, BLUE, 12), "format {format}: {left:?} {right:?}");
            // The title writes other texels: they are shown.
            let other = title.write(2);
            assert_eq!(title.sampled(host), other, "format {format}: an unmapped refresh is the title's");
            // And the mapped ones again: the picture again.
            assert_eq!(title.write(1), mapped);
            let again = title.sampled(host);
            for (level, (shown, made)) in again.iter().zip(&replacement.levels).enumerate() {
                assert_eq!(shown, &made.bytes, "format {format} level {level}: the picture's, on refresh");
            }
        });
    }
}

/// A placed texture staged from a smaller level has no identity, whatever
/// a mapping says, and keeps the title's texels; staged anew from level 0 it
/// has its one identity (the same as had it been staged from level 0 first)
/// and shows the picture.
#[test]
fn a_texture_restaged_from_level_0_has_its_one_identity() {
    let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    let title = Title::new(512, 512, 71, 8);
    let levels = title.write(7);
    let id = title.identity(&levels);
    let (pack, _folder) = pack_for(&[id]);
    with_pack(Some(pack), || {
        let list = host.new_list().unwrap();
        crate::texture::upload_from(&list, title.this, title.texture, 2);
        list.close();
        host.execute(&[&list]);
        let staged = crate::texture::staged_memory(title.this).unwrap().1;
        assert!(staged < title.layout.size, "staged from level 2");
        let shown = read_back(host, title.this, title.texture);
        assert_eq!(shown[2..], levels[2..], "staged from level 2: the title's");
        assert!(!pack.state.lock().unwrap().contains_key(&id), "no identity: nothing asked of the pack");
        // Staged from level 0: its identity; the title's texels while the
        // picture is prepared, then the picture.
        assert_eq!(title.sampled(host), levels);
        settle(pack, &id);
        let Replacement::Ready(replacement) = pack.replacement(&id) else {
            panic!("no picture prepared");
        };
        let shown = title.sampled(host);
        for (level, (shown, made)) in shown.iter().zip(&replacement.levels).enumerate() {
            assert_eq!(shown, &made.bytes, "level {level}: the picture's");
        }
    });
}

/// With the pack on, a texture of a mapped shape whose texels are not the
/// mapped ones is the title's.
#[test]
fn an_unmapped_texture_is_untouched() {
    let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    for (format, width, height, mips) in FORMATS {
        let title = Title::new(width, height + 8, format, mips);
        let elsewhere = Identity { hash: 0x0123_4567_89ab_cdef, ..title.identity(&title.write(3)) };
        let (pack, _folder) = pack_for(&[elsewhere]);
        with_pack(Some(pack), || {
            let written = title.write(4);
            assert!(pack.wants(&title.shape), "its shape is mapped");
            assert_eq!(title.sampled(host), written, "format {format}");
            assert!(!pack.state.lock().unwrap().contains_key(&title.identity(&written)));
        });
    }
}

/// With the pack off, a texture whose identity a pack would map is the
/// title's, and is staged as without the pack (no shape: the loop of old).
#[test]
fn with_the_pack_off_nothing_changes() {
    let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    for (format, width, height, mips) in FORMATS {
        let title = Title::new(width, height + 16, format, mips);
        let written = title.write(5);
        let (_pack, _folder) = pack_for(&[title.identity(&written)]);
        with_pack(None, || {
            assert!(!on() && look(title.shape).is_none(), "nothing to do");
            assert_eq!(title.sampled(host), written, "format {format}");
        });
    }
}

#[cfg(feature = "diagnostics")]
mod log_lines {
    use super::super::log::{MOST_PICTURES, Seen};
    use super::*;

    fn id(hash: u128) -> Identity {
        Identity { hash, shape: Shape { width: 1920, height: 1080, format: 98, mips: 11 } }
    }

    /// A draw over most of the screen: a target of 1280x720 or more, three
    /// quarters of it in the viewport.
    #[test]
    fn a_draw_covers_the_screen_when_its_viewport_covers_most_of_a_screen_sized_target() {
        assert!(covers_screen(1920, 1080, [0.0, 0.0, 1920.0, 1080.0]));
        assert!(covers_screen(1920, 1080, [-10.0, -10.0, 4000.0, 4000.0]), "past the target");
        assert!(covers_screen(1920, 1080, [0.0, 135.0, 1920.0, 810.0]), "three quarters");
        assert!(!covers_screen(1920, 1080, [0.0, 0.0, 960.0, 1080.0]), "half");
        assert!(!covers_screen(1920, 1080, [1900.0, 0.0, 1920.0, 1080.0]), "mostly outside");
        assert!(!covers_screen(1024, 1024, [0.0, 0.0, 1024.0, 1024.0]), "not a screen's target");
    }

    /// One line per identity a draw over the screen reads, however it comes
    /// about (read then detiled, or detiled then read), with its frame.
    #[test]
    fn each_identity_drawn_over_the_screen_is_logged_once() {
        let mut seen = Seen::default();
        assert!(seen.drawn(1, 10).is_empty(), "not detiled yet");
        let first = seen.detiled(1, id(0xa), 11, false);
        assert_eq!(
            first,
            ["texture pack: 0000000000000000000000000000000a-1920x1080-f98-m11 (1920x1080 BC7_UNORM, 11 levels) drawn over most of the screen, frame 11"]
        );
        assert!(seen.detiled(1, id(0xa), 12, false).is_empty());
        assert!(seen.drawn(1, 13).is_empty(), "logged already");
        assert_eq!(seen.detiled(1, id(0xb), 14, false).len(), 1, "a new picture");
        // Detiled first, then read by a draw over the screen.
        assert!(seen.detiled(2, id(0xc), 15, false).is_empty(), "no draw over the screen read it");
        let read = seen.drawn(2, 16);
        assert!(read[0].contains("0000000000000000000000000000000c") && read[0].ends_with("frame 16"), "{read:?}");
        // Another texture with a logged picture: not again.
        assert!(seen.detiled(3, id(0xa), 17, false).is_empty() && seen.drawn(3, 18).is_empty());
        // Released: its address is another texture's next.
        seen.forget(2);
        assert!(seen.drawn(2, 19).is_empty());
        // Uploads, with `=2`: every large texture, drawn or not, once.
        let up = seen.detiled(4, id(0xd), 20, true);
        assert_eq!(
            up,
            ["texture pack: 0000000000000000000000000000000d-1920x1080-f98-m11 (1920x1080 BC7_UNORM, 11 levels) uploaded, frame 20"]
        );
        assert!(seen.detiled(4, id(0xd), 21, true).is_empty());
        // A format the pack cannot write says so.
        let r8 = Identity { shape: Shape { format: 61, ..id(0).shape }, ..id(0xe) };
        assert!(seen.detiled(5, r8, 22, true)[0].contains("R8_UNORM, 11 levels, no encoder here: not replaceable"));
    }

    /// A picture is logged as drawn only for a draw over the screen that
    /// read it: not one detiled long after the last such draw, nor before it.
    #[test]
    fn a_picture_is_logged_for_the_draw_that_read_it() {
        let mut seen = Seen::default();
        seen.detiled(1, id(1), 9, false);
        assert_eq!(seen.drawn(1, 10).len(), 1);
        assert!(seen.detiled(1, id(2), 50, false).is_empty(), "no draw over the screen since frame 10");
        assert_eq!(seen.drawn(1, 51).len(), 1, "the next draw that reads it");
        assert!(seen.detiled(1, id(3), 40, false).is_empty(), "detiled before that draw");
    }

    #[test]
    fn a_movie_surface_is_said_to_be_one_and_then_left_alone() {
        let mut seen = Seen::default();
        let mut lines = Vec::new();
        for frame in 0..100 {
            lines.extend(seen.drawn(9, frame));
            lines.extend(seen.detiled(9, id(0x1000 + u128::from(frame)), frame, false));
        }
        assert_eq!(lines.len(), MOST_PICTURES as usize + 1);
        assert!(lines[MOST_PICTURES as usize].contains("a movie's surface"), "{}", lines[MOST_PICTURES as usize]);
        for line in &lines {
            assert!(line.starts_with("texture pack: ") && !line.contains('\\') && !line.contains('/'), "{line}");
        }
    }
}
