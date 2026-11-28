//! The title's FidelityFX upscaler path patched in at an upscaler's scale, so
//! that DLSS, DLSS frame generation, FSR 3 frame generation and
//! `VICTORY_FSR_PATH` can take its upscale dispatches over or use them. The
//! image is patched before it is mapped, and only for the supported retail
//! build: every field is checked against the hash of its stock bytes first,
//! and another build is refused, not patched.
//!
//! This file stands alone: it uses nothing of the 1440p output's module,
//! which takes the shared items below from here.
//!
//! From DexNom's fork (DexNom/VIctory, `apps/victory/src/native_1440p.rs`,
//! GPL-3.0-or-later).

use anyhow::{Context, Result, ensure};
use victory_hle::streamline::Mode;
use victory_loader::PreparedImage;

/// Whether `bytes` are the stock ones whose BLAKE3 hash (hex) is `hash`.
/// The stock bytes of each place patched are kept as their hash only: no
/// bytes of the title are in this repository.
pub(super) fn same(bytes: &[u8], hash: &str) -> bool {
    blake3::hash(bytes).to_hex().as_str() == hash
}

/// The hash of the one retail image these patches are known for.
pub(super) const RETAIL_BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
/// The title's temporal scene scale (its own temporal pass's render scale).
pub(super) const TEMPORAL_SCALE: u32 = 0x028f_1e7e;

/// The title's renderer selection: it loads renderer 6 (the title's own
/// temporal pass) into a register, then replaces it with the one the console
/// check chose (on Series S with the upscaler there, the FidelityFX path, 7).
/// The renderer number is the place's second byte; the replacement is its
/// last three.
const RENDERER_SELECTION: u32 = 0x007e_9421;
/// Where the renderer number sits in the selection, and where the replacement starts.
const RENDERER_NUMBER: u32 = 1;
const RENDERER_CHOICE: u32 = 5;
/// The FidelityFX path's render size: the console check's result (Series S)
/// is widened, stored as the FidelityFX quality mode, and tested, and a
/// conditional jump takes the scaled branch, whose scale multiplies the
/// output size. The store is the place's bytes 3 to 9; the jump, bytes 12
/// (its opcode) and 13 (its distance).
const UPSCALER_SIZING: u32 = 0x028e_f69a;
const SIZING_STORE: std::ops::Range<usize> = 3..10;
const SIZING_JUMP: usize = 12;
/// The native branch's load of 1.0, and the scaled branch's of the scale.
const NATIVE_LOAD: u32 = 0x028e_f6a8;
const SCALED_LOAD: u32 = 0x028e_f6b2;
/// The scale the scaled branch loads (shared with other code: not written).
const UPSCALER_SCALE: u32 = 0x0615_d95c;

/// The x86-64 encodings our replacements use: load a 32-bit constant into
/// the accumulator, do nothing, jump a short distance.
const MOV_EAX_IMM32: u8 = 0xb8;
const NOP: u8 = 0x90;
const JMP_SHORT: u8 = 0xeb;

/// What a build holds where the upscaler patch goes: the build's hash and,
/// for each instruction patched, the hash of its stock bytes (`same`).
struct Stock<'a> {
    build: &'a str,
    renderer_selection: &'a str,
    upscaler_sizing: &'a str,
    native_load: &'a str,
    scaled_load: &'a str,
}

/// The supported retail build's.
const RETAIL: Stock<'static> = Stock {
    build: RETAIL_BUILD,
    renderer_selection: "f0633abf37df17124108b1e37be64645bd4b2412b0524c2f89334f22bd948464",
    upscaler_sizing: "2ae2ac102d12ca003cb218413ef018e4dba68fc2003654a30c1743e13a820ffd",
    native_load: "2579c24601d173b086a8136edf54ffd82720e6bf186f9139274bb09a45ddadf5",
    scaled_load: "e218882c95d2e16163a409d55b3451519414eb87c4d6220785dc09852d4e14f4",
};

/// Makes the title render through its FidelityFX upscaler path (renderer 7,
/// as Series S does) with the scene at `mode`'s scale of the output, so that
/// DLSS can take its upscale dispatches over (`victory_hle::modules::fidelityfx`).
/// On Series X the stock title uses its own temporal pass instead and never
/// loads its upscaler. The FidelityFX quality mode the title notes is
/// `mode`'s; the temporal scene scale (the title's own pass, and the 1440p
/// output's patch where that is built) is set to the same scale, as the stock
/// Series X image pairs them (2/3 each). All or nothing: the build, every
/// patched instruction (against its stock bytes' hash) and every range to be
/// written are checked first, and an error leaves `image` as it was.
///
/// From DexNom's fork (DexNom/VIctory, `apps/victory/src/native_1440p.rs`,
/// `prepare_fsr_4k`, GPL-3.0-or-later), there for a 4K output.
pub(super) fn prepare_upscaler(image: &mut PreparedImage, mode: Mode) -> Result<()> {
    patch_upscaler(image, mode, &RETAIL)
}

/// The stock Series X scene scale, derived: its 2560-wide scene over its 3840-wide output (2/3),
/// which its upscaler scale and temporal scene scale both hold.
const STOCK_SCENE_SCALE: f32 = 2560.0 / 3840.0;

/// `prepare_upscaler` against the build `stock` describes.
fn patch_upscaler(image: &mut PreparedImage, mode: Mode, stock: &Stock) -> Result<()> {
    ensure!(
        image.original_hash.to_hex().as_str() == stock.build,
        "the upscaler path is only known for the retail build {}",
        stock.build
    );
    ensure!(same(&field::<8>(image, RENDERER_SELECTION)?[..], stock.renderer_selection), "unexpected renderer selection");
    ensure!(same(&field::<14>(image, UPSCALER_SIZING)?[..], stock.upscaler_sizing), "unexpected upscaler render size selection");
    ensure!(same(&field::<10>(image, NATIVE_LOAD)?[..], stock.native_load), "unexpected upscaler native scale load");
    ensure!(same(&field::<8>(image, SCALED_LOAD)?[..], stock.scaled_load), "unexpected upscaler scaled load");
    ensure!(*field::<4>(image, UPSCALER_SCALE)? == STOCK_SCENE_SCALE.to_le_bytes(), "unexpected upscaler scale");
    let temporal = f32::from_le_bytes(*field::<4>(image, TEMPORAL_SCALE)?);
    ensure!(
        temporal == STOCK_SCENE_SCALE || temporal == 1.0,
        "unexpected temporal scene scale {temporal} (stock 2/3, or 1 with the 1440p output)"
    );
    let scale = mode.scale().to_le_bytes();
    // Only the bytes that change are ours; what moves is taken from the image.
    let store = field::<14>(image, UPSCALER_SIZING)?[SIZING_STORE].to_vec();
    let mut sizing = vec![MOV_EAX_IMM32, mode.ffx_mode(), 0, 0, 0];
    sizing.extend_from_slice(&store);
    let mut writes: Vec<(u32, Vec<u8>)> = vec![
        // Renderer 7 every time it is asked (the console check has run): the
        // loaded number becomes 7, and the replacement after it does nothing.
        (RENDERER_SELECTION + RENDERER_NUMBER, vec![7]),
        (RENDERER_SELECTION + RENDERER_CHOICE, vec![NOP; 3]),
        // The mode is loaded as a constant and stored as before (the store
        // moved up to make room), and the conditional jump to the scaled
        // branch becomes an unconditional one over the same distance.
        (UPSCALER_SIZING, sizing),
        (UPSCALER_SIZING + SIZING_JUMP as u32, vec![JMP_SHORT]),
    ];
    if mode != Mode::Quality {
        // The jump leaves the native branch unreached: its load's operand
        // becomes a literal of ours, which the scaled branch loads.
        let literal = NATIVE_LOAD + 4;
        writes.push((literal, scale.to_vec()));
        let displacement = literal as i32 - (SCALED_LOAD as i32 + 8);
        writes.push((SCALED_LOAD + 4, displacement.to_le_bytes().to_vec()));
    }
    writes.push((TEMPORAL_SCALE, scale.to_vec()));
    // Every range resolved before the first byte is written.
    let ranges = writes.iter().map(|(rva, bytes)| file_range(image, *rva, bytes.len())).collect::<Result<Vec<_>>>()?;
    for (range, (_, bytes)) in ranges.into_iter().zip(&writes) {
        image.bytes[range].copy_from_slice(bytes);
    }
    Ok(())
}

/// The `N` bytes of `image` at `rva`, to check and patch in place.
pub(super) fn field<const N: usize>(image: &mut PreparedImage, rva: u32) -> Result<&mut [u8; N]> {
    let range = file_range(image, rva, N)?;
    (&mut image.bytes[range]).try_into().context("invalid display profile field length")
}

/// Where the `len` bytes at `rva` lie in `image`'s file: all of them backed by
/// the file, in one run (the last byte maps as far past the first as it lies
/// past it in memory).
fn file_range(image: &PreparedImage, rva: u32, len: usize) -> Result<std::ops::Range<usize>> {
    let start = image.info.rva_to_file_offset(rva).context("display profile field missing")?;
    let last = u32::try_from(len.saturating_sub(1)).context("incomplete display profile field")?;
    let whole = rva
        .checked_add(last)
        .and_then(|end| image.info.rva_to_file_offset(end))
        .is_some_and(|end| start.checked_add(last) == Some(end));
    let start = start as usize;
    let end = start + len;
    ensure!(whole && end <= image.bytes.len(), "incomplete display profile field");
    Ok(start..end)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Every upscaler mode, each patched in on its own image.
    pub(crate) const MODES: [Mode; 5] =
        [Mode::Dlaa, Mode::Quality, Mode::Balanced, Mode::Performance, Mode::UltraPerformance];

    /// What `prepare_upscaler` made of `original` in `image`: renderer 7, the
    /// mode loaded and stored where the stock store was, the jump made
    /// unconditional over the same distance and landing on the scaled load,
    /// which loads the mode's scale; the shared constant untouched; no byte
    /// changed outside the places written; and a second patch refused
    /// without a write. Also used by the 1440p module's test, after its own patch.
    pub(crate) fn assert_upscaler_patched(original: &PreparedImage, image: &mut PreparedImage, mode: Mode) {
        assert_patched_for(original, image, mode, &RETAIL);
    }

    /// `assert_upscaler_patched` for the build `stock` describes.
    fn assert_patched_for(original: &PreparedImage, image: &mut PreparedImage, mode: Mode, stock: &Stock) {
        let mut was = original.clone();
        let (was_renderer, was_sizing) =
            (*field::<8>(&mut was, RENDERER_SELECTION).unwrap(), *field::<14>(&mut was, UPSCALER_SIZING).unwrap());
        let renderer = *field::<8>(image, RENDERER_SELECTION).unwrap();
        let (number, choice) = (RENDERER_NUMBER as usize, RENDERER_CHOICE as usize);
        assert_eq!(renderer[number], 7);
        assert_eq!(renderer[choice..], [NOP; 3]);
        assert_eq!(renderer[..number], was_renderer[..number]);
        assert_eq!(renderer[number + 1..choice], was_renderer[number + 1..choice]);
        let sizing = *field::<14>(image, UPSCALER_SIZING).unwrap();
        assert_eq!(sizing[..5], [MOV_EAX_IMM32, mode.ffx_mode(), 0, 0, 0]);
        assert_eq!(sizing[5..SIZING_JUMP], was_sizing[SIZING_STORE]);
        assert_eq!((sizing[SIZING_JUMP], sizing[SIZING_JUMP + 1]), (JMP_SHORT, was_sizing[SIZING_JUMP + 1]));
        // The jump lands on the scaled load.
        assert_eq!(UPSCALER_SIZING + 14 + u32::from(sizing[SIZING_JUMP + 1]), SCALED_LOAD);
        // Nothing changed but the places written.
        let mut written = vec![
            (RENDERER_SELECTION + RENDERER_NUMBER, 1),
            (RENDERER_SELECTION + RENDERER_CHOICE, 3),
            (UPSCALER_SIZING, SIZING_JUMP + 1),
            (TEMPORAL_SCALE, 4),
        ];
        if mode != Mode::Quality {
            written.extend([(NATIVE_LOAD + 4, 4), (SCALED_LOAD + 4, 4)]);
        }
        let ranges: Vec<_> = written.into_iter().map(|(rva, len)| file_range(original, rva, len).unwrap()).collect();
        for (at, (a, b)) in original.bytes.iter().zip(&image.bytes).enumerate() {
            assert!(a == b || ranges.iter().any(|r| r.contains(&at)), "a byte changed outside the places written");
        }
        let disp = i32::from_le_bytes(*field::<4>(image, SCALED_LOAD + 4).unwrap());
        let loaded = (SCALED_LOAD + 8).wrapping_add_signed(disp);
        assert_eq!(f32::from_le_bytes(*field::<4>(image, loaded).unwrap()), mode.scale());
        assert_eq!(f32::from_le_bytes(*field::<4>(image, TEMPORAL_SCALE).unwrap()), mode.scale());
        // The shared constant stays as it was.
        assert_eq!(*field::<4>(image, UPSCALER_SCALE).unwrap(), STOCK_SCENE_SCALE.to_le_bytes());
        // Patched once, refused, and the refusal writes nothing.
        let before = image.bytes.clone();
        assert!(patch_upscaler(image, mode, stock).is_err());
        assert!(image.bytes == before, "a refused second patch changed the image");
    }

    /// The hashes a made-up build is checked against, owned (`Stock` borrows them).
    struct Hashes([String; 5]);

    impl Hashes {
        fn stock(&self) -> Stock<'_> {
            let [build, renderer_selection, upscaler_sizing, native_load, scaled_load] = &self.0;
            Stock { build, renderer_selection, upscaler_sizing, native_load, scaled_load }
        }
    }

    /// A made-up image, none of it from the title: at the patched places
    /// only, filler bytes of ours (the scaled load's operand pointing at the
    /// upscaler scale, as the stock one does) and the two scales, each piece
    /// in a section of its own; with the hashes that make it a stock build.
    /// With `split`, the section that holds the native load ends four bytes
    /// into it.
    fn synthetic(split: bool) -> (PreparedImage, Hashes) {
        let third = STOCK_SCENE_SCALE.to_le_bytes();
        let renderer = [0x11; 8];
        let mut sizing = [0x22; 14];
        // The jump's distance, which the patch keeps: to the scaled load.
        sizing[SIZING_JUMP + 1] = u8::try_from(SCALED_LOAD - UPSCALER_SIZING - 14).unwrap();
        let native = [0x33; 10];
        let mut scaled = vec![0x44; 4];
        scaled.extend_from_slice(&(UPSCALER_SCALE as i32 - (SCALED_LOAD as i32 + 8)).to_le_bytes());
        let hash = |bytes: &[u8]| blake3::hash(bytes).to_hex().to_string();
        let build = blake3::hash(b"a made-up build");
        let hashes = Hashes([build.to_hex().to_string(), hash(&renderer), hash(&sizing), hash(&native), hash(&scaled)]);
        let mut branches = Vec::new();
        branches.extend_from_slice(&sizing);
        branches.extend_from_slice(&native);
        branches.extend_from_slice(&scaled);
        let pieces: [(u32, &[u8]); 4] = [
            (RENDERER_SELECTION, &renderer),
            (UPSCALER_SIZING, &branches),
            (UPSCALER_SCALE, &third),
            (TEMPORAL_SCALE, &third),
        ];
        // A stand-in for the headers, so that no piece starts the file.
        let mut bytes = vec![0xcc; 64];
        let mut sections = Vec::new();
        let section = |rva: u32, file_offset: u32, size: u32| victory_pe::SectionInfo {
            name: String::new(),
            rva,
            virtual_size: size,
            file_offset,
            file_size: size,
            characteristics: 0,
        };
        for (rva, piece) in pieces {
            let file_offset = u32::try_from(bytes.len()).unwrap();
            let size = u32::try_from(piece.len()).unwrap();
            bytes.extend_from_slice(piece);
            if split && rva == UPSCALER_SIZING {
                let head = NATIVE_LOAD + 4 - rva;
                let tail = SCALED_LOAD - rva;
                sections.push(section(rva, file_offset, head));
                sections.push(section(SCALED_LOAD, file_offset + tail, size - tail));
            } else {
                sections.push(section(rva, file_offset, size));
            }
        }
        let image = PreparedImage {
            info: victory_pe::ImageInfo {
                machine: 0,
                characteristics: 0,
                timestamp: 0,
                subsystem: 0,
                dll_characteristics: 0,
                image_base: 0,
                size_of_image: 0,
                entry_rva: 0,
                stack_reserve: 0,
                stack_commit: 0,
                sections,
                imports: Vec::new(),
                delay_imports: Vec::new(),
                exports: Vec::new(),
                tls: None,
                has_relocations: false,
                has_exception_table: false,
                has_load_config: false,
                codeview: None,
            },
            bytes,
            original_hash: build,
        };
        (image, hashes)
    }

    /// On the made-up image, every mode patches as on the retail one.
    #[test]
    fn a_synthetic_stock_image_patches_at_every_mode() {
        for mode in MODES {
            let (original, hashes) = synthetic(false);
            let mut image = original.clone();
            patch_upscaler(&mut image, mode, &hashes.stock()).unwrap();
            assert_patched_for(&original, &mut image, mode, &hashes.stock());
        }
    }

    /// The retail build's own checks refuse the made-up image: it is not
    /// that build.
    #[test]
    fn the_retail_checks_refuse_a_made_up_image() {
        let (mut image, _) = synthetic(false);
        let before = image.bytes.clone();
        assert!(prepare_upscaler(&mut image, Mode::Balanced).is_err());
        assert!(image.bytes == before);
    }

    /// All or nothing: whatever check refuses the patch (another build, any
    /// field not as expected, a range the file does not hold whole), the
    /// image is left byte for byte as it was.
    #[test]
    fn a_rejected_patch_leaves_the_image_as_it_was() {
        let (_, hashes) = synthetic(false);
        let mut cases: Vec<(&str, PreparedImage)> = Vec::new();
        let (mut other_build, _) = synthetic(false);
        other_build.original_hash = blake3::hash(b"another build");
        cases.push(("another build", other_build));
        for (name, rva) in [
            ("renderer selection", RENDERER_SELECTION),
            ("upscaler sizing", UPSCALER_SIZING),
            ("native load", NATIVE_LOAD),
            ("scaled load", SCALED_LOAD),
            ("upscaler scale", UPSCALER_SCALE),
            ("temporal scale", TEMPORAL_SCALE),
        ] {
            let (mut image, _) = synthetic(false);
            field::<1>(&mut image, rva).unwrap()[0] ^= 0x01;
            cases.push((name, image));
        }
        let (mut half, _) = synthetic(false);
        *field::<4>(&mut half, TEMPORAL_SCALE).unwrap() = 0.5f32.to_le_bytes();
        cases.push(("temporal scale neither stock nor 1440p", half));
        cases.push(("a section ending inside the native load", synthetic(true).0));
        for (name, image) in cases {
            for mode in MODES {
                let mut image = image.clone();
                let before = image.bytes.clone();
                assert!(patch_upscaler(&mut image, mode, &hashes.stock()).is_err(), "{name}, {}: not refused", mode.name());
                assert!(image.bytes == before, "{name}, {}: refused after writing", mode.name());
            }
        }
    }

    /// With the retail executable named by `VICTORY_DISPLAY_PROFILE_TEST_IMAGE`
    /// (not part of the repository): the upscaler path at each scale on the
    /// stock image (4K output); without it, nothing is checked. The 1440p
    /// module checks the same after its patch.
    #[test]
    fn patches_the_upscaler_path() {
        let Some(path) = std::env::var_os("VICTORY_DISPLAY_PROFILE_TEST_IMAGE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let original = victory_loader::prepare(&bytes).unwrap();
        for mode in MODES {
            let mut image = original.clone();
            prepare_upscaler(&mut image, mode).unwrap();
            assert_upscaler_patched(&original, &mut image, mode);
        }
    }

    /// With the retail executable (as above): each stock hash accepts what a
    /// comparison with the bytes did, the retail image's bytes there, and
    /// refuses them with any one byte changed.
    #[test]
    fn the_stock_hashes_accept_exactly_the_retail_bytes() {
        let Some(path) = std::env::var_os("VICTORY_DISPLAY_PROFILE_TEST_IMAGE") else {
            return;
        };
        let mut image = victory_loader::prepare(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(image.original_hash.to_hex().as_str(), RETAIL.build);
        let sites = [
            (RENDERER_SELECTION, 8, RETAIL.renderer_selection),
            (UPSCALER_SIZING, 14, RETAIL.upscaler_sizing),
            (NATIVE_LOAD, 10, RETAIL.native_load),
            (SCALED_LOAD, 8, RETAIL.scaled_load),
        ];
        for (rva, len, hash) in sites {
            let range = file_range(&image, rva, len).unwrap();
            let there = &mut image.bytes[range];
            assert!(same(there, hash), "{rva:#x}");
            for at in 0..len {
                there[at] ^= 0x01;
                assert!(!same(there, hash), "{rva:#x} with byte {at} changed");
                there[at] ^= 0x01;
            }
        }
    }

    /// Paths whose first segment names nothing of this crate or of a
    /// dependency's private modules: the standard library, outside crates,
    /// and names this file brings in with a checked `use`.
    const FREE_ROOTS: [&str; 14] =
        ["std", "core", "anyhow", "blake3", "Mode", "Self", "f32", "i32", "u32", "u8", "usize", "Vec", "String", "Option"];
    /// The only items of our own crates this file may name, each in open code.
    const OPEN_PATHS: [&str; 5] = [
        "victory_hle::streamline::Mode",
        "victory_loader::PreparedImage",
        "victory_loader::prepare",
        "victory_pe::ImageInfo",
        "victory_pe::SectionInfo",
    ];

    /// `source` as Rust tokens, without comments, strings, char literals and
    /// numbers: identifiers, `::`, and single punctuation characters.
    fn tokens(source: &str) -> Vec<String> {
        let chars: Vec<char> = source.chars().collect();
        let at = |i: usize| chars.get(i).copied();
        // Past a string's closing quote (and `hashes` hashes, for a raw one).
        let skip_string = |mut i: usize, raw: bool, hashes: usize| {
            while i < chars.len() {
                if !raw && chars[i] == '\\' {
                    i += 2;
                } else if chars[i] == '"' && (1..=hashes).all(|h| at(i + h) == Some('#')) {
                    return i + 1 + hashes;
                } else {
                    i += 1;
                }
            }
            i
        };
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                i += 1;
            } else if c == '/' && at(i + 1) == Some('/') {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            } else if c == '/' && at(i + 1) == Some('*') {
                let mut depth = 0;
                while i < chars.len() {
                    if chars[i] == '/' && at(i + 1) == Some('*') {
                        depth += 1;
                        i += 2;
                    } else if chars[i] == '*' && at(i + 1) == Some('/') {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            } else if c == '"' {
                i = skip_string(i + 1, false, 0);
            } else if c == '\'' {
                if at(i + 1) == Some('\\') {
                    // An escaped char: past the escape, to the closing tick.
                    i += 3;
                    while i < chars.len() && chars[i] != '\'' {
                        i += 1;
                    }
                    i += 1;
                } else if at(i + 2) == Some('\'') {
                    i += 3;
                } else {
                    // A lifetime's tick.
                    i += 1;
                }
            } else if c.is_ascii_digit() {
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
            } else if c.is_alphabetic() || c == '_' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let hashes = chars[i..].iter().take_while(|&&h| h == '#').count();
                if matches!(word.as_str(), "r" | "br") && at(i + hashes) == Some('"') {
                    i = skip_string(i + hashes + 1, true, hashes);
                } else if word == "b" && at(i) == Some('"') {
                    i = skip_string(i + 1, false, 0);
                } else if !(word == "b" && at(i) == Some('\'')) {
                    out.push(word);
                }
            } else if c == ':' && at(i + 1) == Some(':') {
                out.push("::".to_string());
                i += 2;
            } else {
                out.push(c.to_string());
                i += 1;
            }
        }
        out
    }

    fn is_ident(token: &str) -> bool {
        token.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
    }

    /// Every path in `tokens`, its segments joined by `::`: one ending in
    /// `::{` or `::*` keeps the `{` or `*`, one starting with `::` keeps
    /// that, and one cut short by a turbofish loses its last `::`.
    fn paths(tokens: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let after_ident = i > 0 && is_ident(&tokens[i - 1]);
            let starts = (is_ident(&tokens[i]) && (i == 0 || tokens[i - 1] != "::"))
                || (tokens[i] == "::" && !after_ident);
            if !starts {
                i += 1;
                continue;
            }
            let mut path = String::new();
            let mut j = i;
            if tokens[j] == "::" {
                path.push_str("::");
                j += 1;
            }
            while let Some(token) = tokens.get(j) {
                if is_ident(token) {
                    path.push_str(token);
                    j += 1;
                } else if (token == "{" || token == "*") && path.ends_with("::") {
                    path.push_str(token);
                    j += 1;
                    break;
                } else {
                    break;
                }
                if tokens.get(j).is_some_and(|t| t == "::") {
                    path.push_str("::");
                    j += 1;
                } else {
                    break;
                }
            }
            out.push(path.trim_end_matches("::").to_string());
            i = j.max(i + 1);
        }
        out
    }

    /// What in `source` reaches outside the file and the open items above:
    /// a module or crate brought in (`mod`, `extern`, `include!`), a path
    /// through this crate (`crate`, `self`, `super` past this file's own
    /// items) or through a dependency other than the open items, a path
    /// from the root (`::name`).
    fn reaches_outside(source: &str) -> Vec<String> {
        reaches_outside_but(source, &[])
    }

    /// [`reaches_outside`] with the paths `also` allowed besides the open
    /// items (exactly as written, braces kept): the 1440p module's check,
    /// which may name this module's items.
    pub(crate) fn reaches_outside_but(source: &str, also: &[&str]) -> Vec<String> {
        let tokens = tokens(source);
        let mut found = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            let next = |n: usize| tokens.get(i + n).map(String::as_str);
            let bad = match token.as_str() {
                "extern" | "include" | "include_bytes" => true,
                "mod" => !(next(1) == Some("tests") && next(2) == Some("{")),
                _ => false,
            };
            if bad {
                found.push(format!("{token} {}", next(1).unwrap_or_default()));
            }
        }
        for path in paths(&tokens) {
            if !path.contains("::") {
                continue;
            }
            let root = path.split("::").next().unwrap_or_default();
            let open = path == "super::*"
                || FREE_ROOTS.contains(&root)
                || OPEN_PATHS.contains(&path.as_str())
                || also.contains(&path.as_str());
            if !open {
                found.push(path);
            }
        }
        found
    }

    /// The public tree has this file without the 1440p module or any of the
    /// files kept out of it: the whole file, its tests included, names only
    /// its own items, outside crates and the open items above.
    #[test]
    fn stands_alone() {
        let found = reaches_outside(include_str!("upscaler_patch.rs"));
        assert!(found.is_empty(), "the upscaler patch reaches outside itself: {found:?}");
    }

    /// The check sees each way of reaching outside, and is not tripped by
    /// the same words in a comment, a string or a char.
    #[test]
    fn the_standalone_check_sees_what_it_claims() {
        for reach in [
            "use victory_loader::write_sites::Site;",
            "use victory_loader::{write_sites};",
            "use victory_hle::{streamline::Mode};",
            "fn f() { crate :: run :: native_1440p :: prepare(); }",
            "fn f() { super :: super :: native_1440p::WIDTH; }",
            "fn f() { self::native_1440p::prepare(); }",
            "fn f() { ::victory_gpu::port::x(); }",
            "mod native_1440p;",
            "mod tests;",
            "#[path = \"other.rs\"] mod other;",
            "extern crate victory_gpu;",
            "include!(\"other.rs\");",
            "pub(super) mod tests { use victory_gpu::titlewrite::X; }",
        ] {
            assert!(!reaches_outside(reach).is_empty(), "missed: {reach}");
        }
        for quiet in [
            "// crate::native_1440p::prepare and super::super::x\nfn f() {}",
            "/* use victory_loader::write_sites; /* nested */ mod x; */ fn f() {}",
            "fn f() -> &'static str { \"crate::native_1440p::prepare\" }",
            "fn f() -> char { '\"' } fn g() -> char { '\\'' }",
            "const S: &str = r#\"super::super::x\"#;",
            "use super::*; pub(crate) fn f() {}",
        ] {
            let found = reaches_outside(quiet);
            assert!(found.is_empty(), "false alarm {found:?}: {quiet}");
        }
    }
}
