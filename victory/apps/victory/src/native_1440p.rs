//! A 2560x1440 SDR display for the supported Series X retail image (the game,
//! 1.0.2.1): the title presents at 3840x2160 on Series X, its scene drawn at
//! 2560x1440 and scaled up by its own temporal pass. With 1440p asked for
//! (`VICTORY_NATIVE_1440P=1`, or `VICTORY_RENDER_1080P=1` for a 1920x1080
//! scene scaled to 1440p), the image's SDR display mode, the scanout
//! reservation, the temporal scene scale and the Settings viewport load are
//! patched before it is mapped: the title then makes a 1440p output, and no
//! 4K pass runs. The same four places make a 1920x1080 output with its scene
//! at 1920x1080 (`VICTORY_OUTPUT_1080P=1`, [`prepare_output`]): no upscaling.
//!
//! Ported from DexNom's fork (DexNom/VIctory, `apps/victory/src/native_1440p.rs`,
//! GPL-3.0-or-later), its native 1440p part only. Every field is checked
//! against the stock bytes first: another build is refused, not patched.
//! The upscaler path's patch, and what both share, are in `upscaler_patch`.

use anyhow::{Result, ensure};
use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
use victory_loader::PreparedImage;

use super::upscaler_patch::{RETAIL_BUILD, TEMPORAL_SCALE, field, same};

pub(super) const WIDTH: u32 = 2560;
pub(super) const HEIGHT: u32 = 1440;
/// The stock Series X output's width, 3840.
const STOCK_OUTPUT_WIDTH: u32 = 3840;
/// The stock temporal scene scale, derived: the 2560-wide scene over the 3840-wide output (2/3).
const STOCK_SCENE_SCALE: f32 = WIDTH as f32 / STOCK_OUTPUT_WIDTH as f32;
/// The 1080p output's size, which is also its scene's (`prepare_output`).
pub(super) const OUTPUT_1080P: (u32, u32) = (1920, 1080);
const SDR_MODE: u32 = 0x0615_c100;
const STOCK_SDR_MODE: &str = "abfb5438d8153b6a0b8d08f33f4f0066e5383925b55aacc5f669131f3f51d4cf";
/// The scanout reservation: a size loaded as a constant (the place's bytes 1
/// to 4), which the title reserves the scanout buffers with.
const SCANOUT_RESERVATION: u32 = 0x0247_ad4a;
const STOCK_SCANOUT_RESERVATION: &str = "092b2fd9854a87744a1283c450072b33fcee8713120f25e33a760e6e3c0a262c";
/// The Settings viewport load: it reads the UI size from a place relative to
/// itself (the distance is the place's bytes 4 to 7).
const UI_VIEWPORT_LOAD: u32 = 0x0580_48d4;
const STOCK_UI_VIEWPORT_LOAD: &str = "977fec47bc1ce1900aa670cbd98438e24f179b3aa5fb248d5f553a1801e26fd1";

/// Patches `image` for a 2560x1440 output; the scene is drawn at 2560x1440,
/// or at 1920x1080 with `render_1080p`. Returns the scene's size.
pub(super) fn prepare(image: &mut PreparedImage, render_1080p: bool) -> Result<(u32, u32)> {
    let scene = if render_1080p { (1920, 1080) } else { (WIDTH, HEIGHT) };
    prepare_output(image, (WIDTH, HEIGHT), scene)
}

/// Patches `image` for an `output`-sized SDR output with the scene drawn at
/// `scene` (no wider than the output): the title's temporal pass then scales
/// by their ratio, 1 (no upscaling) when they are the same. Returns the
/// scene's size. All or nothing: every place is checked first, and an error
/// leaves `image` as it was.
pub(super) fn prepare_output(image: &mut PreparedImage, output: (u32, u32), scene: (u32, u32)) -> Result<(u32, u32)> {
    let (width, height) = output;
    ensure!(
        image.original_hash.to_hex().as_str() == RETAIL_BUILD,
        "{height}p output is only known for the retail build {RETAIL_BUILD}"
    );
    ensure!(scene.0 <= width && scene.0 > 0, "a {}-wide scene for a {width}-wide output", scene.0);

    ensure!(same(&field::<16>(image, SDR_MODE)?[..], STOCK_SDR_MODE), "unexpected Series X SDR mode");

    // Two 64 KiB-aligned, tiled RGBA8 scanout buffers. The reservation is
    // rounded to 2 MiB, matching the title's allocation check.
    let texture = Texture::new(&TextureDesc {
        dimension: Dimension::Tex2d,
        width,
        height,
        depth_or_array: 1,
        mips: 1,
        format: 28,
        samples: 1,
        swizzle: Swizzle::SW_64KB_D_X,
    })
    .map_err(|error| anyhow::anyhow!("{height}p scanout layout: {error:?}"))?;
    let pool_size =
        u32::try_from((texture.size.next_multiple_of(64 * 1024) * 2).next_multiple_of(2 << 20))?;
    ensure!(
        same(&field::<5>(image, SCANOUT_RESERVATION)?[..], STOCK_SCANOUT_RESERVATION),
        "unexpected Series X scanout reservation"
    );

    ensure!(
        *field::<4>(image, TEMPORAL_SCALE)? == STOCK_SCENE_SCALE.to_le_bytes(),
        "unexpected Series X temporal scene scale"
    );

    // Settings lays out its preview panels with a viewport setter of its
    // own that loads the legacy UI size (3840x2160) even when a smaller UI
    // surface was made: it reads the selected SDR mode instead.
    ensure!(
        same(&field::<8>(image, UI_VIEWPORT_LOAD)?[..], STOCK_UI_VIEWPORT_LOAD),
        "unexpected Settings viewport size load"
    );

    // Every place checked whole: the writes below, each inside one, cannot fail.
    let mode = field::<16>(image, SDR_MODE)?;
    mode[..4].copy_from_slice(&width.to_le_bytes());
    mode[4..8].copy_from_slice(&height.to_le_bytes());
    field::<5>(image, SCANOUT_RESERVATION)?[1..].copy_from_slice(&pool_size.to_le_bytes());
    *field::<4>(image, TEMPORAL_SCALE)? = (scene.0 as f32 / width as f32).to_le_bytes();
    let displacement = (SDR_MODE as i32) - (UI_VIEWPORT_LOAD as i32 + 8);
    field::<8>(image, UI_VIEWPORT_LOAD)?[4..].copy_from_slice(&displacement.to_le_bytes());
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::upscaler_patch::prepare_upscaler;
    use super::super::upscaler_patch::tests::{MODES, assert_upscaler_patched, reaches_outside_but};

    /// The three outputs the launcher offers besides the stock 4K one, as
    /// `prepare_output` takes them: 1440p, 1080p upscaled to 1440p, and 1080p.
    const OUTPUTS: [((u32, u32), (u32, u32)); 3] =
        [((WIDTH, HEIGHT), (WIDTH, HEIGHT)), ((WIDTH, HEIGHT), (1920, 1080)), (OUTPUT_1080P, OUTPUT_1080P)];

    /// The retail executable named by `VICTORY_DISPLAY_PROFILE_TEST_IMAGE`
    /// (not part of the repository); without it, nothing is checked.
    fn retail() -> Option<Vec<u8>> {
        std::env::var_os("VICTORY_DISPLAY_PROFILE_TEST_IMAGE").map(|path| std::fs::read(path).unwrap())
    }

    /// With the retail executable: the four fields hold what each output
    /// needs, `prepare` is the 1440p one, and a second patch is refused
    /// without a write.
    #[test]
    fn patches_the_retail_image_for_each_output() {
        let Some(bytes) = retail() else {
            return;
        };
        for (output, scene) in OUTPUTS {
            let mut image = victory_loader::prepare(&bytes).unwrap();
            assert_eq!(prepare_output(&mut image, output, scene).unwrap(), scene);
            let mode = *field::<16>(&mut image, SDR_MODE).unwrap();
            assert_eq!(mode[..4], output.0.to_le_bytes());
            assert_eq!(mode[4..8], output.1.to_le_bytes());
            let scale = f32::from_le_bytes(*field::<4>(&mut image, TEMPORAL_SCALE).unwrap());
            assert_eq!((output.0 as f32 * scale) as u32, scene.0);
            if scene == output {
                assert_eq!(scale, 1.0, "no upscaling");
            }
            let load = *field::<8>(&mut image, UI_VIEWPORT_LOAD).unwrap();
            let disp = i32::from_le_bytes(load[4..].try_into().unwrap());
            assert_eq!(UI_VIEWPORT_LOAD.wrapping_add(8).wrapping_add_signed(disp), SDR_MODE);
            // Patched once, the image is no longer the stock one: refused, nothing written.
            let before = image.bytes.clone();
            assert!(prepare_output(&mut image, output, scene).is_err());
            assert!(image.bytes == before);
        }
        for render_1080p in [false, true] {
            let (mut by_prepare, mut by_output) = (victory_loader::prepare(&bytes).unwrap(), victory_loader::prepare(&bytes).unwrap());
            let scene = prepare(&mut by_prepare, render_1080p).unwrap();
            let (output, expected) = OUTPUTS[usize::from(render_1080p)];
            assert_eq!(scene, expected);
            prepare_output(&mut by_output, output, expected).unwrap();
            assert!(by_prepare.bytes == by_output.bytes, "prepare is the 1440p output's");
        }
    }

    /// With the retail executable: whatever refuses an output (another
    /// build, any of the four places not as expected, a scene wider than the
    /// output), the image is left byte for byte as it was. The places are
    /// spoiled in this test's copy of the image only.
    #[test]
    fn a_refused_output_leaves_the_image_as_it_was() {
        let Some(bytes) = retail() else {
            return;
        };
        for (output, scene) in OUTPUTS {
            let mut cases = Vec::new();
            let mut other = victory_loader::prepare(&bytes).unwrap();
            other.original_hash = blake3::hash(b"another build");
            cases.push(("another build", other, scene));
            for (name, rva) in [
                ("SDR mode", SDR_MODE),
                ("scanout reservation", SCANOUT_RESERVATION),
                ("temporal scale", TEMPORAL_SCALE),
                ("Settings viewport load", UI_VIEWPORT_LOAD),
            ] {
                let mut image = victory_loader::prepare(&bytes).unwrap();
                field::<1>(&mut image, rva).unwrap()[0] ^= 0x01;
                cases.push((name, image, scene));
            }
            cases.push(("a scene wider than the output", victory_loader::prepare(&bytes).unwrap(), (output.0 + 1, output.1)));
            for (name, mut image, scene) in cases {
                let before = image.bytes.clone();
                assert!(prepare_output(&mut image, output, scene).is_err(), "{name} at {output:?}: not refused");
                assert!(image.bytes == before, "{name} at {output:?}: refused after writing");
            }
        }
    }

    /// With the retail executable: the upscaler path at each scale after
    /// the 1440p and the 1080p output patches, checked as `upscaler_patch`'s
    /// own test checks it on the stock image (4K output).
    #[test]
    fn patches_the_upscaler_path_after_1440p() {
        let Some(bytes) = retail() else {
            return;
        };
        for output in [(WIDTH, HEIGHT), OUTPUT_1080P] {
            for mode in MODES {
                let mut image = victory_loader::prepare(&bytes).unwrap();
                prepare_output(&mut image, output, output).unwrap();
                let original = image.clone();
                prepare_upscaler(&mut image, mode).unwrap();
                assert_upscaler_patched(&original, &mut image, mode);
            }
        }
    }

    /// With the retail executable: each stock hash here accepts what a
    /// comparison with the bytes did, the retail image's bytes there, and
    /// refuses them with any one byte changed.
    #[test]
    fn the_stock_hashes_accept_exactly_the_retail_bytes() {
        let Some(bytes) = retail() else {
            return;
        };
        let mut image = victory_loader::prepare(&bytes).unwrap();
        for (rva, hash) in [(SDR_MODE, STOCK_SDR_MODE), (SCANOUT_RESERVATION, STOCK_SCANOUT_RESERVATION), (UI_VIEWPORT_LOAD, STOCK_UI_VIEWPORT_LOAD)] {
            let len = if rva == SDR_MODE { 16 } else if rva == SCANOUT_RESERVATION { 5 } else { 8 };
            let there = field::<16>(&mut image, rva).unwrap();
            let there = &mut there[..len];
            assert!(same(there, hash), "{rva:#x}");
            for at in 0..len {
                there[at] ^= 0x01;
                assert!(!same(there, hash), "{rva:#x} with byte {at} changed");
                there[at] ^= 0x01;
            }
        }
    }

    /// With the retail executable: every output and every upscaler mode (on
    /// the stock output and after each output of the same size as its
    /// scene) makes, byte for byte, the image the code made before its
    /// stock bytes became hashes and its writes held only our bytes (the
    /// whole image's BLAKE3 hash, recorded from that code on 2026-11-28).
    #[test]
    fn the_patched_images_are_the_ones_the_byte_form_made() {
        let Some(bytes) = retail() else {
            return;
        };
        let hash = |i: &PreparedImage| blake3::hash(&i.bytes).to_hex().to_string();
        let outputs = [
            "d4b30d06257d52ba5d48b761ecaa7c318dcea11293c7f69abdb1c3634f83f5f3",
            "c0c46cff64d8f2159f1eb19b7df059c394290cc833013c3ab919f14da00533d9",
            "7ac7d5bc2f03ea39602fddf8251e994fc51738fdf3aa26ebabeff6395e724106",
        ];
        for ((output, scene), expected) in OUTPUTS.into_iter().zip(outputs) {
            let mut image = victory_loader::prepare(&bytes).unwrap();
            prepare_output(&mut image, output, scene).unwrap();
            assert_eq!(hash(&image), expected, "{output:?}, scene {scene:?}");
        }
        let upscaled = [
            [
                "50c47a8d8abc930040be2c1c3e6324dfe2392a7ceedb3c9e627778b672b4ce10",
                "3d0e1deef9cd928d0a892069a3f1f3805febf895f835cc1f494e43db1187f140",
                "e0c9b3e72d81516142408ccc2a961f254b1bf33cfc14b5c8a219711cad1bc3f3",
                "89d68e46cb6c9590b502b21c76e57953aa7c7b66e3a148d6fac1b059e99973f3",
                "363598ca53d14418482f3cad0a208fefa492d0a0469afc8c88d69980059d4e38",
            ],
            [
                "80edeea1eac148056a86aff0bda1f59a01c8f99e6bc3c12c834c6b20944cce58",
                "a3569c9e6372cd3babf2ef039fd64271822bd178d13612d9c73c00e8bd50ea15",
                "564b9faf3d6648dd45d5cac8482be515e6b6a2a501ed64b8edc341311a29338f",
                "c5e6d171f5027d74f6fdfab1c44bf1ebbec559859b9237ad7f239ce3054bb908",
                "ed4e68772c4f2a673f132b3906b301c1156ad325b2b80dd4b71ca0f107716e4f",
            ],
            [
                "0844eeedf6f0c52546343230a2e15accc4409f59cf1ca87017f77c1c698681b5",
                "c5c7e0629a6fa94d9035c786f9b63fc9aa88ac1fd9f9a1fec134b9d90b8b7d74",
                "4252e585561cf4a085de885aa1c029f9538f7a91a8018d65fab1b70156548802",
                "d4d8eb64c541223aebf22d8cb025b562866eaea788fb05e5e42ecfc5fda76e36",
                "ec2b8befadbe10200a3d4d2891e5fca6c1a582137bd02a48a6c4ae7b8b5b9c8a",
            ],
        ];
        for (output, expected) in [None, Some((WIDTH, HEIGHT)), Some(OUTPUT_1080P)].into_iter().zip(upscaled) {
            for (mode, expected) in MODES.into_iter().zip(expected) {
                let mut image = victory_loader::prepare(&bytes).unwrap();
                if let Some(o) = output {
                    prepare_output(&mut image, o, o).unwrap();
                }
                prepare_upscaler(&mut image, mode).unwrap();
                assert_eq!(hash(&image), expected, "{output:?}, {}", mode.name());
            }
        }
    }

    /// The public tree has this file and `upscaler_patch` without the rest
    /// of the private code: the whole file, its tests included, names only
    /// its own items, `upscaler_patch`'s, outside crates and open items.
    #[test]
    fn stands_alone() {
        let found = reaches_outside_but(
            include_str!("native_1440p.rs"),
            &[
                "super::upscaler_patch::{",
                "super::super::upscaler_patch::prepare_upscaler",
                "super::super::upscaler_patch::tests::{",
                "victory_formats::texture::{",
                "Dimension::Tex2d",
                "Swizzle::SW_64KB_D_X",
                "Texture::new",
            ],
        );
        assert!(found.is_empty(), "the 1440p output reaches outside itself: {found:?}");
    }
}
