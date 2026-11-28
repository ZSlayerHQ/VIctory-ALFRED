//! The title's image patched for a 1440p output and for its FidelityFX path
//! at an upscaler's scale: absent from this tree.
//!
//! The private build has a module here that, before the image is loaded,
//! changes the title's code so that the Series X profile renders and outputs
//! 2560x1440 (`VICTORY_NATIVE_1440P`, `VICTORY_RENDER_1080P`), and so that
//! the title renders through its FidelityFX path at the scale of
//! `VICTORY_DLSS_MODE` (what DLSS, DLSS frame generation, FSR 3 frame
//! generation and `VICTORY_FSR_PATH` take over or use). The places it
//! changes are the title's own, so the public tree leaves it out. Here the
//! 1440p switches are refused where they are read (`run.rs`), and the
//! FidelityFX path is never patched in: the run goes on as the base does
//! when that patch fails, with the title's own output and upscaling.

use victory_hle::streamline::Mode;
use victory_loader::PreparedImage;

/// Patches the title's FidelityFX path in at `_mode`'s scale: never here.
pub fn prepare_upscaler(_prepared: &mut PreparedImage, _mode: Mode) -> anyhow::Result<()> {
    anyhow::bail!("this build does not have that patch (the public tree leaves it out)")
}
