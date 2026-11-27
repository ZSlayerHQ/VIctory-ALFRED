//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod adapter;
pub mod bounds;
pub mod bvh;
pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
/// A CPU interpreter of GFX10 code, the reference the translator's fixtures are checked against
/// (test and diagnostics builds only).
#[cfg(any(test, feature = "diagnostics"))]
pub mod oracle;
pub mod resource;
pub mod spill;
pub mod structure;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

/// The runtime's identity of a shader container: its BLAKE3 digest, and the
/// 16-hex id logs and crash reports print (the digest's first 8 bytes). The
/// one place that id is computed; the runtime and the offline tools share it.
pub fn container_id(bytes: &[u8]) -> ([u8; 32], String) {
    let digest = blake3::hash(bytes);
    (*digest.as_bytes(), digest.to_hex()[..16].to_owned())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}

/// The translator's own sources, by which translations kept on disk tell
/// whether this translator made them.
pub fn sources() -> &'static [&'static str] {
    &[
        include_str!("adapter.rs"),
        include_str!("bounds.rs"),
        include_str!("bvh.rs"),
        include_str!("dxbc.rs"),
        include_str!("dxil.rs"),
        include_str!("emit/back_edges.rs"),
        include_str!("emit/bitcode.rs"),
        include_str!("emit/bitcode/census.rs"),
        include_str!("emit/bitcode/opt.rs"),
        include_str!("emit/cbv_raw.rs"),
        include_str!("emit/container.rs"),
        include_str!("emit/d16_load_tests.rs"),
        include_str!("emit/diff_tests.rs"),
        include_str!("emit/early_out_tests.rs"),
        include_str!("emit/first_lane.rs"),
        include_str!("emit/host.rs"),
        include_str!("emit/host_wave_tests.rs"),
        include_str!("emit/host_waves.rs"),
        include_str!("emit/image_lod_tests.rs"),
        include_str!("emit/intersect.rs"),
        include_str!("emit/intersect_tests.rs"),
        include_str!("emit/lane_pair_tests.rs"),
        include_str!("emit/link.rs"),
        include_str!("emit/link_tests.rs"),
        include_str!("emit/loop_bound_tests.rs"),
        include_str!("emit/lower.rs"),
        include_str!("emit/lower/cvt.rs"),
        include_str!("emit/lower/flush.rs"),
        include_str!("emit/lower/linked.rs"),
        include_str!("emit/lower/norm.rs"),
        include_str!("emit/lower/d16.rs"),
        include_str!("emit/lower/slots.rs"),
        include_str!("emit/lower_picked_tests.rs"),
        include_str!("emit/lower_review_tests.rs"),
        include_str!("emit/lower_tests.rs"),
        include_str!("emit/meet.rs"),
        include_str!("emit/meet_tests.rs"),
        include_str!("emit/mod.rs"),
        include_str!("emit/off_is_off_tests.rs"),
        include_str!("emit/own_bits.rs"),
        include_str!("emit/paired.rs"),
        include_str!("emit/paired_tests.rs"),
        include_str!("emit/partial_waves.rs"),
        include_str!("emit/quad.rs"),
        include_str!("emit/quad_read_tests.rs"),
        include_str!("emit/rect.rs"),
        include_str!("emit/sampler.rs"),
        include_str!("emit/sampler_choice.rs"),
        include_str!("emit/sampler_choice_tests.rs"),
        include_str!("emit/sections.rs"),
        include_str!("emit/shader.rs"),
        include_str!("emit/signature_mode_tests.rs"),
        include_str!("emit/signature_order_tests.rs"),
        include_str!("emit/slot_map.rs"),
        include_str!("emit/slot_map_tests.rs"),
        include_str!("emit/threading.rs"),
        include_str!("emit/validator.rs"),
        include_str!("emit/valu_tests.rs"),
        include_str!("emit/vcc_branch_tests.rs"),
        include_str!("emit/vcc_branches.rs"),
        include_str!("emit/waterfall_tests.rs"),
        include_str!("emit/wave.rs"),
        include_str!("emit/wave64_tests.rs"),
        include_str!("gfx10/fields.rs"),
        include_str!("gfx10/mod.rs"),
        include_str!("gfx10/opcodes.rs"),
        include_str!("gfx10/sp3.rs"),
        include_str!("io.rs"),
        include_str!("ir.rs"),
        include_str!("ir/joins.rs"),
        include_str!("ir/lane_pairs.rs"),
        include_str!("lib.rs"),
        include_str!("native.rs"),
        include_str!("oracle/compare.rs"),
        include_str!("oracle/fixture.rs"),
        include_str!("oracle/machine.rs"),
        include_str!("oracle/mod.rs"),
        include_str!("oracle/semantics.rs"),
        include_str!("oracle/tests.rs"),
        include_str!("resource.rs"),
        include_str!("spill.rs"),
        include_str!("structure.rs"),
    ]
}

#[cfg(test)]
mod source_tests {
    /// Every source file is in the list.
    #[test]
    fn every_source_is_listed() {
        fn count(dir: &std::path::Path) -> usize {
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .map(|p| if p.is_dir() { count(&p) } else { usize::from(p.extension().is_some_and(|x| x == "rs")) })
                .sum()
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        assert_eq!(super::sources().len(), count(&dir));
    }
}
