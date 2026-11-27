//! `cfg(victory_avx512)` when the machine that builds has what the BCPack
//! lanes need to run a round at once (AVX-512 F, BW, VL, VBMI2; BMI2,
//! POPCNT): the tests that compare that path are then run, and elsewhere
//! reported ignored instead of passing without having compared anything.
//! The decoder itself asks the processor it runs on.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(victory_avx512)");
    println!("cargo::rerun-if-changed=build.rs");
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi2")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("popcnt")
    {
        println!("cargo::rustc-cfg=victory_avx512");
    }
}
