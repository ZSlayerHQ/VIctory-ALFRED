//! Fingerprint of the translator's own source, for the shader cache key
//! (`victory-gpu`'s `shader_cache`): a rebuilt executable whose translator did
//! not change keeps its cached translations. Also lists every `VICTORY_*`
//! switch the translator reads, whose values join the key at run time (not
//! the names in `NOT_SWITCHES`, which choose a device or a gate).
//! Reads only this crate's `src` folder; writes nothing but cargo directives.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `"VICTORY_..."` names in the sources that are no translation switches: they choose the
/// device a host is made on, or what a gate requires of it, and never change what the translator
/// emits (the host facts that do, its waves, join the key on their own). They stay out of the
/// list, whose values key the shader cache: set, even to the adapter the rule takes anyway, they
/// put every translation under a new key (the hybrid laptop, 2026-11-25: `hits=0 misses=641`
/// at the start screen with `VICTORY_ADAPTER=default` on the card it had without it).
const NOT_SWITCHES: &[&str] = &[
    // `adapter.rs`: the adapter rule's override, and a gate's requirement of a test host.
    "VICTORY_ADAPTER",
    "VICTORY_REQUIRE_GPU",
    // `adapter.rs` names the runtime's software switch (victory-gpu's) in its tests.
    "VICTORY_D3D12_WARP",
    // `oracle/tests.rs`: the folder of an offline coverage census (an ignored test's input).
    "VICTORY_ORACLE_COVERAGE",
];

/// FNV-1a, 64 bits, two lanes with different offsets: a mismatch only costs
/// a cache miss, so this need not be cryptographic.
fn fnv(state: &mut [u64; 2], bytes: &[u8]) {
    for &b in bytes {
        for (i, s) in state.iter_mut().enumerate() {
            *s ^= u64::from(b) ^ (i as u64 * 0x9e);
            *s = s.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let mut list = Vec::new();
    files(&src, &mut list);
    list.sort();
    let mut state = [0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64];
    let mut switches = std::collections::BTreeSet::new();
    for p in &list {
        let rel = p.strip_prefix(&src).unwrap_or(p).to_string_lossy().replace('\\', "/");
        let text = std::fs::read(p).unwrap_or_default();
        fnv(&mut state, rel.as_bytes());
        fnv(&mut state, &(text.len() as u64).to_le_bytes());
        fnv(&mut state, &text);
        let s = String::from_utf8_lossy(&text);
        let mut rest = &s[..];
        while let Some(at) = rest.find("\"VICTORY_") {
            let tail = &rest[at + 1..];
            let end = tail
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(tail.len());
            if tail[end..].starts_with('"') && !NOT_SWITCHES.contains(&&tail[..end]) {
                switches.insert(tail[..end].to_string());
            }
            rest = &tail[end..];
        }
    }
    println!("cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={:016x}{:016x}", state[0], state[1]);
    let joined: Vec<_> = switches.into_iter().collect();
    println!("cargo:rustc-env=VICTORY_SHADER_SWITCHES={}", joined.join(","));
}
