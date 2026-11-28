//! Fingerprints the shader translator, so that cached translations
//! (`src/shader_cache.rs`) last until the translator itself changes and not
//! only until the next build of anything.
//!
//! Beside it, the commit's short hash for the caption's build name with
//! `VICTORY_CAPTION_STATS=1` (`src/host/caption.rs`, `build_name`), as
//! `world`'s build script reads it (005d18b18): one `git rev-parse`, which
//! reads no index and makes no diff, asked with no system or global
//! configuration, none carried in the environment, no optional locks and no
//! prompt, and never the branch. Without git, outside a checkout, or in one
//! that is not this workspace's own, it is `unknown`. It runs again when the
//! checkout's HEAD or HEAD's log changes (a commit, a switch, a reset).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Environment variables that point git at another repository or carry
/// configuration in: removed, so git finds the checkout from this crate's
/// folder and reads no settings from the caller.
const GIT_ENV_REMOVED: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_EXTERNAL_DIFF",
    "GIT_DIFF_OPTS",
    "GIT_PAGER",
    "GIT_ASKPASS",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_EDITOR",
    "GIT_TRACE",
];

/// Says to run again when `path` changes; a path that does not exist is
/// left out (cargo would run the script at every build for it).
fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// The commit's short hash, from one `git rev-parse` run from `dir` with
/// `config` (an empty file) as its global configuration; none outside a
/// checkout that is `workspace`'s own.
fn commit(dir: &Path, config: &Path, workspace: &Path) -> Option<String> {
    let mut command = Command::new("git");
    for name in GIT_ENV_REMOVED {
        command.env_remove(name);
    }
    let out = command
        .current_dir(dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", config)
        .env("GIT_ATTR_NOSYSTEM", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(["--no-optional-locks", "rev-parse", "--show-toplevel", "--absolute-git-dir", "--short=7", "HEAD"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let [top, git_dir, hash] = <[&str; 3]>::try_from(text.lines().collect::<Vec<_>>()).ok()?;
    let same = |a: &Path, b: &Path| matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b);
    if !same(Path::new(top), workspace) {
        return None;
    }
    watch(&Path::new(git_dir).join("HEAD"));
    watch(&Path::new(git_dir).join("logs").join("HEAD"));
    (hash.len() == 7 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then(|| hash.to_owned())
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn main() {
    let here = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let crates = here.join("..");
    let root = crates.join("..");
    // The translator, what it is built from, and the form entries are kept in.
    let mut all = vec![
        crates.join("victory-shader/Cargo.toml"),
        here.join("src/shader_cache.rs"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
    ];
    let source = crates.join("victory-shader/src");
    println!("cargo:rerun-if-changed={}", source.display());
    files(&source, &mut all);
    all.sort();
    let mut hash = blake3::Hasher::new();
    for path in &all {
        println!("cargo:rerun-if-changed={}", path.display());
        // Names and line endings as they are in every checkout.
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let bytes: Vec<u8> = std::fs::read(path).unwrap_or_default().into_iter().filter(|&b| b != 13).collect();
        hash.update(name.as_bytes());
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=VICTORY_TRANSLATOR={}", hash.finalize().to_hex());

    let config = PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default()).join("empty.gitconfig");
    let commit = std::fs::write(&config, b"")
        .ok()
        .and_then(|()| commit(&here, &config, &root))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=VICTORY_GPU_BUILD_COMMIT={commit}");
}
