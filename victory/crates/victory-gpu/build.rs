//! The build's identity, for the window's caption (`host.rs`, `build_name`): the commit's short
//! hash and Cargo's `PROFILE` (`debug` or `release`), nothing else. The window's label can appear
//! in a shared recording, so the branch is never read, and no probe for uncommitted changes runs (Sol's second review:
//! `git status` could still reach a configured helper); a name shows only where
//! `VICTORY_BUILD_NAME` asks for one at run time.
//!
//! The hash is one `git rev-parse` (which reads no index and makes no diff, so no file-system
//! monitor, filter, text conversion or transport runs), asked with no system or global
//! configuration, none carried in the environment, no optional locks and no prompt. Without
//! git, outside a checkout, or in a checkout that is not this workspace's own (a source tarball
//! unpacked inside another), it is `unknown`. Writes nothing but cargo directives and an empty
//! configuration file in `OUT_DIR`.
//!
//! Runs again when the checkout's HEAD or HEAD's log changes (a commit, a switch, a reset), and
//! when the sources of this crate or of the workspace crates it is built from change: those
//! rebuild this crate anyway, so the identity costs no build of its own.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Workspace crates this one is built from (its `Cargo.toml` and theirs): a change to their
/// sources rebuilds this crate whether or not the identity is asked again.
const UPSTREAM: &[&str] = &[
    "victory-core",
    "victory-formats",
    "victory-pipeline",
    "victory-shader",
];

/// Environment variables that point git at another repository or carry configuration in:
/// removed, so git finds the checkout from this crate's folder and reads no settings from
/// the caller.
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

/// Says to run again when `path` changes; a path that does not exist is left out (cargo would
/// run the script at every build for it).
fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// The commit's short hash, from one `git rev-parse` run from `dir` with `config` (an empty
/// file) as its global configuration; none outside a checkout that is `workspace`'s own.
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
        .args([
            "--no-optional-locks",
            "rev-parse",
            "--show-toplevel",
            "--absolute-git-dir",
            "--short=7",
            "HEAD",
        ])
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

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    watch(&manifest.join("src"));
    watch(&manifest.join("Cargo.toml"));
    for upstream in UPSTREAM {
        let dir = manifest.join("..").join(upstream);
        watch(&dir.join("src"));
        watch(&dir.join("Cargo.toml"));
    }

    let config =
        PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default()).join("empty.gitconfig");
    let commit = std::fs::write(&config, b"")
        .ok()
        .and_then(|()| commit(&manifest, &config, &manifest.join("..").join("..")))
        .unwrap_or_else(|| "unknown".to_owned());
    // Cargo's own word for the profile, never a folder's name.
    let profile = match std::env::var("PROFILE").as_deref() {
        Ok("debug") => "debug",
        Ok("release") => "release",
        _ => "build",
    };

    println!("cargo:rustc-env=VICTORY_GPU_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=VICTORY_GPU_BUILD_PROFILE={profile}");
}
