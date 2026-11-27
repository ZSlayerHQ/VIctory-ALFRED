//! A name for every shader hash (derived from the title's database: never commit a table).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use victory_game::sgd5;

/// Files above this size are skipped (read whole into memory).
const MAX_FILE: u64 = 2 << 30;

#[cfg(test)]
thread_local! {
    /// Range comparisons the fallback scan made on this thread: the tests' measure that it walks the boundaries
    /// once rather than every range for every marker.
    static RANGE_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_range_check() {
    RANGE_CHECKS.with(|n| n.set(n.get() + 1));
}

#[cfg(not(test))]
fn note_range_check() {}

/// What the table holds: one row per shader, or one row per pass with every stage's shader.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Names,
    Effects,
}

/// The effect map's columns: the file, the effect and its technique and pass, then each stage's shader id and
/// entry point.
const EFFECT_HEADER: &str = "file\teffect\ttechnique\tpass\tvertex id\tvertex name\tpixel id\tpixel name\tgeometry id\tgeometry name\tslot3 id\tslot3 name\tslot4 id\tslot4 name\tcompute id\tcompute name";

/// One table row: the runtime id, stage, effect (or file), technique/pass, name.
pub(crate) struct Row {
    pub hash: String,
    pub stage: String,
    pub effect: String,
    pub technique: String,
    pub name: String,
}

/// Calls `row` for every shader object of `data` (a file the walker found), one at a time.
/// A database whose tables parse gives each of its shaders a stage and its effect. A marker record outside every
/// parsed database — a damaged database beside a valid one, or a file without one — falls back to its own name,
/// the file's name and no stage: one valid database must not hide the rest of the file.
fn each_row(data: &[u8], file: &Path, row: &mut dyn FnMut(Row)) {
    let databases = sgd5::databases(data);
    for database in &databases {
        for (slot, shaders) in database.shaders.iter().enumerate() {
            for (index, shader) in shaders.iter().enumerate() {
                let Some(container) = shader.record.unpack() else {
                    continue;
                };
                let (effect, technique) =
                    usage(database, slot, index).unwrap_or((file_name(file), String::new()));
                row(Row {
                    hash: victory_shader::container_id(&container).1,
                    stage: sgd5::STAGES[slot].to_owned(),
                    effect,
                    technique,
                    name: shader.record.name.unwrap_or("?").to_owned(),
                });
            }
        }
    }
    // One walk over the markers and the ranges together: the ranges are in file order, so a marker past one moves
    // to the next. Scanning every range for every marker cost databases x markers.
    let mut range = 0;
    for record in sgd5::records(data) {
        while range < databases.len() && record.offset >= databases[range].end {
            note_range_check();
            range += 1;
        }
        let within_a_database = range < databases.len() && {
            note_range_check();
            record.offset >= databases[range].start && record.offset < databases[range].end
        };
        if within_a_database {
            continue;
        }
        let Some(container) = record.unpack() else {
            continue;
        };
        row(Row {
            hash: victory_shader::container_id(&container).1,
            stage: String::new(),
            effect: file_name(file),
            technique: String::new(),
            name: record.name.unwrap_or("?").to_owned(),
        });
    }
}

/// Writes one row per shader to `writer` as the rows are built: neither a file's rows nor one text buffer of them
/// all is kept.
fn write_rows<W: Write + ?Sized>(writer: &mut W, data: &[u8], file: &Path) -> std::io::Result<()> {
    let mut error = None;
    each_row(data, file, &mut |row| {
        if error.is_some() {
            return;
        }
        if let Err(e) = writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            row.hash, row.stage, row.effect, row.technique, row.name
        ) {
            error = Some(e);
        }
    });
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
pub(crate) fn rows_in(data: &[u8], file: &Path) -> Vec<Row> {
    let mut rows = Vec::new();
    each_row(data, file, &mut |row| rows.push(row));
    rows
}

/// Writes one row per pass of every parsed database: the file, the effect, its technique and pass, and each
/// stage's shader id and entry point. The same bounded walk and streaming as the names table.
fn write_effect_rows<W: Write + ?Sized>(writer: &mut W, data: &[u8], file: &Path) -> std::io::Result<()> {
    for database in &sgd5::databases(data) {
        let mut shaders: [Vec<(String, &str)>; 6] = Default::default();
        for (slot, list) in database.shaders.iter().enumerate() {
            shaders[slot] = list
                .iter()
                .map(|shader| {
                    let id = shader
                        .record
                        .unpack()
                        .map(|container| victory_shader::container_id(&container).1)
                        .unwrap_or_default();
                    (id, shader.record.name.unwrap_or("?"))
                })
                .collect();
        }
        for effect in &database.effects {
            for technique in &effect.techniques {
                let first = technique.first as usize;
                let Some(passes) = effect
                    .passes
                    .get(first..first + technique.count as usize)
                else {
                    continue;
                };
                for (at, pass) in passes.iter().flatten().enumerate() {
                    write!(
                        writer,
                        "{}\t{}\t{}\t{}",
                        file_name(file),
                        effect.name,
                        technique.name,
                        at + 1
                    )?;
                    for slot in 0..6 {
                        match pass.shaders[slot].and_then(|index| shaders[slot].get(index)) {
                            Some((id, name)) => write!(writer, "\t{id}\t{name}")?,
                            None => write!(writer, "\t\t")?,
                        }
                    }
                    writeln!(writer)?;
                }
            }
        }
    }
    Ok(())
}

/// The first effect/technique/pass that uses shader `index` of `slot`.
fn usage(database: &sgd5::Database<'_>, slot: usize, index: usize) -> Option<(String, String)> {
    for effect in &database.effects {
        for technique in &effect.techniques {
            let first = technique.first as usize;
            let Some(passes) = effect
                .passes
                .get(first..first + technique.count as usize)
            else {
                continue;
            };
            for (at, pass) in passes.iter().flatten().enumerate() {
                if pass.shaders[slot] == Some(index) {
                    return Some((
                        effect.name.to_owned(),
                        format!("{} pass {}", technique.name, at + 1),
                    ));
                }
            }
        }
    }
    None
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Reads `path` if it is a file of at most `limit` bytes. The size from the metadata is only a first look: a file
/// can change between it and the read, so the read itself stops at the limit.
fn read_bounded(path: &Path, limit: u64) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    if file.metadata().ok()?.len() > limit {
        return None;
    }
    let mut data = Vec::new();
    file.take(limit + 1).read_to_end(&mut data).ok()?;
    (data.len() as u64 <= limit).then_some(data)
}

/// Writes the names table for `root` (a directory or one file) to standard output; a caller redirects it.
pub fn run(root: PathBuf) -> Result<()> {
    run_with(root, MAX_FILE, Mode::Names)
}

/// Writes the effect map for `root`: one row per pass, with every stage's shader.
pub fn run_effects(root: PathBuf) -> Result<()> {
    run_with(root, MAX_FILE, Mode::Effects)
}

fn run_with(root: PathBuf, limit: u64, mode: Mode) -> Result<()> {
    let stdout = std::io::stdout();
    let mut writer = std::io::BufWriter::new(stdout.lock());
    write_mode(&mut writer, &root, limit, mode).context("writing the table to standard output")?;
    writer.flush().context("writing the table to standard output")
}

/// The table of [`run_with`], header and all, into `writer`: the tests write into memory and exercise the limit.
/// A root that cannot be read ends with an error before anything is written, in either mode.
fn write_mode(writer: &mut dyn Write, root: &Path, limit: u64, mode: Mode) -> Result<()> {
    let input = input(root, limit)?;
    if mode == Mode::Effects {
        writeln!(writer, "{EFFECT_HEADER}")?;
    }
    write_input(writer, root, input, limit, mode)
}

/// What a table is made from: the one file `root` names, or the walk of a directory.
enum Input {
    File(Vec<u8>),
    Walk,
}

/// Checks `root` and reads it if it is a file, before anything of a table is written.
fn input(root: &Path, limit: u64) -> Result<Input> {
    // A missing root, or a directory that cannot be opened, would otherwise walk as empty: the walk skips its
    // root error, and the table would come out empty as a success.
    let walked = std::fs::metadata(root)
        .with_context(|| format!("reading the walk root {}", root.display()))?;
    if walked.is_dir() {
        std::fs::read_dir(root)
            .with_context(|| format!("reading the walk root {}", root.display()))?;
    }
    if !root.is_file() {
        return Ok(Input::Walk);
    }
    let data = read_bounded(root, limit).with_context(|| {
        format!(
            "reading {}: not a file of at most {limit} bytes",
            root.display()
        )
    })?;
    Ok(Input::File(data))
}

/// Writes the table for `root` to `writer`; the size limit is the tests' small files.
#[cfg(test)]
fn write_table(writer: &mut dyn Write, root: &Path, limit: u64, mode: Mode) -> Result<()> {
    let input = input(root, limit)?;
    write_input(writer, root, input, limit, mode)
}

/// Writes the rows of `input` (read from `root`) to `writer`.
fn write_input(writer: &mut dyn Write, root: &Path, input: Input, limit: u64, mode: Mode) -> Result<()> {
    if let Input::File(data) = input {
        if sgd5::is_sgd5(&data) {
            match mode {
                Mode::Names => write_rows(writer, &data, root)?,
                Mode::Effects => write_effect_rows(writer, &data, root)?,
            }
        }
    } else {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let Some(data) = read_bounded(path, limit) else {
                continue;
            };
            if sgd5::is_sgd5(&data) {
                match mode {
                    Mode::Names => write_rows(writer, &data, path)?,
                    Mode::Effects => write_effect_rows(writer, &data, path)?,
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    const MAGIC: &[u8; 4] = b"SGD5";
    const MARKER: u32 = 0x000D_DCE7;
    const VS: [u8; 40] = [0x11; 40];
    const PS: [u8; 40] = [0x22; 40];

    fn u32s(out: &mut Vec<u8>, values: &[u32]) {
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn str(out: &mut Vec<u8>, s: &str) {
        out.push(s.len() as u8 + 1);
        out.extend_from_slice(s.as_bytes());
        out.push(0);
    }

    /// A raw-DEFLATE stream with one stored block: no compressor needed.
    fn deflated(payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() <= 0xFFFF);
        let mut out = vec![0x01];
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn shader(out: &mut Vec<u8>, name: &str, kind: u8, payload: &[u8], engine: u64) {
        str(out, name);
        out.push(kind);
        let packed = deflated(payload);
        let n = packed.len() as u32 + 12;
        u32s(out, &[n, MARKER, n, payload.len() as u32]);
        out.extend_from_slice(&packed);
        out.extend_from_slice(&engine.to_le_bytes());
        out.extend_from_slice(&[0; 0x90]);
        u32s(out, &[0]);
    }

    fn effect(out: &mut Vec<u8>, name: &str, stages: [&[u32]; 6], technique: &str, pass: [u8; 24]) {
        effect_with(out, name, stages, technique, &[pass], 0, 1);
    }

    /// [`effect`] with its pass array, and its one technique's first pass and count, given.
    fn effect_with(
        out: &mut Vec<u8>,
        name: &str,
        stages: [&[u32]; 6],
        technique: &str,
        passes: &[[u8; 24]],
        first: u32,
        count: u32,
    ) {
        str(out, name);
        let size_at = out.len();
        u32s(out, &[0]);
        for list in stages {
            u32s(out, &[list.len() as u32]);
            out.extend(list.iter().flat_map(|v| v.to_le_bytes()));
        }
        out.extend_from_slice(&[0; 0x50]);
        for _ in 0..4 {
            u32s(out, &[0]);
        }
        let names = [technique.as_bytes(), &[0]].concat();
        u32s(out, &[1, first << 15 | count << 24]); // one technique, name offset 0
        u32s(out, &[passes.len() as u32]);
        for pass in passes {
            out.extend_from_slice(pass);
        }
        u32s(out, &[0]); // no extra indices
        u32s(out, &[0]); // no local strings
        u32s(out, &[0, 0]); // the two byte lists
        u32s(out, &[0]); // no blocks
        u32s(out, &[names.len() as u32]);
        out.extend_from_slice(&names);
        let sum: usize = stages.iter().map(|list| list.len() + 1).sum();
        let size = sum * 8 + 8 + passes.len() * 32 + names.len();
        out[size_at..size_at + 4].copy_from_slice(&(size as u32).to_le_bytes());
    }

    /// Two effects and three shaders; the two pixel shaders hold the same container under two names.
    fn database() -> Vec<u8> {
        let mut f = MAGIC.to_vec();
        f.extend_from_slice(&[0; 8]);
        u32s(&mut f, &[0]); // no shared blocks
        u32s(&mut f, &[0]); // no string table
        let lists: [&[(&str, u8, &[u8])]; 6] = [
            &[("VS_a", 0x32, &VS)],
            &[("PS_a", 0x32, &PS), ("PS_b", 0x32, &PS)],
            &[],
            &[],
            &[],
            &[],
        ];
        let mut engine = 0x1000u64;
        for list in lists {
            u32s(&mut f, &[list.len() as u32]);
            for (name, kind, payload) in list {
                shader(&mut f, name, *kind, payload, engine);
                engine += 1;
            }
        }
        u32s(&mut f, &[2]);
        // fx1 draws with the vertex shader and the first pixel shader.
        let mut draw = [0u8; 24];
        draw[0] = 1;
        draw[1] = 1;
        effect(&mut f, "fx1", [&[0], &[0], &[], &[], &[], &[]], "main", draw);
        // fx2 draws with the second pixel shader (the duplicated container).
        let mut second = [0u8; 24];
        second[1] = 1;
        effect(&mut f, "fx2", [&[], &[1], &[], &[], &[], &[]], "second", second);
        f
    }

    #[test]
    fn names_every_shader_with_the_runtime_hash_and_its_effect() {
        let db = database();
        let rows = super::rows_in(&db, std::path::Path::new("synthetic.sgd5"));
        assert_eq!(rows.len(), 3);
        let by_name: BTreeMap<&str, _> = rows.iter().map(|row| (row.name.as_str(), row)).collect();
        let vs = victory_shader::container_id(&VS).1;
        let ps = victory_shader::container_id(&PS).1;
        assert_eq!(by_name["VS_a"].hash, vs);
        assert_eq!(by_name["VS_a"].stage, "vertex");
        assert_eq!(by_name["VS_a"].effect, "fx1");
        assert!(by_name["VS_a"].technique.starts_with("main"));
        assert_eq!(by_name["PS_a"].hash, ps);
        assert_eq!(by_name["PS_a"].stage, "pixel");
        assert_eq!(by_name["PS_a"].effect, "fx1");
        assert_eq!(by_name["PS_b"].hash, ps, "the same container under two names");
        assert_eq!(by_name["PS_b"].name, "PS_b");
        assert_eq!(by_name["PS_b"].effect, "fx2");
    }

    /// A fresh directory under the OS temporary folder: unique within this process and never one that already
    /// exists. It fails rather than deleting anything it did not make, and the caller removes it.
    fn temp_dir(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "victory-names-{}-{name}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).expect("a fresh temporary directory that did not exist");
        dir
    }

    #[test]
    fn a_file_over_the_limit_is_skipped() {
        let dir = temp_dir("limit");
        let walked = dir.join("in");
        std::fs::create_dir_all(&walked).unwrap();
        let small = database();
        std::fs::write(walked.join("small.sgd5"), &small).unwrap();
        let mut big = database();
        big.extend_from_slice(&[0; 64]);
        std::fs::write(walked.join("big.sgd5"), &big).unwrap();

        // The limit passes the small database and stops the larger file: the walk writes the small file's three
        // rows only, as it builds them.
        let mut table = Vec::new();
        super::write_table(&mut table, &walked, small.len() as u64, super::Mode::Names).unwrap();
        assert_eq!(String::from_utf8(table).unwrap().lines().count(), 3);

        // The direct path is held to the same limit.
        let mut table = Vec::new();
        assert!(super::write_table(&mut table, &walked.join("big.sgd5"), small.len() as u64, super::Mode::Names).is_err());
        assert!(table.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_valid_database_does_not_hide_a_damaged_one_s_shaders() {
        let mut file = database();
        file.extend(database());
        // The second database's effect size no longer matches its tables, so only the first one parses.
        let second = memchr::memmem::find_iter(&file, b"\x04fx1\0")
            .nth(1)
            .expect("second effect");
        file[second + 4] ^= 1;
        assert_eq!(super::sgd5::databases(&file).len(), 1);
        let rows = super::rows_in(&file, std::path::Path::new("archive/pack.sgd5"));
        // Three rows from the database that parsed, and three from the damaged one's marker records.
        assert_eq!(rows.len(), 6);
        assert_eq!(rows.iter().filter(|row| row.stage.is_empty()).count(), 3);
        assert_eq!(rows.iter().filter(|row| row.effect == "pack.sgd5").count(), 3);
    }

    /// A writer that fails, like a closed pipe.
    struct Closed;

    impl std::io::Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed"))
        }
    }

    #[test]
    fn a_write_error_ends_with_an_error() {
        let dir = temp_dir("closed-pipe");
        let file = dir.join("pack.sgd5");
        std::fs::write(&file, database()).unwrap();
        let error = super::write_table(&mut Closed, &file, super::MAX_FILE, super::Mode::Names).unwrap_err();
        assert!(error.to_string().contains("closed"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_effect_map_lists_each_pass_with_its_shaders() {
        let dir = temp_dir("effect-map");
        let input = dir.join("in");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::write(input.join("materials.sgd5"), database()).unwrap();
        let mut table = Vec::new();
        super::write_mode(&mut table, &input, super::MAX_FILE, super::Mode::Effects).unwrap();
        let text = String::from_utf8(table).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], super::EFFECT_HEADER);
        assert_eq!(lines.len(), 3, "{text}");
        let vs = victory_shader::container_id(&VS).1;
        let ps = victory_shader::container_id(&PS).1;
        let row = |line: &str| line.split('\t').map(str::to_owned).collect::<Vec<_>>();
        // fx1's pass draws the vertex shader and the first pixel name.
        let first = row(lines[1]);
        assert_eq!(&first[1..4], ["fx1", "main", "1"]);
        assert_eq!((first[4].as_str(), first[5].as_str()), (vs.as_str(), "VS_a"));
        assert_eq!((first[6].as_str(), first[7].as_str()), (ps.as_str(), "PS_a"));
        // fx2's pass shows the same container under its other name.
        let second = row(lines[2]);
        assert_eq!(&second[1..4], ["fx2", "second", "1"]);
        assert_eq!((second[6].as_str(), second[7].as_str()), (ps.as_str(), "PS_b"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Two effects; the first holds an invalid pass no technique names, then the pass its technique uses.
    fn database_with_hole() -> Vec<u8> {
        let mut bad = [0u8; 24];
        bad[6] = 9;
        bad[10] = 1;
        let mut draw = [0u8; 24];
        draw[0] = 1;
        draw[1] = 1;
        let mut second = [0u8; 24];
        second[1] = 1;
        let mut f = MAGIC.to_vec();
        f.extend_from_slice(&[0; 8]);
        u32s(&mut f, &[0]); // no shared blocks
        u32s(&mut f, &[0]); // no string table
        let lists: [&[(&str, u8, &[u8])]; 6] = [
            &[("VS_a", 0x32, &VS)],
            &[("PS_a", 0x32, &PS), ("PS_b", 0x32, &PS)],
            &[],
            &[],
            &[],
            &[],
        ];
        let mut engine = 0x1000u64;
        for list in lists {
            u32s(&mut f, &[list.len() as u32]);
            for (name, kind, payload) in list {
                shader(&mut f, name, *kind, payload, engine);
                engine += 1;
            }
        }
        u32s(&mut f, &[2]);
        effect_with(
            &mut f,
            "fx1",
            [&[0], &[0], &[], &[], &[], &[]],
            "main",
            &[bad, draw],
            1,
            1,
        );
        effect(&mut f, "fx2", [&[], &[1], &[], &[], &[], &[]], "second", second);
        f
    }

    #[test]
    fn an_unreferenced_pass_has_no_row_in_the_effect_map() {
        let file = database_with_hole();
        let dir = temp_dir("effect-map-hole");
        let input = dir.join("in");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::write(input.join("materials.sgd5"), &file).unwrap();
        let mut table = Vec::new();
        super::write_mode(&mut table, &input, super::MAX_FILE, super::Mode::Effects).unwrap();
        let text = String::from_utf8(table).unwrap();
        assert_eq!(text.lines().count(), 3, "{text}");
        assert!(text.lines().nth(1).unwrap().contains("\tfx1\tmain\t1\t"));
        assert!(text.lines().nth(2).unwrap().contains("\tfx2\tsecond\t1\t"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_walk_root_ends_with_an_error_and_writes_nothing() {
        let dir = temp_dir("missing-root");
        let mut table = Vec::new();
        let error = super::write_table(&mut table, &dir.join("not-there"), super::MAX_FILE, super::Mode::Names)
            .unwrap_err();
        assert!(error.to_string().contains("not-there"), "{error}");
        assert!(table.is_empty(), "nothing was written");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Both tables, as the command writes them (header and all): a root that
    /// is not there, or a direct file over the limit, ends with an error
    /// before anything is written.
    #[test]
    fn a_bad_root_writes_nothing_in_either_mode() {
        let dir = temp_dir("bad-root");
        let big = dir.join("big.sgd5");
        std::fs::write(&big, database()).unwrap();
        for mode in [super::Mode::Names, super::Mode::Effects] {
            for (root, limit) in [(dir.join("not-there"), super::MAX_FILE), (big.clone(), 16)] {
                let mut table = Vec::new();
                let error = super::write_mode(&mut table, &root, limit, mode).unwrap_err();
                let name = root.file_name().unwrap().to_string_lossy().into_owned();
                assert!(error.to_string().contains(&name), "{error}");
                assert_eq!(String::from_utf8(table).unwrap(), "", "nothing was written");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_direct_input_file_with_any_name_is_read() {
        let dir = temp_dir("direct-name");
        let file = dir.join("materials");
        std::fs::write(&file, database()).unwrap();
        let mut table = Vec::new();
        super::write_table(&mut table, &file, super::MAX_FILE, super::Mode::Names).unwrap();
        assert_eq!(String::from_utf8(table).unwrap().lines().count(), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_fallback_scan_walks_the_database_ranges_once() {
        // Many small databases: scanning every range for every marker costs databases x markers checks.
        let mut file = Vec::new();
        for _ in 0..200 {
            file.extend(database());
        }
        super::RANGE_CHECKS.with(|n| n.set(0));
        let rows = super::rows_in(&file, std::path::Path::new("archive/pack.sgd5"));
        assert_eq!(rows.len(), 600);
        let checks = super::RANGE_CHECKS.with(|n| n.get());
        assert!(checks <= 4 * (200 + 600), "range checks: {checks}");
    }

    #[test]
    fn a_file_whose_tables_do_not_parse_falls_back_to_its_records_and_file_name() {
        let mut db = database();
        let at = memchr::memmem::find(&db, b"\x04fx1\0").expect("effect name") + 4;
        db[at] ^= 1;
        let rows = super::rows_in(&db, std::path::Path::new("archive/pack.sgd5"));
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.stage.is_empty()));
        assert!(rows.iter().all(|row| row.effect == "pack.sgd5"));
        assert!(rows.iter().any(|row| row.name == "PS_b"));
    }
}
