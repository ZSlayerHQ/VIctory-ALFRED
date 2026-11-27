//! Replays the requests `VICTORY_BCPACK_CHECK_KEEP` kept: the serial decode,
//! `unpack` with its chosen lanes, and `unpack` one lane at a time, compared
//! and printed exactly as the check logged them.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;

use victory_formats::bcpack::{self, Pool, Tables, diag};

struct Kept {
    ordinal: u64,
    bc: u8,
    source_size: usize,
    destination_size: usize,
    helpers: usize,
    hash: String,
}

/// Reads a `<hash>.txt` beside a kept `<hash>.bin`.
fn parse(text: &str, hash: &str) -> Result<Kept, String> {
    let mut fields = HashMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=') {
            fields.insert(key.trim(), value.trim());
        }
    }
    let number = |key: &str| -> Result<u64, String> {
        fields
            .get(key)
            .ok_or_else(|| format!("missing {key}"))?
            .parse()
            .map_err(|_| format!("bad {key}"))
    };
    Ok(Kept {
        ordinal: number("ordinal")?,
        bc: u8::try_from(number("bc")?).map_err(|_| "bad bc".to_owned())?,
        source_size: usize::try_from(number("source_size")?)
            .map_err(|_| "bad source_size".to_owned())?,
        destination_size: usize::try_from(number("destination_size")?)
            .map_err(|_| "bad destination_size".to_owned())?,
        helpers: usize::try_from(number("helpers").unwrap_or(0))
            .map_err(|_| "bad helpers".to_owned())?,
        hash: fields
            .get("blake3")
            .map_or_else(|| hash.to_owned(), |value| (*value).to_owned()),
    })
}

#[test]
#[ignore = "set VICTORY_BCPACK_REPLAY to the folder of kept requests"]
fn bcpack_replay() {
    let Some(dir) = std::env::var_os("VICTORY_BCPACK_REPLAY") else {
        eprintln!("bcpack replay: skipped (VICTORY_BCPACK_REPLAY is not set)");
        return;
    };
    let dir = Path::new(&dir);
    let mut bins = Vec::new();
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("bcpack replay: the folder cannot be read"));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|_| panic!("bcpack replay: an entry cannot be read"))
            .path();
        if path.extension() == Some(OsStr::new("bin")) {
            bins.push(path);
        }
    }
    bins.sort();
    let tables = Tables::builtin().expect("bcpack replay: the built-in tables");
    let mut kept = Vec::new();
    let mut helpers = 0usize;
    for bin in &bins {
        let hash = bin
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(bin.with_extension("txt"))
            .unwrap_or_else(|_| panic!("bcpack replay: kept request {hash} has no text file"));
        let params = parse(&text, &hash)
            .unwrap_or_else(|why| panic!("bcpack replay: kept request {hash}: {why}"));
        helpers = helpers.max(params.helpers);
        kept.push((bin.clone(), params));
    }
    // The check's `unpack` calls had this many helpers; the bytes do not
    // depend on it, the comparison just uses the same call.
    let pool = (helpers > 0).then(|| Pool::new(helpers, "bcpack replay", || {}));
    let mut failed: Vec<String> = Vec::new();
    for (bin, params) in &kept {
        let source = std::fs::read(bin).unwrap_or_else(|_| {
            panic!("bcpack replay: kept request {} cannot be read", params.hash)
        });
        assert_eq!(
            source.len(),
            params.source_size,
            "bcpack replay: kept request {} has another size",
            params.hash
        );
        let mut serial = vec![0u8; params.destination_size];
        let mut wide_out = vec![0u8; params.destination_size];
        let mut one_out = vec![0u8; params.destination_size];
        let serial_result = bcpack::decode(params.bc, &source, &mut serial, &tables);
        let wide = bcpack::unpack(params.bc, &source, &mut wide_out, &tables, pool.as_ref());
        let one = bcpack::unpack_one_lane(params.bc, &source, &mut one_out, &tables, pool.as_ref());
        let mut comparison = diag::Comparison::default();
        if serial_result.is_ok() {
            let block = diag::layout(params.bc).map_or(1, |(block, _, _)| block);
            comparison = diag::compare(&serial, &wide_out, &one_out, block);
        }
        if let Err(error) = &serial_result {
            comparison.serial_error = Some(error.0);
        }
        if let Err(error) = &wide {
            comparison.wide = None;
            comparison.wide_error = Some(error.0);
        }
        if let Err(error) = &one {
            comparison.one_lane = None;
            comparison.one_lane_error = Some(error.0);
        }
        let report = diag::Report {
            ordinal: params.ordinal,
            bc: params.bc,
            source_size: source.len(),
            destination_size: params.destination_size,
            hash: &params.hash,
            mode0: serial_result
                .is_ok()
                .then(|| diag::mode0(params.bc, &serial))
                .flatten(),
            comparison,
        };
        println!("{}", report.line());
        if report.comparison.differs() {
            failed.push(params.hash.clone());
        }
    }
    assert!(
        failed.is_empty(),
        "bcpack replay: {} of {} kept requests differ (first: {})",
        failed.len(),
        kept.len(),
        failed.first().map_or("none", String::as_str)
    );
    eprintln!("bcpack replay: {} kept requests, none differ", kept.len());
}
