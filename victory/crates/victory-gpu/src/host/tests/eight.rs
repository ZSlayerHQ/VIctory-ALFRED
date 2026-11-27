//! The eight-load forms of the card hash (`host::hash::pick` with `eight`
//! on) against their one-load forms on the software adapter: over the same
//! patterned and random bytes, the same span constants and the same pass
//! sequence (hash, then compare 1, compare 2 and compare 1 again after bytes
//! changed in the main loop's blocks and in the remainder loop's), the
//! eight-load program leaves the same bytes in `hashes` and `differs` as the
//! one-load program, for the narrow pair and for the wide pair. The shapes:
//! full pieces, skewed first pieces, last pieces clamped by `size` so that 0
//! to 7 steps are left after the eight-load loop, and ranges shorter than
//! eight steps. A child of `wide.rs`, whose device helpers it uses.

use super::{Addrs, Host, Program, Rng, Span, hash, model, model_hash, record, test_host};

const PIECE: u32 = 65_536;

/// One case's span constants.
struct Case {
    name: String,
    first: u32,
    skew: u32,
    size: u32,
    pieces: u32,
}

impl Case {
    fn new(name: impl Into<String>, first: u32, skew: u32, size: u32, pieces: u32) -> Self {
        Self {
            name: name.into(),
            first,
            skew,
            size,
            pieces,
        }
    }
}

/// What a program left in both buffers after each pass of the sequence.
type Snapshots = Vec<(Vec<u8>, Vec<u8>)>;

/// Where `differs` is written from: one dword in, so a store at the wrong
/// offset changes a byte the comparison sees.
const OUT_AT: u32 = 4;

/// Bytes in 4 KiB runs of four kinds: random, a repeated 16-byte block,
/// zeros, and a counter (so identical blocks, zero blocks and distinct
/// blocks all fall in both loops).
fn fill(len: usize, rng: &mut Rng) -> Vec<u8> {
    (0..len)
        .map(|i| match (i / 4096) % 4 {
            0 => rng.byte(),
            1 => ((i % 16) as u8).wrapping_mul(17) ^ 0x3c,
            2 => 0,
            _ => (i as u32).wrapping_mul(0x9E37_79B1).to_le_bytes()[3],
        })
        .collect()
}

/// The steps of one thread of a piece of `blocks` 16-byte blocks: the narrow
/// form's one thread, or the wide form's `lane`.
fn steps(wide: bool, blocks: usize, lane: usize) -> usize {
    if !wide {
        blocks
    } else if blocks > lane {
        (blocks - lane).div_ceil(64)
    } else {
        0
    }
}

/// Whether block `b` of a piece of `blocks` blocks is read by the
/// eight-load loop (else by the remainder loop after it).
fn in_main(wide: bool, b: usize, blocks: usize) -> bool {
    let (lane, turn) = if wide { (b % 64, b / 64) } else { (0, b) };
    turn < steps(wide, blocks, lane) / 8 * 8
}

/// Runs the pass sequence with one program and returns the buffers after
/// each pass: hash (compare 0), then `flips` applied to the bytes, compare
/// 1, compare 2 and compare 1 again. `hashes` starts as `seed_hashes` and
/// `differs` as 0xA5 bytes, so a store where none belongs shows.
fn run(
    host: &Host,
    program: &Program,
    wide: bool,
    case: &Case,
    bytes: &[u8],
    seed_hashes: &[u8],
    flips: &[usize],
) -> Snapshots {
    let differs_len = (case.pieces as usize + 2) * 4;
    let (now, now_at) = host
        .upload_buffer(bytes.len() as u64)
        .expect("an upload buffer");
    let (hashes, hashes_at) = host
        .shared_buffer(seed_hashes.len() as u64)
        .expect("a hash buffer");
    let (differs, differs_at) = host
        .shared_buffer(differs_len as u64)
        .expect("a differs buffer");
    // SAFETY: inside the buffers' mappings.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), now_at, bytes.len());
        std::ptr::copy_nonoverlapping(seed_hashes.as_ptr(), hashes_at, seed_hashes.len());
        std::ptr::write_bytes(differs_at, 0xA5, differs_len);
    }
    // SAFETY: live resources; the addresses are read by the lists below.
    let addrs = unsafe {
        Addrs {
            now: now.GetGPUVirtualAddress(),
            hashes: hashes.GetGPUVirtualAddress(),
            differs: differs.GetGPUVirtualAddress(),
        }
    };
    let mut out = Vec::new();
    for (pass, compare) in [0u32, 1, 2, 1].into_iter().enumerate() {
        if pass == 1 {
            for &at in flips {
                // SAFETY: inside the upload buffer's mapping, after the hash
                // pass's submission was waited for.
                unsafe { *now_at.add(at) ^= 0x80 };
            }
        }
        let list = host.new_list().expect("a host list");
        record(
            &list,
            program,
            wide,
            addrs,
            Span {
                first: case.first,
                skew: case.skew,
                size: case.size,
                pieces: case.pieces,
                out_at: OUT_AT,
                compare,
            },
        );
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mappings, after the queue's fence was reached.
        unsafe {
            out.push((
                std::slice::from_raw_parts(hashes_at, seed_hashes.len()).to_vec(),
                std::slice::from_raw_parts(differs_at, differs_len).to_vec(),
            ));
        }
    }
    out
}

/// The dword piece `p` of a case has in a `differs` snapshot.
fn differs_of(differs: &[u8], p: u32) -> u32 {
    let at = (OUT_AT + p * 4) as usize;
    u32::from_le_bytes(differs[at..at + 4].try_into().unwrap())
}

/// The hash a `hashes` snapshot holds for piece `piece`.
fn hash_of(hashes: &[u8], piece: u32) -> [u32; 4] {
    let at = piece as usize * 16;
    std::array::from_fn(|i| u32::from_le_bytes(hashes[at + i * 4..at + i * 4 + 4].try_into().unwrap()))
}

/// The cases (a) to (d) of the brief, for both pairs; (e) is the pass
/// sequence every case runs.
fn cases() -> Vec<Case> {
    let mut out = vec![
        // (a) full pieces, from piece 0 and from piece 1.
        Case::new("four full pieces", 0, 0, 4 * PIECE, 4),
        Case::new("three full pieces from piece 1", 1, 0, 4 * PIECE, 3),
        Case::new("one full piece", 0, 0, PIECE, 1),
    ];
    // (b) a skewed first piece: 16-byte aligned and not, small and large.
    for skew in [0x10u32, 0x1234, 0x1235, 0x8008, 0xFFF0] {
        out.push(Case::new(format!("skew {skew:#x}"), 0, skew, 3 * PIECE, 3));
    }
    // (c) a last piece clamped by `size` (piece 1 of two): `n` blocks, for
    // every remainder 0 to 7, as the narrow loop leaves it (n % 8) and as
    // the wide loop leaves it (lane 0's steps % 8), with ragged sizes too.
    for r in 0..8u32 {
        for n in [8 + r, 4088 + r, 512 + 64 * r, 3584 + 64 * r + 5] {
            out.push(Case::new(format!("{n} blocks in the last piece"), 0, 0, PIECE + 16 * n, 2));
        }
        for n in [4088 + r, 512 + 64 * r] {
            out.push(Case::new(
                format!("{n} blocks in the last piece, 3 bytes short"),
                0,
                0,
                PIECE + 16 * n - 3,
                2,
            ));
        }
    }
    // (c) with a skew: piece 2 clamped, the copy starting 0x1230 bytes in.
    out.push(Case::new("skew 0x1230, 4093 blocks clamped", 0, 0x1230, 2 * PIECE + 4093 * 16 - 0x1230, 3));
    // (d) shorter than eight steps: in the narrow form (fewer than eight
    // blocks), and in the wide form (fewer than eight a lane).
    for m in 1..8u32 {
        out.push(Case::new(format!("{m} blocks"), 0, 0, 16 * m, 1));
        out.push(Case::new(format!("{m} blocks at piece 2"), 2, 0, 2 * PIECE + 16 * m, 1));
        out.push(Case::new(format!("{} blocks", 64 * m + 1), 0, 0, 16 * (64 * m + 1), 1));
    }
    out.push(Case::new("3 bytes", 0, 0, 3, 1));
    out
}

/// The host device for a pair's test. With no device: a failure when the
/// tests were told to use the software adapter (`VICTORY_D3D12_WARP=1`,
/// `crate::host::on_warp`), whose test must not pass by skipping; else a
/// skip that says so. (`test_host` itself already fails a test whose device
/// was refused or not made under the switch; this holds the test to it
/// whatever path left it without one.)
fn host_or_skip(form: &str) -> Option<&'static Host> {
    let host = test_host();
    if host.is_none() {
        assert!(
            !crate::host::on_warp(),
            "{form} eight-load pair: VICTORY_D3D12_WARP=1 and no host device; the test did not run"
        );
        eprintln!("{form} eight-load pair: skipped, no host device (VICTORY_D3D12_WARP is not set)");
    }
    host
}

/// Runs every case with the one-load and the eight-load program of a pair
/// and holds them to the same bytes after every pass; also holds the
/// eight-load hashes to the model and its flags to the pieces that changed.
fn the_pair_agrees(wide: bool) {
    let form = if wide { "wide" } else { "narrow" };
    let Some(host) = host_or_skip(form) else {
        return;
    };
    let one = Program::new(host, hash::pick(wide, false));
    let eight = Program::new(host, hash::pick(wide, true));
    let mut rng = Rng::new(if wide { 0x2545_f491_4f6c_dd1d } else { 0x9e37_79b9_85eb_ca6b });
    let mut remainders = [false; 8];
    let (mut main_flips, mut rest_flips) = (0usize, 0usize);
    for case in cases() {
        let name = format!("{form}, {}", case.name);
        let len = ((case.size as usize + 15) & !15) + 4096;
        let bytes = fill(len, &mut rng);
        let slots = (case.first + case.pieces) as usize + 1;
        let seed_hashes: Vec<u8> = (0..slots * 16).map(|_| rng.byte()).collect();

        // One flip in the last piece (in the remainder loop's blocks where
        // it has any) and one in the first (in the main loop's, where it has
        // any): each a block of a piece and the piece's own.
        let mut flips = Vec::new();
        let last = case.first + case.pieces - 1;
        for (piece, want_main) in [(last, false), (case.first, true)] {
            let (lo, hi) = model::piece_range(piece, case.skew, case.size);
            let blocks = (hi - lo) / 16;
            if piece == last {
                remainders[steps(wide, blocks, 0) % 8] = true;
            }
            let pick_from = |main: bool| -> Vec<usize> {
                (0..blocks).filter(|&b| in_main(wide, b, blocks) == main).collect()
            };
            let mut chosen = pick_from(want_main);
            let mut main = want_main;
            if chosen.is_empty() {
                chosen = pick_from(!want_main);
                main = !want_main;
            }
            if chosen.is_empty() {
                continue;
            }
            let b = chosen[rng.below(chosen.len())];
            flips.push(lo + b * 16 + rng.below(16));
            if main {
                main_flips += 1;
            } else {
                rest_flips += 1;
            }
            if case.pieces == 1 {
                break;
            }
        }
        let mut after = bytes.clone();
        for &at in &flips {
            after[at] ^= 0x80;
        }

        let want = run(host, &one, wide, &case, &bytes, &seed_hashes, &flips);
        let got = run(host, &eight, wide, &case, &bytes, &seed_hashes, &flips);
        for (pass, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(w.0 == g.0, "{name}: pass {pass}, `hashes` differ");
            assert!(w.1 == g.1, "{name}: pass {pass}, `differs` differ");
        }

        // Not two programs wrong alike: the hash pass is the model's, the
        // first compare flags exactly the pieces whose hash the flips
        // changed, and after compare 2 stored them the next finds none.
        for p in 0..case.pieces {
            let piece = case.first + p;
            assert_eq!(
                hash_of(&got[0].0, piece),
                model_hash(&bytes, wide, piece, case.skew, case.size),
                "{name}: piece {piece}, the model"
            );
            let changed = model_hash(&bytes, wide, piece, case.skew, case.size)
                != model_hash(&after, wide, piece, case.skew, case.size);
            assert_eq!(differs_of(&got[1].1, p), u32::from(changed), "{name}: piece {piece}, compare 1");
            assert_eq!(differs_of(&got[2].1, p), u32::from(changed), "{name}: piece {piece}, compare 2");
            assert_eq!(differs_of(&got[3].1, p), 0, "{name}: piece {piece}, compare 1 after 2");
        }
        assert!(
            flips.is_empty() || (0..case.pieces).any(|p| differs_of(&got[1].1, p) == 1),
            "{name}: a flip changed no piece"
        );
    }
    assert_eq!(remainders, [true; 8], "{form}: every remainder 0 to 7 in a last piece");
    assert!(main_flips > 0 && rest_flips > 0, "{form}: flips in both loops");
}

/// (a) to (e) for the narrow pair: `CARD_COMPARE` and its eight-load form.
#[test]
fn the_narrow_eight_load_form_writes_what_the_one_load_form_writes() {
    the_pair_agrees(false);
}

/// (a) to (e) for the wide pair: `WIDE` and its eight-load form.
#[test]
fn the_wide_eight_load_form_writes_what_the_one_load_form_writes() {
    the_pair_agrees(true);
}

/// The compiler takes all four texts with warnings as errors (`compile`
/// drops a warning when the code builds, so this is where one would show),
/// as the card work compiles them: `main`, `cs_5_1`.
#[test]
fn the_four_programs_compile_without_a_warning() {
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_WARNINGS_ARE_ERRORS, D3DCompile};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;
    for (wide, eight) in [(false, false), (true, false), (false, true), (true, true)] {
        let text = hash::pick(wide, eight);
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                text.as_ptr().cast(),
                text.len(),
                s!("victory"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                D3DCOMPILE_WARNINGS_ARE_ERRORS,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        let said = errors.map(|e| {
            // SAFETY: the blob's own bytes.
            String::from_utf8_lossy(unsafe {
                std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())
            })
            .into_owned()
        });
        assert!(
            compiled.is_ok() && said.as_deref().is_none_or(|s| s.trim().is_empty()),
            "wide {wide}, eight {eight}: {said:?}"
        );
    }
}

/// `pick` without the environment: both switches off is `CARD_COMPARE` and
/// wide alone is `WIDE`, by string equality; the eight-load forms are
/// neither, each holds its eight-load loop and not the one-load loop, and
/// is the one-load text with that loop swapped.
#[test]
fn pick_gives_todays_text_with_eight_off() {
    assert_eq!(hash::pick(false, false), hash::CARD_COMPARE);
    assert_eq!(hash::pick(true, false), hash::WIDE);
    let narrow8 = hash::pick(false, true);
    let wide8 = hash::pick(true, true);
    for (name, text) in [("narrow eight", narrow8), ("wide eight", wide8)] {
        assert_ne!(text, hash::CARD_COMPARE, "{name}");
        assert_ne!(text, hash::WIDE, "{name}");
    }
    assert_ne!(narrow8, wide8);
    assert!(narrow8.contains(hash::NARROW_EIGHT_LOOP) && !narrow8.contains(hash::NARROW_LOOP));
    assert!(wide8.contains(hash::WIDE_EIGHT_LOOP) && !wide8.contains(hash::WIDE_LOOP));
    assert_eq!(
        narrow8.replacen(hash::NARROW_EIGHT_LOOP, hash::NARROW_LOOP, 1),
        hash::CARD_COMPARE
    );
    assert_eq!(wide8.replacen(hash::WIDE_EIGHT_LOOP, hash::WIDE_LOOP, 1), hash::WIDE);
    // Made once: the same text at the same address each time.
    assert!(std::ptr::eq(narrow8, hash::pick(false, true)));
    assert!(std::ptr::eq(wide8, hash::pick(true, true)));
}

/// The line `program` logs once names the program `pick` gives, for each of
/// the four cases: its shape and its loads a step, and both switches.
#[test]
fn the_program_line_names_the_program_in_use() {
    assert_eq!(
        hash::program_text(false, false),
        "card hash program: narrow, one load a step (VICTORY_HASH_WIDE off, VICTORY_HASH_EIGHT off)"
    );
    assert_eq!(
        hash::program_text(true, false),
        "card hash program: wide, one load a step (VICTORY_HASH_WIDE=1, VICTORY_HASH_EIGHT off)"
    );
    assert_eq!(
        hash::program_text(false, true),
        "card hash program: narrow, eight loads a step (VICTORY_HASH_WIDE off, VICTORY_HASH_EIGHT=1)"
    );
    assert_eq!(
        hash::program_text(true, true),
        "card hash program: wide, eight loads a step (VICTORY_HASH_WIDE=1, VICTORY_HASH_EIGHT=1)"
    );
}
