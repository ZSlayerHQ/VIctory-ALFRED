//! The wide card hash (`host::hash::WIDE`) against its CPU model on the
//! software adapter: the GPU result is the model's for the shapes the copy
//! path has (a block, 63, 64 and 65 blocks, a full piece, one byte short,
//! several pieces, a skewed first piece, a ragged tail); every byte of the
//! covered range matters, in every lane and in the tail; swapping blocks
//! matters, in one lane and between lanes; bytes outside `lo..hi` do not;
//! and the compare pass flags exactly the pieces whose bytes changed. The
//! timing test is for a card machine and ignored otherwise.

use std::time::Instant;

use super::{Host, HostList, compile, hash, hash_model_tests as model, test_host};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

/// A deterministic pseudo-random source (xorshift), so a failure names the
/// same bytes every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The shader and its pipeline, for a test's own dispatches.
struct Program {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl Program {
    fn new(host: &Host, source: &str) -> Self {
        let code = compile(source, s!("main"), s!("cs_5_1")).expect("the card hash compiles");
        // SAFETY: plain object creation from our own compiled code.
        unsafe {
            let root: ID3D12RootSignature = host
                .device
                .CreateRootSignature(0, &code)
                .expect("a root signature");
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("the card hash pipeline");
            Self { root, pso }
        }
    }
}

/// One copy for [`gpu_hash_many`]: its bytes (padded past the copy's `size`
/// where the shader's 16-byte rounding reads) and the span constants, with
/// `first` so a case can hash a piece other than its first.
struct Variant<'a> {
    bytes: &'a [u8],
    first: u32,
    skew: u32,
    size: u32,
    pieces: u32,
}

/// The GPU addresses a dispatch writes and reads for one variant.
#[derive(Clone, Copy)]
struct Addrs {
    now: u64,
    hashes: u64,
    differs: u64,
}

/// One dispatch's span constants, as the shader's `Span` cbuffer.
#[derive(Clone, Copy)]
struct Span {
    first: u32,
    skew: u32,
    size: u32,
    pieces: u32,
    out_at: u32,
    compare: u32,
}

/// Runs one hash or compare pass and returns the `pieces` dwords of
/// `differs` it wrote (zero the hash pass leaves there). `out` is the
/// results buffer's mapping in this process.
fn run_pass(host: &Host, program: &Program, wide: bool, addrs: Addrs, span: Span, out: *const u32) -> Vec<u32> {
    let list = host.new_list().expect("a host list");
    record(&list, program, wide, addrs, span);
    list.close();
    host.execute(&[&list]);
    // SAFETY: inside the results mapping, after the queue's fence was
    // reached.
    unsafe { std::slice::from_raw_parts(out, span.pieces as usize).to_vec() }
}

/// Records one dispatch (or the wide form's split dispatches) for a variant.
fn record(list: &HostList, program: &Program, wide: bool, addrs: Addrs, span: Span) {
    // SAFETY: an open list; live resources; our own program.
    unsafe {
        list.list.SetComputeRootSignature(&program.root);
        list.list.SetPipelineState(&program.pso);
        list.list.SetComputeRootShaderResourceView(1, addrs.now);
        list.list.SetComputeRootUnorderedAccessView(2, addrs.hashes);
        list.list
            .SetComputeRootUnorderedAccessView(3, addrs.differs);
    }
    for (offset, groups) in hash::dispatch_parts_as(wide, span.pieces) {
        let constants = [
            span.first + offset,
            span.skew,
            span.size,
            span.pieces - offset,
            span.out_at + offset * 4,
            span.compare,
        ];
        // SAFETY: an open list; six constants at b0.
        unsafe {
            list.list
                .SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.list.Dispatch(groups, 1, 1);
        }
    }
}

/// Hashes every variant in one list (one dispatch or split each) and returns
/// each variant's piece hashes, in piece order from `first`.
fn gpu_hash_many(
    host: &Host,
    program: &Program,
    wide: bool,
    variants: &[Variant],
) -> Vec<Vec<[u32; 4]>> {
    assert!(!variants.is_empty());
    let stride = variants
        .iter()
        .map(|v| (v.bytes.len() + 15) & !15)
        .max()
        .unwrap()
        .max(16);
    let slots = variants
        .iter()
        .map(|v| v.first + v.pieces)
        .max()
        .unwrap()
        .max(1) as usize;
    let count = variants.len();
    let (now, now_at) = host
        .upload_buffer((stride * count) as u64)
        .expect("an upload buffer");
    let (hashes, hashes_at) = host
        .shared_buffer((slots * 16 * count) as u64)
        .expect("a hash buffer");
    let (differs, differs_at) = host
        .shared_buffer((slots * 4 * count) as u64)
        .expect("a differs buffer");
    // SAFETY: inside the buffers' mappings.
    unsafe {
        std::ptr::write_bytes(now_at, 0, stride * count);
        std::ptr::write_bytes(hashes_at, 0, slots * 16 * count);
        std::ptr::write_bytes(differs_at, 0, slots * 4 * count);
        for (i, variant) in variants.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                variant.bytes.as_ptr(),
                now_at.add(i * stride),
                variant.bytes.len(),
            );
        }
    }
    let list = host.new_list().expect("a host list");
    // SAFETY: live resources; the addresses are read by the list below.
    let (now_va, hashes_va, differs_va) = unsafe {
        (
            now.GetGPUVirtualAddress(),
            hashes.GetGPUVirtualAddress(),
            differs.GetGPUVirtualAddress(),
        )
    };
    for (i, v) in variants.iter().enumerate() {
        record(
            &list,
            program,
            wide,
            Addrs {
                now: now_va + (i * stride) as u64,
                hashes: hashes_va + (i * slots * 16) as u64,
                differs: differs_va + (i * slots * 4) as u64,
            },
            Span {
                first: v.first,
                skew: v.skew,
                size: v.size,
                pieces: v.pieces,
                out_at: 0,
                compare: 0,
            },
        );
    }
    list.close();
    host.execute(&[&list]);
    // SAFETY: the mappings, after the queue's fence was reached.
    let out = unsafe { std::slice::from_raw_parts(hashes_at.cast::<[u32; 4]>(), slots * count) };
    variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            (0..v.pieces)
                .map(|p| out[i * slots + (v.first + p) as usize])
                .collect()
        })
        .collect()
}

/// [`gpu_hash_many`] for one copy.
fn gpu_hashes(host: &Host, program: &Program, wide: bool, variant: Variant) -> Vec<[u32; 4]> {
    gpu_hash_many(host, program, wide, &[variant])
        .pop()
        .unwrap()
}

/// The model's hash of one piece of a copy.
fn model_hash(bytes: &[u8], wide: bool, piece: u32, skew: u32, size: u32) -> [u32; 4] {
    let (lo, hi) = model::piece_range(piece, skew, size);
    let blocks = model::blocks(bytes, lo, hi);
    if wide {
        model::wide(&blocks)
    } else {
        model::narrow(&blocks)
    }
}

/// The GPU result is the model's for a random buffer of every shape the
/// copy path cuts: one block, 63, 64 and 65 blocks, a full piece (a copy of
/// one piece), one byte short, several pieces, a skewed first piece and a
/// ragged tail, for both forms.
#[test]
fn the_wide_gpu_hash_is_the_model_for_every_shape() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let narrow = Program::new(host, hash::CARD_COMPARE);
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    let shapes: [(&str, u32, u32); 9] = [
        ("one block", 0, 16),
        ("63 blocks", 0, 63 * 16),
        ("64 blocks", 0, 64 * 16),
        ("65 blocks", 0, 65 * 16),
        ("a full piece", 0, 65_536),
        ("one byte short", 0, 65_535),
        ("several pieces", 0, 3 * 65_536),
        ("a skewed first piece", 0x1234, 2 * 65_536),
        ("a ragged tail", 0, 2 * 65_536 + 13),
    ];
    for (name, skew, size) in shapes {
        let pieces = size.div_ceil(65_536).max(1);
        let padded = ((size as usize + 15) & !15).max(16);
        let bytes: Vec<u8> = (0..padded).map(|_| rng.byte()).collect();
        for (form, is_wide, program) in [("wide", true, &wide), ("narrow", false, &narrow)] {
            let got = gpu_hashes(
                host,
                program,
                is_wide,
                Variant {
                    bytes: &bytes,
                    first: 0,
                    skew,
                    size,
                    pieces,
                },
            );
            for piece in 0..pieces {
                let want = model_hash(&bytes, is_wide, piece, skew, size);
                assert_eq!(got[piece as usize], want, "{name}: {form}, piece {piece}");
            }
        }
    }
}

/// The covered range of one piece, and a few hundred bit positions in it:
/// for every lane the first and the last block of its chain, and every byte
/// of the piece's last (ragged) block.
fn positions(lo: usize, hi: usize) -> Vec<(usize, u8)> {
    let blocks = (hi - lo) / 16;
    assert!(blocks >= 64, "every lane has a block");
    let mut out = Vec::new();
    for lane in 0..64usize {
        let first = lane;
        let last = lane + 64 * ((blocks - 1 - lane) / 64);
        for block in [first, last] {
            out.push((lo + block * 16, 0x01));
            out.push((lo + block * 16 + 15, 0x80));
        }
    }
    for byte in 0..16 {
        out.push((lo + (blocks - 1) * 16 + byte, 0x01));
    }
    out
}

/// Every byte of the covered range matters: flipping one bit at each of a
/// few hundred positions (every lane's first and last block, and the tail)
/// changes the wide hash, for a full piece and for each ragged shape.
#[test]
fn every_byte_of_a_piece_matters_in_the_wide_hash() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let mut rng = Rng::new(0x85eb_ca6b_27d4_eb2f);
    let shapes: [(&str, u32, u32, u32); 4] = [
        // The piece hashed, the copy's skew and size: a full piece; one
        // byte short; a ragged tail in the third piece; a skewed first piece.
        ("a full piece", 0, 0, 65_536),
        ("one byte short", 0, 0, 65_535),
        ("a ragged tail", 2, 0, 2 * 65_536 + 0x4001),
        ("a skewed first piece", 0, 0x1234, 2 * 65_536),
    ];
    for (name, piece, skew, size) in shapes {
        let padded = ((size as usize + 15) & !15).max(16);
        let bytes: Vec<u8> = (0..padded).map(|_| rng.byte()).collect();
        let (lo, hi) = model::piece_range(piece, skew, size);
        let flips = positions(lo, hi);
        let mut variants = Vec::with_capacity(flips.len() + 1);
        variants.push(Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        });
        let mut changed: Vec<Vec<u8>> = Vec::with_capacity(flips.len());
        for &(at, bit) in &flips {
            let mut one = bytes.clone();
            one[at] ^= bit;
            changed.push(one);
        }
        for one in &changed {
            variants.push(Variant {
                bytes: one,
                first: piece,
                skew,
                size,
                pieces: 1,
            });
        }
        let got = gpu_hash_many(host, &wide, true, &variants);
        let base = got[0][0];
        for (i, one) in changed.iter().enumerate() {
            assert_ne!(
                got[i + 1][0],
                base,
                "{name}: the bit at {} changed nothing",
                flips[i].0
            );
            assert_eq!(
                got[i + 1][0],
                model_hash(one, true, piece, skew, size),
                "{name}: the model at {}",
                flips[i].0
            );
        }
    }
}

/// Swapping two different 16-byte blocks changes the wide hash: blocks of
/// one lane (the chain's order), and blocks of two lanes (which chain gets
/// which bytes). A lane-wise XOR-like combine would miss the second.
#[test]
fn swapping_blocks_changes_the_wide_hash() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let mut rng = Rng::new(0xc2b2_ae35_27d4_eb2f);
    let bytes: Vec<u8> = (0..65_536).map(|_| rng.byte()).collect();
    let swapped = |a: usize, b: usize| {
        let mut one = bytes.clone();
        for i in 0..16 {
            one.swap(a * 16 + i, b * 16 + i);
        }
        one
    };
    // Blocks 0 and 64 are steps 0 and 1 of lane 0's chain; blocks 0 and 1
    // are lane 0's and lane 1's first steps.
    let same_lane = swapped(0, 64);
    let other_lanes = swapped(0, 1);
    let far_lanes = swapped(5, 70);
    let variants = [
        Variant {
            bytes: &bytes,
            first: 0,
            skew: 0,
            size: 65_536,
            pieces: 1,
        },
        Variant {
            bytes: &same_lane,
            first: 0,
            skew: 0,
            size: 65_536,
            pieces: 1,
        },
        Variant {
            bytes: &other_lanes,
            first: 0,
            skew: 0,
            size: 65_536,
            pieces: 1,
        },
        Variant {
            bytes: &far_lanes,
            first: 0,
            skew: 0,
            size: 65_536,
            pieces: 1,
        },
    ];
    let got = gpu_hash_many(host, &wide, true, &variants);
    let base = got[0][0];
    for (name, i) in [
        ("two blocks of lane 0", 1),
        ("lane 0's and lane 1's first blocks", 2),
        ("lane 5's and lane 70's first blocks", 3),
    ] {
        assert_ne!(got[i][0], base, "{name}: the swap changed nothing");
        assert_eq!(
            got[i][0],
            model_hash(variants[i].bytes, true, 0, 0, 65_536),
            "{name}: the model"
        );
    }
}

/// Bytes outside a piece's `lo..hi` do not matter, as in the narrow form:
/// a ragged tail's bytes past its rounded end, and a skewed piece's bytes
/// before its rounded start, leave the hash unchanged (both forms).
#[test]
fn bytes_outside_the_piece_range_do_not_matter() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let narrow = Program::new(host, hash::CARD_COMPARE);
    let mut rng = Rng::new(0x27d4_eb2f_9e37_79b9);
    // A ragged tail: piece 2 of a copy of two pieces and 13 bytes.
    let (piece, skew, size) = (2u32, 0u32, 2 * 65_536 + 13);
    let (_, hi) = model::piece_range(piece, skew, size);
    let mut bytes: Vec<u8> = (0..hi + 64).map(|_| rng.byte()).collect();
    let base_wide = gpu_hashes(
        host,
        &wide,
        true,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    let base_narrow = gpu_hashes(
        host,
        &narrow,
        false,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    for at in hi..hi + 64 {
        bytes[at] ^= 0x5a;
    }
    let after_wide = gpu_hashes(
        host,
        &wide,
        true,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    let after_narrow = gpu_hashes(
        host,
        &narrow,
        false,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    assert_eq!(
        after_wide, base_wide,
        "the wide hash saw bytes past the ragged end"
    );
    assert_eq!(
        after_narrow, base_narrow,
        "the narrow hash saw bytes past the ragged end"
    );
    assert_eq!(
        after_wide[0],
        model_hash(&bytes, true, piece, skew, size),
        "the wide model"
    );

    // A skewed piece: piece 1 starts 0x1234 bytes into the copy; its
    // rounded start leaves bytes before it out.
    let (piece, skew, size) = (1u32, 0x1234u32, 2 * 65_536);
    let (lo, hi) = model::piece_range(piece, skew, size);
    let mut bytes: Vec<u8> = (0..hi + 64).map(|_| rng.byte()).collect();
    let base_wide = gpu_hashes(
        host,
        &wide,
        true,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    let base_narrow = gpu_hashes(
        host,
        &narrow,
        false,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    for at in lo - 64..lo {
        bytes[at] ^= 0x5a;
    }
    for at in hi..hi + 64 {
        bytes[at] ^= 0x5a;
    }
    let after_wide = gpu_hashes(
        host,
        &wide,
        true,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    let after_narrow = gpu_hashes(
        host,
        &narrow,
        false,
        Variant {
            bytes: &bytes,
            first: piece,
            skew,
            size,
            pieces: 1,
        },
    );
    assert_eq!(
        after_wide, base_wide,
        "the wide hash saw bytes outside the skewed range"
    );
    assert_eq!(
        after_narrow, base_narrow,
        "the narrow hash saw bytes outside the skewed range"
    );
    assert_eq!(
        after_wide[0],
        model_hash(&bytes, true, piece, skew, size),
        "the wide model"
    );
}

/// The compare pass with the wide program flags exactly the pieces whose
/// bytes changed between the hash pass and the compare pass: 200 pieces,
/// 14 of them (7%) changed at a random byte.
#[test]
fn the_wide_compare_flags_exactly_the_changed_pieces() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    const PIECES: u32 = 200;
    const CHANGED: usize = 14;
    let size = PIECES * 65_536;
    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    let mut bytes: Vec<u8> = (0..size as usize).map(|_| rng.byte()).collect();
    let (now, now_at) = host.upload_buffer(size as u64).expect("an upload buffer");
    // SAFETY: inside the buffer's mapping.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), now_at, bytes.len()) };
    let (hashes, hashes_at) = host
        .shared_buffer(PIECES as u64 * 16)
        .expect("a hash buffer");
    let (differs, differs_at) = host
        .shared_buffer(PIECES as u64 * 4)
        .expect("a differs buffer");
    // SAFETY: inside the mappings.
    unsafe {
        std::ptr::write_bytes(hashes_at, 0, PIECES as usize * 16);
        std::ptr::write_bytes(differs_at, 0, PIECES as usize * 4);
    }
    let hash = host.new_list().expect("a host list");
    // SAFETY: live resources; the addresses are read by the lists below.
    let (now_va, hashes_va, differs_va) = unsafe {
        (
            now.GetGPUVirtualAddress(),
            hashes.GetGPUVirtualAddress(),
            differs.GetGPUVirtualAddress(),
        )
    };
    record(
        &hash,
        &wide,
        true,
        Addrs { now: now_va, hashes: hashes_va, differs: differs_va },
        Span {
            first: 0,
            skew: 0,
            size,
            pieces: PIECES,
            out_at: 0,
            compare: 0,
        },
    );
    hash.close();
    host.execute(&[&hash]);
    // Exactly 7% of the pieces, each at a random byte.
    let mut order: Vec<u32> = (0..PIECES).collect();
    for i in (1..order.len()).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let mut changed = vec![false; PIECES as usize];
    for &piece in &order[..CHANGED] {
        changed[piece as usize] = true;
        let at = piece as usize * 65_536 + rng.below(65_536);
        bytes[at] ^= 0x80;
        // SAFETY: inside the upload buffer's mapping, after the hash pass's
        // submission was waited for.
        unsafe { *now_at.add(at) ^= 0x80 };
    }
    let compare = host.new_list().expect("a host list");
    record(
        &compare,
        &wide,
        true,
        Addrs { now: now_va, hashes: hashes_va, differs: differs_va },
        Span {
            first: 0,
            skew: 0,
            size,
            pieces: PIECES,
            out_at: 0,
            compare: 1,
        },
    );
    compare.close();
    host.execute(&[&compare]);
    // SAFETY: inside the mapping, after the queue's fence was reached.
    let got = unsafe { std::slice::from_raw_parts(differs_at.cast::<u32>(), PIECES as usize) };
    for piece in 0..PIECES as usize {
        assert_eq!(got[piece], u32::from(changed[piece]), "piece {piece}");
    }
    assert_eq!(
        changed.iter().filter(|&&c| c).count(),
        CHANGED,
        "the changed share"
    );
}

/// The compare flags exactly the changed pieces in mode 1 and in mode 2,
/// at a non-zero piece offset (`first = 7`), the changes including the
/// reviewer's two last-block flips: mode 1 does not store its hashes, mode
/// 2 does, and the next compare finds nothing.
#[test]
fn the_wide_compare_flags_exactly_the_changed_pieces_in_both_modes_at_an_offset() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    const FIRST: u32 = 7;
    const PIECES: u32 = 3;
    const SIZE: u32 = 10 * 65_536;
    let mut rng = Rng::new(0x1319_8a2e_0370_7344);
    let mut bytes: Vec<u8> = (0..SIZE as usize).map(|_| rng.byte()).collect();
    let (now, now_at) = host.upload_buffer(SIZE as u64).expect("an upload buffer");
    // SAFETY: inside the buffer's mapping.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), now_at, bytes.len()) };
    let (hashes, _hashes_at) = host
        .shared_buffer(u64::from(FIRST + PIECES) * 16)
        .expect("a hash buffer");
    let (differs, differs_at) = host
        .shared_buffer(u64::from(PIECES) * 4)
        .expect("a differs buffer");
    // SAFETY: inside the mapping.
    unsafe { std::ptr::write_bytes(differs_at, 0, PIECES as usize * 4) };
    // SAFETY: live resources; the addresses are read by the lists.
    let addrs = unsafe {
        Addrs {
            now: now.GetGPUVirtualAddress(),
            hashes: hashes.GetGPUVirtualAddress(),
            differs: differs.GetGPUVirtualAddress(),
        }
    };
    let out = differs_at.cast::<u32>();
    let span = |compare| Span {
        first: FIRST,
        skew: 0,
        size: SIZE,
        pieces: PIECES,
        out_at: 0,
        compare,
    };
    // The hash pass stores the three hashes.
    let _ = run_pass(host, &wide, true, addrs, span(0), out);
    // Piece 7 takes the reviewer's pattern: bit 31 of word 0 in the last
    // blocks of two lanes; piece 9 takes one byte.
    let mut change = |at: usize, bit: u8| {
        bytes[at] ^= bit;
        // SAFETY: inside the upload buffer's mapping, after the hash pass
        // was waited for.
        unsafe { *now_at.add(at) ^= bit };
    };
    for block in [4032usize, 4033] {
        change(FIRST as usize * 65_536 + block * 16 + 3, 0x80);
    }
    change((FIRST as usize + 2) * 65_536 + 1234, 0x40);
    // Mode 1: the two changed pieces, and only them. It stores no hash, so
    // a second mode-1 compare says the same.
    for _ in 0..2 {
        let got = run_pass(host, &wide, true, addrs, span(1), out);
        assert_eq!(got, vec![1, 0, 1], "mode 1 at piece {FIRST}");
    }
    // Mode 2: the same flags, and the hashes are stored.
    let got = run_pass(host, &wide, true, addrs, span(2), out);
    assert_eq!(got, vec![1, 0, 1], "mode 2 at piece {FIRST}");
    let got = run_pass(host, &wide, true, addrs, span(1), out);
    assert_eq!(got, vec![0, 0, 0], "after mode 2 stored the hashes");
    // A new change in the middle piece: only it.
    change((FIRST as usize + 1) * 65_536 + 65, 0x08);
    let got = run_pass(host, &wide, true, addrs, span(1), out);
    assert_eq!(got, vec![0, 1, 0], "only the middle piece changed");
}

/// The compare across the 65,535-group split: a run of 65,536 pieces is
/// hashed and compared in two dispatches, the second holding only pieces
/// past the copy's covered bytes. The changed piece is inside the first
/// part, the reviewer's pattern among its changes, and the split's second
/// dispatch writes neither a flag nor a hash of its own.
#[test]
fn the_wide_compare_flags_across_the_split_in_both_modes() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    const PIECES: u32 = 65_536;
    const SIZE: u32 = 2 * 65_536;
    let mut rng = Rng::new(0x510e_527f_ade6_82d1);
    let mut bytes: Vec<u8> = (0..SIZE as usize).map(|_| rng.byte()).collect();
    assert_eq!(
        hash::dispatch_parts_as(true, PIECES).collect::<Vec<_>>(),
        vec![(0, 65_535), (65_535, 1)]
    );
    let (now, now_at) = host.upload_buffer(SIZE as u64).expect("an upload buffer");
    // SAFETY: inside the buffer's mapping.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), now_at, bytes.len()) };
    let (hashes, _hashes_at) = host
        .shared_buffer(u64::from(PIECES) * 16)
        .expect("a hash buffer");
    let (differs, differs_at) = host
        .shared_buffer(u64::from(PIECES) * 4)
        .expect("a differs buffer");
    // SAFETY: inside the mapping.
    unsafe { std::ptr::write_bytes(differs_at, 0, PIECES as usize * 4) };
    // SAFETY: live resources; the addresses are read by the lists.
    let addrs = unsafe {
        Addrs {
            now: now.GetGPUVirtualAddress(),
            hashes: hashes.GetGPUVirtualAddress(),
            differs: differs.GetGPUVirtualAddress(),
        }
    };
    let out = differs_at.cast::<u32>();
    let span = |compare| Span { first: 0, skew: 0, size: SIZE, pieces: PIECES, out_at: 0, compare };
    let _ = run_pass(host, &wide, true, addrs, span(0), out);
    // Piece 1's last two blocks take the reviewer's sign-bit pattern; piece
    // 0 one byte.
    for block in [4032usize, 4033] {
        let at = 65_536 + block * 16 + 3;
        bytes[at] ^= 0x80;
        // SAFETY: inside the upload buffer's mapping.
        unsafe { *now_at.add(at) ^= 0x80 };
    }
    let at = 777;
    bytes[at] ^= 0x02;
    // SAFETY: inside the upload buffer's mapping.
    unsafe { *now_at.add(at) ^= 0x02 };
    let expected = |piece: u32| u32::from(piece == 0 || piece == 1);
    for _ in 0..2 {
        let got = run_pass(host, &wide, true, addrs, span(1), out);
        assert_eq!(got.len(), PIECES as usize);
        for (piece, &flag) in got.iter().enumerate() {
            assert_eq!(flag, expected(piece as u32), "mode 1, piece {piece}");
        }
    }
    let got = run_pass(host, &wide, true, addrs, span(2), out);
    for (piece, &flag) in got.iter().enumerate() {
        assert_eq!(flag, expected(piece as u32), "mode 2, piece {piece}");
    }
    let got = run_pass(host, &wide, true, addrs, span(1), out);
    assert!(got.iter().all(|&flag| flag == 0), "after mode 2 stored the hashes");
    // A change in the second part's piece 1 (the first part's last piece)
    // still flags only it, through both dispatches.
    let at = 65_536 + 99;
    bytes[at] ^= 0x10;
    // SAFETY: inside the upload buffer's mapping.
    unsafe { *now_at.add(at) ^= 0x10 };
    let got = run_pass(host, &wide, true, addrs, span(1), out);
    for (piece, &flag) in got.iter().enumerate() {
        assert_eq!(flag, u32::from(piece == 1), "after the second change, piece {piece}");
    }
}

/// A wide run over more pieces than one dispatch takes is split, with each
/// part's `first` continuing where the one before ended; the narrow form
/// keeps its single dispatch.
#[test]
fn a_wide_run_is_split_at_the_dispatchs_most_groups() {
    assert_eq!(
        hash::dispatch_parts_as(true, 65_536).collect::<Vec<_>>(),
        vec![(0, 65_535), (65_535, 1)]
    );
    assert_eq!(
        hash::dispatch_parts_as(true, 130_000).collect::<Vec<_>>(),
        vec![(0, 65_535), (65_535, 64_465)]
    );
    assert_eq!(
        hash::dispatch_parts_as(true, 1).collect::<Vec<_>>(),
        vec![(0, 1)]
    );
    assert_eq!(hash::dispatch_parts_as(true, 0).count(), 0);
    assert_eq!(
        hash::dispatch_parts_as(false, 65_536).collect::<Vec<_>>(),
        vec![(0, 1024)]
    );
    assert_eq!(
        hash::dispatch_parts_as(false, 0).collect::<Vec<_>>(),
        vec![(0, 0)]
    );
}

/// The reviewer's finding 1 on the card: bit 31 of word 0 flipped in blocks
/// 4032 and 4033 (the last block of lanes 0 and 1) of a full piece, over
/// zero and random contents, and the swap of two blocks that differ only in
/// that bit. The old fold cancelled both deltas, so the wide hash read the
/// same while the narrow form saw the change.
#[test]
fn the_reviewers_cancelling_pair_changes_the_gpu_hash() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let narrow = Program::new(host, hash::CARD_COMPARE);
    let mut rng = Rng::new(0x6a09_e667_f3bc_c908);
    for (name, zero) in [("a zero piece", true), ("contents", false)] {
        let mut bytes: Vec<u8> = if zero {
            vec![0u8; 65_536]
        } else {
            (0..65_536).map(|_| rng.byte()).collect()
        };
        let base = gpu_hashes(
            host,
            &wide,
            true,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        let narrow_base = gpu_hashes(
            host,
            &narrow,
            false,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        for block in [4032usize, 4033] {
            // Bit 31 of word 0, little-endian: the top bit of the block's
            // fourth byte.
            bytes[block * 16 + 3] ^= 0x80;
        }
        let flipped = gpu_hashes(
            host,
            &wide,
            true,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        let narrow_flipped = gpu_hashes(
            host,
            &narrow,
            false,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        assert_ne!(flipped[0], base[0], "{name}: the two last-block flips read the same");
        assert_ne!(narrow_flipped[0], narrow_base[0], "{name}: the narrow form read the same");
    }
    for (name, zero) in [("a zero piece", true), ("contents", false)] {
        let mut bytes: Vec<u8> = if zero {
            vec![0u8; 65_536]
        } else {
            (0..65_536).map(|_| rng.byte()).collect()
        };
        // The last blocks of lanes 0 and 1 differ only in bit 31 of word 0.
        bytes[4033 * 16 + 3] ^= 0x80;
        let base = gpu_hashes(
            host,
            &wide,
            true,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        for i in 0..16 {
            bytes.swap(4032 * 16 + i, 4033 * 16 + i);
        }
        let swapped = gpu_hashes(
            host,
            &wide,
            true,
            Variant { bytes: &bytes, first: 0, skew: 0, size: 65_536, pieces: 1 },
        );
        assert_ne!(swapped[0], base[0], "{name}: the swapped last blocks read the same");
    }
}

/// Opus's review of `job/hash-wide-fix`, 2, on the card: bit 31 of word k flipped in block t,
/// then bit 31 of word k-1 and bit 16 of word k in block t + 64 (the same lane's next block),
/// every word, three places, over a zero piece and random contents. The wide program's hash
/// changes and is the model's; the narrow program, kept as it was, loses the same pattern
/// between consecutive blocks (t and t + 1), as its model says.
#[test]
fn the_reviews_top_bit_pattern_changes_the_wide_gpu_hash() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let narrow = Program::new(host, hash::CARD_COMPARE);
    let mut rng = Rng::new(0x9216_d5d9_8979_fb1b);
    // Bit `bit` of word `word` of block `block`, little-endian.
    let flip = |bytes: &mut [u8], block: usize, word: usize, bit: usize| {
        bytes[block * 16 + word * 4 + bit / 8] ^= 1 << (bit % 8);
    };
    for zero in [true, false] {
        let base: Vec<u8> = if zero { vec![0u8; 65_536] } else { (0..65_536).map(|_| rng.byte()).collect() };
        let mut cases: Vec<(bool, Vec<u8>)> = vec![(true, base.clone()), (false, base.clone())];
        for k in 0..4 {
            for t in [0usize, 1000, 4031] {
                for (is_wide, next) in [(true, t + 64), (false, t + 1)] {
                    let mut bytes = base.clone();
                    flip(&mut bytes, t, k, 31);
                    flip(&mut bytes, next, (k + 3) % 4, 31);
                    flip(&mut bytes, next, k, 16);
                    cases.push((is_wide, bytes));
                }
            }
        }
        for is_wide in [true, false] {
            let picked: Vec<&Vec<u8>> = cases.iter().filter(|c| c.0 == is_wide).map(|c| &c.1).collect();
            let variants: Vec<Variant> = picked
                .iter()
                .map(|bytes| Variant { bytes: bytes.as_slice(), first: 0, skew: 0, size: 65_536, pieces: 1 })
                .collect();
            let program = if is_wide { &wide } else { &narrow };
            let got = gpu_hash_many(host, program, is_wide, &variants);
            for (bytes, hash) in picked.iter().zip(&got) {
                assert_eq!(hash[0], model_hash(bytes, is_wide, 0, 0, 65_536), "the card is the model (wide: {is_wide})");
            }
            for (i, hash) in got.iter().enumerate().skip(1) {
                if is_wide {
                    assert_ne!(hash[0], got[0][0], "zero: {zero}, case {i}: the wide hash kept the pattern");
                } else {
                    assert_eq!(hash[0], got[0][0], "zero: {zero}, case {i}: the narrow form's known weakness");
                }
            }
        }
    }
}

/// A card machine's figures: both forms on 16 MB and on a single piece.
/// Ignored elsewhere (the software adapter's numbers say nothing about a
/// card).
#[test]
#[ignore = "a card machine: prints the two forms' times"]
fn the_forms_times_on_a_card() {
    let Some(host) = test_host() else {
        return; // no host GPU here
    };
    let wide = Program::new(host, hash::WIDE);
    let narrow = Program::new(host, hash::CARD_COMPARE);
    let mut rng = Rng::new(0x0bad_c0de_0bad_c0de);
    for (name, size, pieces) in [
        ("16 MB", 16 << 20, (16 << 20) / 65_536),
        ("a single piece", 65_536, 1),
    ] {
        let bytes: Vec<u8> = (0..size).map(|_| rng.byte()).collect();
        for (form, is_wide, program) in [("wide", true, &wide), ("narrow", false, &narrow)] {
            let started = Instant::now();
            let _ = gpu_hashes(
                host,
                program,
                is_wide,
                Variant {
                    bytes: &bytes,
                    first: 0,
                    skew: 0,
                    size,
                    pieces,
                },
            );
            println!(
                "{name}, {form}: {:.3} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

/// The eight-load forms against the one-load forms, with these helpers.
#[path = "eight.rs"]
mod eight;
