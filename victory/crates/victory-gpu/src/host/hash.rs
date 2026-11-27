//! The card hash programs and which one this process runs.
//!
//! Both hash a copy's 64 KiB pieces and write 16 bytes a piece into the
//! hash buffer; the same program compares a copy's pieces with those
//! hashes, storing a dword a piece in `differs`. The narrow program gives
//! one thread a piece (a group of 64 threads hashes 64 pieces); the wide
//! program gives one group of 64 threads a piece, each thread a 16-byte
//! step in 64, and folds the 64 lane states after a group barrier.
//!
//! The wide form is another function of a piece's bytes, so it is chosen
//! once (`VICTORY_HASH_WIDE=1`), never while the process runs: every stored
//! hash is written by the same program and only ever compared with a hash
//! from that program, and a switch flipped mid-run would make every piece
//! compared afterwards look changed until its hash was written again. The
//! choice is fixed the first time the card's work is made (`Host::card_ready`
//! → `make_card_work` → [`program`]), which is the first copy that could go
//! to the card's memory, not at process start; both passes and flights
//! already queued use it.
//!
//! Either program has an eight-load form (`VICTORY_HASH_EIGHT=1`, off by
//! default, read once like the wide switch): its main loop issues eight
//! `Load4`s before the eight steps they feed, in ascending address order,
//! then the one-load loop takes what is left. Each thread steps through the
//! same blocks in the same order, so the hash values are bit for bit those of
//! the one-load form ([`NARROW_EIGHT_LOOP`], [`WIDE_EIGHT_LOOP`]); only how
//! many loads a thread has in flight changes. The eight-load texts are made
//! from [`CARD_COMPARE`] and [`WIDE`] at first use by swapping the loop
//! ([`pick`]), so the one-load texts are never edited for them.

use std::sync::OnceLock;

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one thread per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, 1 where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// The wide form of [`CARD_COMPARE`]: one group of 64 threads for a piece.
/// Thread `lane` starts from the same four constants plus `lane` and takes
/// the piece's 16-byte blocks `lane`, `lane+64`, `lane+128`, ... (a stride
/// of 1,024 bytes) through `Step`: the narrow program's step, then a second
/// odd multiply of every word. Each lane's final state then goes through
/// `FinaliseLane`, which mixes the lane number in and mixes each word within
/// itself. After a group barrier thread 0 starts from `Seed(len)` — the
/// piece's covered byte length, so an appended block or a shorter range
/// cannot keep a hash — and folds the 64 lane states in lane order with
/// `Step`, then through `Finalise`. That value is the piece's 128-bit hash.
/// The lane seeds, the finalisers and the seed all make the result depend on
/// which block is where: a fixed delta in two lanes does not cancel in the
/// first fold step, and swapping two unequal blocks changes it, in one lane
/// or in two. What is proved of it and what only sampled:
/// `docs/card-hash-wide.md`. The range is the narrow
/// program's (`lo`/`hi` clamped to the copy, rounded out to 16 bytes, `skew`
/// in piece 0), and so are the stores: 16 bytes a piece in `hashes`, and
/// `1`/`0` a piece in `differs` (the fold is kept, not folded into a
/// dword).
///
/// A dispatch holds at most 65,535 groups, so a run of more pieces is split
/// (see [`dispatch_parts`]); `first` is the dispatch's first piece and
/// `pieces` its count, as in the narrow program.
pub(crate) const WIDE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 parts[64];

// One 16-byte step, in the lanes and in the fold: the narrow program's
// step, then a second odd multiply of every word. For a fixed block each
// part is a bijection of the state (the xor; an odd multiply, invertible
// mod 2^32; the rotate-and-shift, the rotation times one plus a nilpotent
// part, since three shifts by 15 leave nothing; the second odd multiply),
// so the step is one, and a change in one block always reaches the lane's
// final state. The narrow step alone is linear for a top-bit change
// ((a ^ 2^31) * M == (a * M) ^ 2^31): bit 31 of word k leaves exactly bit
// 31 of word k-1 and bit 16 of word k changed whatever the state, and the
// next block can put them back. The second multiply turns that bit 16 into
// a change carried by the state's own bits.
uint4 Step(uint4 h, uint4 v) {
    h = (h ^ v) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
    h = h.yzwx ^ (h >> 15);
    return h * uint4(0x7FEB352Du, 0x846CA68Bu, 0x297A2D39u, 0x5BD1E995u);
}

// The lane's final state: the lane number is mixed in, then three rounds
// each with three xor-shifts and two odd multiplies a word, and a rotation
// of the words after each round. The rotation only moves words: no word
// mixes into another, so an input bit can reach only bits of its own word,
// never the other three (the mixing across words is the chain's and the
// fold's). Each round is a bijection.
uint4 FinaliseLane(uint4 h, uint lane) {
    h = h ^ uint4(0x9E3779B9u ^ lane,
                  0x85EBCA6Bu + lane * 0x9E3779B1u,
                  0xC2B2AE35u ^ lane * 0x85EBCA77u,
                  0x27D4EB2Fu + lane * 0xC2B2AE3Du);
    [unroll]
    for (uint i = 0u; i < 3u; i++) {
        h = h ^ (h >> 16);
        h = h * uint4(0x7FEB352Du, 0x846CA68Bu, 0x9E3779B1u, 0x85EBCA77u);
        h = h ^ (h >> 15);
        h = h * uint4(0x01000193u, 0xC2B2AE3Du, 0x27D4EB2Fu, 0x9E3779B9u);
        h = h ^ (h >> 13);
        h = h.yzwx;
    }
    return h;
}

// The final state after the fold: the same shape of rounds, with other odd
// multipliers; as in `FinaliseLane`, each word is mixed within itself only.
uint4 Finalise(uint4 h) {
    h = h ^ uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [unroll]
    for (uint i = 0u; i < 3u; i++) {
        h = h ^ (h >> 16);
        h = h * uint4(0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu, 0x9E3779B9u);
        h = h ^ (h >> 13);
        h = h * uint4(0x9E3779B1u, 0x01000193u, 0x7FEB352Du, 0x846CA68Bu);
        h = h ^ (h >> 16);
        h = h.yzwx;
    }
    return h;
}

// The fold's seed: the piece's covered byte length, mixed in four ways.
uint4 Seed(uint len) {
    return uint4(0x9E3779B9u ^ len,
                 0x85EBCA6Bu + len,
                 0xC2B2AE35u ^ (len * 0x01000193u),
                 0x27D4EB2Fu + (len * 0x9E3779B1u));
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    if (gid.x >= pieces) return;
    uint piece = first + gid.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint len = hi > lo ? hi - lo : 0u;
    uint4 h = uint4(0x9E3779B9u + lane, 0x85EBCA6Bu + lane, 0xC2B2AE35u + lane, 0x27D4EB2Fu + lane);
    [loop]
    for (uint at = lo + lane * 16u; at < hi; at += 1024u) {
        h = Step(h, now.Load4(at));
    }
    parts[lane] = FinaliseLane(h, lane);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        uint4 f = Seed(len);
        [loop]
        for (uint i = 0u; i < 64u; i++) {
            f = Step(f, parts[i]);
        }
        f = Finalise(f);
        if (compare != 0u) {
            differs.Store(out_at + gid.x * 4u, any(f != hashes.Load4(piece * 16u)) ? 1u : 0u);
            // With the copy-back on its own thread: the next compare of the
            // piece finds what is written from here on.
            if (compare == 2u) hashes.Store4(piece * 16u, f);
        } else {
            hashes.Store4(piece * 16u, f);
        }
    }
}
"#;

/// Whether the card hash is [`WIDE`] rather than [`CARD_COMPARE`]:
/// `VICTORY_HASH_WIDE=1`, off by default. Read once through a `OnceLock`
/// and never again; the first ask is `make_card_work`, so the first card
/// work made fixes the choice for the process. That is because the wide form
/// hashes a piece's bytes to another value: flipping it mid-run would make
/// every piece compared afterwards look changed until its hash was written
/// again (a spurious fetch of up to the declared pieces). A live switch
/// would need the hash buffer kept per mode. With it off, the narrow program
/// and dispatch are used unchanged.
pub(crate) fn wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HASH_WIDE").is_some_and(|v| v == "1"))
}

/// Whether the card hash is the eight-load form of the program [`wide`]
/// picks: `VICTORY_HASH_EIGHT=1`, off by default. Read once through a
/// `OnceLock`, like [`wide`]. The eight-load forms hash to the same values
/// as the one-load forms, so a flip could not make pieces look changed; it
/// is read once all the same, so one process runs one program text.
pub(crate) fn eight() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HASH_EIGHT").is_some_and(|v| v == "1"))
}

/// The program both passes run: [`pick`] of [`wide`] and [`eight`]. With
/// both off, [`CARD_COMPARE`], as before the switches. The first call says
/// once which program the process runs ([`program_text`]), so that a run's
/// log shows whether `VICTORY_HASH_WIDE` and `VICTORY_HASH_EIGHT` took.
pub(crate) fn program() -> &'static str {
    let (wide, eight) = (wide(), eight());
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| tracing::info!("{}", program_text(wide, eight)));
    pick(wide, eight)
}

/// The line [`program`] logs once: the card hash program [`pick`] gives for
/// the two switches, by its shape and its loads a step.
pub(crate) fn program_text(wide: bool, eight: bool) -> &'static str {
    match (wide, eight) {
        (false, false) => "card hash program: narrow, one load a step (VICTORY_HASH_WIDE off, VICTORY_HASH_EIGHT off)",
        (true, false) => "card hash program: wide, one load a step (VICTORY_HASH_WIDE=1, VICTORY_HASH_EIGHT off)",
        (false, true) => "card hash program: narrow, eight loads a step (VICTORY_HASH_WIDE off, VICTORY_HASH_EIGHT=1)",
        (true, true) => "card hash program: wide, eight loads a step (VICTORY_HASH_WIDE=1, VICTORY_HASH_EIGHT=1)",
    }
}

/// The program for the two switches, without the environment: [`WIDE`] or
/// [`CARD_COMPARE`] as they are with `eight` off, else that program with its
/// loop swapped for the eight-load loop (made once, then kept).
pub(crate) fn pick(wide: bool, eight: bool) -> &'static str {
    static NARROW_EIGHT: OnceLock<String> = OnceLock::new();
    static WIDE_EIGHT: OnceLock<String> = OnceLock::new();
    match (wide, eight) {
        (false, false) => CARD_COMPARE,
        (true, false) => WIDE,
        (false, true) => NARROW_EIGHT
            .get_or_init(|| swap_loop(CARD_COMPARE, NARROW_LOOP, NARROW_EIGHT_LOOP))
            .as_str(),
        (true, true) => WIDE_EIGHT
            .get_or_init(|| swap_loop(WIDE, WIDE_LOOP, WIDE_EIGHT_LOOP))
            .as_str(),
    }
}

/// `text` with its one `old` loop replaced by `new`. A loop text that is not
/// found exactly once is a programming error: the eight-load form would not
/// be the one-load program with one loop swapped.
fn swap_loop(text: &str, old: &str, new: &str) -> String {
    assert_eq!(
        text.matches(old).count(),
        1,
        "the card hash's one-load loop appears exactly once"
    );
    text.replacen(old, new, 1)
}

/// The one-load loop of [`CARD_COMPARE`], exactly as it stands there.
pub(crate) const NARROW_LOOP: &str = "    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
";

/// The narrow eight-load loop: eight `Load4`s at `at + 0, 16, ... 112`, then
/// the eight steps in that order, while all eight are below `hi` (`at` and
/// `hi` are multiples of 16, so `at + 112u < hi` is "the eighth is in"); then
/// the one-load loop for the 0 to 7 blocks left. The thread's blocks and
/// their order are the one-load loop's, so is the hash. `at + 112u` cannot
/// wrap, by arithmetic rather than by any cap on `size` (a window copy has
/// none, and `size as u32` truncates): `lo` is `piece * 65536u - skew`, at
/// most 2^32 - 65536; `size` is always a multiple of 64 KiB (a title buffer,
/// or a window on a grid of 64 KiB or more), and so is its truncation, so
/// `hi` is at most 2^32 - 65536 too. Every `at` the loop tests is below
/// `max(lo, hi) + 128`, and adding 112 stays below 2^32.
pub(crate) const NARROW_EIGHT_LOOP: &str = "    uint at = lo;
    // Eight loads issued before the steps that use them, so the thread has
    // eight in flight rather than one; the steps are in address order.
    [loop]
    for (; at + 112u < hi; at += 128u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 16u);
        uint4 v2 = now.Load4(at + 32u);
        uint4 v3 = now.Load4(at + 48u);
        uint4 v4 = now.Load4(at + 64u);
        uint4 v5 = now.Load4(at + 80u);
        uint4 v6 = now.Load4(at + 96u);
        uint4 v7 = now.Load4(at + 112u);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
";

/// The one-load loop of [`WIDE`], exactly as it stands there.
pub(crate) const WIDE_LOOP: &str = "    [loop]
    for (uint at = lo + lane * 16u; at < hi; at += 1024u) {
        h = Step(h, now.Load4(at));
    }
";

/// The wide eight-load loop (the CBJ fork's shape): eight `Load4`s at
/// `at + 0, 1024, ... 7168`, then eight `Step`s in that order, while all
/// eight are below `hi`; then the one-load loop for the 0 to 7 of the lane's
/// blocks left. The lane's blocks and their order are the one-load loop's,
/// so is its state and the fold after the barrier. `at + 7168u` cannot
/// wrap, by arithmetic rather than by any cap on `size` (a window copy has
/// none, and `size as u32` truncates): `lo` is `piece * 65536u - skew`, at
/// most 2^32 - 65536; `size` is always a multiple of 64 KiB (a title buffer,
/// or a window on a grid of 64 KiB or more), and so is its truncation, so
/// `hi` is at most 2^32 - 65536 too. A lane's `at` starts at most 1,008
/// past `lo`, and every `at` the loop tests is below `max(lo + 1008, hi) +
/// 8192`, so adding 7168 stays below 2^32.
pub(crate) const WIDE_EIGHT_LOOP: &str = "    uint at = lo + lane * 16u;
    // Eight loads issued before the steps that use them, so the thread has
    // eight in flight rather than one; the steps are in address order.
    [loop]
    for (; at + 7168u < hi; at += 8192u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 1024u);
        uint4 v2 = now.Load4(at + 2048u);
        uint4 v3 = now.Load4(at + 3072u);
        uint4 v4 = now.Load4(at + 4096u);
        uint4 v5 = now.Load4(at + 5120u);
        uint4 v6 = now.Load4(at + 6144u);
        uint4 v7 = now.Load4(at + 7168u);
        h = Step(h, v0);
        h = Step(h, v1);
        h = Step(h, v2);
        h = Step(h, v3);
        h = Step(h, v4);
        h = Step(h, v5);
        h = Step(h, v6);
        h = Step(h, v7);
    }
    [loop]
    for (; at < hi; at += 1024u) {
        h = Step(h, now.Load4(at));
    }
";

/// One dispatch's share of a run of `pieces` pieces, as `(piece offset,
/// groups)`: the wide form hashes one piece a group and is split at 65,535
/// groups, the most one dispatch dimension takes; the narrow form hashes 64
/// pieces a group and asks for one dispatch, as before.
pub(crate) fn dispatch_parts(pieces: u32) -> impl Iterator<Item = (u32, u32)> {
    dispatch_parts_as(wide(), pieces)
}

/// [`dispatch_parts`] for a given form, so tests can ask for either without
/// the process's choice.
pub(crate) fn dispatch_parts_as(wide: bool, pieces: u32) -> impl Iterator<Item = (u32, u32)> {
    let mut left = pieces;
    let mut at = 0u32;
    let mut narrow_done = false;
    std::iter::from_fn(move || {
        if wide {
            if left == 0 {
                return None;
            }
            let groups = left.min(65_535);
            let part = (at, groups);
            at += groups;
            left -= groups;
            return Some(part);
        }
        if narrow_done {
            return None;
        }
        narrow_done = true;
        Some((0, pieces.div_ceil(64)))
    })
}
