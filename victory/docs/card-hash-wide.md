# The card hash with many threads to a piece

Status: built (branch `job/hash-wide-fix`, 2026-11-26). The wide hash is
behind `VICTORY_HASH_WIDE=1`, off by default; the run-time check behind
`VICTORY_HASH_CHECK=1`, off by default. Either program's eight-load form
is behind `VICTORY_HASH_EIGHT=1`, off by default (branch
`job/compare-eight`): the same hash values, with eight loads a loop turn
(see `host::hash::pick`).

## What runs

A copy in the card's memory has a 16-byte hash a 64 KiB piece
(`crate::shadow::card_hash_bytes`). The same compute program serves both
passes: the hash pass after ranges are sent up (recorded in
`Host::card_before`, before the submission's lists) stores each piece's
hash, and the compare pass after the submission's lists hashes the declared
pieces again and writes a dword a piece to `differs`, `1` where the hash
changed. Both passes run the program `host::hash::program()` returns. The
copy-back then fetches the flagged pieces only, so a changed piece the
compare does not flag is a GPU write that never reaches title memory: hash
quality is correctness here.

The narrow program, `CARD_COMPARE` (the default), gives one thread a piece:
from four constants, one step a 16-byte block,
`h = (h ^ v) * M; h = h.yzwx ^ (h >> 15)`, 4,096 steps for a full piece.

With `VICTORY_HASH_WIDE=1` both passes run `WIDE`: one group of 64 threads
a piece. Thread `t` (`SV_GroupIndex`) starts from the narrow form's four
constants plus `t` and takes the piece's blocks `t`, `t+64`, `t+128`, ...
(a stride of 1,024 bytes) through `Step`:

    h = (h ^ v) * (0x01000193, 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D)
    h = h.yzwx ^ (h >> 15)
    h = h * (0x7FEB352D, 0x846CA68B, 0x297A2D39, 0x5BD1E995)

that is, the narrow step and then a second odd multiply of every word. The
lane's final state goes through `FinaliseLane`: the lane number is XORed
and added into the four words, then three rounds, each with three
xor-shifts and two odd multiplies a word and a rotation of the words. The
rotation only moves words, so the finalisers mix each word within itself
and never one word into another; the mixing across words is the chain's
and the fold's. The 64 finalised states go to `groupshared uint4 parts[64]`;
after `GroupMemoryBarrierWithGroupSync` thread 0 starts from
`f = Seed(len)`, takes `f = Step(f, parts[i])` for `i = 0..63`, and ends
with `Finalise` (the same shape of rounds, other multipliers). `len` is the
piece's covered byte length (`hi - lo`). The store stays 16 bytes a piece,
and `differs` stays `0`/`1`.

The range is what the narrow program computes: the piece clamped to the
copy (`lo`/`hi`), rounded out to 16 bytes, `skew` in piece 0; a ragged tail
is read to its rounded end. A copy of many pieces is one group a piece. A
dispatch holds at most 65,535 groups, so `host::hash::dispatch_parts`
splits a longer run, and each part passes the next `first` and the pieces
left (`piece = first + group`); the compare pass's `out_at` moves with the
part.

The CPU model (`host/tests/hash_model.rs`) is the same function operation
for operation (`step`, `wide_step`, `finalise_lane`, `finalise`, `seed`,
the fold), and `host/tests/wide.rs` holds the card to it on the software
adapter for every shape the copy path cuts.

The idea of splitting a piece among 64 lanes and folding their states comes
from [victory-fork](https://github.com/fxceless/victory-fork) pull request
#1014 (commit `555722468`; its `CARD_COMPARE`). Our
program is written here: its step, lane seeds, finalisers and fold are our
hash's, and its result and result layout are ours (128 bits kept, `0`/`1`
compared).

## One function a process

The wide form is another function of a piece's bytes (the model shows it at
every size tried, down to one block). That is safe because within one
process every stored hash is written by the same program and only ever
compared for equality with a hash that program wrote. The choice is a
`OnceLock` read the first time the card's work is made (`Host::card_ready`
-> `make_card_work` -> `hash::program`), the first copy that could go to the
card's memory; both passes and flights already queued share it. Flipping it
mid-run would make every piece compared afterwards look changed until its
hash was written again; a live switch would need the hash buffer kept per
mode. With the switch off, the narrow program and its dispatch are the same
as before, byte for byte.

## What is proved, and what is only sampled

Proved, for every piece's contents:

- Every part of the wide form is a bijection of the state for fixed input.
  `Step`: the XOR, two odd multiplies (invertible mod 2^32) and the
  rotate-and-shift `y[i] = x[i+1] ^ (x[i] >> 15)`, which is the rotation
  times one plus a nilpotent part (a third shift by 15 leaves nothing), with
  inverse `x[i] = y[i-1] ^ (y[i-2] >> 15) ^ (y[i-3] >> 30)`; the model test
  undoes the step for 100,000 states. Each finaliser round (xor-shifts, odd
  multiplies, a rotation) is one, and so is a fold step in `f` for a fixed
  part and in the part for a fixed `f`. So a change confined to one block
  always changes the hash, and so does any change confined to one lane that
  leaves that lane's final state changed.
- No single-bit change of the state passes `Step` independently of the
  state: for each of the 128 bits, two states leave different changes. The
  narrow step passes the four top bits whatever the state (below).
- A flip of any one of bits 12 to 31 of any word of a block leaves at least
  three bits of the lane's state changed after that block, whatever the
  state, so no one- or two-bit flip of the same lane's next block (64
  blocks on) puts it back; for bit 31 at least 9, 9, 8 and 8 bits (words 0
  to 3). The first multiply turns the flip of bit j into a difference
  `E = z ^ (z ± 2^j * M)` that depends only on `z`'s bits from j up; all of
  them are enumerated (16,777,200), and for none can both of `E` (word k-1)
  and `E >> 15` (word k) leave one bit through their second multiplies.
- Length and bounds are moot by design: `lo`, `hi`, `skew` and `size` are
  fixed per piece, `len` seeds the fold, and bytes outside `lo..hi` are not
  read.

Sampled only (no collision found in any of it):

- Flips of bits 0 to 11 against compensating flips: for every single-bit
  flip in one block, every single- and two-bit flip in the same lane's next
  block, stepped, over nine pairs of blocks (zero, the model's pattern and a
  random piece; a lane's first, middle and last pair): 9,510,912 pairs. The
  same question asked of the difference itself (a flip cancels exactly when
  it is the difference) over every consecutive lane pair of five more random
  pieces: 21,304,442,880 pairs.
- Same-lane pairs 64, 128 and 4,032 blocks apart: 40,000 (block, bit) cases
  a distance, each standing for all 8,256 one- and two-bit flips.
- The review's exact pattern (bit 31 of word k in block t, bits 31 of word
  k-1 and 16 of word k in block t+64) at every word and every pair 64 apart,
  over three pieces: 48,384 whole hashes, all changed (and on the card).
- Two lanes against each other through the fold: the same bit in the last
  blocks of every lane pair (516,096 cases), one lane's last and another's
  second-to-last (1,032,192), two consecutive blocks of one lane (16,384),
  250,000 random correlated edits, swaps and copies.
- Not covered: a change carried along several blocks of a chain before it
  cancels, and collisions in general. This is a non-cryptographic 128-bit
  hash; a collision can hide a changed piece in either form.

Known in the narrow form, and unchanged: its step alone is linear for a
top-bit change (`(a ^ 2^31) * M == (a * M) ^ 2^31`, and the rotate-and-shift
is linear), so bit 31 of word k in block t, then bit 31 of word k-1 and
bit 16 of word k in block t+1, keep the narrow hash for any contents; bit 30
of any word, and bits 29 and 28 of some, do so (with their own pairs) for
about half the states. The model and the card show it
(`the_reviews_top_bit_pattern_changes_the_wide_hash`,
`the_reviews_top_bit_pattern_changes_the_wide_gpu_hash`); the sweep above
finds 78 such pairs among its nine narrow block pairs. The narrow program is
the default and is kept byte for byte; fixing it would change the default's
function and is another job.

## Looking for a missed change in a run

`VICTORY_HASH_CHECK=1` (read once at start, off by default;
`host/hash_check.rs`). About one flight in 16 with declared writes to copies
in the card's memory is sampled, picked by splitmix64 of a counter, so the
sample never falls in step with a frame's count of flights. Its copy-back
lands exactly as with the switch off. After it, the pieces of its ranges the
copy-back did not fetch (by the copy-back's own rule: flagged under no range
of the plan) are fetched by a list, fence and event of the check's own into
8 MiB of PC memory of its own, compared with the CPU side's bytes, and never
merged. Over 8 MiB a landing, pieces are taken by a prime stride from a
generated start. The live stats file has `hash_checked_pieces_total`,
`hash_missed_pieces_total` (compared and different) and `hash_check_skipped_total` (not
compared: over the budget; no memory, list or fence for the check, which
skips and never ends the process; its memory in use by the other thread; on
the relay, the copy queue engaged; a failed fetch); a line is logged every
600 presented frames. With the switch off nothing is sampled, made or
fetched, and the stats file has no such fields.

A count's meaning: with the relay on (the default) the check's fetch runs
behind whatever was sent since the flight, so it can carry a later flight's
writes and count them as misses. A non-zero `hash_missed_pieces_total` there is a
lead, not a collision shown. A clean proof needs a run with the relay off
(`VICTORY_RELAY=0`, `VICTORY_LATE_WAIT` unset), where each submission is
waited for alone. In either mode the title's own CPU writes to those bytes
since the last send count too.

## Tests

`host/tests/hash_model.rs`, without a device: the narrow chain's pinned
value; the covered range's clamping and rounding; the wide form another
function at every size; position and the lane index; length, range and
skew; the review's cancelling pair and swap of the earlier fold; the lane
pair sweeps, the correlated edits; and for the step: its inverse, the
data-dependence of every single-bit change, the enumerated bound for bits
12 to 31, the review's top-bit pattern, the exhaustive and by-difference
compensation sweeps and the same-lane distances above.

`host/tests/wide.rs`, on the software adapter: the card equals the model for
every shape, both forms; every byte matters; swaps matter; bytes outside the
range do not; the compare flags exactly the changed pieces in mode 1 and 2,
at an offset and across the split; the review's patterns on the card; the
split itself; `the_forms_times_on_a_card` is ignored off a card.

`host/tests/fetch.rs`: two ranges of a flown plan in one piece are fetched
under both whichever compare flagged it, and a plan sampled for the check
fetches exactly the same. `host/tests/hash_check.rs`: the sample, the
budget, the fields, and the look on the software adapter (compares only
what was not fetched, counts a difference, merges nothing, skips when it has
no memory, halves a 16 MiB flight to the budget).

A test that asks for a device fails, not skips, when `VICTORY_D3D12_WARP=1`
is set and no device comes up.

## What a card should show

Victory-fork pull request #1014 reports its wide form's hash lists falling
from 22.3/29.4 ms to 5.5/8.5 ms and its compare lists from 11.9/13.6 ms to 2.3/3.3 ms (RTX 3060,
commit `555722468`), about 35 to about 10 ms of card time. Our form does
more a step (a whole 128-bit state, two multiplies, a 64-step fold and
finalisers) but its chain is 64 times shorter, so a card should show the
hash and compare list spans (`VICTORY_GPU_STAMPS=1`) several times smaller.
It saves nothing where the card hash is not on the frame's path (no copies
in the card's memory), where the passes are already small against the
title's lists, or where a one-group piece's dispatch overhead costs more
than the serial chain it replaces; the software adapter's times say nothing
about a card.
