//! `VICTORY_EXACT_BACK=1`: the merge that compares 32 bytes at a time and
//! stores only the 4-byte words that differ gives what the merge that
//! stores the first-to-last stretch gives, byte for byte, on every end and
//! alignment, and stores no word that did not differ.

use crate::shadow::{Reference, exact_merge, merge_piece};

/// A deterministic pseudo-random source (xorshift): the cases are random
/// but a failure names the same bytes every run.
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

/// The card's merge (the first-to-last stretch, `Shadow::card_merge`'s
/// loop, nothing touched when nothing differs), as the reference for
/// `VICTORY_EXACT_BACK=1`.
fn stress_merged(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let Some(first) = now.iter().zip(held.iter()).position(|(g, o)| g != o) else {
        return false;
    };
    let last = now.iter().zip(held.iter()).rposition(|(g, o)| g != o).unwrap_or(first);
    let mut differs = 0u8;
    for i in first..=last {
        let wrote = if now[i] != held[i] { 0xffu8 } else { 0 };
        let merged = (title[i] & !wrote) | (now[i] & wrote);
        differs |= merged ^ title[i];
        title[i] = merged;
        held[i] = merged;
    }
    differs != 0
}

/// Every word wholly inside a stored run differed from the copy's bytes,
/// every differing byte that changed title memory is in a run, and no run
/// is out of the buffers.
fn check_stores(address: usize, old: &[u8], new: &[u8], before: &[u8], after: &[u8], stores: &[(usize, usize)]) {
    for &(off, len) in stores {
        assert!(off + len <= new.len(), "a store inside the buffers");
        if len.is_multiple_of(4) && (address + off).is_multiple_of(4) {
            for word in (off..off + len).step_by(4) {
                assert_ne!(&new[word..word + 4], &old[word..word + 4], "stored a word that did not differ at {word}");
            }
        } else {
            for i in off..off + len {
                assert_ne!(new[i], old[i], "stored a byte that did not differ at {i}");
            }
        }
    }
    for i in 0..new.len() {
        if before[i] != after[i] {
            assert!(
                stores.iter().any(|&(off, len)| i >= off && i < off + len),
                "the title byte at {i} changed and was not stored"
            );
        }
    }
}

/// Random old and new buffers, with changes at the first byte, the last
/// byte, across a 32-byte line of title memory, at an unaligned start and
/// in a final partial word: the exact merge gives the same bytes as today's
/// merge (title memory and the copy's reference), and stores no word that
/// did not differ.
#[test]
fn the_exact_merge_gives_the_same_bytes_and_stores_only_changed_words() {
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    let lengths = [1usize, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 66, 97, 128, 199, 256];
    for &len in &lengths {
        for alignment in 0..8usize {
            for seed in 0..6u8 {
                let address = 0x1_0000 + alignment;
                let old: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
                let mut new = old.clone();
                // The first byte, and the last (in a final partial word
                // whenever the piece does not end on a word boundary).
                new[0] = new[0].wrapping_add(0x21 + seed);
                new[len - 1] = new[len - 1].wrapping_add(0x43);
                // Across a 32-byte line of title memory, which depends on
                // where the piece starts.
                let line = (32 - (address & 31)) & 31;
                for at in [line.saturating_sub(1), line, line + 1] {
                    if at < len {
                        new[at] = new[at].wrapping_add(0x11);
                    }
                }
                // An unaligned start: changes at offsets the first word does
                // not end on.
                for at in [1usize, 3] {
                    if at < len {
                        new[at] ^= 0x22;
                    }
                }
                // And some random changes.
                for _ in 0..(len / 24 + 2) {
                    let at = rng.below(len);
                    new[at] ^= rng.byte() | 1;
                }
                // Title memory, with bytes the title wrote during the
                // flight: some where the card wrote too, some where it did
                // not (the card's byte wins in the first case, the title's
                // in the second, in both merges).
                let title: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
                let title0 = title.clone();
                let mut held1 = old.clone();
                let mut title1 = title.clone();
                let changed1 = merge_piece(&new, &mut held1, &mut title1);
                let mut held2 = old.clone();
                let mut title2 = title.clone();
                let mut stores = Vec::new();
                let changed2 = exact_merge(address, &new, &mut held2, &mut title2, Reference::Now, &mut stores);
                assert_eq!(changed1, changed2, "len {len} at {address:#x}: whether title memory changed");
                assert_eq!(title1, title2, "len {len} at {address:#x}: title memory");
                assert_eq!(held1, held2, "len {len} at {address:#x}: the copy's reference");
                check_stores(address, &old, &new, &title0, &title1, &stores);
            }
        }
    }
}

/// A byte the title wrote during the flight survives exactly as it does
/// today: the title's where the card did not write it, the card's where it
/// did (today's merge lets the card's byte stand there), and only what
/// changed is stored (here the card's byte alone: the stretch begins at it,
/// and the word it is in began before the stretch, where title memory is
/// read no further).
#[test]
fn a_title_byte_written_during_the_flight_survives_as_it_did() {
    let old = vec![0x10u8; 64];
    let mut new = old.clone();
    new[5] = 0x20; // the card wrote it
    let mut title = old.clone();
    title[5] = 0x77; // the title wrote it too: the card's byte stands
    title[6] = 0x99; // the title wrote here; the card did not
    let mut held1 = old.clone();
    let mut title1 = title.clone();
    assert!(merge_piece(&new, &mut held1, &mut title1));
    let mut held2 = old.clone();
    let mut title2 = title.clone();
    let mut stores = Vec::new();
    assert!(exact_merge(0x1_0000, &new, &mut held2, &mut title2, Reference::Now, &mut stores));
    assert_eq!(title1, title2);
    assert_eq!((title1[5], title1[6]), (0x20, 0x99));
    assert_eq!(stores, vec![(5, 1)], "only the byte the card wrote");
    assert_eq!(held1, held2);
}

/// The card's merge of a waited-for submission (the CPU's side takes the
/// title's byte where the card did not write, `Shadow::card_merge`): the
/// exact merge gives the same title memory, the same CPU side and the same
/// word-only stores.
#[test]
fn the_exact_card_merge_gives_the_same_bytes() {
    let mut rng = Rng::new(0xdead_beef_cafe_f00d);
    for len in [1usize, 3, 4, 5, 31, 32, 33, 64, 100] {
        for alignment in 0..8usize {
            let address = 0x2_0000 + alignment;
            let old: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            let mut new: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            // Some bytes are the CPU's side's (the card did not write them),
            // so the title's bytes must reach the CPU's side.
            for i in (0..len).step_by(3) {
                new[i] = old[i];
            }
            let title: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            let title0 = title.clone();
            let mut cpu1 = old.clone();
            let mut title1 = title.clone();
            let changed1 = stress_merged(&new, &mut cpu1, &mut title1);
            let mut cpu2 = old.clone();
            let mut title2 = title.clone();
            let mut stores = Vec::new();
            let changed2 = exact_merge(address, &new, &mut cpu2, &mut title2, Reference::Merged, &mut stores);
            assert_eq!(changed1, changed2);
            assert_eq!(title1, title2, "len {len} at {address:#x}: title memory");
            assert_eq!(cpu1, cpu2, "len {len} at {address:#x}: the CPU's side");
            check_stores(address, &old, &new, &title0, &title1, &stores);
        }
    }
}
