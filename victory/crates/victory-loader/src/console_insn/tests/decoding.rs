//! The decoder against generated encodings and against the opcode tables'
//! cells, whole.

use super::super::decode::{Field, MemoryOperand, Op, Refusal, Segment, decode};
use super::Values;
use super::encode::{insn, memory_forms, modrm, rex};

/// `bytes` decode to `op`, all of them, whatever follows.
#[track_caller]
fn decodes_whole(bytes: &[u8], op: Op) {
    let mut padded = [0; 15];
    padded[..bytes.len()].copy_from_slice(bytes);
    for code in [bytes, &padded[..]] {
        let decoded = decode(code).unwrap_or_else(|e| panic!("{bytes:02x?}: {e:?}"));
        assert_eq!((decoded.op, usize::from(decoded.len)), (op, bytes.len()), "{bytes:02x?}");
    }
}

/// [`decodes_whole`], and every shorter cut of `bytes` is `Truncated`: it
/// could still be one of ours.
#[track_caller]
fn decodes(bytes: &[u8], op: Op) {
    decodes_whole(bytes, op);
    for cut in 0..bytes.len() {
        assert_eq!(decode(&bytes[..cut]), Err(Refusal::Truncated), "{:02x?}", &bytes[..cut]);
    }
}

/// `bytes`, with zeros after them to 15, are not one of ours.
#[track_caller]
fn refused(bytes: &[u8]) {
    let mut padded = [0; 15];
    padded[..bytes.len()].copy_from_slice(bytes);
    assert_eq!(decode(&padded), Err(Refusal::NotOurs), "{bytes:02x?}");
}

/// The implicit operand [rAX] of CLZERO and MONITORX (APM3 pp. 156, 238).
fn at_rax(address32: bool, segment: Segment) -> MemoryOperand {
    MemoryOperand { base: Some(0), index: None, displacement: 0, rip_relative: false, address32, segment }
}

/// A REX for these registers where one is needed (or W is wanted).
fn rex_for(w: bool, r: u8, x: u8, b: u8) -> Option<u8> {
    (w || (r | x | b) >= 8).then(|| rex(w, r, x, b))
}

#[test]
fn every_memory_form_of_the_cache_hints() {
    // CLWB is 66 0F AE /6, CLFLUSHOPT 66 0F AE /7, each with a byte in
    // memory, in any addressing form (APM3 pp. 152, 154; Table A-7, Group
    // 15, row 66, note 6; SDM CLWB, CLFLUSHOPT). A 32-bit address size (67)
    // uses the same forms (1.2.3, 1.7.3); a segment override names its base
    // (1.2.4); REX.R does not extend an opcode digit (1.4.4, p. 23); REX.W
    // has no effect on a byte operand (p. 23).
    let segments = [
        (None, Segment::Flat),
        (Some(0x3e), Segment::Flat),
        (Some(0x64), Segment::Fs),
        (Some(0x65), Segment::Gs),
    ];
    let hints: [(u8, fn(MemoryOperand) -> Op); 2] = [(6, Op::Clwb), (7, Op::Clflushopt)];
    let mut n = 0;
    for (digit, hint) in hints {
        for mem in memory_forms() {
            let encoded = mem.encode(digit);
            for address32 in [false, true] {
                for (override_prefix, segment) in segments {
                    for (w, r) in [(false, 0), (true, 8)] {
                        let mut prefixes = vec![];
                        prefixes.extend(override_prefix);
                        prefixes.push(0x66);
                        if address32 {
                            prefixes.push(0x67);
                        }
                        let rex = rex_for(w, r, encoded.x, encoded.b);
                        let bytes = insn(&prefixes, rex, 0xae, &encoded.bytes, &[]);
                        let op = hint(mem.expected(address32, segment));
                        if n % 61 == 0 {
                            decodes(&bytes, op);
                        } else {
                            decodes_whole(&bytes, op);
                        }
                        n += 1;
                    }
                }
            }
        }
    }
}

/// Whether APM3's tables make an instruction of the 0F map, in the row
/// `row` with ModRM byte `m`, one of ours: Table A-8 row /7 (0F 01 with
/// mod 11, r/m 2 to 5, row none, where 66 is no row but an operand size of
/// no effect); Table A-7 Group 15 row 66, /6 and /7 with memory; Group 9 row
/// F3, /7 with a register; Table A-4 row 7, 66 0F 78 (Group 17, /0) and
/// 66 0F 79, F2 0F 78 and F2 0F 79, with registers.
fn in_the_tables(row: Option<u8>, opcode: u8, m: u8) -> bool {
    let (mode, reg, rm) = (m >> 6, m >> 3 & 7, m & 7);
    match (row, opcode) {
        (None | Some(0x66), 0x01) => mode == 3 && reg == 7 && (2..=5).contains(&rm),
        (Some(0x66), 0xae) => mode != 3 && (reg == 6 || reg == 7),
        (Some(0xf3), 0xc7) => mode == 3 && reg == 7,
        (Some(0x66), 0x78) => mode == 3 && reg == 0,
        (Some(0x66), 0x79) | (Some(0xf2), 0x78 | 0x79) => mode == 3,
        _ => false,
    }
}

#[test]
fn exactly_the_tables_cells_are_ours() {
    // Every opcode of the 0F map, every ModRM byte, every prefix row; room
    // after ModRM for a SIB (base 101), a displacement and immediates.
    for row in [None, Some(0x66), Some(0xf2), Some(0xf3)] {
        for opcode in 0..=255u8 {
            for m in 0..=255u8 {
                let bytes = insn(row.as_slice(), None, opcode, &[m, 0x25, 0, 0, 0, 0, 0, 0], &[]);
                let ours = decode(&bytes).is_ok();
                assert_eq!(ours, in_the_tables(row, opcode, m), "{bytes:02x?}");
            }
        }
    }
    // Outside the 0F map, nothing is: every first byte that is neither a
    // prefix, a REX nor 0F.
    let prefix_or_escape = |b: u8| matches!(b, 0x0f | 0x26 | 0x2e | 0x36 | 0x3e | 0x40..=0x4f | 0x64..=0x67 | 0xf0 | 0xf2 | 0xf3);
    for b in (0..=255u8).filter(|&b| !prefix_or_escape(b)) {
        refused(&[b, 0x0f, 0x01, modrm(3, 7, 4)]);
    }
}

#[test]
fn row_7_of_table_a8() {
    // 0F 01 with ModRM 11 111 r/m (APM3 Table A-8, p. 540): r/m 2 MONITORX,
    // 3 MWAITX, 4 CLZERO, 5 RDPRU. 67 makes [rAX] [EAX] (1.2.3), FS and GS
    // add their base (1.2.4), the null segments do nothing (1.2.4), 66 and
    // REX change nothing here (1.2.1, 1.2.7).
    use Segment::{Flat, Fs, Gs};
    let cases: [(&[u8], bool, Segment); 6] = [
        (&[], false, Flat),
        (&[0x67], true, Flat),
        (&[0x66], false, Flat),
        (&[0x64], false, Fs),
        (&[0x65, 0x67], true, Gs),
        (&[0x2e, 0x36, 0x26], false, Flat),
    ];
    for (prefixes, address32, segment) in cases {
        for rex in [None, Some(rex(true, 8, 8, 8))] {
            let op = |rm| insn(prefixes, rex, 0x01, &[modrm(3, 7, rm)], &[]);
            decodes(&op(2), Op::Monitorx(at_rax(address32, segment)));
            decodes(&op(3), Op::Mwaitx);
            decodes(&op(4), Op::Clzero(at_rax(address32, segment)));
            decodes(&op(5), Op::Rdpru);
        }
    }
}

#[test]
fn every_register_of_rdpid() {
    // RDPID is F3 0F C7 /7 with a register (APM3 p. 307; Table A-7, Group
    // 9, row F3; SDM RDPID: "ModRM.MOD = 011B required"); REX.B extends
    // r/m, REX.R not the /7 (1.4.4); 66 and REX.W do not apply (APM3 p. 307;
    // SDM RDPID), F3 keeping the row with 66 either side of it.
    for reg in 0..16 {
        for (w, r) in [(false, 0), (true, 8)] {
            for prefixes in [&[0xf3][..], &[0x66, 0xf3], &[0xf3, 0x66], &[0x3e, 0xf3]] {
                let bytes = insn(prefixes, rex_for(w, r, 0, reg), 0xc7, &[modrm(3, 7, reg)], &[]);
                decodes(&bytes, Op::Rdpid { reg });
            }
        }
    }
}

#[test]
fn every_register_pair_of_extrq_and_insertq() {
    // EXTRQ 66 0F 78 /0 ib ib (xmm in r/m: "/0", APM3 2.5.2) and 66 0F 79 /r;
    // INSERTQ F2 0F 78 /r ib ib and F2 0F 79 /r (APM4 pp. 173, 196): reg the
    // destination (REX.R), r/m the source (REX.B), mod 11 ("Uo", p. 525).
    let mut values = Values::new(0x5e4a);
    for dst in 0..16 {
        for src in 0..16 {
            let (length, index) = (values.next() as u8, values.next() as u8);
            let imm = Field::Immediate { length, index };
            let pair = rex_for(false, dst, 0, src);
            let ext = |row: &[u8], opcode, tail: &[u8], imm: &[u8]| insn(row, pair, opcode, tail, imm);
            decodes(
                &insn(&[0x66], rex_for(false, 0, 0, dst), 0x78, &[modrm(3, 0, dst)], &[length, index]),
                Op::Extrq { dst, field: imm },
            );
            // REX.R on the /0 digit extends nothing (1.4.4).
            decodes(
                &insn(&[0x66], Some(rex(false, 8, 0, dst)), 0x78, &[modrm(3, 0, dst)], &[length, index]),
                Op::Extrq { dst, field: imm },
            );
            decodes(&ext(&[0x66], 0x79, &[modrm(3, dst, src)], &[]), Op::Extrq { dst, field: Field::Register(src) });
            decodes(&ext(&[0xf2], 0x78, &[modrm(3, dst, src)], &[length, index]), Op::Insertq { dst, src, field: imm });
            let held = Op::Insertq { dst, src, field: Field::Register(src) };
            decodes(&ext(&[0xf2], 0x79, &[modrm(3, dst, src)], &[]), held);
            // F2 with 66 either side keeps the F2 row: INSERTQ, as the CPU
            // does (`emulation::mixed_prefixes_agree_with_the_cpu`); F3 with
            // 66 is the F3 row, blank (Table A-4).
            decodes(&ext(&[0xf2, 0x66], 0x79, &[modrm(3, dst, src)], &[]), held);
            decodes(&ext(&[0x66, 0xf2], 0x78, &[modrm(3, dst, src)], &[length, index]), Op::Insertq { dst, src, field: imm });
            refused(&ext(&[0x66, 0xf3], 0x79, &[modrm(3, dst, src)], &[]));
            refused(&ext(&[0xf3, 0x66], 0x78, &[modrm(3, dst, src)], &[length, index]));
        }
    }
}

#[test]
fn prefix_rules() {
    let clzero = |prefixes: &[u8], rex: Option<u8>| insn(prefixes, rex, 0x01, &[modrm(3, 7, 4)], &[]);
    let clwb = |prefixes: &[u8], rex: Option<u8>| insn(prefixes, rex, 0xae, &[modrm(0, 6, 0)], &[]);
    let rdpid = |prefixes: &[u8]| insn(prefixes, None, 0xc7, &[modrm(3, 7, 0)], &[]);
    let monitorx = |prefixes: &[u8]| insn(prefixes, None, 0x01, &[modrm(3, 7, 2)], &[]);
    // The legacy prefixes in any order (1.2.1): 66, 67 and 65 in all six.
    let three = [0x66, 0x67, 0x65];
    for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let prefixes = order.map(|i| three[i]);
        decodes(&clzero(&prefixes, None), Op::Clzero(at_rax(true, Segment::Gs)));
    }
    // LOCK anywhere among them is #UD on these (1.2.5).
    for at in 0..=2 {
        let mut prefixes = vec![0x3e, 0x67];
        prefixes.insert(at, 0xf0);
        refused(&clzero(&prefixes, None));
        let mut prefixes = vec![0x66, 0x3e];
        prefixes.insert(at, 0xf0);
        refused(&clwb(&prefixes, None));
    }
    // Two of one group: undefined (1.2.1). The CS, DS, ES and SS overrides
    // are no group's in 64-bit mode (1.2.4): any number of them.
    refused(&clwb(&[0x66, 0x66], None));
    refused(&clzero(&[0x67, 0x67], None));
    refused(&monitorx(&[0x64, 0x65]));
    refused(&monitorx(&[0x65, 0x65]));
    refused(&rdpid(&[0xf3, 0xf3]));
    refused(&rdpid(&[0xf2, 0xf3]));
    decodes(&clzero(&[0x2e, 0x36, 0x3e, 0x26, 0x3e, 0x2e], None), Op::Clzero(at_rax(false, Segment::Flat)));
    // One REX, right before 0F (1.2.7).
    refused(&[vec![rex(true, 0, 0, 0)], clzero(&[0x3e], None)].concat());
    refused(&[vec![rex(false, 0, 0, 8)], clzero(&[], Some(rex(false, 0, 0, 8)))].concat());
    refused(&[vec![rex(false, 0, 0, 8)], clwb(&[0x66], None)].concat());
    // 15 bytes at most (1.1.2): null prefixes up to 15 fit, one more does not.
    let fill = |count: usize| clzero(&vec![0x3e; count], None);
    decodes(&fill(12), Op::Clzero(at_rax(false, Segment::Flat)));
    assert_eq!(decode(&fill(13)), Err(Refusal::NotOurs));
    assert_eq!(decode(&fill(13)[..15]), Err(Refusal::NotOurs));
    // VEX (C5, C4), XOP (8F) and EVEX (62) escapes: none of these uses them
    // (1.2.8).
    for escape in [0xc5, 0xc4, 0x8f, 0x62] {
        refused(&[escape, 0xf9, 0x79, modrm(3, 0, 1)]);
    }
}
