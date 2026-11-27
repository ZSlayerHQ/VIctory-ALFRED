//! The pipeline's linked pair: a mesh program whose outputs pass D3D12's 32 KiB, translated
//! again with the pixel program of its pipeline ([`super::linked`]). The console containers
//! are synthetic, built here as `victory-shader`'s `link_tests` builds them.

use super::super::shader;
use super::*;

const TRIANGLE: u32 = 1 << 10 | 2 << 20;

fn mesh_code() -> Vec<u32> {
    vec![
        0xD765_0001, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001_00C1,
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E12_02FF, // v_mov_b32 v9, prim
        TRIANGLE,
        0xF800_020F, // exp param0 v7, v7, v7, v7 (TEXCOORD1: `stage_data`)
        0x0707_0707,
        0xF800_021F, // exp param1 v7, v7, v7, v7 (TEXCOORD2)
        0x0707_0707,
        0xF800_0941, // exp prim v9, off, off, off done
        0x0000_0009,
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ]
}

/// `v_interp_mov_f32 v2, p0, attr<attr>.z` and `exp mrt0, v2, v2, v2, v2 done vm`.
fn pixel_code_reading(attr: u32) -> Vec<u32> {
    vec![0xC80A_0202 | attr << 10, 0xF800_180F, 2 * 0x0101_0101, 0xBF81_0000]
}

/// One signature part's rows: (semantic, index, system value, register, components,
/// components read) and whether it is an input signature.
fn signature(rows: &[(&str, u32, u32, u32, u8, u8)], input: bool) -> Vec<u8> {
    let mut out = (rows.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&8u32.to_le_bytes());
    let mut names = Vec::new();
    let mut offsets = Vec::new();
    let mut next = 8 + 32 * rows.len() as u32;
    for (name, ..) in rows {
        offsets.push(next);
        next += name.len() as u32 + 1;
        names.extend(name.bytes());
        names.push(0);
    }
    for (n, &(_, index, sv, row, cols, used)) in rows.iter().enumerate() {
        let mask = ((1u16 << cols) - 1) as u8;
        let rw = if input { used } else { !used & 0xf };
        for v in [0, offsets[n], index, sv, 3, row] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[mask, rw, 0, 0]);
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    out.extend_from_slice(&names);
    out
}

/// The `DXIL` part of a container around one native program: its program header, the `XDAT`
/// index and the `XBCH`-packed `FSSM` blob (`victory-shader`'s `native.rs` lays them out).
fn native_part(stage: victory_shader::HwStage, code: &[u32], data: &[u8]) -> Vec<u8> {
    use victory_shader::HwStage;
    const SLOTS: [(HwStage, usize); 8] = [
        (HwStage::Ls, 0x08),
        (HwStage::Hs, 0x10),
        (HwStage::Es, 0x18),
        (HwStage::Gs, 0x20),
        (HwStage::Vs, 0x28),
        (HwStage::Ps, 0x30),
        (HwStage::Ms, 0x48),
        (HwStage::Cs, 0x60),
    ];
    let mut payload = 6u64.to_le_bytes().to_vec();
    payload.resize(0x68, 0);
    for (_, at) in SLOTS {
        payload[at..at + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEF_u64.to_le_bytes());
    }
    let at = SLOTS.iter().find(|s| s.0 == stage).unwrap().1;
    let record = payload.len() as u64;
    payload[at..at + 8].copy_from_slice(&record.to_le_bytes());
    let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let code_at = record + 0x54 + data.len() as u64;
    payload.extend_from_slice(&code_at.to_le_bytes());
    payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
    payload.extend_from_slice(&[0; 4]);
    payload.extend_from_slice(&(1u32 << 5).to_le_bytes());
    payload.resize(record as usize + 0x50, 0);
    payload.extend_from_slice(&(31u32 | 31 << 20).to_le_bytes());
    payload.extend_from_slice(data);
    payload.extend_from_slice(&code);
    let mut blob = b"FSSM".to_vec();
    blob.extend_from_slice(&0x101u32.to_le_bytes());
    blob.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    blob.extend_from_slice(&payload);
    let mut buf = vec![0u8; zlib_rs::compress_bound(blob.len())];
    let (stream, rc) = zlib_rs::compress_slice(&mut buf, &blob, zlib_rs::DeflateConfig::new(6));
    assert_eq!(rc, zlib_rs::ReturnCode::Ok);
    let mut packed = b"XBCH".to_vec();
    packed.extend_from_slice(&3u32.to_le_bytes());
    packed.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    packed.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    packed.extend_from_slice(stream);
    let mut part = vec![0u8; 0x18];
    part.extend_from_slice(b"XDAT");
    part.extend_from_slice(&1u32.to_le_bytes());
    part.extend_from_slice(&0x24u32.to_le_bytes());
    let first = part.len() + 12;
    for v in [0x0du32, first as u32, packed.len() as u32] {
        part.extend_from_slice(&v.to_le_bytes());
    }
    part.extend_from_slice(&packed);
    part
}

/// Stage data (`native.rs` reads the I/O out of it): no export patches, and either pixel
/// inputs or the mesh's thread and vertex maxima.
fn stage_data(ps_inputs: &[(u8, u8)], mesh: Option<(u16, u16, u16)>) -> Vec<u8> {
    let mut data = vec![0u8; 0x1000];
    let patches = 0x14c;
    for mrt in 0..8 {
        let at = patches + mrt * (4 + 10 * 10 * 4);
        data[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    }
    data[0x124..0x128].copy_from_slice(&(ps_inputs.len() as u32).to_le_bytes());
    for (n, &(row, attr)) in ps_inputs.iter().enumerate() {
        let at = 0x24 + 8 * n;
        // usage 6 (generic), row, mask 0xf, attribute, second row, flat
        data[at] = 6;
        data[at + 1] = row;
        data[at + 2] = 0xf;
        data[at + 3] = attr;
        data[at + 5] = row;
        data[at + 7] = 1;
    }
    if let Some((threads, vertices, primitives)) = mesh {
        let mut info = [0u8; 8];
        info[0..2].copy_from_slice(&threads.to_le_bytes());
        info[2..4].copy_from_slice(&vertices.to_le_bytes());
        info[4..6].copy_from_slice(&primitives.to_le_bytes());
        info[7] = 2; // triangles
        data[0x448..0x450].copy_from_slice(&info);
        // The parameter exports: param0 is row 2 (TEXCOORD1), param1 row 3 (TEXCOORD2), the
        // two the fixtures' pixel programs read.
        data[0x188..0x18c].copy_from_slice(&2u32.to_le_bytes());
        for (n, row) in [2u8, 3].into_iter().enumerate() {
            // usage 6 (generic), row, parameter
            data[0xc8 + 6 * n..][..3].copy_from_slice(&[6, row, n as u8]);
        }
    }
    data
}

/// A DXBC container from parts (the header, then each part's offset, fourcc and data).
fn container(parts: &[(victory_shader::FourCC, &[u8])]) -> Vec<u8> {
    let mut out = b"DXBC".to_vec();
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    let total = 32 + 4 * parts.len() + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut at = 32 + 4 * parts.len();
    for (_, data) in parts {
        out.extend_from_slice(&(at as u32).to_le_bytes());
        at += 8 + data.len();
    }
    for (fourcc, data) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// A mesh container: `outputs` float4 TEXCOORDs after the position, 128 lanes and vertices.
fn mesh_container(outputs: u32, vertices: u16) -> Vec<u8> {
    use victory_shader::FourCC;
    let mut rows = vec![("SV_Position", 0, 1, 0, 4, 15)];
    for i in 0..outputs {
        rows.push(("TEXCOORD", i, 0, 1 + i, 4, 0));
    }
    let osg1 = signature(&rows, false);
    let data = stage_data(&[], Some((128, vertices, 32)));
    let part = native_part(victory_shader::HwStage::Ms, &mesh_code(), &data);
    container(&[(FourCC::OSG1, &osg1), (FourCC::DXIL, &part)])
}

/// A pixel container declaring `inputs` float4 TEXCOORDs (row n for TEXCOORD n) and reading
/// TEXCOORD2's z.
fn pixel_container(inputs: u32) -> Vec<u8> {
    pixel_container_reading(inputs, 2)
}

/// A pixel container declaring `inputs` float4 TEXCOORDs (row n for TEXCOORD n) and reading
/// TEXCOORD`attr`'s z.
fn pixel_container_reading(inputs: u32, attr: u32) -> Vec<u8> {
    use victory_shader::FourCC;
    let mut rows = Vec::new();
    for i in 0..inputs {
        rows.push(("TEXCOORD", i, 0, i, 4, 0));
    }
    let isg1 = signature(&rows, true);
    let osg1 = signature(&[("SV_Target", 0, 64, 0, 4, 15)], false);
    let data = stage_data(&[(0, 0), (1, 1), (2, 2)], None);
    let part = native_part(victory_shader::HwStage::Ps, &pixel_code_reading(attr), &data);
    container(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1), (FourCC::DXIL, &part)])
}

/// A mesh program whose outputs pass 32 KiB as translated alone is made from the pair linked
/// with its pixel program, once per pair; a mesh that fits keeps its own translation.
#[test]
fn a_mesh_over_the_limit_is_made_from_the_linked_pair() {
    let ms = shader("mesh", &mesh_container(17, 128));
    let ps = shader("pixel", &pixel_container(17));
    let over = &ms.translated().expect("the mesh translates").host.dxil;
    assert!(
        !victory_shader::emit::mesh_fits(over).unwrap(),
        "the mesh fits 32 KiB as translated alone"
    );
    let pair = linked(&ms, &ps).expect("the pair is linked");
    assert!(victory_shader::emit::mesh_fits(&pair.0.dxil).unwrap());
    let again = linked(&ms, &ps).expect("the pair is kept");
    assert!(Arc::ptr_eq(&pair, &again), "the pair was translated again");
    // 64 components at 128 vertices are exactly 32,768 bytes: no pair is made.
    let fits = shader("mesh", &mesh_container(15, 128));
    assert!(linked(&fits, &ps).is_none());
}

/// The arbitrary semantics of an emitted program's output signature, by index.
fn texcoords_written(dxil: &[u8]) -> Vec<u32> {
    let c = victory_shader::Container::parse(dxil).unwrap();
    let part = c.part(victory_shader::FourCC::OSG1).expect("an output signature").data;
    let u32_at = |at: usize| u32::from_le_bytes(part[at..at + 4].try_into().unwrap());
    (0..u32_at(0) as usize)
        .map(|i| u32_at(4) as usize + 32 * i)
        .filter(|&at| u32_at(at + 12) == 0)
        .map(|at| u32_at(at + 8))
        .collect()
}

/// A pipeline maker's refusal, as the card gives one.
fn refusal() -> windows::core::Error {
    windows::core::Error::from(windows::Win32::Foundation::E_INVALIDARG)
}

/// A maker that writes down what it is asked for in `asked` and refuses the linked pair
/// with the next of `refusals` (none left: it makes it).
fn maker<'a>(
    asked: &'a std::cell::RefCell<Vec<&'static str>>,
    refusals: &'a std::cell::RefCell<Vec<windows::core::HRESULT>>,
) -> impl Fn(Option<&LinkedPair>) -> windows::core::Result<&'static str> + Copy + 'a {
    move |pair| {
        asked.borrow_mut().push(if pair.is_some() { "linked" } else { "shared" });
        match pair {
            Some(_) if !refusals.borrow().is_empty() => {
                Err(windows::core::Error::from(refusals.borrow_mut().remove(0)))
            }
            Some(_) => Ok("the linked pair"),
            None => Ok("the shared pair"),
        }
    }
}

/// Finding 1 of the review of the linked pair: the card refuses the pipeline of a linked
/// pair. The pipeline is then made of the shared translations (the draw is not dropped), and
/// the next pipeline of that pair does not try the linked pair again.
#[test]
fn a_linked_pipeline_the_card_refuses_is_made_of_the_shared_pair() {
    // A pair of this test's own: what is remembered of it must not reach another test's.
    let ms = shader("mesh", &mesh_container(18, 128));
    let ps = shader("pixel", &pixel_container(18));
    assert!(linked(&ms, &ps).is_some(), "the pair is linked");
    let asked = std::cell::RefCell::new(Vec::new());
    let make = |pair: Option<&LinkedPair>| {
        asked.borrow_mut().push(if pair.is_some() { "linked" } else { "shared" });
        match pair {
            Some(_) => Err(refusal()),
            None => Ok("a pipeline of the shared pair"),
        }
    };
    let made = mesh_pair_pipeline(&ms, Some(&ps), "a state", make);
    assert_eq!(
        made.ok(),
        Some("a pipeline of the shared pair"),
        "the linked pipeline was refused and none was made of the shared pair: the draw is dropped"
    );
    assert_eq!(*asked.borrow(), ["linked", "shared"]);
    asked.borrow_mut().clear();
    let again = mesh_pair_pipeline(&ms, Some(&ps), "a state", make);
    assert_eq!(again.ok(), Some("a pipeline of the shared pair"));
    assert_eq!(
        *asked.borrow(),
        ["shared"],
        "the linked pair was tried again for a pair whose linked pipeline the card refused"
    );
    // The second review: what the card refused is that pipeline, not the pair.
    assert!(
        linked(&ms, &ps).is_some(),
        "the pair is not linked any more, for any state"
    );
}

/// The second review of the linked pair, its required change 2: a refusal is of the linked
/// pipeline that was asked for, the pair in one state (a sample count the card does not
/// take, say). The same two programs in another state are still asked for linked, and each
/// state remembers its own refusal.
#[test]
fn a_refusal_is_remembered_for_its_state_alone() {
    use windows::Win32::Foundation::E_INVALIDARG;
    let ms = shader("mesh", &mesh_container(21, 128));
    let ps = shader("pixel", &pixel_container(21));
    let asked = std::cell::RefCell::new(Vec::new());
    let refusals = std::cell::RefCell::new(vec![E_INVALIDARG]);
    let make = maker(&asked, &refusals);
    assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), "state A", make).ok(), Some("the shared pair"));
    assert_eq!(
        mesh_pair_pipeline(&ms, Some(&ps), "state B", make).ok(),
        Some("the linked pair"),
        "the refusal in state A kept state B from its linked pair"
    );
    assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), "state A", make).ok(), Some("the shared pair"));
    assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), "state B", make).ok(), Some("the linked pair"));
    assert_eq!(*asked.borrow(), ["linked", "shared", "linked", "shared", "linked"]);
}

/// Only a refusal that would come again is remembered. A linked pipeline that failed for
/// want of memory, or because the device was lost, is made of the shared pair that once
/// (the draw is kept), and the next pipeline of that pair and state asks for the linked
/// pair again.
#[test]
fn a_passing_failure_is_not_remembered() {
    use windows::Win32::Foundation::E_OUTOFMEMORY;
    use windows::Win32::Graphics::Dxgi::{
        DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
    };
    let ms = shader("mesh", &mesh_container(22, 128));
    let ps = shader("pixel", &pixel_container(22));
    let passing = [E_OUTOFMEMORY, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET];
    let asked = std::cell::RefCell::new(Vec::new());
    let refusals = std::cell::RefCell::new(passing.to_vec());
    let make = maker(&asked, &refusals);
    for failure in passing {
        asked.borrow_mut().clear();
        assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), "a state", make).ok(), Some("the shared pair"));
        assert_eq!(
            *asked.borrow(),
            ["linked", "shared"],
            "the linked pair was not asked for again (its next failure: {failure:?})"
        );
    }
    assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), "a state", make).ok(), Some("the linked pair"));
}

/// The third review of the linked pair, its required change 1: only a result that says the
/// host refuses this pipeline's description is remembered. A generic failure (`E_FAIL`), a
/// driver's internal error (which is a lost device) and a failed interface cast say nothing
/// of the description: the pipeline is made of the shared pair that once, and the next
/// pipeline of the same pair and state is asked for linked again, and is made linked when
/// the host takes it.
#[test]
fn a_failure_that_is_no_refusal_leaves_the_linked_pair_eligible() {
    use windows::Win32::Foundation::{E_FAIL, E_NOINTERFACE};
    use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DRIVER_INTERNAL_ERROR;
    let ms = shader("mesh", &mesh_container(23, 128));
    let ps = shader("pixel", &pixel_container(23));
    for failure in [E_FAIL, DXGI_ERROR_DRIVER_INTERNAL_ERROR, E_NOINTERFACE] {
        let state = format!("a state of {failure:?}");
        let asked = std::cell::RefCell::new(Vec::new());
        let refusals = std::cell::RefCell::new(vec![failure]);
        let make = maker(&asked, &refusals);
        assert_eq!(mesh_pair_pipeline(&ms, Some(&ps), &state, make).ok(), Some("the shared pair"));
        assert_eq!(*asked.borrow(), ["linked", "shared"]);
        asked.borrow_mut().clear();
        assert_eq!(
            mesh_pair_pipeline(&ms, Some(&ps), &state, make).ok(),
            Some("the linked pair"),
            "{failure:?} was remembered as the host's refusal of the linked pipeline"
        );
        assert_eq!(*asked.borrow(), ["linked"]);
    }
}

/// Two workers can both find no entry for a pair and both translate it. What the first puts
/// in the table stays: the second gets the first one's pair back, not its own, and a pair
/// the first found not to link is not made linked by the second.
#[test]
fn a_late_publisher_keeps_the_first_publication() {
    let ms = shader("mesh", &mesh_container(17, 128));
    let ps = shader("pixel", &pixel_container(17));
    let pair = linked(&ms, &ps).expect("the pair is linked");
    let copy = || Some(Arc::new((pair.0.clone(), pair.1.clone())));
    // Keys of this test's own: no container has these digests.
    let key = ([0xA1; 32], [0xB2; 32]);
    let first = publish(key, copy()).expect("the first publication");
    let late = publish(key, copy()).expect("the pair has an entry");
    assert!(Arc::ptr_eq(&first, &late), "the late publisher replaced the first publication");
    let key = ([0xA1; 32], [0xB3; 32]);
    assert!(publish(key, None).is_none());
    assert!(publish(key, copy()).is_none(), "the late publisher linked a pair found not to link");
}

/// A mesh program that fits 32 KiB, and one drawn without a pixel program, are made of the
/// shared translations and nothing else is asked for; a pair that links is made of the linked
/// pair alone when the card takes it.
#[test]
fn a_pipeline_is_asked_for_once() {
    let asked = std::cell::RefCell::new(Vec::new());
    let make = |pair: Option<&LinkedPair>| {
        asked.borrow_mut().push(if pair.is_some() { "linked" } else { "shared" });
        Ok::<(), windows::core::Error>(())
    };
    let ps = shader("pixel", &pixel_container(19));
    let fits = shader("mesh", &mesh_container(15, 128));
    mesh_pair_pipeline(&fits, Some(&ps), "a state", make).unwrap();
    let over = shader("mesh", &mesh_container(19, 128));
    mesh_pair_pipeline(&over, None, "a state", make).unwrap();
    assert_eq!(*asked.borrow(), ["shared", "shared"]);
    asked.borrow_mut().clear();
    mesh_pair_pipeline(&over, Some(&ps), "a state", make).unwrap();
    assert_eq!(*asked.borrow(), ["linked"]);
}

/// Finding 3: the id the logs print is the first 64 bits of a container's digest. Two pixel
/// programs whose ids agree there, drawn with the same mesh program, each get the pair made
/// for them: the second is not handed the first one's, whose mesh program writes another
/// attribute than it reads.
#[test]
fn pixel_programs_whose_ids_agree_in_64_bits_get_their_own_pairs() {
    let ms = shader("mesh", &mesh_container(17, 128));
    let agreeing = |container: Vec<u8>| {
        Arc::new(Shader {
            stage: "pixel",
            hash: "0123456789abcdef".to_owned(),
            container,
            translated: std::sync::OnceLock::new(),
        })
    };
    let reads_2 = agreeing(pixel_container_reading(17, 2));
    let reads_1 = agreeing(pixel_container_reading(17, 1));
    let first = linked(&ms, &reads_2).expect("the first pair is linked");
    let second = linked(&ms, &reads_1).expect("the second pair is linked");
    assert_eq!(texcoords_written(&first.0.dxil), [2]);
    assert_eq!(
        texcoords_written(&second.0.dxil),
        [1],
        "the second pixel program was handed the first one's pair"
    );
    assert!(!Arc::ptr_eq(&first, &second));
}

/// `VICTORY_MESH_LINK=0` and nothing else turns linking off (unset, it is on).
#[test]
fn only_0_turns_linking_off() {
    let os = |s: &'static str| Some(std::ffi::OsStr::new(s));
    assert!(link_on(None));
    assert!(!link_on(os("0")));
    for on in ["1", "", "00", "off", "false", " 0"] {
        assert!(link_on(os(on)), "{on:?}");
    }
}
