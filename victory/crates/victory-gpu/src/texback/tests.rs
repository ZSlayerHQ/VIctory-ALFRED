//! The copy-back's first half on synthetic textures: the switch's parser,
//! selection, and on the host device (WARP with `VICTORY_D3D12_WARP=1`)
//! [`record`] then [`Back::retile_into`] against `Layout::tile` of the texels
//! written. Every kind, size and value here is invented.

use super::*;
use crate::resource::{host_resource, test_texture_of};
use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::s;

const UNORDERED_ACCESS: u32 = 0x8;
const COPY_DEST: u32 = 0x400;
const R32_FLOAT: u32 = 41;
const R32G32_FLOAT: u32 = 16;
const BC1_UNORM: u32 = 71;
/// The title's resource flag for unordered access.
const TITLE_UAV: u32 = 4;

fn k(width: u64, height: u32, format: u32) -> Kind {
    Kind {
        width,
        height,
        format,
    }
}

// --- the switch ---

#[test]
fn the_parser_reads_kinds_and_skips_what_is_not_one() {
    let p = parse("48x24x41,16x16x2");
    assert_eq!(p.kinds, [k(48, 24, 41), k(16, 16, 2)]);
    assert!(p.malformed.is_empty());
    // Space around entries, a repeat and a trailing comma.
    let p = parse(" 48x24x41 , 16x16x2,48x24x41, ");
    assert_eq!(p.kinds, [k(48, 24, 41), k(16, 16, 2)]);
    assert!(p.malformed.is_empty());
    // Off: empty, space, `0`.
    for off in ["", "   ", "0", " 0 "] {
        assert_eq!(parse(off), Parsed::default(), "{off:?} is off");
    }
    // `1` is not a kind and gives no default list.
    let p = parse("1");
    assert!(p.kinds.is_empty());
    assert_eq!(p.malformed, ["1"]);
    // Malformed entries, each skipped on its own.
    for bad in [
        "48x24",
        "48x24x",
        "x24x41",
        "48x24x41x1",
        "48X24X41",
        "48x0x41",
        "0x24x41",
        "48x24x0",
        "-4x24x41",
        "+4x24x41",
        "4.5x2x2",
        "abc",
        "48 x24x41",
        "99999999999999999999x1x1",
        "1x4294967296x1",
    ] {
        let p = parse(bad);
        assert!(p.kinds.is_empty(), "{bad:?} is not a kind");
        assert_eq!(p.malformed, [bad], "{bad:?} is said");
    }
    // A malformed entry among kinds: the kinds stay.
    let p = parse("48x24x41,bad,1,16x16x2");
    assert_eq!(p.kinds, [k(48, 24, 41), k(16, 16, 2)]);
    assert_eq!(p.malformed, ["bad", "1"]);
}

// --- selection ---

fn shape() -> Shape {
    Shape {
        two_d: true,
        width: 48,
        height: 24,
        samples: 1,
        planes: 1,
        format: R32_FLOAT,
    }
}

#[test]
fn selection_takes_listed_2d_single_sample_single_plane_textures_only() {
    let kinds = [k(48, 24, R32_FLOAT), k(8, 8, R32G32_FLOAT)];
    assert!(wanted_shape(&kinds, &shape()));
    assert!(wanted_shape(
        &kinds,
        &Shape {
            width: 8,
            height: 8,
            format: R32G32_FLOAT,
            ..shape()
        }
    ));
    let unwanted = [
        (
            "3D",
            Shape {
                two_d: false,
                ..shape()
            },
        ),
        (
            "multi-sample",
            Shape {
                samples: 4,
                ..shape()
            },
        ),
        (
            "two planes",
            Shape {
                planes: 2,
                ..shape()
            },
        ),
        (
            "no console layout",
            Shape {
                planes: 0,
                ..shape()
            },
        ),
        (
            "another format",
            Shape {
                format: R32G32_FLOAT,
                ..shape()
            },
        ),
        (
            "another width",
            Shape {
                width: 47,
                ..shape()
            },
        ),
        (
            "width and height swapped",
            Shape {
                width: 24,
                height: 48,
                ..shape()
            },
        ),
    ];
    for (what, s) in unwanted {
        assert!(!wanted_shape(&kinds, &s), "{what}");
    }
    assert!(!wanted_shape(&[], &shape()), "no kinds");
}

/// A title texture's shape is read from its description.
#[test]
fn a_title_textures_shape_is_its_description() {
    let (tex, _, _) = test_texture_of(48, 24, R32_FLOAT, 2, 3, TITLE_UAV);
    assert_eq!(crate::resource::texture_kind(tex), Some(shape()));
    assert_eq!(crate::resource::texture_kind(std::ptr::null()), None);
    assert_eq!(
        crate::resource::texture_kind(crate::resource::test_buffer(256)),
        None
    );
}

/// Runs the child test `name` in a process of its own with `env` set and
/// the switch as `switch` gives it (`None`: unset).
fn child(name: &str, switch: Option<&str>, env: &[(&str, &str)]) {
    let exe = std::env::current_exe().expect("the test binary");
    let mut command = std::process::Command::new(exe);
    command
        .args([
            "--exact",
            &format!("texback::tests::{name}"),
            "--ignored",
            "--test-threads",
            "1",
            "--nocapture",
        ])
        .envs(env.iter().copied());
    match switch {
        Some(value) => command.env(SWITCH, value),
        None => command.env_remove(SWITCH),
    };
    let out = command.output().expect("the child ran");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    for line in text.lines().filter(|l| l.contains("NOT RUN")) {
        eprintln!("{name}: {line}");
    }
    assert!(
        out.status.success() && text.contains("1 passed"),
        "{name}: {text}"
    );
}

#[test]
fn off_is_off_with_the_switch_unset() {
    child("off_is_off_with_the_switch_unset_child", None, &[]);
}

#[test]
#[ignore = "a child process of the test above (VICTORY_TEXTURE_BACK unset)"]
fn off_is_off_with_the_switch_unset_child() {
    assert!(kinds().is_empty() && !on());
    let (tex, _, _) = test_texture_of(48, 24, R32_FLOAT, 1, 1, TITLE_UAV);
    assert!(!wanted(tex), "nothing is wanted with the switch off");
}

#[test]
fn the_switch_is_read_once_into_its_kinds() {
    child(
        "the_switch_is_read_once_into_its_kinds_child",
        Some("48x24x41, 1, bad,8x8x16"),
        &[],
    );
    for off in ["", "0"] {
        child("the_switch_off_wants_nothing_child", Some(off), &[]);
    }
    child("the_switch_off_wants_nothing_child", Some("1"), &[]);
}

#[test]
#[ignore = "a child process of the test above (VICTORY_TEXTURE_BACK set)"]
fn the_switch_is_read_once_into_its_kinds_child() {
    assert_eq!(kinds(), [k(48, 24, R32_FLOAT), k(8, 8, R32G32_FLOAT)]);
    assert!(on());
    let (listed, _, _) = test_texture_of(48, 24, R32_FLOAT, 1, 1, TITLE_UAV);
    let (other, _, _) = test_texture_of(48, 24, R32G32_FLOAT, 1, 1, TITLE_UAV);
    let (array, _, _) = test_texture_of(8, 8, R32G32_FLOAT, 2, 2, TITLE_UAV);
    assert!(wanted(listed));
    assert!(!wanted(other));
    assert!(
        wanted(array),
        "arrays and mip chains of a listed kind are wanted"
    );
}

#[test]
#[ignore = "a child process of the test above (VICTORY_TEXTURE_BACK empty, 0 or 1)"]
fn the_switch_off_wants_nothing_child() {
    assert!(kinds().is_empty() && !on());
    let (tex, _, _) = test_texture_of(1, 1, R32_FLOAT, 1, 1, TITLE_UAV);
    assert!(!wanted(tex));
}

// --- record and retile, on the host device ---

fn host() -> &'static crate::host::Host {
    crate::host::test_host().expect("a host device (VICTORY_D3D12_WARP=1 for WARP)")
}

/// A compute program of ours that writes texel `(x, y)` of mip `level` of
/// array slice `slice` of a `ty` texture through a view of that one
/// subresource: `value` of `v = x + 1000 y + 100000 level + 1000000 slice`.
struct Writer {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: usize,
    /// The next view slot: a slot a recorded list names is never written
    /// again.
    next: std::cell::Cell<usize>,
}

impl Writer {
    fn new(ty: &str, value: &str, views: u32) -> Writer {
        let source = format!(
            "#define RS \"DescriptorTable(UAV(u0)), RootConstants(num32BitConstants=2, b0)\"\n\
             RWTexture2DArray<{ty}> t : register(u0);\n\
             cbuffer c : register(b0) {{ uint level; uint slice; }};\n\
             [RootSignature(RS)]\n[numthreads(8, 8, 1)]\n\
             void main(uint3 id : SV_DispatchThreadID) {{\n\
             uint w, h, n; t.GetDimensions(w, h, n);\n\
             if (id.x >= w || id.y >= h) {{ return; }}\n\
             uint v = id.x + id.y * 1000u + level * 100000u + slice * 1000000u;\n\
             t[uint3(id.xy, 0)] = {value};\n}}\n"
        );
        let host = host();
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too; plain object
        // creation.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .expect("compiled");
            let code = code.expect("code");
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host
                .device
                .CreateRootSignature(0, bytes)
                .expect("a root signature");
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: views,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("a view heap");
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            Writer {
                root,
                pso,
                heap,
                step,
                next: std::cell::Cell::new(0),
            }
        }
    }

    /// In `list`: title texture `tex` to unordered access, then every mip
    /// of every slice written, one dispatch each, through views in slots of
    /// the heap not used before.
    fn write(
        &self,
        list: &HostList,
        tex: This,
        format: u32,
        size: (u32, u32),
        mips: u32,
        slices: u32,
    ) {
        crate::list::ensure_state(list, tex, UNORDERED_ACCESS);
        let texture = host_resource(tex).expect("a host copy");
        // SAFETY: an open list, our pipeline and heap, a live texture; each
        // view is of one subresource of it.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            for slice in 0..slices {
                for level in 0..mips {
                    let i = self.next.get();
                    self.next.set(i + 1);
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT(format as i32),
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                                MipSlice: level,
                                FirstArraySlice: slice,
                                ArraySize: 1,
                                PlaneSlice: 0,
                            },
                        },
                    };
                    let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: self.heap.GetCPUDescriptorHandleForHeapStart().ptr + i * self.step,
                    };
                    host()
                        .device
                        .CreateUnorderedAccessView(texture, None, Some(&desc), cpu);
                    let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: self.heap.GetGPUDescriptorHandleForHeapStart().ptr
                            + (i * self.step) as u64,
                    };
                    list.list.SetComputeRootDescriptorTable(0, gpu);
                    let constants = [level, slice];
                    list.list
                        .SetComputeRoot32BitConstants(1, 2, constants.as_ptr().cast(), 0);
                    let (w, h) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
                    list.list.Dispatch(w.div_ceil(8), h.div_ceil(8), 1);
                }
            }
        }
    }
}

/// The bytes of texel `(x, y)` of mip `level` of slice `slice` as the
/// writer stores it, with `parts` floats (1: `v`; 2: `v`, `v + 0.5`).
fn texel(x: u32, y: u32, level: u32, slice: u32, parts: u32) -> Vec<u8> {
    let v = (x + y * 1000 + level * 100_000 + slice * 1_000_000) as f32;
    let mut out = v.to_le_bytes().to_vec();
    if parts == 2 {
        out.extend((v + 0.5).to_le_bytes());
    }
    out
}

/// `fill` bytes of the layout's size with each subresource tiled in from
/// tight linear rows `rows(mip, slice)` of `row_bytes(mip)` bytes.
fn tiled(
    layout: &Layout,
    fill: u8,
    mips: u32,
    slices: u32,
    rows: impl Fn(u32, u32) -> (Vec<u8>, usize),
) -> Vec<u8> {
    let mut out = vec![fill; layout.size as usize];
    for slice in 0..slices {
        for mip in 0..mips {
            let (linear, row) = rows(mip, slice);
            layout
                .tile(&mut out, mip, slice, &linear, row, linear.len())
                .unwrap_or_else(|e| panic!("mip {mip} slice {slice}: {e:?}"));
        }
    }
    out
}

/// Byte for byte, with the first difference named.
fn same(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(at) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!(
            "{what}: first difference at byte {at} of {}: {:#04x}, wanted {:#04x}",
            got.len(),
            got[at],
            want[at]
        );
    }
}

/// The title memory of a texture, copied.
fn title_bytes(memory: *mut u8, layout: &Layout) -> Vec<u8> {
    // SAFETY: the texture's title memory, its layout's size.
    unsafe { std::slice::from_raw_parts(memory, layout.size as usize).to_vec() }
}

/// A `width` x `height` title texture of `format` (`parts` floats a texel),
/// `mips` levels and `slices` slices, written whole by our compute pass,
/// recorded in the same list, run, and re-tiled: the span is
/// `Layout::tile` of the texels written, byte for byte, and the title's
/// memory is what it was.
fn written_then_retiled(
    format: u32,
    ty: &str,
    value: &str,
    parts: u32,
    size: (u32, u32),
    mips: u32,
    slices: u32,
) {
    let host = host();
    let (tex, memory, layout) = test_texture_of(
        u64::from(size.0),
        size.1,
        format,
        mips as u16,
        slices as u16,
        TITLE_UAV,
    );
    let before = title_bytes(memory, &layout);
    let writer = Writer::new(ty, value, mips * slices);
    let list = host.new_list().expect("a host list");
    writer.write(&list, tex, format, size, mips, slices);
    let back = record(&list, tex).expect("recorded");
    assert_eq!(back.title_span(), (memory as usize, layout.size as usize));
    let texels: u64 = (0..mips)
        .map(|m| u64::from((size.0 >> m).max(1)) * u64::from((size.1 >> m).max(1)))
        .sum();
    assert!(
        back.bytes() >= texels * u64::from(parts) * 4 * u64::from(slices),
        "the read-back buffer holds every texel"
    );
    list.close();
    host.execute(&[&list]);
    // SAFETY: a plain query.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is not removed");
    let mut span = vec![0xA5; layout.size as usize];
    back.retile_into(&mut span).expect("re-tiled");
    let want = tiled(&layout, 0xA5, mips, slices, |mip, slice| {
        let (w, h) = ((size.0 >> mip).max(1), (size.1 >> mip).max(1));
        let linear: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| texel(x, y, mip, slice, parts)))
            .collect();
        (linear, (w * parts * 4) as usize)
    });
    same(&span, &want, "the re-tiled span");
    same(
        &title_bytes(memory, &layout),
        &before,
        "title memory, untouched",
    );
}

#[test]
fn a_written_r32_float_texture_is_recorded_and_retiled_exactly() {
    written_then_retiled(R32_FLOAT, "float", "float(v)", 1, (64, 40), 1, 1);
}

#[test]
fn a_written_five_level_r32g32_float_chain_is_recorded_and_retiled_exactly() {
    written_then_retiled(
        R32G32_FLOAT,
        "float2",
        "float2(float(v), float(v) + 0.5)",
        2,
        (48, 20),
        5,
        1,
    );
}

#[test]
fn a_written_array_of_two_slices_is_recorded_and_retiled_exactly() {
    written_then_retiled(R32_FLOAT, "float", "float(v)", 1, (32, 24), 2, 2);
}

/// A BC1 texture of three levels filled by a copy from our upload buffer
/// (bytes made up for each block), recorded and re-tiled: the tiler on
/// block elements (8 bytes, a block of 4 x 4 texels).
#[test]
fn a_bc1_texture_filled_by_a_copy_is_recorded_and_retiled_exactly() {
    let host = host();
    let (w, h, mips) = (64u32, 32u32, 3u32);
    let (tex, memory, layout) = test_texture_of(u64::from(w), h, BC1_UNORM, mips as u16, 1, 0);
    let before = title_bytes(memory, &layout);
    let texture = host_resource(tex).expect("a host copy");
    // SAFETY: plain queries of a live texture, out arrays of `mips` entries.
    let (footprints, counts, total) = unsafe {
        let desc = texture.GetDesc();
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mips as usize];
        let mut counts = vec![0u32; mips as usize];
        let mut total = 0u64;
        host.device.GetCopyableFootprints(
            &desc,
            0,
            mips,
            0,
            Some(footprints.as_mut_ptr()),
            Some(counts.as_mut_ptr()),
            None,
            Some(&mut total),
        );
        (footprints, counts, total)
    };
    let block = |mip: u32, x: u32, y: u32, b: u32| (mip * 71 + x * 13 + y * 29 + b * 5 + 1) as u8;
    let blocks = |mip: u32| ((w >> mip).max(1).div_ceil(4), (h >> mip).max(1).div_ceil(4));
    let (upload, at) = host.upload_buffer(total).expect("an upload buffer");
    for (mip, (f, &rows)) in footprints.iter().zip(&counts).enumerate() {
        let mip = mip as u32;
        let (bw, bh) = blocks(mip);
        assert_eq!(rows, bh, "a row of blocks a footprint row");
        for y in 0..bh {
            for x in 0..bw {
                for b in 0..8 {
                    let to = f.Offset as usize + (y * f.Footprint.RowPitch + x * 8 + b) as usize;
                    // SAFETY: inside the upload buffer's mapping of `total` bytes.
                    unsafe { at.add(to).write(block(mip, x, y, b)) };
                }
            }
        }
    }
    let list = host.new_list().expect("a host list");
    crate::list::ensure_state(&list, tex, COPY_DEST);
    for (i, f) in footprints.iter().enumerate() {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *f,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture in copy dest, the upload buffer
        // holding every footprint.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    let back = record(&list, tex).expect("recorded");
    assert!(
        list.transition(tex, COPY_DEST).is_none(),
        "the texture is left in copy dest"
    );
    list.close();
    host.execute(&[&list]);
    // SAFETY: a plain query.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is not removed");
    let mut span = vec![0x5A; layout.size as usize];
    back.retile_into(&mut span).expect("re-tiled");
    let want = tiled(&layout, 0x5A, mips, 1, |mip, _| {
        let (bw, bh) = blocks(mip);
        let linear: Vec<u8> = (0..bh)
            .flat_map(|y| (0..bw).flat_map(move |x| (0..8).map(move |b| block(mip, x, y, b))))
            .collect();
        (linear, (bw * 8) as usize)
    });
    same(&span, &want, "the re-tiled BC1 span");
    same(
        &title_bytes(memory, &layout),
        &before,
        "title memory, untouched",
    );
}

/// `record` leaves the texture in the state it found it in: the list still
/// has it in unordered access, a dispatch writing it after the record and a
/// later list's use of it validate, and the debug layer says nothing.
#[test]
fn record_leaves_the_texture_in_the_state_it_found() {
    child(
        "record_leaves_the_texture_in_the_state_it_found_child",
        None,
        &[("VICTORY_D3D12_DEBUG", "1")],
    );
}

#[test]
#[ignore = "a child process of the test above (the debug layer on)"]
fn record_leaves_the_texture_in_the_state_it_found_child() {
    let host = host();
    if crate::host::take_host_debug_errors(host).is_none() {
        eprintln!("NOT RUN (no debug layer): the debug layer's check; the rest is checked");
    }
    let size = (32u32, 16u32);
    let (tex, _, _) = test_texture_of(u64::from(size.0), size.1, R32_FLOAT, 1, 1, TITLE_UAV);
    let writer = Writer::new("float", "float(v)", 3);
    let list = host.new_list().expect("a host list");
    writer.write(&list, tex, R32_FLOAT, size, 1, 1);
    let back = record(&list, tex).expect("recorded");
    assert!(
        list.transition(tex, UNORDERED_ACCESS).is_none(),
        "the list has the texture in unordered access"
    );
    // A dispatch writing it after the record, in the same list.
    writer.write(&list, tex, R32_FLOAT, size, 1, 1);
    list.close();
    host.execute(&[&list]);
    // A later list that uses it as the record left it.
    let next = host.new_list().expect("a host list");
    writer.write(&next, tex, R32_FLOAT, size, 1, 1);
    next.close();
    host.execute(&[&next]);
    // SAFETY: a plain query.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is not removed");
    drop(back);
    if let Some(errors) = crate::host::take_host_debug_errors(host) {
        assert!(errors.is_empty(), "the debug layer said: {errors:#?}");
    }
}
