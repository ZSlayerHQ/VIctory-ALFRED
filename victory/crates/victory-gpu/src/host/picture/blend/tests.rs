use super::*;
use crate::host::picture::{pixel_bytes, write_row};

fn warp() -> Option<&'static Host> {
    if !crate::switch::on("VICTORY_WARP") || !crate::switch::on("VICTORY_D3D12_WARP") {
        return None;
    }
    Some(crate::host::host().expect("WARP device"))
}

fn target(host: &Host, format: DXGI_FORMAT, (w, h): (u32, u32)) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(w),
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture = None;
    // SAFETY: valid heap and texture descriptions.
    unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            None,
            &mut texture,
        )
    }
    .unwrap();
    texture.unwrap()
}

fn clear(host: &Host, list: &HostList, texture: &ID3D12Resource) {
    let rtv = host.rtv(Some(texture), None);
    // SAFETY: the target starts in the render-target state.
    unsafe {
        list.list.ClearRenderTargetView(rtv, &[0.0, 0.0, 1.0, 1.0], None);
        list.list.ResourceBarrier(&[transition(
            texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
    }
}

#[test]
fn warp_blends_coverage_and_keeps_the_old_opaque_pixels() {
    let Some(host) = warp() else { return };
    let size = (17, 9);
    let at = (7, 11);
    let opaque = [238, 72, 25, 255u8];
    let mut canvas = opaque.repeat((size.0 * size.1) as usize);
    canvas[3] = 0;
    canvas[7] = 128;
    for format in [
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
    ] {
        let mut blend = Blend::default();
        assert!(blend.ready(host, format), "{format:?}");
        let back = target(host, format, (64, 32));
        let mut rgba = Buffers::overlay(host, format, size, 4, true).unwrap();
        let mut rgb = Buffers::new(host, format, size, 4).unwrap();
        rgba.write_canvas(&canvas, size);
        rgb.write_canvas(&opaque.repeat((size.0 * size.1) as usize), size);
        let list = host.new_list().unwrap();
        clear(host, &list, &back);
        rgba.draw(&blend, host, &list, &back, size, at);
        rgb.draw(&blend, host, &list, &back, size, (32, at.1));
        // SAFETY: both routes restore COPY_DEST. Readback follows them in the same list.
        let (read, footprint) = unsafe {
            list.list.ResourceBarrier(&[transition(
                &back,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            crate::capture::read_back(&host.device, &list.list, &back).unwrap()
        };
        list.close();
        host.execute_sync_internal(&list);

        let mut mapped = std::ptr::null_mut();
        let pitch = footprint.Footprint.RowPitch as usize;
        let bpp = pixel_bytes(format).unwrap() as usize;
        // SAFETY: the copy completed and filled this footprint.
        unsafe { read.Map(0, None, Some(&mut mapped)) }.unwrap();
        let pixels = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), pitch * 32) };
        let pixel = |x: u32, y: u32| &pixels[y as usize * pitch + x as usize * bpp..][..bpp];
        let mut background = [0; 8];
        let mut expected = [0; 8];
        write_row(&[0, 0, 255, 255], format, &mut background);
        write_row(&opaque, format, &mut expected);
        assert_eq!(pixel(0, 0), &background[..bpp], "{format:?}");
        assert_eq!(pixel(at.0, at.1), &background[..bpp], "{format:?}");
        let centre = pixel(at.0 + size.0 / 2, at.1 + size.1 / 2);
        assert_eq!(pixel(32, at.1), &expected[..bpp], "{format:?}");
        assert_eq!(centre, &expected[..bpp], "{format:?}");
        let edge = pixel(at.0 + 1, at.1);
        assert_ne!(edge, centre, "{format:?}");
        assert_ne!(edge, &background[..bpp], "{format:?}");
        if format == DXGI_FORMAT_R8G8B8A8_UNORM {
            assert_eq!(edge, &[119, 36, 140, 255]);
        }
        unsafe { read.Unmap(0, None) };
    }
}

#[test]
fn warp_times_one_overlay_at_4k() {
    let Some(host) = warp() else { return };
    const SAMPLES: u32 = 16;
    let size = (1600, 1200);
    let at = (1120, 480);
    let format = DXGI_FORMAT_B8G8R8A8_UNORM;
    let back = target(host, format, (3840, 2160));
    let mut blend = Blend::default();
    assert!(blend.ready(host, format));
    let mut rgba = Buffers::overlay(host, format, size, 4, true).unwrap();
    let mut rgb = Buffers::new(host, format, size, 4).unwrap();
    let mut canvas = [255, 132, 251, 255].repeat((size.0 * size.1) as usize);
    for y in 0..44u32 {
        for x in 0..44 - y {
            for (x, y) in [
                (x, y),
                (size.0 - 1 - x, y),
                (x, size.1 - 1 - y),
                (size.0 - 1 - x, size.1 - 1 - y),
            ] {
                canvas[((y * size.0 + x) * 4 + 3) as usize] = if x.min(size.0 - 1 - x) + y.min(size.1 - 1 - y) >= 40 {
                    128
                } else {
                    0
                };
            }
        }
    }
    rgba.write_canvas(&canvas, size);
    rgb.write_canvas(&canvas, size);
    let list = host.new_list().unwrap();
    clear(host, &list, &back);
    // Warm up both routes and upload the canvas before timing the steady-state present work.
    rgb.draw(&blend, host, &list, &back, size, at);
    rgba.draw(&blend, host, &list, &back, size, at);

    let mut queries = None;
    let mut read = None;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let count = SAMPLES * 4;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(count) * 8,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: timestamp queries on a direct list, with an aligned readback buffer.
    unsafe {
        host.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: count,
                    NodeMask: 0,
                },
                &mut queries,
            )
            .unwrap();
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut read,
            )
            .unwrap();
    }
    let queries: ID3D12QueryHeap = queries.unwrap();
    let read: ID3D12Resource = read.unwrap();
    let mut cpu = [Vec::new(), Vec::new()];
    for sample in 0..SAMPLES {
        for pass in 0..2 {
            let index = (sample * 4) + pass * 2;
            let route = ((sample + pass) % 2) as usize;
            let buffers = if route == 0 { &mut rgb } else { &mut rgba };
            unsafe { list.list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, index) };
            let start = std::time::Instant::now();
            buffers.draw(&blend, host, &list, &back, size, at);
            cpu[route].push(start.elapsed());
            unsafe { list.list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, index + 1) };
        }
    }
    unsafe {
        list.list
            .ResolveQueryData(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, count, &read, 0)
    };
    list.close();
    host.execute_sync_internal(&list);

    let mut mapped = std::ptr::null_mut();
    // SAFETY: all queries were resolved and the list finished before mapping.
    unsafe { read.Map(0, None, Some(&mut mapped)) }.unwrap();
    let stamps = unsafe { std::slice::from_raw_parts(mapped.cast::<u64>(), count as usize) };
    let frequency = unsafe { host.queue.GetTimestampFrequency() }.unwrap() as f64;
    let mut gpu = [Vec::new(), Vec::new()];
    for (i, pair) in stamps.chunks_exact(2).enumerate() {
        assert!(pair[1] > pair[0]);
        gpu[(i / 2 + i % 2) % 2].push(pair[1] - pair[0]);
    }
    unsafe { read.Unmap(0, None) };
    for (route, buffers) in [rgb, rgba].iter().enumerate() {
        gpu[route].sort_unstable();
        cpu[route].sort_unstable();
        let middle = SAMPLES as usize / 2;
        let gpu_us = (gpu[route][middle - 1] + gpu[route][middle]) as f64 * 5e5 / frequency;
        let cpu_us = (cpu[route][middle - 1] + cpu[route][middle]).as_secs_f64() * 5e5;
        // Allocation information includes D3D12's alignment for each committed resource.
        let bytes = [0, 1].map(|upload| unsafe {
            let resource = if upload == 0 { &buffers.card } else { &buffers.uploads[0].0 };
            host.device.GetResourceAllocationInfo(0, &[resource.GetDesc()]).SizeInBytes
        });
        eprintln!(
            "4K WARP {}: {gpu_us:.3} us GPU, {cpu_us:.3} us CPU recording, {} payload bytes ({} default, {} upload allocated), {} canvas bytes",
            if route == 0 { "copy" } else { "blend" },
            buffers.bytes(),
            bytes[0],
            bytes[1] * buffers.uploads.len() as u64,
            canvas.capacity()
        );
    }
}
