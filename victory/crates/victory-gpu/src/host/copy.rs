//! Copies of title memory at submission: a kernel read of the ranges the title may remap, a
//! sparse fallback for holes, and the bytes-moved count the copies counters add.

use std::sync::OnceLock;

use super::{for_pieces, read};

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// Returns the bytes it copied: `len` when the copy ran (a sparse fallback
/// fills holes with zeros, so the destination still takes all of them), and 0
/// when it left the destination alone (copy mode off, an empty range, or a
/// title destination that is not there). The copies counters add this, not
/// the length asked for.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) -> usize {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return len;
        }
        2 => return 0,
        _ => {}
    }
    if len == 0 {
        return 0;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return 0;
    }
    // The kernel writes the destination (and the sparse fallback may write
    // zeros into it from user mode): its blocks stay open until the copy
    // and any retries are done, whatever guard is armed meanwhile. A guard
    // re-armed between an opening and the write would make the kernel's
    // copy fail on the block instead of faulting into the guard.
    let _writing = title_dst.then(|| crate::watch::kernel_writing(dst as usize, len));
    if title_dst {
        // A title copy command, which the card would run: the census
        // counts the destination as the card's.
        #[cfg(feature = "diagnostics")]
        crate::memclass::card(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len, crate::kernelstat::Caller::CopyMapped);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return len;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return 0;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    crate::kernelstat::fallback(crate::kernelstat::Caller::CopyMapped);
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
    len
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS};
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Query);
        let size = crate::kernelstat::virtual_query(crate::kernelstat::Place::Copy, at as usize, &mut info);
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            crate::kernelstat::fallback(crate::kernelstat::Caller::ReadSparse);
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n, crate::kernelstat::Caller::ReadSparse) {
            // A mapping may change between query and read.
            crate::kernelstat::fallback(crate::kernelstat::Caller::ReadSparse);
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

/// Reads `len` bytes of title memory at `src` into our memory at `dst`
/// through the kernel for `caller`, pages that cannot be read as zeros (as
/// [`copy_mapped`] reads a source): a diagnostic's look, counted under its
/// own caller and kept out of the copies' own counters (`crate::copystat`).
pub(crate) fn read_for(caller: crate::kernelstat::Caller, src: *const u8, dst: *mut u8, len: usize) {
    if len == 0 || read(src, dst, len, caller) {
        return;
    }
    crate::kernelstat::fallback(caller);
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n, caller) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n, caller) {
                    // SAFETY: inside the destination, which is ours.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n, crate::kernelstat::Caller::ReadPieces) {
            crate::kernelstat::fallback(crate::kernelstat::Caller::ReadPieces);
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n, crate::kernelstat::Caller::ReadPieces) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

