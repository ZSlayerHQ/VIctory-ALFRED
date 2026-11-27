//! The thread that lands flights sent ahead, and what a landing does: hand
//! what shaders wrote back into title memory, fetch pieces the card wrote
//! from its memory, and finish each flight's lists.

use std::sync::atomic::Ordering;

use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::budget::{Phase as Ph, phase as ph};

use super::fetch;
use super::hash_check;
use super::{
    COPY_BACK, CardPlan, Declared, FINISHED, Flight, HOST_COPY, Host, HostList, LINE_LANDED, LINE_MOVED, Late, Line,
    QUEUE_ORDER, RELAY_MOVED, RELAY_STATS, RelayFetch, bury, fail, host, join_ranges, landable, last_notes, line,
    read_backs, walls_landed,
};
#[cfg(feature = "diagnostics")]
use super::flight_stamps_landed;

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
pub(super) fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    let _p = ph(Ph::CardWait);
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::stalls::card_waited(crate::stalls::Site::Relay, waiting);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
pub(super) fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            walls_landed(HOST_COPY, *from, to - from, to - from, written, 0, None);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its big heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (tenths of a second) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            walls_landed(HOST_COPY, from, to - from, to - from, written, 0, None);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up. `beside`, with `VICTORY_LAND_SCAN=1`, is
    /// looked through while the fetch is in flight: after its list is sent
    /// and before this thread waits for it. Plans made with
    /// `VICTORY_FETCH_QUEUE=1` fetch on a copy queue instead ([`fetch`]).
    fn relay_card_back(
        &self,
        plans: &[&CardPlan],
        fetch: &mut Option<RelayFetch>,
        beside: Option<&mut crate::host::land::RelayBack>,
        exact: bool,
    ) -> bool {
        // `exact`: the landing's merge mode, read once by `relay_land` and
        // kept to its last piece (`VICTORY_EXACT_BACK`, live).
        let mut beside = beside;
        let (pieces, total) = fetch::pieces(plans);
        if let Some(fetched) = self.fetch_on_copy_queue(plans, &pieces, total, beside.as_deref_mut(), exact) {
            return fetched;
        }
        if pieces.is_empty() {
            // Nothing to fetch: every send so far is older than the next fetch
            // (with tickets out, the oldest one's order: `fetch::prune_bound`).
            crate::shadow::relay_fetched(fetch::prune_bound(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner())));
            return false;
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        #[cfg(feature = "diagnostics")]
        let mut stamps = crate::stamps::on().then_some(&mut f.stamps);
        // The fetch's phases, timed apart ([`RELAY_STATS`]): recording its
        // list, sending it (the queue's order taken, the list, the signals),
        // and the wait for the card, which runs whatever was sent before it
        // first.
        let waiting = std::time::Instant::now();
        let (mut recorded, mut sending_done) = (waiting, waiting);
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<(u64, bool)> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                crate::events::begin(&list, crate::events::RELAY_FETCH);
                // A pair around the copies, resolved in the fetch's list.
                #[cfg(feature = "diagnostics")]
                let begun = stamps.as_deref_mut().and_then(|s| {
                    s.restart();
                    s.begin(&list)
                });
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                crate::events::end(&list);
                #[cfg(feature = "diagnostics")]
                if let Some(s) = stamps.as_deref_mut() {
                    s.end(crate::stamps::Kind::Relay, begun, &list, true);
                    s.resolve(&list);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                recorded = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                if let Some(s) = stamps.as_deref_mut() {
                    s.sent(&fence, sent);
                }
                let after = self.send_drawing_fetch(&lists, &fence, sent)?;
                sending_done = std::time::Instant::now();
                // The fetch is in flight, behind what was sent before it:
                // the PC-side scan runs under it (`VICTORY_LAND_SCAN=1`).
                if let Some(back) = beside.as_deref_mut() {
                    back.look();
                }
                let reached = fetch::drawing_waited(&self.device, &fence, event, sent)?;
                #[cfg(feature = "diagnostics")]
                crate::stalls::card_waited(crate::stalls::Site::RelayFetch, sending_done);
                Ok((after, reached))
            })()
        };
        let after = match ran {
            Ok((after, true)) => after,
            Ok((_, false)) => fail(format_args!(
                "host GPU device removed during a fetch of what shaders wrote to the card's memory (VICTORY_RELAY=1)"
            )),
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        let fetched = std::time::Instant::now();
        let us = fetched.saturating_duration_since(waiting).as_micros() as u64;
        for (stat, from, to) in [(3, waiting, recorded), (5, recorded, sending_done), (6, sending_done, fetched)] {
            RELAY_STATS[stat].fetch_add(to.saturating_duration_since(from).as_micros() as u64, Ordering::Relaxed);
        }
        RELAY_STATS[8].fetch_add(total, Ordering::Relaxed);
        // The fetch has run: its pair is read.
        #[cfg(feature = "diagnostics")]
        if crate::stamps::on() {
            f.stamps.landed();
        }
        fetch::land_pieces(&pieces, data, total, us, after, exact);
        true
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let _p = ph(Ph::RelayLand);
        let landing = std::time::Instant::now();
        // Whether it brings copies in the card's memory back.
        #[cfg(feature = "diagnostics")]
        let mut fetched = false;
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<&CardPlan> = flights.iter().filter_map(|f| f.card.as_ref()).collect();
                // With `VICTORY_LAND_SCAN=1`: the copies in the PC's memory
                // are looked through while the fetch from the card's memory
                // is in flight. The list of what to look through is the same
                // either way; only when it is made differs.
                let beside = crate::host::land::scan_beside();
                // The landing's merge mode, read once here and kept to its last piece.
                let exact = crate::host::land::exact();
                let mut back = if beside { crate::host::land::declared(&flights) } else { Vec::new() };
                let mut hand = beside.then(|| crate::host::land::RelayBack::new(&back, exact));
                let pieces = self.relay_card_back(&plans, fetch, hand.as_mut(), exact);
                hash_check::look_landed(self, &plans);
                #[cfg(feature = "diagnostics")]
                {
                    fetched = pieces;
                }
                #[cfg(not(feature = "diagnostics"))]
                let _ = pieces;
                if !beside {
                    back = crate::host::land::declared(&flights);
                    hand = Some(crate::host::land::RelayBack::new(&back, exact));
                }
                if let Some(hand) = hand {
                    hand.merge();
                }
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                let done = std::time::Instant::now();
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                RELAY_STATS[11].fetch_add(done.elapsed().as_micros() as u64, Ordering::Relaxed);
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    #[cfg(feature = "diagnostics")]
                    flight_stamps_landed(&flight);
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        self.log_messages();
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
            // What a wait this landing ends was released by.
            #[cfg(feature = "diagnostics")]
            crate::stalls::landed(fetched);
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }
}
