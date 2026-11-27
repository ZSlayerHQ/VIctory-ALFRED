//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. A poll thread turns host state into readings (a new one per change)
//! and runs device callbacks on connect and disconnect.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

fn fallback(keys: &[KeyState]) -> PadState {
    let on = |vk: u8| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: FALLBACK_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z'),
        right_trigger: on(b'C'),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself: its `id` is filled in; the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index < PADS { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source);
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            (pads, source.keys())
        };
        let events = self.state().poll(&self.devices, pads, keys);
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        let any_pad = pads.iter().any(Option::is_some);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then(|| Value::Pad(fallback(&keys)));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    let _entry = victory_gpu::hle_call();
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    let _entry = victory_gpu::hle_call();
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    let history = &state.slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; no callback has been registered
// (the caller polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (a caller may register while holding a lock its
/// callback takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    let _entry = victory_gpu::hle_call();
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    let _entry = victory_gpu::hle_call();
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub.returned.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {
    let _entry = victory_gpu::hle_call();
}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    let _entry = victory_gpu::hle_call();
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; trigger levels arrive in this call's parameters.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.index >= PADS {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    lock(&device.hub().source).rumble(device.index, speed(low), speed(high));
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    let _entry = victory_gpu::hle_call();
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let _entry = victory_gpu::hle_call();
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    let _entry = victory_gpu::hle_call();
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let _entry = victory_gpu::hle_call();
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    let _entry = victory_gpu::hle_call();
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    let _entry = victory_gpu::hle_call();
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    let _entry = victory_gpu::hle_call();
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let _entry = victory_gpu::hle_call();
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    let _entry = victory_gpu::hle_call();
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    let _entry = victory_gpu::hle_call();
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing as *const (),           // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, the keyboard while our process has focus.
#[cfg(windows)]
mod host {
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId,
    };

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        /// This poll's scripted keys: evaluated in pad(0) and taken by keys() (diagnostics only). A present
        /// between the two views would otherwise leave a key down on the pad and up on the keyboard in one poll.
        #[cfg(feature = "diagnostics")]
        scripted_poll: Option<Vec<u8>>,
    }

    // The sequence parser is shared with the live `LIVE_KEYS` switch; it
    // lives in victory-gpu so the switch can refuse a bad sequence before
    // it is applied. The hold rule and its evaluation stay here.
    #[cfg(any(feature = "diagnostics", test))]
    use victory_gpu::scripted_keys::{KEY_PRESS, Hold, KeyScript, ScriptedKey, key_script};


    /// How long a count that does not advance may keep a frame hold down: after this, the key is released and
    /// the log says so once, naming the entry and the presents it got. No key is ever held without an end.
    #[cfg(any(feature = "diagnostics", test))]
    const STALL: Duration = Duration::from_secs(30);

    /// What a frame hold remembers between polls: the present count its press was seen at, the count at the last
    /// poll and the poll time it last advanced (a count that stops), whether it fired with no count at all (one
    /// press per entry, ever), and whether each once-only warning has been said.
    #[cfg(any(feature = "diagnostics", test))]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct FrameHold {
        pressed_at: Option<u64>,
        last: Option<u64>,
        advanced: Option<Duration>,
        fired_without_count: bool,
        warned_no_count: bool,
        warned_stall: bool,
    }

    /// The keys down at `now`: a time hold from its start for its whole length; a frame hold from the poll that
    /// first finds its start reached until `frames` of the title's presents have happened since (`holds` keeps
    /// what each entry remembers between polls, an entry an index). With no present yet the count cannot be read:
    /// the entry fires at most once, as the short press if a poll comes in time and as nothing if not, and the log
    /// says which once. A count that stops advancing for [`STALL`] releases the hold with one warning.
    #[cfg(any(feature = "diagnostics", test))]
    fn keys_down_with(
        keys: &[ScriptedKey],
        now: Duration,
        presents: Option<u64>,
        holds: &mut Vec<FrameHold>,
    ) -> Vec<u8> {
        holds.resize(keys.len(), FrameHold::default());
        keys.iter()
            .enumerate()
            .filter_map(|(i, key)| {
                let held = &mut holds[i];
                match key.hold {
                    Hold::Time(hold) => (key.at..key.at + hold).contains(&now).then_some(key.key),
                    Hold::Frames(frames) => {
                        if now < key.at {
                            return None;
                        }
                        let Some(now_presents) = presents else {
                            // No count ever seen: the entry fires at most once, as its short press across the polls
                            // inside the press window (or as nothing, when the first due poll is past it), and the
                            // log says which once.
                            if !held.fired_without_count {
                                held.fired_without_count = true;
                                if !held.warned_no_count {
                                    held.warned_no_count = true;
                                    if now < key.at + KEY_PRESS {
                                        tracing::warn!(
                                            key = key.key,
                                            "a VICTORY_KEYS hold counts the title's presents but none has been presented: pressed once instead"
                                        );
                                    } else {
                                        tracing::warn!(
                                            key = key.key,
                                            "a VICTORY_KEYS hold counts the title's presents but none has been presented, and its press window is past: nothing was pressed for it"
                                        );
                                    }
                                }
                            }
                            return (now < key.at + KEY_PRESS).then_some(key.key);
                        };
                        if held.fired_without_count || held.warned_stall {
                            return None;
                        }
                        let pressed = *held.pressed_at.get_or_insert(now_presents);
                        if held.last != Some(now_presents) {
                            held.last = Some(now_presents);
                            held.advanced = Some(now);
                        }
                        if now_presents - pressed >= frames {
                            return None;
                        }
                        if now.saturating_sub(held.advanced.unwrap_or(now)) >= STALL {
                            held.warned_stall = true;
                            tracing::warn!(
                                key = key.key,
                                presents = now_presents,
                                "a VICTORY_KEYS frame hold's count stopped: the key is released"
                            );
                            return None;
                        }
                        Some(key.key)
                    }
                }
            })
            .collect()
    }

    /// [`keys_down_with`] for a time-only list (`now`, nothing to remember between polls). Test-only.
    #[cfg(test)]
    fn keys_down(keys: &[ScriptedKey], now: Duration) -> Vec<u8> {
        keys_down_with(keys, now, None, &mut Vec::new())
    }

    /// Every key down, each clock's frame holds remembered in `holds`: the absolute entries at `since_start`,
    /// and the `+` entries at `since_after` (nothing when the marker has not appeared).
    #[cfg(any(feature = "diagnostics", test))]
    fn keys_down_at_with(
        script: &KeyScript,
        since_start: Duration,
        since_after: Option<Duration>,
        presents: Option<u64>,
        holds: &mut (Vec<FrameHold>, Vec<FrameHold>),
    ) -> Vec<u8> {
        let mut down = keys_down_with(&script.absolute, since_start, presents, &mut holds.0);
        if let Some(since) = since_after {
            down.extend(keys_down_with(&script.relative, since, presents, &mut holds.1));
        }
        down
    }

    /// [`keys_down_at_with`] with nothing between polls (a time-only list, as the tests build). Test-only.
    #[cfg(test)]
    fn keys_down_at(script: &KeyScript, since_start: Duration, since_after: Option<Duration>) -> Vec<u8> {
        keys_down_at_with(script, since_start, since_after, None, &mut (Vec::new(), Vec::new()))
    }

    impl Host {
        /// This poll's scripted keys: pad(0) evaluated them and left them here; with no pad in slot 0 this is the
        /// poll's one evaluation either way, and the pad and the keyboard read the same list.
        #[cfg(feature = "diagnostics")]
        fn scripted_once(&mut self) -> Vec<u8> {
            self.scripted_poll.take().unwrap_or_else(scripted)
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            #[cfg(all(test, feature = "diagnostics"))]
            let connected = PAD_FOR_TESTS.with(|p| p.get()) || unsafe { XInputGetState(slot as u32, &mut state) } == 0;
            #[cfg(not(all(test, feature = "diagnostics")))]
            let connected = unsafe { XInputGetState(slot as u32, &mut state) } == 0;
            if !connected {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            #[allow(unused_mut)]
            let mut pad = convert(&state.Gamepad);
            // The scripted keys (`VICTORY_KEYS`) also press the first
            // pad's buttons, by the stand-in pad's table: with a real pad
            // connected the stand-in is not there, and a title that listens
            // to the pad (the title's start screen) saw nothing.
            #[cfg(feature = "diagnostics")]
            if slot == 0 {
                let down = scripted();
                for vk in &down {
                    if let Some(&(_, button)) = super::FALLBACK_BUTTONS.iter().find(|(key, _)| *key == *vk) {
                        pad.buttons |= button;
                    }
                }
                self.scripted_poll = Some(down);
            }
            Some(pad)
        }

        fn keys(&mut self) -> Vec<KeyState> {
            #[cfg(feature = "diagnostics")]
            let mut down = self.scripted_once();
            #[cfg(not(feature = "diagnostics"))]
            let mut down = Vec::new();
            #[cfg(all(test, feature = "diagnostics"))]
            let focused = focused() && !PHYSICAL_KEYS_FOR_TESTS.with(|p| p.get());
            #[cfg(not(all(test, feature = "diagnostics")))]
            let focused = focused();
            if focused {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            down.into_iter()
                .map(|vk| KeyState {
                    // SAFETY: plain query.
                    scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                    virtual_key: vk,
                    // shortcut: no characters (code point 0); the runtime
                    // reports scan codes and virtual keys.
                    ..Default::default()
                })
                .collect()
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: low,
                wRightMotorSpeed: high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>[:<hold>],...`
    /// presses each key that long after input starts, for its hold (0.2 s
    /// unless given: seconds, or `<N>f` for N of the title's presents —
    /// counted when the title presents, whether or not the frame reaches the
    /// window — from the press, so a route ends on the same view whatever the
    /// frame rate; with nothing presented yet, one press at most and one
    /// warning), with or without focus. A key is a letter, a digit, or
    /// `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`, `Left`, `Right` (the
    /// stand-in pad's table maps them to buttons). An entry written
    /// `+<seconds>:<key>[:<hold>]` counts from the moment the file named by
    /// `VICTORY_KEYS_AFTER` first exists, so a harness can start a script
    /// when the scene is playable. Without that variable, `+` entries are
    /// skipped with one warning.
    #[cfg(feature = "diagnostics")]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, KeyScript)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            (Instant::now(), key_script(&list))
        });
        static AFTER: LazyLock<Option<std::path::PathBuf>> = LazyLock::new(|| {
            std::env::var_os("VICTORY_KEYS_AFTER").map(std::path::PathBuf::from)
        });
        /// Each clock's frame holds, what they remember between polls (absolute, relative).
        static HOLDS: std::sync::Mutex<(Vec<FrameHold>, Vec<FrameHold>)> =
            std::sync::Mutex::new((Vec::new(), Vec::new()));
        #[cfg(test)]
        {
            // The one-poll fixture's seams: a script it supplies, a count it sets, and holds of its own.
            let supplied = SCRIPT_FOR_TESTS.with(|s| s.borrow().as_ref().map(|(start, script)| (*start, std::rc::Rc::clone(script))));
            if let Some((start, script)) = supplied {
                let since_start = start.elapsed();
                let presents = presented();
                let after = if script.relative.is_empty() { None } else { AFTER.as_deref() };
                let mut down = HOLDS_FOR_TESTS.with(|h| {
                    let mut holds = h.borrow_mut();
                    evaluate(since_start, &script, after, presents, &mut holds)
                });
                down.extend(live_scripted(presents));
                return down;
            }
        }
        // The order a poll read these at 5051676b5: the clock, then the title's presents, then the holds; and the
        // marker file is looked for only when a script has '+' entries.
        let (start, script) = &*SCRIPT;
        let since_start = start.elapsed();
        let presents = presented();
        let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
        let after = if script.relative.is_empty() { None } else { AFTER.as_deref() };
        let mut down = evaluate(since_start, script, after, presents, &mut holds);
        drop(holds);
        down.extend(live_scripted(presents));
        down
    }

    /// A live `LIVE_KEYS` sequence's playback between polls: the run it
    /// belongs to, each frame hold's memory, and whether its done line has
    /// been let out.
    #[cfg(feature = "diagnostics")]
    struct LivePlay {
        id: u64,
        holds: Vec<FrameHold>,
        done: bool,
    }

    /// The run playing in a real session.
    #[cfg(feature = "diagnostics")]
    static LIVE_PLAY: Mutex<Option<LivePlay>> = Mutex::new(None);

    /// The live sequence's keys at this poll: the same delivery path as
    /// `VICTORY_KEYS`, with every time counted from the frame boundary that
    /// applied the change. When every press has ended the run is reported
    /// once through [`victory_gpu::live_keys_done`].
    #[cfg(feature = "diagnostics")]
    fn live_scripted(presents: Option<u64>) -> Vec<u8> {
        #[cfg(test)]
        if let Some(run) = LIVE_FOR_TESTS.with(|l| l.borrow().clone()) {
            return LIVE_PLAY_FOR_TESTS.with(|play| play_live(&mut play.borrow_mut(), run, presents));
        }
        let mut play = LIVE_PLAY.lock().unwrap_or_else(|e| e.into_inner());
        match victory_gpu::live_keys() {
            Some(run) => play_live(&mut play, run, presents),
            None => {
                *play = None;
                Vec::new()
            }
        }
    }

    /// One poll of `run` through a playback state: the keys down, and the
    /// done line once when the last of them has ended.
    #[cfg(feature = "diagnostics")]
    fn play_live(play: &mut Option<LivePlay>, run: victory_gpu::KeysRun, presents: Option<u64>) -> Vec<u8> {
        if play.as_ref().is_none_or(|play| play.id != run.id) {
            *play = Some(LivePlay {
                id: run.id,
                holds: Vec::new(),
                done: false,
            });
        }
        let play = play.as_mut().expect("the run was just stored");
        let since = run.start.elapsed();
        let down = keys_down_with(&run.presses, since, presents, &mut play.holds);
        if !play.done && live_finished(&run.presses, since, presents, &play.holds) {
            play.done = true;
            victory_gpu::live_keys_done(run.id);
        }
        down
    }

    /// Whether every press of a live sequence has ended: each timed hold
    /// past its end, each frame hold released by its presents, its no-count
    /// fallback's short press over, or its stall warning.
    #[cfg(feature = "diagnostics")]
    fn live_finished(keys: &[ScriptedKey], since: Duration, presents: Option<u64>, holds: &[FrameHold]) -> bool {
        keys.iter().zip(holds).all(|(key, held)| match key.hold {
            Hold::Time(hold) => since >= key.at.saturating_add(hold),
            Hold::Frames(frames) => {
                if since < key.at {
                    return false;
                }
                if held.warned_stall {
                    return true;
                }
                if held.fired_without_count {
                    return since >= key.at + KEY_PRESS;
                }
                held.pressed_at
                    .zip(presents)
                    .is_some_and(|(pressed, now)| now.saturating_sub(pressed) >= frames)
            }
        })
    }

    /// The keys the script has down: one evaluation over the elapsed `since_start`, `script`, the marker
    /// `after` and the title's presents so far.
    #[cfg(feature = "diagnostics")]
    fn evaluate(
        since_start: Duration,
        script: &KeyScript,
        after: Option<&std::path::Path>,
        presents: Option<u64>,
        holds: &mut (Vec<FrameHold>, Vec<FrameHold>),
    ) -> Vec<u8> {
        if script.relative.is_empty() {
            keys_down_with(&script.absolute, since_start, presents, &mut holds.0)
        } else if let Some(path) = after {
            let since_after = marker_elapsed(path);
            keys_down_at_with(script, since_start, since_after, presents, holds)
        } else {
            warn_relative_without_marker(script.relative.len());
            keys_down_with(&script.absolute, since_start, presents, &mut holds.0)
        }
    }

    /// The title's presents so far; none yet (a frame hold cannot count then) is `None`. Tests set it.
    #[cfg(feature = "diagnostics")]
    fn presented() -> Option<u64> {
        #[cfg(test)]
        if let Some(count) = PRESENT_FOR_TESTS.with(|p| p.get()) {
            return Some(count);
        }
        let presented = victory_gpu::queue::presented();
        (presented >= 1).then_some(presented)
    }

    // The index fixture's seams (`one_poll_evaluates_the_script_once_for_both_views`): a script it supplies and
    // holds of its own, a present count it sets, and a pad it makes look connected in slot 0.
    #[cfg(all(test, feature = "diagnostics"))]
    thread_local! {
        static SCRIPT_FOR_TESTS: std::cell::RefCell<Option<(Instant, std::rc::Rc<KeyScript>)>> = const { std::cell::RefCell::new(None) };
        static HOLDS_FOR_TESTS: std::cell::RefCell<(Vec<FrameHold>, Vec<FrameHold>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        static PRESENT_FOR_TESTS: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
        static PAD_FOR_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// The physical keyboard reads as empty, so a person holding a key cannot change what a fixture sees.
        static PHYSICAL_KEYS_FOR_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// A live run the test supplies in place of the registry's, and its playback memory.
        static LIVE_FOR_TESTS: std::cell::RefCell<Option<victory_gpu::KeysRun>> = const { std::cell::RefCell::new(None) };
        static LIVE_PLAY_FOR_TESTS: std::cell::RefCell<Option<LivePlay>> = const { std::cell::RefCell::new(None) };
    }

    /// Tests: the scripted keys come from `list` until the guard drops, the pad in slot 0 reads as connected,
    /// and the present count is `presented`.
    #[cfg(all(test, feature = "diagnostics"))]
    fn scripted_for_tests(list: &str, presented: u64) -> impl Drop {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                SCRIPT_FOR_TESTS.with(|s| *s.borrow_mut() = None);
                HOLDS_FOR_TESTS.with(|h| *h.borrow_mut() = (Vec::new(), Vec::new()));
                PRESENT_FOR_TESTS.with(|p| p.set(None));
                PAD_FOR_TESTS.with(|p| p.set(false));
                PHYSICAL_KEYS_FOR_TESTS.with(|p| p.set(false));
            }
        }
        SCRIPT_FOR_TESTS.with(|s| *s.borrow_mut() = Some((Instant::now(), std::rc::Rc::new(key_script(list)))));
        HOLDS_FOR_TESTS.with(|h| *h.borrow_mut() = (Vec::new(), Vec::new()));
        PRESENT_FOR_TESTS.with(|p| p.set(Some(presented)));
        PAD_FOR_TESTS.with(|p| p.set(true));
        PHYSICAL_KEYS_FOR_TESTS.with(|p| p.set(true));
        Reset
    }

    /// Tests: the present count the scripted keys read from now on.
    #[cfg(all(test, feature = "diagnostics"))]
    fn set_presented_for_tests(presented: u64) {
        PRESENT_FOR_TESTS.with(|p| p.set(Some(presented)));
    }

    /// Tests: a live run of `list` that started `elapsed` ago, in the
    /// registry's place until the guard drops. The keyboard and pad read it
    /// through the same `scripted()` evaluation.
    #[cfg(all(test, feature = "diagnostics"))]
    fn live_for_tests(list: &str, elapsed: Duration, presented: u64) -> impl Drop {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                LIVE_FOR_TESTS.with(|l| *l.borrow_mut() = None);
                LIVE_PLAY_FOR_TESTS.with(|p| *p.borrow_mut() = None);
            }
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let presses = victory_gpu::scripted_keys::live_keys_script(list).expect("the fixture parses");
        let run = victory_gpu::KeysRun {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            start: Instant::now() - elapsed,
            count: presses.len(),
            presses: std::sync::Arc::from(presses),
        };
        LIVE_FOR_TESTS.with(|l| *l.borrow_mut() = Some(run));
        LIVE_PLAY_FOR_TESTS.with(|p| *p.borrow_mut() = None);
        PRESENT_FOR_TESTS.with(|p| p.set(Some(presented)));
        Reset
    }

    /// Tests: move the supplied live run's clock back, `elapsed` in all.
    #[cfg(all(test, feature = "diagnostics"))]
    fn live_elapsed_for_tests(elapsed: Duration) {
        let start = Instant::now() - elapsed;
        LIVE_FOR_TESTS.with(|l| {
            if let Some(run) = l.borrow_mut().as_mut() {
                run.start = start;
            }
        });
    }

    /// How long ago the marker `path` first existed, checked at most five times a second: `None` until it does.
    #[cfg(feature = "diagnostics")]
    fn marker_elapsed(path: &std::path::Path) -> Option<Duration> {
        use std::sync::Mutex;
        use std::time::Instant;

        struct Clock {
            seen: Option<Instant>,
            last_check: Option<Instant>,
        }
        static CLOCK: Mutex<Clock> = Mutex::new(Clock {
            seen: None,
            last_check: None,
        });
        const CHECK_EVERY: Duration = Duration::from_millis(200);
        let mut clock = CLOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if clock.seen.is_none()
            && clock
                .last_check
                .is_none_or(|last| last.elapsed() >= CHECK_EVERY)
        {
            clock.last_check = Some(Instant::now());
            if path.exists() {
                clock.seen = Some(Instant::now());
            }
        }
        clock.seen.map(|seen| seen.elapsed())
    }

    #[cfg(feature = "diagnostics")]
    fn warn_relative_without_marker(entries: usize) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "VICTORY_KEYS has {entries} '+'-relative entries but VICTORY_KEYS_AFTER is not set: they are skipped"
            );
        }
    }

    /// The foreground window is one of ours (the title's or our fallback).
    fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }

    /// `+5:S:90f` is a hold counted in presented frames, starting five seconds after the marker: parsed, not
    /// skipped. A malformed frame count (`90x`, `f`, `0f`, a negative one) is skipped as a malformed entry is.
    #[test]
    fn a_hold_in_presented_frames_parses() {
        assert_eq!(
            key_script("+5:S:90f").relative,
            [ScriptedKey {
                at: Duration::from_secs(5),
                hold: Hold::Frames(90),
                key: 0x53,
            }],
            "+5:S:90f parses as 90 presented frames from five seconds after the marker"
        );
        let script = key_script("+5:S:90x,+5:S:f,+5:S:0f,+5:S:-5f");
        assert!(script.relative.is_empty(), "a malformed frame hold is skipped");
    }

    /// A hold counting the title's presents stays down while fewer than its count have been presented since the
    /// press, however much time passes, and is released at the first poll after the last of them, however little
    /// time passes. The poll that first finds its start reached captures the present count it counts from.
    #[test]
    fn a_hold_in_presented_frames_ends_with_the_presents_not_with_time() {
        let keys = key_script("5:W:90f").absolute;
        let key = |now: Duration, presents: Option<u64>, holds: &mut Vec<FrameHold>| keys_down_with(&keys, now, presents, holds);
        let mut holds = Vec::new();
        assert_eq!(key(Duration::from_secs(4), Some(1_000), &mut holds), Vec::<u8>::new(), "before its start");
        assert_eq!(holds, [FrameHold::default()], "nothing captured before the start");
        assert_eq!(key(Duration::from_secs(5), Some(1_000), &mut holds), vec![0x57], "pressed at its start");
        assert_eq!(holds[0].pressed_at, Some(1_000), "the present it counts from is the one at its press");
        // Eighty-nine presents in, and an hour of time on top of the press: still down.
        assert_eq!(key(Duration::from_secs(3_600), Some(1_089), &mut holds), vec![0x57], "89 of 90 presents");
        // The first poll after the 90th present, a moment later: up.
        assert_eq!(key(Duration::from_millis(3_600_001), Some(1_090), &mut holds), Vec::<u8>::new(), "the 90th present ends it");
    }

    /// With no present yet the count cannot be read: a frame hold is one press, said in the log once, never a key
    /// held for ever.
    #[test]
    fn a_frame_hold_with_no_frame_presented_is_one_press_and_says_so_once() {
        let keys = key_script("5:W:90f").absolute;
        let mut holds = Vec::new();
        assert_eq!(keys_down_with(&keys, Duration::from_secs(5), None, &mut holds), vec![0x57]);
        assert!(holds[0].warned_no_count, "the fallback was said in the log");
        assert_eq!(keys_down_with(&keys, Duration::from_millis(5_199), None, &mut holds), vec![0x57]);
        assert_eq!(keys_down_with(&keys, Duration::from_millis(5_200), None, &mut holds), Vec::<u8>::new());
    }

    /// A count that arrives after the no-count fallback fired cannot press the entry again: the fallback is one
    /// press per entry, ever.
    #[test]
    fn a_count_arriving_after_the_fallback_fired_does_not_press_again() {
        let keys = key_script("0:W:90f").absolute;
        let mut holds = Vec::new();
        // No count yet: pressed once, for the short press, and said.
        assert_eq!(keys_down_with(&keys, Duration::ZERO, None, &mut holds), vec![0x57]);
        assert!(holds[0].warned_no_count);
        assert!(holds[0].fired_without_count);
        assert_eq!(keys_down_with(&keys, Duration::from_millis(200), None, &mut holds), Vec::<u8>::new());
        // A count arrives: the entry has fired, and stays up.
        assert_eq!(keys_down_with(&keys, Duration::from_millis(300), Some(1), &mut holds), Vec::<u8>::new(), "no second press");
        assert_eq!(keys_down_with(&keys, Duration::from_millis(400), Some(2), &mut holds), Vec::<u8>::new());
    }

    /// A first due poll after the short press window, with no count: nothing is pressed, and the entry counts as
    /// fired.
    #[test]
    fn a_late_first_poll_with_no_count_presses_nothing() {
        let keys = key_script("0:W:90f").absolute;
        let mut holds = Vec::new();
        assert_eq!(keys_down_with(&keys, Duration::from_secs(2), None, &mut holds), Vec::<u8>::new(), "no press, and none claimed");
        assert!(holds[0].warned_no_count);
        assert!(holds[0].fired_without_count);
        assert_eq!(keys_down_with(&keys, Duration::from_secs(3), Some(1), &mut holds), Vec::<u8>::new(), "late count: still no press");
    }

    /// A count that stops advancing releases the hold after thirty seconds of poll time, with one warning, and
    /// the key never comes back.
    #[test]
    fn a_stalled_count_releases_the_frame_hold_and_says_so_once() {
        let keys = key_script("5:W:90f").absolute;
        let mut holds = Vec::new();
        assert_eq!(keys_down_with(&keys, Duration::from_secs(5), Some(1_000), &mut holds), vec![0x57], "pressed");
        assert_eq!(keys_down_with(&keys, Duration::from_secs(6), Some(1_005), &mut holds), vec![0x57], "advancing");
        // No present for thirty seconds, the count stuck at 1,005: released, and said once.
        assert_eq!(keys_down_with(&keys, Duration::from_secs(36), Some(1_005), &mut holds), Vec::<u8>::new(), "released by the stall");
        assert!(holds[0].warned_stall, "the stall was said in the log");
        assert_eq!(keys_down_with(&keys, Duration::from_secs(37), Some(1_010), &mut holds), Vec::<u8>::new(), "never held again");
    }

    /// One input poll samples the script once: the pad's sample and the keyboard's read that same evaluation, so a
    /// present crossing the hold's release threshold between them in one poll cannot leave the pad down and the
    /// keyboard up. Through the production sampling path: a fake pad in slot 0, a script the test supplies, and a
    /// present count the test sets.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn one_poll_reads_one_evaluation_across_the_release_threshold() {
        // Enter held for five of the title's presents from the start.
        let _seam = scripted_for_tests("0:Enter:5f", 100);
        let mut host = Host::default();
        // Poll 1: pressed, on both views.
        let pad = host.pad(0).expect("the fake pad is connected");
        assert_ne!(pad.buttons & A, 0, "the pad view is down");
        assert!(host.keys().iter().any(|k| k.virtual_key == 0x0D), "the keyboard view is down");
        // Poll 2: the pad samples at 104 presents (four of five), the game presents once (105, the release) before
        // the keyboard's sample in the same poll; both read the pad's evaluation.
        set_presented_for_tests(104);
        let pad = host.pad(0).expect("the fake pad is connected");
        assert_ne!(pad.buttons & A, 0, "the pad is down at 104");
        set_presented_for_tests(105);
        assert!(
            host.keys().iter().any(|k| k.virtual_key == 0x0D),
            "the keyboard reads the pad's evaluation of 104, not its own at 105"
        );
        // Poll 3: past the threshold on both, whatever order they sample in.
        let pad = host.pad(0).expect("the fake pad is connected");
        assert_eq!(pad.buttons & A, 0, "up at 105");
        assert!(!host.keys().iter().any(|k| k.virtual_key == 0x0D), "up on the keyboard too");
    }

    /// A live sequence plays through the same evaluation as `VICTORY_KEYS`:
    /// the pad and the keyboard see its press, its times count from the run's
    /// start, and its one-second hold is over two seconds in. No live run
    /// leaves both views exactly as the launch script alone has them.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_live_sequence_plays_through_the_scripted_path_and_ends() {
        let _seam = scripted_for_tests("", 100);
        let live = live_for_tests("0:X:1", Duration::from_millis(100), 100);
        let mut host = Host::default();
        let pad = host.pad(0).expect("the fake pad is connected");
        assert_ne!(pad.buttons & X, 0, "the pad sees the live key");
        assert!(host.keys().iter().any(|k| k.virtual_key == 0x58), "the keyboard sees it");
        live_elapsed_for_tests(Duration::from_secs(2));
        let pad = host.pad(0).expect("the fake pad is connected");
        assert_eq!(pad.buttons & X, 0, "up after its hold");
        assert!(!host.keys().iter().any(|k| k.virtual_key == 0x58), "up on the keyboard too");
        drop(live);
        assert!(!host.keys().iter().any(|k| k.virtual_key == 0x58), "no run, no press");
    }

    #[test]
    fn a_held_key_is_down_across_its_hold_and_up_after_it() {
        let keys = key_script("300:W:20").absolute;
        assert_eq!(
            keys,
            [ScriptedKey {
                at: Duration::from_secs(300),
                hold: Hold::Time(Duration::from_secs(20)),
                key: 0x57,
            }]
        );
        assert_eq!(keys_down(&keys, Duration::from_secs(299)), Vec::<u8>::new());
        assert_eq!(keys_down(&keys, Duration::from_secs(300)), vec![0x57]);
        assert_eq!(keys_down(&keys, Duration::from_secs(319)), vec![0x57]);
        assert_eq!(keys_down(&keys, Duration::from_secs(320)), Vec::<u8>::new());
    }

    #[test]
    fn two_holds_that_overlap_are_both_down() {
        // W is held 300 to 325, A 321 to 326.
        let keys = key_script("300:W:25,321:A:5").absolute;
        assert_eq!(keys_down(&keys, Duration::from_secs(310)), vec![0x57]);
        assert_eq!(keys_down(&keys, Duration::from_secs(322)), vec![0x57, 0x41]);
        assert_eq!(keys_down(&keys, Duration::from_secs(325)), vec![0x41]);
        assert_eq!(keys_down(&keys, Duration::from_secs(326)), Vec::<u8>::new());
    }

    #[test]
    fn an_entry_without_a_hold_is_down_for_a_fifth_of_a_second() {
        let keys = key_script("30:Enter").absolute;
        assert_eq!(
            keys,
            [ScriptedKey {
                at: Duration::from_secs(30),
                hold: Hold::Time(KEY_PRESS),
                key: 0x0D,
            }]
        );
        assert_eq!(keys_down(&keys, Duration::from_millis(30_199)), vec![0x0D]);
        assert_eq!(keys_down(&keys, Duration::from_millis(30_200)), Vec::<u8>::new());
    }

    #[test]
    fn malformed_entries_are_skipped_and_the_rest_kept() {
        let keys = key_script("300:W:20,nope,310:Q:-1,320:A:notanumber,330:B:601,340:C:0,350:!:5,,355:D").absolute;
        // Kept: 300:W:20, 340:C:0 (a hold of zero), 355:D.
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0].key, 0x57);
        assert_eq!(keys[1].key, 0x43);
        assert_eq!(keys[1].hold, Hold::Time(Duration::ZERO));
        assert_eq!(keys[2].key, 0x44);
        assert_eq!(keys[2].hold, Hold::Time(KEY_PRESS));
    }

    #[test]
    fn plus_entries_count_from_the_marker_and_absolute_entries_from_the_start() {
        let script = key_script("30:W,+5:J:10,+20:A");
        assert_eq!(script.absolute.len(), 1);
        assert_eq!(script.relative.len(), 2);
        // Before the marker exists, only the absolute entry counts, from input start.
        assert_eq!(keys_down_at(&script, Duration::from_secs(30), None), vec![0x57]);
        assert_eq!(keys_down_at(&script, Duration::from_secs(31), None), Vec::<u8>::new());
        // Five seconds after the marker, J is down.
        assert_eq!(
            keys_down_at(&script, Duration::from_secs(0), Some(Duration::from_secs(5))),
            vec![0x4A]
        );
        // Fifteen seconds after the marker, J's hold is over and A is not due yet.
        assert_eq!(
            keys_down_at(&script, Duration::from_secs(0), Some(Duration::from_secs(15))),
            Vec::<u8>::new()
        );
        // Twenty seconds after the marker, A is down.
        assert_eq!(
            keys_down_at(&script, Duration::from_secs(0), Some(Duration::from_secs(20))),
            vec![0x41]
        );
        // Both clocks count at once: W at 30 s of input and A at 20 s after the marker.
        assert_eq!(
            keys_down_at(&script, Duration::from_secs(30), Some(Duration::from_secs(20))),
            vec![0x57, 0x41]
        );
    }

    #[test]
    fn a_plus_entry_parses_like_an_absolute_one_and_malformed_ones_are_skipped() {
        let script = key_script("+,+30:Enter:1,+5:!,nope,+:W");
        assert_eq!(script.absolute, Vec::<ScriptedKey>::new());
        assert_eq!(
            script.relative,
            [ScriptedKey {
                at: Duration::from_secs(30),
                hold: Hold::Time(Duration::from_secs(1)),
                key: 0x0D,
            }]
        );
        // The plain list keeps its old meaning: a `+` entry is not an absolute one.
        assert_eq!(key_script("+5:J").absolute, Vec::<ScriptedKey>::new());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))));
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(&hub.object, std::ptr::null(), GAMEPAD_KIND, 1, 0, context, Some(slow), &mut token),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }
}
