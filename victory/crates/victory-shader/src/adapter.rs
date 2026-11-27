//! Which graphics adapter a host device is made on, and the line that says so: one rule for
//! the runtime's device (`victory-gpu`'s host) and for every test host (`victory-gpu`'s and
//! [`crate::emit::host`]'s), kept here because both crates can use it.
//!
//! Before it, devices were made on Windows' default adapter, the one its list of adapters
//! starts with. On a laptop with an integrated adapter and a card that is the integrated
//! one, and every "real card" test there ran on it without anyone knowing.
//!
//! The rule ([`choose`]): by default the hardware adapter Windows ranks first for high
//! performance (`IDXGIFactory6::EnumAdapterByGpuPreference`), software adapters skipped;
//! Windows' default adapter, as before, where that interface is missing or ranks no hardware
//! adapter. `VICTORY_ADAPTER` ([`SWITCH`]) overrides it:
//!
//! - `default`: Windows' default adapter, as before the rule;
//! - `high-performance`, `minimum-power`: the first hardware adapter Windows ranks for that;
//! - any other text: the first hardware adapter, in Windows' order, whose description
//!   contains it, letter case aside (never the software adapter, which is the software
//!   switches'; as job/wave-meet's shader host takes `VICTORY_ADAPTER=<text>`).
//!
//! Where no adapter matches (or Windows ranks none for the two above) no device is made, and
//! the message lists the adapters' descriptions. A software switch wins over the switch and
//! keeps its meaning: `VICTORY_D3D12_WARP` in the runtime (diagnostics builds),
//! `VICTORY_WARP=1` in the shader test host, and the test hosts that ask for the software
//! adapter themselves ([`TEST_SOFTWARE`]).
//!
//! A hardware adapter is taken only where it can run the host ([`BASELINE`], [`settle`]):
//! where the default rule's ranked adapter cannot, Windows' default adapter is tried once,
//! and the line says so and why; an adapter `VICTORY_ADAPTER` chose is refused instead.
//!
//! What was used is said in one line ([`line_for`]), the adapter found by the device's own
//! LUID: the runtime logs it when it makes its device, and a test host prints it once a
//! process ([`say_once`]), so a gate's output shows the adapter. `VICTORY_REQUIRE_GPU`
//! ([`REQUIRE`], [`requirement`]) makes a test host that asked for hardware fail, not skip,
//! where it has none, none of the kind named, or one whose adapter DXGI does not describe
//! ([`verdict`]); a test host that asked for the software adapter itself is not held to it.

/// The switch that overrides the rule.
pub const SWITCH: &str = "VICTORY_ADAPTER";

/// The switch by which a gate requires hardware of a test host.
pub const REQUIRE: &str = "VICTORY_REQUIRE_GPU";

/// How a test host that asks for the software adapter itself says it chose it.
pub const TEST_SOFTWARE: &str = "the test's request for the software adapter";

/// An adapter as DXGI describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    pub description: String,
    pub vendor: u32,
    pub device: u32,
    /// Its LUID, the high part above the low part.
    pub luid: u64,
    /// Flagged software, or Microsoft's Basic Render Driver (vendor 0x1414, device 0x8c).
    pub software: bool,
    /// Dedicated video memory, bytes.
    pub dedicated: u64,
    /// The user-mode driver's version, four 16-bit parts from the highest, where DXGI says.
    pub driver: Option<u64>,
}

/// The adapters there are, in Windows' order (`IDXGIFactory1::EnumAdapters1`, whose first is
/// Windows' default adapter, the one `D3D12CreateDevice` takes without one), and how Windows
/// ranks them for high performance and for minimum power: indices into `adapters`, `None`
/// without `IDXGIFactory6`.
#[derive(Clone, Debug, Default)]
pub struct Listing {
    pub adapters: Vec<Adapter>,
    pub high_performance: Option<Vec<usize>>,
    pub minimum_power: Option<Vec<usize>>,
}

/// The adapter a device is to be made on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Windows' default adapter: none is passed to `D3D12CreateDevice` (as before the rule).
    Windows,
    /// `Listing::adapters[i]`.
    Listed(usize),
    /// The software adapter (`IDXGIFactory4::EnumWarpAdapter`).
    Software,
}

/// What [`choose`] picked, and how, for the line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub pick: Pick,
    pub how: String,
    /// What follows where the picked adapter cannot run the host ([`settle`]).
    pub on_fail: OnFail,
}

/// What follows where the picked adapter cannot run the host ([`BASELINE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnFail {
    /// The default rule's ranked adapter: Windows' default adapter is tried, once.
    Windows,
    /// `VICTORY_ADAPTER`'s: refused, with why ([`Unmade::Refused`]); no other is tried.
    Refuse,
    /// Windows' default adapter by the default rule, or the software adapter: no device
    /// ([`Unmade::Failed`]), as before the rule.
    NoDevice,
}

/// What the host needs of a hardware device, its baseline: one made at feature level 12_0
/// that supports shader model 6.6 (every translated shader is, [`crate::emit`]). The software
/// adapter is not held to it here (the switches' and the tests' own request; the shader host
/// asks it of every device it makes).
pub const BASELINE: &str = "feature level 12_0 and shader model 6.6";

/// The shader model the host needs (`D3D_SHADER_MODEL_6_6`).
pub const SHADER_MODEL: i32 = 0x66;

/// Why a device whose highest shader model is `model` (`D3D_SHADER_MODEL`, `0x65` for 6.5)
/// cannot run the host; `None`: it can.
pub fn below_baseline(model: i32) -> Option<String> {
    (model < SHADER_MODEL).then(|| {
        format!(
            "shader model {}.{} only, the host needs 6.6",
            model >> 4,
            model & 0xf
        )
    })
}

/// Why no device was made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmade {
    /// `VICTORY_ADAPTER` names no adapter there is, a ranking Windows does not give, or an
    /// adapter that cannot run the host: none is made (the runtime does not start; a test
    /// fails), and the message lists the adapters' descriptions.
    Refused(String),
    /// DXGI or D3D12 failed, the software adapter is not one, or the default rule found no
    /// adapter that can run the host: there is no device (the runtime goes on without one, a
    /// test skips), as before the rule.
    Failed(String),
}

impl std::fmt::Display for Unmade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unmade::Refused(text) | Unmade::Failed(text) => f.write_str(text),
        }
    }
}

/// The rule (the module's documentation). `switch`: `VICTORY_ADAPTER`'s value, if set;
/// `software`: the software switch in effect, named, which wins. `Err`: the switch names no
/// adapter there is, and the message lists them; no device is to be made.
pub fn choose(
    listing: &Listing,
    switch: Option<&str>,
    software: Option<&str>,
) -> Result<Choice, String> {
    let value = switch.map(str::trim).filter(|v| !v.is_empty());
    if let Some(why) = software {
        let how = match value {
            Some(value) => format!("{why}, over {SWITCH}={value}"),
            None => why.to_owned(),
        };
        return Ok(Choice {
            pick: Pick::Software,
            how,
            on_fail: OnFail::NoDevice,
        });
    }
    let Some(value) = value else {
        let ranked = listing.high_performance.as_deref();
        let (pick, how, on_fail) = match first_hardware(listing, ranked) {
            Some(i) => (
                Pick::Listed(i),
                "the default rule (the first hardware adapter for high performance)",
                OnFail::Windows,
            ),
            None if ranked.is_none() => (
                Pick::Windows,
                "the default rule (no IDXGIFactory6: Windows' default adapter)",
                OnFail::NoDevice,
            ),
            None => (
                Pick::Windows,
                "the default rule (no hardware adapter ranked: Windows' default adapter)",
                OnFail::NoDevice,
            ),
        };
        return Ok(Choice {
            pick,
            how: how.to_owned(),
            on_fail,
        });
    };
    let how = format!("{SWITCH}={value}");
    let part = value.to_lowercase();
    let pick = match part.as_str() {
        "default" => Ok(Pick::Windows),
        "high-performance" => by_rank(
            listing,
            listing.high_performance.as_deref(),
            "high performance",
        ),
        "minimum-power" => by_rank(listing, listing.minimum_power.as_deref(), "minimum power"),
        _ => listing
            .adapters
            .iter()
            .position(|a| !a.software && a.description.to_lowercase().contains(&part))
            .map(Pick::Listed)
            .ok_or_else(|| format!("no hardware adapter's description contains \"{value}\"")),
    };
    match pick {
        Ok(pick) => Ok(Choice {
            pick,
            how,
            on_fail: OnFail::Refuse,
        }),
        Err(why) => Err(format!("{how}: {why}; {}", names(listing))),
    }
}

/// The choice made good: `make(pick)` gives a device on that adapter that can run the host,
/// or why it cannot ([`BASELINE`]); where it cannot, [`Choice::on_fail`] says what follows,
/// and the default rule's second try is Windows' default adapter, once. `Ok`: the device and
/// how it was chosen, a second try said in it.
pub fn settle<D>(
    listing: &Listing,
    choice: Choice,
    mut make: impl FnMut(Pick) -> Result<D, String>,
) -> Result<(D, String), Unmade> {
    let first = match make(choice.pick) {
        Ok(device) => return Ok((device, choice.how)),
        Err(why) => why,
    };
    let what = named(listing, choice.pick);
    match choice.on_fail {
        OnFail::NoDevice => Err(Unmade::Failed(first)),
        OnFail::Refuse => Err(Unmade::Refused(format!(
            "{}: {what} cannot run the host ({first}); no other adapter is tried for an adapter {SWITCH} chose; {}",
            choice.how,
            names(listing)
        ))),
        OnFail::Windows => match make(Pick::Windows) {
            Ok(device) => Ok((
                device,
                format!(
                    "the default rule (fell back to Windows' default adapter: the first hardware adapter for high performance, {what}, cannot run the host: {first})"
                ),
            )),
            Err(again) => Err(Unmade::Failed(format!(
                "the first hardware adapter for high performance, {what}, cannot run the host ({first}), nor can Windows' default adapter ({again})"
            ))),
        },
    }
}

/// `pick`'s adapter, for a message.
fn named(listing: &Listing, pick: Pick) -> String {
    match pick {
        Pick::Listed(i) => listing.adapters.get(i).map_or_else(
            || format!("adapter {i}"),
            |a| format!("\"{}\"", a.description),
        ),
        Pick::Windows => "Windows' default adapter".to_owned(),
        Pick::Software => "the software adapter".to_owned(),
    }
}

/// The first adapter in `ranked` that is not a software one.
fn first_hardware(listing: &Listing, ranked: Option<&[usize]>) -> Option<usize> {
    ranked?
        .iter()
        .copied()
        .find(|&i| listing.adapters.get(i).is_some_and(|a| !a.software))
}

fn by_rank(listing: &Listing, ranked: Option<&[usize]>, what: &str) -> Result<Pick, String> {
    match ranked {
        None => Err(format!(
            "Windows ranks no adapters for {what} here (no IDXGIFactory6)"
        )),
        Some(_) => first_hardware(listing, ranked)
            .map(Pick::Listed)
            .ok_or_else(|| format!("Windows ranks no hardware adapter for {what}")),
    }
}

/// The adapters, for a refusal: each with its place in Windows' order (`#0` first; an
/// adapter only the ranking gave comes last) and what the default rule and a name do with
/// it, so that entries of one description are told apart. The LUID stays out: internal.
fn names(listing: &Listing) -> String {
    if listing.adapters.is_empty() {
        return "the adapters: none listed".to_owned();
    }
    let ranked = listing.high_performance.as_deref();
    let pick = first_hardware(listing, ranked);
    let entries: Vec<String> = listing
        .adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut notes = Vec::new();
            if a.software {
                notes.push("software: the default rule and a name skip it".to_owned());
            } else {
                match ranked.map(|r| r.iter().position(|&j| j == i)) {
                    Some(Some(at)) if pick == Some(i) => notes.push(format!(
                        "high-performance rank {}: the default rule's pick",
                        at + 1
                    )),
                    Some(Some(at)) => notes.push(format!("high-performance rank {}", at + 1)),
                    Some(None) => notes.push(
                        "not ranked for high performance: the default rule never takes it"
                            .to_owned(),
                    ),
                    None => {}
                }
                let lower = a.description.to_lowercase();
                if let Some(first) = listing.adapters[..i]
                    .iter()
                    .position(|b| !b.software && b.description.to_lowercase() == lower)
                {
                    notes.push(format!(
                        "the description of #{first}, which a name takes first"
                    ));
                }
            }
            if notes.is_empty() {
                format!("#{i} \"{}\"", a.description)
            } else {
                format!("#{i} \"{}\" ({})", a.description, notes.join("; "))
            }
        })
        .collect();
    format!("the adapters, in Windows' order: {}", entries.join(", "))
}

/// The line that says which adapter a device is on and how it was chosen:
/// `host adapter: <description> (vendor 0x<vvvv>, device 0x<dddd>, driver <a.b.c.d>,
/// <n> MiB dedicated), chosen by <how>`, the driver `unknown` where DXGI does not say it.
pub fn line(adapter: &Adapter, how: &str) -> String {
    let driver = match adapter.driver {
        Some(v) => format!(
            "{}.{}.{}.{}",
            v >> 48,
            (v >> 32) & 0xffff,
            (v >> 16) & 0xffff,
            v & 0xffff
        ),
        None => "unknown".to_owned(),
    };
    format!(
        "host adapter: {} (vendor {:#06x}, device {:#06x}, driver {driver}, {} MiB dedicated), chosen by {how}",
        adapter.description,
        adapter.vendor,
        adapter.device,
        adapter.dedicated >> 20,
    )
}

/// [`line`] for the adapter a device's LUID names, or, where DXGI does not describe it:
/// `host adapter: unknown (DXGI does not describe the device's adapter), chosen by <how>`.
pub fn line_for(adapter: Option<&Adapter>, how: &str) -> String {
    match adapter {
        Some(adapter) => line(adapter, how),
        None => format!(
            "host adapter: unknown (DXGI does not describe the device's adapter), chosen by {how}"
        ),
    }
}

/// What a gate requires of a test host that asks for hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Requirement {
    /// A hardware adapter (`VICTORY_REQUIRE_GPU=1`).
    Hardware,
    /// A hardware adapter whose description contains this, letter case aside
    /// (`VICTORY_REQUIRE_GPU=nvidia`).
    Named(String),
}

/// `VICTORY_REQUIRE_GPU`'s value as a requirement: unset, empty or `0` none.
pub fn requirement(value: Option<&str>) -> Option<Requirement> {
    match value.map(str::trim) {
        None | Some("" | "0") => None,
        Some("1") => Some(Requirement::Hardware),
        Some(part) => Some(Requirement::Named(part.to_owned())),
    }
}

/// What a test host got, for the requirement: the adapter a device it made is on, by the
/// device's own LUID.
#[derive(Clone, Copy, Debug)]
pub enum Got<'a> {
    /// No device was made.
    Nothing,
    /// A device was made, and DXGI does not describe the adapter its LUID names: it is not
    /// known to be hardware.
    Undescribed,
    /// A device was made on this adapter.
    On(&'a Adapter),
}

impl<'a> From<Option<&'a Adapter>> for Got<'a> {
    /// A device made: on the adapter its LUID names, `None` where DXGI does not describe it.
    fn from(on: Option<&'a Adapter>) -> Got<'a> {
        on.map_or(Got::Undescribed, Got::On)
    }
}

/// Whether what a test host made for hardware meets `required`. `Err`: why the test is to
/// fail. An adapter DXGI does not describe never does: hardware is identified, not assumed.
pub fn meets(required: &Requirement, got: Got<'_>) -> Result<(), String> {
    let switch = match required {
        Requirement::Hardware => format!("{REQUIRE}=1"),
        Requirement::Named(part) => format!("{REQUIRE}={part}"),
    };
    let on = match got {
        Got::Nothing => {
            return Err(format!(
                "{switch} and no hardware device was made: the test did not run"
            ));
        }
        Got::Undescribed => {
            return Err(format!(
                "{switch} and DXGI does not describe the device's adapter: it is not known to be hardware, and the test did not run on it"
            ));
        }
        Got::On(on) => on,
    };
    if on.software {
        return Err(format!(
            "{switch} and the device is the software adapter ({}): the test did not run on hardware",
            on.description
        ));
    }
    match required {
        Requirement::Named(part)
            if !on.description.to_lowercase().contains(&part.to_lowercase()) =>
        {
            Err(format!(
                "{switch} and the device is on {}, whose description does not contain \"{part}\"",
                on.description
            ))
        }
        _ => Ok(()),
    }
}

/// `VICTORY_REQUIRE_GPU`'s requirement as it is set.
pub fn required() -> Option<Requirement> {
    requirement(std::env::var(REQUIRE).ok().as_deref())
}

/// [`meets`] with `VICTORY_REQUIRE_GPU` as it is set; `Ok` where it is not.
pub fn require(got: Got<'_>) -> Result<(), String> {
    match required() {
        Some(required) => meets(&required, got),
        None => Ok(()),
    }
}

/// What a test host asked for: kept apart from what a switch then chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    /// Hardware: the rule's adapter, or the software adapter where a software switch says so.
    /// `VICTORY_REQUIRE_GPU` holds for it.
    Hardware,
    /// The software adapter itself (a fixture that must not reach a card): no switch moves
    /// it, and `VICTORY_REQUIRE_GPU` does not hold for it.
    Software,
}

/// The software request [`choose`] is given for a test host: the test's own where it asked
/// for the software adapter ([`TEST_SOFTWARE`]), else the software switch in effect, named.
pub fn software_for(asked: Asked, switch: Option<&str>) -> Option<&str> {
    match asked {
        Asked::Software => Some(TEST_SOFTWARE),
        Asked::Hardware => switch,
    }
}

/// What a test host does with what it got ([`verdict`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The test runs on the device.
    Run,
    /// The test skips (no device of the kind asked here), saying why.
    Skip(String),
    /// The test fails, saying why.
    Fail(String),
}

/// What a test host does: `asked`, what it asked for; `required`, `VICTORY_REQUIRE_GPU`'s
/// requirement; `switched`, a software switch was in effect for it; `made`, the adapter the
/// device made is on (by its LUID, `None` where DXGI does not describe it), or why none was.
///
/// - A refusal (`VICTORY_ADAPTER`) fails, whatever was asked.
/// - Asked for the software adapter: it runs where the device is on it, skips where there is
///   none; the requirement is not its own.
/// - Asked for hardware, with the requirement: it runs only where the device is on hardware
///   of the kind required, identified; no device, the software adapter (a switch's too) or an
///   adapter not described fails.
/// - Asked for hardware, without it: it runs, on the software adapter too where a switch put
///   it there (the line says so); skips where there is no device, or where the rule's adapter
///   is the software one without a switch (a machine without a card).
pub fn verdict(
    asked: Asked,
    required: Option<&Requirement>,
    switched: bool,
    made: Result<Option<&Adapter>, &Unmade>,
) -> Verdict {
    let why = match made {
        Err(Unmade::Refused(text)) => return Verdict::Fail(text.clone()),
        Err(Unmade::Failed(why)) => Some(why),
        Ok(_) => None,
    };
    match (asked, required) {
        (Asked::Software, _) => match made {
            Ok(Some(on)) if !on.software => Verdict::Fail(format!(
                "the test asked for the software adapter, and the device is on {}",
                on.description
            )),
            Ok(_) => Verdict::Run,
            Err(_) => Verdict::Skip(why.cloned().unwrap_or_default()),
        },
        (Asked::Hardware, Some(required)) => {
            let got = match made {
                Ok(on) => Got::from(on),
                Err(_) => Got::Nothing,
            };
            match meets(required, got) {
                Ok(()) => Verdict::Run,
                Err(failed) => Verdict::Fail(match why {
                    Some(why) => format!("{failed} ({why})"),
                    None => failed,
                }),
            }
        }
        (Asked::Hardware, None) => match made {
            Ok(Some(on)) if on.software && !switched => Verdict::Skip(
                "the chosen adapter is the software adapter: no hardware device here".to_owned(),
            ),
            Ok(_) => Verdict::Run,
            Err(_) => Verdict::Skip(why.cloned().unwrap_or_default()),
        },
    }
}

#[cfg(all(windows, any(test, feature = "adapter")))]
pub use live::{
    Made, device_adapter, listing, make_device, say_once, software_adapter, test_device,
};

/// The rule on the adapters there are: DXGI's listing, and the device made where it picks.
#[cfg(all(windows, any(test, feature = "adapter")))]
mod live {
    use std::io::Write;
    use std::sync::Mutex;

    use windows::Win32::Foundation::LUID;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::{
        D3D_SHADER_MODEL_6_6, D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL,
        D3D12CreateDevice, ID3D12Device,
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
        DXGI_GPU_PREFERENCE, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        DXGI_GPU_PREFERENCE_MINIMUM_POWER, IDXGIAdapter1, IDXGIDevice, IDXGIFactory4,
        IDXGIFactory6,
    };
    use windows::core::Interface;

    use super::{
        Adapter, Asked, Listing, Pick, SWITCH, Unmade, Verdict, below_baseline, choose, required,
        settle, software_for, verdict,
    };

    /// A device made on the adapter the rule picked: the adapter it is on, found by the
    /// device's own LUID (`None` where DXGI does not describe it), and how it was chosen.
    pub struct Made {
        pub device: ID3D12Device,
        pub adapter: Option<Adapter>,
        pub how: String,
    }

    impl Made {
        /// The line that says so ([`super::line_for`]).
        pub fn line(&self) -> String {
            super::line_for(self.adapter.as_ref(), &self.how)
        }
    }

    fn luid(luid: LUID) -> u64 {
        (u64::from(luid.HighPart as u32) << 32) | u64::from(luid.LowPart)
    }

    /// `adapter` as DXGI describes it, without its driver's version ([`driver`]).
    fn described(adapter: &IDXGIAdapter1) -> windows::core::Result<Adapter> {
        // SAFETY: a plain query of a live adapter.
        let desc = unsafe { adapter.GetDesc1() }?;
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        Ok(Adapter {
            description: String::from_utf16_lossy(&desc.Description[..len]),
            vendor: desc.VendorId,
            device: desc.DeviceId,
            luid: luid(desc.AdapterLuid),
            software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c),
            dedicated: desc.DedicatedVideoMemory as u64,
            driver: None,
        })
    }

    /// The user-mode driver's version, as DXGI gives it for `IDXGIDevice`. Asked of the
    /// adapter a device is made on only.
    fn driver(adapter: &IDXGIAdapter1) -> Option<u64> {
        // SAFETY: a plain query of a live adapter, with an interface id of a static.
        unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
            .ok()
            .map(|v| v as u64)
    }

    /// The order Windows ranks the adapters in for `preference`, as indices into `listing`
    /// (one it did not list before is added).
    fn ranked(
        factory: &IDXGIFactory6,
        preference: DXGI_GPU_PREFERENCE,
        listing: &mut Listing,
        handles: &mut Vec<IDXGIAdapter1>,
    ) -> Vec<usize> {
        let mut order = Vec::new();
        for i in 0..64 {
            // SAFETY: plain DXGI enumeration; the end of the list is an error.
            let Ok(adapter) =
                (unsafe { factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, preference) })
            else {
                break;
            };
            let Ok(found) = described(&adapter) else {
                continue;
            };
            let at = match listing.adapters.iter().position(|a| a.luid == found.luid) {
                Some(at) => at,
                None => {
                    listing.adapters.push(found);
                    handles.push(adapter);
                    listing.adapters.len() - 1
                }
            };
            order.push(at);
        }
        order
    }

    /// The adapters there are, as the rule takes them, and their DXGI objects (one per
    /// listed adapter). Nothing is opened: no device is made.
    pub fn listing() -> windows::core::Result<(Listing, Vec<IDXGIAdapter1>)> {
        // SAFETY: plain DXGI object creation and enumeration.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }?;
        let mut listing = Listing::default();
        let mut handles = Vec::new();
        for i in 0..64 {
            // SAFETY: as above; the end of the list is an error.
            let Ok(adapter) = (unsafe { factory.EnumAdapters1(i) }) else {
                break;
            };
            if let Ok(found) = described(&adapter) {
                listing.adapters.push(found);
                handles.push(adapter);
            }
        }
        if let Ok(factory) = factory.cast::<IDXGIFactory6>() {
            let high = ranked(
                &factory,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                &mut listing,
                &mut handles,
            );
            let low = ranked(
                &factory,
                DXGI_GPU_PREFERENCE_MINIMUM_POWER,
                &mut listing,
                &mut handles,
            );
            listing.high_performance = Some(high);
            listing.minimum_power = Some(low);
        }
        Ok((listing, handles))
    }

    /// The software adapter, refused where DXGI does not say it is one (`EnumWarpAdapter`
    /// may give hardware, notably under Proton).
    pub fn software_adapter() -> Result<(IDXGIAdapter1, Adapter), Unmade> {
        let failed = |e: windows::core::Error| Unmade::Failed(format!("no software adapter: {e}"));
        // SAFETY: plain DXGI queries.
        let adapter: IDXGIAdapter1 = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumWarpAdapter())
        }
        .map_err(failed)?;
        let found = described(&adapter).map_err(failed)?;
        if !found.software {
            return Err(Unmade::Failed(format!(
                "DXGI did not identify the WARP adapter as software (vendor {:#x}, device {:#x}); refusing device",
                found.vendor, found.device
            )));
        }
        Ok((adapter, found))
    }

    /// The adapter `device` is on, found by its LUID, with its driver's version.
    pub fn device_adapter(device: &ID3D12Device) -> Option<Adapter> {
        // SAFETY: plain DXGI queries of a live device's adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).ok()?;
            let mut found = described(&adapter).ok()?;
            found.driver = driver(&adapter);
            Some(found)
        }
    }

    /// Whether `device`, made at feature level 12_0, can run the host ([`super::BASELINE`]):
    /// its shader model is asked, and a query it does not answer is a no.
    fn baseline(device: &ID3D12Device) -> Result<(), String> {
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        // SAFETY: writes `model` of the given size.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut model).cast(),
                size_of_val(&model) as u32,
            )
        }
        .map_err(|e| format!("its device does not answer the shader model query: {e}"))?;
        below_baseline(model.HighestShaderModel.0).map_or(Ok(()), Err)
    }

    /// A device of feature level 12_0 on `pick`'s adapter that can run the host: a hardware
    /// one held to [`super::BASELINE`], the software adapter (checked to be one) not.
    fn made_on(pick: Pick, handles: &[IDXGIAdapter1]) -> Result<ID3D12Device, String> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: device creation on a live adapter, or on Windows' default one.
        let created = unsafe {
            match pick {
                Pick::Software => {
                    let (handle, _) = software_adapter().map_err(|e| e.to_string())?;
                    D3D12CreateDevice(&handle, D3D_FEATURE_LEVEL_12_0, &mut device)
                }
                Pick::Listed(i) => {
                    let handle = handles.get(i).ok_or("the adapter is not listed")?;
                    D3D12CreateDevice(handle, D3D_FEATURE_LEVEL_12_0, &mut device)
                }
                // D3D12 picks: what it took is known from the device's LUID only.
                Pick::Windows => D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device),
            }
        };
        created.map_err(|e| format!("D3D12CreateDevice: {e}"))?;
        let device = device.ok_or("D3D12CreateDevice made no device")?;
        if pick != Pick::Software {
            baseline(&device)?;
        }
        Ok(device)
    }

    /// A device on the adapter the rule picks, with `VICTORY_ADAPTER` as it is set, made
    /// good by [`settle`] (the default rule's second try, an explicit choice's refusal);
    /// `software`: the software request in effect, named. The adapter is the one the
    /// device's LUID names, whichever was asked for. A layer to be on for the device (debug,
    /// device-removed data) is switched on before.
    pub fn make_device(software: Option<&str>) -> Result<Made, Unmade> {
        // No listing (no DXGI factory): Windows' default adapter, as before.
        let (listing, handles) = listing().unwrap_or_default();
        let switch = std::env::var(SWITCH).ok();
        let choice = choose(&listing, switch.as_deref(), software).map_err(Unmade::Refused)?;
        let (device, how) = settle(&listing, choice, |pick| made_on(pick, &handles))?;
        let adapter = device_adapter(&device);
        Ok(Made {
            device,
            adapter,
            how,
        })
    }

    /// Prints `line` on the process's standard error, once a process: a test host's device,
    /// so that a gate's output shows the adapter. Written to the stream itself, not with
    /// `eprintln!`, whose output a passing test holds back.
    pub fn say_once(line: &str) {
        static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if said.iter().all(|s| s != line) {
            said.push(line.to_owned());
            let _ = writeln!(std::io::stderr().lock(), "{line}");
        }
    }

    /// A test host's device ([`make_device`]), its line said once a process ([`say_once`]).
    /// `asked`: what the test asked for; `switch`: the software switch in effect, named,
    /// which puts a test that asked for hardware on the software adapter (and leaves it a
    /// test that asked for hardware). Panics where [`verdict`] fails it, so that the test
    /// fails instead of skipping; `Err`: it skips, and why.
    pub fn test_device(asked: Asked, switch: Option<&str>) -> Result<Made, String> {
        let made = make_device(software_for(asked, switch));
        if let Ok(made) = &made {
            say_once(&made.line());
        }
        let on = made.as_ref().map(|made| made.adapter.as_ref());
        match verdict(asked, required().as_ref(), switch.is_some(), on) {
            Verdict::Run => made.map_err(|e| e.to_string()),
            Verdict::Skip(why) => Err(why),
            Verdict::Fail(why) => panic!("{why}"),
        }
    }

    /// A device made by the rule is on the adapter it was chosen for, and its line names
    /// that adapter as the device's own: on the software adapter with `VICTORY_WARP=1` (as
    /// the shader host), else on the one the rule picks here.
    #[test]
    fn a_device_is_on_the_adapter_its_line_names() {
        let software = std::env::var("VICTORY_WARP")
            .is_ok_and(|v| v == "1")
            .then_some("VICTORY_WARP=1");
        let made = match make_device(software) {
            Ok(made) => made,
            Err(Unmade::Failed(why)) => {
                eprintln!("no device here ({why}); skipped");
                return;
            }
            Err(refused) => panic!("{refused}"),
        };
        let line = made.line();
        say_once(&line);
        let own = device_adapter(&made.device).expect("the device's adapter, by its LUID");
        let chosen = made.adapter.as_ref().expect("the adapter described");
        assert_eq!(chosen.luid, own.luid, "{line}");
        assert_eq!(chosen.description, own.description);
        assert_eq!(
            line,
            super::line(&own, &made.how),
            "the line is the device's own adapter's"
        );
        assert!(
            line.starts_with(&format!("host adapter: {} (vendor ", own.description)),
            "{line}"
        );
        // The rule's own verdict for a request for hardware: on a machine without a card
        // (GitHub's runner) the rule's adapter is the software one, which is that request's
        // skip ([`verdict`]); everywhere else the device is on the software adapter only
        // where the switch put it there.
        if let Some(why) = machine_without_a_card(software.is_some(), &own) {
            eprintln!("{why}; the hardware-or-switch check skipped ({line})");
            return;
        }
        assert_eq!(software.is_some(), own.software, "{line}");
    }

    #[cfg(test)]
    /// Why the live check of the adapter's kind does not apply here: `Some` where no switch
    /// asked for the software adapter and the rule's adapter is it anyway, a machine without
    /// a card, by [`verdict`]'s skip for a request for hardware.
    fn machine_without_a_card(switched: bool, own: &Adapter) -> Option<String> {
        match verdict(Asked::Hardware, None, switched, Ok(Some(own))) {
            Verdict::Skip(why) => Some(why),
            Verdict::Run | Verdict::Fail(_) => None,
        }
    }

    #[test]
    fn only_a_machine_without_a_card_skips_the_live_kind_check() {
        let card = Adapter { software: false, ..software_listed() };
        let warp = software_listed();
        assert!(machine_without_a_card(false, &warp).is_some(), "no switch, the rule's adapter is software");
        assert_eq!(machine_without_a_card(true, &warp), None, "the switch put it there: checked");
        assert_eq!(machine_without_a_card(false, &card), None, "a card: checked");
        assert_eq!(machine_without_a_card(true, &card), None, "a card under the switch: checked (and fails)");
    }

    #[cfg(test)]
    fn software_listed() -> Adapter {
        Adapter {
            description: "Microsoft Basic Render Driver".to_owned(),
            vendor: 0x1414,
            device: 0x8c,
            luid: 1,
            software: true,
            dedicated: 0,
            driver: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(description: &str, vendor: u32, device: u32, luid: u64, software: bool) -> Adapter {
        Adapter {
            description: description.to_owned(),
            vendor,
            device,
            luid,
            software,
            dedicated: 0,
            driver: None,
        }
    }

    /// A laptop: Windows' default adapter is the integrated one, the card is ranked first
    /// for high performance, the software adapter last in both orders.
    fn laptop() -> Listing {
        Listing {
            adapters: vec![
                adapter("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9a49, 1, false),
                adapter(
                    "NVIDIA GeForce RTX 5080 Laptop GPU",
                    0x10de,
                    0x2c59,
                    2,
                    false,
                ),
                adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 3, true),
            ],
            high_performance: Some(vec![1, 0, 2]),
            minimum_power: Some(vec![0, 1, 2]),
        }
    }

    fn pick(listing: &Listing, switch: Option<&str>) -> Pick {
        choose(listing, switch, None)
            .unwrap_or_else(|e| panic!("{e}"))
            .pick
    }

    #[test]
    fn the_default_rule_takes_the_card_ranked_first_not_windows_default_adapter() {
        let choice = choose(&laptop(), None, None).unwrap();
        assert_eq!(choice.pick, Pick::Listed(1));
        assert_eq!(
            choice.how,
            "the default rule (the first hardware adapter for high performance)"
        );
        // An empty or blank switch is no switch.
        assert_eq!(pick(&laptop(), Some("")), Pick::Listed(1));
        assert_eq!(pick(&laptop(), Some("  ")), Pick::Listed(1));
    }

    /// The main PC as DXGI lists it (2026-11-25): the card, then a second adapter with the
    /// card's description and ids but its own LUID (the virtual display adapter, which
    /// renders on the card), ranked after the software adapter; Windows' default is the
    /// card. The rule takes the same adapter as before, and so does the card's name.
    #[test]
    fn on_the_main_pc_the_rule_takes_the_adapter_windows_took() {
        let card = "NVIDIA GeForce RTX 5090";
        let listing = Listing {
            adapters: vec![
                adapter(card, 0x10de, 0x2b85, 10, false),
                adapter(card, 0x10de, 0x2b85, 11, false),
                adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 12, true),
            ],
            high_performance: Some(vec![0, 2, 1]),
            minimum_power: Some(vec![0, 2, 1]),
        };
        assert_eq!(pick(&listing, None), Pick::Listed(0));
        assert_eq!(pick(&listing, Some("nvidia")), Pick::Listed(0));
        assert_eq!(pick(&listing, Some("high-performance")), Pick::Listed(0));
    }

    #[test]
    fn the_default_rule_skips_a_software_adapter_ranked_first() {
        let mut listing = laptop();
        listing.high_performance = Some(vec![2, 1, 0]);
        assert_eq!(pick(&listing, None), Pick::Listed(1));
    }

    #[test]
    fn the_default_rule_is_windows_default_adapter_without_the_ranking_or_hardware() {
        let mut listing = laptop();
        listing.high_performance = None;
        let choice = choose(&listing, None, None).unwrap();
        assert_eq!(choice.pick, Pick::Windows);
        assert_eq!(
            choice.how,
            "the default rule (no IDXGIFactory6: Windows' default adapter)"
        );
        // A hosted runner: the software adapter alone.
        let alone = Listing {
            adapters: vec![adapter(
                "Microsoft Basic Render Driver",
                0x1414,
                0x8c,
                3,
                true,
            )],
            high_performance: Some(vec![0]),
            minimum_power: Some(vec![0]),
        };
        let choice = choose(&alone, None, None).unwrap();
        assert_eq!(choice.pick, Pick::Windows);
        assert_eq!(
            choice.how,
            "the default rule (no hardware adapter ranked: Windows' default adapter)"
        );
        // Nothing listed at all (no DXGI factory): as before.
        assert_eq!(pick(&Listing::default(), None), Pick::Windows);
    }

    #[test]
    fn each_switch_value_picks_its_adapter() {
        let listing = laptop();
        let choice = choose(&listing, Some("default"), None).unwrap();
        assert_eq!(
            choice,
            Choice {
                pick: Pick::Windows,
                how: "VICTORY_ADAPTER=default".to_owned(),
                on_fail: OnFail::Refuse,
            }
        );
        assert_eq!(pick(&listing, Some("high-performance")), Pick::Listed(1));
        assert_eq!(pick(&listing, Some("minimum-power")), Pick::Listed(0));
        assert_eq!(pick(&listing, Some("Minimum-Power")), Pick::Listed(0));
        let choice = choose(&listing, Some(" high-performance "), None).unwrap();
        assert_eq!(choice.how, "VICTORY_ADAPTER=high-performance");
    }

    #[test]
    fn a_part_of_a_description_letter_case_aside_takes_the_first_match() {
        let mut listing = laptop();
        assert_eq!(pick(&listing, Some("geforce")), Pick::Listed(1));
        assert_eq!(pick(&listing, Some("INTEL")), Pick::Listed(0));
        // Never the software adapter, which is the software switches'.
        let refused = choose(&listing, Some("basic render"), None).unwrap_err();
        assert!(
            refused.starts_with("VICTORY_ADAPTER=basic render: no hardware adapter's"),
            "{refused}"
        );
        // Two that match: the first in Windows' order, whatever the ranking.
        listing
            .adapters
            .push(adapter("NVIDIA RTX A2000", 0x10de, 0x2531, 4, false));
        listing.high_performance = Some(vec![3, 1, 0, 2]);
        assert_eq!(pick(&listing, Some("nvidia")), Pick::Listed(1));
        let choice = choose(&listing, Some("NVIDIA"), None).unwrap();
        assert_eq!(choice.how, "VICTORY_ADAPTER=NVIDIA");
    }

    #[test]
    fn no_match_is_refused_with_every_adapters_description() {
        let refused = choose(&laptop(), Some("radeon"), None).unwrap_err();
        assert_eq!(
            refused,
            "VICTORY_ADAPTER=radeon: no hardware adapter's description contains \"radeon\"; the adapters, \
             in Windows' order: #0 \"Intel(R) Iris(R) Xe Graphics\" (high-performance rank 2), \
             #1 \"NVIDIA GeForce RTX 5080 Laptop GPU\" (high-performance rank 1: the default rule's pick), \
             #2 \"Microsoft Basic Render Driver\" (software: the default rule and a name skip it)"
        );
        let refused = choose(&Listing::default(), Some("nvidia"), None).unwrap_err();
        assert!(refused.ends_with("the adapters: none listed"), "{refused}");
        // A ranking Windows does not give is refused too, not guessed.
        let mut listing = laptop();
        listing.minimum_power = None;
        let refused = choose(&listing, Some("minimum-power"), None).unwrap_err();
        assert!(
            refused.starts_with("VICTORY_ADAPTER=minimum-power: Windows ranks no adapters"),
            "{refused}"
        );
        listing.high_performance = Some(vec![2]);
        let refused = choose(&listing, Some("high-performance"), None).unwrap_err();
        assert!(
            refused.contains("ranks no hardware adapter for high performance"),
            "{refused}"
        );
    }

    #[test]
    fn a_software_switch_wins_over_the_switch() {
        let listing = laptop();
        let choice = choose(&listing, None, Some("VICTORY_D3D12_WARP")).unwrap();
        assert_eq!(
            choice,
            Choice {
                pick: Pick::Software,
                how: "VICTORY_D3D12_WARP".to_owned(),
                on_fail: OnFail::NoDevice,
            }
        );
        let choice = choose(&listing, Some("nvidia"), Some("VICTORY_WARP=1")).unwrap();
        assert_eq!(choice.pick, Pick::Software);
        assert_eq!(choice.how, "VICTORY_WARP=1, over VICTORY_ADAPTER=nvidia");
        // Even a value that would be refused.
        let choice = choose(&listing, Some("radeon"), Some(TEST_SOFTWARE)).unwrap();
        assert_eq!(choice.pick, Pick::Software);
    }

    #[test]
    fn the_line_says_the_adapter_and_how_it_was_chosen() {
        let mut card = adapter(
            "NVIDIA GeForce RTX 5080 Laptop GPU",
            0x10de,
            0x2c59,
            2,
            false,
        );
        card.dedicated = 16 << 30;
        card.driver = Some((32 << 48) | (15 << 16) | 8180);
        assert_eq!(
            line(
                &card,
                "the default rule (the first hardware adapter for high performance)"
            ),
            "host adapter: NVIDIA GeForce RTX 5080 Laptop GPU (vendor 0x10de, device 0x2c59, driver 32.0.15.8180, \
             16384 MiB dedicated), chosen by the default rule (the first hardware adapter for high performance)"
        );
        let warp = adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 3, true);
        assert_eq!(
            line(&warp, "VICTORY_D3D12_WARP"),
            "host adapter: Microsoft Basic Render Driver (vendor 0x1414, device 0x008c, driver unknown, \
             0 MiB dedicated), chosen by VICTORY_D3D12_WARP"
        );
    }

    #[test]
    fn the_requirement_takes_one_or_a_part_of_a_description() {
        assert_eq!(requirement(None), None);
        assert_eq!(requirement(Some("")), None);
        assert_eq!(requirement(Some("0")), None);
        assert_eq!(requirement(Some("1")), Some(Requirement::Hardware));
        assert_eq!(
            requirement(Some("nvidia")),
            Some(Requirement::Named("nvidia".to_owned()))
        );
        let card = adapter(
            "NVIDIA GeForce RTX 5080 Laptop GPU",
            0x10de,
            0x2c59,
            2,
            false,
        );
        let integrated = adapter("Intel(R) Iris(R) Xe Graphics", 0x8086, 0x9a49, 1, false);
        let warp = adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 3, true);
        let hardware = Requirement::Hardware;
        let nvidia = Requirement::Named("nvidia".to_owned());
        assert_eq!(meets(&hardware, Got::On(&card)), Ok(()));
        assert_eq!(meets(&hardware, Got::On(&integrated)), Ok(()));
        assert_eq!(meets(&nvidia, Got::On(&card)), Ok(()));
        assert_eq!(
            meets(&nvidia, Got::On(&integrated)),
            Err("VICTORY_REQUIRE_GPU=nvidia and the device is on Intel(R) Iris(R) Xe Graphics, whose description \
                 does not contain \"nvidia\""
                .to_owned())
        );
        assert_eq!(
            meets(&hardware, Got::On(&warp)),
            Err("VICTORY_REQUIRE_GPU=1 and the device is the software adapter (Microsoft Basic Render Driver): \
                 the test did not run on hardware"
                .to_owned())
        );
        assert!(
            meets(&Requirement::Named("basic".to_owned()), Got::On(&warp)).is_err(),
            "software is never hardware"
        );
        assert_eq!(
            meets(&hardware, Got::Nothing),
            Err(
                "VICTORY_REQUIRE_GPU=1 and no hardware device was made: the test did not run"
                    .to_owned()
            )
        );
    }

    /// The laptop's card, ranked first, cannot run the host (here: a driver giving shader
    /// model 6.5): the default rule tries Windows' default adapter, once, and its line says
    /// that it fell back and why. No card is needed: `settle` is fed the candidates' fates.
    #[test]
    fn the_default_rule_falls_back_to_windows_default_adapter_once() {
        let listing = laptop();
        let short = below_baseline(0x65).unwrap();
        assert_eq!(short, "shader model 6.5 only, the host needs 6.6");
        let mut tried = Vec::new();
        let (device, how) = settle(&listing, choose(&listing, None, None).unwrap(), |pick| {
            tried.push(pick);
            if pick == Pick::Listed(1) {
                Err(short.clone())
            } else {
                Ok(pick)
            }
        })
        .unwrap();
        assert_eq!(tried, [Pick::Listed(1), Pick::Windows]);
        assert_eq!(device, Pick::Windows);
        assert_eq!(
            how,
            "the default rule (fell back to Windows' default adapter: the first hardware adapter for high \
             performance, \"NVIDIA GeForce RTX 5080 Laptop GPU\", cannot run the host: shader model 6.5 only, \
             the host needs 6.6)"
        );
        assert_eq!(
            line(&listing.adapters[0], &how),
            format!(
                "host adapter: Intel(R) Iris(R) Xe Graphics (vendor 0x8086, device 0x9a49, driver unknown, \
                 0 MiB dedicated), chosen by {how}"
            )
        );
        // A pick that can run the host is the only one tried, its line as before.
        let mut tried = Vec::new();
        let (_, how) = settle(&listing, choose(&listing, None, None).unwrap(), |pick| {
            tried.push(pick);
            Ok::<_, String>(())
        })
        .unwrap();
        assert_eq!(tried, [Pick::Listed(1)]);
        assert_eq!(
            how,
            "the default rule (the first hardware adapter for high performance)"
        );
        // Neither can: no device, both reasons said; tried once each.
        let mut tried = Vec::new();
        let unmade = settle(&listing, choose(&listing, None, None).unwrap(), |pick| {
            tried.push(pick);
            Err::<(), _>(format!("{pick:?} cannot"))
        })
        .unwrap_err();
        assert_eq!(tried, [Pick::Listed(1), Pick::Windows]);
        assert_eq!(
            unmade,
            Unmade::Failed(
                "the first hardware adapter for high performance, \"NVIDIA GeForce RTX 5080 Laptop GPU\", \
                 cannot run the host (Listed(1) cannot), nor can Windows' default adapter (Windows cannot)"
                    .to_owned()
            )
        );
        // Windows' default adapter by the rule (no ranking): no second try, no device.
        let mut unranked = laptop();
        unranked.high_performance = None;
        let mut tried = Vec::new();
        let unmade = settle(&unranked, choose(&unranked, None, None).unwrap(), |pick| {
            tried.push(pick);
            Err::<(), _>("D3D12CreateDevice: no".to_owned())
        })
        .unwrap_err();
        assert_eq!(tried, [Pick::Windows]);
        assert_eq!(unmade, Unmade::Failed("D3D12CreateDevice: no".to_owned()));
    }

    /// An adapter `VICTORY_ADAPTER` chose stays the one asked for: where it cannot run the
    /// host, the choice is refused, saying why, and no other adapter is tried.
    #[test]
    fn an_explicit_choice_that_cannot_run_the_host_is_refused_not_replaced() {
        let listing = laptop();
        for (value, pick, what) in [
            (
                "nvidia",
                Pick::Listed(1),
                "\"NVIDIA GeForce RTX 5080 Laptop GPU\"",
            ),
            (
                "high-performance",
                Pick::Listed(1),
                "\"NVIDIA GeForce RTX 5080 Laptop GPU\"",
            ),
            (
                "minimum-power",
                Pick::Listed(0),
                "\"Intel(R) Iris(R) Xe Graphics\"",
            ),
            ("default", Pick::Windows, "Windows' default adapter"),
        ] {
            let choice = choose(&listing, Some(value), None).unwrap();
            assert_eq!(choice.pick, pick, "{value}");
            let mut tried = Vec::new();
            let unmade = settle(&listing, choice, |pick| {
                tried.push(pick);
                Err::<(), _>(below_baseline(0x65).unwrap())
            })
            .unwrap_err();
            assert_eq!(tried, [pick], "{value}: no other adapter is tried");
            assert_eq!(
                unmade,
                Unmade::Refused(format!(
                    "VICTORY_ADAPTER={value}: {what} cannot run the host (shader model 6.5 only, the host needs \
                     6.6); no other adapter is tried for an adapter VICTORY_ADAPTER chose; the adapters, in \
                     Windows' order: #0 \"Intel(R) Iris(R) Xe Graphics\" (high-performance rank 2), \
                     #1 \"NVIDIA GeForce RTX 5080 Laptop GPU\" (high-performance rank 1: the default rule's pick), \
                     #2 \"Microsoft Basic Render Driver\" (software: the default rule and a name skip it)"
                ))
            );
        }
        // The software adapter a switch chose is not replaced either: no device, as before.
        let choice = choose(&listing, Some("nvidia"), Some("VICTORY_D3D12_WARP")).unwrap();
        let unmade = settle(&listing, choice, |_| {
            Err::<(), _>("no software adapter: x".to_owned())
        })
        .unwrap_err();
        assert_eq!(unmade, Unmade::Failed("no software adapter: x".to_owned()));
        assert_eq!(below_baseline(0x66), None);
        assert_eq!(below_baseline(0x67), None);
    }

    /// A device whose adapter DXGI does not describe (by the device's LUID) is not known to be
    /// hardware: the requirement fails on it, and its line says it is unknown.
    #[test]
    fn an_adapter_dxgi_does_not_describe_fails_the_requirement() {
        let failed = meets(&Requirement::Hardware, Got::Undescribed).unwrap_err();
        assert_eq!(
            failed,
            "VICTORY_REQUIRE_GPU=1 and DXGI does not describe the device's adapter: it is not known to be \
             hardware, and the test did not run on it"
        );
        assert!(meets(&Requirement::Named("nvidia".to_owned()), Got::Undescribed).is_err());
        assert!(matches!(Got::from(None), Got::Undescribed));
        assert_eq!(
            line_for(None, "VICTORY_ADAPTER=default"),
            "host adapter: unknown (DXGI does not describe the device's adapter), chosen by VICTORY_ADAPTER=default"
        );
        let card = adapter(
            "NVIDIA GeForce RTX 5080 Laptop GPU",
            0x10de,
            0x2c59,
            2,
            false,
        );
        assert_eq!(line_for(Some(&card), "x"), line(&card, "x"));
    }

    /// The hybrid laptop's gate (2026-11-25): two virtual displays enumerate under the
    /// integrated adapter's name. A refusal's list gives each entry's place in Windows' order
    /// and what the default rule does with it, so that a person can see which entry a name
    /// takes; the LUID stays out of it.
    #[test]
    fn a_refusal_tells_entries_of_one_name_apart() {
        let intel = "Intel(R) Graphics";
        let listing = Listing {
            adapters: vec![
                adapter(intel, 0x8086, 0x7d55, 0x1111, false),
                adapter(
                    "NVIDIA GeForce RTX 5080 Laptop GPU",
                    0x10de,
                    0x2c59,
                    0x2222,
                    false,
                ),
                adapter(intel, 0x8086, 0x7d55, 0x3333, false),
                adapter(intel, 0x8086, 0x7d55, 0x4444, false),
                adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 0x5555, true),
            ],
            high_performance: Some(vec![1, 0, 4, 2]),
            minimum_power: Some(vec![0, 1, 4, 2]),
        };
        let refused = choose(&listing, Some("nosuchcard"), None).unwrap_err();
        assert_eq!(
            refused,
            "VICTORY_ADAPTER=nosuchcard: no hardware adapter's description contains \"nosuchcard\"; \
             the adapters, in Windows' order: \
             #0 \"Intel(R) Graphics\" (high-performance rank 2), \
             #1 \"NVIDIA GeForce RTX 5080 Laptop GPU\" (high-performance rank 1: the default rule's pick), \
             #2 \"Intel(R) Graphics\" (high-performance rank 4; the description of #0, which a name takes first), \
             #3 \"Intel(R) Graphics\" (not ranked for high performance: the default rule never takes it; \
             the description of #0, which a name takes first), \
             #4 \"Microsoft Basic Render Driver\" (software: the default rule and a name skip it)"
        );
        for luid in ["1111", "2222", "3333", "4444", "5555"] {
            assert!(!refused.contains(luid), "{refused}");
        }
        // Without the ranking: places and the software mark only.
        let mut unranked = listing.clone();
        unranked.high_performance = None;
        let refused = choose(&unranked, Some("nosuchcard"), None).unwrap_err();
        assert!(
            refused.ends_with(
                "#3 \"Intel(R) Graphics\" (the description of #0, which a name takes first), \
                 #4 \"Microsoft Basic Render Driver\" (software: the default rule and a name skip it)"
            ),
            "{refused}"
        );
        assert!(
            refused.contains("#1 \"NVIDIA GeForce RTX 5080 Laptop GPU\", #2"),
            "{refused}"
        );
    }

    fn warp() -> Adapter {
        adapter("Microsoft Basic Render Driver", 0x1414, 0x8c, 3, true)
    }

    fn card() -> Adapter {
        adapter(
            "NVIDIA GeForce RTX 5080 Laptop GPU",
            0x10de,
            0x2c59,
            2,
            false,
        )
    }

    /// (a) A test that asked for hardware, under a software switch, runs on the software
    /// adapter and its line says the switch chose it; without the switch, the rule's adapter
    /// being the software one (no card) is a skip.
    #[test]
    fn a_hardware_leg_under_the_software_switch_runs_on_software_and_says_so() {
        let software = software_for(Asked::Hardware, Some("VICTORY_WARP=1"));
        assert_eq!(software, Some("VICTORY_WARP=1"));
        let choice = choose(&laptop(), None, software).unwrap();
        assert_eq!(choice.pick, Pick::Software);
        assert!(
            line(&warp(), &choice.how).ends_with("chosen by VICTORY_WARP=1"),
            "{}",
            choice.how
        );
        assert_eq!(
            verdict(Asked::Hardware, None, true, Ok(Some(&warp()))),
            Verdict::Run
        );
        assert!(matches!(
            verdict(Asked::Hardware, None, false, Ok(Some(&warp()))),
            Verdict::Skip(_)
        ));
        assert_eq!(software_for(Asked::Hardware, None), None);
    }

    /// (b) A test that asked for hardware, with `VICTORY_REQUIRE_GPU` set, fails where it has
    /// no hardware: no device, the software adapter (a switch's too), or an adapter not
    /// described; it runs on hardware of the kind required.
    #[test]
    fn a_hardware_leg_with_the_requirement_and_no_hardware_fails() {
        let hardware = Requirement::Hardware;
        let failed = Unmade::Failed("D3D12CreateDevice: no adapter".to_owned());
        assert_eq!(
            verdict(Asked::Hardware, Some(&hardware), false, Err(&failed)),
            Verdict::Fail(
                "VICTORY_REQUIRE_GPU=1 and no hardware device was made: the test did not run \
                 (D3D12CreateDevice: no adapter)"
                    .to_owned()
            )
        );
        for (switched, on) in [(false, Some(&warp())), (true, Some(&warp())), (false, None)] {
            assert!(
                matches!(
                    verdict(Asked::Hardware, Some(&hardware), switched, Ok(on)),
                    Verdict::Fail(_)
                ),
                "{switched} {on:?}"
            );
        }
        assert_eq!(
            verdict(Asked::Hardware, Some(&hardware), false, Ok(Some(&card()))),
            Verdict::Run
        );
        let intel = Requirement::Named("intel".to_owned());
        assert!(matches!(
            verdict(Asked::Hardware, Some(&intel), false, Ok(Some(&card()))),
            Verdict::Fail(_)
        ));
        // Without the requirement the same is a skip.
        assert_eq!(
            verdict(Asked::Hardware, None, false, Err(&failed)),
            Verdict::Skip("D3D12CreateDevice: no adapter".to_owned())
        );
        // A refusal fails whatever was asked.
        let refused = Unmade::Refused("VICTORY_ADAPTER=radeon: no".to_owned());
        for asked in [Asked::Hardware, Asked::Software] {
            assert_eq!(
                verdict(asked, None, false, Err(&refused)),
                Verdict::Fail("VICTORY_ADAPTER=radeon: no".to_owned())
            );
        }
    }

    /// (c) A test that asked for the software adapter itself runs with `VICTORY_REQUIRE_GPU`
    /// set, a software switch in effect or not, and is labelled as its own request; with no
    /// software adapter it skips.
    #[test]
    fn a_software_leg_with_the_requirement_still_runs() {
        assert_eq!(
            software_for(Asked::Software, Some("VICTORY_WARP=1")),
            Some(TEST_SOFTWARE)
        );
        for required in [
            Requirement::Hardware,
            Requirement::Named("nvidia".to_owned()),
        ] {
            for switched in [false, true] {
                assert_eq!(
                    verdict(
                        Asked::Software,
                        Some(&required),
                        switched,
                        Ok(Some(&warp()))
                    ),
                    Verdict::Run,
                    "{required:?} {switched}"
                );
            }
        }
        let none = Unmade::Failed("no software adapter: x".to_owned());
        assert_eq!(
            verdict(
                Asked::Software,
                Some(&Requirement::Hardware),
                false,
                Err(&none)
            ),
            Verdict::Skip("no software adapter: x".to_owned())
        );
        // Never a card for a test that asked for the software adapter.
        assert!(matches!(
            verdict(Asked::Software, None, false, Ok(Some(&card()))),
            Verdict::Fail(_)
        ));
    }
}
