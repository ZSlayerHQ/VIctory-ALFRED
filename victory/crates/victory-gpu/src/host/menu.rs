//! The F1 settings menu. F1 shows and hides it while our window has the
//! focus; Up/Down pick a setting, Left/Right change it, Enter runs the line
//! picked (save, save and restart, preload shaders, reset, close), Esc
//! closes. Preloading saves, ends this run and has the launcher make every
//! shader before the title starts again (`VICTORY_PRELOAD=1`, `victory
//! preload`); it asks for a second Enter first. While it is
//! shown the title sees no keyboard (`crate::SETTINGS_MENU_OPEN`, read by the
//! input module); a pad still works.
//!
//! The settings are read once at start, so a change is saved to the file
//! `VICTORY_MENU_FILE` names (the launcher, `victory play`, sets it to
//! `victory-settings.txt` under its own folder and applies it at every
//! start) and takes effect at the next start; "Save and restart" starts the
//! launcher again (`VICTORY_MENU_PLAY`, with `VICTORY_MENU_ARGS`) and ends
//! this run. The `VICTORY_MENU_...` settings are the launcher's own choices
//! (DLSS mode, frame generation, frame rate, console, picture size), which it
//! turns into the layer's switches; the others are the switches themselves.
//!
//! In every build, as the stats panel is. Drawn on the processor and placed
//! on the swap chain's buffer, in the middle of the picture: on its panel
//! (our own art, `art.rs`: read from `VICTORY_MENU_ART`, else `menu-art`
//! beside the program), or with `VICTORY_MENU_PANEL=0` or without the art as
//! the stats panel is, white dot letters on a black box. The picture is made
//! again only when something on it changes; while the menu is closed it holds
//! nothing (its buffers are retired a few presents later), and the frame-rate
//! line in the corner has small buffers of its own.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_DOWN, VK_ESCAPE, VK_F1, VK_LEFT, VK_RETURN, VK_RIGHT, VK_UP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::font::{colours, glyph};
#[cfg(test)]
use super::picture;
use super::picture::{Blend, Buffers};
use super::{Host, HostList};

mod art;
pub(super) mod chip;
mod panel;
pub(super) mod status;

const GLYPH: (u32, u32) = (5, 7);
const CELL: (u32, u32) = (6, 9);
/// As wide as the widest line the menu can show (a help line), no wider (tested).
const COLUMNS: u32 = 112;
/// Lines of the box: a title, the keys, a blank, the rows shown (group
/// headings, settings, a blank, the actions), a blank, the help line of the
/// row picked (two more lines for the preload), a note and the repository line.
const ROWS: usize = 20;
const LINES: u32 = ROWS as u32 + 9;
/// The words are kept in mixed case for the lettering to come; today's dot
/// font has capitals only, so they are painted in capitals ([`dots`]).
const TITLE: &str = "Settings";
const KEYS: &str = "Up / Down: move \u{b7} Left / Right: change \u{b7} Enter: select \u{b7} F1 or Esc: close";
/// The action that preloads shaders, and what is said of it while it is picked.
const PRELOAD: usize = 2;
const ABOUT_PRELOAD: [&str; 3] = [
    "Compiles every shader up front, then starts the game again.",
    "This can take a long time, depending on your PC.",
    "Needed again after an update of this build.",
];
/// The help lines of the other actions, in their order (the preload's is [`ABOUT_PRELOAD`]).
const ACTION_HELP: [&str; 5] = [
    "Saves your changes and restarts the game so they take effect.",
    "Saves your changes. They take effect the next time the game starts.",
    "",
    "Clears your saved changes. The defaults apply at the next launch.",
    "Closes this menu.",
];
/// The note after the first Enter on it: the second one does it.
const CONFIRM_PRELOAD: &str = "Press Enter again to save, close the game and precompile the shaders.";
/// The notes after the actions.
const NO_FILE: &str = "Nothing to save to: start the game with victory play.";
const SAVED: &str = "Saved. Applies at the next launch.";
const NOT_SAVED: &str = "Could not save the settings file.";
const RESTART_YOURSELF: &str = "Saved. Restart the game yourself to apply.";
const RESTART_FAILED: &str = "Saved, but the restart failed. Restart the game yourself.";
const RESET: &str = "Reset. Defaults apply at the next launch.";
/// The note line while no other note is up: what the row picked does once
/// changed, else what the `*` beside a row means.
const CHANGED_NOW: &str = "Changed: applied live.";
const CHANGED_LATER: &str = "Changed: applies after a restart.";
const LEGEND: &str = "* = Changed: applies after a restart.";
/// The last line of the box: where this build comes from.
const REPO: &str = "VIctory-ALFRED, built on VIctory-NVIDIA and foxhound";
const RIM: u32 = 4;
const UPLOADS: usize = 4;

/// A setting: the variable, its name in the menu, the values it takes and
/// how each is shown (`""`: not set, the build's own default), whether a
/// change applies at once, the help line shown while it is picked, the group
/// it is shown in, and other variables that move with it (`ties`).
struct Item {
    var: &'static str,
    label: &'static str,
    values: &'static [(&'static str, &'static str)],
    live: bool,
    /// One to three lines.
    help: &'static [&'static str],
    group: &'static str,
    /// Variables set with this one: each with its value for each of this
    /// item's values, by index. One that is another row's variable moves that
    /// row (shown, and saved as it then stands); one that is no row's is
    /// saved with this row. A change of this row saves all of them.
    ties: &'static [(&'static str, &'static [&'static str])],
}

impl Item {
    const fn new(group: &'static str, var: &'static str, label: &'static str, values: &'static [(&'static str, &'static str)], help: &'static [&'static str]) -> Item {
        Item { var, label, values, live: false, help, group, ties: &[] }
    }

    /// Applies while playing (`VICTORY_...` read at each use).
    const fn live(self) -> Item {
        Item { live: true, ..self }
    }

    const fn tied(self, ties: &'static [(&'static str, &'static [&'static str])]) -> Item {
        Item { ties, ..self }
    }
}

const ON_UNSET: &[(&str, &str)] = &[("1", "On"), ("", "Off")];
const UNSET_ON: &[(&str, &str)] = &[("", "Off"), ("1", "On")];
const STAGES: &[(&str, &str)] = &[("2", "2"), ("1", "1"), ("0", "Off")];

const GRAPHICS: &str = "Graphics";
const FRAME_RATE: &str = "Frame rate";
const LIGHT: &str = "Lighting and water";
const OVERLAY: &str = "Overlay";
const INPUT: &str = "Input";
const ADVANCED: &str = "Advanced";

/// The row that turns the title's own ray-traced bounce light on: its two
/// switches, with the estimates of light, haze and reflections cleared; off,
/// the play defaults of all five (the first value of each estimate's row,
/// and the two switches unset).
const BOUNCE: &str = "VICTORY_RT_RESOLVE";

/// The first value of each is what the launcher starts with (shown as recommended).
/// In the order shown: by group, each group's rows together.
#[rustfmt::skip]
const ITEMS: &[Item] = &[
    // The launcher's own choices, then the layer's switches, within each group.
    Item::new(GRAPHICS, "VICTORY_MENU_DLSS", "DLSS upscaling (not in this build)", &[("quality", "Quality"), ("balanced", "Balanced"), ("performance", "Performance"), ("dlaa", "DLAA (native)"), ("off", "Off")],
        &["NVIDIA DLSS upscaling: renders below the output resolution and reconstructs it.",
          "DLAA renders at native resolution with DLSS anti-aliasing. Needs an RTX card and NVIDIA's files."]),
    Item::new(GRAPHICS, "VICTORY_DLSS_PRESET", "DLSS model preset (not in this build)", &[("", "Default"), ("k", "K"), ("j", "J"), ("m", "M"), ("l", "L")],
        &["Which DLSS model preset is used. Default lets the installed DLSS version choose."]),
    // Not "Render and output resolution": with DLSS on, the game renders below the output.
    Item::new(GRAPHICS, "VICTORY_MENU_RES", "Resolution", &[("1440", "2560 x 1440 (not in this build)"), ("1080to1440", "1920 x 1080 upscaled to 1440p (not in this build)"), ("4k", "3840 x 2160")],
        &["3840 x 2160 is the console's own; 2560 x 1440 is native 1440p;",
          "1080p upscaled uses the game's own upscaler. Only 3840 x 2160 is in this build."]),
    Item::new(GRAPHICS, "VICTORY_MENU_CONSOLE", "Console profile", &[("series-x", "Series X (full detail)"), ("series-s", "Series S (lower detail, faster)")],
        &["Which console the game runs as. It sets the game's own detail and resolution targets."]),
    Item::new(GRAPHICS, "VICTORY_FULLSCREEN", "Borderless fullscreen", UNSET_ON,
        &["A borderless window that covers the display."]),
    Item::new(FRAME_RATE, "VICTORY_MENU_FRAMEGEN", "Frame generation (not in this build)", &[("0", "Off"), ("2", "2x"), ("3", "3x"), ("4", "4x"), ("5", "5x"), ("6", "6x")],
        &["Inserts generated frames between rendered ones: 2x doubles the displayed frame rate, up to 6x.",
          "The game still simulates at the rendered rate, and input latency rises. Above 2x needs multi frame generation."]),
    // DLSS Frame Generation where it runs (RTX 40 and later), else AMD's FSR 3 (2X only).
    Item::new(FRAME_RATE, "VICTORY_MENU_FG_TYPE", "Frame gen type (not in this build)", &[("auto", "Auto"), ("dlss", "DLSS"), ("fsr", "FSR 3")],
        &["DLSS Frame Generation (RTX 40 and 50) or AMD FSR 3 frame generation (other cards). Auto uses DLSS where it runs."]),
    // Someone else's mod (apps\victory\src\play.rs): loaded at the next start where its file was put.
    Item::new(FRAME_RATE, "VICTORY_MFG_UNLOCKER", "MFG unlocker (not in this build)", UNSET_ON,
        &["A third-party mod that enables DLSS frame generation on RTX 20 and 30. Used at the next start only if",
          "its file was put in place; turn it off if the game then fails to launch."]),
    Item::new(FRAME_RATE, "VICTORY_MENU_FPS", "FPS limit", &[("1000", "Unlimited"), ("120", "120"), ("60", "60"), ("30", "30 (console default)")],
        &["Caps the rendered frame rate. 30 is the console's own cap."]),
    // Name and help as before: whether it gives ray-traced light without the BVH builder is
    // not what the code says (`pipeline::make`: without it the resolve shades no hits).
    Item::new(LIGHT, BOUNCE, "Real bounce light (best picture, slower)", UNSET_ON,
        &["The game's own bounced light: lit walls and floors in shade. Costs about a quarter of the frame rate today."])
        .tied(&[
            ("VICTORY_ACTIVATE_RAYS", &["", "1"]),
            ("VICTORY_AMBIENT", &["0.8", ""]),
            ("VICTORY_AMBIENT_AIR", &["1.5", ""]),
            ("VICTORY_AMBIENT_SPECULAR", &["1", ""]),
        ]),
    // The fallbacks run whenever the BVH builder does not (`ambient::on`), not by the row above.
    Item::new(LIGHT, "VICTORY_AMBIENT", "Ambient light fallback", &[("0.8", "On (0.8)"), ("1", "Brighter (1.0)"), ("0.5", "Dimmer (0.5)"), ("", "Off")],
        &["An approximate ambient term. The number is its intensity."]),
    Item::new(LIGHT, "VICTORY_AMBIENT_SPECULAR", "Reflection fallback", ON_UNSET,
        &["Approximate specular reflections."]),
    Item::new(LIGHT, "VICTORY_AMBIENT_AIR", "Fog brightness clamp", &[("1.5", "1.5"), ("1", "1"), ("2", "2"), ("", "None")],
        &["Clamps how bright volumetric fog gets with the ambient fallback."]),
    // The builder is the title's own compute programs, on the card.
    Item::new(LIGHT, "VICTORY_BVH_BUILDER", "Ray tracing: full BVH (experimental)", UNSET_ON,
        &["Builds the game's full ray-tracing acceleration structure (BVH). A few frames a second for now."]),
    Item::new(LIGHT, "VICTORY_WATER_LIGHT", "Water lighting", &[("", "Off (clean)"), ("1", "On (artifacts for now)")],
        &["The water's own tiled lighting passes. On, they still draw dark blocks."]),
    Item::new(LIGHT, "VICTORY_LEAVE_OUT", "Heat haze and decal defrag", &[("02384a60fda6f2d0,731a39171b568298", "Skip (faster)"), ("", "Run (slower)")],
        &["Two effects that are expensive for now."]),
    Item::new(OVERLAY, "VICTORY_FPS_LINE", "FPS counter", &[("1", "On"), ("0", "Off")],
        &["The FPS counter in the top-left corner."]).live(),
    Item::new(OVERLAY, "VICTORY_PANEL", "Performance overlay (F3)", &[("0", "Off"), ("1", "On")],
        &["Frame time, CPU and GPU timings and draw calls, for troubleshooting."]).live(),
    Item::new(INPUT, "VICTORY_MOUSE_CAPTURE", "Lock cursor to window", UNSET_ON,
        &["Confines the mouse cursor to the game window."]),
    Item::new(INPUT, "VICTORY_IME_ENGLISH", "Force English keyboard layout", UNSET_ON,
        &["Types with the English layout whatever the Windows input language is."]),
    Item::new(ADVANCED, "VICTORY_EARLY_DEPTH", "Early depth test (early-Z)", ON_UNSET,
        &["Lets the GPU reject hidden pixels before shading them. Faster."]),
    Item::new(ADVANCED, "VICTORY_WC_NO_BACK", "Skip large GPU readbacks", ON_UNSET,
        &["Keeps large buffers on the GPU instead of copying them back to system memory. Faster."]),
    Item::new(ADVANCED, "VICTORY_SOFT_VIEWS", "Soft views (no hitch at cuts)", &[("copies", "On"), ("", "Off")],
        &["Avoids a short hitch when a cutscene changes scene."]),
    Item::new(ADVANCED, "VICTORY_TEXTURE_NO_WAIT", "Async texture uploads", ON_UNSET,
        &["The game carries on while a texture uploads; it may pop in a moment late."]),
    Item::new(ADVANCED, "VICTORY_RELAY", "Command relay", STAGES,
        &["How draw commands are handed to the GPU. 2 is fastest."]),
    Item::new(ADVANCED, "VICTORY_CARD_STEPS", "GPU-side copies", STAGES,
        &["Moves small memory copies from the CPU to the GPU. 2 is fastest."]),
    // Help as before: it batches the clamps of indirect executions, not small draws (`draw::batches`).
    Item::new(ADVANCED, "VICTORY_INDIRECT_BATCH", "Draw call batching", STAGES,
        &["Sends small pieces of drawing together. 2 is fastest."]),
    // Name and help as before: the smallest title buffer whose copy is in the card's memory
    // (`shadow::card_limits`), not a comparison instead of the CPU.
    Item::new(ADVANCED, "VICTORY_CARD_BUFFERS_MIN_KB", "Compare on the card from", &[("64", "64 KB"), ("1024", "1024 KB")],
        &["Blocks this size or larger are compared on the card."]),
];
/// The lines after the settings.
const ACTIONS: [&str; 5] =
    ["Apply and restart", "Save (applies at next launch)", "Precompile shaders (restarts the game)", "Reset to defaults", "Close"];

/// The width of the labels' column: the longest label.
const LABEL: usize = {
    let mut i = 0;
    let mut most = 0;
    while i < ITEMS.len() {
        if ITEMS[i].label.len() > most {
            most = ITEMS[i].label.len();
        }
        i += 1;
    }
    most
};

/// A row of the list: a group's heading, a blank, or a line the cursor can
/// rest on (an item, or an action after them, as `Menu::at` counts them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    Heading(&'static str),
    Blank,
    Pick(usize),
}

/// The list's rows in order: each group's heading before its items, a blank, the actions.
fn rows() -> Vec<Row> {
    let mut out = Vec::with_capacity(ITEMS.len() + ACTIONS.len() + 8);
    let mut group = "";
    for (i, item) in ITEMS.iter().enumerate() {
        if item.group != group {
            group = item.group;
            out.push(Row::Heading(group));
        }
        out.push(Row::Pick(i));
    }
    out.push(Row::Blank);
    out.extend((0..ACTIONS.len()).map(|a| Row::Pick(ITEMS.len() + a)));
    out
}

/// The dots of a character: today's font, in capitals, and the apostrophe,
/// semicolon and middle dot the menu's words have, which that font lacks.
fn dots(c: char) -> [u8; 7] {
    match c {
        '\'' => [0b00100, 0b00100, 0b01000, 0, 0, 0, 0],
        ';' => [0, 0b01100, 0b01100, 0, 0b01100, 0b00100, 0b01000],
        '\u{b7}' => [0, 0, 0, 0b01100, 0b01100, 0, 0],
        c if c.is_ascii() => glyph((c as u8).to_ascii_uppercase()),
        _ => glyph(b'#'),
    }
}

fn get(var: &str) -> String {
    std::env::var(var).unwrap_or_default()
}

/// The first start of a new victory.exe: the seconds it plays before it starts
/// again by itself (`VICTORY_WARMUP_RESTART`, set by the launcher while the exe
/// has no mark under `%LOCALAPPDATA%\victory\warmed`). On the RTX 5070, every
/// new build's first run stayed at a flat 14 fps (Present ~40 ms) for as long as
/// it ran, and every later run was normal (27-29 fps), even after a first run of
/// 45 seconds; a copy of an exe already run was fast at once (2026-11-27). Likely
/// the driver's cache of the program; not ours to fix, so it is run through once.
pub(crate) fn warmup_seconds() -> Option<u64> {
    get("VICTORY_WARMUP_RESTART").parse().ok().filter(|s| *s > 0)
}

/// The start's notice lines for the warm-up start ([`warmup_seconds`]).
pub(crate) fn warmup_notice(seconds: u64) -> Vec<String> {
    vec![
        "FIRST START OF THIS VERSION: A ONE-TIME WARM-UP".to_owned(),
        format!("THE GAME RESTARTS BY ITSELF IN ABOUT {seconds} SECONDS"),
        "THIS HAPPENS ONCE PER NEW VERSION, AND IS NEEDED:".to_owned(),
        "WITHOUT IT THE FIRST RUN STAYS AT ABOUT 14 FPS".to_owned(),
        String::new(),
    ]
}

/// Starts `command_line` in a console window of its own, with that console's input and output.
/// (`std::process::Command` always hands its child this process's standard handles: the restarted
/// launcher then wrote into the old launcher's pipe and log, not its own window, and kept the old
/// launcher waiting on that pipe.)
fn new_console(command_line: &str) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{CREATE_NEW_CONSOLE, CreateProcessW, PROCESS_INFORMATION, STARTUPINFOW};
    let mut line: Vec<u16> = command_line.encode_utf16().chain(Some(0)).collect();
    // SAFETY: plain structs, zero is their empty value.
    let mut startup: STARTUPINFOW = unsafe { std::mem::zeroed() };
    startup.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    let mut process: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `line` is a writable, terminated UTF-16 string; no handles inherited; this
    // process's environment and folder.
    let made = unsafe {
        CreateProcessW(
            std::ptr::null(),
            line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            CREATE_NEW_CONSOLE,
            std::ptr::null(),
            std::ptr::null(),
            &startup,
            &mut process,
        )
    };
    if made == 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the two handles CreateProcessW gave, closed once.
    unsafe {
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
    }
    Ok(())
}

/// Whether a change of `item` shows at once. Frame generation does when this run started with
/// it ready (the launcher asks for that whenever Streamline is there).
fn live(item: &Item) -> bool {
    item.live || (item.var == "VICTORY_MENU_FRAMEGEN" && crate::streamline::framegen_live())
}

/// Whether the frame-rate line is shown at the top while the menu is closed
/// (`VICTORY_FPS_LINE=0`: not; the menu changes it).
fn fps_line_on() -> &'static AtomicBool {
    static ON: OnceLock<AtomicBool> = OnceLock::new();
    ON.get_or_init(|| AtomicBool::new(std::env::var_os("VICTORY_FPS_LINE").is_none_or(|v| v != "0")))
}

/// That line: the frames rendered a second and, while frame generation shows more, those shown.
fn fps_line() -> Option<String> {
    if !fps_line_on().load(Ordering::Relaxed) {
        return None;
    }
    // The stats panel has that corner (F3), and says the same.
    if super::panel::shown().load(Ordering::Relaxed) {
        return None;
    }
    let fps = f64::from(super::METER_FPS_TENTHS.load(Ordering::Relaxed)) / 10.0;
    if fps <= 0.0 {
        return None;
    }
    Some(match crate::streamline::shown_per_frame() {
        1 => format!("FPS: {fps:.1}"),
        per => format!("FPS: {fps:.1}  FG: {:.1}", fps * f64::from(per)),
    })
}

/// The index of `value` among the item's values (an unknown value: shown as
/// itself, at the end).
fn index_of(item: &Item, value: &str) -> Option<usize> {
    item.values.iter().position(|(v, _)| *v == value)
}

/// What the menu shows for `value`: its label (the first, what the launcher
/// starts with, marked as recommended on the row picked), an unknown value as
/// itself, and a value the launcher did not set (a start without the launcher,
/// `victory play`) as "Default" rather than a blank.
fn shown_value(item: &Item, value: &str, picked: bool) -> String {
    match index_of(item, value) {
        Some(0) if picked => format!("{} (recommended)", item.values[0].1),
        Some(i) => item.values[i].1.to_owned(),
        None if value.is_empty() => "Default".to_owned(),
        None => value.to_owned(),
    }
}

/// Keys pressed, gathered on a thread of their own every 10 ms: a frame
/// lasts 50 ms and more, and a short press would fall between two frames.
fn keys() -> &'static Mutex<Vec<u16>> {
    static KEYS: OnceLock<Mutex<Vec<u16>>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let _ = std::thread::Builder::new().name("settings menu keys".into()).spawn(|| {
            const WATCHED: [u16; 7] = [VK_F1, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN, VK_ESCAPE];
            let mut was = [false; WATCHED.len()];
            loop {
                std::thread::sleep(Duration::from_millis(10));
                let mut process = 0;
                // SAFETY: plain queries; room for the answer.
                unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
                let ours = process == std::process::id();
                for (i, &vk) in WATCHED.iter().enumerate() {
                    // SAFETY: a plain query.
                    let down = ours && unsafe { GetAsyncKeyState(i32::from(vk)) } < 0;
                    if down && !was[i] {
                        if vk == VK_F1 {
                            crate::F1_PRESSES.fetch_add(1, Ordering::Relaxed);
                        }
                        keys().lock().unwrap_or_else(|e| e.into_inner()).push(vk);
                    }
                    was[i] = down;
                }
            }
        });
        Mutex::new(Vec::new())
    })
}

/// The whole-number scale the box is drawn at in a picture `height` pixels high.
const fn scale_for(height: u32) -> u32 {
    if height < 1080 { 1 } else { height / 540 }
}

/// The box's size in pixels.
const fn size(scale: u32) -> (u32, u32) {
    size_of((COLUMNS, LINES), scale)
}

/// The size in pixels of a box of `columns` by `lines` characters.
const fn size_of((columns, lines): (u32, u32), scale: u32) -> (u32, u32) {
    ((columns * CELL.0 + 2 * RIM) * scale, (lines * CELL.1 + 2 * RIM) * scale)
}

/// The frame-rate line's box, in characters.
const CORNER: (u32, u32) = (32, 2);

/// `lines`: each line's characters as their dots ([`dots`]), in a box `columns` wide.
fn lit(lines: &[Vec<[u8; 7]>], columns: u32, x: u32, y: u32) -> bool {
    let (Some(x), Some(y)) = (x.checked_sub(RIM), y.checked_sub(RIM)) else {
        return false;
    };
    if x % CELL.0 >= GLYPH.0 || y % CELL.1 >= GLYPH.1 || x / CELL.0 >= columns {
        return false;
    }
    let Some(c) = lines.get((y / CELL.1) as usize).and_then(|l| l.get((x / CELL.0) as usize)) else {
        return false;
    };
    (c[(y % CELL.1) as usize] >> (GLYPH.0 - 1 - x % CELL.0)) & 1 != 0
}

/// `lines` in a box of `chars` (columns, lines), white on black, rows `pitch` bytes apart.
fn painted(lines: &[String], chars: (u32, u32), scale: u32, colours: [&[u8]; 2], pitch: usize) -> Vec<u8> {
    let (width, height) = size_of(chars, scale);
    let bytes = colours[0].len();
    let lines: Vec<Vec<[u8; 7]>> = lines.iter().map(|l| l.chars().map(dots).collect()).collect();
    let lines = lines.as_slice();
    let mut out = vec![0; pitch * height as usize];
    for y in 0..height / scale {
        let top = pitch * (y * scale) as usize;
        for x in 0..width / scale {
            let colour = colours[usize::from(lit(lines, chars.0, x, y))];
            for i in 0..scale {
                let at = top + (x * scale + i) as usize * bytes;
                out[at..at + bytes].copy_from_slice(colour);
            }
        }
        for i in 1..scale as usize {
            out.copy_within(top..top + pitch, top + i * pitch);
        }
    }
    out
}

/// `lines` in dot letters in a box of `chars`, into the next of `buffers`.
fn write_dots(buffers: &mut Buffers, lines: &[String], chars: (u32, u32), scale: u32) {
    let Some(colours) = colours(buffers.format) else { return };
    let picture = painted(lines, chars, scale, colours, buffers.pitch as usize);
    buffers.next().copy_from_slice(&picture);
}

/// Presents a menu's buffers are kept after it closes or is resized (the
/// card may still copy from them), before they are retired.
const LEAVE: u32 = 8;

/// How the open menu is drawn.
enum Look {
    /// Step one's dot letters, white on black, at a whole-number scale.
    Dots { scale: u32 },
    /// On the panel: the art at the menu's size, and the picture drawn on it.
    Panel { scaled: Box<art::Scaled>, canvas: Vec<u8> },
}

impl Look {
    fn size(&self) -> (u32, u32) {
        match self {
            Look::Dots { scale } => size(*scale),
            Look::Panel { scaled, .. } => scaled.size,
        }
    }

    /// The bytes it holds on the processor's side.
    fn bytes(&self) -> u64 {
        match self {
            Look::Dots { .. } => 0,
            Look::Panel { scaled, canvas } => (scaled.bytes() + canvas.capacity()) as u64,
        }
    }
}

/// The open menu: for which swap chain format and window size, how it is drawn, and its buffers.
struct Shown {
    key: (DXGI_FORMAT, u32, u32),
    look: Look,
    buffers: Buffers,
}

/// The menu of a presenter.
pub(super) struct Menu {
    /// The values chosen, by item (as at start until changed).
    values: Vec<String>,
    /// The values at start.
    started: Vec<String>,
    /// The line picked: an item, or an action after them.
    at: usize,
    note: String,
    changed: bool,
    /// While it is open.
    shown: Option<Shown>,
    /// The frame-rate line's buffers while it is shown (the menu closed), and the line they hold.
    corner: Option<(Buffers, String)>,
    /// The corner's chip, when its art is there.
    chip: chip::ChipView,
    /// The corner's lines, and when they were looked at.
    corner_lines: Option<(std::time::Instant, Vec<chip::Line>)>,
    /// Buffers let go, with the presents they are still kept for.
    leaving: Vec<(u32, Buffers)>,
    /// Drawn on the panel when its art is there (`VICTORY_MENU_PANEL`, unless 0).
    panel: bool,
    failed: bool,
}

impl Menu {
    pub(super) fn new() -> Menu {
        // Start the key thread now: F1 must be seen while the menu is closed.
        let _ = keys();
        // `VICTORY_MENU_OPEN=1`: shown from the start (to look at it in a run nobody presses keys in).
        if std::env::var_os("VICTORY_MENU_OPEN").is_some_and(|v| v == "1") {
            crate::SETTINGS_MENU_OPEN.store(true, Ordering::Relaxed);
        }
        let values: Vec<String> = ITEMS.iter().map(|i| get(i.var)).collect();
        let panel = std::env::var_os("VICTORY_MENU_PANEL").is_none_or(|v| v != "0");
        Menu::with(values, panel)
    }

    fn with(values: Vec<String>, panel: bool) -> Menu {
        Menu {
            started: values.clone(),
            values,
            at: 0,
            note: String::new(),
            changed: true,
            shown: None,
            corner: None,
            chip: chip::ChipView::new(panel),
            corner_lines: None,
            leaving: Vec::new(),
            panel,
            failed: false,
        }
    }

    fn open() -> bool {
        crate::SETTINGS_MENU_OPEN.load(Ordering::Relaxed)
    }

    /// Item `i`'s line: the label, the value shown, and a mark: `*` changed
    /// since the start, `NOW` applies while playing.
    fn item_line(&self, i: usize, cursor: char) -> String {
        let item = &ITEMS[i];
        let value = &self.values[i];
        let shown = shown_value(item, value, i == self.at);
        let mark = match (*value != self.started[i], live(item)) {
            (true, true) => " *LIVE",
            (true, false) => " *",
            (false, true) => " LIVE",
            (false, false) => "",
        };
        format!("{cursor} {:<width$} < {shown} >{mark}", item.label, width = LABEL)
    }

    /// The help of the line picked: a setting's, or an action's.
    fn help(&self) -> Vec<&'static str> {
        match ITEMS.get(self.at) {
            Some(item) => item.help.to_vec(),
            None if self.at == ITEMS.len() + PRELOAD => ABOUT_PRELOAD.to_vec(),
            None => vec![ACTION_HELP[self.at - ITEMS.len()]],
        }
    }

    /// The note: one an action left, else whether the setting picked is changed and when that applies.
    fn note(&self) -> Option<&str> {
        if !self.note.is_empty() {
            return Some(&self.note);
        }
        let item = ITEMS.get(self.at)?;
        (self.values[self.at] != self.started[self.at]).then(|| if live(item) { CHANGED_NOW } else { CHANGED_LATER })
    }

    /// Whether any setting is changed since the start (marked `*`).
    fn any_changed(&self) -> bool {
        self.values != self.started
    }

    /// What the panel shows ([`panel::Page`]): the same rows and words as [`Menu::lines`], in mixed case.
    fn page(&self) -> panel::Page {
        let entries = rows()
            .into_iter()
            .map(|row| match row {
                Row::Heading(group) => panel::Entry::Heading(group),
                Row::Blank => panel::Entry::Blank,
                Row::Pick(pick) => {
                    let picked = pick == self.at;
                    let Some(item) = ITEMS.get(pick) else {
                        return panel::Entry::Line { label: ACTIONS[pick - ITEMS.len()], mark: "", value: String::new(), picked };
                    };
                    let value = &self.values[pick];
                    let mark = match (*value != self.started[pick], live(item)) {
                        (true, true) => "  * Live",
                        (true, false) => "  *",
                        (false, true) => "  Live",
                        (false, false) => "",
                    };
                    panel::Entry::Line { label: item.label, mark, value: shown_value(item, value, picked), picked }
                }
            })
            .collect();
        let note = self.note().map(str::to_owned).or_else(|| self.any_changed().then(|| LEGEND.to_owned()));
        panel::Page { entries, help: self.help(), note, keys: KEYS, repo: REPO }
    }

    fn lines(&self) -> Vec<String> {
        let rows = rows();
        let total = rows.len();
        let at = rows.iter().position(|r| *r == Row::Pick(self.at)).unwrap_or(0);
        let first = at.saturating_sub(ROWS / 2).min(total.saturating_sub(ROWS));
        let mut out = vec![TITLE.to_owned(), KEYS.to_owned(), String::new()];
        for row in &rows[first..(first + ROWS).min(total)] {
            let line = match *row {
                Row::Heading(group) => group.to_owned(),
                Row::Blank => String::new(),
                Row::Pick(pick) => {
                    let cursor = if pick == self.at { '>' } else { ' ' };
                    if pick < ITEMS.len() { self.item_line(pick, cursor) } else { format!("{cursor} {}", ACTIONS[pick - ITEMS.len()]) }
                }
            };
            out.push(line.chars().take(COLUMNS as usize).collect());
        }
        while out.len() < 3 + ROWS {
            out.push(String::new());
        }
        out.push(String::new());
        let help = self.help();
        out.extend((0..3).map(|i| help.get(i).copied().unwrap_or_default().to_owned()));
        out.push(self.note().unwrap_or(LEGEND).to_owned());
        out.push(REPO.to_owned());
        out
    }

    /// Moves the cursor by `step` (+1 / -1) through the lines it can rest on
    /// (headings and blanks are not among them), round at the ends.
    fn go(&mut self, step: isize) {
        let total = (ITEMS.len() + ACTIONS.len()) as isize;
        self.at = (self.at as isize + step).rem_euclid(total) as usize;
    }

    /// Changes item `i` by `step` (+1 / -1) through its values, and the rows tied to it.
    fn change(&mut self, i: usize, step: isize) {
        let item = &ITEMS[i];
        let n = item.values.len() as isize;
        let now = index_of(item, &self.values[i]).map_or(-1, |x| x as isize);
        let next = if now < 0 { if step > 0 { 0 } else { n - 1 } } else { (now + step).rem_euclid(n) };
        self.values[i] = item.values[next as usize].0.to_owned();
        if live(item) {
            self.apply_live(i);
        }
        for (var, by_value) in item.ties {
            if let Some(j) = ITEMS.iter().position(|o| o.var == *var) {
                self.values[j] = by_value[next as usize].to_owned();
                if live(&ITEMS[j]) {
                    self.apply_live(j);
                }
            }
        }
        self.note.clear();
    }

    /// A setting the build reads at each use: changed in this run's environment at once.
    fn apply_live(&self, i: usize) {
        let (var, value) = (ITEMS[i].var, &self.values[i]);
        // SAFETY: the variables changed here are read by this build with
        // std::env::var at each use; Windows' environment block is locked by
        // the process heap's routines, and std holds its own lock.
        unsafe {
            if value.is_empty() {
                std::env::remove_var(var);
            } else {
                std::env::set_var(var, value);
            }
        }
        match var {
            // Off (0) or the frames shown for each one rendered, from the next present on.
            "VICTORY_MENU_FRAMEGEN" => crate::streamline::set_framegen(value.parse().unwrap_or(0)),
            "VICTORY_FPS_LINE" => fps_line_on().store(value != "0", Ordering::Relaxed),
            "VICTORY_PANEL" => super::panel::shown().store(value == "1", Ordering::Relaxed),
            _ => {}
        }
    }

    /// The settings file `file` with the changed values written over it: each
    /// changed row's variable, and with it every variable tied to it (a row's
    /// as that row now stands, else the tie's value for this row's value).
    fn saved(&self, file: &str) -> String {
        let mut kept: Vec<(String, String)> = file
            .lines()
            .filter_map(|l| l.split_once('=').map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned())))
            .filter(|(k, _)| k.starts_with("VICTORY_"))
            .collect();
        let mut put = |var: &str, value: &str| {
            kept.retain(|(k, _)| k != var);
            kept.push((var.to_owned(), value.to_owned()));
        };
        for (i, item) in ITEMS.iter().enumerate() {
            if self.values[i] == self.started[i] {
                continue;
            }
            put(item.var, &self.values[i]);
            let Some(chosen) = index_of(item, &self.values[i]) else { continue };
            for (var, by_value) in item.ties {
                match ITEMS.iter().position(|o| o.var == *var) {
                    Some(j) => put(var, &self.values[j]),
                    None => put(var, by_value[chosen]),
                }
            }
        }
        let mut text = String::from("# Saved by the F1 menu; applied over the play settings at every start (an empty value: unset).\r\n");
        for (k, v) in &kept {
            text.push_str(&format!("{k}={v}\r\n"));
        }
        text
    }

    /// Writes the changed values over those in the settings file.
    fn save(&mut self) -> bool {
        let Some(path) = std::env::var_os("VICTORY_MENU_FILE") else {
            self.note = NO_FILE.to_owned();
            return false;
        };
        let text = self.saved(&std::fs::read_to_string(&path).unwrap_or_default());
        match std::fs::write(&path, text) {
            Ok(()) => {
                tracing::info!(path = %std::path::Path::new(&path).display(), "settings menu: saved");
                self.started = self.values.clone();
                self.note = SAVED.to_owned();
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "settings menu: not saved");
                self.note = NOT_SAVED.to_owned();
                false
            }
        }
    }

    /// Starts the launcher again in a new console, a few seconds on, and ends this run.
    /// `preload`: the launcher makes every shader first (`VICTORY_PRELOAD=1`, for that start alone).
    fn restart(&mut self, preload: bool) {
        let Some(play) = std::env::var_os("VICTORY_MENU_PLAY") else {
            self.note = RESTART_YOURSELF.to_owned();
            return;
        };
        let play = play.to_string_lossy().replace('\'', "''");
        let args = get("VICTORY_MENU_ARGS");
        // The preload is asked by VICTORY_PRELOAD=1, which both launchers read (victory.exe's is
        // `--preload`, the script's `-Preload`: the script's flag given to victory.exe was refused,
        // and the window closed before it could be read, 2026-11-27). A launcher that fails keeps
        // its window until Enter.
        let command = format!(
            "Start-Sleep 3; {}& '{play}' {args}; if ($LASTEXITCODE) {{ Read-Host 'The start failed (see above). Press Enter to close this window' }}",
            if preload { "$env:VICTORY_PRELOAD='1'; " } else { "" }
        );
        match new_console(&format!("powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \"{}\"", command.replace('"', "\\\""))) {
            Ok(()) => {
                tracing::info!(command, preload, "settings menu: restarting");
                // Ended at once: std::process::exit runs the DLLs' unloading, which with
                // Streamline loaded did not come back (a windowless leftover process).
                // SAFETY: this process, with nothing of its own left to save.
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
                    TerminateProcess(GetCurrentProcess(), 0);
                }
                std::process::exit(0);
            }
            Err(e) => {
                tracing::warn!(error = %e, "settings menu: restart failed");
                self.note = RESTART_FAILED.to_owned();
            }
        }
    }

    /// The warm-up start ([`warmup_seconds`]): once its time is up, this exe is
    /// marked as warmed up (`VICTORY_WARMUP_MARK`) and the game starts again.
    fn warmup(&mut self) {
        static BEGAN: OnceLock<std::time::Instant> = OnceLock::new();
        static DONE: AtomicBool = AtomicBool::new(false);
        let Some(seconds) = warmup_seconds() else { return };
        if DONE.load(Ordering::Relaxed) {
            return;
        }
        if BEGAN.get_or_init(std::time::Instant::now).elapsed() < Duration::from_secs(seconds) {
            return;
        }
        DONE.store(true, Ordering::Relaxed);
        let mark = std::path::PathBuf::from(get("VICTORY_WARMUP_MARK"));
        if !mark.as_os_str().is_empty() {
            if let Some(dir) = mark.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&mark, "this victory.exe was warmed up (its first start, then a restart)\r\n");
        }
        tracing::info!(seconds, mark = %mark.display(), "first start of this victory.exe: warmed up; the game starts again");
        self.restart(false);
    }

    /// Removes the settings file: the launcher's own settings at the next start.
    fn reset(&mut self) {
        if let Some(path) = std::env::var_os("VICTORY_MENU_FILE") {
            let _ = std::fs::remove_file(path);
        }
        self.note = RESET.to_owned();
    }

    /// A frame was presented: the keys pressed since the last one.
    pub(super) fn frame(&mut self) {
        self.warmup();
        let pressed = std::mem::take(&mut *keys().lock().unwrap_or_else(|e| e.into_inner()));
        if pressed.is_empty() {
            return;
        }
        for vk in pressed {
            let open = Self::open();
            if vk == VK_F1 || (open && vk == VK_ESCAPE) {
                crate::SETTINGS_MENU_OPEN.store(!open && vk == VK_F1, Ordering::Relaxed);
                self.note.clear();
                self.changed = true;
                continue;
            }
            if !open {
                continue;
            }
            // Only an Enter right after the first one starts a preload.
            let confirmed = self.note == CONFIRM_PRELOAD;
            if confirmed {
                self.note.clear();
            }
            match vk {
                VK_UP => self.go(-1),
                VK_DOWN => self.go(1),
                VK_LEFT if self.at < ITEMS.len() => self.change(self.at, -1),
                VK_RIGHT if self.at < ITEMS.len() => self.change(self.at, 1),
                VK_RETURN if self.at < ITEMS.len() => self.change(self.at, 1),
                VK_RETURN => match self.at - ITEMS.len() {
                    0 => {
                        if self.save() {
                            self.restart(false);
                        }
                    }
                    1 => {
                        self.save();
                    }
                    PRELOAD if !confirmed => self.note = CONFIRM_PRELOAD.to_owned(),
                    PRELOAD => {
                        if self.save() {
                            self.restart(true);
                        }
                    }
                    3 => self.reset(),
                    _ => crate::SETTINGS_MENU_OPEN.store(false, Ordering::Relaxed),
                },
                _ => {}
            }
            self.changed = true;
        }
    }

    /// Records the box onto `back` (in the copy-destination state):
    /// the menu while it is open, else the frame-rate line, else nothing.
    pub(super) fn draw(&mut self, blend: &mut Blend, host: &Host, list: &HostList, back: &ID3D12Resource, (width, height): (u32, u32), format: DXGI_FORMAT) {
        self.leave();
        self.chip.tick();
        if self.failed {
            return;
        }
        if !Self::open() {
            self.close();
            self.corner(blend, host, list, back, (width, height), format);
            return;
        }
        self.let_go_of_corner();
        self.chip.release();
        let key = (format, width, height);
        if self.shown.as_ref().is_none_or(|s| s.key != key) {
            self.close();
            let art = if self.panel { Self::art() } else { None };
            self.shown = self.show(blend, host, key, art);
            self.failed = self.shown.is_none();
            self.changed = true;
        }
        if std::mem::take(&mut self.changed) {
            self.paint();
        }
        let Some(shown) = &mut self.shown else { return };
        let (w, h) = shown.look.size();
        let (w, h) = (w.min(width), h.min(height));
        shown.buffers.draw(blend, host, list, back, (w, h), ((width - w) / 2, (height - h) / 2));
    }

    /// The art, read and decoded ([`art::Art::read`]); `None` (said once) without it.
    fn art() -> Option<art::Art> {
        static SAID: AtomicBool = AtomicBool::new(false);
        let dir = art::folder()?;
        match art::Art::read(&dir) {
            Ok(art) => Some(art),
            Err(e) => {
                if !SAID.swap(true, Ordering::Relaxed) {
                    tracing::warn!(dir = %dir.display(), error = %e, "settings menu: no panel art; drawn in dot letters");
                }
                None
            }
        }
    }

    /// The open menu for a swap chain of `format` in a window `width` by
    /// `height`: on the panel if `art` is given, else in dot letters.
    fn show(&self, blend: &mut Blend, host: &Host, key: (DXGI_FORMAT, u32, u32), art: Option<art::Art>) -> Option<Shown> {
        let (format, width, height) = key;
        let began = std::time::Instant::now();
        let art = art.filter(|art| !art.alpha() || blend.ready(host, format));
        let alpha = art.as_ref().is_some_and(art::Art::alpha);
        let look = match art {
            Some(art) => Look::Panel { scaled: Box::new(art.at(art::panel_size((width, height)))), canvas: Vec::new() },
            None => Look::Dots { scale: scale_for(height) },
        };
        let buffers = Buffers::overlay(host, format, look.size(), UPLOADS, alpha)?;
        let (w, h) = look.size();
        tracing::info!(
            panel = matches!(look, Look::Panel { .. }),
            w,
            h,
            ms = began.elapsed().as_secs_f64() * 1e3,
            processor_bytes = look.bytes(),
            card_bytes = buffers.bytes(),
            "settings menu: shown"
        );
        Some(Shown { key, look, buffers })
    }

    /// Writes the open menu's picture into its next upload buffer.
    fn paint(&mut self) {
        let lines = match self.shown.as_ref().map(|s| &s.look) {
            Some(Look::Dots { .. }) => Some(self.lines()),
            _ => None,
        };
        let page = if lines.is_none() { Some(self.page()) } else { None };
        let Some(shown) = &mut self.shown else { return };
        let began = std::time::Instant::now();
        match (&mut shown.look, lines, page) {
            (Look::Dots { scale }, Some(lines), _) => write_dots(&mut shown.buffers, &lines, (COLUMNS, LINES), *scale),
            (Look::Panel { scaled, canvas }, _, Some(page)) => {
                panel::compose(scaled, &page, canvas);
                shown.buffers.write_canvas(canvas, scaled.size);
            }
            _ => {}
        }
        tracing::debug!(ms = began.elapsed().as_secs_f64() * 1e3, "settings menu: picture made");
    }

    /// Closed: what the open menu held is let go (its buffers once the card is done with them).
    fn close(&mut self) {
        if let Some(shown) = self.shown.take() {
            self.leaving.push((LEAVE, shown.buffers));
            let (processor_bytes, card_bytes) = self.held();
            tracing::debug!(processor_bytes, card_bytes, "settings menu: closed; still held until the card is done");
        }
    }

    fn let_go_of_corner(&mut self) {
        if let Some((buffers, _)) = self.corner.take() {
            self.leaving.push((LEAVE, buffers));
        }
    }

    /// A present: buffers let go are kept for [`LEAVE`] presents, then retired.
    fn leave(&mut self) {
        for (left, _) in &mut self.leaving {
            *left = left.saturating_sub(1);
        }
        while let Some(i) = self.leaving.iter().position(|(left, _)| *left == 0) {
            super::retire(self.leaving.swap_remove(i).1);
        }
    }

    /// The bytes held for the menu: on the processor's side, and buffers on
    /// the card or for it (those still leaving included).
    fn held(&self) -> (u64, u64) {
        let (chip_cpu, chip_card) = self.chip.held();
        let cpu = self.shown.as_ref().map_or(0, |s| s.look.bytes()) + chip_cpu;
        let card = self.shown.iter().map(|s| s.buffers.bytes()).sum::<u64>()
            + chip_card
            + self.corner.iter().map(|(b, _)| b.bytes()).sum::<u64>()
            + self.leaving.iter().map(|(_, b)| b.bytes()).sum::<u64>();
        (cpu, card)
    }

    /// The menu closed: the frame-rate line and the shaders' status in the top
    /// left corner, on the chip when its art is there, else in dot letters in
    /// buffers of their own size; nothing (and nothing held) when neither is wanted.
    fn corner(&mut self, blend: &mut Blend, host: &Host, list: &HostList, back: &ID3D12Resource, (width, height): (u32, u32), format: DXGI_FORMAT) {
        let lines = self.corner_lines();
        if lines.is_empty() {
            self.let_go_of_corner();
            self.chip.release();
            return;
        }
        if self.chip.ready(height) && self.chip.prepare(blend, host, (width, height), format, &lines) {
            self.let_go_of_corner();
            self.chip.draw(blend, host, list, back, (width, height));
            return;
        }
        let texts: Vec<String> = lines.into_iter().map(|(l, _)| l).collect();
        let scale = scale_for(height);
        let size = size_of(CORNER, scale);
        if self.corner.as_ref().is_none_or(|(b, _)| (b.format, b.size) != (format, size)) {
            self.let_go_of_corner();
            let Some(buffers) = Buffers::new(host, format, size, UPLOADS) else {
                self.failed = true;
                return;
            };
            self.corner = Some((buffers, String::new()));
        }
        let Some((buffers, painted)) = &mut self.corner else { return };
        let joined = texts.join("\n");
        if *painted != joined {
            write_dots(buffers, &texts, CORNER, scale);
            *painted = joined;
        }
        let longest = texts.iter().map(|t| t.chars().count()).max().unwrap_or(0).min(CORNER.0 as usize) as u32;
        let w = (longest * CELL.0 + 2 * RIM) * scale;
        let h = (texts.len() as u32 * CELL.1 + 2 * RIM) * scale;
        let (w, h) = (w.min(width).min(size.0), h.min(height).min(size.1));
        buffers.copy_out(list, back, (w, h), (0, 0));
    }

    /// The corner's lines now, looked at again at most every [`CORNER_EVERY`]
    /// (the counter changes about once a second; the status's count may change
    /// every frame, and the chip is made again only when its lines change).
    fn corner_lines(&mut self) -> Vec<chip::Line> {
        let now = std::time::Instant::now();
        if let Some((at, lines)) = &self.corner_lines
            && now.duration_since(*at) < CORNER_EVERY
        {
            return lines.clone();
        }
        let lines = corner_lines_from(fps_line(), status::now(), super::panel::shown().load(Ordering::Relaxed));
        self.corner_lines = Some((now, lines.clone()));
        lines
    }
}

/// How often the corner's lines are looked at.
const CORNER_EVERY: Duration = Duration::from_millis(250);

/// The corner's lines: the frame rate if it is shown, and the shaders'
/// status unless the details panel (F3), which says it itself, is up.
fn corner_lines_from(fps: Option<String>, status: status::Status, details_shown: bool) -> Vec<chip::Line> {
    let mut lines: Vec<chip::Line> = fps.into_iter().map(|l| (l, art::LABEL)).collect();
    if !details_shown && let Some(words) = status.words() {
        lines.push((words, status.colour()));
    }
    lines
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A menu as at a start with `values`, nothing changed.
    fn menu_with(values: Vec<String>) -> Menu {
        Menu { changed: false, ..Menu::with(values, false) }
    }

    /// The play defaults: the first value of every row.
    fn play_defaults() -> Menu {
        menu_with(ITEMS.iter().map(|i| i.values[0].0.to_owned()).collect())
    }

    /// The variables a settings file sets, in its order.
    fn set_in(file: &str) -> Vec<(String, String)> {
        file.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
            .collect()
    }

    fn sorted(mut set: Vec<(String, String)>) -> Vec<(String, String)> {
        set.sort();
        set
    }

    fn pairs(set: &[(&str, &str)]) -> Vec<(String, String)> {
        set.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()
    }

    fn item(var: &str) -> usize {
        ITEMS.iter().position(|i| i.var == var).unwrap_or_else(|| panic!("no row sets {var}"))
    }

    #[test]
    fn every_line_fits_and_every_character_has_a_picture() {
        let mut menu = menu_with(ITEMS.iter().map(|_| String::new()).collect());
        for at in 0..ITEMS.len() + ACTIONS.len() {
            menu.at = at;
            for line in menu.lines() {
                assert!(line.chars().count() <= COLUMNS as usize, "{line}");
                for c in line.chars() {
                    assert_ne!(dots(c), glyph(b'#'), "{line}: {c}");
                }
            }
        }
    }

    #[test]
    fn the_link_is_always_right_under_the_note() {
        // The words of the note, the help lines, the actions, the keys and the last line are the
        // wording for PC gamers, each technical statement checked against the code.
        let mut menu = menu_with(ITEMS.iter().map(|_| String::new()).collect());
        for note in ["", "SAVED"] {
            menu.note = note.to_owned();
            for at in 0..ITEMS.len() + ACTIONS.len() {
                menu.at = at;
                let lines = menu.lines();
                assert_eq!(lines.len(), LINES as usize);
                let n = lines.len();
                // Nothing is changed: the note, else what `*` means.
                assert_eq!(lines[n - 2], if note.is_empty() { LEGEND } else { note });
                assert_eq!(lines[n - 1], REPO);
                // What a preload costs is said while it is picked; a row's or action's help line while it is.
                let about = &lines[n - 5..n - 2];
                if at == ITEMS.len() + PRELOAD {
                    assert_eq!(about, ABOUT_PRELOAD);
                } else if let Some(item) = ITEMS.get(at) {
                    let mut help = item.help.to_vec();
                    help.resize(3, "");
                    assert_eq!(about, &help[..]);
                } else {
                    assert_eq!(about, [ACTION_HELP[at - ITEMS.len()], "", ""]);
                }
            }
        }
        assert_eq!(ACTIONS[PRELOAD], "Precompile shaders (restarts the game)");
        assert_eq!(REPO, "VIctory-ALFRED, built on VIctory-NVIDIA and foxhound");
        assert!(CONFIRM_PRELOAD.len() <= COLUMNS as usize);
        assert_eq!(CONFIRM_PRELOAD, "Press Enter again to save, close the game and precompile the shaders.");
        assert!(KEYS.ends_with("F1 or Esc: close"));
        assert_eq!(ABOUT_PRELOAD[0], "Compiles every shader up front, then starts the game again.");
        assert_eq!(SAVED, "Saved. Applies at the next launch.");
        // This build has no DLSS or frame generation: their rows say so.
        for var in ["VICTORY_MENU_DLSS", "VICTORY_DLSS_PRESET", "VICTORY_MENU_FRAMEGEN", "VICTORY_MENU_FG_TYPE", "VICTORY_MFG_UNLOCKER"] {
            assert!(ITEMS[item(var)].label.ends_with("(not in this build)"), "{var}");
        }
        assert_eq!(ITEMS[item("VICTORY_MENU_RES")].label, "Resolution");
        assert!(ITEMS[item("VICTORY_MFG_UNLOCKER")].help[0].starts_with("A third-party mod"));
        // This build downloads nothing: the mod is used only where its file was put.
        assert!(ITEMS[item("VICTORY_MFG_UNLOCKER")].help.iter().all(|l| !l.contains("ownload")));
        assert!(ITEMS.iter().all(|i| (1..=3).contains(&i.help.len())));
        assert_eq!(ITEMS.iter().filter(|i| i.var == "VICTORY_MFG_UNLOCKER").count(), 1);
    }

    #[test]
    fn the_corner_shows_the_counter_and_the_status_and_nothing_when_both_are_off() {
        use status::Status;
        let at = std::time::Instant::now();
        // Off and nothing waiting: no lines, so nothing is made or held (`Menu::corner`).
        assert!(corner_lines_from(None, Status::Hidden, false).is_empty());
        let fps = || Some("FPS: 60.0".to_owned());
        assert_eq!(corner_lines_from(fps(), Status::Hidden, false), [("FPS: 60.0".to_owned(), art::LABEL)]);
        // The status under the counter, in its colour; alone when the counter is off.
        let both = corner_lines_from(fps(), Status::Preparing(12), false);
        assert_eq!(both[1], ("PREPARING SHADERS (12 left)".to_owned(), art::HEADING));
        assert_eq!(corner_lines_from(None, Status::Ready(at), false), [("SHADERS READY".to_owned(), art::VALUE)]);
        // With the details panel (F3) up, it says the status itself.
        assert_eq!(corner_lines_from(fps(), Status::Preparing(12), true).len(), 1);
        // Both fit the dot letters' corner box.
        assert!(both.iter().all(|(l, _)| l.chars().count() <= CORNER.0 as usize));
        let menu = play_defaults();
        assert_eq!(menu.held(), (0, 0));
    }

    #[test]
    fn values_cycle_both_ways() {
        let mut menu = play_defaults();
        let n = ITEMS[0].values.len();
        for _ in 0..n {
            menu.change(0, 1);
        }
        assert_eq!(menu.values[0], ITEMS[0].values[0].0);
        menu.change(0, -1);
        assert_eq!(menu.values[0], ITEMS[0].values[n - 1].0);
    }

    #[test]
    fn a_changed_row_says_when_it_applies() {
        let mut menu = play_defaults();
        // Unchanged: the note says what `*` means; a row that applies while playing says so.
        menu.at = item("VICTORY_FPS_LINE");
        assert!(menu.lines().iter().any(|l| l.starts_with("> FPS counter") && l.ends_with(" LIVE")));
        // Set here, not by `change`: that would change this process's environment at once.
        menu.values[menu.at] = "0".to_owned();
        let lines = menu.lines();
        assert!(lines.iter().any(|l| l.starts_with("> FPS counter") && l.ends_with(" *LIVE")), "{lines:#?}");
        assert_eq!(lines[lines.len() - 2], CHANGED_NOW);
        menu.at = item("VICTORY_MENU_DLSS");
        menu.change(menu.at, 1);
        let lines = menu.lines();
        assert!(lines.iter().any(|l| l.starts_with("> DLSS upscaling") && l.ends_with("< Balanced > *")), "{lines:#?}");
        assert_eq!(lines[lines.len() - 2], CHANGED_LATER);
        // The first value is what the launcher starts with: recommended, on the row picked only.
        menu.change(menu.at, -1);
        let lines = menu.lines();
        assert!(lines.iter().any(|l| l.starts_with("> DLSS upscaling") && l.ends_with("< Quality (recommended) >")), "{lines:#?}");
        assert!(lines.iter().any(|l| l.starts_with("  Resolution") && l.ends_with("< 2560 x 1440 (not in this build) >")), "{lines:#?}");
        // Not set: the default.
        assert_eq!(shown_value(&ITEMS[item("VICTORY_MENU_FPS")], "", true), "Default");
    }

    #[test]
    fn every_label_choice_and_help_line_fits_its_column() {
        let fixed = [TITLE, KEYS, LEGEND, CHANGED_LATER, CHANGED_NOW, REPO, CONFIRM_PRELOAD];
        let notes = [NO_FILE, SAVED, NOT_SAVED, RESTART_YOURSELF, RESTART_FAILED, RESET];
        let mut widest = 0;
        for line in fixed.iter().chain(&notes).chain(&ABOUT_PRELOAD).chain(&ACTION_HELP) {
            assert!(line.chars().count() <= COLUMNS as usize, "{line}");
            widest = widest.max(line.chars().count());
        }
        for action in ACTIONS {
            let line = format!("> {action}");
            assert!(line.len() <= COLUMNS as usize, "{line}");
            widest = widest.max(line.len());
        }
        for row in rows() {
            if let Row::Heading(group) = row {
                assert!(group.len() <= COLUMNS as usize, "{group}");
                widest = widest.max(group.len());
            }
        }
        // Each row, at each of its values, changed or not (the longest mark).
        let mut menu = menu_with(ITEMS.iter().map(|_| String::new()).collect());
        for (i, item) in ITEMS.iter().enumerate() {
            assert!(item.label.is_ascii() && item.label.len() <= LABEL, "{}", item.label);
            assert!((1..=3).contains(&item.help.len()), "{}", item.label);
            for line in item.help {
                assert!(!line.is_empty() && line.chars().count() <= COLUMNS as usize, "{line}");
                widest = widest.max(line.chars().count());
            }
            // Picked: the longest, with the recommended mark.
            menu.at = i;
            for (value, _) in item.values {
                menu.values[i] = (*value).to_owned();
                for started in [(*value).to_owned(), format!("not {value}")] {
                    menu.started[i] = started;
                    let line = menu.item_line(i, '>');
                    assert!(line.chars().count() <= COLUMNS as usize, "{line}");
                    widest = widest.max(line.chars().count());
                }
            }
        }
        // As wide as the widest line needs, and no wider.
        assert_eq!(widest, COLUMNS as usize);
        // The whole box fits a 1280 by 720 window at the scale it gets there (1), and the usual larger ones.
        assert_eq!(scale_for(720), 1);
        for (width, height) in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)] {
            let (w, h) = size(scale_for(height));
            assert!(w <= width && h <= height, "{w} x {h} in {width} x {height}");
        }
    }

    #[test]
    fn rows_are_in_groups_and_the_cursor_never_rests_on_a_heading() {
        // Each group's rows together, under one heading.
        let headings: Vec<&str> = rows().iter().filter_map(|r| if let Row::Heading(g) = r { Some(*g) } else { None }).collect();
        assert_eq!(headings, [GRAPHICS, FRAME_RATE, LIGHT, OVERLAY, INPUT, ADVANCED]);
        let total = ITEMS.len() + ACTIONS.len();
        for step in [1, -1] {
            let mut menu = play_defaults();
            let mut seen = vec![false; total];
            for _ in 0..2 * total {
                menu.go(step);
                seen[menu.at] = true;
                let lines = menu.lines();
                let cursor: Vec<usize> = (0..lines.len()).filter(|&l| lines[l].starts_with('>')).collect();
                assert_eq!(cursor.len(), 1, "{lines:#?}");
                let line = &lines[cursor[0]];
                assert!(!headings.contains(&line.as_str()) && !line[1..].trim().is_empty(), "{line}");
                // On an item: its line, with its group's heading in view above it.
                if let Some(item) = ITEMS.get(menu.at) {
                    assert!(line.starts_with(&format!("> {}", item.label)), "{line}");
                    assert!(lines[..cursor[0]].iter().any(|l| l == item.group), "{} has no heading over it: {lines:#?}", item.label);
                } else {
                    assert_eq!(line, &format!("> {}", ACTIONS[menu.at - ITEMS.len()]));
                }
            }
            assert!(seen.iter().all(|s| *s), "every line picked going {step}");
        }
    }

    #[test]
    fn the_bounce_light_row_writes_and_clears_exactly_the_five_variables() {
        let bounce = item(BOUNCE);
        // Off, the estimates' ties are their rows' play defaults.
        for (var, by_value) in ITEMS[bounce].ties {
            assert_eq!(by_value.len(), ITEMS[bounce].values.len(), "{var}");
            if let Some(row) = ITEMS.iter().find(|i| i.var == *var) {
                assert_eq!(by_value[0], row.values[0].0, "{var}");
            }
        }
        let mut menu = play_defaults();
        assert!(set_in(&menu.saved("")).is_empty());
        // On: the two switches set, the three estimates cleared, and their rows show it.
        menu.change(bounce, 1);
        assert_eq!(menu.values[bounce], "1");
        for var in ["VICTORY_AMBIENT", "VICTORY_AMBIENT_AIR", "VICTORY_AMBIENT_SPECULAR"] {
            assert_eq!(menu.values[item(var)], "", "{var}");
        }
        // A file's other lines are kept, and what it said of the five is replaced.
        let mut on = set_in(&menu.saved("VICTORY_MENU_DLSS=off\r\nVICTORY_AMBIENT=0.5\r\n"));
        assert_eq!(on.remove(0), ("VICTORY_MENU_DLSS".to_owned(), "off".to_owned()));
        let five_on = pairs(&[
            ("VICTORY_ACTIVATE_RAYS", "1"),
            ("VICTORY_AMBIENT", ""),
            ("VICTORY_AMBIENT_AIR", ""),
            ("VICTORY_AMBIENT_SPECULAR", ""),
            ("VICTORY_RT_RESOLVE", "1"),
        ]);
        assert_eq!(sorted(on), five_on);
        // Saved, then off: the play defaults of the five.
        menu.started = menu.values.clone();
        menu.change(bounce, -1);
        assert_eq!(menu.values[bounce], "");
        let five_off = pairs(&[
            ("VICTORY_ACTIVATE_RAYS", ""),
            ("VICTORY_AMBIENT", "0.8"),
            ("VICTORY_AMBIENT_AIR", "1.5"),
            ("VICTORY_AMBIENT_SPECULAR", "1"),
            ("VICTORY_RT_RESOLVE", ""),
        ]);
        assert_eq!(sorted(set_in(&menu.saved(""))), five_off);
        // Another row's change writes that row's variable alone.
        let mut menu = play_defaults();
        menu.change(item("VICTORY_AMBIENT"), 1);
        assert_eq!(set_in(&menu.saved("")), pairs(&[("VICTORY_AMBIENT", "1")]));
    }

    /// The sizes of window the panel is checked at.
    const WINDOWS: [(u32, u32); 4] = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)];

    /// The longest note the panel can show, and every other.
    fn notes() -> Vec<&'static str> {
        vec![LEGEND, CHANGED_LATER, CHANGED_NOW, CONFIRM_PRELOAD, NO_FILE, SAVED, NOT_SAVED, RESTART_YOURSELF, RESTART_FAILED, RESET]
    }

    #[test]
    fn the_panel_shows_the_same_rows_and_words() {
        let mut menu = play_defaults();
        for at in 0..ITEMS.len() + ACTIONS.len() {
            menu.at = at;
            let page = menu.page();
            assert_eq!(page.entries.len(), rows().len());
            let picked: Vec<usize> = (0..page.entries.len())
                .filter(|&i| matches!(page.entries[i], panel::Entry::Line { picked: true, .. }))
                .collect();
            assert_eq!(picked.len(), 1);
            assert_eq!(rows()[picked[0]], Row::Pick(at));
            assert_eq!(page.help, menu.help());
            // Recommended on the row picked only.
            for (i, entry) in page.entries.iter().enumerate() {
                if let (panel::Entry::Line { value, picked, .. }, Row::Pick(p)) = (entry, rows()[i]) {
                    assert_eq!(value.ends_with("(recommended)"), *picked && p < ITEMS.len(), "{value}");
                }
            }
        }
        // Nothing changed: no note; a change: the note says when it applies.
        assert_eq!(menu.page().note, None);
        menu.at = item("VICTORY_MENU_DLSS");
        menu.change(menu.at, 1);
        assert_eq!(menu.page().note.as_deref(), Some(CHANGED_LATER));
        menu.at = 0;
        menu.go(-1);
        assert_eq!(menu.page().note.as_deref(), Some(LEGEND));
    }

    #[test]
    fn every_label_choice_help_and_note_fits_the_panel() {
        let art = art::tests::made();
        for window in WINDOWS {
            let scaled = art.at(art::panel_size(window));
            let menu = play_defaults();
            let page = menu.page();
            let (text_width, keys) = panel::widths(&scaled, &page);
            assert!(keys <= text_width, "the keys at {window:?}");
            let gap = scaled.row.width("  ");
            let arrow = scaled.row.width("\u{2039}");
            let w = scaled.size.0 as f32;
            for item in ITEMS {
                let marks = if item.live { ["", "  * Live"] } else { ["", "  *"] };
                for (value, _) in item.values {
                    for picked in [false, true] {
                        let shown = shown_value(item, value, picked);
                        let (end, start) = panel::spans(&scaled, item.label, marks[1], &shown, picked);
                        // The left arrow sits at its place, or left of a long value: never on the label.
                        let left = if picked { (art::X_ARROW_LEFT * w).min(start - scaled.row.width(" ") - arrow) } else { start };
                        assert!(end + gap <= left, "{} < {shown} > at {window:?}: {end} {left}", item.label);
                    }
                }
            }
            for row in rows() {
                if let Row::Heading(group) = row {
                    assert!(scaled.small.width(&group.to_uppercase()) <= text_width, "{group}");
                }
            }
            // Help and the longest note under the list: room for at least six lines above them.
            let mut menu = play_defaults();
            for at in 0..ITEMS.len() + ACTIONS.len() {
                menu.at = at;
                for note in notes() {
                    let mut page = menu.page();
                    page.note = Some(note.to_owned());
                    let (n, bottom) = panel::lines_under(&scaled, &page);
                    assert!(n <= 5, "{n} lines under the list at {at}, {window:?}");
                    assert!(bottom - art::FIRST_BASELINE_Y >= 5.0 * art::ROW_PITCH, "{at}: {bottom}");
                    for line in page.help.iter().copied().chain([note]) {
                        for part in panel::wrapped(&scaled.small, line, text_width) {
                            assert!(scaled.small.width(&part) <= text_width || !part.contains(' '), "{part}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_picked_line_and_its_heading_are_always_on_the_panel() {
        let mut menu = play_defaults();
        for at in 0..ITEMS.len() + ACTIONS.len() {
            menu.at = at;
            let page = menu.page();
            let (lo, hi) = panel::window(&page.entries, art::LAST_BASELINE_Y - 3.0 * (art::KEYS_BASELINE_Y - art::HELP_BASELINE_Y));
            let picked = rows().iter().position(|r| *r == Row::Pick(at)).unwrap_or(usize::MAX);
            assert!((lo..=hi).contains(&picked), "{at}: {lo}..={hi}");
            if let Some(item) = ITEMS.get(at) {
                let heading = rows().iter().position(|r| *r == Row::Heading(item.group)).unwrap_or(usize::MAX);
                assert!((lo..=hi).contains(&heading), "{}: {lo}..={hi}", item.label);
            }
            assert!(hi - lo >= 6, "{at}: {lo}..={hi}");
        }
    }

    #[test]
    fn the_panel_is_drawn_with_the_bar_and_the_lettering() {
        let art = art::tests::made();
        let scaled = art.at(art::panel_size((1920, 1080)));
        let mut menu = play_defaults();
        menu.at = item("VICTORY_MENU_FRAMEGEN");
        let mut canvas = Vec::new();
        panel::compose(&scaled, &menu.page(), &mut canvas);
        assert_eq!(canvas.len(), scaled.panel.len());
        let (w, h) = (scaled.size.0 as usize, scaled.size.1 as usize);
        let px = |x: usize, y: usize| [canvas[(y * w + x) * 4], canvas[(y * w + x) * 4 + 1], canvas[(y * w + x) * 4 + 2]];
        // The bar: its fill at its right end, on some row of the list.
        let x = (art::BAR_RIGHT * w as f32) as usize - 2;
        let bar: Vec<usize> = (0..h).filter(|&y| px(x, y) == art::BAR_FILL).collect();
        assert!(!bar.is_empty() && bar.len() <= (art::BAR_HEIGHT * h as f32).ceil() as usize + 1, "{}", bar.len());
        // Lettering: pixels the panel did not have, in the label's colour, left of the middle.
        let lettered = (0..h).flat_map(|y| (0..w / 2).map(move |x| (x, y))).filter(|&(x, y)| px(x, y) == art::LABEL).count();
        assert!(lettered > 1000, "{lettered}");
        // A change of row moves the bar.
        menu.go(1);
        let mut again = Vec::new();
        panel::compose(&scaled, &menu.page(), &mut again);
        assert!(again != canvas);
    }

    #[test]
    fn with_the_switch_at_0_the_picture_is_the_dot_letters_exactly() {
        let Some(host) = super::super::host() else {
            eprintln!("no host device: not run");
            return;
        };
        let mut menu = Menu { changed: false, ..Menu::with(ITEMS.iter().map(|i| i.values[0].0.to_owned()).collect(), false) };
        assert!(!menu.panel);
        for (format, height) in [(DXGI_FORMAT_B8G8R8A8_UNORM, 720), (DXGI_FORMAT_R16G16B16A16_FLOAT, 2160)] {
            menu.shown = menu.show(&mut Blend::default(), host, (format, height * 16 / 9, height), None);
            assert!(matches!(menu.shown.as_ref().map(|s| &s.look), Some(Look::Dots { .. })));
            menu.paint();
            let lines = menu.lines();
            let shown = menu.shown.as_mut().expect("shown");
            let scale = scale_for(height);
            let colours = colours(format).expect("a format of ours");
            let expected = painted(&lines, (COLUMNS, LINES), scale, colours, shown.buffers.pitch as usize);
            // SAFETY: the upload buffer just written, of this length.
            let written = unsafe { std::slice::from_raw_parts(shown.buffers.uploads[shown.buffers.turn].1, expected.len()) };
            assert!(written == expected.as_slice(), "{format:?}");
            menu.close();
        }
    }

    #[test]
    fn the_menu_holds_nothing_once_closed() {
        let Some(host) = super::super::host() else {
            eprintln!("no host device: not run");
            return;
        };
        let mut menu = Menu { changed: false, ..Menu::with(ITEMS.iter().map(|i| i.values[0].0.to_owned()).collect(), true) };
        assert_eq!(menu.held(), (0, 0));
        menu.shown = menu.show(&mut Blend::default(), host, (DXGI_FORMAT_R10G10B10A2_UNORM, 3840, 2160), Some(art::tests::made()));
        menu.paint();
        let (cpu, card) = menu.held();
        // The art at 1600 x 1200 and the picture drawn on it, and five buffers of the picture.
        assert!(cpu >= 2 * 1600 * 1200 * 4, "{cpu}");
        assert_eq!(card, 1600 * 4 * 1200 * (UPLOADS as u64 + 1));
        eprintln!("held while open at 3840 x 2160: {cpu} bytes on the processor's side, {card} for the card");
        menu.close();
        assert!(menu.shown.is_none());
        assert_eq!(menu.held().0, 0, "the art and the picture are let go at once");
        assert_eq!(menu.leaving.len(), 1, "the buffers wait for the card");
        for _ in 0..LEAVE {
            menu.leave();
        }
        assert_eq!(menu.held(), (0, 0));
        assert!(menu.leaving.is_empty() && menu.corner.is_none());
    }

    /// What a change costs at 3840 by 2160 (and 1920 by 1080): the art read and
    /// scaled when the menu opens, the picture made on a change, for each swap
    /// chain format. With `VICTORY_MENU_ART`: the real art (else art made here,
    /// of the same sizes); with `VICTORY_MENU_PROOF=<file.ppm>`: the picture at
    /// 1920 by 1080 written there. Run in an optimised profile:
    /// `cargo test --profile diag -p victory-gpu --lib menu::tests::cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, in an optimised build"]
    fn cost() {
        use std::time::Instant;
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let real = std::env::var_os("VICTORY_MENU_ART").map(std::path::PathBuf::from);
        let read = |dir: &std::path::Path| {
            let t = Instant::now();
            let art = art::Art::read(dir).expect("the art");
            (art, ms(t))
        };
        let (art, read_ms) = match &real {
            Some(dir) => read(dir),
            None => (art::tests::made(), 0.0),
        };
        println!("art: {}; read and decoded in {read_ms:.1} ms", if real.is_some() { "real" } else { "made here" });
        let mut menu = play_defaults();
        menu.at = item("VICTORY_MENU_FRAMEGEN");
        for window in [(3840, 2160), (1920, 1080)] {
            let t = Instant::now();
            let scaled = art.at(art::panel_size(window));
            let scale_ms = ms(t);
            let mut canvas = Vec::new();
            panel::compose(&scaled, &menu.page(), &mut canvas);
            let (w, h) = scaled.size;
            println!("{window:?}: panel {w} x {h}, scaled in {scale_ms:.1} ms, holding {} bytes with the picture", scaled.bytes() + canvas.len());
            for format in [DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT] {
                let pitch = (w * picture::pixel_bytes(format).unwrap_or(4)).next_multiple_of(256) as usize;
                let mut out = vec![0u8; pitch * h as usize];
                let runs = 30;
                let (mut compose_ms, mut write_ms) = (Vec::new(), Vec::new());
                for i in 0..runs {
                    menu.go(if i % 2 == 0 { 1 } else { -1 });
                    let page = menu.page();
                    let t = Instant::now();
                    panel::compose(&scaled, &page, &mut canvas);
                    compose_ms.push(ms(t));
                    let t = Instant::now();
                    picture::write_out(&canvas, scaled.size, format, &mut out, pitch);
                    write_ms.push(ms(t));
                }
                let mid = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    (v[v.len() / 2], v[v.len() - 1])
                };
                let ((c, c_most), (o, o_most)) = (mid(&mut compose_ms), mid(&mut write_ms));
                println!(
                    "  {format:?}: a change {:.2} ms (drawn {c:.2}, written out {o:.2}; slowest {:.2}), {} bytes a buffer",
                    c + o,
                    c_most + o_most,
                    out.len()
                );
            }
            if window == (1920, 1080)
                && let Some(proof) = std::env::var_os("VICTORY_MENU_PROOF")
            {
                let mut ppm = format!("P6 {w} {h} 255\n").into_bytes();
                ppm.extend(canvas.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
                std::fs::write(proof, ppm).expect("the proof written");
            }
        }
    }
}
