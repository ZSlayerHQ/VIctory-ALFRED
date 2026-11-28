//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing. Its program that resolves the diffuse light
//! of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the
//! sky's light outdoors, nothing indoors (walls are black where no lamp or
//! sun reaches) and, in blocks of the picture's tiles, what the probes'
//! memory held. Right after that program the light is raised to an estimate
//! and bounded by a multiple of it.
//!
//! That a stand-in can be put there, and in which of the program's images,
//! is beloooo's find (VIctory-Belobug-AMD, 06031237, 4f45ab88, cd9f78ce,
//! GPL-3.0-or-later), which puts a constant there, shaded in creases from the
//! scene's depth. The estimate here is worked out from the frame instead, so
//! it follows the hour, the room and its lamps by itself: the light the
//! visible surfaces send out (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before) is averaged in 64x36
//! cells, and every point gets the mean of the cells weighted by the share of
//! its surroundings they cover, as a directional value (first-order harmonics
//! at 160x90 points, taken between them by depth).
//!
//! What it assumes: what is not on screen sends out what is (with a slow
//! mean of the last seconds, so that turning into a dark corner is not black
//! at once). No occlusion but the creases'.
//!
//! `VICTORY_AMBIENT`:
//! - unset or `0`/`off`: the title's own output.
//! - a number: that share of the estimate (1: all of it).
//! - `constant:<level>`: the constant, as beloooo's fork has it.
//!
//! `VICTORY_AMBIENT_SHADE=<strength>,<radius in metres>` sets the creases.
//! `VICTORY_AMBIENT_ONCE=1`: one bounce of the title's own light, without the
//! stand-in's light coming back to it ([`once`]).
//!
//! The air of a room is lit by the same probes: without rays they hold the
//! sky's light indoors too, a brown veil at sunset. `VICTORY_AMBIENT_AIR`
//! bounds what the air scatters by a multiple of the light around.
//!
//! Reflections are traced too, at half the picture's size: rays the
//! title's screen-space march resolves (`CS_ResolveScreenSpace`) take the
//! light seen on screen; the others go to the structures, find nothing, and
//! `CS_RayMiss` gives each the sky's light from the sky's cube maps (in a
//! cutscene frame, capture haze2: 10 to 30 against 0.01 to 0.1 for the room
//! the eyes and floors are in, and blue). `VICTORY_AMBIENT_SPECULAR` puts in,
//! right after that program, the same estimate taken along the reflected
//! direction ([`Role::Miss`]); the screen-space resolve that comes next
//! still writes what it found over it. That is the opaque layer's; the alpha
//! layer's misses are left as the title has them unless
//! `VICTORY_AMBIENT_SPECULAR_ALPHA` says otherwise ([`alpha`]: the estimate,
//! or the title's sky bounded by a multiple of the light around).

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host};

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program
    /// that writes that volume before it is summed up along the view).
    Air,
    /// Gives the reflection rays that found nothing the sky's light
    /// (`CS_RayMiss`, not `CS_RayMiss_Volumetrics` nor the screen probes').
    Miss,
}

/// The program's part in this, by its name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else if is_miss(container) {
        Some(Role::Miss)
    } else {
        None
    }
}

/// `CS_RayMiss_` and the eight hex digits of a variant's hash.
fn is_miss(container: &[u8]) -> bool {
    const NAME: &[u8] = b"CS_RayMiss_";
    container.windows(NAME.len() + 9).any(|w| {
        w.starts_with(NAME) && w[NAME.len()..NAME.len() + 8].iter().all(u8::is_ascii_hexdigit) && w[NAME.len() + 8] == b'_'
    })
}

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers)
/// and reads the scene's depth, and where the other writes the light of
/// visible surfaces: root parameter and place in its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
/// Where the light the air scatters is.
pub(crate) const AIR: (u8, u32) = (1, 32);
/// Where the miss program writes the reflected light (R11G11B10, half the
/// picture's size) and where it reads the depth it is given (a 32-bit depth
/// of the picture's size).
pub(crate) const MISS_LIGHT: (u8, u32) = (1, 104);
pub(crate) const MISS_DEPTH: (u8, u32) = (0, 1000);
/// The miss program's first root constant: which surfaces its rays are of.
pub(crate) const MISS_OPAQUE: u32 = 0;
pub(crate) const MISS_ALPHA: u32 = 1;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default()))
}

/// Whether a stand-in is on (for the diffuse light or for reflections):
/// set, and the title's rays find nothing.
pub(crate) fn on() -> bool {
    (set() != Set::Off || specular().is_some()) && !crate::pipeline::builder_runs()
}

/// `VICTORY_AMBIENT_SPECULAR=<share>[,<sky>[,occluded]]`: reflections whose
/// rays found nothing take `share` of the estimate along the reflected
/// direction instead of the sky's light; `sky` (0 to 1, 0 if not given) of
/// them stays the title's sky, made no brighter than the estimate. Unset or
/// 0: the title's sky. `occluded`: the estimate is no brighter than the
/// diffuse light at the point and less where the reflected ray runs into the
/// depth (the `OCCLUDED` part of `reflect_light`); without it, the passes as
/// they were.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Specular {
    share: f32,
    sky: f32,
    occluded: bool,
}

fn parse_specular(set: &str) -> Option<Specular> {
    let mut fields = set.split(',');
    let number = |t: Option<&str>| t.and_then(|t| t.trim().parse::<f32>().ok()).filter(|v| v.is_finite());
    let share = number(fields.next()).filter(|v| *v > 0.0)?;
    let sky = number(fields.next()).map_or(0.0, |v| v.clamp(0.0, 1.0));
    let occluded = fields.next().is_some_and(|t| t.trim() == "occluded");
    Some(Specular { share, sky, occluded })
}

fn specular() -> Option<Specular> {
    static SET: OnceLock<Option<Specular>> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_SPECULAR").unwrap_or_default();
        if set.split(',').nth(2).is_some_and(|t| !t.trim().is_empty() && t.trim() != "occluded") {
            tracing::warn!(value = %set, "VICTORY_AMBIENT_SPECULAR: a third field that is not `occluded` is not taken");
        }
        parse_specular(&set)
    })
}

/// What becomes of the alpha layer's reflections (glass, the eyes' wet
/// layer, decals, foam and wakes on water) whose rays found nothing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Alpha {
    /// The title's sky light, whatever is around (as the base has it).
    Title,
    /// The estimate along the reflected direction, as opaque surfaces get
    /// (wrong on water: the depth read there is the bottom's).
    Estimate,
    /// The title's sky light, no brighter than this multiple of the light
    /// around ([`Role::Miss`]'s bound pass).
    Bound(f32),
}

/// The multiples a bound may be set to.
const BOUND_RANGE: std::ops::RangeInclusive<f32> = 0.1..=16.0;

/// `VICTORY_AMBIENT_SPECULAR_ALPHA`, parsed; `None`: not one of its forms.
fn parse_alpha(set: &str) -> Option<Alpha> {
    match set.trim() {
        "" | "0" | "off" | "title" => Some(Alpha::Title),
        "1" | "estimate" => Some(Alpha::Estimate),
        "bound" => Some(Alpha::Bound(1.0)),
        set => set
            .strip_prefix("bound:")
            .and_then(|k| k.trim().parse::<f32>().ok())
            .filter(|k| BOUND_RANGE.contains(k))
            .map(Alpha::Bound),
    }
}

/// `VICTORY_AMBIENT_SPECULAR_ALPHA`:
/// - unset, `0`, `off` or `title`: the alpha layer is left as the title has
///   it;
/// - `1` or `estimate`: the estimate along the reflected direction, as the
///   opaque layer gets;
/// - `bound` or `bound:<k>` (k from 0.1 to 16): the title's sky light, no
///   brighter than 1 (or k) times the light around: the slow geometric mean
///   of the luminance of what the picture sends out. That reads neither the
///   surface's depth nor its normal, so the wrong (opaque) depth behind glass
///   and water does not matter.
///
/// Anything else: as unset, with a warning.
pub(crate) fn alpha() -> Alpha {
    static SET: OnceLock<Alpha> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_SPECULAR_ALPHA").unwrap_or_default();
        parse_alpha(&set).unwrap_or_else(|| {
            tracing::warn!(
                value = %set,
                "VICTORY_AMBIENT_SPECULAR_ALPHA is none of unset, 0, off, title, 1, estimate, bound, bound:<0.1 to 16>: \
                 the alpha layer is left as the title has it"
            );
            Alpha::Title
        })
    })
}

/// [`alpha`] as it can be had: a bound whose pipeline could not be made is
/// off (`make` said so), and the alpha layer is left as the title has it.
pub(crate) fn alpha_in_effect() -> Alpha {
    match alpha() {
        Alpha::Bound(_) if pass().is_none_or(|p| p.bound.is_none()) => Alpha::Title,
        a => a,
    }
}

/// What is done after a dispatch of the miss program ([`route`]).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum MissRoute {
    /// The estimate along the reflected direction (place, gather, reflect).
    Estimate,
    /// The title's sky light bounded by that multiple of the light around.
    Bound(f32),
    /// Left as the title had it: water, layers not known, the alpha layer
    /// with [`Alpha::Title`]. Counted as such.
    Left,
    /// The estimate is wanted but there is no depth it can be made from
    /// (none found, or one the format and size check refuses): left as the
    /// title had it, counted as not ready. Never another layer's path.
    NoDepth,
}

/// Where a dispatch of the miss program goes, by the layer its rays are of
/// (its first root constant), whether a depth was found in its tables,
/// whether that depth passed the format and size check, and the alpha
/// layer's switch ([`alpha_in_effect`]). [`MissRoute::Left`] does not depend
/// on the depth, so it can be decided before the tables are looked at.
pub(crate) fn route(layer: u32, depth_found: bool, depth_valid: bool, alpha: Alpha) -> MissRoute {
    let estimate = if depth_found && depth_valid { MissRoute::Estimate } else { MissRoute::NoDepth };
    match (layer, alpha) {
        (MISS_OPAQUE, _) | (MISS_ALPHA, Alpha::Estimate) => estimate,
        (MISS_ALPHA, Alpha::Bound(k)) => MissRoute::Bound(k),
        _ => MissRoute::Left,
    }
}

/// `VICTORY_AMBIENT_ONCE=1`: the estimate is one bounce of the light the
/// title's own lighting put there, not of the stand-in's too.
///
/// The light the visible surfaces send out (the estimate's source) is that of
/// the frame before, which the stand-in had raised: its own light comes back
/// to it, reflected. With a share `level` of what surfaces send out, and
/// surfaces sending out a share `a` of the light they get (`e` their own),
/// the light settles at `level * e / (1 - level * a)` instead of
/// `level * (a * l + e)`: at 0.8 and walls that send out 0.6, 1.9 times
/// one bounce, more near pale surfaces and lamps (the overexposed rooms).
/// With this switch the raise notes, at the 25 places of each cell that the
/// next keep reads, which share of the light there is the title's own; the
/// keep counts each place's light by that share. Light that is the
/// surface's own (lamps) is counted down as well where the stand-in raised
/// it: shortcut, no albedo to tell the two apart.
fn once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_AMBIENT_ONCE").is_some_and(|v| v == "1"))
}

/// The most the air scatters, as a multiple of the light around
/// (`VICTORY_AMBIENT_AIR=<multiple>`; unset or 0: what the title has).
fn air() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_AIR").unwrap_or_default();
        set.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v > 0.0).unwrap_or(0.0)
    })
}

/// Strength and radius of the creases' shade.
fn shade() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=12, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#ifdef DITHER_FRAME
#undef RS
#define RS "RootConstants(num32BitConstants=13, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#endif
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; float once;
#ifdef DITHER_FRAME
    uint frame;
#endif
};
#ifdef DITHER_FRAME
// The interleaved-gradient noise moved with the frame (`VICTORY_AMBIENT_DITHER=frame`), as its
// usual animated form has it: the texel shifted by 5.588238 per frame over 64 frames. At frame 0
// it is the still noise.
#define DITHERED(p) frac(52.9829189 * frac(dot(float2(p) + 5.588238 * float(frame % 64), float2(0.06711056, 0.00583715))))
#endif
Texture2D<float> depth : register(t0);
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are, from SHARES on the share of the title's own
// light at each of the 25 places of a cell the keep reads (x; y: written).
#define SHARES (2 * CELLS + 1)
// After the shares, with a bound on (GEO_MEAN): the slow geometric mean of
// the cells' luminance (x; w: written).
#define GEO (SHARES + 25 * CELLS)
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean and a direction.
RWStructuredBuffer<float4> points : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

// The surface's direction at a texel (towards the eye), from the depth around it: the
// neighbours on the side nearer its own depth. With NORMAL_SPAN (texels, `VICTORY_AMBIENT_NORMALS`)
// it is the mean of that and the directions across 2, 4, ... NORMAL_SPAN texels on each side,
// stopping at the first span that crosses a jump in depth: a curved mesh's faces then blend
// into one another instead of each taking one flat direction, and edges keep their own.
float3 facing(int2 p, float2 size, float3 c) {
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
#if NORMAL_SPAN > 1
    // A texel's width in metres at that depth; a span may change depth by 8 of them a texel.
    float texel = 2.0 * tan_y * c.z / size.y;
    float3 sum = n;
    [loop] for (int s = 2; s <= NORMAL_SPAN; s *= 2) {
        float3 l2 = seen(p + int2(-s, 0), size), r2 = seen(p + int2(s, 0), size);
        float3 u2 = seen(p + int2(0, -s), size), d2 = seen(p + int2(0, s), size);
        float most = 8.0 * s * texel;
        if (abs(l2.z - c.z) > most || abs(r2.z - c.z) > most || abs(u2.z - c.z) > most || abs(d2.z - c.z) > most) break;
        float3 m = normalize(cross(d2 - u2, r2 - l2) + float3(0, 0, -1e-9));
        if (dot(m, c) > 0.0) m = -m;
        sum += m;
    }
    n = normalize(sum);
#endif
    return n;
}

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void keep(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        float3 c = max(radiance[p], 0.0);
        c = any(isnan(c)) || any(isinf(c)) ? 0.0 : c;
        // Lamps and glints count, but not for everything.
        c /= max(1.0, dot(c, LUMA) / 20.0);
        // Not the stand-in's own light, reflected (`once`).
        float4 share = cells[SHARES + (at.y * CX + at.x) * 25 + y * 5 + x];
        sum += once > 0.0 && share.y > 0.0 ? c * saturate(share.x) : c;
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void mean() {
    float3 sum = 0;
#ifdef GEO_MEAN
    // Also the mean of the logarithms of the luminance of the cells that send
    // out anything: a window with the sunset in it is a third of a cell and
    // 200 times the room, and the plain mean is the window's; this one is the
    // room's. Only with a bound on (it is 2,304 logarithms in one thread).
    float logs = 0, lit = 0;
    for (uint i = 0; i < CELLS; i++) {
        sum += cells[i].rgb;
        float l = dot(cells[i].rgb, LUMA);
        if (l > 1e-5) { logs += log(l); lit += 1.0; }
    }
#else
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
#endif
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, follow) : now, 1.0);
#ifdef GEO_MEAN
    float4 wasg = cells[GEO];
    float g = lit > 0.0 ? exp(logs / lit) : 0.0;
    cells[GEO] = float4(wasg.a > 0.0 ? lerp(wasg.x, g, follow) : g, 0.0, 0.0, 1.0);
#endif
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void place(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    cells[CELLS + 1 + at.y * CX + at.x] = float4(p, p.z * p.z);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void gather(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, light.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, light.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, light.b);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is).
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void clear_air(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef REFLECT
RWTexture2D<float3> reflected : register(u0);
#ifdef OCCLUDED
// How open a point is, as `raise` takes it (the creases' shade).
float open_at(int2 p, float2 size, float3 c, float3 n) {
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
#ifdef DITHER_FRAME
    noise = DITHERED(p);
#endif
    if (!(strength > 0.0 && px >= 1.5 && c.z < 400.0)) return 1.0;
    float sum = 0.0;
    const int N = 12;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
// How open the reflected ray's way out is: marched over the depth in 5 steps
// (closer together near the point), out to a metre or up to 4 for points
// farther away. At the first step behind the depth (by more than a little,
// by less than a wall's thickness) the ray is taken as blocked and the fill
// scaled down, the more the nearer the hit: 0.3 at the point, 0.7 at the
// march's end. Off the screen, towards the eye, or clear: open.
//
// Scaled, not the light of the hit's point: that light is in the cells only,
// 64x36 of the frame before, and a hit's cell can hold a window or a lamp
// (the bright squares again); and where the hit is on screen the title's own
// screen-space resolve, which comes next, writes it over this texel anyway.
// What stays of this value is where that march found nothing, and there a
// darker fill is the safer guess for a ray into a wall or a corner.
float ray_open(float3 c, float3 ray, float2 size) {
    float reach = clamp(0.3 * c.z, 1.0, 4.0);
    float aspect = size.x / size.y;
    [unroll] for (int i = 1; i <= 5; i++) {
        float t = reach * float(i * i) / 25.0;
        float3 q = c + ray * t;
        if (q.z <= near) return 1.0;
        float2 ndc = float2(q.x / (tan_y * aspect * q.z), -q.y / (tan_y * q.z));
        if (any(abs(ndc) >= 1.0)) return 1.0;
        float z = seen(int2((ndc * 0.5 + 0.5) * size), size).z;
        float behind = q.z - z;
        if (behind > 0.01 + 0.01 * z && behind < 0.5 + 0.25 * z) return 0.3 + 0.4 * t / reach;
    }
    return 1.0;
}
#endif
// Each texel of the reflections' image (half the picture's size): the light
// the points around send along the reflected direction, as `raise` takes it
// along the normal; `level` is the share, `constant` the share of the
// title's sky kept, made no brighter than that.
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void reflect_light(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float2 size = float2(width, height);
    int2 p = int2((float2(at.xy) + 0.5) * size / float2(w, h));
    float3 c = seen(p, size);
    // Nothing is reflected by the sky.
    if (c.z >= 999.0) return;
    float3 n = facing(p, size, c);
    float3 v = normalize(c);
    float3 ray = v - 2.0 * dot(v, n) * n;
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float wgt = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += wgt * points[at0];
        sg += wgt * points[at0 + 1];
        sb += wgt * points[at0 + 2];
        sum += wgt;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    // Radiance from the first moments: three times the direction's share
    // for an even spread of directions, a little less for a smooth image.
    float3 found = all + 2.0 * float3(dot(sr.xyz, ray), dot(sg.xyz, ray), dot(sb.xyz, ray)) / sum;
#ifdef OCCLUDED
    // No brighter than the diffuse light at the point (`raise`'s level there,
    // with the reflections' share for the diffuse one: this pass is not given
    // that), instead of 3 times the points' mean; less where the ray runs
    // into the depth.
    float3 diffuse = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    diffuse = clamp(diffuse, 0.2 * all, 2.5 * all);
    diffuse = lerp(diffuse, cells[CELLS].rgb, 0.2) * level * open_at(p, size, c, n);
    found = max(found, 0.1 * all);
    found = min(lerp(found, cells[CELLS].rgb, 0.2) * level, diffuse) * ray_open(c, ray, size);
#else
    found = clamp(found, 0.1 * all, 3.0 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level;
#endif
    found = any(isnan(found)) || any(isinf(found)) ? 0.0 : max(found, 0.0);
    float3 was = reflected[at.xy];
    was = any(isnan(was)) || any(isinf(was)) ? 0.0 : max(was, 0.0);
    float has = dot(was, LUMA);
    was *= min(1.0, dot(found, LUMA) / max(has, 1e-6));
    reflected[at.xy] = lerp(found, was, constant);
}
#endif

#ifdef BOUND
RWTexture2D<float3> reflected : register(u0);
// The alpha layer's reflections keep the title's sky light, but no more than
// `level` times the light around: the slow geometric mean of the luminance of
// what the picture sent out (GEO; until a mean has been written, the plain
// mean's luminance), one value for the whole picture. Nothing of the texel's
// own depth or normal is read. What is taken away goes towards the colour
// around, so what is left of a blue sky indoors is not a blue sheen. Both
// ends of the blend have luminance `cap`, so the result has too.
//
// Adapted from upstream VIctory pull request #63 (its BOUND pass and log
// mean; `Pass::bound`).
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void bound_sky(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float3 was = reflected[at.xy];
    if (any(isnan(was)) || any(isinf(was))) { reflected[at.xy] = 0.0; return; }
    was = max(was, 0.0);
    float has = dot(was, LUMA);
    if (has <= 0.0) return;
    float3 room = max(cells[CELLS].rgb, 0.0);
    float4 geo = cells[GEO];
    float around = geo.w > 0.0 ? geo.x : dot(room, LUMA);
    float cap = level * around;
    if (has <= cap) return;
    float kept = cap / has;
    float3 toned = room * (cap / max(dot(room, LUMA), 1e-6));
    reflected[at.xy] = lerp(toned, was * kept, kept);
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void raise(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 n = facing(p, size, c);
    // How open it is (as in beloooo's cd9f78ce).
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
#ifdef DITHER_FRAME
    noise = DITHERED(p);
#endif
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(round(o)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    // Between the four points around, those at its own depth first; which
    // four is dithered a little.
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * points[at0];
        sg += w * points[at0 + 1];
        sb += w * points[at0 + 2];
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level * open;
    float3 was = unpack(light[p]);
    // Magenta/violet garbage the probes' memory held (green far under red and
    // blue): dropped, as VIctory-Belobug-AMD does (ambient.rs, "Filter magenta
    // garbage probes"). Seen on an RTX 5080 as a purple-blue glow on a lit
    // sheet of paper in an indoor scene; not an AMD-only fault.
    if (was.g < 0.5 * min(was.r, was.b) && max(was.r, was.b) > 0.02) was = 0.0;
    // And the bluer kind of it (blue over 2.5x green and over red): left as
    // a faint violet by the line above. The sky's light keeps green above
    // 0.4x blue, sunsets keep blue low, so neither is touched.
    if (was.b > 2.5 * was.g && was.b > was.r && was.b > 0.02) was = 0.0;
    // What the program left counts as far as it can be true: the sky's light
    // outdoors, not the blocks of other memory.
    float top = max(dot(found, LUMA) * most, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6));
    float3 now = max(was, found);
    light[p] = pack(now);
    if (once > 0.0) {
        // At the places the next keep reads, the share of the light that is
        // the title's own.
        float2 gc = (float2(p) + 0.5) / size * float2(CX, CY);
        uint2 cell = min(uint2(gc), uint2(CX - 1, CY - 1));
        uint2 k = min(uint2((gc - float2(cell)) * 5.0), uint2(4, 4));
        uint2 q = uint2((float2(cell) + (float2(k) + 0.5) / 5.0) * size / float2(CX, CY));
        if (q.x == uint(p.x) && q.y == uint(p.y)) {
            float share = dot(was, LUMA) / max(dot(now, LUMA), 1e-6);
            cells[SHARES + (cell.y * CX + cell.x) * 25 + k.y * 5 + k.x] = float4(share, 1.0, 0.0, 0.0);
        }
    }
}
#endif
"#;

/// Bytes of the cells' buffer and of the points'.
const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (2 * CELLS + 1 + 25 * CELLS) * 16;
const POINT_BYTES: u64 = 160 * 90 * 48;

/// The cells' buffer: one more value (the geometric mean, `GEO`) with a
/// bound on, as the base has it without.
const fn cell_bytes(geo: bool) -> u64 {
    if geo { CELL_BYTES + 16 } else { CELL_BYTES }
}

struct Pass {
    /// Made with `DITHER_FRAME`: the root signature has 13 constants.
    dither: bool,
    root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    reflect: ID3D12PipelineState,
    /// The alpha layer's bound ([`Alpha::Bound`], `bound_sky`): made only
    /// with it on, and then the mean keeps the geometric mean too.
    ///
    /// The bound and its log-mean luminance are adapted from upstream VIctory
    /// pull request #63. Taken: the BOUND pass and the log mean. Changed here: off unless
    /// asked for, routed on layer and checked depth, the log mean only with
    /// the bound on, a failed pipeline turns the bound off alone, its own
    /// count in the ten-second line.
    bound: Option<ID3D12PipelineState>,
    cells: ID3D12Resource,
    points: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

impl Pass {
    /// How many root constants its root signature has: the base's 12, and
    /// the frame with [`dither`].
    fn constants(&self) -> usize {
        if self.dither { 13 } else { 12 }
    }
}

/// `VICTORY_AMBIENT_NORMALS=<texels>` (2 to 32; unset, 0 or 1: off): the stand-in takes a
/// surface's direction over that span of the depth around it, not from its neighbours alone
/// (a normal from the nearest depth is one flat direction across each triangle, so a
/// curved mesh in shade shows its faces). Off by default: the passes as they were.
fn normal_span() -> u32 {
    static SPAN: OnceLock<u32> = OnceLock::new();
    *SPAN.get_or_init(|| parse_span(&std::env::var("VICTORY_AMBIENT_NORMALS").unwrap_or_default()))
}

fn parse_span(value: &str) -> u32 {
    value.trim().parse::<u32>().ok().filter(|&n| n >= 2).map_or(0, |n| n.min(32))
}

/// `VICTORY_AMBIENT_DITHER=frame`: the interleaved-gradient noise
/// that turns the creases' shade samples (and which points `raise` takes)
/// moves with the frame, so the upscaler averages it away instead of keeping
/// it as a still grain on the floor. Unset, `off`, `0` or anything else: off,
/// and the passes and their constants are the base's (`DITHER_FRAME` is not
/// defined, 12 root constants).
fn dither() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_DITHER").unwrap_or_default();
        if !matches!(set.trim(), "" | "off" | "0" | "frame") {
            tracing::warn!(value = %set, "VICTORY_AMBIENT_DITHER is none of unset, off, 0, frame: the dither stays still");
        }
        parse_dither(&set)
    })
}

fn parse_dither(value: &str) -> bool {
    value.trim() == "frame"
}

/// The source of one pass: `part` and the span defined, then `more` (defines
/// after the base's two; none: the base's source), then with `dither`
/// `DITHER_FRAME`. Without it the string is the base's.
fn source(part: &str, span: u32, more: &str, dither: bool) -> String {
    let dither = if dither { "#define DITHER_FRAME\n" } else { "" };
    format!("#define {part}\n#define NORMAL_SPAN {span}\n{more}{dither}{SOURCE}")
}

/// The passes. `bound`: the alpha layer's bound is wanted ([`Alpha::Bound`]);
/// if its pipeline cannot be made it is off, with a warning, and the rest
/// are made as without it. `occluded`: the reflections' estimate with its
/// march and cap ([`Specular`]'s third field). `dither`: the noise moves with
/// the frame ([`dither`]); every pass then has `DITHER_FRAME` defined and the
/// root signature a 13th constant, the frame.
fn make(
    device: &ID3D12Device,
    buffer: impl Fn(u64) -> Option<ID3D12Resource>,
    span: u32,
    bound: bool,
    occluded: bool,
    dither: bool,
) -> windows::core::Result<Pass> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |part: &str, more: &str, entry| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&source(part, span, more, dither), entry, s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let keep = pipeline("KEEP", "", s!("keep"))?;
    let bound = if bound {
        pipeline("BOUND", "", s!("bound_sky"))
            .inspect_err(|e| {
                tracing::warn!(
                    error = %e,
                    "no bound for the alpha layer's reflections (VICTORY_AMBIENT_SPECULAR_ALPHA): it is left as the \
                     title has it; the other stand-ins run as before"
                );
            })
            .ok()
    } else {
        None
    };
    let geo = bound.is_some();
    let mean = pipeline("KEEP", if geo { "#define GEO_MEAN\n" } else { "" }, s!("mean"))?;
    let place = pipeline("PLACE", "", s!("place"))?;
    let gather = pipeline("GATHER", "", s!("gather"))?;
    let raise = pipeline("RAISE", "", s!("raise"))?;
    let air = pipeline("AIR", "", s!("clear_air"))?;
    let reflect = pipeline("REFLECT", if occluded { "#define OCCLUDED\n" } else { "" }, s!("reflect_light"))?;
    let made = |bytes| buffer(bytes).ok_or_else(windows::core::Error::empty);
    Ok(Pass {
        dither,
        root: root.ok_or_else(windows::core::Error::empty)?,
        keep,
        mean,
        place,
        gather,
        raise,
        air,
        reflect,
        bound,
        cells: made(cell_bytes(geo))?,
        points: made(POINT_BYTES)?,
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let bound = matches!(alpha(), Alpha::Bound(_));
        let occluded = specular().is_some_and(|s| s.occluded);
        let dither = dither();
        if dither {
            tracing::info!("the stand-in's dither moves with the frame (VICTORY_AMBIENT_DITHER=frame)");
        }
        make(&host.device, |bytes| host.uav_buffer(bytes), normal_span(), bound, occluded, dither)
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the passes read and write, as views in the heap the list has set.
pub(crate) struct Views {
    /// The image the title's program just wrote.
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The scene's depth, for [`Role::Resolve`].
    pub(crate) depth: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The size of the image, and how many slices it has.
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
    /// The size of the depth (the image's but for [`Role::Miss`]).
    pub(crate) depth_size: [u32; 2],
}

/// For the line every ten seconds: what the stand-in for reflections did.
struct Said {
    /// Dispatches of the miss program it followed, frames they were in,
    /// and those it left as they were: of water, and before a frame's
    /// light was kept (or no tables).
    ran: AtomicU64,
    frames: AtomicU64,
    water: AtomicU64,
    not_ready: AtomicU64,
    last_frame: AtomicU64,
    /// Bit 0 opaque surfaces, 1 alpha, 2 water, 3 others.
    layers: AtomicU64,
    /// Dispatches of the alpha layer whose sky light was bounded (not
    /// counted in `ran`, `water` or `not_ready`).
    bounded: AtomicU64,
}

static SAID_MISS: Said = Said {
    ran: AtomicU64::new(0),
    frames: AtomicU64::new(0),
    water: AtomicU64::new(0),
    not_ready: AtomicU64::new(0),
    last_frame: AtomicU64::new(u64::MAX),
    layers: AtomicU64::new(0),
    bounded: AtomicU64::new(0),
};

/// What came of a dispatch of the miss program, counted for the line.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum MissNote {
    /// The estimate, on surfaces of that layer.
    Ran(u32),
    /// The alpha layer's sky light, bounded ([`MissRoute::Bound`]).
    Bounded,
    Water,
    NotReady,
}

/// Counts what came of a dispatch of the miss program; every ten seconds an
/// info line says it, so a run shows that the stand-in ran and where.
pub(crate) fn note_miss(note: MissNote) {
    use AtomicOrdering::Relaxed;
    if specular().is_none() {
        return;
    }
    let s = &SAID_MISS;
    match note {
        MissNote::Ran(layer) => {
            s.ran.fetch_add(1, Relaxed);
            s.layers.fetch_or(1 << layer.min(3), Relaxed);
            let frame = crate::frame::current();
            if s.last_frame.swap(frame, Relaxed) != frame {
                s.frames.fetch_add(1, Relaxed);
            }
        }
        MissNote::Bounded => {
            s.bounded.fetch_add(1, Relaxed);
        }
        MissNote::Water => {
            s.water.fetch_add(1, Relaxed);
        }
        MissNote::NotReady => {
            s.not_ready.fetch_add(1, Relaxed);
        }
    }
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    let Some(since) = *last else {
        *last = Some(now);
        return;
    };
    if now.duration_since(since).as_secs_f32() < 10.0 {
        return;
    }
    *last = Some(now);
    let layers = s.layers.swap(0, Relaxed);
    let named: Vec<&str> = [(1, "opaque"), (2, "alpha"), (4, "water"), (8, "other")]
        .iter()
        .filter(|(bit, _)| layers & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    let Specular { share, sky, occluded } = specular().unwrap_or(Specular { share: 0.0, sky: 0.0, occluded: false });
    // With the field, as it was set; without, the line is the base's.
    let sky = if occluded { format!("{sky},occluded") } else { sky.to_string() };
    // With a bound on, its own count; without, the line is the base's.
    let bounded = s.bounded.swap(0, Relaxed);
    let bounded = match alpha_in_effect() {
        Alpha::Bound(k) => format!(
            "{bounded} of the alpha layer kept the title's sky light, bounded at {k} times the light around \
             (VICTORY_AMBIENT_SPECULAR_ALPHA); "
        ),
        _ => String::new(),
    };
    tracing::info!(
        "a stand-in for reflections (VICTORY_AMBIENT_SPECULAR={share},{sky}) in the last {:.0} s: \
         {} frames, {} dispatches of CS_RayMiss followed (surfaces: {}), each by an estimate along \
         the reflected direction put in right after it (before CS_ResolveScreenSpace); {bounded}{} of water or the alpha layer \
         and {} before a frame's light was kept left as the title had them",
        now.duration_since(since).as_secs_f32(),
        s.frames.swap(0, Relaxed),
        s.ran.swap(0, Relaxed),
        if named.is_empty() { "none".to_owned() } else { named.join(", ") },
        s.water.swap(0, Relaxed),
        s.not_ready.swap(0, Relaxed),
    );
}

/// After the title's program of that role, in `list` with the program's view
/// heap set. Replaces the list's compute root signature and pipeline; false
/// if it recorded nothing. `bound`, for [`Role::Miss`]: the image is the
/// alpha layer's and only bounded, by that multiple ([`MissRoute::Bound`]).
pub(crate) fn after(list: &HostList, role: Role, views: &Views, bound: Option<f32>) -> bool {
    use std::sync::atomic::Ordering;
    let Some(pass) = pass() else {
        return false;
    };
    // The share of the estimate, and the constant (for reflections: the
    // share of the title's sky kept); the keep runs for either stand-in.
    // For the bound: its multiple, nothing of the title's sky blended.
    let (level, constant) = match (role, bound, set(), specular()) {
        (Role::Miss, Some(k), _, Some(_)) => (k, 0.0),
        (Role::Miss, None, _, Some(s)) => (s.share, s.sky),
        (Role::Miss, _, _, None) => return false,
        (_, Some(_), _, _) => return false,
        (Role::Radiance, None, Set::Off, Some(_)) => (0.0, 0.0),
        (_, None, Set::Off, _) => return false,
        (_, None, Set::Estimate(level), _) => (level, 0.0f32),
        (_, None, Set::Constant(level), _) => (0.0, level),
    };
    if bound.is_some() && pass.bound.is_none() {
        return false;
    }
    let estimate = role == Role::Miss || constant == 0.0;
    if role != Role::Radiance && estimate && !KEPT.load(Ordering::Relaxed) {
        if role == Role::Miss {
            note_miss(MissNote::NotReady);
        }
        return false;
    }
    if role == Role::Air && (constant > 0.0 || air() == 0.0) {
        return false;
    }
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 5 {
            *said += 1;
            tracing::info!(?role, size = ?views.size, level, constant, "a stand-in for the indirect light");
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    let [strength, radius] = shade();
    let floats = [level, constant, strength, radius, 0.1, 0.45];
    let mut words = [0u32; 13];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    words[6] = views.depth_size[0];
    words[7] = views.depth_size[1];
    words[8] = 0.04f32.to_bits();
    words[9] = 4.0f32.to_bits();
    words[10] = air().to_bits();
    words[11] = if once() { 1.0f32 } else { 0.0 }.to_bits();
    // With the dither moving, the frame (its low bits: the shader takes it
    // modulo 64); without, the base's 12 constants alone are given.
    words[12] = crate::frame::current() as u32;
    if !record(&list.list, pass, role, &words[..pass.constants()], views, constant, bound.is_some()) {
        return false;
    }
    if role == Role::Radiance {
        KEPT.store(true, Ordering::Relaxed);
    }
    true
}

/// The passes of `role`, recorded in `l` with the heap of `views` set and the
/// constants `words` (as many as [`Pass::constants`]). `bound`: for
/// [`Role::Miss`], the bound alone (false, with nothing recorded, if there is
/// no bound pipeline or another role).
fn record(
    l: &ID3D12GraphicsCommandList,
    pass: &Pass,
    role: Role,
    words: &[u32],
    views: &Views,
    constant: f32,
    bound: bool,
) -> bool {
    let bound = match (bound, role, &pass.bound) {
        (false, ..) => None,
        (true, Role::Miss, Some(pso)) => Some(pso),
        (true, ..) => return false,
    };
    if words.len() != pass.constants() {
        debug_assert!(false, "{} constants for a root signature of {}", words.len(), pass.constants());
        return false;
    }
    // SAFETY: an open list with the heap of the views set; the title's
    // program has just used both views, so their textures are in the states
    // these views need; buffers the list keeps.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, words.len() as u32, words.as_ptr().cast(), 0);
        // The bound reads no depth, and there may be none to give.
        if bound.is_none() {
            l.SetComputeRootDescriptorTable(1, views.depth);
        }
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
            }
            Role::Air => run(&pass.air, views.size[0].div_ceil(8), views.size[1].div_ceil(8), views.slices.div_ceil(2)),
            Role::Resolve => {
                if constant == 0.0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(&pass.raise, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
            Role::Miss => match bound {
                // The bound reads the cells' means alone, as the last keep
                // left them.
                Some(pso) => run(pso, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1),
                None => {
                    // The points afresh, from this frame's depth: the diffuse
                    // stand-in makes them later in the frame.
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                    run(&pass.reflect, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
                }
            },
        }
    }
    true
}

/// `VICTORY_AMBIENT_LIST_VIEWS=1` (diagnostics builds): after a dispatch of
/// a program with a [`Role`], the shader views and unordered views its root
/// tables hold, once per distinct set, so that one run names the title's
/// normal target among the resolve's views (a curved
/// mesh shows flat faces in shade where the direction is taken from the
/// depth). Each view: the root slot and place in the table (dwords), the
/// resource's dimension, size, layers or depth, levels and format (the host
/// copy's description, or the title's for a placed texture without one),
/// the view's format where it differs, and whether the resource is a render
/// target or a depth. Nothing is made for it: the host copies are those the
/// dispatch made (`host_made`, `texture_at_made`).
///
/// Off (unset, anything but `1`): nothing is read, logged or allocated.
#[cfg(feature = "diagnostics")]
pub(crate) mod views {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

    use super::Role;
    use crate::descriptor::View;
    use crate::object::This;

    fn parse(value: Option<&std::ffi::OsStr>) -> bool {
        value.is_some_and(|v| v == "1")
    }

    /// Whether the listing is on.
    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| parse(std::env::var_os("VICTORY_AMBIENT_LIST_VIEWS").as_deref()))
    }

    /// The DXGI formats' names, by number (0 to 115; the rest are video and
    /// feedback formats no view here has).
    const NAMES: [&str; 116] = [
        "UNKNOWN", "R32G32B32A32_TYPELESS", "R32G32B32A32_FLOAT", "R32G32B32A32_UINT", "R32G32B32A32_SINT",
        "R32G32B32_TYPELESS", "R32G32B32_FLOAT", "R32G32B32_UINT", "R32G32B32_SINT", "R16G16B16A16_TYPELESS",
        "R16G16B16A16_FLOAT", "R16G16B16A16_UNORM", "R16G16B16A16_UINT", "R16G16B16A16_SNORM", "R16G16B16A16_SINT",
        "R32G32_TYPELESS", "R32G32_FLOAT", "R32G32_UINT", "R32G32_SINT", "R32G8X24_TYPELESS",
        "D32_FLOAT_S8X24_UINT", "R32_FLOAT_X8X24_TYPELESS", "X32_TYPELESS_G8X24_UINT", "R10G10B10A2_TYPELESS",
        "R10G10B10A2_UNORM", "R10G10B10A2_UINT", "R11G11B10_FLOAT", "R8G8B8A8_TYPELESS", "R8G8B8A8_UNORM",
        "R8G8B8A8_UNORM_SRGB", "R8G8B8A8_UINT", "R8G8B8A8_SNORM", "R8G8B8A8_SINT", "R16G16_TYPELESS",
        "R16G16_FLOAT", "R16G16_UNORM", "R16G16_UINT", "R16G16_SNORM", "R16G16_SINT", "R32_TYPELESS", "D32_FLOAT",
        "R32_FLOAT", "R32_UINT", "R32_SINT", "R24G8_TYPELESS", "D24_UNORM_S8_UINT", "R24_UNORM_X8_TYPELESS",
        "X24_TYPELESS_G8_UINT", "R8G8_TYPELESS", "R8G8_UNORM", "R8G8_UINT", "R8G8_SNORM", "R8G8_SINT",
        "R16_TYPELESS", "R16_FLOAT", "D16_UNORM", "R16_UNORM", "R16_UINT", "R16_SNORM", "R16_SINT", "R8_TYPELESS",
        "R8_UNORM", "R8_UINT", "R8_SNORM", "R8_SINT", "A8_UNORM", "R1_UNORM", "R9G9B9E5_SHAREDEXP",
        "R8G8_B8G8_UNORM", "G8R8_G8B8_UNORM", "BC1_TYPELESS", "BC1_UNORM", "BC1_UNORM_SRGB", "BC2_TYPELESS",
        "BC2_UNORM", "BC2_UNORM_SRGB", "BC3_TYPELESS", "BC3_UNORM", "BC3_UNORM_SRGB", "BC4_TYPELESS", "BC4_UNORM",
        "BC4_SNORM", "BC5_TYPELESS", "BC5_UNORM", "BC5_SNORM", "B5G6R5_UNORM", "B5G5R5A1_UNORM", "B8G8R8A8_UNORM",
        "B8G8R8X8_UNORM", "R10G10B10_XR_BIAS_A2_UNORM", "B8G8R8A8_TYPELESS", "B8G8R8A8_UNORM_SRGB",
        "B8G8R8X8_TYPELESS", "B8G8R8X8_UNORM_SRGB", "BC6H_TYPELESS", "BC6H_UF16", "BC6H_SF16", "BC7_TYPELESS",
        "BC7_UNORM", "BC7_UNORM_SRGB", "AYUV", "Y410", "Y416", "NV12", "P010", "P016", "420_OPAQUE", "YUY2", "Y210",
        "Y216", "NV11", "AI44", "IA44", "P8", "A8P8", "B4G4R4A4_UNORM",
    ];

    fn name(format: u32) -> &'static str {
        NAMES.get(format as usize).copied().unwrap_or("?")
    }

    /// What a table entry is, for its line.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Found {
        /// A texture: its description, the view's format (0: the
        /// resource's), and whether the description is the title's (a
        /// placed texture with no host copy yet).
        Texture { desc: D3D12_RESOURCE_DESC, view: u32, title: bool },
        /// A buffer of that many bytes.
        Buffer(u64),
        /// A texture with no host copy made.
        NotMade,
        /// No view of a resource there (none, or a constant buffer or
        /// sampler where the program reads a view).
        Nothing,
    }

    /// What the view recorded at a table's place shows.
    pub(crate) fn found(view: Option<View>) -> Found {
        let pc = |words: Option<[u32; 10]>| words.map_or(0, |d| crate::resource::pc_format(d[0]));
        let texture = |r: This, view: u32| match crate::resource::host_made(r) {
            // SAFETY: a live resource.
            Some(host) => Found::Texture { desc: unsafe { host.GetDesc() }, view, title: false },
            None => match crate::resource::buffer_bytes(r) {
                Some((_, bytes)) => Found::Buffer(bytes),
                None => Found::NotMade,
            },
        };
        match view {
            Some(View::Shader(r, d)) => texture(r as This, pc(d)),
            Some(View::Unordered(r, _, d)) => texture(r as This, pc(d)),
            Some(View::Placed(_, r, _)) if r[0] == 1 => Found::Buffer(u64::from(r[4]) | u64::from(r[5]) << 32),
            Some(View::Placed(_, r, d)) => {
                let view = crate::resource::pc_format(d[0]);
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                match crate::resource::texture_at_made(address, &r) {
                    Some(made) => texture(made, view),
                    None => Found::Texture { desc: described(&r), view, title: true },
                }
            }
            _ => Found::Nothing,
        }
    }

    /// A resource description kept as words (`D3D12_RESOURCE_DESC`'s layout;
    /// words 1 and 13 are filler).
    fn described(r: &[u32; 14]) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION(r[0] as i32),
            Alignment: u64::from(r[2]) | u64::from(r[3]) << 32,
            Width: u64::from(r[4]) | u64::from(r[5]) << 32,
            Height: r[6],
            DepthOrArraySize: r[7] as u16,
            MipLevels: (r[7] >> 16) as u16,
            Format: DXGI_FORMAT(crate::resource::pc_format(r[8]) as i32),
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: r[9], Quality: r[10] },
            Layout: D3D12_TEXTURE_LAYOUT(r[11] as i32),
            Flags: D3D12_RESOURCE_FLAGS(r[12] as i32),
        }
    }

    /// One view's line: root slot and place in its table (dwords), kind,
    /// then what is there.
    pub(crate) fn line(slot: u8, offset: u32, uav: bool, found: &Found) -> String {
        let kind = if uav { "uav" } else { "srv" };
        let what = match *found {
            Found::Texture { desc, view, title } => {
                let (dimension, layers) = match desc.Dimension {
                    D3D12_RESOURCE_DIMENSION_TEXTURE1D => ("1d", "array"),
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D => ("2d", "array"),
                    D3D12_RESOURCE_DIMENSION_TEXTURE3D => ("3d", "depth"),
                    D3D12_RESOURCE_DIMENSION_BUFFER => ("buffer", "array"),
                    _ => ("?", "array"),
                };
                let format = desc.Format.0 as u32;
                let mut text = format!(
                    "{dimension} {}x{} {layers} {} mips {} {}({format})",
                    desc.Width,
                    desc.Height,
                    desc.DepthOrArraySize,
                    desc.MipLevels,
                    name(format)
                );
                if view != 0 && view != format {
                    text.push_str(&format!(" view {}({view})", name(view)));
                }
                if desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
                    text.push_str(" target");
                }
                if desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
                    text.push_str(" depth");
                }
                if desc.SampleDesc.Count > 1 {
                    text.push_str(&format!(" samples {}", desc.SampleDesc.Count));
                }
                if title {
                    text.push_str(" (title's description: no host copy)");
                }
                text
            }
            Found::Buffer(bytes) => format!("buffer {bytes} bytes"),
            Found::NotMade => "texture, no host copy".to_owned(),
            Found::Nothing => "-".to_owned(),
        };
        format!("s{slot}+{offset} {kind} {what}")
    }

    /// The most sets listed for a role.
    const MOST: usize = 32;

    /// The sets listed so far.
    #[derive(Default)]
    pub(crate) struct Seen {
        sets: HashSet<(String, Vec<String>)>,
        per_role: [usize; 4],
    }

    impl Seen {
        /// The number of `lines` among `role`'s sets listed (with `program`'s
        /// name) if they are a set not listed before and fewer than [`MOST`]
        /// were; `Err(())` the first time one is not for the cap.
        pub(crate) fn first(&mut self, role: Role, program: &str, lines: &[String]) -> Result<Option<usize>, ()> {
            let count = &mut self.per_role[role as usize];
            if *count > MOST {
                return Ok(None);
            }
            let key = (format!("{role:?} {program}"), lines.to_vec());
            if self.sets.contains(&key) {
                return Ok(None);
            }
            if *count == MOST {
                *count += 1;
                return Err(());
            }
            self.sets.insert(key);
            *count += 1;
            Ok(Some(*count - 1))
        }
    }

    /// Logs `lines`, the views of `program` (of `role`), if they are a set
    /// not listed before.
    pub(crate) fn note(role: Role, program: &str, lines: &[String]) {
        static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
        let first = SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Seen::default).first(role, program, lines);
        match first {
            Ok(Some(set)) => {
                let frame = crate::frame::current();
                tracing::info!(?role, program, set, frame, views = lines.len(), "stand-in views: a set of the program's tables");
                for view in lines {
                    tracing::info!(?role, set, "stand-in view {view}");
                }
            }
            Ok(None) => {}
            Err(()) => tracing::info!(?role, most = MOST, "stand-in views: more sets of this role are not listed"),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        #[test]
        fn the_switch() {
            assert!(!parse(None));
            assert!(!parse(Some("".as_ref())));
            assert!(!parse(Some("0".as_ref())));
            assert!(!parse(Some("yes".as_ref())));
            assert!(parse(Some("1".as_ref())));
        }

        #[test]
        fn names_by_number() {
            for (format, text) in [
                (DXGI_FORMAT_UNKNOWN, "UNKNOWN"),
                (DXGI_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM"),
                (DXGI_FORMAT_R10G10B10A2_UNORM, "R10G10B10A2_UNORM"),
                (DXGI_FORMAT_R11G11B10_FLOAT, "R11G11B10_FLOAT"),
                (DXGI_FORMAT_R16G16_SNORM, "R16G16_SNORM"),
                (DXGI_FORMAT_R32_UINT, "R32_UINT"),
                (DXGI_FORMAT_X24_TYPELESS_G8_UINT, "X24_TYPELESS_G8_UINT"),
                (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, "R9G9B9E5_SHAREDEXP"),
                (DXGI_FORMAT_BC5_SNORM, "BC5_SNORM"),
                (DXGI_FORMAT_B8G8R8X8_UNORM_SRGB, "B8G8R8X8_UNORM_SRGB"),
                (DXGI_FORMAT_BC7_UNORM_SRGB, "BC7_UNORM_SRGB"),
                (DXGI_FORMAT_B4G4R4A4_UNORM, "B4G4R4A4_UNORM"),
            ] {
                assert_eq!(name(format.0 as u32), text);
            }
            assert_eq!(name(130), "?");
        }

        fn texture(dimension: D3D12_RESOURCE_DIMENSION, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
            D3D12_RESOURCE_DESC {
                Dimension: dimension,
                Alignment: 0,
                Width: 2560,
                Height: 1440,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: flags,
            }
        }

        #[test]
        fn a_views_line() {
            let normal = texture(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                DXGI_FORMAT_R10G10B10A2_TYPELESS,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            );
            let found = Found::Texture { desc: normal, view: DXGI_FORMAT_R10G10B10A2_UNORM.0 as u32, title: false };
            assert_eq!(
                line(0, 472, false, &found),
                "s0+472 srv 2d 2560x1440 array 1 mips 1 R10G10B10A2_TYPELESS(23) view R10G10B10A2_UNORM(24) target"
            );
            // The view's format only where it differs; a depth; a volume.
            let depth = texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            let found = Found::Texture { desc: depth, view: DXGI_FORMAT_R32_TYPELESS.0 as u32, title: false };
            assert_eq!(line(0, 464, false, &found), "s0+464 srv 2d 2560x1440 array 1 mips 1 R32_TYPELESS(39) depth");
            let mut volume = texture(D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            volume.Width = 160;
            volume.Height = 90;
            volume.DepthOrArraySize = 64;
            let found = Found::Texture { desc: volume, view: 0, title: true };
            assert_eq!(
                line(1, 32, true, &found),
                "s1+32 uav 3d 160x90 depth 64 mips 1 R16G16B16A16_FLOAT(10) (title's description: no host copy)"
            );
            assert_eq!(line(1, 8, false, &Found::Buffer(4096)), "s1+8 srv buffer 4096 bytes");
            assert_eq!(line(1, 16, false, &Found::Nothing), "s1+16 srv -");
        }

        #[test]
        fn a_placed_description_as_words() {
            let mut words = [0u32; 14];
            words[0] = 3;
            words[4] = 2560;
            words[6] = 1440;
            words[7] = 1 | 5 << 16;
            words[8] = 24;
            words[9] = 1;
            words[12] = 1;
            let d = described(&words);
            assert_eq!((d.Dimension, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels), (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 2560, 1440, 1, 5));
            assert_eq!((d.Format, d.Flags), (DXGI_FORMAT_R10G10B10A2_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
        }

        #[test]
        fn once_per_distinct_set() {
            let mut seen = Seen::default();
            let a = vec!["s1+176 uav 2d 2560x1440 array 1 mips 1 R32_UINT(42)".to_owned()];
            let b = vec!["s1+176 uav 2d 1280x720 array 1 mips 1 R32_UINT(42)".to_owned()];
            assert_eq!(seen.first(Role::Resolve, "p", &a), Ok(Some(0)));
            assert_eq!(seen.first(Role::Resolve, "p", &a), Ok(None), "the same set again");
            assert_eq!(seen.first(Role::Resolve, "p", &b), Ok(Some(1)), "another set");
            assert_eq!(seen.first(Role::Resolve, "p", &b), Ok(None));
            assert_eq!(seen.first(Role::Resolve, "q", &a), Ok(Some(2)), "another program of the role");
            assert_eq!(seen.first(Role::Miss, "p", &a), Ok(Some(0)), "another role counts its own");
        }

        #[test]
        fn sets_stop_at_the_most() {
            let mut seen = Seen::default();
            for i in 0..MOST {
                assert_eq!(seen.first(Role::Air, "p", &[i.to_string()]), Ok(Some(i)));
            }
            assert_eq!(seen.first(Role::Air, "p", &["0".to_owned()]), Ok(None), "one listed is still not new");
            assert_eq!(seen.first(Role::Air, "p", &["new".to_owned()]), Err(()), "said once");
            assert_eq!(seen.first(Role::Air, "p", &["newer".to_owned()]), Ok(None));
            assert_eq!(seen.first(Role::Resolve, "p", &["new".to_owned()]), Ok(Some(0)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
    }

    #[test]
    fn specular_settings() {
        assert_eq!(parse_specular(""), None);
        assert_eq!(parse_specular("0"), None);
        assert_eq!(parse_specular("off"), None);
        let set = |share, sky| Some(Specular { share, sky, occluded: false });
        assert_eq!(parse_specular("1"), set(1.0, 0.0));
        assert_eq!(parse_specular("0.6, 0.25"), set(0.6, 0.25));
        assert_eq!(parse_specular("0.6,7"), set(0.6, 1.0));
        assert_eq!(parse_specular("0.6,x"), set(0.6, 0.0));
        // The third field.
        let occluded = |share, sky| Some(Specular { share, sky, occluded: true });
        assert_eq!(parse_specular("1,0,occluded"), occluded(1.0, 0.0));
        assert_eq!(parse_specular("0.6, 0.25 , occluded "), occluded(0.6, 0.25));
        assert_eq!(parse_specular("1,,occluded"), occluded(1.0, 0.0));
        assert_eq!(parse_specular("1,0,"), set(1.0, 0.0));
        assert_eq!(parse_specular("1,0,occlude"), set(1.0, 0.0));
        assert_eq!(parse_specular("1,0,Occluded"), set(1.0, 0.0));
        assert_eq!(parse_specular("0,0,occluded"), None);
    }

    /// Every form of `VICTORY_AMBIENT_SPECULAR_ALPHA`; `None` is off with a
    /// warning ([`alpha`]).
    #[test]
    fn alpha_layer_settings() {
        for off in ["", "0", "off", "title", " title "] {
            assert_eq!(parse_alpha(off), Some(Alpha::Title), "{off:?}");
        }
        for estimate in ["1", "estimate", " estimate "] {
            assert_eq!(parse_alpha(estimate), Some(Alpha::Estimate), "{estimate:?}");
        }
        assert_eq!(parse_alpha("bound"), Some(Alpha::Bound(1.0)));
        assert_eq!(parse_alpha(" bound "), Some(Alpha::Bound(1.0)));
        assert_eq!(parse_alpha("bound:2.5"), Some(Alpha::Bound(2.5)));
        assert_eq!(parse_alpha("bound: 6 "), Some(Alpha::Bound(6.0)));
        assert_eq!(parse_alpha("bound:0.1"), Some(Alpha::Bound(0.1)));
        assert_eq!(parse_alpha("bound:16"), Some(Alpha::Bound(16.0)));
        for other in [
            "bound:0.09", "bound:16.5", "bound:0", "bound:-1", "bound:nan", "bound:inf", "bound:", "bound:x", "bound6",
            "Bound", "2", "on", "yes", "estimate:1",
        ] {
            assert_eq!(parse_alpha(other), None, "{other:?}");
        }
    }

    /// Every layer and depth case, for every form of the switch.
    #[test]
    fn miss_routes() {
        use MissRoute::*;
        let alphas = [Alpha::Title, Alpha::Estimate, Alpha::Bound(1.0), Alpha::Bound(4.0)];
        // Found, valid; valid without found is not a case the caller gives,
        // but it is no depth either.
        let depths = [(false, false), (true, false), (true, true), (false, true)];
        for alpha in alphas {
            for (found, valid) in depths {
                let usable = found && valid;
                let estimate = if usable { Estimate } else { NoDepth };
                let at = |layer| route(layer, found, valid, alpha);
                // Opaque: the estimate with a usable depth, else nothing.
                assert_eq!(at(MISS_OPAQUE), estimate, "opaque {alpha:?} {found} {valid}");
                // Alpha: as the switch says; the estimate never falls through
                // to anything else without depth.
                let want = match alpha {
                    Alpha::Title => Left,
                    Alpha::Estimate => estimate,
                    Alpha::Bound(k) => Bound(k),
                };
                assert_eq!(at(MISS_ALPHA), want, "alpha {alpha:?} {found} {valid}");
                // Water and layers not known: left, whatever is set.
                for other in [2, 3, 7, u32::MAX] {
                    assert_eq!(at(other), Left, "layer {other} {alpha:?} {found} {valid}");
                }
            }
            // What is left does not depend on the depth (the caller decides it
            // before looking at the tables).
            for layer in [MISS_OPAQUE, MISS_ALPHA, 2, u32::MAX] {
                let left = route(layer, false, false, alpha) == Left;
                for (found, valid) in depths {
                    assert_eq!(route(layer, found, valid, alpha) == Left, left, "layer {layer} {alpha:?}");
                }
            }
        }
    }

    /// The cells' buffer with the bound off is the base's.
    #[test]
    fn the_cells_grow_only_with_a_bound() {
        assert_eq!(cell_bytes(false), (2 * 64 * 36 + 1 + 25 * 64 * 36) * 16);
        assert_eq!(cell_bytes(true), cell_bytes(false) + 16);
    }

    /// The software adapter, and what the passes' fixtures need of it: their
    /// own device (not the host's, which may be the card), list and heap.
    pub(super) mod warp {
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        pub(crate) struct Rig {
            pub(crate) device: ID3D12Device,
            queue: ID3D12CommandQueue,
            allocator: ID3D12CommandAllocator,
            pub(crate) list: ID3D12GraphicsCommandList,
            fence: ID3D12Fence,
            done: u64,
            event: HANDLE,
            heap: ID3D12DescriptorHeap,
            step: u32,
            /// Upload buffers the list still reads.
            held: Vec<ID3D12Resource>,
            _turn: std::sync::MutexGuard<'static, ()>,
        }

        impl Rig {
            /// None without a software adapter.
            pub(crate) fn new() -> Option<Self> {
                let turn = crate::indirect::tests::warp_turn();
                // SAFETY: plain DXGI and D3D12 creations.
                unsafe {
                    let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                    let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                    let mut device: Option<ID3D12Device> = None;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                    let device = device?;
                    let queue: ID3D12CommandQueue = device
                        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                            ..Default::default()
                        })
                        .ok()?;
                    let allocator: ID3D12CommandAllocator =
                        device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                    let list: ID3D12GraphicsCommandList =
                        device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
                    let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                    let event = CreateEventW(None, false, false, None).ok()?;
                    let heap: ID3D12DescriptorHeap = device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: 8,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    list.SetDescriptorHeaps(&[Some(heap.clone())]);
                    Some(Self { device, queue, allocator, list, fence, done: 0, event, heap, step, held: Vec::new(), _turn: turn })
                }
            }

            fn buffer(&self, kind: D3D12_HEAP_PROPERTIES, bytes: u64, uav: bool, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes.max(4),
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                    ..Default::default()
                };
                let mut buffer: Option<ID3D12Resource> = None;
                // SAFETY: a plain creation.
                unsafe {
                    self.device.CreateCommittedResource(&kind, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut buffer).unwrap();
                }
                buffer.unwrap()
            }

            /// A buffer shaders write and the test reads and writes through a
            /// mapping (the passes' cells and points).
            pub(crate) fn mapped(&self, bytes: u64) -> Option<ID3D12Resource> {
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                };
                Some(self.buffer(heap, bytes, true, D3D12_RESOURCE_STATE_COMMON))
            }

            /// A 2D texture of 32-bit texels holding `texels`, left in `state`.
            pub(crate) fn texture(
                &mut self,
                format: DXGI_FORMAT,
                size: [u32; 2],
                texels: &[u32],
                uav: bool,
                state: D3D12_RESOURCE_STATES,
            ) -> ID3D12Resource {
                let [w, h] = size;
                assert_eq!(texels.len(), (w * h) as usize);
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(w),
                    Height: h,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
                    ..Default::default()
                };
                let mut texture: Option<ID3D12Resource> = None;
                let pitch = (w * 4).next_multiple_of(256);
                let upload = self.buffer(
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    u64::from(pitch * h),
                    false,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                );
                // SAFETY: plain creation; the mapping is of `pitch * h` bytes;
                // the copy and the barrier are recorded in our open list.
                unsafe {
                    self.device
                        .CreateCommittedResource(
                            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut texture,
                        )
                        .unwrap();
                    let texture = texture.unwrap();
                    let mut data = std::ptr::null_mut();
                    upload.Map(0, None, Some(&mut data)).unwrap();
                    for (y, row) in texels.chunks(w as usize).enumerate() {
                        std::ptr::copy_nonoverlapping(row.as_ptr(), data.cast::<u8>().add(y * pitch as usize).cast(), row.len());
                    }
                    upload.Unmap(0, None);
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&upload),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 0,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: w, Height: h, Depth: 1, RowPitch: pitch },
                            },
                        },
                    };
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    self.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                    self.transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, state);
                    self.held.push(upload);
                    texture
                }
            }

            pub(crate) fn transition(&self, resource: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) {
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            // SAFETY: a borrowed pointer the barrier does not keep.
                            pResource: unsafe { std::mem::transmute_copy(resource) },
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: from,
                            StateAfter: to,
                        }),
                    },
                    ..Default::default()
                };
                // SAFETY: our open list.
                unsafe { self.list.ResourceBarrier(&[barrier]) };
            }

            fn handles(&self, at: u32) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
                // SAFETY: our heap; `at` is within its 8 places.
                unsafe {
                    let (mut cpu, mut gpu) =
                        (self.heap.GetCPUDescriptorHandleForHeapStart(), self.heap.GetGPUDescriptorHandleForHeapStart());
                    cpu.ptr += (at * self.step) as usize;
                    gpu.ptr += u64::from(at * self.step);
                    (cpu, gpu)
                }
            }

            /// A 2D UAV of `texture` at place `at` of the heap.
            pub(crate) fn uav(&self, at: u32, texture: &ID3D12Resource, format: DXGI_FORMAT) -> D3D12_GPU_DESCRIPTOR_HANDLE {
                let (cpu, gpu) = self.handles(at);
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                };
                // SAFETY: a live texture and a place of our heap.
                unsafe { self.device.CreateUnorderedAccessView(texture, None, Some(&desc), cpu) };
                gpu
            }

            /// A 2D SRV of `texture` at place `at` of the heap.
            pub(crate) fn srv(&self, at: u32, texture: &ID3D12Resource, format: DXGI_FORMAT) -> D3D12_GPU_DESCRIPTOR_HANDLE {
                let (cpu, gpu) = self.handles(at);
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                    },
                };
                // SAFETY: a live texture and a place of our heap.
                unsafe { self.device.CreateShaderResourceView(texture, Some(&desc), cpu) };
                gpu
            }

            /// Runs what is recorded and waits; the list is open again after,
            /// with the heap set.
            pub(crate) fn submit(&mut self) {
                self.done += 1;
                // SAFETY: our list, queue and fence.
                unsafe {
                    self.list.Close().unwrap();
                    self.queue.ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
                    self.queue.Signal(&self.fence, self.done).unwrap();
                    self.fence.SetEventOnCompletion(self.done, self.event).unwrap();
                    WaitForSingleObject(self.event, INFINITE);
                    self.allocator.Reset().unwrap();
                    self.list.Reset(&self.allocator, None).unwrap();
                    self.list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                }
                self.held.clear();
            }

            /// The texels of a 2D texture of 32-bit texels that is in `state`,
            /// after what is recorded has run.
            pub(crate) fn read(&mut self, texture: &ID3D12Resource, size: [u32; 2], state: D3D12_RESOURCE_STATES) -> Vec<u32> {
                let [w, h] = size;
                let pitch = (w * 4).next_multiple_of(256);
                let back = self.buffer(
                    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    u64::from(pitch * h),
                    false,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                self.transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                // SAFETY: as in `texture`, the other way.
                unsafe {
                    let format = texture.GetDesc().Format;
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&back),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 0,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: w, Height: h, Depth: 1, RowPitch: pitch },
                            },
                        },
                    };
                    self.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
                self.transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
                self.submit();
                let mut texels = vec![0u32; (w * h) as usize];
                // SAFETY: the copy has run; the mapping is of `pitch * h` bytes.
                unsafe {
                    let mut data = std::ptr::null_mut();
                    back.Map(0, None, Some(&mut data)).unwrap();
                    for (y, row) in texels.chunks_mut(w as usize).enumerate() {
                        std::ptr::copy_nonoverlapping(data.cast::<u8>().add(y * pitch as usize).cast(), row.as_mut_ptr(), row.len());
                    }
                    back.Unmap(0, None);
                }
                texels
            }
        }

        /// The float4s of a mapped buffer from `at` on.
        pub(crate) fn float4s(buffer: &ID3D12Resource, at: u64, n: usize) -> Vec<[f32; 4]> {
            let mut out = vec![[0f32; 4]; n];
            // SAFETY: a mapped buffer of the rig's, `at + n` float4s long; the
            // GPU is done with it.
            unsafe {
                let mut data = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                std::ptr::copy_nonoverlapping(data.cast::<[f32; 4]>().add(at as usize), out.as_mut_ptr(), n);
                buffer.Unmap(0, None);
            }
            out
        }

        /// Writes float4s into a mapped buffer from `at` on.
        pub(crate) fn put_float4s(buffer: &ID3D12Resource, at: u64, values: &[[f32; 4]]) {
            // SAFETY: as in `float4s`.
            unsafe {
                let mut data = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                std::ptr::copy_nonoverlapping(values.as_ptr(), data.cast::<[f32; 4]>().add(at as usize), values.len());
                buffer.Unmap(0, None);
            }
        }

        /// One channel of R11G11B10_FLOAT (5 exponent bits, `m` mantissa
        /// bits, no sign), rounded to nearest; what the fixtures put in.
        fn small(v: f32, m: u32) -> u32 {
            // The largest finite value.
            let top = (30 << m) | ((1 << m) - 1);
            if v.is_nan() {
                return (31 << m) | 1;
            }
            if v.is_infinite() {
                return 31 << m;
            }
            if v <= 0.0 {
                return 0;
            }
            let bits = v.to_bits();
            let e = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if e <= 0 {
                return 0;
            }
            let full = (e as u32) << m | ((bits & 0x7f_ffff) >> (23 - m));
            let round = (bits >> (22 - m)) & 1;
            (full + round).min(top)
        }

        fn unsmall(v: u32, m: u32) -> f32 {
            let (e, f) = (v >> m, v & ((1 << m) - 1));
            match e {
                31 if f == 0 => f32::INFINITY,
                31 => f32::NAN,
                0 => f as f32 / (1 << m) as f32 * 2f32.powi(-14),
                _ => (1.0 + f as f32 / (1 << m) as f32) * 2f32.powi(e as i32 - 15),
            }
        }

        pub(crate) fn r11g11b10(c: [f32; 3]) -> u32 {
            small(c[0], 6) | small(c[1], 6) << 11 | small(c[2], 5) << 22
        }

        pub(crate) fn rgb(v: u32) -> [f32; 3] {
            [unsmall(v & 0x7ff, 6), unsmall((v >> 11) & 0x7ff, 6), unsmall(v >> 22, 5)]
        }

        pub(crate) fn luma(c: [f32; 3]) -> f32 {
            0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        }
    }

    /// The passes' constants as `after` lays them out.
    fn words(level: f32, constant: f32, strength: f32, depth_size: [u32; 2]) -> [u32; 12] {
        let floats = [level, constant, strength, 0.6, 0.1, 0.45];
        let mut words = [0u32; 12];
        for (to, v) in words.iter_mut().zip(floats) {
            *to = v.to_bits();
        }
        words[6] = depth_size[0];
        words[7] = depth_size[1];
        words[8] = 0.04f32.to_bits();
        words[9] = 4.0f32.to_bits();
        words
    }

    /// Where the geometric mean is, in float4s.
    const GEO: u64 = 2 * CELLS + 1 + 25 * CELLS;

    /// The mean with a bound on keeps the geometric mean of the cells'
    /// luminance beside the plain mean; without, the buffer is the base's.
    #[test]
    fn warp_the_mean_keeps_the_log_mean_with_a_bound() {
        let Some(mut rig) = warp::Rig::new() else {
            return;
        };
        let off = make(&rig.device, |b| rig.mapped(b), 0, false, false, false).expect("the passes");
        assert!(off.bound.is_none());
        // SAFETY: a live buffer.
        assert_eq!(unsafe { off.cells.GetDesc() }.Width, CELL_BYTES, "the base's buffer without a bound");
        let pass = make(&rig.device, |b| rig.mapped(b), 0, true, false, false).expect("the passes, with the bound");
        // Half the cells send out 1, half 4 (grey: luminance as much).
        let cells: Vec<[f32; 4]> = (0..CELLS).map(|i| if i % 2 == 0 { [1.0, 1.0, 1.0, 1.0] } else { [4.0, 4.0, 4.0, 1.0] }).collect();
        warp::put_float4s(&pass.cells, 0, &cells);
        let w = words(1.0, 0.0, 0.0, [64, 36]);
        // SAFETY: our open list; the mean reads and writes the cells alone.
        unsafe {
            rig.list.SetComputeRootSignature(&pass.root);
            rig.list.SetComputeRoot32BitConstants(0, 12, w.as_ptr().cast(), 0);
            rig.list.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
            rig.list.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
            rig.list.SetPipelineState(&pass.mean);
            rig.list.Dispatch(1, 1, 1);
        }
        rig.submit();
        let mean = warp::float4s(&pass.cells, CELLS, 1)[0];
        let geo = warp::float4s(&pass.cells, GEO, 1)[0];
        assert!((mean[1] - 2.5).abs() < 1e-4 && mean[3] == 1.0, "the plain mean: {mean:?}");
        assert!((geo[0] - 2.0).abs() < 1e-3 && geo[3] == 1.0, "the geometric mean: {geo:?}");
    }

    /// The bound on a synthetic half-size image of the reflections: below the
    /// ceiling as it was, above it at the ceiling's luminance, NaN and Inf
    /// 0; with the bound off nothing is recorded and nothing changes.
    #[test]
    fn warp_the_bound_caps_the_alpha_layers_sky() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R11G11B10_FLOAT;
        let Some(mut rig) = warp::Rig::new() else {
            return;
        };
        let size = [16u32, 8];
        let below = warp::r11g11b10([0.25, 0.5, 0.125]);
        let above = warp::r11g11b10([8.0, 4.0, 2.0]);
        let nan = (31 << 6 | 1) | warp::r11g11b10([0.0, 1.0, 1.0]);
        let inf = 31 << 6;
        let mut texels = vec![below; 16 * 8];
        texels[..4].copy_from_slice(&[above, nan, inf, 0]);
        texels[9] = warp::r11g11b10([0.5, 1.0, 0.5]);
        // Room grey 0.5, geometric mean 0.5, k 2: the ceiling is 1.
        let k = 2.0f32;
        let ceiling = 1.0f32;
        let pass = make(&rig.device, |b| rig.mapped(b), 0, true, false, false).expect("the passes, with the bound");
        warp::put_float4s(&pass.cells, CELLS, &[[0.5, 0.5, 0.5, 1.0]]);
        warp::put_float4s(&pass.cells, GEO, &[[0.5, 0.0, 0.0, 1.0]]);
        let image = rig.texture(DXGI_FORMAT_R11G11B10_FLOAT, size, &texels, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let view = rig.uav(0, &image, DXGI_FORMAT_R11G11B10_FLOAT);
        let views = Views { image: view, depth: view, size, slices: 1, depth_size: [32, 16] };
        assert!(record(&rig.list, &pass, Role::Miss, &words(k, 0.0, 1.0, [32, 16]), &views, 0.0, true));
        let out = rig.read(&image, size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (i, (&was, &now)) in texels.iter().zip(&out).enumerate() {
            match i {
                0 => {
                    let l = warp::luma(warp::rgb(now));
                    assert!((l - ceiling).abs() < 0.03 * ceiling, "above the ceiling: luminance {l}, {:?}", warp::rgb(now));
                }
                1 | 2 => assert_eq!(now, 0, "texel {i} (NaN or Inf): {:?}", warp::rgb(now)),
                _ => assert_eq!(now, was, "texel {i}, below the ceiling: {:?} for {:?}", warp::rgb(now), warp::rgb(was)),
            }
        }
        // The bound off: no pipeline, nothing recorded, the image as it was.
        let off = make(&rig.device, |b| rig.mapped(b), 0, false, false, false).expect("the passes");
        let image = rig.texture(DXGI_FORMAT_R11G11B10_FLOAT, size, &texels, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let view = rig.uav(1, &image, DXGI_FORMAT_R11G11B10_FLOAT);
        let views = Views { image: view, depth: view, size, slices: 1, depth_size: [32, 16] };
        assert!(!record(&rig.list, &off, Role::Miss, &words(k, 0.0, 1.0, [32, 16]), &views, 0.0, true));
        assert_eq!(route(MISS_ALPHA, true, true, Alpha::Title), MissRoute::Left, "the switch off routes nothing");
        assert_eq!(rig.read(&image, size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), texels, "the bound off changes nothing");
    }

    /// The reflections' estimate over a synthetic corner: a floor 1.5 m below
    /// the eye, and a wall 5 m ahead on the left half of the view only (the
    /// right half is floor out to the sky). The room's light is the same
    /// everywhere, so without the field every floor texel gets the same fill
    /// (the old result). With it, a floor texel whose reflected ray runs into
    /// the wall gets less than its mirror on the right, whose ray runs into
    /// open space and keeps the old fill. (The creases' shade is off here, so
    /// that what differs is the march's.)
    #[test]
    fn warp_occluded_reflections_are_less_where_the_ray_meets_a_wall() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R32_FLOAT};
        let Some(mut rig) = warp::Rig::new() else {
            return;
        };
        let depth_size = [320u32, 180];
        let size = [160u32, 90];
        let (near, tan_y) = (0.1f32, 0.45f32);
        // The depth as the title keeps it (near / distance; 0 the sky).
        let mut depth = Vec::new();
        for y in 0..depth_size[1] {
            for x in 0..depth_size[0] {
                let ndc = [
                    (x as f32 + 0.5) / depth_size[0] as f32 * 2.0 - 1.0,
                    (y as f32 + 0.5) / depth_size[1] as f32 * 2.0 - 1.0,
                ];
                let down = ndc[1] * tan_y;
                let floor = if down > 0.0 { 1.5 / down } else { f32::INFINITY };
                let wall = if ndc[0] < 0.0 { 5.0 } else { f32::INFINITY };
                let z = floor.min(wall);
                depth.push(if z.is_finite() { (near / z).to_bits() } else { 0 });
            }
        }
        // Floor texels 4.4 m away, on the left (towards the wall) and right.
        let (left, right) = ([40usize, 78usize], [119usize, 78usize]);
        let run = |rig: &mut warp::Rig, occluded: bool, place: u32| -> Vec<f32> {
            let pass = make(&rig.device, |b| rig.mapped(b), 0, false, occluded, false).expect("the passes");
            let mut cells = vec![[1.0f32, 1.0, 1.0, 1.0]; CELLS as usize + 1];
            cells[CELLS as usize] = [1.0, 1.0, 1.0, 1.0];
            warp::put_float4s(&pass.cells, 0, &cells);
            let seen = rig.texture(DXGI_FORMAT_R32_FLOAT, depth_size, &depth, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            let sky = vec![warp::r11g11b10([10.0, 12.0, 20.0]); (size[0] * size[1]) as usize];
            let image = rig.texture(DXGI_FORMAT_R11G11B10_FLOAT, size, &sky, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let views = Views {
                image: rig.uav(place, &image, DXGI_FORMAT_R11G11B10_FLOAT),
                depth: rig.srv(place + 1, &seen, DXGI_FORMAT_R32_FLOAT),
                size,
                slices: 1,
                depth_size,
            };
            assert!(record(&rig.list, &pass, Role::Miss, &words(1.0, 0.0, 0.0, depth_size), &views, 0.0, false));
            rig.read(&image, size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).into_iter().map(|t| warp::luma(warp::rgb(t))).collect()
        };
        let at = |out: &[f32], [x, y]: [usize; 2]| out[y * size[0] as usize + x];
        let old = run(&mut rig, false, 0);
        let new = run(&mut rig, true, 2);
        let (old_left, old_right) = (at(&old, left), at(&old, right));
        let (new_left, new_right) = (at(&new, left), at(&new, right));
        eprintln!("fill without the field {old_left} | {old_right}; with it {new_left} | {new_right}");
        // Without the field: the old fill, the same on both sides.
        assert!((old_left - 1.0).abs() < 0.02 && (old_right - 1.0).abs() < 0.02, "the old fill: {old_left}, {old_right}");
        // With it: open space keeps that fill, the wall takes some away.
        assert!((new_right - old_right).abs() < 0.02, "open space: {new_right} against {old_right}");
        assert!(new_left < 0.8 * new_right, "into the wall {new_left}, into open space {new_right}");
        // And the whole band of floor next to the wall, on average.
        let band = |out: &[f32], xs: std::ops::Range<usize>| xs.clone().map(|x| at(out, [x, 78])).sum::<f32>() / xs.len() as f32;
        assert!(band(&new, 30..50) < 0.8 * band(&new, 110..130), "the floor before the wall against the open floor");
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"CS_TemporalResolve"), None);
        assert_eq!(role(b"..CS_RayMiss_de6b91a6_Wrapped.."), Some(Role::Miss));
        assert_eq!(role(b"..CS_RayMiss_Volumetrics_3c5d8c00_Wrapped.."), None);
        assert_eq!(role(b"..CS_ScreenProbe_RayMiss_80a88544_Wrapped.."), None);
    }

    /// The keep and raise as they act on the light, frame after frame, for
    /// surfaces that send out `a` of the light they get and the title's own
    /// light `title` on each (`VICTORY_AMBIENT_ONCE`'s model): the light it
    /// settles at on each surface.
    fn settle(level: f32, a: f32, title: &[f32], once: bool) -> Vec<f32> {
        let mut light = title.to_vec();
        let mut share = vec![1.0f32; title.len()];
        for _ in 0..500 {
            // Keep: what the surfaces sent out (the frame before), counted by
            // the share of the title's own light there with `once`.
            let sent: f32 = light
                .iter()
                .zip(&share)
                .map(|(l, s)| a * l * if once { *s } else { 1.0 })
                .sum::<f32>()
                / title.len() as f32;
            // Raise: never below the title's own light.
            for (i, t) in title.iter().enumerate() {
                light[i] = t.max(level * sent);
                share[i] = t / light[i].max(1e-6);
            }
        }
        light
    }

    /// Without it the stand-in's own light comes back to it: a lit wall and a
    /// dark one settle at `level a t / 2 / (1 - level a / 2)` on the dark one,
    /// a third more than one bounce at walls that send out 0.6, three
    /// quarters more at 0.9. With it, one bounce of the title's light.
    #[test]
    fn once_is_one_bounce_of_the_titles_light() {
        for a in [0.6f32, 0.9] {
            let one_bounce = 0.8 * a * 1.0 / 2.0;
            let fed = settle(0.8, a, &[1.0, 0.0], false)[1];
            let once = settle(0.8, a, &[1.0, 0.0], true)[1];
            assert!(
                (fed - one_bounce / (1.0 - 0.8 * a / 2.0)).abs() < 1e-4,
                "a={a}: fed back {fed}"
            );
            assert!((once - one_bounce).abs() < 1e-4, "a={a}: once {once}");
            assert!(fed > 1.3 * once);
        }
        // Where the title's light is above the estimate nothing changes.
        assert_eq!(settle(0.8, 0.6, &[1.0, 1.0], true), vec![1.0, 1.0]);
    }

    /// The passes compile with the system's compiler and the host takes them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device, |bytes| host.uav_buffer(bytes), 0, false, false, false).expect("the passes");
        // and with the wider surface direction (`VICTORY_AMBIENT_NORMALS`)
        make(&host.device, |bytes| host.uav_buffer(bytes), 8, false, false, false).expect("the passes, with a span of 8");
        // and with the alpha layer's bound (`VICTORY_AMBIENT_SPECULAR_ALPHA=bound`)
        let pass = make(&host.device, |bytes| host.uav_buffer(bytes), 0, true, false, false).expect("the passes, with the bound");
        assert!(pass.bound.is_some(), "the bound's pipeline");
        // and with the reflections' march and cap (`VICTORY_AMBIENT_SPECULAR=<share>,<sky>,occluded`)
        make(&host.device, |bytes| host.uav_buffer(bytes), 0, false, true, false).expect("the passes, occluded");
        make(&host.device, |bytes| host.uav_buffer(bytes), 8, true, true, false).expect("the passes, all but the dither on");
        // and with the dither moving with the frame (`VICTORY_AMBIENT_DITHER=frame`)
        let pass = make(&host.device, |bytes| host.uav_buffer(bytes), 0, false, false, true).expect("the passes, dithered");
        assert_eq!(pass.constants(), 13, "the frame's constant");
        make(&host.device, |bytes| host.uav_buffer(bytes), 8, true, true, true).expect("the passes, all on");
    }

    #[test]
    fn the_dither_moves_only_for_frame() {
        assert!(parse_dither("frame"));
        assert!(parse_dither(" frame "));
        for off in ["", "off", "0", "1", "on", "Frame", "frames", "frame:2"] {
            assert!(!parse_dither(off), "{off:?} is off");
        }
    }

    /// `SOURCE` with its `DITHER_FRAME` blocks taken out: the base's source.
    fn without_dither(source: &str) -> String {
        let mut out = String::new();
        let mut inside = false;
        for line in source.split_inclusive('\n') {
            match line.trim() {
                "#ifdef DITHER_FRAME" => inside = true,
                "#endif" if inside => inside = false,
                _ if !inside => out.push_str(line),
                _ => {}
            }
        }
        out
    }

    /// With the dither off every pass compiles to the same bytes as the source
    /// without the dither's blocks (the base's), for each set of defines the
    /// passes are made with; with it on every pass differs (its root signature
    /// has the frame).
    #[test]
    fn off_the_passes_are_the_bases() {
        let base = without_dither(SOURCE);
        assert_ne!(base, SOURCE, "the dither's blocks are in the source");
        assert!(!base.contains("DITHER") && !base.contains("uint frame"), "nothing of the dither is left outside its blocks");
        let passes = [
            ("KEEP", "", s!("keep")),
            ("KEEP", "#define GEO_MEAN\n", s!("mean")),
            ("PLACE", "", s!("place")),
            ("GATHER", "", s!("gather")),
            ("RAISE", "", s!("raise")),
            ("AIR", "", s!("clear_air")),
            ("REFLECT", "", s!("reflect_light")),
            ("REFLECT", "#define OCCLUDED\n", s!("reflect_light")),
            ("BOUND", "", s!("bound_sky")),
        ];
        let compile = |text: &str, entry| crate::host::compile(text, entry, s!("cs_5_1")).expect("the pass compiles");
        let mut compared = 0;
        for span in [0, 8] {
            for (part, more, entry) in passes {
                let off = compile(&source(part, span, more, false), entry);
                let was = compile(&format!("#define {part}\n#define NORMAL_SPAN {span}\n{more}{base}"), entry);
                assert!(off == was, "{part} {more:?} span {span}: off is not the base's bytes");
                let on = compile(&source(part, span, more, true), entry);
                assert!(on != off, "{part} {more:?} span {span}: the dither changes nothing");
                compared += 1;
            }
        }
        assert_eq!(compared, 18);
    }

    /// The creases' shade (`raise` with a constant level) over a synthetic
    /// corner: a floor 1.5 m below the eye and a wall 5 m ahead on the left
    /// half. With the dither off the frame is not given; with it on, frame 0
    /// (and 64) is the still noise and frame 1 moves the samples.
    #[test]
    fn warp_the_dither_moves_with_the_frame() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT};
        let Some(mut rig) = warp::Rig::new() else {
            return;
        };
        let size = [160u32, 90];
        let (near, tan_y) = (0.1f32, 0.45f32);
        let mut depth = Vec::new();
        for y in 0..size[1] {
            for x in 0..size[0] {
                let ndc = [(x as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0, (y as f32 + 0.5) / size[1] as f32 * 2.0 - 1.0];
                let down = ndc[1] * tan_y;
                let floor = if down > 0.0 { 1.5 / down } else { f32::INFINITY };
                let wall = if ndc[0] < 0.0 { 5.0 } else { f32::INFINITY };
                let z = floor.min(wall);
                depth.push(if z.is_finite() { (near / z).to_bits() } else { 0 });
            }
        }
        let off = make(&rig.device, |b| rig.mapped(b), 0, false, false, false).expect("the passes");
        let on = make(&rig.device, |b| rig.mapped(b), 0, false, false, true).expect("the passes, dithered");
        // Level 0, constant 1, the creases' shade at full strength.
        let base = words(0.0, 1.0, 1.0, size);
        let run = |rig: &mut warp::Rig, pass: &Pass, frame: Option<u32>| -> Vec<u32> {
            let mut w = base.to_vec();
            w.extend(frame);
            let seen = rig.texture(DXGI_FORMAT_R32_FLOAT, size, &depth, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            let light = rig.texture(DXGI_FORMAT_R32_UINT, size, &vec![0; (size[0] * size[1]) as usize], true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let views = Views {
                image: rig.uav(0, &light, DXGI_FORMAT_R32_UINT),
                depth: rig.srv(1, &seen, DXGI_FORMAT_R32_FLOAT),
                size,
                slices: 1,
                depth_size: size,
            };
            assert!(record(&rig.list, pass, Role::Resolve, &w, &views, 1.0, false));
            rig.read(&light, size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let still = run(&mut rig, &off, None);
        let shaded = still.iter().filter(|&&t| t != still[0]).count();
        assert!(shaded > 100, "the corner is shaded somewhere: {shaded} texels");
        let first = run(&mut rig, &on, Some(0));
        let next = run(&mut rig, &on, Some(1));
        let again = run(&mut rig, &on, Some(64));
        let moved = still.iter().zip(&next).filter(|(a, b)| a != b).count();
        eprintln!("{shaded} texels shaded; {moved} differ at frame 1");
        assert!(first == still, "frame 0 is the still noise");
        assert!(again == first, "the noise repeats after 64 frames");
        assert!(moved > shaded / 4, "frame 1 moves the samples: {moved} of {shaded} shaded texels differ");
    }

    #[test]
    fn the_normal_span_is_off_unless_two_or_more() {
        assert_eq!(parse_span(""), 0);
        assert_eq!(parse_span("0"), 0);
        assert_eq!(parse_span("1"), 0);
        assert_eq!(parse_span("off"), 0);
        assert_eq!(parse_span("8"), 8);
        assert_eq!(parse_span(" 4 "), 4);
        assert_eq!(parse_span("100"), 32);
    }
}
