# The settings menu's panel, modelled and rendered in Blender: our own artwork (a sunset-coast neon theme), no file
# or logo of any game. Renders the empty panel (the image the runtime draws its rows on) and a preview with rows.
#
#   blender -b --factory-startup -P scripts/overlay-art/menu_panel.py -- <out dir> [samples]
#
# Outputs <out dir>/panel.png (art only) and <out dir>/preview.png (art with example rows). The image is exactly the
# panel, 4 by 3 and opaque: the runtime copies it into the picture as a rectangle, with no blending on the card.
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else os.path.abspath("overlay-out")
SAMPLES = int(argv[1]) if len(argv) > 1 else 96
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
ROT = (math.pi / 2, 0.0, 0.0)  # local XY plane to world XZ, facing the camera at -Y

# Palette
NAVY = (0.012, 0.010, 0.045)
INK = (0.004, 0.004, 0.012)
PINK = (1.0, 0.10, 0.42)
CYAN = (0.10, 0.95, 0.95)
ORANGE = (1.0, 0.45, 0.12)
GOLD = (1.0, 0.82, 0.35)
PURPLE = (0.20, 0.03, 0.40)


def nodes_of(mat):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    return mat.node_tree


def principled(name, base, rough=0.35, metal=0.0, coat=0.0, emit=None, strength=0.0):
    mat = bpy.data.materials.new(name)
    tree = nodes_of(mat)
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if coat and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def emission(name, colour, strength):
    mat = bpy.data.materials.new(name)
    tree = nodes_of(mat)
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*colour, 1.0)
    emit.inputs["Strength"].default_value = strength
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def ramp_emission(name, ref, stops, strength, axis="Y", lo=-1.0, hi=1.0):
    """Emission whose colour runs along one local axis of `ref` (so separate objects share one gradient)."""
    mat = bpy.data.materials.new(name)
    tree = nodes_of(mat)
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    coord.object = ref
    sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    span = tree.nodes.new("ShaderNodeMapRange")
    span.inputs["From Min"].default_value = lo
    span.inputs["From Max"].default_value = hi
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    elements = ramp.color_ramp.elements
    elements[0].position, elements[0].color = stops[0][0], (*stops[0][1], 1.0)
    elements[1].position, elements[1].color = stops[-1][0], (*stops[-1][1], 1.0)
    for pos, colour in stops[1:-1]:
        el = elements.new(pos)
        el.color = (*colour, 1.0)
    tree.links.new(coord.outputs["Object"], sep.inputs[0])
    tree.links.new(sep.outputs[axis], span.inputs["Value"])
    tree.links.new(span.outputs[0], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    emit.inputs["Strength"].default_value = strength
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def link(obj):
    scene.collection.objects.link(obj)
    return obj


def box(name, cx, cz, w, h, y0, y1, mat, bevel=0.0, segments=6):
    """A box in world space: x and z extents w and h around (cx, cz), depth from y0 (front) to y1 (back)."""
    x0, x1, z0, z1 = cx - w / 2, cx + w / 2, cz - h / 2, cz + h / 2
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7), (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = link(bpy.data.objects.new(name, mesh))
    obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
    return obj


def plane(name, cx, cz, w, h, y, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    mesh.update()
    obj = link(bpy.data.objects.new(name, mesh))
    obj.rotation_euler = ROT
    obj.scale = (w / 2, h / 2, 1.0)
    obj.location = (cx, y, cz)
    obj.data.materials.append(mat)
    return obj


def disc(name, cx, cz, r, y, mat, n=96):
    verts = [(0.0, 0.0, 0.0)] + [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = link(bpy.data.objects.new(name, mesh))
    obj.rotation_euler = ROT
    obj.scale = (r, r, 1.0)
    obj.location = (cx, y, cz)
    obj.data.materials.append(mat)
    return obj


def rounded_rect(cx, cz, hw, hh, r, y, per_corner=14):
    pts = []
    for (sx, sz, a0) in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        ox, oz = cx + sx * (hw - r), cz + sz * (hh - r)
        for i in range(per_corner + 1):
            a = math.radians(a0 + 90.0 * i / per_corner)
            pts.append((ox + r * math.cos(a), y, oz + r * math.sin(a)))
    return pts


def tube(name, pts, radius, mat, cyclic=True):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 6
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for p, (x, y, z) in zip(spline.points, pts):
        p.co = (x, y, z, 1.0)
    spline.use_cyclic_u = cyclic
    obj = link(bpy.data.objects.new(name, curve))
    obj.data.materials.append(mat)
    return obj


def text(name, body, x, z, size, mat, y=-0.075, align="LEFT", extrude=0.0, bevel=0.0):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.extrude = extrude
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    obj = link(bpy.data.objects.new(name, curve))
    obj.rotation_euler = ROT
    obj.location = (x, y, z)
    obj.data.materials.append(mat)
    return obj


def palm(name, x, z, height, lean, y, mat, fronds=9, seed=0):
    """A palm in silhouette: a leaning trunk and arched fronds with hanging leaflets, all thin tubes."""
    trunk = []
    for i in range(13):
        t = i / 12
        trunk.append((x + lean * t * t, y, z + height * t))
    tube(f"{name}-trunk", trunk, 0.022, mat, cyclic=False)
    top_x, top_z = trunk[-1][0], trunk[-1][2]
    for f in range(fronds):
        angle = math.radians(-70 + 320 * f / (fronds - 1)) + 0.11 * math.sin(seed + f * 1.7)
        length = height * (0.42 + 0.10 * math.cos(seed * 1.3 + f))
        dx, dz = math.sin(angle), math.cos(angle)
        spine = []
        for i in range(11):
            t = i / 10
            droop = 0.55 * length * t * t
            spine.append((top_x + dx * length * t, y, top_z + dz * length * t * 0.75 - droop + 0.04))
        tube(f"{name}-frond{f}", spine, 0.012, mat, cyclic=False)
        for i in range(2, 11):
            px, _, pz = spine[i]
            leaf = 0.16 * length * (1.0 - 0.45 * i / 10)
            tube(f"{name}-leaf{f}-{i}", [(px, y, pz), (px + 0.25 * leaf * dx, y, pz - leaf)], 0.007, mat, cyclic=False)


# --- Materials
slab_mat = principled("slab", NAVY, rough=0.22, metal=0.55, coat=1.0)
plate_mat = principled("plate", (0.006, 0.006, 0.020), rough=0.30, metal=0.35, coat=1.0)
frame_mat = principled("frame", (0.03, 0.02, 0.07), rough=0.18, metal=0.9, coat=1.0)
title_face = emission("title-face", (1.0, 0.93, 0.97), 1.15)
title_body = principled("title-body", (0.30, 0.01, 0.20), rough=0.25, metal=0.6, emit=(0.55, 0.02, 0.35), strength=0.35)
pink_tube = emission("pink-tube", PINK, 2.3)
cyan_tube = emission("cyan-tube", CYAN, 2.0)
black_mat = emission("silhouette", (0.0, 0.0, 0.0), 0.0)

# --- The panel: a back slab, a raised plate for the rows, a framed sky window on top
W, H = 8.0, 6.0
box("slab", 0.0, 0.0, W + 0.4, H + 0.4, 0.10, 0.34, slab_mat)
box("plate", 0.0, -0.68, 7.44, 4.20, -0.04, 0.12, plate_mat, bevel=0.035, segments=4)
SKY_Z0, SKY_Z1 = 1.50, 2.78
sky = plane("sky", 0.0, (SKY_Z0 + SKY_Z1) / 2, 7.44, SKY_Z1 - SKY_Z0, 0.095, black_mat)
sky_mat = ramp_emission("sky-gradient", sky, [(0.0, (1.0, 0.70, 0.22)), (0.20, ORANGE), (0.52, (0.95, 0.10, 0.48)), (1.0, PURPLE)], 1.0)
sky.data.materials.clear()
sky.data.materials.append(sky_mat)

# The sun: a disc rising behind the frame's lower bar, with the classic slats showing the sky through it.
SUN_X, SUN_Z, SUN_R = 2.15, 1.52, 0.86
sun = disc("sun", SUN_X, SUN_Z, SUN_R, 0.085, black_mat)
sun_mat = ramp_emission("sun-gradient", sun, [(0.0, (1.0, 0.25, 0.50)), (0.45, (1.0, 0.55, 0.20)), (1.0, (1.0, 0.90, 0.45))], 1.25)
sun.data.materials.clear()
sun.data.materials.append(sun_mat)
for i, (dz, thick) in enumerate(((0.14, 0.060), (0.30, 0.050), (0.44, 0.040), (0.56, 0.030), (0.66, 0.022))):
    plane(f"slat{i}", SUN_X, SUN_Z + dz, 2 * SUN_R + 0.1, thick, 0.080, sky_mat)

# Palms in silhouette, at three depths.
palm("palm-a", 3.25, 1.50, 1.12, -0.22, 0.060, black_mat, seed=1)
palm("palm-b", 3.58, 1.50, 0.86, 0.14, 0.066, black_mat, fronds=8, seed=4)
palm("palm-c", 0.92, 1.50, 0.70, 0.10, 0.072, black_mat, fronds=8, seed=7)

# The frame around the sky window, raised above it.
for name, cx, cz, w, h in (
    ("frame-top", 0.0, SKY_Z1 + 0.035, 7.58, 0.07),
    ("frame-bottom", 0.0, SKY_Z0 - 0.035, 7.58, 0.07),
    ("frame-left", -3.755, (SKY_Z0 + SKY_Z1) / 2, 0.07, SKY_Z1 - SKY_Z0 + 0.14),
    ("frame-right", 3.755, (SKY_Z0 + SKY_Z1) / 2, 0.07, SKY_Z1 - SKY_Z0 + 0.14),
):
    box(name, cx, cz, w, h, -0.05, 0.12, frame_mat, bevel=0.012, segments=3)

# Neon: a pink tube round the whole panel, a cyan one round the rows' plate.
tube("neon-outer", rounded_rect(0.0, 0.0, W / 2 - 0.105, H / 2 - 0.105, 0.17, 0.02), 0.024, pink_tube)
tube("neon-plate", rounded_rect(0.0, -0.68, 3.78, 2.16, 0.12, 0.02), 0.012, cyan_tube)

# The title, in extruded chrome, catching the pink and cyan lights.
text("title-body", "SETTINGS", -3.395, 1.835, 0.66, title_body, y=0.045, extrude=0.03, bevel=0.016)
text("title", "SETTINGS", -3.42, 1.86, 0.66, title_face, y=-0.02)

# --- Light and camera
world = bpy.data.worlds.new("world")
tree = nodes_of(world)
background = next((n for n in tree.nodes if n.type == "BACKGROUND"), None)
if background is not None:
    background.inputs["Color"].default_value = (0.02, 0.02, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.10
scene.world = world


def area(name, location, colour, power, size, aim=(0.0, 0.0, 0.0)):
    light = bpy.data.lights.new(name, "AREA")
    light.color = colour
    light.energy = power
    light.size = size
    obj = link(bpy.data.objects.new(name, light))
    obj.location = location
    direction = (aim[0] - location[0], aim[1] - location[1], aim[2] - location[2])
    from mathutils import Vector

    obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return obj


area("key-pink", (-5.5, -6.5, 4.5), PINK, 900.0, 5.0)
area("key-cyan", (5.5, -6.5, -3.5), CYAN, 700.0, 5.0)
area("fill", (0.0, -9.0, 0.5), (1.0, 0.95, 0.9), 35.0, 8.0)

camera_data = bpy.data.cameras.new("camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = W
camera = link(bpy.data.objects.new("camera", camera_data))
camera.location = (0.0, -12.0, 0.0)
camera.rotation_euler = ROT
scene.camera = camera

scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.resolution_x = 1600
scene.render.resolution_y = 1200
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
try:
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
except Exception:
    pass

# --- Example rows, for the preview only
label_mat = emission("label", (0.90, 0.94, 1.0), 1.25)
value_mat = emission("value", (0.35, 1.0, 0.95), 1.5)
head_mat = emission("heading", (1.0, 0.42, 0.72), 1.5)
soft_mat = emission("soft", (0.62, 0.68, 0.82), 1.0)
pick_mat = principled("pick", (0.01, 0.10, 0.13), rough=0.25, metal=0.4, emit=(0.03, 0.40, 0.46), strength=0.22)
white_mat = emission("white", (1.0, 1.0, 1.0), 1.4)

ROWS = [
    ("heading", "PICTURE", ""),
    ("row", "Sharper picture (DLSS)", "Quality"),
    ("row", "Picture size", "3840 x 2160"),
    ("row", "Console mode", "Series X  (full detail)"),
    ("heading", "SMOOTHNESS", ""),
    ("pick", "Extra frames (smoother motion)", "Off"),
    ("row", "Frame rate limit", "None"),
    ("heading", "LIGHT AND WATER", ""),
    ("row", "Real bounce light  (best picture, slower)", "Off"),
    ("row", "Quick light in shade", "On"),
    ("row", "Light in water", "Off"),
    ("heading", "ON SCREEN", ""),
    ("row", "Show frame rate", "On"),
]
preview = []
z = 1.12
for kind, left, right in ROWS:
    if kind == "heading":
        z -= 0.06
        preview.append(text(f"h-{left}", left, -3.46, z, 0.150, head_mat))
        z -= 0.235
        continue
    if kind == "pick":
        preview.append(box("pick-bar", 0.0, z + 0.062, 7.16, 0.26, -0.062, -0.03, pick_mat, bevel=0.012, segments=3))
        preview.append(tube("pick-line", [(-3.56, -0.064, z - 0.072), (3.56, -0.064, z - 0.072)], 0.008, cyan_tube, cyclic=False))
        preview.append(text("pick-l", "‹", 1.02, z, 0.21, value_mat))
        preview.append(text("pick-r", "›", 3.42, z, 0.21, value_mat))
    preview.append(text(f"l-{left}", left, -3.30, z, 0.190, label_mat))
    preview.append(text(f"v-{left}", right, 3.28 if kind != "pick" else 3.20, z, 0.190, white_mat if kind == "pick" else value_mat, align="RIGHT"))
    z -= 0.25
preview.append(text("help", "Adds in-between frames so movement looks smoother. Takes effect after a restart.", -3.46, -2.36, 0.150, soft_mat))
preview.append(text("keys", "Up / Down: move      Left / Right: change      Enter: select      F1: close", -3.46, -2.63, 0.145, soft_mat))


def glow(path, radius, gain):
    """A bloom done on the saved image: the bright parts, blurred, screened back; the glow may pass the edge."""
    image = bpy.data.images.load(path)
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    alpha = px[..., 3:4]
    pre = px[..., :3] * alpha
    bright = np.clip(pre - 0.70, 0.0, None) * 2.2

    def blur(a, r):
        for axis in (0, 1):
            for _ in range(3):
                c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(3)], mode="edge"), axis=axis)
                a = (np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis) - np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)) / (2 * r + 1)
        return a

    halo = blur(bright, radius) * gain + blur(bright, max(2, radius // 4)) * gain * 0.6
    out = 1.0 - (1.0 - np.clip(pre, 0, 1)) * (1.0 - np.clip(halo, 0, 1))
    shadow = np.roll(blur(np.repeat(alpha, 3, axis=2), 26)[..., :1], -14, axis=0) * 0.60
    out_alpha = np.maximum(np.maximum(alpha, shadow), np.clip(halo.max(axis=2, keepdims=True) * 0.9, 0, 1))
    rgb = np.where(out_alpha > 1e-4, out / np.maximum(out_alpha, 1e-4), 0.0)
    px = np.concatenate([np.clip(rgb, 0, 1), out_alpha], axis=2).astype(np.float32)
    image.pixels.foreach_set(px.reshape(-1))
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


for name, show in (("panel", False), ("preview", True)):
    for obj in preview:
        obj.hide_render = not show
    path = os.path.join(OUT, f"{name}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    glow(path, 20, 0.34)
    print(f"wrote {path}")
