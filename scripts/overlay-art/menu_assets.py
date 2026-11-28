# The settings menu's assets for the runtime, from the rendered panel: the panel and two sizes of lettering as QOI
# images (a format decoded in about 60 lines, so the runtime needs no new dependency), each character's advance, and
# where things sit on the panel. Also a proof image drawn from those assets alone, the way the runtime will draw.
#
#   blender -b --factory-startup -P scripts/overlay-art/menu_assets.py -- <panel.png> <out dir>
#
# The lettering is Blender's built-in font, rendered here as coverage (how much of each pixel a letter covers).
import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PANEL, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)

PPU = 200  # pixels to a world unit: the panel is 8 by 6 units, 1600 by 1200 pixels
PANEL_W, PANEL_H = 8.0, 6.0
CHARS = [chr(c) for c in range(32, 127)] + ["‹", "›", "·", "×"]
SIZES = {"row": 0.190, "small": 0.150}
COLS = 16
PAD = 6  # pixels kept free left of each character's origin, for letters that reach left of it
ROT = (math.pi / 2, 0.0, 0.0)


def make_text(scene, body, size, x=0.0, z=0.0, mat=None):
    curve = bpy.data.curves.new("t", "FONT")
    curve.body = body
    curve.size = size
    obj = bpy.data.objects.new("t", curve)
    scene.collection.objects.link(obj)
    obj.rotation_euler = ROT
    obj.location = (x, 0.0, z)
    if mat is not None:
        curve.materials.append(mat)
    return obj


def width(scene, body, size):
    obj = make_text(scene, body, size)
    bpy.context.view_layer.update()
    w = obj.dimensions.x
    if w <= 0.0:
        box = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).bound_box
        w = max(p[0] for p in box) - min(p[0] for p in box)
    bpy.data.objects.remove(obj, do_unlink=True)
    return w


def pixels_of(path):
    image = bpy.data.images.load(path)
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    bpy.data.images.remove(image)
    return np.clip(px.reshape(h, w, 4)[::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)  # top row first


def save_png(path, rgb):
    h, w, _ = rgb.shape
    image = bpy.data.images.new("out", w, h, alpha=False)
    px = np.concatenate([rgb[::-1].astype(np.float32) / 255.0, np.ones((h, w, 1), np.float32)], axis=2)
    image.pixels.foreach_set(px.reshape(-1))
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def qoi_encode(rgb):
    """QOI, three channels. `rgb` is (height, width, 3) bytes, top row first."""
    h, w, _ = rgb.shape
    out = bytearray(b"qoif" + w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([3, 0]))
    index = [(0, 0, 0)] * 64
    pr, pg, pb = 0, 0, 0
    run = 0
    for r, g, b in rgb.reshape(-1, 3).tolist():
        if (r, g, b) == (pr, pg, pb):
            run += 1
            if run == 62:
                out.append(0xC0 | 61)
                run = 0
            continue
        if run:
            out.append(0xC0 | (run - 1))
            run = 0
        slot = (r * 3 + g * 5 + b * 7 + 255 * 11) % 64
        if index[slot] == (r, g, b):
            out.append(slot)
        else:
            index[slot] = (r, g, b)
            dr, dg, db = ((r - pr + 128) & 255) - 128, ((g - pg + 128) & 255) - 128, ((b - pb + 128) & 255) - 128
            if -2 <= dr <= 1 and -2 <= dg <= 1 and -2 <= db <= 1:
                out.append(0x40 | (dr + 2) << 4 | (dg + 2) << 2 | (db + 2))
            elif -32 <= dg <= 31 and -8 <= dr - dg <= 7 and -8 <= db - dg <= 7:
                out.append(0x80 | (dg + 32))
                out.append((dr - dg + 8) << 4 | (db - dg + 8))
            else:
                out += bytes([0xFE, r, g, b])
        pr, pg, pb = r, g, b
    if run:
        out.append(0xC0 | (run - 1))
    return bytes(out + bytes([0, 0, 0, 0, 0, 0, 0, 1]))


def qoi_decode(data):
    """The decoder the runtime needs, here to prove each file reads back exactly."""
    assert data[:4] == b"qoif" and data[12] == 3
    w, h = int.from_bytes(data[4:8], "big"), int.from_bytes(data[8:12], "big")
    out = bytearray(w * h * 3)
    index = [(0, 0, 0)] * 64
    r, g, b = 0, 0, 0
    at, i = 14, 0
    while i < w * h:
        tag = data[at]
        at += 1
        count = 1
        if tag == 0xFE:
            r, g, b = data[at], data[at + 1], data[at + 2]
            at += 3
        elif tag >> 6 == 0:
            r, g, b = index[tag]
        elif tag >> 6 == 1:
            r, g, b = (r + (tag >> 4 & 3) - 2) & 255, (g + (tag >> 2 & 3) - 2) & 255, (b + (tag & 3) - 2) & 255
        elif tag >> 6 == 2:
            dg = (tag & 63) - 32
            nxt = data[at]
            at += 1
            r, g, b = (r + dg + (nxt >> 4) - 8) & 255, (g + dg) & 255, (b + dg + (nxt & 15) - 8) & 255
        else:
            count = (tag & 63) + 1
        index[(r * 3 + g * 5 + b * 7 + 255 * 11) % 64] = (r, g, b)
        for _ in range(count):
            out[i * 3:i * 3 + 3] = bytes((r, g, b))
            i += 1
    return np.frombuffer(bytes(out), dtype=np.uint8).reshape(h, w, 3)


def write_qoi(name, rgb):
    data = qoi_encode(rgb)
    assert np.array_equal(qoi_decode(data), rgb), f"{name}: the QOI file does not read back"
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)
    return len(data)


def bake(name, size):
    """One size of lettering: a grid of cells, each character at its origin, as coverage."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    base_width = width(scene, "HH", size)
    advance = {ch: max(0.0, width(scene, "H" + ch + "H", size) - base_width) for ch in CHARS}
    cell_w = math.ceil(max(advance.values()) * PPU) + 2 * PAD
    cell_h = math.ceil(size * 1.5 * PPU)
    baseline = round(size * 1.04 * PPU)  # pixels from a cell's top down to the line the letters stand on
    rows = math.ceil(len(CHARS) / COLS)
    px_w, px_h = COLS * cell_w, rows * cell_h

    mat = bpy.data.materials.new("white")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    tree = mat.node_tree
    tree.nodes.clear()
    out_node = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    tree.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
    for i, ch in enumerate(CHARS):
        if ch == " ":
            continue
        col, row = i % COLS, i // COLS
        make_text(scene, ch, size, x=(col * cell_w + PAD) / PPU, z=-(row * cell_h + baseline) / PPU, mat=mat)

    cam_data = bpy.data.cameras.new("camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(px_w, px_h) / PPU
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (px_w / PPU / 2, -10.0, -px_h / PPU / 2)
    cam.rotation_euler = ROT
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.render.resolution_x, scene.render.resolution_y = px_w, px_h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    path = os.path.join(OUT, f"font-{name}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    coverage = pixels_of(path)[..., 3]
    size_bytes = write_qoi(f"font-{name}.qoi", np.repeat(coverage[..., None], 3, axis=2))
    metrics = {
        "image": f"font-{name}.qoi",
        "width": px_w,
        "height": px_h,
        "columns": COLS,
        "cell": [cell_w, cell_h],
        "origin_x": PAD,
        "baseline": baseline,
        "first_char": 32,
        "extra_chars": ["‹", "›", "·", "×"],
        "advance": [round(advance[ch] * PPU, 2) for ch in CHARS],
        "bytes": size_bytes,
    }
    return coverage, metrics


def draw(canvas, font, metrics, body, x, baseline, colour, right=False):
    """Blend `body` onto `canvas` by coverage: what the runtime does, to prove the assets are enough."""
    advance = metrics["advance"]
    cell_w, cell_h = metrics["cell"]
    total = sum(advance[CHARS.index(ch)] for ch in body if ch in CHARS)
    pen = x - total if right else x
    for ch in body:
        if ch not in CHARS:
            continue
        i = CHARS.index(ch)
        col, row = i % COLS, i // COLS
        src = font[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w].astype(np.float32) / 255.0
        dx, dy = int(round(pen)) - metrics["origin_x"], int(round(baseline)) - metrics["baseline"]
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + cell_w, canvas.shape[1]), min(dy + cell_h, canvas.shape[0])
        if x1 > x0 and y1 > y0:
            a = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx, None]
            canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + np.array(colour, np.float32) * a
        pen += advance[i]


panel = pixels_of(PANEL)[..., :3]
assert panel.shape[:2] == (int(PANEL_H * PPU), int(PANEL_W * PPU)), panel.shape
panel_bytes = write_qoi("panel.qoi", panel)

fonts, metrics = {}, {}
for name, size in SIZES.items():
    fonts[name], metrics[name] = bake(name, size)


def fx(x):  # a world x on the panel, as a fraction of its width
    return round((x + PANEL_W / 2) / PANEL_W, 5)


def fy(z):  # a world z on the panel, as a fraction of its height, from the top
    return round((PANEL_H / 2 - z) / PANEL_H, 5)


layout = {
    "panel": {"image": "panel.qoi", "width": int(PANEL_W * PPU), "height": int(PANEL_H * PPU), "bytes": panel_bytes},
    "fonts": metrics,
    "note": "x and y are fractions of the panel's width and height, y from the top; text sits on its baseline",
    "lines": {
        "first_baseline_y": fy(1.12),
        "row_pitch": round(0.25 / PANEL_H, 5),
        "heading_gap_before": round(0.06 / PANEL_H, 5),
        "heading_pitch": round(0.235 / PANEL_H, 5),
        "last_baseline_y": fy(-2.05),
    },
    "x": {
        "heading": fx(-3.46),
        "label": fx(-3.30),
        "value_right": fx(3.28),
        "picked_value_right": fx(3.20),
        "arrow_left": fx(1.02),
        "arrow_right": fx(3.42),
        "help": fx(-3.46),
    },
    "pick_bar": {
        "left": fx(-3.58),
        "right": fx(3.58),
        "top_above_baseline": round(0.192 / PANEL_H, 5),
        "height": round(0.26 / PANEL_H, 5),
        "underline_below_baseline": round(0.072 / PANEL_H, 5),
        "underline_height": round(0.014 / PANEL_H, 5),
        "fill": [22, 104, 118],
        "underline": [70, 240, 238],
    },
    "help_baseline_y": fy(-2.36),
    "keys_baseline_y": fy(-2.63),
    "colours": {
        "heading": [255, 122, 188],
        "label": [240, 245, 255],
        "value": [120, 255, 244],
        "picked_value": [255, 255, 255],
        "soft": [168, 180, 214],
    },
    "font_for": {"label": "row", "value": "row", "heading": "small", "help": "small", "keys": "small"},
}
with open(os.path.join(OUT, "layout.json"), "w", encoding="utf-8") as f:
    json.dump(layout, f, indent=2, ensure_ascii=True)

# --- The proof: the same rows as the preview, drawn from the assets alone.
H, W = panel.shape[:2]
canvas = qoi_decode(open(os.path.join(OUT, "panel.qoi"), "rb").read()).astype(np.float32)
col = layout["colours"]
rows = [
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
y = layout["lines"]["first_baseline_y"] * H
for kind, left, right in rows:
    if kind == "heading":
        y += layout["lines"]["heading_gap_before"] * H
        draw(canvas, fonts["small"], metrics["small"], left, layout["x"]["heading"] * W, y, col["heading"])
        y += layout["lines"]["heading_pitch"] * H
        continue
    if kind == "pick":
        bar = layout["pick_bar"]
        top = int(round(y - bar["top_above_baseline"] * H))
        x0, x1 = int(bar["left"] * W), int(bar["right"] * W)
        canvas[top:top + int(round(bar["height"] * H)), x0:x1] = bar["fill"]
        under = int(round(y + bar["underline_below_baseline"] * H))
        canvas[under:under + max(1, int(round(bar["underline_height"] * H))), x0:x1] = bar["underline"]
        draw(canvas, fonts["row"], metrics["row"], "‹", layout["x"]["arrow_left"] * W, y, col["value"])
        draw(canvas, fonts["row"], metrics["row"], "›", layout["x"]["arrow_right"] * W, y, col["value"])
    picked = kind == "pick"
    draw(canvas, fonts["row"], metrics["row"], left, layout["x"]["label"] * W, y, col["label"])
    draw(canvas, fonts["row"], metrics["row"], right, layout["x"]["picked_value_right" if picked else "value_right"] * W, y,
         col["picked_value" if picked else "value"], right=True)
    y += layout["lines"]["row_pitch"] * H
draw(canvas, fonts["small"], metrics["small"], "Adds in-between frames so movement looks smoother. Takes effect after a restart.",
     layout["x"]["help"] * W, layout["help_baseline_y"] * H, col["soft"])
draw(canvas, fonts["small"], metrics["small"], "Up / Down: move      Left / Right: change      Enter: select      F1: close",
     layout["x"]["help"] * W, layout["keys_baseline_y"] * H, col["soft"])
save_png(os.path.join(OUT, "proof.png"), np.clip(canvas + 0.5, 0, 255).astype(np.uint8))
print("assets:", json.dumps({"panel_qoi": panel_bytes, **{k: v["bytes"] for k, v in metrics.items()}}))
