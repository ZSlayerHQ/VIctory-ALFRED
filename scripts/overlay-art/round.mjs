// Run after rendering and encoding the RGB panel, before running chip.mjs.
// node round.mjs [assets dir] [panel preview.ppm]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { qoiDecode, qoiEncode, writeLayout, writePreview } from './chip.mjs';

function round(panel) {
  const { w, h, pixels } = panel;
  const count = w * h;
  const outside = new Uint8Array(count);
  const queue = new Int32Array(count);
  let tail = 0;
  const visit = i => {
    if (outside[i]) return;
    const p = i * 3;
    // The bright pink tube blocks the flood. Its core is continuous and distinct from the slab.
    if (pixels[p] >= 245 && pixels[p + 1] <= 180 && pixels[p + 2] >= 205) return;
    outside[i] = 1;
    queue[tail++] = i;
  };
  for (let x = 0; x < w; x += 1) {
    visit(x);
    visit((h - 1) * w + x);
  }
  for (let y = 0; y < h; y += 1) {
    visit(y * w);
    visit(y * w + w - 1);
  }
  for (let head = 0; head < tail; head += 1) {
    const i = queue[head];
    const x = i % w;
    if (x > 0) visit(i - 1);
    if (x < w - 1) visit(i + 1);
    if (i >= w) visit(i - w);
    if (i < count - w) visit(i + w);
  }
  if (outside[Math.floor(h / 2) * w + Math.floor(w / 2)] || ![0, w - 1, count - w, count - 1].every(i => outside[i])) {
    throw new Error('the pink outline must enclose the centre and exclude all four corners');
  }

  const rounded = Buffer.alloc(count * 4);
  const nearest = new Int32Array(count);
  const distance = new Uint8Array(count).fill(255);
  const reach = 4;
  for (let i = 0; i < count; i += 1) {
    if (outside[i]) continue;
    pixels.copy(rounded, i * 4, i * 3, i * 3 + 3);
    rounded[i * 4 + 3] = 255;
    if (!(outside[i - 1] || outside[i + 1] || outside[i - w] || outside[i + w])) continue;
    const x = i % w;
    const y = Math.floor(i / w);
    for (let dy = -reach; dy <= reach; dy += 1) for (let dx = -reach; dx <= reach; dx += 1) {
      const d = dx * dx + dy * dy;
      const j = i + dy * w + dx;
      if (d <= reach * reach && x + dx >= 0 && x + dx < w && outside[j] && d < distance[j]) {
        distance[j] = d;
        nearest[j] = i;
      }
    }
  }
  for (let i = 0; i < count; i += 1) {
    if (distance[i] === 255) continue;
    const core = nearest[i];
    const x = i % w;
    const y = Math.floor(i / w);
    // Compare the blue of the glow to the local slab and to the tube, along the outward direction.
    const step = (reach + 1) / Math.sqrt(distance[i]);
    const sx = Math.max(0, Math.min(w - 1, Math.round(x + (x - core % w) * step)));
    const sy = Math.max(0, Math.min(h - 1, Math.round(y + (y - Math.floor(core / w)) * step)));
    const slab = pixels[(sy * w + sx) * 3 + 2] + 2;
    const tube = pixels[core * 3 + 2];
    const a = Math.round(255 * Math.max(0, pixels[i * 3 + 2] - slab) / Math.max(1, tube - slab));
    if (a === 0 || [0, w - 1, count - w, count - 1].includes(i)) continue;
    // Straight alpha. Extend the tube colour into the ramp to keep the slab out of the blend.
    pixels.copy(rounded, i * 4, core * 3, core * 3 + 3);
    rounded[i * 4 + 3] = Math.min(254, a);
  }
  return { w, h, channels: 4, pixels: rounded };
}

const [dir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'assets'), preview] = process.argv.slice(2);
const file = path.join(dir, 'panel.qoi');
let data = fs.readFileSync(file);
let panel = qoiDecode(data);
if (panel.channels === 3) {
  panel = round(panel);
  data = qoiEncode(panel);
  if (!qoiDecode(data).pixels.equals(panel.pixels)) throw new Error('the encoded panel does not decode to the same pixels');
  fs.writeFileSync(file, data);
}
writeLayout(dir, layout => Object.assign(layout.panel, { bytes: data.length, channels: panel.channels }));
if (preview) writePreview(preview, panel);
console.log(`panel.qoi: ${panel.w} by ${panel.h}, ${panel.channels} channels, ${data.length} bytes`);
