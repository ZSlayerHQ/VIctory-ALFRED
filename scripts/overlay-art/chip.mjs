// The counter chip's art, cut from the settings panel itself so the two match: the panel's bottom-left corner (its
// pink and cyan tubes and the glass), mirrored into four corners, with an edge strip and a piece of glass between
// them. The result is a nine-slice at half the panel's scale: corners stay as they are, edges and the centre repeat.
//   node scripts/overlay-art/chip.mjs [assets dir]      writes chip.qoi and the "chip" entry of layout.json
//   node scripts/overlay-art/chip.mjs [assets dir] --preview <out.ppm> <width> <height>   a chip drawn at that size
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dir = args[0] && !args[0].startsWith('--') ? args.shift() : path.join(HERE, 'assets');

/// QOI to { w, h, channels, pixels }.
export function qoiDecode(data) {
  if (data.toString('latin1', 0, 4) !== 'qoif') throw new Error('not a QOI image');
  const w = data.readUInt32BE(4);
  const h = data.readUInt32BE(8);
  const channels = data[12];
  if (channels !== 3 && channels !== 4) throw new Error('expected RGB or RGBA QOI');
  const pixels = Buffer.alloc(w * h * channels);
  const seen = new Uint8Array(64 * 4);
  let [r, g, b, a] = [0, 0, 0, 255];
  let at = 14;
  let run = 0;
  for (let px = 0; px < w * h; px += 1) {
    if (run > 0) run -= 1;
    else {
      const byte = data[at++];
      if (byte === 0xfe) [r, g, b] = [data[at++], data[at++], data[at++]];
      else if (byte === 0xff) [r, g, b, a] = [data[at++], data[at++], data[at++], data[at++]];
      else if (byte >> 6 === 0) [r, g, b, a] = [seen[byte * 4], seen[byte * 4 + 1], seen[byte * 4 + 2], seen[byte * 4 + 3]];
      else if (byte >> 6 === 1) {
        r = (r + ((byte >> 4) & 3) - 2) & 255;
        g = (g + ((byte >> 2) & 3) - 2) & 255;
        b = (b + (byte & 3) - 2) & 255;
      } else if (byte >> 6 === 2) {
        const next = data[at++];
        const dg = (byte & 63) - 32;
        r = (r + dg - 8 + (next >> 4)) & 255;
        g = (g + dg) & 255;
        b = (b + dg - 8 + (next & 15)) & 255;
      } else run = byte & 63;
      const slot = ((r * 3 + g * 5 + b * 7 + a * 11) % 64) * 4;
      [seen[slot], seen[slot + 1], seen[slot + 2], seen[slot + 3]] = [r, g, b, a];
    }
    const offset = px * channels;
    pixels[offset] = r;
    pixels[offset + 1] = g;
    pixels[offset + 2] = b;
    if (channels === 4) pixels[offset + 3] = a;
  }
  return { w, h, channels, pixels };
}

/// { w, h, channels, pixels } to QOI.
export function qoiEncode({ w, h, channels, pixels }) {
  const out = Buffer.alloc(14 + w * h * (channels + 1) + 8);
  out.write('qoif', 0, 'latin1');
  out.writeUInt32BE(w, 4);
  out.writeUInt32BE(h, 8);
  out[12] = channels;
  out[13] = 0;
  const seen = new Uint32Array(64);
  let at = 14;
  let [pr, pg, pb, pa] = [0, 0, 0, 255];
  let run = 0;
  for (let px = 0; px < w * h; px += 1) {
    const offset = px * channels;
    const [r, g, b] = [pixels[offset], pixels[offset + 1], pixels[offset + 2]];
    const a = channels === 4 ? pixels[offset + 3] : 255;
    if (r === pr && g === pg && b === pb && a === pa) {
      run += 1;
      if (run === 62 || px === w * h - 1) {
        out[at++] = 0xc0 | (run - 1);
        run = 0;
      }
      continue;
    }
    if (run > 0) {
      out[at++] = 0xc0 | (run - 1);
      run = 0;
    }
    const slot = (r * 3 + g * 5 + b * 7 + a * 11) % 64;
    const value = (r | g << 8 | b << 16 | a << 24) >>> 0;
    if (seen[slot] === value) out[at++] = slot;
    else {
      seen[slot] = value;
      const [dr, dg, db] = [((r - pr + 128) & 255) - 128, ((g - pg + 128) & 255) - 128, ((b - pb + 128) & 255) - 128];
      const [drg, dbg] = [dr - dg, db - dg];
      if (a !== pa) {
        out[at++] = 0xff;
        out[at++] = r;
        out[at++] = g;
        out[at++] = b;
        out[at++] = a;
      } else if (dr >= -2 && dr <= 1 && dg >= -2 && dg <= 1 && db >= -2 && db <= 1) out[at++] = 0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2);
      else if (dg >= -32 && dg <= 31 && drg >= -8 && drg <= 7 && dbg >= -8 && dbg <= 7) {
        out[at++] = 0x80 | (dg + 32);
        out[at++] = ((drg + 8) << 4) | (dbg + 8);
      } else {
        out[at++] = 0xfe;
        out[at++] = r;
        out[at++] = g;
        out[at++] = b;
      }
    }
    [pr, pg, pb, pa] = [r, g, b, a];
  }
  Buffer.from([0, 0, 0, 0, 0, 0, 0, 1]).copy(out, at);
  return out.subarray(0, at + 8);
}

const image = (w, h, channels) => ({ w, h, channels, pixels: Buffer.alloc(w * h * channels) });
const get = (im, x, y) => {
  const at = (y * im.w + x) * im.channels;
  return im.pixels.subarray(at, at + im.channels);
};
const put = (im, x, y, p) => p.copy(im.pixels, (y * im.w + x) * im.channels);
function crop(im, x0, y0, w, h) {
  const out = image(w, h, im.channels);
  for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) put(out, x, y, get(im, x0 + x, y0 + y));
  return out;
}
function flip(im, across) {
  const out = image(im.w, im.h, im.channels);
  for (let y = 0; y < im.h; y += 1)
    for (let x = 0; x < im.w; x += 1) put(out, x, y, get(im, across === 'x' ? im.w - 1 - x : x, across === 'y' ? im.h - 1 - y : y));
  return out;
}
function paste(to, im, x0, y0) {
  for (let y = 0; y < im.h; y += 1) for (let x = 0; x < im.w; x += 1) put(to, x0 + x, y0 + y, get(im, x, y));
}
function halve(im) {
  const out = image(im.w / 2, im.h / 2, im.channels);
  for (let y = 0; y < out.h; y += 1) {
    for (let x = 0; x < out.w; x += 1) {
      const samples = [get(im, 2 * x, 2 * y), get(im, 2 * x + 1, 2 * y), get(im, 2 * x, 2 * y + 1), get(im, 2 * x + 1, 2 * y + 1)];
      const alpha = samples.reduce((sum, p) => sum + (p[3] ?? 255), 0);
      const pixel = get(out, x, y);
      for (let c = 0; c < 3; c += 1) {
        pixel[c] = alpha ? Math.round(samples.reduce((sum, p) => sum + p[c] * (p[3] ?? 255), 0) / alpha) : 0;
      }
      if (out.channels === 4) pixel[3] = Math.round(alpha / 4);
    }
  }
  return out;
}

// Where the pieces sit in the 1600 by 1200 panel: its bottom-left corner, a strip of its bottom edge and of its
// left edge where the tubes run straight, and a piece of the glass.
const CORNER = 64;
const EDGE = 16;
/// The nine-slice at the panel's own scale, 144 by 144.
export function chipAtlas(panel) {
  if (panel.w !== 1600 || panel.h !== 1200) throw new Error(`the panel is ${panel.w} by ${panel.h}, not 1600 by 1200`);
  // Put the straight sides of the chip at the first fully covered row/column of the pink tube.
  const inset = (x, y, dx, dy) => {
    for (let i = 0; i < CORNER; i += 1) {
      if ((get(panel, x + dx * i, y + dy * i)[3] ?? 255) === 255) return i;
    }
    throw new Error('the chip cut does not reach the pink outline');
  };
  const x = Math.max(...Array.from({ length: EDGE }, (_, i) => inset(0, 1040 + i, 1, 0)));
  const bottomY = Math.max(...Array.from({ length: EDGE }, (_, i) => inset(720 + i, panel.h - 1, 0, -1)));
  const y = panel.h - CORNER - bottomY;
  const corner = crop(panel, x, y, CORNER, CORNER);
  const bottom = crop(panel, 720, y, EDGE, CORNER);
  const left = crop(panel, x, 1040, CORNER, EDGE);
  const glass = crop(panel, 100, 1090, EDGE, EDGE);
  const size = CORNER + EDGE + CORNER;
  const atlas = image(size, size, panel.channels);
  paste(atlas, flip(corner, 'y'), 0, 0);
  paste(atlas, flip(bottom, 'y'), CORNER, 0);
  paste(atlas, flip(flip(corner, 'y'), 'x'), CORNER + EDGE, 0);
  paste(atlas, left, 0, CORNER);
  paste(atlas, glass, CORNER, CORNER);
  paste(atlas, flip(left, 'x'), CORNER + EDGE, CORNER);
  paste(atlas, corner, 0, CORNER + EDGE);
  paste(atlas, bottom, CORNER, CORNER + EDGE);
  paste(atlas, flip(corner, 'x'), CORNER + EDGE, CORNER + EDGE);
  if (atlas.channels === 4) {
    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
      if (((x >= CORNER && x < CORNER + EDGE) || (y >= CORNER && y < CORNER + EDGE)) && get(atlas, x, y)[3] !== 255) {
        throw new Error('the chip edge strips and centre must be fully covered');
      }
    }
  }
  return atlas;
}

/// A chip of `w` by `h` drawn from the nine-slice: corners as they are, edges and the centre repeated.
export function drawChip(atlas, corner, w, h) {
  const edge = atlas.w - 2 * corner;
  const out = image(w, h, atlas.channels);
  const from = (at, length) => (at < corner ? at : at >= length - corner ? atlas.w - (length - at) : corner + ((at - corner) % edge));
  for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) put(out, x, y, get(atlas, from(x, w), from(y, h)));
  return out;
}

export function writePreview(out, im) {
  const rgb = Buffer.alloc(im.w * im.h * 3);
  for (let y = 0; y < im.h; y += 1) for (let x = 0; x < im.w; x += 1) {
    const p = get(im, x, y);
    const checker = ((x >> 4) + (y >> 4)) % 2 ? 170 : 240;
    const a = (p[3] ?? 255) / 255;
    for (let c = 0; c < 3; c += 1) rgb[(y * im.w + x) * 3 + c] = Math.round(p[c] * a + checker * (1 - a));
  }
  fs.writeFileSync(out, Buffer.concat([Buffer.from(`P6\n${im.w} ${im.h}\n255\n`), rgb]));
}

export function writeLayout(dir, update) {
  const layoutPath = path.join(dir, 'layout.json');
  const raw = fs.readFileSync(layoutPath, 'utf8');
  const layout = JSON.parse(raw);
  update(layout);
  const eol = raw.includes('\r\n') ? '\r\n' : '\n';
  fs.writeFileSync(layoutPath, (JSON.stringify(layout, null, raw.match(/^\{\r?\n(\s+)/)?.[1].length ?? 2) + '\n').replace(/\n/g, eol));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const panel = qoiDecode(fs.readFileSync(path.join(dir, 'panel.qoi')));
  const chip = halve(chipAtlas(panel));
  const corner = CORNER / 2;
  if (args[0] === '--preview') {
    const [, out, w, h] = args;
    const drawn = drawChip(chip, corner, Number(w), Number(h));
    writePreview(out, drawn);
    console.log(`preview ${drawn.w} by ${drawn.h}`);
  } else {
    const data = qoiEncode(chip);
    const back = qoiDecode(data);
    if (!back.pixels.equals(chip.pixels)) throw new Error('the encoded chip does not decode to the same pixels');
    fs.writeFileSync(path.join(dir, 'chip.qoi'), data);
    writeLayout(dir, layout => {
      layout.chip = {
        image: 'chip.qoi',
        width: chip.w,
        height: chip.h,
        channels: chip.channels,
        bytes: data.length,
        corner,
        edge: EDGE / 2,
        padding: 30,
        note: 'a nine-slice at half the panel scale: the four corners are drawn as they are, the edges and the centre repeat; text starts `padding` pixels inside each side',
      };
    });
    console.log(`chip.qoi: ${chip.w} by ${chip.h}, ${data.length} bytes; corners ${corner}, edges ${EDGE / 2}`);
  }
}
