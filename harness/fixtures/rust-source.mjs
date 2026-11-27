// Reading the runtime's Rust source for the fixtures' drift test: its string literals as the compiler sees them, a
// function's body, a constant's value, and a `format!` string turned into a regular expression. Small on purpose:
// enough of Rust's lexical rules (comments, char literals, raw strings, escapes, the line continuation) to read the
// strings of victory-gpu, not a parser.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

/// The victory checkout: VICTORY_REPO, else the one beside harness/ (the outer repository) or the outer repository's
/// from a lane worktree (worktrees/<lane>/harness/fixtures). Null where there is none (the public tree).
export function victoryRoot() {
  const roots = [
    process.env.VICTORY_REPO,
    path.join(here, '..', '..', 'victory'),
    path.join(here, '..', '..', '..', '..', 'victory'),
  ];
  for (const root of roots) {
    if (root && fs.existsSync(path.join(root, 'crates', 'victory-gpu', 'src', 'live.rs'))) return root;
  }
  return null;
}

/// One source file of victory-gpu (`src/<relative>`), its line endings made LF.
export function gpuSource(root, relative) {
  return fs.readFileSync(path.join(root, 'crates', 'victory-gpu', 'src', relative), 'utf8').replace(/\r\n/g, '\n');
}

const CHAR_LITERAL = /^'(?:\\(?:[nrt0\\'"]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\})|[^\\'\n])'/;

function decodeEscape(src, i) {
  // src[i] is the backslash.
  const c = src[i + 1];
  const simple = { n: '\n', t: '\t', r: '\r', 0: '\0', '\\': '\\', '"': '"', "'": "'" };
  if (c in simple) return [simple[c], i + 2];
  if (c === 'x') return [String.fromCharCode(parseInt(src.slice(i + 2, i + 4), 16)), i + 4];
  if (c === 'u') {
    const close = src.indexOf('}', i);
    return [String.fromCodePoint(parseInt(src.slice(i + 3, close), 16)), close + 1];
  }
  if (c === '\n') {
    // The line continuation: the newline and every whitespace character after it are not in the string.
    let j = i + 2;
    while (j < src.length && /\s/.test(src[j])) j += 1;
    return ['', j];
  }
  throw new Error(`an escape this reader does not know: \\${c}`);
}

/// Every string literal of `src` with its place: { start, end, value }. Comments and char literals are skipped;
/// `masked` is the source with every literal and comment blanked, for brace matching.
export function scan(src) {
  const literals = [];
  const mask = src.split('');
  const blank = (from, to) => {
    for (let k = from; k < to; k += 1) if (mask[k] !== '\n') mask[k] = ' ';
  };
  let i = 0;
  while (i < src.length) {
    if (src.startsWith('//', i)) {
      const end = src.indexOf('\n', i);
      const stop = end < 0 ? src.length : end;
      blank(i, stop);
      i = stop;
      continue;
    }
    if (src.startsWith('/*', i)) {
      const end = src.indexOf('*/', i + 2);
      const stop = end < 0 ? src.length : end + 2;
      blank(i, stop);
      i = stop;
      continue;
    }
    const raw = /^b?r(#*)"/.exec(src.slice(i, i + 12));
    if (raw && !/[A-Za-z0-9_]/.test(src[i - 1] ?? '')) {
      const close = `"${raw[1]}`;
      const from = i + raw[0].length;
      const end = src.indexOf(close, from);
      literals.push({ start: i, end: end + close.length, value: src.slice(from, end) });
      blank(i, end + close.length);
      i = end + close.length;
      continue;
    }
    if (src[i] === '"' || (src[i] === 'b' && src[i + 1] === '"' && !/[A-Za-z0-9_]/.test(src[i - 1] ?? ''))) {
      const start = i;
      i += src[i] === 'b' ? 2 : 1;
      let value = '';
      while (i < src.length && src[i] !== '"') {
        if (src[i] === '\\') {
          const [text, next] = decodeEscape(src, i);
          value += text;
          i = next;
        } else {
          value += src[i];
          i += 1;
        }
      }
      i += 1;
      literals.push({ start, end: i, value });
      blank(start, i);
      continue;
    }
    if (src[i] === "'") {
      const char = CHAR_LITERAL.exec(src.slice(i, i + 16));
      if (char) {
        blank(i, i + char[0].length);
        i += char[0].length;
        continue;
      }
    }
    i += 1;
  }
  return { literals, masked: mask.join('') };
}

/// The body of `fn <name>` (the first one after `from`, a text the source holds before it, if given), from its
/// opening brace to its closing one: { start, end, text }.
export function fnBody(src, name, { after = null } = {}) {
  const { masked } = scan(src);
  const base = after ? masked.indexOf(after) : 0;
  if (base < 0) throw new Error(`no ${after} in the source`);
  const re = new RegExp(String.raw`\bfn ${name}\b`, 'g');
  re.lastIndex = base;
  const m = re.exec(masked);
  if (!m) throw new Error(`no fn ${name} in the source`);
  const open = masked.indexOf('{', m.index);
  let depth = 0;
  for (let k = open; k < masked.length; k += 1) {
    if (masked[k] === '{') depth += 1;
    else if (masked[k] === '}') {
      depth -= 1;
      if (depth === 0) return { start: open, end: k + 1, text: src.slice(open, k + 1) };
    }
  }
  throw new Error(`fn ${name} has no end`);
}

/// The literals inside [start, end) of `src`.
export function literalsIn(src, start = 0, end = src.length) {
  return scan(src).literals.filter((lit) => lit.start >= start && lit.end <= end);
}

/// The one literal of `src` (or of `fn <test>`'s body) whose text contains `find`. Throws when none or several do,
/// so a fixture that names a pinned string fails when the runtime's test no longer pins it.
export function literalWith(src, find, { fn = null, where = (value) => value.includes(find) } = {}) {
  const range = fn ? fnBody(src, fn) : { start: 0, end: src.length };
  const found = literalsIn(src, range.start, range.end).filter((lit) => where(lit.value));
  if (found.length !== 1) {
    throw new Error(`${found.length} literals${fn ? ` in fn ${fn}` : ''} contain ${JSON.stringify(find)}`);
  }
  return found[0].value;
}

/// The literals of every `for <variable> in [ ... ]` array in `fn <test>`'s body: what the test checks present
/// (`key`) or absent (`gone`), by the loop's variable name.
export function loopArrays(src, test, variable) {
  const body = fnBody(src, test);
  const { masked } = scan(src);
  const out = [];
  const re = new RegExp(String.raw`for ${variable} in \[`, 'g');
  re.lastIndex = body.start;
  for (let m = re.exec(masked); m && m.index < body.end; m = re.exec(masked)) {
    let depth = 0;
    let k = m.index + m[0].length - 1;
    for (; k < masked.length; k += 1) {
      if (masked[k] === '[') depth += 1;
      else if (masked[k] === ']') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    out.push(...literalsIn(src, m.index, k).map((lit) => lit.value));
  }
  return out;
}

/// The values of the file's `const NAME: T = <number or string>;`, by name.
export function constants(src) {
  const { literals, masked } = scan(src);
  const out = {};
  for (const m of masked.matchAll(/\bconst ([A-Z][A-Z0-9_]*): [^=;]+= ([^;]+);/g)) {
    const value = m[2].trim();
    if (/^-?[\d_]+(\.\d+)?$/.test(value)) out[m[1]] = Number(value.replace(/_/g, ''));
  }
  // A string constant's literal is blanked in the masked text: read it beside the name.
  for (const m of src.matchAll(/\bconst ([A-Z][A-Z0-9_]*): &str = "/g)) {
    const lit = literals.find((l) => l.start === m.index + m[0].length - 1);
    if (lit) out[m[1]] = lit.value;
  }
  return out;
}

const escapeRe = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/// A Rust format string as an anchored regular expression: its text exactly, `{{` and `}}` as braces, each argument
/// a lazy group (named after the argument where it has an identifier's name), and an argument naming one of
/// `values` (the file's constants) as that value, formatted by its precision.
export function formatRegex(format, values = {}) {
  let re = '^';
  const names = new Set();
  const order = [];
  for (let i = 0; i < format.length; ) {
    if (format.startsWith('{{', i) || format.startsWith('}}', i)) {
      re += escapeRe(format[i]);
      i += 2;
      continue;
    }
    if (format[i] === '{') {
      const close = format.indexOf('}', i);
      const spec = format.slice(i + 1, close);
      const [name, precision] = spec.split(':');
      i = close + 1;
      if (name && Object.hasOwn(values, name)) {
        const value = values[name];
        const digits = /^\.(\d+)$/.exec(precision ?? '');
        re += escapeRe(digits && typeof value === 'number' ? value.toFixed(Number(digits[1])) : String(value));
        continue;
      }
      order.push(name || null);
      if (name && /^[A-Za-z_]\w*$/.test(name) && !names.has(name)) {
        names.add(name);
        re += `(?<${name}>[\\s\\S]*?)`;
      } else re += '([\\s\\S]*?)';
      continue;
    }
    re += escapeRe(format[i]);
    i += 1;
  }
  return { re: new RegExp(`${re}$`), arguments: order };
}
