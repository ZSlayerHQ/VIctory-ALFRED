#!/usr/bin/env node
// The live switches the runtime registers, in one place: live.mjs prints them for live.sh, and live-drive.mjs
// refuses a plan that names anything else. The set and the kinds must equal the runtime's `switches!` table in
// `victory-gpu/src/live.rs` (live-names.test.mjs reads it); the order here is not the runtime's and means nothing.
// The shortest way to see them is `harness/live.sh names`.
export const LIVE_NAMES = Object.freeze([
  'VICTORY_KEEP_TEXTURES',
  'VICTORY_TEXTURE_LOW',
  'VICTORY_TEXTURE_HIGH',
  'VICTORY_LATE_WAITS',
  'LIVE_SKIP_DISPATCH',
  'LIVE_SKIP_DRAW',
  'LIVE_KEYS',
  'LIVE_MARK',
  'VICTORY_LAND_SCAN',
  'VICTORY_EXACT_BACK',
  'VICTORY_UPLOAD_POOL',
  'VICTORY_TEXTURE_RECHECK',
  'VICTORY_FETCH_QUEUE',
  'VICTORY_LIVE_HUD',
  'VICTORY_TEXTURE_MADE',
  'VICTORY_USER_COPY',
  'VICTORY_USER_STORE',
]);

const LIVE_NAME_SET = new Set(LIVE_NAMES);

/// True for a name the runtime's live registry knows.
export function isLiveName(name) {
  return LIVE_NAME_SET.has(name);
}

/// Each switch's kind, as the runtime's registry declares it (`victory-gpu/src/live.rs`, `SPECS`).
export const LIVE_KINDS = Object.freeze({
  VICTORY_KEEP_TEXTURES: 'onoff',
  VICTORY_TEXTURE_LOW: 'integer',
  VICTORY_TEXTURE_HIGH: 'integer',
  VICTORY_LATE_WAITS: 'onoff',
  LIVE_SKIP_DISPATCH: 'list',
  LIVE_SKIP_DRAW: 'list',
  LIVE_KEYS: 'keys',
  // A plan's mark (live.rs `Kind::Mark`): a whole number, zero or more, that changes nothing; a new one cuts a segment.
  LIVE_MARK: 'mark',
  VICTORY_LAND_SCAN: 'onoff',
  VICTORY_EXACT_BACK: 'onoff',
  VICTORY_UPLOAD_POOL: 'onoff',
  VICTORY_TEXTURE_RECHECK: 'onoff',
  VICTORY_FETCH_QUEUE: 'onoff',
  VICTORY_LIVE_HUD: 'onoff',
  VICTORY_TEXTURE_MADE: 'onoff',
  VICTORY_USER_COPY: 'onoff',
  VICTORY_USER_STORE: 'onoff',
});

/// The switch the stats file and segment lines leave out while it is off (`live.rs`, `listed_in_lines`).
export const UNLISTED_WHEN_OFF = 'VICTORY_LIVE_HUD';

/// A value as the runtime holds and prints it (`live.rs`, `parse_value` and `Value::text`): on/off for a flag
/// (anything but empty, 0, false, off or no is on), a whole number clamped to 1..100, a comma list without blanks,
/// a mark's whole number, zero or more, a key sequence's trimmed text. Null where the runtime would refuse the text and
/// keep the old value.
export function normalizeLiveValue(name, text) {
  const value = String(text ?? '').trim();
  switch (LIVE_KINDS[name]) {
    case 'onoff':
      return value === '' || /^(0|false|off|no)$/i.test(value) ? 'off' : 'on';
    case 'integer': {
      if (!/^[+-]?\d+$/.test(value)) return null;
      return String(Math.min(100, Math.max(1, Number(value))));
    }
    case 'mark': {
      if (!/^[+-]?\d+$/.test(value) || Number(value) < 0) return null;
      return String(Number(value));
    }
    case 'list':
      return value
        .split(',')
        .map((id) => id.trim())
        .filter(Boolean)
        .join(',');
    default:
      return value;
  }
}
