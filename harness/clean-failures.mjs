#!/usr/bin/env node
// Turn a VIctory log into "stage|hash|reason" lines for shader translation failures.
// usage: node clean-failures.mjs <log> <out>
import fs from 'node:fs';
import { parseTranslationFailures } from './parse-run.mjs';

const [, , logPath, outPath] = process.argv;
if (!logPath || !outPath) {
  console.error('usage: node clean-failures.mjs <log> <out>');
  process.exit(2);
}
const text = fs.readFileSync(logPath, 'utf8');
const out = parseTranslationFailures(text).map(
  ({ stage, hash, reason }) => `${stage}|${hash}|${reason}`,
);
fs.writeFileSync(outPath, out.join('\n') + (out.length ? '\n' : ''));
console.log(`${out.length} failures -> ${outPath}`);
