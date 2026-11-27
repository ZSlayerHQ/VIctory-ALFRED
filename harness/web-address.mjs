// The one rule for web addresses in what the tools print: a line that holds one is dropped whole, never masked, and
// counted ("N lines withheld"). Reason: the tools' output can be shared, and a masked address leaks whatever the mask missed
// (the check of 2026-11-26 found one masked only up to a comma). summarize-run.mjs and the live MCP server's
// log_find use it; harness/public-lines.sh carries the same rule in bash for the run's console.
//
// An address is a scheme of two or more letters before `://` (any case: `https://`, `HTTP://`, `file:///`), or a
// `www.` host. A drive path (`E:/x`, `E://x`, `C:\x`), a UNC share (`\\server\share`) and a colon before slashes in a
// path are not: the scheme's two letters are what tells them apart.
export const WEB_ADDRESS_RE = /(?:^|[^A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]+:\/\/|\bwww\.[A-Za-z0-9-]+\.[A-Za-z]/i;

/// Whether `text` holds a web address.
export function hasWebAddress(text) {
  return WEB_ADDRESS_RE.test(String(text));
}

/// The lines of `lines` without any that holds a web address, and how many were withheld.
export function withoutWebAddresses(lines) {
  const kept = [];
  let withheld = 0;
  for (const line of lines) {
    if (hasWebAddress(line)) withheld += 1;
    else kept.push(line);
  }
  return { lines: kept, withheld };
}

/// The note a tool prints for the lines it withheld.
export function withheldText(count) {
  return `${count} line${count === 1 ? '' : 's'} withheld (they hold a web address)`;
}

/// A JSON-able value with every string that holds a web address taken out: from an array the element is dropped,
/// elsewhere the string becomes null. `withheld` counts them.
export function dropWebAddresses(value) {
  let withheld = 0;
  const walk = (item) => {
    if (typeof item === 'string') {
      if (!hasWebAddress(item)) return item;
      withheld += 1;
      return null;
    }
    if (Array.isArray(item)) {
      const out = [];
      for (const child of item) {
        if (typeof child === 'string' && hasWebAddress(child)) withheld += 1;
        else out.push(walk(child));
      }
      return out;
    }
    if (item && typeof item === 'object') {
      const out = {};
      for (const [key, child] of Object.entries(item)) out[key] = walk(child);
      return out;
    }
    return item;
  };
  const result = walk(value);
  return { value: result, withheld };
}
