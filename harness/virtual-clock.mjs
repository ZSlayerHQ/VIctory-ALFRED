// A clock for tests that must not depend on the machine's speed (the hosted runner's failure of 2026-11-26): time
// moves only when a test or the code under test sleeps on it, and whatever falls due on the way runs in order.
// live-fake-runtime.mjs, the live MCP server's tests and the watchdog's tests use it.

/// A virtual clock: `now()` in ms (from the real time it was made, so a stats object's `unix_ms` is near the reader's
/// clock), `every(ms, fn)` and `after(ms, fn)` schedule work on it, and `sleep(ms)` moves it on by `ms`, running
/// whatever falls due on the way in order, then resolves on the next turn of the event loop. Nothing on it waits on
/// real time, so a test driven by it gives the same result on a fast machine and a slow, loaded one.
export function virtualClock(start = Date.now()) {
  let t = start;
  let order = 0;
  const timers = new Set();
  const runUntil = (until) => {
    for (;;) {
      let next = null;
      for (const timer of timers) if (timer.at <= until && (!next || timer.at < next.at || (timer.at === next.at && timer.order < next.order))) next = timer;
      if (!next) break;
      t = Math.max(t, next.at);
      if (next.every) next.at += next.every;
      else timers.delete(next);
      next.fn();
    }
    t = Math.max(t, until);
  };
  const add = (ms, fn, every) => {
    const timer = { at: t + ms, every, fn, order: (order += 1) };
    timers.add(timer);
    return () => timers.delete(timer);
  };
  return {
    now: () => t,
    every: (ms, fn) => add(ms, fn, ms),
    after: (ms, fn) => add(ms, fn, 0),
    sleep: (ms) => {
      runUntil(t + Math.max(0, ms));
      return new Promise((resolve) => setImmediate(resolve));
    },
  };
}
