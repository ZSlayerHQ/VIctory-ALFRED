// Ending this run's game, for every watcher that can end it (settled-watch, watchdog, heavy-watch, disk-guard).
//
// The process ids come from a fresh selection at the moment of the kill, never from an earlier one. A game that ends
// by itself leaves its id free, Windows hands ids out again within seconds, and the harness takes 15 to 35 s to stop
// its watchers after a game ends: a watcher that force-ended a cached id in that window could end whatever program
// had been given it. The fresh selection (run-process.ps1) only returns a victory.exe of this run.

/// Selects afresh and ends what the selection returns now. `select` gives a list of process ids, or null when it
/// could not tell (PowerShell failed: not evidence that the game is gone); `end` ends one id and says whether it did.
/// Returns { outcome, fresh, ended }: 'unknown' (could not tell; nothing was ended, try again later), 'gone' (the
/// selection is empty: the game ended by itself), 'ended' (at least one was ended) or 'failed' (none could be).
export function endSelected(select, end) {
  const fresh = select();
  if (fresh === null) return { outcome: 'unknown', fresh: null, ended: [] };
  if (!fresh.length) return { outcome: 'gone', fresh, ended: [] };
  const ended = fresh.filter((pid) => end(pid));
  return { outcome: ended.length ? 'ended' : 'failed', fresh, ended };
}
