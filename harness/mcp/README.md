# The live MCP server

`live-server.mjs` is a small Model Context Protocol server over the harness's live tools. It reads the current
run's numbers and flips its switches, so an agent does not have to type shell commands or read raw log lines. It
has no dependencies beyond Node and speaks newline-delimited JSON-RPC 2.0 on standard input and output.

It reads `harness/run.lock` for whether a run is up, and takes the newest `harness/logs/*.log` for the numbers.
The control file is `live.txt` in that log's capture folder, and the stats file is `stats.json` beside it.
`--log`, `--file` and `--stats` (or `HARNESS_LOG`, `VICTORY_LIVE` and `VICTORY_LIVE_STATS`) override the search.
Nothing it returns carries a folder path, a process id or a command line, and `log_find` masks paths and 16-hex
ids.

## Tools

| Tool | Use it for |
|---|---|
| `game_status` | Whether a run holds the lock, its build, console and switches, uptime, the latest fps, frame time, parts and video memory; the stats object's shape, seq, age and whether it is stale (a stale object's figures are withheld), the segments closed (`segments_closed_total`) and the one under way (`segment_open`, in `segment_json`'s names). A stats object's longest frame is `frame_ms_worst` (it has no 95th percentile); a frame line's window gives `frame_ms_p95`. |
| `frame_parts` | Mean, median and p95 of frame time and each part over the last N seconds (default 10, at most 120). |
| `live_get` | Every live switch's kind and current value (`LIVE_MARK` among them, kind `mark`), where it was read, and whether its latest change has a runtime marker; a requested value is compared as the runtime holds it (`1` is `on`), and a stale object's values are marked as the last known (`stats_stale`, `stale_note`) and acknowledge nothing. |
| `live_set` | Set one or more switches between frames, wait up to 15 s for the markers, and restore the file if nothing answers. |
| `segments` | The live segment table from the log as objects, newest last, short stretches marked; the segment under way last, `open` true. |
| `run_plan` | Run an A/B/A plan through `live-drive.mjs` with its stop rules, each state confirmed from the run's `stats.json`, and return the verdict rows and whether the baseline was confirmed after it. |
| `send_keys` | Play a timed `seconds:Key` sequence through `LIVE_KEYS` and wait for its done line (or say the build lacks it). |
| `log_find` | Search the run log for a warning or counter the other tools do not cover; paths (a folder name with a space included) and ids are redacted, and a line holding a web address is never returned: it is counted in `withheld` (`harness/web-address.mjs`). |

There is deliberately no tool that starts, ends or kills a run, and no tool that takes a file path. The only file
a tool writes is the control file; a running plan also writes its sidecar beside the log.

## The stats file's two shapes

Both are read through `harness/stats-object.mjs`; `stats_shape` says which (`interval`: the runtime's object, one a
second covering every frame since the last; `sample`: the pre-rebuild one-second sample). A field a shape
lacks is `null`, never zero. `parts` are the presenting thread's budget phases (diagnostics builds), checked to sum to
`budget.frame_ms`; `summed_across_threads` (the older `parts`, and the frame line's parenthesised figures) overlap
each other and the frame and are not parts of it. An object is new by its `seq`, and stale past 5 s by its own
`unix_ms` (or its seq standing that long; the older shape: 15 s by the file's time): its figures are then withheld.
The shapes are fixed by `harness/fixtures/runtime-stats.mjs` and `runtime-lines.mjs`, copied from the runtime's own
tests and re-read from its source by `fixtures/runtime-shapes.test.mjs`; where an older note's field table disagrees,
the fixtures are right.

## Add it to `.mcp.json`

```json
{
  "mcpServers": {
    "victory-live": {
      "command": "node",
      "args": ["harness/mcp/live-server.mjs"]
    }
  }
}
```

No environment is needed. The script's path is relative to the repository root, where `harness/run.lock` and
`harness/logs/` live; the server resolves them from its own location, so the client's working folder does not
matter.
