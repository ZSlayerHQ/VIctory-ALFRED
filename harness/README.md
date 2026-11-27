# The run harness

`world-run.sh` holds the run lock, checks for the recorder and calls `run-world.sh`; `run-world.sh` starts one run,
watches it, records it and writes a summary beside its log. The tests use the same scripts with stand-ins for the
game.

## Run settings

These are read from the environment before a run starts. An explicit value always wins over a default.

| Setting | Default | What it does |
|---|---|---|
| `WATCHDOG` | a timed run: `1`; an open, untimed session: `warn` | What the frozen-run watchdog does when the frame counter stands still once the scene has settled. `1` writes the stall line, saves the tail file and ends the run. `warn` writes the stall line, saves the tail file once per stall, keeps watching, writes the resume line when the counter moves again, and never ends the run. `0` turns the watchdog off. |
| `WATCHDOG_STALL_S` | `45` | How many seconds the counter may stand still before the watchdog acts, in either mode. |
| `HARNESS_MIN_FREE_GB` | `30` | Refuse to start a run when the logs' drive has less free space. |
| `HARNESS_STOP_FREE_GB` | `5` | End a running game when free space falls below this; `off` disables the guard. |
| `VICTORY_WARM_GENTLE` | unset | `1` or `0` forces a `--warm` warm-up gentle or raised; unset, it is gentle while the recorder is running. |
| `VICTORY_RUN_LOCK` | `harness/run.lock` | The run lock a `--save` restore must find held by this run. |
| `HARNESS_ROUTES_DIR` | `harness/routes` | Where `--route <name>` reads `<name>.keys` from. |

A run is timed when a hold ends it on the harness's own timer: `--heavy-seconds` or `--settled-seconds` (a route
may supply either). Such a run defaults to `WATCHDOG=1`, because a frozen timed leg is over and ending it frees the
lock. An open, untimed session defaults to `WATCHDOG=warn`, because its counter can stand still while the window is
minimised or covered, during a long load after a fast travel, or in the title's own pause: ending the session on a
false alarm is worse than the freeze.
