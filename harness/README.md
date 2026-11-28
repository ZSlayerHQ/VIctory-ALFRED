# The run harness

`world-run.sh` holds the run lock, checks for the recorder and calls `run-world.sh`; `run-world.sh` starts one run,
watches it, records it and writes a summary beside its log. The tests use the same scripts with stand-ins for the
game.

## Run settings

These are read from the environment before a run starts. An explicit value always wins over a default.

| Setting | Default | What it does |
|---|---|---|
| `HARNESS_MENU_FILE` (or `--menu-file <file>`) | none | The file the F1 settings menu saves to, for a session someone plays. The menu is handed the file, and at the next start each `VICTORY_NAME=value` line in it goes over the settings set (an empty value leaves that switch out); the picture size, console and frame rate limit chosen in the menu become their switches as the build's own launcher makes them, and so does DLSS when `STREAMLINE_DIR` (in `harness/local.env`) names a folder with NVIDIA's files; without it the run says DLSS is off and why. The menu's frame generation (a factor of 2 to 6) becomes `VICTORY_FRAMEGEN=1`, `VICTORY_FRAMEGEN_X` and `VICTORY_TEXTURE_ROOM_PERCENT=80` (type `fsr`: `VICTORY_FSR_FRAMEGEN=1` in place of `VICTORY_FRAMEGEN`) only over that DLSS and with `VICTORY_THIN=1` or `VICTORY_ASYNC_SUBMIT=1` in the run's final switches; otherwise the run says why it is off. The command line and the harness flags win. The log header has `harness env: menu-file=<name> (N values)`. |
| `HARNESS_BOOT_VIDEO` (or `--boot-video <file>`) | none | The loading cover, for a session someone plays: the video is shown over the whole primary screen (ffplay: silent, looping, always on top) from before the game starts until a frame line in the run's log shows the world drawn at a playable rate (`boot-cover.mjs`), for ten minutes at most and never past the run's end. `--boot-volume N` is the game's level while it shows; `--volume` applies again when it ends. A measured run does not use it. |
| `HARNESS_SETTINGS` (or `--settings <name>`) | `noone` | The named set of runtime switches a run starts with: every `VICTORY_NAME=value` line of `harness/settings/<name>.env`. `none` applies nothing. A switch given on the command line or by a harness flag wins over the file. The log header has `harness env: settings=<name> (N switches)` and each switch; the summary prints `Settings:`. |
| `WATCHDOG` | a timed run: `1`; an open, untimed session: `warn` | What the frozen-run watchdog does when the frame counter stands still once the scene has settled. `1` writes the stall line, saves the tail file and ends the run. `warn` writes the stall line, saves the tail file once per stall, keeps watching, writes the resume line when the counter moves again, and never ends the run. `0` turns the watchdog off. |
| `WATCHDOG_STALL_S` | `45` | How many seconds the counter may stand still before the watchdog acts, in either mode. |
| `HARNESS_MIN_FREE_GB` | `30` | Refuse to start a run when the logs' drive has less free space. |
| `HARNESS_STOP_FREE_GB` | `5` | End a running game when free space falls below this; `off` disables the guard. |
| `VICTORY_WARM_GENTLE` | unset | `1` or `0` forces a `--warm` warm-up gentle or raised; unset, it is gentle while the recorder is running. |
| `VICTORY_RUN_LOCK` | `harness/run.lock` | The run lock a `--save` restore must find held by this run. |
| `HARNESS_ROUTES_DIR` | `harness/routes` | Where `--route <name>` reads `<name>.keys` from. |

`--cold-pipelines` starts a run with no pipeline library for its build: the library is set aside in
`<cache_dir>/pipelines-cold-aside/<build>/`, put back as it was when the run ends, and the cold run's own discarded,
so a cold check never changes later runs (`pipeline-library.sh` says how, including a run killed outright).

A run is timed when a hold ends it on the harness's own timer: `--heavy-seconds` or `--settled-seconds` (a route
may supply either). Such a run defaults to `WATCHDOG=1`, because a frozen timed leg is over and ending it frees the
lock. An open, untimed session defaults to `WATCHDOG=warn`, because its counter can stand still while the window is
minimised or covered, during a long load after a fast travel, or in the title's own pause: ending the session on a
false alarm is worse than the freeze.
