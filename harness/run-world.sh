#!/usr/bin/env bash
# One VIctory world run, with the diagnostics that matter, condensed at exit.
# usage: run-world.sh [--timing] [--volume N|off] [--no-record] [--no-presentmon] [--aftermath] [--warm] [--live] [--frames 2400,2600] [--keys '55:Enter,...']
#                     [--route <name>] [--protocol team-270] [--dump-targets] [--print-env] [--settled-seconds N] [--settled-after-s N]
#                     [--settled-timeout N] [--world-timeout N]
#                     [--settled-fps F] [--settled-waiting K]
#                     [--console series-x|series-s] [--profile diag|release]
#                     [--copy-mode plain|none] [--heavy-seconds N] [--tag name] [--require-quiet] [--quiet-wait N]
#                     [--cold-pipelines]
# VICTORY_BIN overrides the binary (default $VICTORY_DIR/target/$PROFILE/victory.exe). GAME_EXE (the title's
# executable in the dump), VICTORY_CAPTURES (the root each run's capture folder goes under; CAPTURE names one run's
# folder outright) and VICTORY_SAVE_BACKUPS come from the environment, else harness/local.env on a machine that has
# one. There is no machine default: GAME_EXE is required, and the capture root falls back to a relative
# `victory-captures` folder in the working directory.
# VICTORY_WARM_GENTLE=1 or 0 forces the --warm warm-up gentle or raised; unset, it is gentle while OBS is running (obs.sh).
# VICTORY_RUN_LOCK names the run lock (default harness/run.lock); a --save restore checks it carries this run's token.
# The frozen-run watchdog. A timed run (--heavy-seconds or --settled-seconds: a hold that ends on the harness's own
# timer) defaults to WATCHDOG=1, and a counter standing still there ends the run. An open, untimed session defaults
# to WATCHDOG=warn: its counter can stand still while the window is minimised or covered, during a long load after a
# fast travel, or in the title's own pause, and ending the owner's session on a false alarm is worse than the freeze.
# WATCHDOG=warn writes the stall line, saves the tail file once per stall, keeps watching and writes the resume line;
# WATCHDOG=0 turns it off; WATCHDOG=1 ends the run at a stall. An explicit WATCHDOG=1, 0 or warn always wins.
# WATCHDOG_STALL_S (default 45) is how many seconds its frame counter may stand still once the scene has settled.
# --live makes an empty control file in the run's capture folder and points VICTORY_LIVE at it; harness/live.sh sets
# a switch there while the title runs, and harness/live-report.mjs reads the segments it cuts from the log. The same
# flag points VICTORY_LIVE_STATS at stats.json there, which the runtime rewrites atomically about once a second with
# one object covering every frame since the one before; harness/mcp/live-server.mjs reads both files, and
# stats-recorder.mjs appends each object to stats.jsonl beside them (presentmon-compare reads that history).
# The body is one function, called once at the end, so bash has read every line before it runs anything: an edit to
# this file during a run cannot change the rest of that run (world-run.sh runs a copy of it, too).
usage() {
  printf '%s\n' "usage: run-world.sh [--timing] [--volume N|off] [--no-record] [--no-presentmon] [--aftermath] [--warm] [--live] [--require-quiet] [--quiet-wait N]" \
    "                     [--frames 2400,2600] [--keys '55:Enter,...'] [--route <name>] [--protocol team-270] [--dump-targets]" \
    "                     [--print-env] [--settled-seconds N] [--settled-after-s N] [--settled-timeout N] [--world-timeout N]" \
    "                     [--settled-fps F] [--settled-waiting K] [--console series-x|series-s]" \
    "                     [--profile diag|release] [--settings noone|none|<name>] [--menu-file <file>] [--boot-video <file>] [--boot-volume N] [--copy-mode plain|none] [--heavy-seconds N] [--tag name]" \
    "                     [--require-quiet] [--quiet-wait N] [--cold-pipelines]" \
    | public_line
}

# A heartbeat from the game's own frame-rate line, about every ten seconds while it runs. Built here, so only the
# frame line's numbers reach the terminal, never the line itself. The log is read from its last 64 KiB only: a long
# run's log is many megabytes, and the newest frame line is always in the tail.
HEARTBEAT_TAIL_BYTES=65536
heartbeat_loop() {
  local line beat
  while sleep 10; do
    line=$(tail -c "$HEARTBEAT_TAIL_BYTES" "$LOG" 2>/dev/null | grep -aE ' fps, [0-9.]+ ms/frame' | tail -1)
    [ -n "$line" ] || continue
    beat=$(heartbeat_line "$line")
    [ -n "$beat" ] && printf '%s\n' "$beat" | public_line
  done
}

# One --require-quiet sample. load-line.mjs wait samples, keeps sampling every 15 s while the machine is not quiet,
# and stops at --quiet-wait seconds. Its waiting lines go through the public filter and into the log; a quiet result
# is left in QUIET_LINE. A refusal is said here, and the caller exits 3.
quiet_sample() {
  local out rc
  QUIET_LINE=
  # The waiting lines are said as they come, through the public filter, and go into the log: a wait can last many
  # minutes, and a silent one looks like a hang (2026-11-28: a run waited 20 minutes for another session's test
  # suite without a word, because the lines came out only when the wait returned).
  exec 3>&1
  out=$(
    node "$TOOLS/load-line.mjs" wait --quiet-wait "$QUIET_WAIT" 2>>"$LOG" | while IFS= read -r line; do
      printf '%s\n' "$line"
      case "$line" in
        'harness: waiting for quiet:'*)
          printf '%s\n' "$line" | public_line >&3
          printf '%s\n' "$line" >>"$LOG"
          ;;
      esac
    done
    exit "${PIPESTATUS[0]}"
  )
  rc=$?
  exec 3>&-
  if [ "$rc" = 3 ]; then
    QUIET_REASON=$(printf '%s\n' "$out" | tail -1)
    say_err "refused: ${QUIET_REASON:-the start load is not quiet}"
    return 3
  fi
  if [ "$rc" != 0 ]; then
    say_err "refused: the start load could not be read"
    return 3
  fi
  QUIET_LINE=$(printf '%s\n' "$out" | grep '^harness: load at ' | tail -1)
}
main() {
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
# world-run.sh runs a copy of this script beside copies of the files it sources; TOOLS is the harness checkout those
# copies came from, where the helpers, routes and logs stay.
TOOLS=${WORLD_RUN_HARNESS_DIR:-$HERE}
. "$HERE/local-env.sh"
load_local_env "$TOOLS"
# The terminal's output can be shared: one filter (public-lines.sh) decides what may reach it, and the header's path rule.
. "$HERE/public-lines.sh"
VICTORY_DIR=${VICTORY_DIR:-$TOOLS/../victory}
GAME_EXE=${GAME_EXE:-}
PROFILE=diag
TIMING=0
# PresentMon runs whenever PRESENTMON_EXE names it, unless --no-presentmon; --presentmon is still accepted.
PRESENTMON=auto
AFTERMATH=0
WARM=0
LIVE=0
LIVE_FILE=""
LIVE_STATS=""
LIVE_HISTORY=""
FRAMES=""
KEYS=""
ROUTE=""
PROTOCOL=""
CONSOLE=series-x
CONSOLE_GIVEN=""
COPY_MODE=""
LOOP_LIMIT=""
SKIP=""
DUMP_TARGETS=""
PRINT_ENV=""
# --cold-pipelines: the run starts with no pipeline library for its build (below, where the library is put in place).
COLD_PIPELINES=""
HEAVY_SECONDS=""
SETTLED_SECONDS=""
SETTLED_AFTER=""
SETTLED_TIMEOUT=""
WORLD_TIMEOUT=""
SETTLED_FPS=""
SETTLED_WAITING=""
TAG="run"
RECORD=1
DRED=1
VOLUME=0
REQUIRE_QUIET=0
# With --require-quiet, how long to keep sampling for a quiet machine before refusing; 0 refuses at once.
QUIET_WAIT=600
SAVE=""
SAVE_AS=""
# --settings: the named set of runtime switches the run starts with (harness/settings/<name>.env); none applies nothing.
SETTINGS=${HARNESS_SETTINGS:-noone}
# --menu-file: the file the F1 settings menu saves to and the run starts from (below); none unless asked.
MENU_FILE=${HARNESS_MENU_FILE:-}
# --boot-video: a video shown over the screen from the start until the world is on it (the loading cover, below);
# none unless asked. --boot-volume: the game's volume while it shows; --volume applies again when it ends.
BOOT_VIDEO=${HARNESS_BOOT_VIDEO:-}
BOOT_VOLUME=""
GPU_TRACE=0
TRACE_AFTER_MS=""
TRACE_FRAMES=1
RENDERDOC=0
CAPTURE_AFTER_MS=""
HELP=0
# Free space on the logs' drive: refuse to start below MIN_FREE_GB, and end the run below STOP_FREE_GB.
MIN_FREE_GB=${HARNESS_MIN_FREE_GB:-30}
STOP_FREE_GB=${HARNESS_STOP_FREE_GB:-5}
# The frozen-run watchdog's mode is chosen after the arguments and routes are read: a timed run defaults to 1 (end
# the frozen leg), an open, untimed session to warn (never end the owner's session on a false alarm). An explicit
# WATCHDOG=1, 0 or warn wins. WATCHDOG_STALL_S is the same for both modes.
WATCHDOG=${WATCHDOG-}
WATCHDOG_STALL_S=${WATCHDOG_STALL_S:-45}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) HELP=1; shift ;;
    --timing) TIMING=1; shift ;;
    --presentmon) PRESENTMON=1; shift ;;
    --no-presentmon) PRESENTMON=0; shift ;;
    --aftermath) AFTERMATH=1; shift ;;
    --warm) WARM=1; shift ;;
    --live) LIVE=1; shift ;;
    --frames) FRAMES=$2; shift 2 ;;
    --keys) KEYS=$2; shift 2 ;;
    --route) ROUTE=$2; shift 2 ;;
    --protocol) PROTOCOL=$2; shift 2 ;;
    --console) CONSOLE=$2; CONSOLE_GIVEN=1; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --settings) SETTINGS=$2; shift 2 ;;
    --menu-file) MENU_FILE=$2; shift 2 ;;
    --boot-video) BOOT_VIDEO=$2; shift 2 ;;
    --boot-volume) BOOT_VOLUME=$2; shift 2 ;;
    --copy-mode) COPY_MODE=$2; shift 2 ;;
    --loop-limit) LOOP_LIMIT=$2; shift 2 ;;
    --skip) SKIP=$2; shift 2 ;;
    --dump-targets) DUMP_TARGETS=1; shift ;;
    --print-env) PRINT_ENV=1; shift ;;
    --cold-pipelines) COLD_PIPELINES=1; shift ;;
    --heavy-seconds) HEAVY_SECONDS=$2; shift 2 ;;
    --settled-seconds) SETTLED_SECONDS=$2; shift 2 ;;
    --settled-after-s) SETTLED_AFTER=$2; shift 2 ;;
    --settled-timeout) SETTLED_TIMEOUT=$2; shift 2 ;;
    --world-timeout) WORLD_TIMEOUT=$2; shift 2 ;;
    --settled-fps) SETTLED_FPS=$2; shift 2 ;;
    --settled-waiting) SETTLED_WAITING=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --require-quiet) REQUIRE_QUIET=1; shift ;;
    --quiet-wait) QUIET_WAIT=$2; shift 2 ;;
    --no-record) RECORD=0; shift ;;
    --no-dred) DRED=0; shift ;;
    --volume) VOLUME=$2; shift 2 ;;
    --save) SAVE=$2; shift 2 ;;
    --gpu-trace) GPU_TRACE=1; shift ;;
    --renderdoc) RENDERDOC=1; shift ;;
    --capture-after-ms) CAPTURE_AFTER_MS=$2; shift 2 ;;
    --trace-after-ms) TRACE_AFTER_MS=$2; shift 2 ;;
    --trace-frames) TRACE_FRAMES=$2; shift 2 ;;
    --save-as) SAVE_AS=$2; shift 2 ;;
    *) say_err "unknown argument: $1"; exit 2 ;;
  esac
done

# --help is a question, not a run: it answers before the lock, the binary or anything else is needed.
if [ "$HELP" = 1 ]; then
  usage
  exit 0
fi

# --protocol: a named protocol from the team's rules, as a preset over the flags it sets — team-270 is their 270 s
# test: the route team-270, and a settled window of 275 s, unless given — 5 s past the protocol's last window, so
# PresentMon's last row (a second or two short of the game's end) still covers it. The summary reports their two
# windows from the `harness env: protocol=` line below.
if [ -n "$PROTOCOL" ]; then
  case "$PROTOCOL" in
    team-270)
      [ -z "$ROUTE" ] && ROUTE=team-270
      [ -z "$SETTLED_SECONDS" ] && SETTLED_SECONDS=275
      ;;
    *)
      say_err "--protocol: unknown protocol '$PROTOCOL' (team-270)"
      exit 2
      ;;
  esac
fi

# --route: a named key route from harness/routes/<name>.keys, and its default --heavy-seconds or --settled-seconds
# unless given. The route file is one <seconds>:<key>[:<hold>] entry per line with # comments (route-keys.mjs
# reads it).
if [ -n "$ROUTE" ] && [ -n "$KEYS" ]; then
  say_err "--keys and --route together: give one or the other"
  exit 2
fi
# HARNESS_ROUTES_DIR: where the routes are read from, the harness's own folder by default (a test points it at a
# fixture route of its own).
ROUTES=${HARNESS_ROUTES_DIR:-$TOOLS/routes}
if [ -n "$ROUTE" ]; then
  KEYS=$(node "$TOOLS/route-keys.mjs" keys "$ROUTES" "$ROUTE") || exit 2
  ROUTE_HEAVY=$(node "$TOOLS/route-keys.mjs" heavy-seconds "$ROUTES" "$ROUTE") || exit 2
  if [ -z "$HEAVY_SECONDS" ] && [ -n "$ROUTE_HEAVY" ]; then
    HEAVY_SECONDS=$ROUTE_HEAVY
  fi
  ROUTE_SETTLED=$(node "$TOOLS/route-keys.mjs" settled-seconds "$ROUTES" "$ROUTE") || exit 2
  if [ -z "$SETTLED_SECONDS" ] && [ -n "$ROUTE_SETTLED" ]; then
    SETTLED_SECONDS=$ROUTE_SETTLED
  fi
  ROUTE_AFTER=$(node "$TOOLS/route-keys.mjs" settled-after "$ROUTES" "$ROUTE") || exit 2
  if [ -z "$SETTLED_AFTER" ] && [ -n "$ROUTE_AFTER" ]; then
    SETTLED_AFTER=$ROUTE_AFTER
  fi
fi

# A profiled or captured run is one at a time, and its delay is the flag's: without it the trace or capture would
# begin during the loading screen, so it refuses.
if [ "$GPU_TRACE" = 1 ] && [ "$RENDERDOC" = 1 ]; then
  say_err "--gpu-trace and --renderdoc together: give one or the other"
  exit 2
fi
if [ "$RENDERDOC" = 1 ] && [ -z "$CAPTURE_AFTER_MS" ]; then
  say_err "--renderdoc needs --capture-after-ms <ms>: refusing to capture the loading screen"
  exit 2
fi
if [ "$RENDERDOC" = 1 ] && ! [[ "$CAPTURE_AFTER_MS" =~ ^[0-9]+$ ]]; then
  say_err "--capture-after-ms needs a whole number, not '$CAPTURE_AFTER_MS'"
  exit 2
fi
if [ "$GPU_TRACE" = 1 ] && [ -z "$TRACE_AFTER_MS" ]; then
  say_err "--gpu-trace needs --trace-after-ms <ms>: refusing to trace the loading screen"
  exit 2
fi
if [ "$GPU_TRACE" = 1 ]; then
  for PAIR in "trace-after-ms:$TRACE_AFTER_MS" "trace-frames:$TRACE_FRAMES"; do
    NAME=${PAIR%%:*}
    VALUE=${PAIR#*:}
    if ! [[ "$VALUE" =~ ^[0-9]+$ ]] || [ "$VALUE" -lt 0 ]; then
      say_err "--$NAME needs a whole number, not '$VALUE'"
      exit 2
    fi
  done
fi

# A run ends by one clock: the heavy scene's age or the settled scene's. Never both.
if [ -n "$SETTLED_SECONDS" ] && [ -n "$HEAVY_SECONDS" ]; then
  say_err "--settled-seconds and --heavy-seconds together: give one or the other"
  exit 2
fi

# A run holds the harness lock in world-run.sh; calling this script directly would skip it and the OBS refusal.
# --print-env starts nothing and --help answers a question, so both are allowed without it.
if [ "$PRINT_ENV" != 1 ] && [ "${WORLD_RUN_HOLDS_LOCK:-}" != 1 ]; then
  say_err "refused: run this through world-run.sh, which holds the run lock (WORLD_RUN_HOLDS_LOCK=1); --print-env and --help work directly"
  exit 2
fi

SAVE_CONFIG=${VICTORY_CONFIG:-$TOOLS/../victory/victory.toml}
SAVE_CONFIG_DIR=$(cd "$(dirname "$SAVE_CONFIG")" && { pwd -W 2>/dev/null || pwd; })
SAVE_CONFIG="$SAVE_CONFIG_DIR/$(basename "$SAVE_CONFIG")"
export VICTORY_CONFIG="$SAVE_CONFIG"
# The run lock world-run.sh holds. A --save restore must find it held by this run, not another one.
LOCK=${VICTORY_RUN_LOCK:-$TOOLS/run.lock}

STAMP=${WORLD_RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
# HARNESS_LOG_DIR names the log folder for a test; the real harness keeps its logs beside itself.
LOG_DIR=${HARNESS_LOG_DIR:-$TOOLS/logs}
LOG=$LOG_DIR/$STAMP-$TAG.log
CAPTURES=${VICTORY_CAPTURES:-victory-captures}
CAPTURE=${CAPTURE:-${CAPTURES%/}/$STAMP-$TAG}
mkdir -p "$LOG_DIR" "$CAPTURE"
# --live: the control file the runtime re-reads at frame boundaries. It starts empty (today's behaviour), and the
# terminal sees only its file name; the log's header names it the same way.
if [ "$LIVE" = 1 ]; then
  LIVE_FILE=${CAPTURE%/}/live.txt
  LIVE_STATS=${CAPTURE%/}/stats.json
  LIVE_HISTORY=${CAPTURE%/}/stats.jsonl
  : >"$LIVE_FILE"
  rm -f "$LIVE_STATS" "$LIVE_HISTORY"
fi

BIN=${VICTORY_BIN:-$VICTORY_DIR/target/$PROFILE/victory.exe}
if [ ! -x "$BIN" ]; then
  say_err "no victory binary for profile $PROFILE — build it: (cargo build --profile $PROFILE -p victory --features diagnostics)"
  exit 2
fi
# GAME_EXE names the title's executable in the dump; it has no default, so a run without one stops here with a
# message rather than guessing a path.
if [ -z "$GAME_EXE" ]; then
  say_err "refused: GAME_EXE is not set — put it in harness/local.env or the environment"
  exit 2
fi
BIN_DIR=$(cd "$(dirname "$BIN")" && { pwd -W 2>/dev/null || pwd; })
BIN="$BIN_DIR/$(basename "$BIN")"
# The runtime's data guard refuses to start the title from inside a git work tree: run from the snapshot folder
# when the binary is in one, otherwise from a fresh folder under the capture folder, and refuse loudly if neither
# is outside a work tree.
RUN_CWD=$(node "$TOOLS/run-cwd.mjs" choose --bin "$BIN" --lane "$VICTORY_DIR" --capture "$CAPTURE") || exit 2

if [ -n "$HEAVY_SECONDS" ] && ! [[ "$HEAVY_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  say_err "--heavy-seconds needs a number, not '$HEAVY_SECONDS'"
  exit 2
fi
for PAIR in "settled-seconds:$SETTLED_SECONDS" "settled-after-s:$SETTLED_AFTER" "settled-timeout:$SETTLED_TIMEOUT" "world-timeout:$WORLD_TIMEOUT" "settled-fps:$SETTLED_FPS" "settled-waiting:$SETTLED_WAITING"; do
  NAME=${PAIR%%:*}
  VALUE=${PAIR#*:}
  if [ -n "$VALUE" ] && ! [[ "$VALUE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    say_err "--$NAME needs a number, not '$VALUE'"
    exit 2
  fi
done
if ! [[ "$QUIET_WAIT" =~ ^[0-9]+$ ]]; then
  say_err "--quiet-wait needs a whole number of seconds, not '$QUIET_WAIT'"
  exit 2
fi
if [ -z "$WATCHDOG" ]; then
  if [ -n "$HEAVY_SECONDS" ] || [ -n "$SETTLED_SECONDS" ]; then
    WATCHDOG=1
  else
    WATCHDOG=warn
  fi
fi
case "$WATCHDOG" in
  0|off|no|false) WATCHDOG=0 ;;
  1|on|yes|true) WATCHDOG=1 ;;
  warn|warn-only) WATCHDOG=warn ;;
  *) say_err "refused: WATCHDOG is 1, 0 or warn, not '$WATCHDOG'"; exit 2 ;;
esac
if ! node -e 'const n = Number(process.argv[1]); process.exit(Number.isFinite(n) && n > 0 ? 0 : 1)' "$WATCHDOG_STALL_S"; then
  say_err "WATCHDOG_STALL_S needs a positive number of seconds, not '$WATCHDOG_STALL_S'"
  exit 2
fi

if [ -n "$BOOT_VIDEO" ]; then
  if [ ! -f "$BOOT_VIDEO" ]; then
    say_err "--boot-video: no such file"
    exit 2
  fi
  # --print-env starts no player, so a machine without one can still be asked what a run would get.
  if [ -z "$PRINT_ENV" ] && ! command -v ffplay >/dev/null 2>&1; then
    say_err "--boot-video: ffplay is not on the path (it comes with ffmpeg)"
    exit 2
  fi
fi
case "$BOOT_VOLUME" in
  '') ;;
  *[!0-9]*)
    say_err "--boot-volume needs a whole number from 0 to 100, not '$BOOT_VOLUME'"
    exit 2
    ;;
  *)
    if [ -z "$BOOT_VIDEO" ]; then
      say_err "--boot-volume goes with --boot-video: it is the game's volume while the video shows"
      exit 2
    fi
    ;;
esac

ENV_ARGS=(VICTORY_LOG=info "VICTORY_WAVE64_EMULATE=emulate" "VICTORY_CONFIG=$SAVE_CONFIG")
# DRED (the card's crash breadcrumbs) is on unless --no-dred, a timing diagnostic: a hang in such a run leaves no
# breadcrumbs. The runtime turns DRED on when the variable is set at all, so it is left out, not set to 0.
[ "$DRED" = 1 ] && ENV_ARGS+=("VICTORY_D3D12_DRED=1")
[ "$TIMING" = 1 ] && ENV_ARGS+=("VICTORY_GPU_TIMING=1")
ENV_ARGS+=("VICTORY_CAPTURE=$CAPTURE")
[ -n "$FRAMES" ] && ENV_ARGS+=("VICTORY_CAPTURE_FRAMES=$FRAMES")
[ -n "$KEYS" ] && ENV_ARGS+=("VICTORY_KEYS=$KEYS")
# The settled watcher's sentinel: VICTORY_KEYS_AFTER names it, and a "+..." entry in VICTORY_KEYS counts from its
# first appearance (job/keys-after in the runtime). A stale one from an earlier run would start the clock at launch.
AFTER=""
if [ -n "$SETTLED_SECONDS" ]; then
  AFTER="$CAPTURE/keys-after"
  rm -f "$AFTER"
  ENV_ARGS+=("VICTORY_KEYS_AFTER=$AFTER")
fi
[ -n "$COPY_MODE" ] && ENV_ARGS+=("VICTORY_COPY_MODE=$COPY_MODE")
[ "$LIVE" = 1 ] && ENV_ARGS+=("VICTORY_LIVE=$LIVE_FILE")
[ "$LIVE" = 1 ] && ENV_ARGS+=("VICTORY_LIVE_STATS=$LIVE_STATS")
[ -n "$LOOP_LIMIT" ] && ENV_ARGS+=("VICTORY_SHADER_LOOP_LIMIT=$LOOP_LIMIT")
[ -n "$SKIP" ] && ENV_ARGS+=("VICTORY_SKIP_PIPELINES=$SKIP")
[ -n "$DUMP_TARGETS" ] && ENV_ARGS+=("VICTORY_DUMP_TARGETS=1" "VICTORY_DUMP_UNSAFE=1")
# --print-env: a test's view of what the run would get; nothing is started.
# Aftermath (diag build, NVIDIA): the crash dump, its JSON and the shader debug info land in the capture folder.
# Markers are on with it: a hang run is for the dump, not for frame times.
if [ "$AFTERMATH" = 1 ]; then
  ROOT=$(cd "$TOOLS/.." && { pwd -W 2>/dev/null || pwd; })
  ENV_ARGS+=("VICTORY_AFTERMATH=$CAPTURE" "VICTORY_AFTERMATH_MARKERS=1"
    "VICTORY_AFTERMATH_DLL=${VICTORY_AFTERMATH_DLL:-$ROOT/tools/nsight-aftermath-sdk-2026.3/lib/x64}")
fi

# --menu-file: the file the F1 settings menu saves to. The build's own launcher (`play`) hands the menu that file
# and turns it into switches at the next start; the harness starts `run`, which does neither, so without this the
# menu cannot save and a saved choice is never applied. Each VICTORY_NAME=value line goes over the settings set (an
# empty value leaves that switch out), and the launcher choices `run` understands become their switches as `play`
# makes them: the picture size, the console, the frame rate limit, and DLSS when STREAMLINE_DIR (harness/local.env or
# the environment) names a folder that holds NVIDIA's files (the runtime checks their signatures itself), and frame
# generation over that DLSS (after the settings set, below). The command line and the flags above win over the file.
MENU_COUNT=0
MENU_SKIPPED=0
MENU_UNSET=" "
MENU_PUT=" "
MENU_FG=0
MENU_FG_TYPE=auto
MENU_NAME=""
MENU_DLSS=""
MENU_DLSS_WHY=""
MENU_FG_WHY=""
if [ -n "$MENU_FILE" ]; then
  case "$MENU_FILE" in /*|[A-Za-z]:[\\/]*) ;; *) MENU_FILE=$PWD/$MENU_FILE ;; esac
  MENU_NAME=${MENU_FILE##*[\\/]}
  MENU_BY_FLAGS=" "
  for E in "${ENV_ARGS[@]}"; do MENU_BY_FLAGS+="${E%%=*} "; done
  menu_put() { # NAME VALUE: one switch from the menu, unless the command line or a flag already set it
    [ -n "${!1+x}" ] && return 0
    case "$MENU_BY_FLAGS" in *" $1 "*) return 0 ;; esac
    if [ -n "$2" ]; then ENV_ARGS+=("$1=$2"); else MENU_UNSET+="$1 "; fi
    MENU_PUT+="$1 "
    MENU_COUNT=$((MENU_COUNT + 1))
  }
  MENU_RES=""
  MENU_FPS=""
  MENU_CONSOLE=""
  if [ -f "$MENU_FILE" ]; then
    while IFS= read -r PAIR || [ -n "$PAIR" ]; do
      PAIR=${PAIR%$'\r'}
      case "$PAIR" in ''|'#'*) continue ;; esac
      NAME=${PAIR%%=*}
      VALUE=${PAIR#*=}
      case "$NAME" in
        "$PAIR"|VICTORY_*[!A-Z0-9_]*) MENU_SKIPPED=$((MENU_SKIPPED + 1)) ;;
        VICTORY_MENU_RES) MENU_RES=$VALUE ;;
        VICTORY_MENU_FPS) MENU_FPS=$VALUE ;;
        VICTORY_MENU_CONSOLE) MENU_CONSOLE=$VALUE ;;
        VICTORY_MENU_FG_TYPE) MENU_FG_TYPE=$VALUE ;;
        VICTORY_MENU_FRAMEGEN) MENU_FG=$VALUE ;;
        VICTORY_MENU_DLSS) MENU_DLSS=$VALUE ;;
        VICTORY_MENU_*) ;;
        VICTORY_*) menu_put "$NAME" "$VALUE" ;;
        *) MENU_SKIPPED=$((MENU_SKIPPED + 1)) ;;
      esac
    done < "$MENU_FILE"
  fi
  case "$MENU_CONSOLE" in
    series-x|series-s) [ -z "$CONSOLE_GIVEN" ] && CONSOLE=$MENU_CONSOLE ;;
  esac
  if [ "$CONSOLE" != series-s ]; then
    case "$MENU_RES" in
      1440) menu_put VICTORY_NATIVE_1440P 1 ;;
      1080to1440) menu_put VICTORY_RENDER_1080P 1 ;;
    esac
  fi
  case "$MENU_FPS" in
    ''|*[!0-9]*) ;;
    *)
      MENU_FPS=$((10#$MENU_FPS))
      if [ "$MENU_FPS" -ge 1 ] && [ "$MENU_FPS" -le 1000 ]; then
        menu_put VICTORY_FRAME_INTERVAL_US "$(((1000000 + MENU_FPS / 2) / MENU_FPS))"
      fi
      ;;
  esac
  case "$MENU_FG_TYPE" in auto|dlss|fsr) ;; *) MENU_FG_TYPE=auto ;; esac
  # 0 is off and 2 to 6 a factor; anything else is ignored, as an unknown DLSS mode is.
  case "$MENU_FG" in [2-6]) ;; *) MENU_FG=0 ;; esac
  # DLSS as `play` sets it: a mode the build knows, and a folder with NVIDIA's files. Without the folder the choice
  # stays saved in the menu's file and the run says why it is off.
  case "$MENU_DLSS" in
    dlaa|quality|balanced|performance)
      if [ -n "${STREAMLINE_DIR:-}" ] && [ -f "$STREAMLINE_DIR/sl.interposer.dll" ]; then
        menu_put VICTORY_DLSS 1
        menu_put VICTORY_DLSS_MODE "$MENU_DLSS"
        menu_put VICTORY_STREAMLINE_DIR "$STREAMLINE_DIR"
      else
        MENU_DLSS_WHY="the menu asks for DLSS ($MENU_DLSS), but STREAMLINE_DIR names no folder with NVIDIA's files: DLSS is off"
      fi
      ;;
  esac
fi

# --settings: a named set of runtime switches, one NAME=value a line in harness/settings/<name>.env, added to every
# run. The default is the base build's own launcher profile: the harness starts `run`, which applies none, and a night
# of runs was measured at bare defaults before anyone saw it (2026-11-27). A switch already given on the command
# line or by a flag above wins over the file; `--settings none` applies nothing. Every switch is in the log header
# with the rest of the environment, under one `harness env: settings=` line naming the set.
SETTINGS_COUNT=0
SETTINGS_PAIRS=()
case "$SETTINGS" in
  none) ;;
  *[!a-z0-9-]*|'')
    say_err "--settings: a name of small letters, digits and dashes, or none (not '$SETTINGS')"
    exit 2
    ;;
  *)
    SETTINGS_FILE=$TOOLS/settings/$SETTINGS.env
    if [ ! -f "$SETTINGS_FILE" ]; then
      say_err "--settings: no harness/settings/$SETTINGS.env (none applies nothing)"
      exit 2
    fi
    SET_BY_FLAGS=" "
    for E in "${ENV_ARGS[@]}"; do SET_BY_FLAGS+="${E%%=*} "; done
    while IFS= read -r PAIR || [ -n "$PAIR" ]; do
      PAIR=${PAIR%$'\r'}
      case "$PAIR" in ''|'#'*) continue ;; esac
      NAME=${PAIR%%=*}
      case "$NAME" in
        VICTORY_*[!A-Z0-9_]*|"$PAIR")
          say_err "--settings: $SETTINGS.env has a line that is not VICTORY_NAME=value"
          exit 2
          ;;
        VICTORY_*) ;;
        *)
          say_err "--settings: $SETTINGS.env has a line that is not VICTORY_NAME=value"
          exit 2
          ;;
      esac
      [ -n "${!NAME+x}" ] && continue
      case "$SET_BY_FLAGS" in *" $NAME "*) continue ;; esac
      case "$MENU_UNSET" in *" $NAME "*) continue ;; esac
      ENV_ARGS+=("$PAIR")
      SETTINGS_PAIRS+=("$PAIR")
      SETTINGS_COUNT=$((SETTINGS_COUNT + 1))
    done < "$SETTINGS_FILE"
    ;;
esac

# With a menu file the menu is told what this run really has, so it shows the true choices and can save: the file,
# the console, the picture size, the frame rate limit, DLSS and frame generation as the run's switches give them.
# No VICTORY_MENU_PLAY is given, so its "Save and restart" saves and asks for the restart by hand.
if [ -n "$MENU_FILE" ]; then
  menu_final() { # NAME: its value in the run's environment (the command line, else what the harness passes)
    if [ -n "${!1+x}" ]; then printf '%s' "${!1}"; return 0; fi
    local ENTRY FOUND=""
    for ENTRY in "${ENV_ARGS[@]}"; do [ "${ENTRY%%=*}" = "$1" ] && FOUND=${ENTRY#*=}; done
    printf '%s' "$FOUND"
  }
  MENU_SHOWN_RES=4k
  [ "$(menu_final VICTORY_NATIVE_1440P)" = 1 ] && MENU_SHOWN_RES=1440
  [ "$(menu_final VICTORY_RENDER_1080P)" = 1 ] && MENU_SHOWN_RES=1080to1440
  MENU_SHOWN_DLSS=off
  if [ "$(menu_final VICTORY_DLSS)" = 1 ]; then
    MENU_SHOWN_DLSS=$(menu_final VICTORY_DLSS_MODE)
    case "$MENU_SHOWN_DLSS" in dlaa|quality|balanced|performance) ;; *) MENU_SHOWN_DLSS=quality ;; esac
  fi
  # Frame generation as `play` sets it, decided on the run's final switches (the settings set, the menu, the command
  # line): it needs DLSS upscaling on and the line of submissions (VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1); the
  # menu's type fsr (FSR 3) puts VICTORY_FSR_FRAMEGEN=1 in place of VICTORY_FRAMEGEN. Off, the run says why.
  fg_put() { # NAME [VALUE]: over the settings set unless the command line, a flag or the menu file gave it; no VALUE leaves it out
    [ -n "${!1+x}" ] && return 0
    case "$MENU_BY_FLAGS$MENU_PUT" in *" $1 "*) return 0 ;; esac
    local ENTRY KEPT=() SET_KEPT=()
    for ENTRY in "${ENV_ARGS[@]}"; do [ "${ENTRY%%=*}" = "$1" ] || KEPT+=("$ENTRY"); done
    ENV_ARGS=("${KEPT[@]}")
    for ENTRY in ${SETTINGS_PAIRS[@]+"${SETTINGS_PAIRS[@]}"}; do [ "${ENTRY%%=*}" = "$1" ] || SET_KEPT+=("$ENTRY"); done
    SETTINGS_PAIRS=(${SET_KEPT[@]+"${SET_KEPT[@]}"})
    SETTINGS_COUNT=${#SETTINGS_PAIRS[@]}
    if [ -n "${2:-}" ]; then ENV_ARGS+=("$1=$2"); MENU_COUNT=$((MENU_COUNT + 1)); fi
  }
  MENU_SHOWN_FG=0
  MENU_FG_KEY=VICTORY_FRAMEGEN
  [ "$MENU_FG_TYPE" = fsr ] && MENU_FG_KEY=VICTORY_FSR_FRAMEGEN
  if [ "$MENU_FG" != 0 ]; then
    MENU_FG_ASK="the menu asks for frame generation (${MENU_FG}x), but"
    if [ -n "${VICTORY_FRAMEGEN+x}" ] && [ "$VICTORY_FRAMEGEN" != 1 ]; then
      MENU_FG_WHY="$MENU_FG_ASK the command line's VICTORY_FRAMEGEN=$VICTORY_FRAMEGEN keeps it off"
    elif [ -n "${!MENU_FG_KEY+x}" ] && [ "${!MENU_FG_KEY}" != 1 ]; then
      MENU_FG_WHY="$MENU_FG_ASK the command line's $MENU_FG_KEY=${!MENU_FG_KEY} keeps it off"
    elif [ "$MENU_SHOWN_DLSS" = off ]; then
      MENU_FG_WHY="$MENU_FG_ASK it needs DLSS upscaling on with NVIDIA's files: frame generation is off"
    elif [ "$(menu_final VICTORY_THIN)" != 1 ] && [ "$(menu_final VICTORY_ASYNC_SUBMIT)" != 1 ]; then
      MENU_FG_WHY="$MENU_FG_ASK it needs VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1: frame generation is off"
    else
      if [ "$MENU_FG_TYPE" = fsr ]; then
        fg_put VICTORY_FSR_FRAMEGEN 1
        fg_put VICTORY_FRAMEGEN
      else
        fg_put VICTORY_FRAMEGEN 1
      fi
      fg_put VICTORY_FRAMEGEN_X "$MENU_FG"
      fg_put VICTORY_TEXTURE_ROOM_PERCENT 80
      # The factor the runtime reads: 0 is off, a number clamps to 2..6, anything else is 2.
      MENU_SHOWN_FG=$(menu_final VICTORY_FRAMEGEN_X)
      case "$MENU_SHOWN_FG" in
        ''|*[!0-9]*) MENU_SHOWN_FG=2 ;;
        *)
          MENU_SHOWN_FG=$((10#$MENU_SHOWN_FG))
          if [ "$MENU_SHOWN_FG" -gt 6 ]; then MENU_SHOWN_FG=6; elif [ "$MENU_SHOWN_FG" -eq 1 ]; then MENU_SHOWN_FG=2; fi
          ;;
      esac
      if [ "$(menu_final "$MENU_FG_KEY")" != 1 ]; then
        MENU_SHOWN_FG=0
        MENU_FG_WHY="$MENU_FG_ASK a line of the menu file leaves $MENU_FG_KEY off"
      fi
    fi
  fi
  ENV_ARGS+=("VICTORY_MENU_FILE=$MENU_FILE" "VICTORY_MENU_CONSOLE=$CONSOLE" "VICTORY_MENU_RES=$MENU_SHOWN_RES"
    "VICTORY_MENU_DLSS=$MENU_SHOWN_DLSS" "VICTORY_MENU_FRAMEGEN=$MENU_SHOWN_FG" "VICTORY_MENU_FG_TYPE=$MENU_FG_TYPE")
  MENU_INTERVAL=$(menu_final VICTORY_FRAME_INTERVAL_US)
  case "$MENU_INTERVAL" in
    ''|*[!0-9]*) ;;
    *)
      MENU_INTERVAL=$((10#$MENU_INTERVAL))
      [ "$MENU_INTERVAL" -gt 0 ] && ENV_ARGS+=("VICTORY_MENU_FPS=$(((1000000 + MENU_INTERVAL / 2) / MENU_INTERVAL))")
      ;;
  esac
fi

say "log:     $LOG"
say "capture: $CAPTURE"
[ "$LIVE" = 1 ] && say "live:      $(basename "$LIVE_FILE")"
# PresentMon is the one frame-time count independent of the runtime's own meter: say at start when it cannot run
# (never its path: the terminal's output can be shared), and start no watcher that would only fail.
if [ -z "${PRESENTMON_EXE:-}" ]; then
  say "PresentMon: not configured (set PRESENTMON_EXE in harness/local.env)"
  PRESENTMON=0
elif [ ! -f "$PRESENTMON_EXE" ]; then
  say "PresentMon: not configured (PRESENTMON_EXE names no file; set it in harness/local.env)"
  PRESENTMON=0
elif [ "$PRESENTMON" != 0 ]; then
  PRESENTMON=1
fi
say "profile: $PROFILE  console: $CONSOLE  timing: $TIMING  presentmon: $PRESENTMON"
say "settings: $SETTINGS ($SETTINGS_COUNT switches)"
if [ -n "$MENU_FILE" ]; then
  MENU_FG_SAID=off
  if [ "$MENU_SHOWN_FG" != 0 ]; then
    MENU_FG_SAID="${MENU_SHOWN_FG}x (DLSS)"
    [ "$MENU_FG_TYPE" = fsr ] && MENU_FG_SAID="${MENU_SHOWN_FG}x (FSR 3)"
  fi
  say "menu file: $MENU_NAME ($MENU_COUNT values applied; DLSS $MENU_SHOWN_DLSS; frame generation $MENU_FG_SAID)"
  [ -n "$MENU_DLSS_WHY" ] && say "menu file: $MENU_DLSS_WHY"
  [ -n "$MENU_FG_WHY" ] && say "menu file: $MENU_FG_WHY"
  [ "$MENU_SKIPPED" -gt 0 ] && say "menu file: $MENU_SKIPPED lines skipped (not VICTORY_NAME=value)"
fi
say ""

# --print-env: what the run would get, then stop before anything is started. The tests use it to pin the env, and
# with a capture flag it also prints the command that tool would run.
if [ -n "$PRINT_ENV" ]; then
  # The harness's own watchdog choice, so a test and an operator can see the default the run would use.
  printf 'watchdog=%s\n' "$WATCHDOG"
  printf 'harness env: settings=%s (%s switches)\n' "$SETTINGS" "$SETTINGS_COUNT"
  [ -n "$MENU_FILE" ] && printf 'harness env: menu-file=%s (%s values)\n' "$MENU_NAME" "$MENU_COUNT"
  [ -n "$BOOT_VIDEO" ] && printf 'harness env: boot-video=%s%s\n' "${BOOT_VIDEO##*[\\/]}" "${BOOT_VOLUME:+ (the game at $BOOT_VOLUME while it shows)}"
  [ -n "$WORLD_TIMEOUT" ] && printf 'harness env: world-timeout=%s\n' "$WORLD_TIMEOUT"
  # Only under --cold-pipelines; no library is touched here.
  [ -n "$COLD_PIPELINES" ] && printf 'harness env: pipelines=cold (the build'"'"'s library is set aside for the run and put back after it)\n'
  # Only under --protocol: an ordinary run's --print-env is unchanged.
  if [ -n "$PROTOCOL" ]; then
    printf 'harness env: protocol=%s\n' "$PROTOCOL"
    printf 'harness env: route=%s\n' "$ROUTE"
    printf 'harness env: settled-seconds=%s\n' "$SETTLED_SECONDS"
  fi
  printf '%s\n' "${ENV_ARGS[@]}"
  printf '%s\n' run "$GAME_EXE" --execute --console "$CONSOLE" > "$CAPTURE/game-args.txt"
  printf '%s\n' "${ENV_ARGS[@]}" > "$CAPTURE/env.txt"
  if [ "$GPU_TRACE" = 1 ]; then
    printf 'gpu-trace: '
    node "$TOOLS/gpu-trace.mjs" print --bin "$BIN" --dir "$VICTORY_DIR" \
      --game-args-file "$CAPTURE/game-args.txt" --env-file "$CAPTURE/env.txt" \
      --after-ms "$TRACE_AFTER_MS" --frames "$TRACE_FRAMES" --out "$CAPTURE/gputrace"
  fi
  if [ "$RENDERDOC" = 1 ]; then
    printf 'renderdoc: '
    node "$TOOLS/renderdoc.mjs" print --bin "$BIN" --dir "$VICTORY_DIR" \
      --game-args-file "$CAPTURE/game-args.txt" --out "$CAPTURE/rdoc" --after-ms "$CAPTURE_AFTER_MS"
  fi
  exit 0
fi

# Free space is guarded twice: a run that starts on a drive with too little room would lose its log first, so it is
# refused here; disk-guard.mjs then watches every poll while the game runs.
if ! DISK_CHECK=$(node "$TOOLS/disk-guard.mjs" check --path "$LOG_DIR" --min "$MIN_FREE_GB" 2>&1); then
  say_err "refused: ${DISK_CHECK:-the free-space check failed}"
  exit 2
fi

# The log starts with what the run was, one setting per line, so experiment-table.mjs can read it back.
{
  # The build line: a lane checkout's commit, or, for a snapshot build, the commit, branch and profile from
  # snapshot.txt beside VICTORY_BIN (build-line.mjs says why).
  node "$TOOLS/build-line.mjs" "$VICTORY_DIR" "${VICTORY_BIN:-}" "$PROFILE" "$CONSOLE"
  # VICTORY_LIVE is named by the `harness env: live=` line below, as a file name only.
  for E in "${ENV_ARGS[@]}"; do
    case "$E" in VICTORY_LIVE=*) continue ;; esac
    harness_env_line "$E"
  done
  [ -n "${VICTORY_BIN:-}" ] && harness_env_line "VICTORY_BIN=$VICTORY_BIN"
  # A warm-up's bar feed writes its own errors to a file beside its log, never to null (a feed that died silently
  # left the bar stale on the 09:19 warm-up).
  [ "$WARM" = 1 ] && echo "harness: the warm feed's errors go to $LOG.warm.feed"
  # A variable given on the command line (VICTORY_D3D12_DEBUG=1 ./run-world.sh) is passed through to the game
  # but is not in ENV_ARGS; record every other VICTORY_* variable too so the experiment table can tell runs apart.
  SET_NAMES=" VICTORY_BIN "
  for E in "${ENV_ARGS[@]}"; do SET_NAMES+="${E%%=*} "; done
  for NAME in $(compgen -e | sort); do
    case "$NAME" in VICTORY_*) ;; *) continue ;; esac
    case "$SET_NAMES" in *" $NAME "*) continue ;; esac
    harness_env_line "$NAME=${!NAME}"
  done
  echo "capture: $(harness_public_path "$CAPTURE")"
  echo "harness env: cwd=$(harness_public_path "$RUN_CWD")"
  echo "harness env: settings=$SETTINGS ($SETTINGS_COUNT switches)"
  if [ -n "$MENU_FILE" ]; then echo "harness env: menu-file=$MENU_NAME ($MENU_COUNT values)"; fi
  if [ -n "$BOOT_VIDEO" ]; then echo "harness env: boot-video=${BOOT_VIDEO##*[\\/]}${BOOT_VOLUME:+ (the game at $BOOT_VOLUME while it shows)}"; fi
  # The volumes the run reads and writes, once: the game's, and the cache's and data's when on other drives.
  # Each helper's exit is recorded in the log; a failure is a line to read, not a silence to blame on the game.
  node "$TOOLS/disk-usage.mjs" --config "$SAVE_CONFIG" "game=$(dirname "$GAME_EXE")" 2>&1
  echo "harness: disk-usage exit=$?"
  [ "$LIVE" = 1 ] && echo "harness env: live=$(basename "$LIVE_FILE")"
  [ "$LIVE" = 1 ] && echo "harness env: live-stats=$(basename "$LIVE_STATS")"
  [ -n "$ROUTE" ] && echo "harness env: route=$ROUTE"
  [ -n "$PROTOCOL" ] && echo "harness env: protocol=$PROTOCOL"
  [ -n "$SETTLED_SECONDS" ] && echo "harness env: settled-seconds=$SETTLED_SECONDS"
  [ -n "$SETTLED_TIMEOUT" ] && echo "harness env: settled-timeout=$SETTLED_TIMEOUT"
  [ -n "$WORLD_TIMEOUT" ] && echo "harness env: world-timeout=$WORLD_TIMEOUT"
  [ -n "$COLD_PIPELINES" ] && echo "harness env: pipelines=cold"
  [ -n "$SETTLED_FPS" ] && echo "harness env: settled-fps=$SETTLED_FPS"
  [ -n "$SETTLED_WAITING" ] && echo "harness env: settled-waiting=$SETTLED_WAITING"
  [ "$GPU_TRACE" = 1 ] && echo "harness env: gpu-trace=after ${TRACE_AFTER_MS} ms, ${TRACE_FRAMES} frames, output $(harness_public_path "$CAPTURE")/gputrace"
  [ "$RENDERDOC" = 1 ] && echo "harness env: renderdoc=after ${CAPTURE_AFTER_MS} ms, output $(harness_public_path "$CAPTURE")/rdoc"
  if [ "$RECORD" = 1 ]; then echo "harness env: record=on"; else echo "harness env: record=off"; fi
  echo "harness env: volume=$VOLUME"
  case "$WATCHDOG" in
    1) echo "harness env: watchdog=on" ;;
    warn) echo "harness env: watchdog=warn" ;;
    *) echo "harness env: watchdog=off" ;;
  esac
  [ "$WATCHDOG" != 0 ] && echo "harness env: watchdog-stall-s=$WATCHDOG_STALL_S"
} >"$LOG"

# --require-quiet: sample once before the warm-up too, so a refusal never costs a five-minute warm-up. The lock is
# still held while this waits, and the save restore and pipeline library have not been touched yet; the same wait
# runs again after the warm-up, on the usual start load line.
if [ "$REQUIRE_QUIET" = 1 ] && [ "$WARM" = 1 ]; then
  quiet_sample
  PRE_QUIET_RC=$?
  printf 'harness: quiet before warm-up exit=%s\n' "$PRE_QUIET_RC" >>"$LOG"
  [ "$PRE_QUIET_RC" = 0 ] || exit 3
  printf '%s\n' "$QUIET_LINE" >>"$LOG"
fi

# --save: the title's save folder is made to hold exactly the named snapshot before the game starts. The current
# folder is copied to <VICTORY_SAVE_BACKUPS, default $HOME/victory-save-backups>/before-runs/<stamp>-<name> and verified
# first, the snapshot is staged and verified beside the live folder, and only then swapped in by renames, with the old
# folder put back if that fails (save-snapshot.mjs restore() says how). Exit 1: the save is as it was; exit 3: a
# rollback failed, and the verified backup holds the previous save.
if [ -n "$SAVE" ]; then
  say "restore: $SAVE"
  node "$TOOLS/save-snapshot.mjs" restore "$SAVE" --config "$SAVE_CONFIG" --lock "$LOCK" ${WORLD_RUN_HOLDS_LOCK:+--caller-holds-lock} >>"$LOG" 2>&1
  RESTORE_RC=$?
  case "$RESTORE_RC" in
    0) ;;
    1) say_err "save restore failed; the save is as it was; see the log"; exit 2 ;;
    3) say_err "save restore failed AND its rollback failed: the save may be changed; the old folder and the verified backup the log names both hold the previous save"; exit 2 ;;
    4) say "save restored; its old folder could not be removed (the log names it): remove it before the next restore" ;;
    2) say_err "save restore did not run (exit 2: its arguments); see the log"; exit 2 ;;
    *) say_err "save restore ended unexpectedly (exit $RESTORE_RC): check the save folder against the backup the log names"; exit 2 ;;
  esac
fi

# One pipeline library per build (pipeline-library.sh says why and how). This build's own is put in place now, whatever
# build ran last, and put away when the run ends: below once the game has ended, and by the EXIT trap if the harness is
# stopped or fails first. It prints one line, and refuses to start if the folder cannot be made this build's.
# --cold-pipelines: the build's library is set aside instead, and at the end put back as it was while the cold run's
# own is discarded, so a cold check never changes how warm a later run starts. A cold run killed too hard for the
# trap is put right by the next start, with or without the flag.
PIPE_TOOLS=$TOOLS
. "$HERE/pipeline-library.sh"
PIPE_LOG=$LOG
pipeline_library_enter "$(pipeline_library_cache_dir "$SAVE_CONFIG")" "$(pipeline_library_key "$BIN")" ${COLD_PIPELINES:+cold} || exit 2
# The loading cover's player (--boot-video, below) is ended with the run, however the run ends.
COVER_PID=""
cover_end() {
  [ -n "$COVER_PID" ] && taskkill //PID "$COVER_PID" //T //F >/dev/null 2>&1
  COVER_PID=""
  return 0
}
trap 'cover_end; pipeline_library_leave' EXIT
trap 'exit 130' INT TERM

# --warm: run the title's own pre-compile (`victory warm`) so every shader is in the cache this run reads and its
# compute pipelines are made on this GPU, before the game starts. The loop limit and the wave mode are part of a
# translation, so they are the run's own.
if [ "$WARM" = 1 ]; then
  # The per-program report the translator writes (name, stage, outcome, reason, ms, DXIL bytes), read by
  # `shader-compile-report.mjs --programs`; a stale file from an earlier run is removed first.
  REPORT_TSV="$CAPTURE/translate-report.tsv"
  rm -f "$REPORT_TSV"
  WARM_ENV=(VICTORY_LOG=info "VICTORY_WAVE64_EMULATE=emulate" "VICTORY_TRANSLATE_REPORT=$REPORT_TSV" "VICTORY_CONFIG=$SAVE_CONFIG")
  [ -n "$LOOP_LIMIT" ] && WARM_ENV+=("VICTORY_SHADER_LOOP_LIMIT=$LOOP_LIMIT")
  # The settings set's switches go to the warm-up too: a translation made with other switches is not found by the run
  # (switches given on the command line reach it through the environment, as they reach the game).
  WARM_ENV+=(${SETTINGS_PAIRS[@]+"${SETTINGS_PAIRS[@]}"})
  # The command that translates a folder into the shader cache: `shaders` where the build has it (the base: its
  # `warm` makes pipelines from its book and takes no folder), else `warm` (the two older builds).
  WARM_CMD=warm
  "$BIN" shaders --help >/dev/null 2>&1 && WARM_CMD=shaders
  WARM_T0=$(date +%s)
  PRECOMPILE_JSON=${VICTORY_PRECOMPILE_JSON:-$HOME/.config/victory-harness/stream/precompile.json}
  say "warm-up: warm (see the log)"
  # The stream overlay reads this file: progress at most twice a second while the warm-up runs, then its end
  # state from `final` with the warm-up's exit code. The warm log prints no totals, so the bar counts its
  # per-program and per-pipeline lines against the newest earlier warm log in this folder (precompile-progress.mjs).
  FEED_LOG="$LOG.warm.feed"
  node "$TOOLS/precompile-progress.mjs" follow --log "$LOG.warm" --out "$PRECOMPILE_JSON"     --started-ms "$((WARM_T0 * 1000))" >"$FEED_LOG" 2>&1 &
  PRE_PID=$!
  # Raised, as the game is: other lanes' compilers starve the warm-up at normal priority (852 s against 682 s).
  # While OBS is running it is gentle instead (obs.sh): below normal and half the logical processors, so a recorder
  # running beside it keeps its share; VICTORY_WARM_GENTLE=1 or 0 forces either. The selection shares run-process.ps1's rules and
  # matches only a victory.exe created after this warm-up began whose command line has " warm ", so it can never
  # select a real game.
  . "$HERE/obs.sh"
  warm_mode_decide
  warm_mode_line
  WARM_ARGS=()
  WARM_PS_FLAGS=-Translate
  if [ "$WARM_MODE" = gentle ]; then
    WARM_ARGS=(--jobs "$(warm_jobs)")
    WARM_PS_FLAGS="-Translate -Gentle"
  fi
  ( sleep 4; RUN_WORLD_WARM_STARTED_EPOCH="$WARM_T0" \
      powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/raise-priority.ps1" $WARM_PS_FLAGS >&2 ) > >(public_line) 2>&1 &
  WARM_RAISER=$!
  (cd "$VICTORY_DIR" && env "${WARM_ENV[@]}" "$BIN" "$WARM_CMD" ${WARM_ARGS[@]+"${WARM_ARGS[@]}"} "$(dirname "$GAME_EXE")") >"$LOG.warm" 2>&1
  WARM_RC=$?
  wait "$WARM_RAISER" 2>/dev/null
  kill -0 "$PRE_PID" 2>/dev/null && kill "$PRE_PID" 2>/dev/null
  wait "$PRE_PID" 2>/dev/null
  node "$TOOLS/precompile-progress.mjs" final --log "$LOG.warm" --out "$PRECOMPILE_JSON"     --started-ms "$((WARM_T0 * 1000))" --exit "$WARM_RC" >>"$FEED_LOG" 2>&1
  printf 'harness: feed exit=%s\n' "$?" >>"$LOG"
  WARM_LINE="warm-up: exit=$WARM_RC after $(( $(date +%s) - WARM_T0 )) s"
  printf '%s\n' "$WARM_LINE" >>"$LOG"
  say "$WARM_LINE"
fi

# The load this leg runs under, so a frame time can be read with the machine that produced it in view. One line just
# before the game starts and one when it ends; load-line.mjs samples CPU over five seconds, free and committed
# memory, and the builders running by name. Each helper's exit is recorded like the others'. The check runs before
# any watcher starts, so a refusal leaves nothing behind. With --require-quiet the start sample waits for a quiet
# machine (still holding the lock) instead of refusing at once.
if [ "$REQUIRE_QUIET" = 1 ]; then
  quiet_sample
  LOAD_START_RC=$?
  LOAD_START=$QUIET_LINE
else
  LOAD_START=$(node "$TOOLS/load-line.mjs" start 2>>"$LOG")
  LOAD_START_RC=$?
fi
printf '%s\n' "$LOAD_START" >>"$LOG"
printf 'harness: load start exit=%s\n' "$LOAD_START_RC" >>"$LOG"
# --require-quiet: refuse before the game starts when the start sample is not comparable (exit 3), naming the
# threshold that failed. Off by default. The wait above has already refused a machine that never went quiet.
if [ "$REQUIRE_QUIET" = 1 ]; then
  if [ "$LOAD_START_RC" != 0 ]; then
    exit 3
  fi
  if ! QUIET_REASON=$(node "$TOOLS/load-line.mjs" check "$LOAD_START" 2>>"$LOG"); then
    say_err "refused: ${QUIET_REASON:-the start load is not quiet}"
    exit 3
  fi
fi

# PresentMon starts on its own when the heavy 3D scene begins, so it cannot miss the window.
PM_PID=""
if [ "$PRESENTMON" = 1 ]; then
  node "$TOOLS/presentmon.mjs" watch "$LOG" "$CAPTURE/presentmon.csv" > >(public_line) 2>&1 &
  PM_PID=$!
fi

# The game is raised to above-normal priority so other lanes' compilers cannot starve its pipeline workers
# (raise-priority.ps1 says why, picks this run's process, and prints one line saying what it raised). The line goes to
# the log as well: it names the game's process id, the one presentmon-compare keeps PresentMon's rows to.
RUN_STARTED=$(date +%s)
( sleep 4
  PRIORITY_LINE=$(RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/raise-priority.ps1" 2>&1)
  PRIORITY_LINE=${PRIORITY_LINE//$'\r'/}
  printf '%s\n' "$PRIORITY_LINE" >>"$LOG"
  printf '%s\n' "$PRIORITY_LINE" >&2 ) > >(public_line) 2>&1 &

# The game is muted (0 in Windows' mixer, the maintainer, 2026-11-27) unless told otherwise (--volume N, or --volume off to leave it alone):
# the title is loud. set-volume.ps1 touches only this run's game, each audio session once.
BOOT_VOLUME_PID=""
if [ -n "$BOOT_VOLUME" ]; then
  # While the loading cover shows, the game is heard at --boot-volume; the cover's end stops this setter and starts
  # the usual one (a setter sets each audio session once, so two at once would each win some sessions).
  ( sleep 4; RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/set-volume.ps1" -Percent "$BOOT_VOLUME" >>"$LOG" 2>&1 ) &
  BOOT_VOLUME_PID=$!
elif [ "$VOLUME" != off ]; then
  ( sleep 4; RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED"       powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/set-volume.ps1" -Percent "$VOLUME" >>"$LOG" 2>&1 ) &
fi

# --boot-video: the loading cover. A player (ffplay: full screen, always on top, silent, looping) covers the screen
# before the game starts, so the game's window opens under it and keeps the keyboard. boot-cover.mjs follows this
# run's log and ends the player when the world is on screen, after ten minutes, or with the run. It is a window and
# not a pass in the runtime: while the world loads the title presents about three frames a second, and a video
# drawn in its frames would stutter. The video's own sound is never played.
if [ -n "$BOOT_VIDEO" ]; then
  SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0 ffplay -loglevel quiet -an -loop 0 -fs -alwaysontop -window_title loading \
    "$BOOT_VIDEO" >/dev/null 2>&1 &
  COVER_MSYS=$!
  sleep 1
  COVER_PID=$(cat "/proc/$COVER_MSYS/winpid" 2>/dev/null)
  if [ -z "$COVER_PID" ]; then
    say "loading cover: the player did not start; the run goes on without it"
  else
    say "loading cover: ${BOOT_VIDEO##*[\\/]} until the world is on screen"
    (
      node "$TOOLS/boot-cover.mjs" watch "$LOG" "$COVER_PID" --ceiling-seconds 600 >>"$LOG" 2>&1
      if [ -n "$BOOT_VOLUME_PID" ]; then
        kill "$BOOT_VOLUME_PID" 2>/dev/null
        if [ "$VOLUME" != off ]; then
          RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/set-volume.ps1" -Percent "$VOLUME" >>"$LOG" 2>&1
        fi
      fi
    ) &
  fi
fi

# Every run records its own window at two frames a second, into its capture folder; --no-record turns it off.
# record-screen.mjs selects only this run's game (run-process.ps1 -Strict) and ends only its own ffmpeg child.
REC_PID=""
if [ "$RECORD" = 1 ]; then
  RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" \
    node "$TOOLS/record-screen.mjs" "$CAPTURE/screen.mp4" >>"$LOG" 2>&1 &
  REC_PID=$!
fi

# --heavy-seconds: end the game N seconds after the first frame line that is heavy (draws + dropped >= 1000).
# It appends "harness: ended by --heavy-seconds ..." to the log before the summary below reads it.
HEAVY_PID=""
if [ -n "$HEAVY_SECONDS" ]; then
  RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" \
    node "$TOOLS/heavy-watch.mjs" "$LOG" "$HEAVY_SECONDS" > >(public_line) 2>&1 &
  HEAVY_PID=$!
fi
SETTLE_PID=""
if [ -n "$SETTLED_SECONDS" ]; then
  SETTLE_ARGS=(--after-file "$AFTER")
  [ -n "$SETTLED_TIMEOUT" ] && SETTLE_ARGS+=(--timeout "$SETTLED_TIMEOUT")
  [ -n "$WORLD_TIMEOUT" ] && SETTLE_ARGS+=(--world-timeout "$WORLD_TIMEOUT")
  [ -n "$SETTLED_FPS" ] && SETTLE_ARGS+=(--floor-fps "$SETTLED_FPS")
  [ -n "$SETTLED_WAITING" ] && SETTLE_ARGS+=(--waiting "$SETTLED_WAITING")
  [ -n "$SETTLED_AFTER" ] && SETTLE_ARGS+=(--settle-after-s "$SETTLED_AFTER")
  RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" \
    node "$TOOLS/settled-watch.mjs" "$LOG" "$SETTLED_SECONDS" "${SETTLE_ARGS[@]}" > >(public_line) 2>&1 &
  SETTLE_PID=$!
fi

# The frozen-run watchdog: it arms at the settle (the settled watcher's own rule, which an open, untimed session
# reaches without the watcher above running). Mode 1 ends the run when the frame counter stands still; warn mode
# reports the stall and the resume and never ends it. Its counter is the stats file for a --live run, and the newest
# frame line in the log otherwise.
WATCHDOG_PID=""
if [ "$WATCHDOG" != 0 ]; then
  WD_ARGS=(--stall-seconds "$WATCHDOG_STALL_S")
  [ "$WATCHDOG" = warn ] && WD_ARGS+=(--warn)
  [ "$LIVE" = 1 ] && WD_ARGS+=(--stats "$LIVE_STATS")
  [ -n "$SETTLED_FPS" ] && WD_ARGS+=(--settled-fps "$SETTLED_FPS")
  [ -n "$SETTLED_WAITING" ] && WD_ARGS+=(--settled-waiting "$SETTLED_WAITING")
  [ -n "$SETTLED_AFTER" ] && WD_ARGS+=(--settled-after-s "$SETTLED_AFTER")
  RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" \
    node "$TOOLS/watchdog.mjs" "$LOG" "${WD_ARGS[@]}" > >(public_line) 2>&1 &
  WATCHDOG_PID=$!
fi

# The stats recorder (a --live run): the runtime rewrites stats.json about once a second, so every new object is
# appended to stats.jsonl beside it, for presentmon-compare's window. It ends by itself after the load-end line.
STATS_REC_PID=""
if [ "$LIVE" = 1 ]; then
  node "$TOOLS/stats-recorder.mjs" "$LIVE_STATS" "$LIVE_HISTORY" --log "$LOG" >>"$LOG" 2>&1 &
  STATS_REC_PID=$!
fi

# The free-space guard: every poll, and under the floor it appends "harness: ended: disk below N GB" and ends this
# run's game. It exits by itself once the game is gone.
DISK_PID=""
if [ "$STOP_FREE_GB" != off ]; then
  RUN_WORLD_GAME_EXE="$GAME_EXE" RUN_WORLD_STARTED_EPOCH="$RUN_STARTED" \
    node "$TOOLS/disk-guard.mjs" watch --log "$LOG" --path "$LOG_DIR" --below "$STOP_FREE_GB" > >(public_line) 2>&1 &
  DISK_PID=$!
fi

# The terminal's heartbeat, from the game's own frame-rate line, while the game runs.
heartbeat_loop &
HEARTBEAT_PID=$!

# A capture flag runs the game under its tool, with the same arguments and environment, written to files so paths
# with spaces survive. Without one, the game starts directly.
if [ "$GPU_TRACE" = 1 ] || [ "$RENDERDOC" = 1 ]; then
  printf '%s\n' run "$GAME_EXE" --execute --console "$CONSOLE" > "$CAPTURE/game-args.txt"
  printf '%s\n' "${ENV_ARGS[@]}" > "$CAPTURE/env.txt"
fi
# The game's output goes to the log only; the terminal sees the heartbeat and the harness's own filtered lines.
if [ "$GPU_TRACE" = 1 ]; then
  node "$TOOLS/gpu-trace.mjs" run --bin "$BIN" --dir "$VICTORY_DIR" \
    --game-args-file "$CAPTURE/game-args.txt" --env-file "$CAPTURE/env.txt" \
    --after-ms "$TRACE_AFTER_MS" --frames "$TRACE_FRAMES" --out "$CAPTURE/gputrace" \
    >>"$LOG" 2>&1
  RC=$?  # ngfx.exe's own exit code, which is the traced game's
elif [ "$RENDERDOC" = 1 ]; then
  node "$TOOLS/renderdoc.mjs" run --bin "$BIN" --dir "$VICTORY_DIR" \
    --game-args-file "$CAPTURE/game-args.txt" --out "$CAPTURE/rdoc" --after-ms "$CAPTURE_AFTER_MS" \
    >>"$LOG" 2>&1
  RC=$?
else
  (cd "$VICTORY_DIR" && env "${ENV_ARGS[@]}" "$BIN" run "$GAME_EXE" --execute --console "$CONSOLE") \
    >>"$LOG" 2>&1
  RC=$?  # victory.exe's own exit code
fi

cover_end
# The game has ended: its library is stored under its build's name before the slow part below (the EXIT trap is the
# same call, for a run that never gets here). The end load is taken first, while the leg's own processes are gone.
node "$TOOLS/load-line.mjs" end >>"$LOG" 2>&1
printf 'harness: load end exit=%s\n' "$?" >>"$LOG"
pipeline_library_leave

# The watchers end the game themselves, so give them a moment to record that in the log before the summary.
if [ -n "$DISK_PID" ]; then
  for _ in $(seq 1 50); do
    kill -0 "$DISK_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$DISK_PID" 2>/dev/null && kill "$DISK_PID" 2>/dev/null
  wait "$DISK_PID" 2>/dev/null
fi
if [ -n "$HEARTBEAT_PID" ]; then
  kill -0 "$HEARTBEAT_PID" 2>/dev/null && kill "$HEARTBEAT_PID" 2>/dev/null
  wait "$HEARTBEAT_PID" 2>/dev/null
fi
if [ -n "$HEAVY_PID" ]; then
  for _ in $(seq 1 50); do
    kill -0 "$HEAVY_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$HEAVY_PID" 2>/dev/null && kill "$HEAVY_PID" 2>/dev/null
  wait "$HEAVY_PID" 2>/dev/null
fi
if [ -n "$SETTLE_PID" ]; then
  for _ in $(seq 1 50); do
    kill -0 "$SETTLE_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$SETTLE_PID" 2>/dev/null && kill "$SETTLE_PID" 2>/dev/null
  wait "$SETTLE_PID" 2>/dev/null
fi
if [ -n "$WATCHDOG_PID" ]; then
  for _ in $(seq 1 50); do
    kill -0 "$WATCHDOG_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$WATCHDOG_PID" 2>/dev/null && kill "$WATCHDOG_PID" 2>/dev/null
  wait "$WATCHDOG_PID" 2>/dev/null
fi
if [ -n "$STATS_REC_PID" ]; then
  for _ in $(seq 1 50); do
    kill -0 "$STATS_REC_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$STATS_REC_PID" 2>/dev/null && kill "$STATS_REC_PID" 2>/dev/null
  wait "$STATS_REC_PID" 2>/dev/null
fi

if [ -n "$PM_PID" ]; then
  kill -0 "$PM_PID" 2>/dev/null && kill "$PM_PID" 2>/dev/null
  wait "$PM_PID" 2>/dev/null
fi

# The recorder ends when the game process does; give it its clean stop (and its own last line) before the summary.
if [ -n "$REC_PID" ]; then
  for _ in $(seq 1 150); do
    kill -0 "$REC_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$REC_PID" 2>/dev/null && kill "$REC_PID" 2>/dev/null
  wait "$REC_PID" 2>/dev/null
fi
# The sheets are what an agent opens to review a run; a failure here never fails the run.
if [ "$RECORD" = 1 ] && [ -f "$CAPTURE/screen.mp4" ]; then
  node "$TOOLS/contact-sheets.mjs" "$CAPTURE/screen.mp4" >>"$LOG" 2>&1 || say_err "contact sheets: failed"
fi

# --save-as: the save folder as the run left it becomes a new snapshot; never overwrites an existing name.
if [ -n "$SAVE_AS" ]; then
  if node "$TOOLS/save-snapshot.mjs" save-as "$SAVE_AS" --config "$SAVE_CONFIG" >>"$LOG" 2>&1; then
    say "saved: $SAVE_AS"
  else
    say_err "save-as failed; see the log"
  fi
fi

# The driver's own account of a GPU fault is in the Windows System log; append it before the summary reads the log.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TOOLS/driver-events.ps1" "$RUN_STARTED" "$(date +%s)" >>"$LOG" 2>>"$LOG"
printf 'harness: driver-events exit=%s\n' "$?" >>"$LOG"

# The game's own exit code, before the parser reads the log: a non-zero code with no failure line is a run
# something else stopped, not a log that simply ends.
echo "harness: victory exit=$RC" >>"$LOG"

say ""
# The summary is written to a temporary file and renamed, so a crash cannot leave half of one; its exit is recorded.
SUMMARY_RC=1
if node "$TOOLS/summarize-run.mjs" "$LOG" >"$LOG.md.tmp" 2>>"$LOG"; then
  if mv -f "$LOG.md.tmp" "$LOG.md"; then SUMMARY_RC=0; fi
fi
rm -f "$LOG.md.tmp"
printf 'harness: summary exit=%s\n' "$SUMMARY_RC" >>"$LOG"
# The run's result file in its capture folder, for the stream's best-run card (scripts/run-result.mjs). It is made
# from the summary and the log's head; a failure here costs the run nothing.
RESULT_TOOL=$TOOLS/../scripts/run-result.mjs
if [ "$SUMMARY_RC" = 0 ] && [ -f "$RESULT_TOOL" ] && [ -n "${CAPTURE:-}" ] && [ -d "$CAPTURE" ]; then
  RESULT_LINE=$(RUN_RESULT_LOGS=$(dirname "$LOG") node "$RESULT_TOOL" "$CAPTURE" 2>&1 | head -1)
  printf 'harness: result file: %s\n' "$RESULT_LINE" >>"$LOG"
fi
[ "$SUMMARY_RC" = 0 ] && cat "$LOG.md" | public_line
say "victory exit=$RC"
[ "$GPU_TRACE" = 1 ] && say "profiled: this code is Nsight's, not the title's"
[ "$RENDERDOC" = 1 ] && say "captured: this code is RenderDoc's, not the title's"
exit "$RC"
}
main "$@"
exit
