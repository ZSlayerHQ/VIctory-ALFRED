#!/usr/bin/env bash
# Drive an open live session: live-drive.mjs does the work.
# usage: live-drive.sh run <plan.json> --log <run log> [--file <control file>] [--stats <stats file>]
#        live-drive.sh report --log <run log> [--plan <n>] [--json]
# The control file comes from --file, else VICTORY_LIVE; the stats file from --stats, else VICTORY_LIVE_STATS, else
# stats.json beside the control file. Every state is confirmed from the stats file before an arm starts. The plan's
# fields, the lock, the sidecars and the verdict rules are in live-drive.mjs's header.
HERE=$(cd "$(dirname "$0")" && pwd)
exec node "$HERE/live-drive.mjs" "$@"
