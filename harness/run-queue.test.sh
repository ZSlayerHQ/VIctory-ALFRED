#!/usr/bin/env bash
# Tests run-queue.sh with stand-ins for world-run.sh. No game, GPU, cargo or run lock is used.
# Three required cases: a stub with a chosen exit code, a stub that outlives --max-seconds,
# and a queue whose first line fails while the second passes. A fourth checks lane resolution.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
QUEUE="$HERE/run-queue.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/run-queue-test.XXXXXX") || exit 1
# The queue ends "the game" when a run outlasts --max-seconds. A test must never match a real one.
export GAME_EXE="C:/run-queue-test/no-such-game.exe"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok() { PASS=$((PASS + 1)); echo "ok - $1"; }
bad() { FAIL=$((FAIL + 1)); echo "not ok - $1"; }

make_snapshot() {
  mkdir -p "$1"
  : >"$1/victory.exe"
  cat >"$1/snapshot.txt" <<EOF
commit: 0000000000000000000000000000000000000000
branch: test
profile: diag
features: diagnostics
date: 2026-11-25T00:00:00Z
EOF
}

cat >"$TMP/stub-exit.sh" <<'EOF'
#!/usr/bin/env bash
tag=run
while [ $# -gt 0 ]; do
  case "$1" in --tag) tag=$2; shift 2 ;; *) shift ;; esac
done
mkdir -p "$STUB_LOG_DIR"
log="$STUB_LOG_DIR/$tag.log"
printf '2026-11-25T01:00:00Z INFO victory: started\n2026-11-25T01:00:04Z INFO victory: run complete\n' >"$log"
echo "log:     $log"
echo "stub exit ${STUB_EXIT:-0}"
exit "${STUB_EXIT:-0}"
EOF

cat >"$TMP/stub-by-tag.sh" <<'EOF'
#!/usr/bin/env bash
tag=run
while [ $# -gt 0 ]; do
  case "$1" in --tag) tag=$2; shift 2 ;; *) shift ;; esac
done
mkdir -p "$STUB_LOG_DIR"
log="$STUB_LOG_DIR/$tag.log"
printf '2026-11-25T01:00:00Z INFO victory: started\n2026-11-25T01:00:04Z INFO victory: run complete\n' >"$log"
echo "log:     $log"
case "$tag" in
  fail*) echo "stub failing $tag"; exit 1 ;;
  *) echo "stub passing $tag"; exit 0 ;;
esac
EOF

cat >"$TMP/stub-sleep.sh" <<'EOF'
#!/usr/bin/env bash
tag=slow
while [ $# -gt 0 ]; do
  case "$1" in --tag) tag=$2; shift 2 ;; *) shift ;; esac
done
echo "log:     $STUB_LOG_DIR/$tag.log"
echo "stub sleeping for $tag"
echo $$ >"$STUB_SLEEP_PID"
exec sleep 30
EOF

cat >"$TMP/stub-env.sh" <<'EOF'
#!/usr/bin/env bash
tag=run
while [ $# -gt 0 ]; do
  case "$1" in --tag) tag=$2; shift 2 ;; *) shift ;; esac
done
mkdir -p "$STUB_LOG_DIR"
log="$STUB_LOG_DIR/$tag.log"
printf '2026-11-25T01:00:00Z INFO victory: started\n' >"$log"
echo "log:     $log"
printf 'VICTORY_DIR=%s\nVICTORY_BIN=%s\n' "${VICTORY_DIR:-}" "${VICTORY_BIN-<unset>}" >"$STUB_ENV_OUT"
exit 0
EOF

# 1. A stub that prints a log: line and exits with the chosen code.
make_snapshot "$TMP/snap-a"
cat >"$TMP/q1" <<EOF
# one experiment with a chosen failure code
one $TMP/snap-a --heavy-seconds 5
EOF
STUB_LOG_DIR="$TMP/logs1" STUB_EXIT=7 RUN_QUEUE_WRAPPER="$TMP/stub-exit.sh" \
  bash "$QUEUE" "$TMP/q1" >"$TMP/out1" 2>&1
rc=$?
[ "$rc" = 0 ] && ok "chosen exit: queue finished" || bad "chosen exit: queue exit $rc"
grep -q '| one |' "$TMP/out1" && ok "chosen exit: row present" || bad "chosen exit: no row"
grep -q '| 7 |' "$TMP/out1" && ok "chosen exit: recorded exit 7" || bad "chosen exit: exit 7 not recorded"
grep -q 'log ends after 4 s' "$TMP/out1" && ok "chosen exit: summary read the log" || bad "chosen exit: no summary"
[ -f "$TMP/q1.result.md" ] && ok "chosen exit: result file written" || bad "chosen exit: no result file"
grep -q '| one |' "$TMP/q1.result.md" 2>/dev/null && ok "chosen exit: result file has the row" || bad "chosen exit: result file has no row"

# 2. A stub that sleeps past --max-seconds; the queue must end it and go on.
cat >"$TMP/q2" <<EOF
slow $TMP/snap-a --heavy-seconds 1
EOF
start=$(date +%s)
STUB_LOG_DIR="$TMP/logs2" STUB_SLEEP_PID="$TMP/sleep.pid" RUN_QUEUE_WRAPPER="$TMP/stub-sleep.sh" \
  RUN_QUEUE_KILL_GRACE=1 bash "$QUEUE" --max-seconds 2 "$TMP/q2" >"$TMP/out2" 2>&1
rc=$?
elapsed=$(( $(date +%s) - start ))
[ "$rc" = 0 ] && ok "sleeping stub: queue finished" || bad "sleeping stub: queue exit $rc"
[ "$elapsed" -le 20 ] && ok "sleeping stub: stopped after ${elapsed}s" || bad "sleeping stub: took ${elapsed}s"
sleep_pid=$(cat "$TMP/sleep.pid" 2>/dev/null)
if [ -n "$sleep_pid" ] && kill -0 "$sleep_pid" 2>/dev/null; then
  bad "sleeping stub: process $sleep_pid survived"
else
  ok "sleeping stub: process tree killed"
fi
grep -q '| slow |' "$TMP/out2" && ok "sleeping stub: row present" || bad "sleeping stub: no row"
grep -q '| 124 |' "$TMP/out2" && ok "sleeping stub: exit 124" || bad "sleeping stub: exit 124 missing"
grep -q 'timed out after 2 s' "$TMP/out2" && ok "sleeping stub: ended as timed out" || bad "sleeping stub: no timeout text"

# 3. A failing first line, then a passing second: both rows, in order.
make_snapshot "$TMP/snap-b"
cat >"$TMP/q3" <<EOF
fail $TMP/snap-a
pass $TMP/snap-b --heavy-seconds 5
EOF
STUB_LOG_DIR="$TMP/logs3" RUN_QUEUE_WRAPPER="$TMP/stub-by-tag.sh" \
  bash "$QUEUE" "$TMP/q3" >"$TMP/out3" 2>&1
rc=$?
[ "$rc" = 0 ] && ok "fail then pass: queue finished" || bad "fail then pass: queue exit $rc"
fail_line=$(grep -n '| fail |' "$TMP/out3" | cut -d: -f1 | head -n 1)
pass_line=$(grep -n '| pass |' "$TMP/out3" | cut -d: -f1 | head -n 1)
[ -n "$fail_line" ] && [ -n "$pass_line" ] && [ "$fail_line" -lt "$pass_line" ] \
  && ok "fail then pass: both rows, in order" || bad "fail then pass: rows missing or out of order"
awk -F'|' '/\| fail \|/ {gsub(/ /,"",$4); if ($4==1) found=1} END {exit !found}' "$TMP/out3" \
  && ok "fail then pass: failing exit recorded" || bad "fail then pass: failing exit not 1"
awk -F'|' '/\| pass \|/ {gsub(/ /,"",$4); if ($4==0) found=1} END {exit !found}' "$TMP/out3" \
  && ok "fail then pass: passing exit recorded" || bad "fail then pass: passing exit not 0"
grep -q '| fail |' "$TMP/q3.result.md" 2>/dev/null && grep -q '| pass |' "$TMP/q3.result.md" 2>/dev/null \
  && ok "fail then pass: result file has both" || bad "fail then pass: result file incomplete"

# 4. A lane gives VICTORY_DIR and clears an inherited VICTORY_BIN.
mkdir -p "$TMP/lane/target/release"
cat >"$TMP/q4" <<EOF
lane $TMP/lane --profile release
EOF
STUB_LOG_DIR="$TMP/logs4" STUB_ENV_OUT="$TMP/env4" VICTORY_BIN=/should/be/cleared \
  RUN_QUEUE_WRAPPER="$TMP/stub-env.sh" bash "$QUEUE" "$TMP/q4" >"$TMP/out4" 2>&1
rc=$?
LANE_WIN=$(cygpath -m "$TMP/lane" 2>/dev/null || printf '%s' "$TMP/lane")
[ "$rc" = 0 ] && ok "lane: queue finished" || bad "lane: queue exit $rc"
grep -q "^VICTORY_DIR=$LANE_WIN\$" "$TMP/env4" 2>/dev/null \
  && ok "lane: VICTORY_DIR set to the lane" || bad "lane: VICTORY_DIR wrong"
grep -q '^VICTORY_BIN=<unset>$' "$TMP/env4" 2>/dev/null \
  && ok "lane: inherited VICTORY_BIN cleared" || bad "lane: inherited VICTORY_BIN leaked"
grep -q 'target/release/victory.exe' "$TMP/out4" \
  && ok "lane: binary shows the chosen profile" || bad "lane: binary path wrong"

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
