# Sourced by run-world.sh: one Direct3D 12 pipeline library per build.
#
# The runtime keeps its library (host-pipelines.bin, and host-pipelines.next and early-pipelines.txt beside it) in
# <cache_dir>/pipelines. <cache_dir> is VICTORY_CACHE_DIR, else cache_dir in the config file VICTORY_CONFIG names, else
# %LOCALAPPDATA%/victory/cache (victory-core config.rs); the `pipelines` name is fixed in victory-hle d3d12x.rs. No
# variable moves that folder alone (VICTORY_CACHE_DIR would move the shader cache and images too), so the harness
# swaps the folder's files instead, by rename on the one volume.
#
# Layout under the cache folder: pipelines/ holds the files of the build named in pipelines/library-build.txt (no
# marker: nothing is in place); pipelines-by-build/<build>/ holds each other build's files, and its .last-used file
# orders the six kept. A folder that holds another build's files, or unmarked ones, is put aside under that name
# (unmarked-<time>) before this build's own are put in place, so a run can only ever write into its own build's
# library, even after a harness that was killed outright.
#
# A cold start (run-world.sh --cold-pipelines; pipeline_library_enter's third argument `cold`) moves the build's own
# library, stored or in place, to pipelines-cold-aside/<build>/ and marks the empty live folder `cold <build>` (no
# build's name has a space). When the run ends its own files are discarded and the library goes back to the store as
# it was. A start that finds either left behind by a killed cold run discards and puts back before anything else.
#
# Never print the cache folder's path (its output can appear in a shared recording): output names builds and sizes only, and every
# command that could name a path has its error output dropped.

# The folder this file lives in. run-world.sh sources a copy from the run's capture folder and points PIPE_TOOLS at
# the harness checkout, where the helper module below stays.
PIPE_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPE_TOOLS=${PIPE_TOOLS:-$PIPE_HERE}
PIPE_MARKER=library-build.txt
PIPE_KEEP=${PIPELINE_LIBRARY_KEEP:-6}
PIPE_RETRIES=${PIPELINE_LIBRARY_RETRIES:-10}
PIPE_ASIDE_NAME=pipelines-cold-aside
PIPE_ON=
PIPE_COLD=

# Prints a line to the terminal (through run-world.sh's public filter when it is loaded) and, when PIPE_LOG names the
# run's log, to the log as well.
_pipe_say() {
  if declare -F public_line >/dev/null 2>&1; then
    printf '%s\n' "$*" | public_line
  else
    echo "$*"
  fi
  [ -n "${PIPE_LOG:-}" ] && echo "harness: $*" >>"$PIPE_LOG"
  return 0
}

# The library files in folder $1, one name a line: every regular file but the marker.
_pipe_files() {
  local f
  for f in "$1"/*; do
    [ -f "$f" ] || continue
    [ "${f##*/}" = "$PIPE_MARKER" ] && continue
    printf '%s\n' "${f##*/}"
  done
}

# The size of folder $1's library files in whole MB.
_pipe_mb() {
  local name size total=0
  while IFS= read -r name; do
    size=$(stat -c %s "$1/$name" 2>/dev/null) || continue
    total=$((total + size))
  done < <(_pipe_files "$1")
  echo $((total / 1048576))
}

# Moves folder $1's library files into folder $2 (made if absent), each retried while something still holds it (a
# virus scanner just after the game ends). Fails if any file stays.
_pipe_move_all() {
  local name try rc=0
  mkdir -p "$2" 2>/dev/null || return 1
  while IFS= read -r name; do
    try=1
    until mv -f "$1/$name" "$2/$name" 2>/dev/null; do
      if [ "$try" -ge "$PIPE_RETRIES" ]; then
        rc=1
        break
      fi
      try=$((try + 1))
      sleep 1
    done
  done < <(_pipe_files "$1")
  return $rc
}

# Deletes folder $1's library files (a cold run's own), each retried like a move. Fails if any file stays.
_pipe_discard_all() {
  local name try rc=0
  while IFS= read -r name; do
    try=1
    until rm -f "$1/$name" 2>/dev/null && [ ! -e "$1/$name" ]; do
      if [ "$try" -ge "$PIPE_RETRIES" ]; then
        rc=1
        break
      fi
      try=$((try + 1))
      sleep 1
    done
  done < <(_pipe_files "$1")
  return $rc
}

# Moves store-style folder $1 (library files and .last-used) to $2 as it is: one rename when $2 is absent, else file
# by file into it. Nothing to do when $1 is absent. Fails if anything stays in $1.
_pipe_move_folder() {
  local try=1
  [ -d "$1" ] || return 0
  if [ ! -e "$2" ]; then
    until mv "$1" "$2" 2>/dev/null; do
      [ "$try" -ge "$PIPE_RETRIES" ] && break
      try=$((try + 1))
      sleep 1
    done
    [ -d "$1" ] || return 0
  fi
  _pipe_move_all "$1" "$2" || return 1
  if [ -e "$1/.last-used" ]; then
    mv -f "$1/.last-used" "$2/.last-used" 2>/dev/null || return 1
  fi
  rmdir "$1" 2>/dev/null
}

# Puts build $1's library back from the cold aside folder into the store, and removes that folder when it is empty.
_pipe_put_back() {
  _pipe_move_folder "$PIPE_COLDROOT/$1" "$PIPE_ROOT/$1" || return 1
  rmdir "$PIPE_COLDROOT" 2>/dev/null
  return 0
}

# First at every start: a cold run killed before its put-back left its own files in place (marked `cold <build>`)
# and the build's library aside. The first are discarded, the second put back, and one line says so. Fails (the run
# must not start) if either cannot be done: the run would otherwise write into, or save over, the wrong library.
_pipe_recover() {
  local held d name mb said=
  held=$(cat "$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null)
  case $held in
    "cold "*)
      if ! _pipe_discard_all "$PIPE_LIVE"; then
        _pipe_say "pipeline library: an interrupted cold run's own files could not be discarded; run refused"
        return 1
      fi
      rm -f "$PIPE_LIVE/$PIPE_MARKER"
      ;;
    *) held= ;;
  esac
  for d in "$PIPE_COLDROOT"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    mb=$(_pipe_mb "$d")
    if ! _pipe_put_back "$name"; then
      _pipe_say "pipeline library: $name could not be put back from an interrupted cold run; run refused"
      return 1
    fi
    _pipe_say "pipeline library: $name put back ($mb MB) after a cold run that was interrupted"
    said=1
  done
  if [ -n "$held" ] && [ -z "$said" ]; then
    _pipe_say "pipeline library: an interrupted cold run's own files (${held#cold }) discarded"
  fi
  return 0
}

# The live folder's files go into the store under name $1; the marker is removed only once they are all there.
_pipe_put_away() {
  _pipe_move_all "$PIPE_LIVE" "$PIPE_ROOT/$1" || return 1
  : >"$PIPE_ROOT/$1/.last-used" 2>/dev/null
  rm -f "$PIPE_LIVE/$PIPE_MARKER"
}

# The names in the store, most recently used first.
_pipe_by_use() {
  local d
  for d in "$PIPE_ROOT"/*/; do
    [ -d "$d" ] || continue
    printf '%s %s\n' "$(stat -c %Y "${d}.last-used" 2>/dev/null || stat -c %Y "$d" 2>/dev/null || echo 0)" "$(basename "$d")"
  done | sort -rn | cut -d' ' -f2-
}

# Deletes the store's folders beyond the PIPE_KEEP most recently used. Only folders this file made, by name.
_pipe_prune() {
  local name n=0
  while IFS= read -r name; do
    n=$((n + 1))
    [ "$n" -le "$PIPE_KEEP" ] && continue
    case $name in '' | . | .. | */*) continue ;; esac
    [ "$name" = "$PIPE_KEY" ] && continue
    rm -rf "${PIPE_ROOT:?}/$name" 2>/dev/null && _pipe_say "pipeline library: $name dropped (only the $PIPE_KEEP most recent builds are kept)"
  done < <(_pipe_by_use)
  return 0
}

# pipeline_library_key <victory.exe>: the build's name. A snapshot (<builds root>/<name>/victory.exe) is its
# folder's name; a lane checkout's <lane>/target/<profile>/victory.exe is lane-<lane>-<profile>. A WARP run adds
# -warp: the host rejects a library made for another adapter, and the next save would replace it.
pipeline_library_key() {
  local dir parent key
  dir=$(dirname "$1")
  parent=$(dirname "$dir")
  if [ "$(basename "$parent")" = target ]; then
    key="lane-$(basename "$(dirname "$parent")")-$(basename "$dir")"
  else
    key=$(basename "$dir")
  fi
  [ "${VICTORY_D3D12_WARP:-}" = 1 ] && key="$key-warp"
  key=$(printf '%s' "$key" | tr -c 'A-Za-z0-9._-' '_')
  case $key in '' | .*) key="build$key" ;; esac
  printf '%s' "$key"
}

# pipeline_library_cache_dir <config file>: the cache folder the runtime will use, or nothing when it cannot be told.
pipeline_library_cache_dir() {
  local dir=${VICTORY_CACHE_DIR:-}
  if [ -z "$dir" ]; then
    # The module goes second: disk-usage.mjs runs its own main when it is process.argv[1].
    dir=$(node --input-type=module -e '
      import { pathToFileURL } from "node:url";
      const m = await import(pathToFileURL(process.argv[2]).href);
      try { process.stdout.write(m.configDirs(process.argv[1]).cache || ""); } catch {}
    ' "$1" "$PIPE_TOOLS/disk-usage.mjs" 2>/dev/null)
  fi
  if [ -z "$dir" ] && [ -n "${LOCALAPPDATA:-}" ]; then
    dir="$LOCALAPPDATA/victory/cache"
  fi
  case $dir in [A-Za-z]:[/\\]* | /*) printf '%s' "$dir" ;; esac
}

# A cold start, after any other build's files are set aside: the build's own library (stored, or in place after a
# harness killed outright, which wins as it would for a warm start) moves to the aside folder, and the live folder is
# marked as the cold run's. $1 is the marker found, $2 the note on what was set aside. Prints one line.
_pipe_enter_cold() {
  local held=$1 aside=$2 to=$PIPE_COLDROOT/$PIPE_KEY mb
  if ! mkdir -p "$PIPE_COLDROOT" 2>/dev/null ||
    ! _pipe_move_folder "$PIPE_ROOT/$PIPE_KEY" "$to" ||
    { [ "$held" = "$PIPE_KEY" ] && ! _pipe_move_all "$PIPE_LIVE" "$to"; } ||
    ! printf '%s\n' "cold $PIPE_KEY" >"$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null; then
    _pipe_put_back "$PIPE_KEY"
    _pipe_say "pipeline library: could not set $PIPE_KEY aside for a cold start; run refused"
    return 1
  fi
  PIPE_ON=1
  if [ -d "$to" ]; then
    mb=$(_pipe_mb "$to")
    _pipe_say "pipeline library: $PIPE_KEY cold start: its library ($mb MB) set aside until the run ends$aside"
  else
    rmdir "$PIPE_COLDROOT" 2>/dev/null
    _pipe_say "pipeline library: $PIPE_KEY cold start: none stored$aside"
  fi
  return 0
}

# pipeline_library_enter <cache folder> <build> [cold]: puts the build's library in place, setting aside whatever else
# is there, and prints one line; with `cold`, the build's own library is set aside too (_pipe_enter_cold). First it
# puts back what a killed cold run left (_pipe_recover). Returns 1 (the run must not start) if another build's files
# cannot be set aside or this build's cannot be put in place: a run would then write into the wrong library. A cold
# start with no cache folder is refused too: it could not be told cold.
pipeline_library_enter() {
  PIPE_ON=
  PIPE_KEY=$2
  PIPE_COLD=
  [ "${3:-}" = cold ] && PIPE_COLD=1
  PIPE_LIVE=$1/pipelines
  PIPE_ROOT=$1/pipelines-by-build
  PIPE_COLDROOT=$1/$PIPE_ASIDE_NAME
  if [ -z "$1" ] || ! mkdir -p "$PIPE_LIVE" "$PIPE_ROOT" 2>/dev/null; then
    if [ -n "$PIPE_COLD" ]; then
      _pipe_say "pipeline library: cache folder not found, so no cold start can be made; run refused"
      return 1
    fi
    _pipe_say "pipeline library: cache folder not found, left alone"
    return 0
  fi
  _pipe_recover || return 1
  local held aside="" mb
  held=$(cat "$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null)
  if [ -n "$(_pipe_files "$PIPE_LIVE")" ]; then
    if [ "$held" = "$PIPE_KEY" ]; then
      if [ -n "$PIPE_COLD" ]; then
        _pipe_enter_cold "$held" ""
        return
      fi
      PIPE_ON=1
      _pipe_say "pipeline library: $PIPE_KEY already in place ($(_pipe_mb "$PIPE_LIVE") MB)"
      return 0
    fi
    [ -n "$held" ] || held=unmarked-$(date +%Y%m%d-%H%M%S)
    mb=$(_pipe_mb "$PIPE_LIVE")
    if ! _pipe_put_away "$held"; then
      _pipe_say "pipeline library: could not set $held aside; run refused"
      return 1
    fi
    aside=", $held set aside ($mb MB)"
  fi
  if [ -n "$PIPE_COLD" ]; then
    _pipe_enter_cold "" "$aside"
    return
  fi
  if ! printf '%s\n' "$PIPE_KEY" >"$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null; then
    _pipe_say "pipeline library: could not mark the folder as $PIPE_KEY's; run refused"
    return 1
  fi
  PIPE_ON=1
  local stored=$PIPE_ROOT/$PIPE_KEY
  if [ -n "$(_pipe_files "$stored")" ]; then
    mb=$(_pipe_mb "$stored")
    if ! _pipe_move_all "$stored" "$PIPE_LIVE"; then
      _pipe_say "pipeline library: could not put $PIPE_KEY in place; run refused"
      return 1
    fi
    _pipe_say "pipeline library: $PIPE_KEY restored ($mb MB)$aside"
  else
    _pipe_say "pipeline library: $PIPE_KEY none stored, starting empty$aside"
  fi
  rm -f "$stored/.last-used"
  rmdir "$stored" 2>/dev/null
  return 0
}

# pipeline_library_leave: stores the library the run wrote under its build's name and prunes the store. Safe to call
# twice (run-world.sh calls it when the game ends and again from its EXIT trap). Silent unless something goes wrong.
# After a cold start it stores nothing: _pipe_leave_cold.
pipeline_library_leave() {
  [ -n "${PIPE_ON:-}" ] || return 0
  PIPE_ON=
  if [ -n "${PIPE_COLD:-}" ]; then
    _pipe_leave_cold
    return
  fi
  local held
  held=$(cat "$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null)
  [ "$held" = "$PIPE_KEY" ] || return 0
  if [ -z "$(_pipe_files "$PIPE_LIVE")" ]; then
    rm -f "$PIPE_LIVE/$PIPE_MARKER"
    return 0
  fi
  if ! _pipe_put_away "$PIPE_KEY"; then
    _pipe_say "pipeline library: $PIPE_KEY could not be stored; it stays in place and the next run sets it aside"
    return 1
  fi
  _pipe_prune
}

# The end of a cold run: the build's library goes back to the store as it was, and the files the cold run made are
# discarded, not stored. Why: a cold run checks a fault of the cold start, and storing what it made would replace the
# build's full library with a partial one; a cold check must not change how warm any later run starts. Says one line.
# What cannot be done here is done at the next start (_pipe_recover).
_pipe_leave_cold() {
  local rc=0 back="it had none to put back"
  [ -d "$PIPE_COLDROOT/$PIPE_KEY" ] && back="$PIPE_KEY's library put back ($(_pipe_mb "$PIPE_COLDROOT/$PIPE_KEY") MB)"
  if ! _pipe_put_back "$PIPE_KEY"; then
    _pipe_say "pipeline library: $PIPE_KEY could not be put back after the cold run; the next run puts it back"
    rc=1
  fi
  if [ "$(cat "$PIPE_LIVE/$PIPE_MARKER" 2>/dev/null)" = "cold $PIPE_KEY" ]; then
    if _pipe_discard_all "$PIPE_LIVE"; then
      rm -f "$PIPE_LIVE/$PIPE_MARKER"
    else
      _pipe_say "pipeline library: the cold run's own files could not be discarded; the next run discards them"
      rc=1
    fi
  fi
  [ "$rc" = 0 ] && _pipe_say "pipeline library: the cold run's own files discarded, $back"
  return $rc
}
