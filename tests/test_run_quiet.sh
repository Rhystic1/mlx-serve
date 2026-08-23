#!/bin/bash
# `mlx-serve run` (TTY REPL mode) must not print `[discovery] skip …` lines.
#
# Regression: the REPL log-quieting (info → warn) sat AFTER the models-root
# discovery scan in main(), so every unsupported dir under ~/.mlx-serve/models
# (LTX, a ViT classifier, partial downloads, …) printed an info-level
# skip line into the chat REPL greeting. The quieting must take effect BEFORE
# discovery runs; an explicit --log-level keeps the diagnostics reachable.
#
# Hermetic: fake $HOME with one unsupported model dir; no weights, no load —
# `run` exits at config parse of a nonexistent model dir, after discovery has
# already scanned (and logged, or not) the models root. Needs a pty (`script`)
# because REPL mode only engages when stdin is a TTY.
set -u

BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
PORT="${1:-11321}"

if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi

# REPL mode only engages on a TTY, and `script` is how we give a child one.
# Git Bash ships no `script`, and Windows has no pty a POSIX tool can hand a
# native process anyway -- without this guard the transcript comes back EMPTY,
# check 1 ("no skip lines") passes because there are no lines at all, and only
# check 2 fails. A vacuous pass is worse than a skip.
if ! command -v script >/dev/null 2>&1; then
    echo "SKIP: no script(1) on this host — REPL mode needs a pty"
    exit 0
fi

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
FAKE_HOME="$SCRATCH/home"
mkdir -p "$FAKE_HOME/.mlx-serve/models/fake-vit-classifier"
printf '{"model_type": "vit"}\n' > "$FAKE_HOME/.mlx-serve/models/fake-vit-classifier/config.json"

# Run `mlx-serve run` under a pty, transcript to $1. Bounded wait so a
# regression can't hang the suite; the process normally exits on its own
# (config parse failure on the nonexistent model dir).
#
# `script(1)` is TWO different programs with the same name, and the difference
# is exactly how you hand it a command:
#   BSD/macOS:   script -q <file> <cmd> <args...>
#   util-linux:  script -q -c "<cmd string>" <file>   (Linux)
# Feeding the BSD form to util-linux fails with rc=1 and writes NO transcript,
# which reads as "the server printed nothing" -- so before the non-empty
# assertion below existed, this test reported a quieting regression that was
# really an argv-order difference. Detect the flavour rather than the OS: a Mac
# with util-linux from brew on PATH is the same program as Linux's.
SCRIPT_IS_UTIL_LINUX=0
if script --version 2>&1 | grep -qi util-linux; then
    SCRIPT_IS_UTIL_LINUX=1
fi

run_case() {
    local out="$1"; shift
    : > "$out"
    local pid
    if [ "$SCRIPT_IS_UTIL_LINUX" -eq 1 ]; then
        # One shell string, so every argument is quoted back into it.
        local cmd
        printf -v cmd '%q run %q --port %q' "$BIN" /nonexistent-model-dir "$PORT"
        local a
        for a in "$@"; do printf -v cmd '%s %q' "$cmd" "$a"; done
        HOME="$FAKE_HOME" script -q -c "$cmd" "$out" </dev/null >/dev/null 2>&1 &
    else
        HOME="$FAKE_HOME" script -q "$out" "$BIN" run /nonexistent-model-dir --port "$PORT" "$@" </dev/null >/dev/null 2>&1 &
    fi
    pid=$!
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 0
}

PASS=0
FAIL=0
check() { # $1 = description, $2 = 0/1 (0 = ok)
    if [ "$2" -eq 0 ]; then
        echo "PASS: $1"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $1"
        FAIL=$((FAIL + 1))
    fi
}

# 1. Default `run`: no skip lines in the REPL transcript.
run_case "$SCRATCH/default.txt"
# An empty transcript would satisfy the check below without proving anything --
# the fixture must have produced SOME output for "no skip lines in it" to mean
# "the quieting worked".
if [ ! -s "$SCRATCH/default.txt" ]; then
    check "run (default) produced a transcript at all" 1
else
    check "run (default) produced a transcript at all" 0
fi
if grep -q "\[discovery\] skip" "$SCRATCH/default.txt"; then
    check "run (default) does not print [discovery] skip lines" 1
else
    check "run (default) does not print [discovery] skip lines" 0
fi

# 2. Explicit --log-level info: skip lines still reachable (also proves the
#    fixture actually produces them — guards against a vacuous check 1).
run_case "$SCRATCH/info.txt" --log-level info
if grep -q "\[discovery\] skip" "$SCRATCH/info.txt"; then
    check "run --log-level info still prints [discovery] skip lines" 0
else
    check "run --log-level info still prints [discovery] skip lines" 1
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
