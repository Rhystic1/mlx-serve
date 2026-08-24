#!/bin/bash
# Guard: `--remote-prefill` must be a PURE OPTIMISATION. The feature is allowed
# to not work; it is never allowed to fail a request.
#
# That invariant is what this script exists for, and the important half needs no
# remote at all: arm [1] points the client at a port nothing is listening on and
# asserts the request still answers, with one `[remote-prefill] fell back:` line
# saying why. A silent success would be just as wrong as a failure — the
# operator has to be able to tell that the remote is not being used.
#
# Arm [2] is the same request with the flag absent, and its answer must be
# byte-identical: remote prefill changes WHERE the prompt is prefilled, never
# what the model says. Greedy (temperature 0) is what makes that comparable.
#
# Arm [3] only runs when REMOTE_PREFILL_URL names a real worker (CI has none, so
# it skips): it asserts the log says `engaged N tokens` and that the answer
# still matches the local one. N is the FULL prompt count — the worker prefills
# all but the last token (remote_prefill.prefillSpan), and the count on the wire
# stays the full one.
#
# Usage: ./tests/test_remote_prefill.sh [model] [port]
#   REMOTE_PREFILL_URL=http://<ip>:<port>   enables arm [3]

set -u

MODEL=${1:-~/models/LFM2.5-2.6B-heretic-Q4_K_M.gguf}
PORT=${2:-8139}
PASS=0
FAIL=0
TOTAL=0

MODEL=$(eval echo "$MODEL")
if [ ! -e "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi

EXE=./zig-out/bin/mlx-serve
[ -x "$EXE" ] || EXE=./zig-out/bin/mlx-serve.exe
if [ ! -x "$EXE" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

# `python3` on macOS/Linux, `python` on this Windows host. A missing
# interpreter used to take every `2>/dev/null ||` fallback below, which shipped
# a TWO-TOKEN prompt and made the whole script assert nothing while reporting
# "fell back: prompt below the round-trip threshold" as if that were the answer.
#
# Detection must RUN the interpreter, not just find it: Windows ships an App
# Execution Alias named `python3` that exists on PATH, satisfies `command -v`,
# and then prints "Python was not found" to stderr and exits non-zero.
PY=""
for c in python3 python; do
    if [ "$("$c" -c 'print(1)' 2>/dev/null)" = "1" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "SKIP: no working python interpreter for request/response shaping"; exit 0; fi

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

TMP=$(mktemp -d)
SRV_PID=""   # seeded: the EXIT trap runs on the SKIP path too
trap 'kill ${SRV_PID:-} 2>/dev/null || true; rm -rf "$TMP"' EXIT

# The prompt must clear the client's own economic gate (MIN_REMOTE_TOKENS=256)
# or the request never leaves the box and arms [1]/[3] assert nothing. Built
# from repeated text so the token count is comfortably past it.
LONG=$("$PY" -c "print('The quick brown fox jumps over the lazy dog. ' * 120)")
if [ ${#LONG} -lt 3000 ]; then echo "FAIL: prompt builder produced ${#LONG} chars — arms would assert nothing"; exit 1; fi

start_server() {  # $1 = log path, $2.. = extra flags
    local log="$1"; shift
    "$EXE" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 4096 --log-level debug "$@" > "$log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 120); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV_PID" 2>/dev/null || { echo "server died early; log:"; tail -20 "$log"; return 1; }
        sleep 1
    done
    return 1
}

stop_server() {
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
}

ask() {  # $1 = output file
    curl -s -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$("$PY" - "$LONG" <<'PYEOF'
import json,sys
print(json.dumps({"model":"mlx-serve",
                  "messages":[{"role":"user","content":sys.argv[1]+"\n\nReply with exactly one short sentence."}],
                  "temperature":0,"max_tokens":24}))
PYEOF
)" > "$1" 2>&1
}

content_of() {  # $1 = json file → the assistant text, empty when absent
    "$PY" -c "import json,sys
try:
    print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])
except Exception:
    pass" "$1" 2>/dev/null
}

echo "=== remote prefill ==="

# ── [1] A dead remote must not cost the request ──────────────────────────────
# Port 9 is discard: nothing serves HTTP there, so every attempt fails somewhere
# between connect and read. Whichever way it fails, the answer must arrive.
echo "[1] unreachable remote → request still answers, fallback is logged"
if start_server "$TMP/dead.log" --remote-prefill "http://127.0.0.1:9"; then
    ask "$TMP/dead.json"
    CONTENT=$(content_of "$TMP/dead.json")
    if [ -n "$CONTENT" ]; then
        run_test "request answered despite an unreachable remote" PASS
    else
        run_test "request answered despite an unreachable remote" FAIL "no content; body: $(head -c 300 "$TMP/dead.json")"
    fi
    # The operator must be able to SEE that the remote was not used. A silent
    # local prefill is indistinguishable from a working remote.
    if grep -q '\[remote-prefill\] fell back:' "$TMP/dead.log"; then
        run_test "fallback logged with a reason" PASS
        echo "      $(grep -m1 '\[remote-prefill\] fell back:' "$TMP/dead.log" | sed 's/^ *//')"
    else
        run_test "fallback logged with a reason" FAIL "no '[remote-prefill] fell back:' line in the log"
    fi
    # It must never claim success it did not have.
    if grep -q '\[remote-prefill\] engaged' "$TMP/dead.log"; then
        run_test "does not claim engagement when the remote is dead" FAIL "log says engaged"
    else
        run_test "does not claim engagement when the remote is dead" PASS
    fi
    stop_server
else
    run_test "server booted with --remote-prefill" FAIL "server did not come up"
fi

# ── [2] The flag must not change the answer ──────────────────────────────────
echo "[2] same request with no flag → identical greedy answer"
if start_server "$TMP/plain.log"; then
    ask "$TMP/plain.json"
    BASE=$(content_of "$TMP/plain.json")
    stop_server
    if [ -n "$BASE" ] && [ "$BASE" = "${CONTENT:-}" ]; then
        run_test "fallback answer is byte-identical to the no-flag answer" PASS
    elif [ -z "$BASE" ]; then
        run_test "fallback answer is byte-identical to the no-flag answer" FAIL "baseline produced no content"
    else
        run_test "fallback answer is byte-identical to the no-flag answer" FAIL "answers differ"
    fi
else
    run_test "server booted without the flag" FAIL "server did not come up"
fi

# ── [3] A real worker, when one is configured ────────────────────────────────
if [ -z "${REMOTE_PREFILL_URL:-}" ]; then
    echo "[3] SKIP: set REMOTE_PREFILL_URL=http://<ip>:<port> to test against a live worker"
else
    echo "[3] live worker at $REMOTE_PREFILL_URL"
    if start_server "$TMP/live.log" --remote-prefill "$REMOTE_PREFILL_URL"; then
        ask "$TMP/live.json"
        LIVE=$(content_of "$TMP/live.json")
        if grep -q '\[remote-prefill\] engaged' "$TMP/live.log"; then
            run_test "remote prefill engaged" PASS
            echo "      $(grep -m1 '\[remote-prefill\] engaged' "$TMP/live.log" | sed 's/^ *//')"
        else
            # Not a failure of the invariant, but the arm cannot prove anything.
            run_test "remote prefill engaged" FAIL "fell back: $(grep -m1 '\[remote-prefill\] fell back:' "$TMP/live.log" | sed 's/^ *//')"
        fi
        # The whole point: a remotely-prefilled answer is the SAME answer.
        if [ -n "$LIVE" ] && [ "$LIVE" = "${BASE:-}" ]; then
            run_test "remote-prefilled answer matches the local one" PASS
        else
            run_test "remote-prefilled answer matches the local one" FAIL "differs from the local baseline"
        fi
        stop_server
    else
        run_test "server booted against the live worker" FAIL "server did not come up"
    fi
fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
