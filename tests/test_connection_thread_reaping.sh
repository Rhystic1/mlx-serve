#!/bin/bash
# Regression: one completed HTTP request must not retain one pthread stack.
#
# The accept loop used to discard each joinable std.Thread handle. The handler
# returned, but macOS kept its stack mapping and page-table bookkeeping until a
# join/detach that never came. A long Claude Code session therefore grew from
# the model's 4-5 GB working set toward 10 GB while MLX cache_bytes stayed tiny.


# The measurement IS vmmap: this test counts the server's Stack regions before
# and after a burst of connections, which is how a leaked joinable thread shows
# up. vmmap is a macOS tool with no portable equivalent (Windows stack
# reservations are not enumerable this way), so off Darwin there is nothing to
# measure and a run reports "nothing was measured" as a FAILURE. Skip by name
# instead; porting this means finding a different observable, not a different
# spelling of vmmap.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "SKIP: thread reaping is measured with vmmap (macOS-only)"
    exit 0
fi
set -u

PORT="${1:-11483}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
REQUESTS="${CONN_REAP_REQUESTS:-300}"
EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
SERVER_PID=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$EMPTY_DIR" "$LOG"
}
trap cleanup EXIT

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

"$BINARY" --serve --model-dir "$EMPTY_DIR" --host 127.0.0.1 \
    --port "$PORT" --log-file off >"$LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        READY=1
        break
    fi
    sleep 0.1
done
if [ "$READY" -ne 1 ]; then
    echo "[fail] server did not start"
    cat "$LOG"
    exit 1
fi

sleep 0.5
stack_regions() {
    vmmap "$SERVER_PID" 2>/dev/null | grep -Ec '^Stack[[:space:]]+[[:xdigit:]]+-'
}

BEFORE="$(stack_regions)"
if [ "$BEFORE" -lt 1 ]; then
    echo "[fail] vmmap reported no stack regions for pid $SERVER_PID — nothing was measured"
    exit 1
fi
for _ in $(seq 1 "$REQUESTS"); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
        echo "[fail] health request failed"
        exit 1
    }
done
sleep 1
AFTER="$(stack_regions)"
if [ "$AFTER" -lt 1 ]; then
    echo "[fail] vmmap reported no stack regions after the requests — server died or vmmap failed"
    exit 1
fi
DELTA=$((AFTER - BEFORE))

echo "connection thread reaping: before=$BEFORE after=$AFTER delta=$DELTA requests=$REQUESTS"
if [ "$DELTA" -gt 8 ]; then
    echo "[fail] completed requests retained $DELTA stack regions"
    exit 1
fi

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "[fail] server stopped responding"
    exit 1
}
echo "[pass] connection thread stacks are reaped"
