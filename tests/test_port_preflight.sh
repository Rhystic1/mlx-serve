#!/bin/bash
# Guard: the boot-time "is this port taken" check must ANSWER, fast, and be
# right about a live server.
#
# It used to answer by CONNECTING to the port and reading a connection refusal
# as "free". That is a different question, and its cost is not bounded by
# anything local: on a host that blackholes a SYN to a closed port instead of
# refusing it (WSL2 mirrored networking, live 2026-08-21) `connect()` sits in
# SYN-SENT for the full TCP retry ladder, so boot hung for minutes inside a
# check that exists to fail FAST — with no output past the log banner, which
# with --lan-share on the command line reads as an mDNS hang.
#
# The hang class is a WALL-CLOCK property, so it is asserted with a timeout,
# not with output. The second arm is the one the first arm can't see: a probe
# that never blocks is worthless if it also reports a running server as free
# (the real listener sets SO_REUSEPORT, so two instances would bind one port
# and silently split traffic — this check is the only thing preventing that).
#
# Usage: ./tests/test_port_preflight.sh [model] [port]

set -u

MODEL=${1:-~/models/LFM2.5-2.6B-heretic-Q4_K_M.gguf}
PORT=${2:-8137}
PASS=0
FAIL=0
TOTAL=0

MODEL=$(eval echo "$MODEL")
if [ ! -e "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

LOG=/tmp/mlx-serve-portpreflight.log
LOG2=/tmp/mlx-serve-portpreflight-2.log

echo "=== boot port pre-flight ==="

# [1] Free port: boot must reach the listener. A connect-probe hang shows up
#     here on any host that blackholes closed-port SYNs; everywhere else this
#     arm just proves the bind probe releases the port it tested.
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 --port $PORT \
    >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

UP=no
for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=yes; break; }
    kill -0 $SERVER_PID 2>/dev/null || break
    sleep 1
done
if [ "$UP" = yes ]; then
    run_test "a free port boots to a listening server" PASS
else
    run_test "a free port boots to a listening server" FAIL "no /health within 90s (see $LOG)"
fi

# [2] The probe must not hang before the args line. A pre-flight that blocks
#     shows as a boot with the log banner and nothing after it.
if grep -q "\[args\] serve:" "$LOG" 2>/dev/null; then
    run_test "boot gets past the pre-flight into arg reporting" PASS
else
    run_test "boot gets past the pre-flight into arg reporting" FAIL "no '[args] serve:' in $LOG"
fi

# [3] Held port: a second instance must refuse BY NAME, and refuse promptly —
#     the whole point is to not spend a model load finding out.
START=$(date +%s)
timeout 30 ./zig-out/bin/mlx-serve --model "$MODEL" --serve --host 127.0.0.1 \
    --port $PORT >"$LOG2" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
if [ "$RC" -eq 1 ] && grep -q "already in use" "$LOG2"; then
    run_test "a port held by a live server is refused by name" PASS
else
    run_test "a port held by a live server is refused by name" FAIL "rc=$RC, log: $(tail -2 "$LOG2" | tr '\n' ' ')"
fi
if [ "$ELAPSED" -le 15 ]; then
    run_test "the refusal costs no model load (${ELAPSED}s)" PASS
else
    run_test "the refusal costs no model load (${ELAPSED}s)" FAIL "took ${ELAPSED}s"
fi

# [4] The probe must leave nothing behind: the live server still serves after
#     being probed (a probe that stole the port, or that bound over it with
#     SO_REUSEPORT and kept it, would show here).
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    run_test "the probed server is still serving" PASS
else
    run_test "the probed server is still serving" FAIL "/health stopped answering after the probe"
fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
