#!/usr/bin/env bash
# Layer offload over ggml RPC (rpc-offload-plan.md Part 1).
#
# Gated on RPC_WORKER=host:port (a peer running `mlx-serve --rpc-serve <port>`)
# and RPC_TEST_GGUF=<path> so CI skips it. Proves, against a LIVE worker:
#   [1] a dead endpoint is a NAMED load failure (never a half-loaded model)
#   [2] the split load assigns layers to BOTH a local device and RPC0
#   [3] a chat request decodes coherently and reports tok/s
# Byte-identity is not the bar (cross-backend); coherence is.
set -u
cd "$(dirname "$0")/.."

if [ -z "${RPC_WORKER:-}" ] || [ -z "${RPC_TEST_GGUF:-}" ]; then
  echo "SKIP: set RPC_WORKER=host:port and RPC_TEST_GGUF=/path/model.gguf"
  exit 0
fi
BIN=${BIN:-zig-out/bin/mlx-serve}
[ -x "$BIN" ] || BIN="$BIN.exe"
PORT=${PORT:-8177}
LOG=$(mktemp)
pass=0; fail=0
ok()  { echo "PASS [$1] $2"; pass=$((pass+1)); }
bad() { echo "FAIL [$1] $2"; fail=$((fail+1)); }
cleanup() { pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; taskkill //F //IM mlx-serve.exe >/dev/null 2>&1 || true; }
trap cleanup EXIT
PY=$(command -v python || command -v python3)

# [1] dead endpoint
DEAD=127.0.0.1:59999
timeout 60 "$BIN" --serve --host 127.0.0.1 --port "$PORT" --model "$RPC_TEST_GGUF" --rpc "$DEAD" > "$LOG" 2>&1
if grep -q "worker unreachable: $DEAD" "$LOG" && ! grep -q "engine ready" "$LOG"; then ok 1 "dead worker refused by name"; else bad 1 "dead worker not refused (see $LOG)"; fi

# [2] split load
"$BIN" --serve --host 127.0.0.1 --port "$PORT" --model "$RPC_TEST_GGUF" --rpc "$RPC_WORKER" > "$LOG" 2>&1 &
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "engine ready" "$LOG" && break; sleep 1; done
rpc_layers=$(grep -c "assigned to device RPC0" "$LOG")
local_layers=$(grep -c "assigned to device" "$LOG")
if [ "$rpc_layers" -gt 0 ] && [ "$local_layers" -gt "$rpc_layers" ]; then ok 2 "layers split: $rpc_layers on RPC0 of $local_layers"; else bad 2 "no split (rpc=$rpc_layers total=$local_layers)"; fi

# [3] coherent decode + tok/s
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
RESP=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to five in words, comma separated.\"}],\"max_tokens\":120,\"temperature\":0,\"enable_thinking\":false}")
CONTENT=$(printf '%s' "$RESP" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
TPS=$(printf '%s' "$RESP" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["timings"]["predicted_per_second"])' 2>/dev/null)
lc=$(printf '%s' "$CONTENT" | tr 'A-Z' 'a-z')
if printf '%s' "$lc" | grep -q "one" && printf '%s' "$lc" | grep -q "five" && [ -n "$TPS" ]; then ok 3 "coherent ($TPS tok/s): $CONTENT"; else bad 3 "incoherent or no timings: $RESP"; fi

echo "rpc_offload: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
