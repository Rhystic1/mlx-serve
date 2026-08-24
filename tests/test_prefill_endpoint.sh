#!/usr/bin/env bash
# POST /v1/prefill -- the remote-prefill WORKER (server side of
# nvidia-prefill-plan.md). Pins the wire contract in src/remote_prefill.zig from
# the outside: a 200 is a raw llama.cpp sequence-state blob with every identity
# header the consumer checks, the blob describes ONLY this request (two
# identical requests -> byte-identical blobs, so the dedicated seq is cleared
# between them), and each malformed shape is refused by name with a 400.
#
# Needs a GGUF: PREFILL_TEST_GGUF=<path> (falls back to LLAMA_TEST_MODEL).
# Skips cleanly without one so CI stays green.
set -u
cd "$(dirname "$0")/.."

GGUF="${PREFILL_TEST_GGUF:-${LLAMA_TEST_MODEL:-}}"
if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
  echo "SKIP: set PREFILL_TEST_GGUF (or LLAMA_TEST_MODEL) to a .gguf to run this test"
  exit 0
fi
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: $BIN not built (zig build -Doptimize=ReleaseFast)"; exit 1; }
PORT="${PREFILL_TEST_PORT:-18093}"
TMP="$(mktemp -d)"
pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

"$BIN" --model "$GGUF" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 4096 > "$TMP/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 90); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server never became healthy"; tail -20 "$TMP/server.log"; exit 1; }

MODEL_ID="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"
MODEL_BYTES="$(stat -c %s "$GGUF")"

# A ~600-token prompt through the server's own tokenizer, so the ids belong to
# this vocabulary by construction.
python3 -c "import json;print(json.dumps({'content':('The quick brown fox jumps over the lazy dog near the riverbank at dawn. '*40).strip()}))" > "$TMP/tok.json"
curl -s -X POST "http://127.0.0.1:$PORT/tokenize" -H 'Content-Type: application/json' --data-binary @"$TMP/tok.json" > "$TMP/toks.json"
N="$(python3 -c "import json;t=json.load(open('$TMP/toks.json'))['tokens'];json.dump({'model':'$MODEL_ID','tokens':t},open('$TMP/req.json','w'));print(len(t))")"
[ "$N" -gt 100 ] && ok "tokenized a real prompt ($N ids)" || bad "tokenize returned $N ids"

post() { curl -s -o "$2" -D "$3" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/prefill" -H 'Content-Type: application/json' --data-binary "$1"; }
hdr() { tr -d '\r' < "$1" | awk -v k="$2" 'BEGIN{IGNORECASE=1} tolower($1)==tolower(k":"){print $2}'; }

echo "[1] a well-formed request answers a binary state blob with every identity header"
code="$(post @"$TMP/req.json" "$TMP/blob1.bin" "$TMP/h1.txt")"
[ "$code" = "200" ] && ok "200" || bad "status $code: $(head -c 300 "$TMP/blob1.bin")"
[ "$(hdr "$TMP/h1.txt" Content-Type)" = "application/octet-stream" ] && ok "application/octet-stream" || bad "content-type $(hdr "$TMP/h1.txt" Content-Type)"
[ "$(hdr "$TMP/h1.txt" X-Prefill-Version)" = "2" ] && ok "X-Prefill-Version: 2" || bad "version header"
[ "$(hdr "$TMP/h1.txt" X-Prefill-Swa)" = "windowed" ] && ok "X-Prefill-Swa: windowed (worker session prunes sliding layers)" || bad "swa header $(hdr "$TMP/h1.txt" X-Prefill-Swa)"
case "$(hdr "$TMP/h1.txt" X-Prefill-Kv-Type)" in f16|q8_0|q4_0) ok "X-Prefill-Kv-Type names the cache type" ;; *) bad "kv-type header $(hdr "$TMP/h1.txt" X-Prefill-Kv-Type)" ;; esac
[ "$(hdr "$TMP/h1.txt" X-Prefill-Model)" = "$MODEL_ID" ] && ok "X-Prefill-Model echoes the request" || bad "model header $(hdr "$TMP/h1.txt" X-Prefill-Model)"
[ "$(hdr "$TMP/h1.txt" X-Prefill-Tokens)" = "$N" ] && ok "X-Prefill-Tokens == N ($N)" || bad "tokens header $(hdr "$TMP/h1.txt" X-Prefill-Tokens)"
BL="$(stat -c %s "$TMP/blob1.bin")"
[ "$(hdr "$TMP/h1.txt" X-Prefill-Bytes)" = "$BL" ] && [ "$BL" -gt 1000 ] && ok "X-Prefill-Bytes == body length ($BL)" || bad "bytes header $(hdr "$TMP/h1.txt" X-Prefill-Bytes) vs body $BL"
[ "$(hdr "$TMP/h1.txt" X-Prefill-Model-Bytes)" = "$MODEL_BYTES" ] && ok "X-Prefill-Model-Bytes == GGUF size" || bad "model-bytes header $(hdr "$TMP/h1.txt" X-Prefill-Model-Bytes) vs $MODEL_BYTES"
V="$(hdr "$TMP/h1.txt" X-Prefill-Vocab)"; [ "${V:-0}" -gt 1000 ] 2>/dev/null && ok "X-Prefill-Vocab present ($V)" || bad "vocab header '$V'"

echo "[2] the blob describes THIS request only: a repeat is byte-identical (seq cleared between jobs)"
code="$(post @"$TMP/req.json" "$TMP/blob2.bin" "$TMP/h2.txt")"
[ "$code" = "200" ] && cmp -s "$TMP/blob1.bin" "$TMP/blob2.bin" && ok "second blob identical" || bad "second request: status $code, cmp=$(cmp "$TMP/blob1.bin" "$TMP/blob2.bin" 2>&1 | head -1)"

echo "[3] a different prompt yields a different blob (the state is not a constant)"
python3 -c "import json;t=json.load(open('$TMP/toks.json'))['tokens'];json.dump({'model':'$MODEL_ID','tokens':t[:len(t)//2]},open('$TMP/req_half.json','w'))"
code="$(post @"$TMP/req_half.json" "$TMP/blob3.bin" "$TMP/h3.txt")"
[ "$code" = "200" ] && ! cmp -s "$TMP/blob1.bin" "$TMP/blob3.bin" && ok "half-prompt blob differs" || bad "half-prompt: status $code"

echo "[4] malformed shapes are refused by name"
for case in '{"tokens":[1,2,3]}|model' '{"model":"'"$MODEL_ID"'"}|tokens' '{"model":"'"$MODEL_ID"'","tokens":[5]}|at least 2' '{"model":"'"$MODEL_ID"'","tokens":[1,-2]}|non-negative' '[1,2]|object' '{nope|Invalid JSON'; do
  body="${case%%|*}"; needle="${case##*|}"
  code="$(post "$body" "$TMP/e.bin" "$TMP/he.txt")"
  if [ "$code" = "400" ] && grep -q "$needle" "$TMP/e.bin"; then ok "400 mentioning '$needle'"; else bad "expected 400/'$needle', got $code: $(head -c 200 "$TMP/e.bin")"; fi
done

echo "[5] the server still serves chat after prefill jobs (the dedicated session never touched the chat cache)"
code="$(curl -s -o "$TMP/chat.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8}')"
[ "$code" = "200" ] && grep -q '"content"' "$TMP/chat.json" && ok "chat 200" || bad "chat status $code"

grep -q '\[prefill\] .* tokens in .* ms' "$HOME/.mlx-serve/logs/mlx-serve-$PORT.log" 2>/dev/null && ok "[prefill] engagement line logged" || bad "no [prefill] line in ~/.mlx-serve/logs/mlx-serve-$PORT.log"

echo; echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
