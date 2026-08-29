#!/usr/bin/env bash
# MiniMax-H3 Acc (PDD) distillation end-to-end.
#
# Acc is NOT a style LoRA: trunk adapter + 32-interval PDD head bank, trained
# 8-step (or regrouped 4-step) Euler. This script pins the named 400s (Turbo
# combo, missing file) on a real pack, then — when the Acc file is on disk —
# a 512×512 × 22-frame × 4-step generate that returns rgb8 + stereo pcm.
# No pixel-oracle vs Comfy. SKIPs without an H3 pack (like test_minimax_h3.sh);
# the generate arm SKIPs without the Acc file.
#
# Usage: [H3_MODEL=<dir>] [H3_ACC=<file>] ./tests/test_h3_acc.sh [port]
set -uo pipefail
PORT="${1:-11362}"
MODEL="${H3_MODEL:-$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TASKS=$(python3 - "$MODEL/config.json" <<'PY'
import json, sys
try:
    print(",".join(json.load(open(sys.argv[1])).get("tasks", [])))
except Exception:
    print("")
PY
)
case ",$TASKS," in (*,ref2va,*) PART=ref2va; ACC_NAME="MiniMax-H3-Ref2VA-Acc-8Step.safetensors";;
                 *) PART=fl2va; ACC_NAME="MiniMax-H3-FL2VA-Acc-8Step.safetensors";; esac
WRONG_NAME="MiniMax-H3-FL2VA-Acc-8Step.safetensors"
[ "$PART" = fl2va ] && WRONG_NAME="MiniMax-H3-Ref2VA-Acc-8Step.safetensors"

ACC="${H3_ACC:-}"
if [ -z "$ACC" ]; then
  for c in "$MODEL/$ACC_NAME" \
           "$HOME/.mlx-serve/models/alibaba-pai/MiniMax-H3-Acc-LoRAs/$ACC_NAME"; do
    [ -f "$c" ] && ACC="$c" && break
  done
fi
WRONG_ACC=""
for c in "$MODEL/$WRONG_NAME" \
         "$HOME/.mlx-serve/models/alibaba-pai/MiniMax-H3-Acc-LoRAs/$WRONG_NAME"; do
  [ -f "$c" ] && WRONG_ACC="$c" && break
done

echo "pack=$MODEL partition=$PART acc=${ACC:-missing}"

LOG=/tmp/test_h3_acc_server.log
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
  sleep 1
done
rc=0

# [1] Acc + Turbo is refused by name, before any Acc file is read.
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","turbo":true,"acc":true}' -o /tmp/h3_acc_turbo.json -w "%{http_code}")
if [ "$code" = "400" ] && grep -q "mutually exclusive" /tmp/h3_acc_turbo.json; then
  echo "PASS: Acc+Turbo -> named 400"
else
  echo "FAIL: Acc+Turbo returned $code ($(head -c 160 /tmp/h3_acc_turbo.json))"; rc=1
fi

# [2] Missing Acc file names the HF repo.
if [ -z "$ACC" ]; then
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","acc":true}' -o /tmp/h3_acc_miss.json -w "%{http_code}")
  if [ "$code" = "400" ] && grep -q "alibaba-pai/MiniMax-H3-Acc-LoRAs" /tmp/h3_acc_miss.json; then
    echo "PASS: missing Acc file -> named 400 pointing at HF"
  else
    echo "FAIL: missing Acc returned $code ($(head -c 200 /tmp/h3_acc_miss.json))"; rc=1
  fi
else
  echo "SKIP: Acc file present — missing-file 400 covered by test_media_eviction_gate.sh"
fi

# [3] Wrong-partition Acc file is a named 400.
if [ -n "$WRONG_ACC" ]; then
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"x\",\"acc\":true,\"acc_path\":\"$WRONG_ACC\"}" \
    -o /tmp/h3_acc_part.json -w "%{http_code}")
  if [ "$code" = "400" ] && grep -qi "partition" /tmp/h3_acc_part.json; then
    echo "PASS: wrong-partition Acc -> named 400"
  else
    echo "FAIL: wrong-partition Acc returned $code ($(head -c 200 /tmp/h3_acc_part.json))"; rc=1
  fi
else
  echo "SKIP: other partition's Acc file not on disk"
fi

# [4] Live generate: 512×512 × 22 frames × 4 Acc steps. Transport is rgb8 +
# stereo pcm (the app muxes mp4). Generous wall budget — staged TE+DiT load
# dominates, not the four Euler steps.
if [ -z "$ACC" ]; then
  echo "SKIP: no Acc file at $MODEL/$ACC_NAME (set H3_ACC or download from https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)"
  [ "$rc" -eq 0 ] && exit 0
  exit "$rc"
fi

OUT=/tmp/test_h3_acc.json
START=$(date +%s)
code=$(curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"a calico cat blinking on a sunlit windowsill. overall_soundscape: soft rain.\",\"num_frames\":22,\"width\":512,\"height\":512,\"steps\":4,\"seed\":7,\"acc\":true,\"acc_path\":\"$ACC\"}" \
  -o "$OUT" -w "%{http_code}")
ELAPSED=$(( $(date +%s) - START ))
if [ "$code" != "200" ]; then
  echo "FAIL: Acc generation http $code after ${ELAPSED}s"; head -c 400 "$OUT"; rc=1
else
  python3 - "$OUT" "$ELAPSED" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
elapsed = int(sys.argv[2])
assert d["format"] == "rgb8", d.get("format")
assert d["fps"] == 24, d.get("fps")
F, H, W = d["frames"], d["height"], d["width"]
assert F == 22, f"requested 22 frames (on the 17k+5 ladder), got {F}"
assert (H, W) == (512, 512), (H, W)
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"rgb len {len(raw)} != {F*H*W*3}"
lo, hi = min(raw), max(raw)
assert hi - lo > 20, f"frames look uniform ({lo}..{hi})"
assert d.get("audio_format") == "pcm_s16le", d.get("audio_format")
assert d.get("audio_channels") == 2, d.get("audio_channels")
sr = d["audio_sample_rate"]
pcm = base64.b64decode(d["audio_data"])
n_frames_per_ch = len(pcm) // (2 * 2)
adur, vdur = n_frames_per_ch / sr, F / 24.0
assert abs(adur - vdur) < 0.06, f"audio {adur:.3f}s vs video {vdur:.3f}s"
# Staged load + 4 Acc steps at 512² should finish well under 30 min.
assert elapsed < 1800, f"Acc generate took {elapsed}s"
print(f"PASS: Acc generate -> {F}f {W}x{H} rgb8 range {lo}..{hi}, audio {adur:.3f}s stereo in {elapsed}s")
PY
  [ $? -eq 0 ] || rc=1
  if grep -q "acc=" "$LOG" || grep -qi "acc " "$LOG"; then
    echo "PASS: Acc engaged in the server log"
  else
    echo "FAIL: Acc generate succeeded but the log never mentioned Acc"
    grep -E "video|lora|acc" "$LOG" | tail -8
    rc=1
  fi
fi

exit "$rc"
