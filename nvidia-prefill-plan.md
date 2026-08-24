# NVIDIA remote prefill ("nvidia prefix cache") — PoC plan

Target model: gemma-4-12b (GGUF on both sides for v1). Branch: testing/windows_linux.

## Shape of the PoC (v1 — llama.cpp on BOTH sides)

The Linux/WSL box (llama.cpp CUDA) prefills; the Mac (llama.cpp Metal, embedded
engine) decodes. Both load the SAME gemma-4-12b GGUF. KV travels as the
llama.cpp sequence-state blob (`llama_state_seq_get_data` /
`llama_state_seq_set_data`) — upstream's own cross-backend serialization, so
there is NO layout translation to write. MLX-side import (translating KV into
our MLX KVCache for the QAT pack) is v2, explicitly out of scope here.

Flow:
1. Mac gets a chat request for a GGUF model and `--remote-prefill <url>` is set.
2. Mac tokenizes/renders as usual, POSTs the token ids to the remote box.
3. Remote runs prefill only (no sampling), extracts the seq state blob, streams
   it back.
4. Mac `llama_state_seq_set_data` into the request's slot, then decodes locally
   from position N.
5. Fallback: any remote error/timeouts → local prefill, request must still
   succeed. Remote prefill must never be able to fail a request.

## Task split

### linux-x64 — server side (prefill worker)
- Verify the branch builds + tests in WSL; build llama.cpp with CUDA
  (`scripts/build-llama-cuda.sh`, §3.3 says CUDA compile unverified — verify,
  fix, confirm the GPU actually engages, not the CPU backend).
- Get a gemma-4-12b GGUF (Q4_K_M or similar) working on the CUDA build.
- New endpoint `POST /v1/prefill` on the existing server (embedded-engine
  path): body = `{"model": ..., "tokens": [ids]}` → runs decode of the whole
  prompt batch with logits off, then `llama_state_seq_get_size/get_data`,
  responds with the blob (binary body, `application/octet-stream`, token count
  + model id echoed in headers). Wire whatever shim exports are missing through
  `lib/llama_shim` + `src/arch/llama_ffi.zig`.
- Keep it single-purpose: one seq, clear the seq after responding
  (`llama_memory_seq_rm`), refuse when a gen is in flight if that keeps it simple.

### windows-x64 — client side (consumer)
- Same repo, shared Zig code (it compiles for both hosts): implement
  `--remote-prefill <base-url>` on the serve path. Before local prefill of an
  embedded-engine request: POST tokens, receive blob, `llama_state_seq_set_data`,
  decode continues from the restored position. Timeout + ANY failure → silent
  local-prefill fallback (log one line: `[remote-prefill] engaged N tokens` /
  `fell back: <why>`).
- Test loop entirely on the Windows/WSL pair: WSL CUDA server as remote,
  Windows CPU build as consumer (localhost networking WSL<->Windows works).
- Unit tests for the client glue (request framing, fallback), TDD where behaviour
  can regress; shell test `tests/test_remote_prefill.sh` gated on a
  `REMOTE_PREFILL_URL` env so it skips on CI.

### mac (me) — integration + review
- Keep the branch green on macOS (merge-fallout fixes already pushed).
- Final e2e: Mac decode against the WSL CUDA prefill over LAN.

## Rules of engagement
- Coordinate through this branch; FEWER, BIGGER pushes (user's instruction).
  Pull before you start and before every push; announce pushes to m4max.
- Don't touch MLX-side files (transformer.zig, prefix_cache.zig etc.) — v1 is
  embedded-engine only. server.zig edits: linux owns the new endpoint handler,
  windows owns the client + flag; keep them in separate hunks/files where
  possible (suggest new file `src/remote_prefill.zig` for shared logic).
- Blob compatibility: both sides must run the same llama.cpp build (b10472
  vendored) and the same GGUF file. Assert model id + token count echo before
  restoring; version-gate the endpoint response with a header.

## Known risks (accepted for PoC)
- State blob format is llama.cpp-version-coupled — fine, we pin b10472.
- gemma-4 SWA cache: llama.cpp may not serialize pruned sliding-window state
  for arbitrary lengths — if `state_seq_get_data` refuses/truncates on SWA
  models, first fall back to full-cache mode (`swa_full=true`) on both sides
  and note the memory cost.
- Wire cost: blob is per-layer KV f16 — a few hundred MB at long contexts;
  fine on LAN for the PoC.

## Shrink round (2026-08-23, linux-x64)

- `swa_full=false` verified round-trip-safe (windowed blob → full consumer) but
  is NOT a size lever on b10472: the blob is already window-bounded on SWA
  layers. Measured gemma-4-12b: fixed ~335 MB (f16) / ~178 MB (q8_0) + 16.4 /
  8.7 KB per token. Worker session windowed anyway (VRAM).
- q8_0 KV: works with FA forced; restore refused across types by llama.cpp.
  Wire v2 adds `X-Prefill-Kv-Type` + `X-Prefill-Swa`. Run `--kv-quant 8` on
  both sides. See docs/gotchas/server-http.md.
