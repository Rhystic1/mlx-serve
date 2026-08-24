# Layer offload over the network + worker discovery — plan

Branch: testing/windows_linux. Read WHAT_WHY.md and nvidia-prefill-plan.md
first; this builds on the same embedded llama.cpp engine and the same
Windows/Linux CUDA worker box. Owner: windows-x64. m4max reviews and runs the
Mac side of the e2e.

## Goal

Run a GGUF model that does not fit on one machine by splitting its layers
across the Mac (Metal) and the NVIDIA box (CUDA), using llama.cpp's ggml RPC
backend as the transport. Then make workers discoverable so the user never
types an address. Then show it in the built-in web console.

## Part 1: RPC layer offload

llama.cpp already has the whole mechanism. Upstream shape:
- worker side: `ggml_backend_rpc_start_server(backend, endpoint, cache_dir,
  free_mem, total_mem)` serves one local backend (CUDA0 here) over TCP. The
  stock `rpc-server` tool is a thin wrapper around it.
- consumer side: `ggml_backend_rpc_add_device("host:port")` (name may differ at
  b10472, check ggml-rpc.h) returns a ggml device; put it in
  `llama_model_params.devices` alongside the local ones before `llama_model_load`
  and llama.cpp assigns layers across devices by `tensor_split` /
  `n_gpu_layers` exactly as it does across multiple GPUs.

What we build:
1. Build: GGML_RPC=ON in scripts/build-llama-cuda.sh and fetch-llama.sh, stage
   libggml-rpc beside the other backends on every host (the same silent-CPU
   fallback trap as the CUDA runtime applies: verify it loaded, not just that
   the file exists). macOS build too, since the Mac is the consumer.
2. Worker mode in OUR binary, not a separate tool: `mlx-serve --rpc-serve
   [host:]port` starts the ggml RPC server in-process on the box's best backend
   and keeps the normal HTTP server up (health, /v1/models, metrics). One
   binary, one process, same logs. Shim exports via lib/llama_shim +
   src/arch/llama_ffi.zig.
3. Consumer flag: `--rpc host:port[,host:port...]` on the serve path. Plumb
   into the llama engine's model params before load. `--tensor-split` or an
   auto split from advertised free memory (worker reports it via the RPC
   protocol's get_alloc_size/free mem).
4. Memory preflight: the load-refusal path (`loadRequirementBytes`) must count
   remote device memory as capacity, or every model this feature exists for
   gets refused before it tries. Refusals name both local and remote numbers.
5. Test: a model that fits NEITHER box alone. Candidates on the Mac's external
   drive: unsloth/Qwen3.8-27B-GGUF (pick a quant > 16 GB and < mini + 5060 Ti).
   Prove: refuses alone on each box, loads split, decodes coherently, tok/s
   reported. Byte-identity is NOT the bar (cross-backend); coherence is.
   tests/test_rpc_offload.sh gated on RPC_WORKER env so CI skips it.
6. Measure: tok/s and prefill for the split model, plus the hop tax on a model
   that DOES fit locally (split it anyway, compare vs local) so the cost per
   boundary is a number, not a guess.

Rules: fall back is NOT silent here (unlike remote prefill): if an RPC device
is configured and unreachable, the load fails with a named error; a model
half-loaded on a dead peer is worse than no model. TDD where behaviour can
regress; fewer bigger pushes; no Claude-Session trailers; message m4max on
push or block.

## Part 2: worker discovery over the LAN layer

The existing LAN layer (lan.zig, lan_mdns.zig on Windows/Linux, Bonjour on
Mac) already advertises servers and their model lists. Extend, do not
reinvent:
1. Advertise capabilities in the TXT record: `prefill=1` when /v1/prefill is
   live (with the model id, KV type, SWA mode it serves), `rpc=<port>` plus
   free/total device memory when --rpc-serve is up.
2. Consumer: `--remote-prefill auto` picks the first discovered peer serving
   the byte-identical model with matching KV type; `--rpc auto` picks
   discovered RPC workers. Explicit URLs keep working and win over auto.
3. Reuse the existing keyless-gate rules (`routeClass` x `SharedSet`) so a
   peer only offers prefill/RPC for models it shares.
4. Tests: the LAN integration suite already has 23 cases on Linux; add the
   capability advert + auto-pick cases there. Mac interop is m4max's run.

Known traps from the prefill round, all still apply: WSL2 mirrored networking
makes localhost reach the wrong worker (check the model-bytes header), the
Windows firewall only allows the ports it was told about, and a flag the
server does not know is refused at boot (good) but a flag that exists on the
MLX side is a silent no-op on GGUF (`--kv-quant` vs `--llama-kv-quant`).

## Part 3: console UI

GET / on the consumer gets a "Cluster" panel: local + remote devices, which
layer range lives where (derive from the split; llama.cpp logs it at load),
per-device memory, live tok/s, prefill worker in use and whether the last
request engaged it. Read-only first; config (pick peers, set the split) comes
after Part 2 works. The page renders on an empty server and fetches
client-side (see the index.html rules in CLAUDE.md); pure logic in a helper
with a hermetic test, per the html_console_test.mjs pattern.

## Out of scope

MLX-side anything (m4max owns MLX KV import separately). Windows/Linux MLX.
Speed: this buys capacity; the split model runs at the slowest stage's pace.

## Part 1 status (2026-08-23, windows-x64)

Steps 1-4 shipped and proven on a loopback pair on the Windows box (worker
and consumer on the same 5060 Ti): `--rpc-serve`, `--rpc`, `--tensor-split`,
the preflight (remote free counts as capacity, refusal names both sides), and
the no-silent-fallback rule (dead endpoint = named load failure). Build:
GGML_RPC=ON + a libggml-rpc.so stage check in build-llama-cuda.sh; the
Windows release zip already ships ggml-rpc.dll. Measured hop tax (step 6,
LFM2.5-2.6B Q4, one boundary, loopback): decode 170 → ~125 tok/s, prefill
~3000 → ~1500 tok/s. Story: docs/gotchas/server-http.md.

Open, needs the Mac: step 5 (a model that fits neither box; the Mac must have
RPC in its XCFramework — `zig build test -Dtest-filter="RPC backend is
LOADED"` answers that) and the cross-machine numbers.

## Part 3 status (2026-08-24, windows-x64 / m4max split)

Data half shipped: `GET /v1/cluster` (schema agreed with m4max, who owns the
console tab). Layer assignment is llama.cpp's own truth, captured at load via
`llama_log_set`; per-hop timing is `hops:null` until ggml exposes it.
Blocked for the Mac-as-consumer split: the macOS llama asset in
fetch-llama.sh has no ggml-rpc backend (nm-confirmed) — needs an
RPC-enabled macOS build (windows-x64 owns that next).

## Part 2 status (2026-08-24, windows-x64)

Shipped: TXT `rpc=<port>`/`pf=1` + caps from the peer's /v1/cluster;
`--rpc auto`, `--remote-prefill auto` (same model, same kv type); peers in
/v1/cluster carry `caps`. Open: the LAN shell-suite case (needs a Linux/Mac
runner), and the box-alone "refuses" arm is a Mac property only — Windows
spills to system RAM (see docs/gotchas/server-http.md).
