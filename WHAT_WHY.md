# What is this branch, and why

This is the Windows / Linux port of mlx-serve, plus the first cross-machine
feature built on top of it: remote prefill, where an NVIDIA GPU does the heavy
prompt processing and your Mac does the decoding.

mlx-serve started as a native Zig server for MLX models on Apple Silicon. MLX
does not exist on Windows and its CUDA backend is Linux only, so off-Apple this
branch swaps the inference floor to llama.cpp (GGUF models, CUDA on both
Windows and Linux) and keeps everything else. That "everything else" is the
point of this document.

## Why not just run llama.cpp directly?

llama.cpp is the engine. mlx-serve is the server you actually want in front of
it. The whole HTTP, tooling and management stack is native Zig, one binary, no
Python, and it runs identically on macOS, Windows and Linux:

- **Four API surfaces, properly.** OpenAI chat/completions and Responses,
  Anthropic /v1/messages (Claude Code works out of the box), and the Ollama
  /api/* surface, all off one server. Streaming, usage accounting, cached-token
  reporting, context-overflow errors that name both numbers.
- **Tool calling that survives real models.** Models mangle tool calls
  constantly: broken JSON, wrong tag families, truncated calls, params buried
  in the wrong level. mlx-serve parses every tag dialect, repairs what is
  repairable, coerces types to the tool schema, and never ships invalid
  arguments. This pipeline is pinned by a replay harness of real captured
  traffic. llama.cpp gives you none of this.
- **A real CLI.** `mlx-serve pull` with resumable downloads, model aliases,
  `list`, a `run` REPL, and `mlx-serve launch <agent>` which configures claude,
  codex, aider, opencode and friends against your local server in one command.
- **Model discovery and management.** Point it at model folders, it finds and
  classifies everything (GGUF and MLX side by side), serves a live /v1/models
  with real context lengths, loads and unloads on demand, and refuses loads
  that will not fit in memory with an error that tells you why.
- **LAN sharing.** Servers find each other with zero config (Bonjour/mDNS,
  hand-rolled on Linux and Windows), and any box can transparently serve a
  model that actually lives on a peer. One flag.
- **Observability.** Prometheus + JSON metrics, per-request timings, a status
  bar, structured logs that make post-mortems possible.
- **A built-in web console.** Chat UI with streaming, markdown, voice, and
  monitoring, served straight from the binary at GET /. No extra install.
- **Robustness you only notice when it is missing.** Stall timeouts, stream
  keepalives, repetition-loop detection that stops runaway generations,
  graceful cancel on disconnect, API keys, single-flight admission.

If you use llama.cpp directly you get an engine and a demo server. This gives
you the product around it.

## Remote prefill: the NVIDIA prefix cache

A chat request has two phases. Prefill reads the whole prompt (fast on big
GPUs, slow on small Macs), decode generates tokens one by one (Macs are fine at
this). Remote prefill splits them across machines: a box with an NVIDIA card
runs the prompt through the model, exports the resulting KV state, and ships it
to your Mac, which restores it and starts decoding immediately.

Both machines load the same GGUF, byte for byte. The state travels in
llama.cpp's own cross-backend format, so a CUDA worker feeds a Metal consumer
with no translation layer. One flag on the consumer (`--remote-prefill
http://worker:8080`), one endpoint on the worker (`POST /v1/prefill`). Any
failure, mismatch or timeout falls back to local prefill silently; remote
prefill can never break a request.

Measured, gemma-4-12b Q4_K_M, ~3.4k-token prompt, gigabit LAN, RTX 5060 Ti
worker, fresh server per run, greedy:

| Consumer | KV wire format | Local (alone) | With remote prefill | Verdict |
|---|---|---|---|---|
| M4 mini 16 GB | f16 (~570 MB) | ~32.2 s | ~16.9 s | ~1.9x faster end to end |
| M4 Max | f16 (~570 MB) | 7.0 s | 9.0 s | loses, transfer eats the win |
| M4 Max | q8_0 (~210 MB) | 7.7 s | 6.9 s | wins by ~0.8 s |

The mini wins big because its own prefill is slow (~6.7 ms/token) and the GPU
does that part far faster. The Max lost on f16 because it is nearly as fast as
the worker (~2.0 vs ~0.8 ms/token) and a 570 MB transfer ate the difference.
Quantizing the exported KV cache to q8_0 cut the wire roughly in half and
flipped the Max to a win.

### What the wire actually costs

The exported state is not linear in prompt length. gemma-4 is 43 sliding-window
layers plus 6 global ones, so the blob is a fixed window payload plus a small
per-token term:

| KV type | Fixed | Per token | At 4k | At 16k | At 32k |
|---|---|---|---|---|---|
| f16 | ~335 MB | 16.4 KB | ~400 MB | ~600 MB | ~860 MB |
| q8_0 | ~178 MB | 8.7 KB | ~213 MB | ~317 MB | ~457 MB |

On gigabit that fixed part is ~1.6 s of transfer you pay once per request, and
the per-token part is under 0.1 ms/token. Everything else is the worker's
prefill (~0.8 ms/token on a 5060 Ti with q8). So remote costs roughly
`1.6 s + 0.9 ms x tokens` and local costs `your Mac's ms/token x tokens`. The
client fits this from observed replies and refuses a worker that cannot pay for
itself on your machine.

### Where it pays: other Macs, bigger prompts

Measured for the M4 mini and M4 Max; the rest are extrapolated from GPU core
count and memory bandwidth against those two points, same 5060 Ti worker, q8_0,
gigabit. Numbers are prefill wall time, local vs remote. Treat the extrapolated
rows as estimates, not measurements.

| Consumer | Local prefill | Break-even | 4k prompt | 16k prompt | 32k prompt |
|---|---|---|---|---|---|
| M1 / M2 MacBook Air, 8 GPU cores (est.) | ~9 ms/tok | ~200 tok | 36 s vs 5 s | 144 s vs 16 s | 288 s vs 30 s |
| M4 mini / MacBook Air, 10 cores (measured) | ~6.7 ms/tok | ~300 tok | 27 s vs 5 s | 107 s vs 16 s | 214 s vs 30 s |
| M4 Pro, 20 cores (est.) | ~3.5 ms/tok | ~600 tok | 14 s vs 5 s | 56 s vs 16 s | 112 s vs 30 s |
| M4 Max, 40 cores (measured) | ~2.0 ms/tok | ~2k tok | 8 s vs 5 s | 32 s vs 16 s | 64 s vs 30 s |
| M3 Ultra, 80 cores (est.) | ~1.2 ms/tok | ~5k tok | loses | 19 s vs 16 s | 38 s vs 30 s |

Two things make the real picture better than this table for long prompts.
Local prefill is not really linear: attention cost grows with the square of
the context, so a 32k prompt on a small Mac is worse than 8x a 4k one, while
the remote path stays close to linear. And on memory-tight machines the mini
showed a second effect: decode after a long local prefill ran ~1.7x slower than
decode after a remote one, because the GPU had just spent 20+ seconds at full
load. Neither is in the table.

A faster worker moves every row. A 4090 prefills roughly 3x faster than the
5060 Ti used here, which pushes the M4 Max break-even well under 1k tokens and
makes even an Ultra a mild win at long contexts.

The shape of it: the slower your Mac and the longer your prompts, the more
this gives you. The target user is not the Ultra owner, it is the base mini or
MacBook Air owner with any gaming PC on the same network, running agent
workloads where every turn re-reads a 10k to 50k token context.

## TODO: layer offload over the network

Remote prefill needs both machines to hold the full model. The next step is
the opposite trade: split the model itself across machines, so you can run a
model no single box can hold. llama.cpp already ships the transport (the ggml
RPC backend); what it lacks is everything around it, which is exactly what
mlx-serve is for:

- wire `--rpc` through the embedded engine and the build scripts
- discovery of RPC-capable peers over the existing LAN layer
- a pretty UI for it: the built-in web console gets a view of which machine
  holds which layers, live per-hop stats, and soon config so you can set the
  split without touching a terminal

Per decode token only a few KB of activations cross each split boundary, so
this works on ordinary gigabit. It buys capacity, not speed: a 120B model
across a Mac and a 16 GB GPU runs at the pace of the slowest stage. But it
runs.

## Status

- Windows and Linux builds serve GGUF models with CUDA today, full test suite
  green on both.
- Remote prefill works end to end (Mac consumer, Windows CUDA worker) and is
  measured, not just demoed.
- Blob shrink round (q8_0 + windowed KV export) landed and measured: M4 Max
  went from a loss to a win at 3.3k tokens.
- Layer offload is a design, not code.
