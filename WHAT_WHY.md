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
worker:

| Consumer | Local (alone) | With remote prefill | Verdict |
|---|---|---|---|
| M4 mini 16 GB | ~32.2 s | ~16.9 s | ~1.9x faster end to end |
| M4 Max | 7.0 s | 9.0 s | not worth it (yet) |

The mini wins big because its own prefill is slow and the GPU does that part
4.5x faster. The Max loses because it is nearly as fast as the worker and the
KV transfer ate the difference. The current round cuts the wire cost roughly
in half (quantized KV cache, windowed sliding-window export), which moves the
break-even point down; the economics are computed per model and per machine,
and the client simply declines to use a worker that cannot pay for itself.

The slower your Mac, the more this gives you. The target user is not the
Ultra owner, it is the base mini or MacBook Air owner with any gaming PC on
the same network.

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
- Blob shrink round (quantized + windowed KV export) is landing now.
- Layer offload is a design, not code.
