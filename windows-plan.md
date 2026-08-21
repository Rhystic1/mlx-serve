# Windows / Linux port — state and remaining work

Branch: `testing/windows_linux`. Four commits on top of `main`:

```
45cf570  fix: GGUF inference on Windows -- zero-layer KV shell, host-aware model ids
99ff538  docs: windows-plan.md
9f5566f  feat: GGUF-only Windows build (llama.cpp + CUDA), MLX gated out
f947fb5  build: cross-platform toolchain fetch (Windows/Linux hosts)
```

Read this whole file before touching anything. The traps in **§6** are the ones
that cost the most time the first time round, and none of them are guessable.

---

## 1. What this port is (and deliberately is not)

**MLX is not ported.** It has no Windows build at all, its CUDA backend is
Linux-only, and ~600 call sites across `transformer.zig` / `deepseek_v4.zig` go
through `mlx_fast_metal_kernel`, which is Metal by construction. llama.cpp has
first-class CUDA on both Windows and Linux, so off-Apple the server keeps the
entire HTTP / tool-calling / discovery / templating stack and swaps only the
inference floor.

This was viable because the engine boundary was already clean:
`model_registry.LoadedModel` holds mutually-exclusive engine fields, and
`scheduler.zig`'s `is_embedded` path already skipped every MLX per-slot
structure (zero KV layers, no SSM entries, no `legacy_gen`).

Selected by `-Dgguf-only`, which defaults ON for every non-macOS target. A Mac
can build the portable configuration with `-Dgguf-only` to check it.

---

## 2. Verified working today

`zig build` and `zig build test` both pass on Windows. **907 tests pass, 18
skipped, 0 fail.**

```
$ mlx-serve.exe --version
mlx-serve 26.8.10
mlx unknown                       <- honestly absent
nax unavailable (built without MLX)
ggml 0.20.1 (60eeeb608)
llama.cpp b10472

$ mlx-serve.exe --serve --host 127.0.0.1 --port 18093
$ curl /health     -> {"status":"ok"}
$ curl /v1/models  -> {"object":"list","data":[]}
$ curl /v1/nope    -> 404                       (route-existence rule intact)
$ curl /props      -> available_bytes: 16022376448
```

CUDA is live end to end — a test binary linked against the staged `llama.dll`
reported `ggml_cuda_init: found 1 CUDA devices … RTX 5060 Ti, compute
capability 12.0`. The prebuilt binary covers sm_120 (Blackwell).

**Inference works.** Verified against `Qwen3.8-27B-UD-IQ3_XXS` (11.9 GB IQ3_XXS)
on an RTX 5060 Ti:

- non-streaming completion returns the right text, with sane usage + timings
- SSE streaming emits the correct chunk lifecycle
- tool calling returns a well-formed `tool_call`, valid JSON arguments,
  `finish_reason: "tool_calls"`

Two bugs stood between "model loaded" and "token produced", both found by
serving a real model and reading the response rather than the code — see §6.8.
Expect more of the same shape: things that compile and load but have never
executed on this host.

---

## 3. Remaining work, in priority order

### 3.1 Actually serve a model  ← DO THIS FIRST

Nothing here has run inference. Until a GGUF model produces tokens, the port is
unproven where it matters.

```bash
.zig-toolchain/zig.exe build -Doptimize=ReleaseFast
# put any small GGUF in %USERPROFILE%\.mlx-serve\models\<org>\<repo>\
./zig-out/bin/mlx-serve.exe --serve --host 127.0.0.1 --port 8080
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

Then work outward: streaming, `/v1/messages` (Anthropic), `/api/chat` (Ollama),
tool calling, `continue_final_message`. Expect problems in the **connection
lifecycle** rather than in llama.cpp — see §6.2, the accept loop does not behave
the way it does on POSIX.

Watch for: `n_gpu_layers` defaults to 999 (`arch/llama.zig` `OpenOptions`), so
offload should be automatic; confirm it in the log rather than assuming.

### 3.2 Run the shell integration suite

166 scripts in `tests/`, none of which have been run on Windows. They are bash,
so Git Bash should carry most of them, but they assume POSIX tools and paths.
Start with the ones that do not need MLX models:

```
tests/test_index_page.sh          tests/test_route_404_no_load.sh
tests/test_models_capabilities.sh tests/test_model_rescan.sh
tests/test_multi_model_dir.sh     tests/test_loop_stop_signal.sh
```

Triage each into: passes / needs a portable fix / genuinely macOS-only (gate it
by name, do not delete).

### 3.3 Linux

**Upstream publishes no prebuilt Linux CUDA binary** — only cpu, vulkan, sycl
and openvino. `scripts/fetch-llama.sh` already refuses Linux **by name** and
points at `scripts/build-llama-cuda.sh`, which does not exist yet. Write it:
fetch the source at `LLAMA_TAG`, `cmake -DGGML_CUDA=ON`, stage
`lib/llama/lib/libllama.so` + headers to match the layout `verifyLlamaStage`
expects.

Then the code side, which should be small — `build.zig` already has the `else`
arm with `$ORIGIN` rpaths, and `status.zig` needs a `status_linux.zig` sibling
(`/proc/meminfo`, `/proc/stat`, NVML — the NVML code in `status_windows.zig`
ports almost verbatim, only `LoadLibraryA` becomes `dlopen`).

Do NOT quietly fall back to the cpu tarball. A build that reports CUDA support
it does not have is worse than one that refuses.

### 3.4 Claw back the three features the user asked for

The shim (`lib/llama_shim/llama_shim.h`, 25 functions) exposes text generation
only. llama.cpp offers more, and `mtmd.dll` already ships beside `llama.dll` in
the staged set:

- **Embeddings** — keeps `/v1/embeddings` alive on GGUF embedding models.
  Smallest of the three. `gen_stub.computeEmbeddingsBatch` currently refuses by
  name; that is the seam.
- **Vision via mtmd** — `mtmd.h` and `mtmd-helper.h` are already staged in
  `lib/llama/include`. Note mtmd does its OWN preprocessing, so it does not go
  through `qwen_vision`/`muse_vision`/`lfm2_vision` and does not resurrect
  `vision_stub.zig`.
- **Draft-model speculative decode** — llama.cpp has its own. Partly recovers
  what was lost with MTP/DFlash. Needs shim work plus scheduler wiring; do it
  last, and measure before claiming anything (`/bench` skill rules apply).

### 3.5 Un-skip what deserves it

18 tests skip. Most are pre-existing env-gated live tests. The ones this port
added, each with its reason stated in code:

| Skipped | Why | Worth revisiting? |
|---|---|---|
| 2 × HF-cache symlink tests | Windows needs Developer Mode to create symlinks | Only if CI enables it |
| 7 × `prefillMemoryNeeded` / `resolvePrefillChunk` | MLX billing math; the terms come from `transformer.zig` + `ane.zig` | No — behaviour is absent, not untested |
| 1 × `aneGateHeadroom` | ANE is Apple hardware | No |
| 1 × `lanShareDenial` | needs a Lan that can share; stub never does | Only if Avahi lands |

Known cosmetic wart, **pre-existing and not a port regression**:
`/v1/models` reports `"quantization":"0-bit"` for GGUF models. `quant_bits` is
an affine-safetensors concept that is never set on the GGUF path (and
`gguf_meta.zig` does not parse the ggml file type at all), so macOS reports the
same. Fixing it means plumbing the GGML type through and changes macOS-visible
API output — worth doing deliberately, not as a drive-by.

The security-relevant half (`routeClass` × `SharedSet`) still runs everywhere —
that was the point of splitting `lan.zig`.

### 3.6 CI

`.github/workflows/ci.yml` is `runs-on: macos-26` only. A
`windows-latest` job would need: `scripts/fetch-zig.sh` (already works there),
`scripts/fetch-llama.sh` (already works there), `zig build -Dgguf-only`,
`zig build test`. No brew, no mlx build, no Xcode steps.

### 3.7 Things deliberately left broken

Each is named at its call site; do not "fix" one by making it silently succeed.

- **WebP input** — `lib/webp_stub/` returns NULL from `WebPDecodeRGB`, which the
  single call site already treats as "cannot decode". PNG/JPEG still work via
  stb. Lift by vendoring real libwebp.
- **LAN sharing** — `--lan-share` / `--lan-discover` are no-ops. Linux could use
  Avahi; Windows has no equivalent. The streaming proxy itself is portable, but
  without discovery there is nothing to proxy to.
- **Media generation, ds4, ANE, MTP/DFlash/PLD/drafter, MLX safetensors** — all
  refuse by name.

---

## 4. Environment

```bash
./scripts/fetch-zig.sh      # -> .zig-toolchain/zig.exe  (0.17.0-dev nightly, pinned)
./scripts/fetch-llama.sh    # -> lib/llama/{bin,include}  (b10472, CUDA 13.3)
.zig-toolchain/zig.exe build
```

Both staged trees are gitignored. `fetch-llama.sh` takes `LLAMA_CUDA_VER`
(default 13.3; upstream also ships 12.4 for x64).

Dev box this was built on: Windows 11, RTX 5060 Ti 16 GB (sm_120), CUDA 13.3,
WSL2 Ubuntu available but unused so far.

**Always `-Doptimize=ReleaseFast` for anything perf-shaped** (project rule:
Debug is 2–4× slower and produces fake regressions). Plain `zig build` is fine
for compile-error iteration.

---

## 5. Map of what was added

**Gating**
- `src/build_cfg.zig` — `ds4_enabled` / `llama_enabled` / `mlx_enabled` /
  `ane_enabled` / `media_gen_enabled`. Names capabilities, not flags — see §6.4.
- `build.zig` — `-Dgguf-only`; exe and test module setup collapsed into one
  `makeSharedModule` (they had already drifted).

**Split out of MLX files, shared by both builds** (§6.5 explains why sharing
rather than duplicating):
`model.zig`+`model_weights.zig`, `lan_policy.zig`+`lan_bonjour.zig`,
`gen_common.zig`, `kv_quant_config.zig`, `dflash_contract.zig`, `loop_detect.zig`

**Stubs** (things that genuinely do not exist):
`mlx_stub.zig`, `transformer_stub.zig`, `generate_stub.zig`, `gen_stub.zig`,
`spec_stub.zig`, `vision_stub.zig`, `mlx_cache_stub.zig`, `ane_stub.zig`,
`dsv4_stub.zig`, `diffusion_stub.zig`, `model_weights_stub.zig`,
`lan_transport_stub.zig`, `lib/ds4_metal_sources_stub.zig`, `lib/webp_stub/`

**Host layer**
- `src/platform.zig` — sleep, shutdown handling, `pollSocket`/`peerClosed`,
  `homeDir`, `setEnv`, `connectedPair`, `tmpDirPath`. Everything here is reached
  from threads carrying no `std.Io` handle; that is *why* it cannot use `std.Io`.
- `src/status_windows.zig` — a real implementation, not a stub. See §6.3.

---

## 6. Traps

### 6.1 lld links a DLL directly under `-gnu`, and refuses under `-msvc`

The llama.cpp Windows release ships DLLs with **no import libraries and no
headers**. Under the gnu ABI lld links the `.dll` itself; under msvc it errors
`bad file type. Did you specify a DLL instead of an import library?`. So
`build.zig` pins `x86_64-windows-gnu`. The boundary is `llama.h`, pure C, so the
ABI difference is not observable. Headers come from the source tarball at the
same tag.

`ggml` is a **separate DLL** on Windows (the macOS XCFramework merges
llama+ggml+ggml-metal into one dylib), hence `linkSystemLibrary("ggml-base")`.
The whole DLL set must ship beside the exe even though only two are linked —
ggml dlopens its backends **by filename** at init, so a missing `ggml-cuda.dll`
silently downgrades the backend instead of failing the link.

### 6.2 Winsock: the accept loop does not work the way POSIX's does

Two separate failures, found by instrumenting rather than guessing:

1. `WSAPoll` returns **WSANOTINITIALISED (10093)** unless `WSAStartup` ran in
   this process. `std.Io.net` starts Winsock lazily for its own sockets and had
   not yet. `platform.ensureWinsock()` fixes it.
2. Even then it returns **WSAENOTSOCK (10038)** on `std.Io.net`'s socket handle.

So **there is no 1-second poll timeout on Windows** — the accept loop blocks in
`accept()`. Shutdown works because the console-control handler makes a loopback
connection to the server's own port to unblock it; without that, Ctrl+C hangs,
because the handler claims the event and suppresses the default terminator.

The symptom while this was wrong is worth memorising: **clients sat in
CLOSE_WAIT and got no response**, because Windows completes the TCP handshake
from the backlog *before* the app calls `accept()`. The connection looks
established while the server has never seen it.

If you make `pollSocket` work on Windows properly, delete the loopback wakeup
and restore the timeout — but prove it with `/props` and a Ctrl+C, not by
reading the docs.

### 6.3 `status.zig`'s numbers are load-bearing

`getTotalMemBytes` / `getAvailableMemBytes` feed the **model-load admission gate
and the auto-context sizer**. A zero-returning stub does not blank a panel, it
makes every load decision on wrong numbers. `status_windows.zig` is therefore a
real implementation (GlobalMemoryStatusEx, K32GetProcessMemoryInfo,
GetSystemTimes) and Linux needs the same treatment, not a stub.

GPU% goes through NVML loaded at runtime via `LoadLibrary`, so a missing driver
cannot stop the server from starting. Init failure is permanent for the process
— do not retry it on a 500 ms timer.

### 6.4 `build_cfg` names capabilities because the two reductions cut opposite ways

`ios` = MLX only, both GGUF engines stubbed. `gguf_only` = the exact inverse,
llama.cpp is the whole floor. `if (opts.ios)` at a ds4 import reads as "ds4 is
off" even in the build where ds4 is the only thing on. Note `llama_enabled` is
deliberately NOT gated on `gguf_only`.

### 6.5 Split, don't stub — when the thing is pure

Six files were cleanly ordered pure-then-MLX. Each portable half is **shared**,
not duplicated, because each is load-bearing somewhere:

- `loop_detect.zig` was a genuine **behaviour bug**, not a test artifact. A
  llama.cpp generation loops exactly as readily as an MLX one; stubbing
  `degenerateTail` shipped a server with no loop-stop guard. Five scheduler
  tests went red→green when it was extracted.
- `lan_policy.zig` carries `routeClass` × `SharedSet` — the keyless security
  gate. Leaving it inside the Bonjour file would have made those 8 tests
  macOS-only.
- `dflash_contract.zig` — two copies of a contract are two contracts.

Apply the same test before adding any new stub: **is this pure? then share it.**

### 6.6 Zig test collection ignores comptime gating

`test { if (cond) _ = @import("x"); }` **still registers x's tests** when the
branch is dead. Collection walks the *reference* graph, not the post-analysis
one. The `if/else` expression form does not help either. The only thing that
works is selecting a different test **root file** from build.zig — hence
`src/tests.zig` (portable) vs `src/tests_all.zig` (adds MLX/ds4/ANE).

Corollary: **any MLX file transitively imported by a portable file gets its
tests compiled.** That is why three live tower-parity tests moved into
`*_parity_test.zig` files, and why `format_corpus_test.zig` could not keep its
`mtp.zig` import (it dragged transformer → deepseek_v4 → kv_quant → the vision
towers into the link).

If you see undefined `mlx_*` symbols in a build that compiled fine, this is why.
`zig build test 2>&1 | grep "test_zcu.obj:"` names the emitting function, which
is how to find the culprit.

### 6.8 Compiles + loads is not runs

Both bugs that blocked the first token had this shape, and neither was visible
by reading the diff:

- `Slot.init` builds a KVCache for **every** slot and asks for **zero layers**
  on an embedded-engine slot (ds4/llama own their KV). The stub refused instead
  of returning the shell, so every GGUF request failed 500 `MlxUnavailable`
  *after tokenizing successfully*. Zero layers allocate nothing — there is no
  MLX to be missing. **When adding a stub, ask what the caller passes on the
  GGUF path**, not just whether the feature exists.
- Six sites hand-rolled basename with `'/'`. A Windows path has none, so the
  whole path became the model id and `bytes_on_disk` came back null.

The general lesson: a stub that refuses is right for a feature that is absent,
and wrong for a shared code path that merely *degenerates* on this backend.

### 6.7 This Zig nightly

No `std.fs`, no `std.posix.open`, no `std.Thread.Mutex`, no
`std.crypto.random`, no `std.time.nanoTimestamp`, no `std.time.Timer`.
Everything file-shaped is `std.Io.*` and needs an `Io` handle;
`std.Io.Mutex` is the codebase idiom (17 uses). `std.c.open` cannot even be
*declared* under the Windows calling convention.

`mlx_stub.zig` **aliases the real types** (`const real = @import("mlx.zig")`)
and stubs only functions. A type is not a symbol — an extern is emitted only if
called — so this costs no link dependency, while parallel type declarations made
MLX-side modules see two incompatible `mlx_array` types.

---

## 7. Before this merges to main

**The macOS build has never been compiled since these changes.** `build.zig`,
`server.zig`, `main.zig`, `scheduler.zig`, `model.zig`, `generate.zig`,
`lan.zig`, `kv_quant.zig`, `dflash.zig`, `mtp.zig` and the three vision modules
were all edited. Everything is gated and macOS *should* be unchanged, but that
needs a real `zig build && zig build test` on a Mac.

Highest-risk spots for a macOS regression, in order:

1. The six file splits — every one moved declarations between files and added
   re-exports. `model.zig` and `kv_quant.zig` re-export into hot paths.
2. `build.zig`'s `makeSharedModule` — the exe and test modules had *drifted*
   (the test module carried a `linkSystemLibrary("c++")` the exe did not);
   unifying them is correct but changes the macOS link line.
3. The tests moved out of `format_corpus_test.zig`, `lfm2_vision.zig`,
   `muse_vision.zig` and `qwen_vision.zig` — they must still run on macOS via
   `tests_all.zig`, not silently vanish. Check the macOS test COUNT, not just
   that it is green.

The Swift app (`app/`) was not touched and is untested against any of this.
