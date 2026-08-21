# Windows / Linux port — state and remaining work

Branch: `testing/windows_linux`.

**§3 is what is left. §6 is the traps** — none of them are guessable. §6.9–§6.12
came out of serving real traffic and running the shell suite; §6.13–§6.14 came
out of building on a second host. Read §6 before touching anything; read §3 to
pick up work.

| | |
|---|---|
| §3.1 WSL packages | **blocked on the user** — one apt line, unblocks §3.2 + §3.3 |
| §3.2 Linux build | script written, refusals verified; **build never run** |
| §3.3 LAN transport | not started; interop contract **verified against a live Mac** |
| §3.4 Vision (mtmd) | not started |
| §3.5 Spec decode | not started |
| §3.6 CI | windows job added, **never run on a runner** |
| §7 macOS | **never compiled since this branch** — merge gate |

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

## 2. Verified working today (Windows)

`zig build` and `zig build test` both pass. **920 tests pass, 15 skipped, 0
fail** with `LLAMA_TEST_MODEL` + `LLAMA_TEST_EMBED_MODEL` pointed at real GGUFs
(915/20 with neither set). CUDA is live end to end (RTX 5060 Ti, sm_120).

Exercised against real models, non-streaming and streaming:

- `/v1/chat/completions` — tool calling, `continue_final_message`,
  `json_schema` grammar mode, 4-way concurrency, mid-stream cancellation
- `/v1/messages` (Anthropic / Claude Code) — SSE lifecycle + `tool_use` blocks
- `/v1/responses` — including the WebSocket upgrade and its error turns
- `/v1/completions`, `/api/chat`, `/api/generate`, `/api/tags`
- `/v1/embeddings` — GGUF embedders serve it; `tests/test_embeddings.sh` 10/10
  against nomic-embed
- shell suite: **54 pass / 88 skip / 11 fail** of 166 (`tests/run.sh <script>`);
  all 11 are missing prerequisites that fail the same on a Mac

Five real bugs were found this way, every one of which produced **plausible
output rather than an error** — the lessons are §6.8–§6.10:

| Bug | Symptom |
|---|---|
| Discarded `llama_memory_seq_rm` refusal | repeat requests echoed the prompt, then returned empty; multi-turn broken on every hybrid checkpoint (Qwen3.5/3.8) |
| `startsWith(id, "/")` as "is absolute" | register-by-path unreachable on Windows |
| `homeDir` ignored `HOME` | six test scripts read and WROTE the real `~/.mlx-serve` |
| `n_ctx_train` never asked | a 2048-window embedder accepted 8192-token inputs |
| pooling type never asked | an embedding model advertised `chat` and answered with garbage |

Plus `fetch-zig.sh` was downloading `zig.tar` and extracting `zig.tar.xz`, so
**macOS and Linux could not stage a toolchain at all** since commit `f947fb5`.
Windows was unaffected by luck (`zig.zip` matched). Fixed; guarded by
`tests/test_toolchain_fetch.sh`.

Two embedding contracts worth not re-breaking: results are **L2-normalized**
(the MLX encoder normalizes, and `dimensions` truncates-then-renormalizes
assuming unit input), and **encoder-only is detected from the declared pooling
type**, not `llama_model_has_encoder` — llama.cpp reserves that for T5-style
encoder-decoders and answers false for every BERT-family embedder.

---

## 3. What is left

### 3.1 Install the WSL packages — blocks §3.2 and §3.3

One line, needs a password so it cannot be automated:

```bash
wsl.exe -d Ubuntu -- sudo apt-get update && \
wsl.exe -d Ubuntu -- sudo apt-get install -y cmake g++ pkg-config \
    libavahi-compat-libdnssd-dev avahi-daemon
```

Already prepared and waiting on it:

- `~/mlx-serve-linux` — a separate Linux clone with the branch's working tree
  applied, and Zig staged and working. Separate on purpose: see §6.14.
- `tests/probe_avahi_dnssd.c` — the Avahi probe (§3.3), never compiled.

WSL2 Ubuntu 26.04 here has systemd and dbus, so avahi-daemon can actually run.

### 3.2 Linux

`scripts/build-llama-cuda.sh` exists and both its refusal paths are verified
(wrong host, missing cmake — each exits 1 and names the fix). **The build
itself has never run**, so the compile, the staging layout and the Zig-side
Linux link are all UNVERIFIED — `build.zig`'s `else` arm gained
`linkSystemLibrary("ggml")` + `("ggml-base")` and that line has never been fed
to a linker.

With cmake + g++ (§3.1) a CPU-only llama.cpp build into a scratch dir is enough
to compile-check the Zig side. A real CUDA build needs the toolkit (~4 GB).

Then: `status_linux.zig` does not exist. It cannot be a zero-returning stub —
those numbers feed the model-load admission gate and the auto-context sizer
(§6.3). The NVML code in `status_windows.zig` ports nearly verbatim; only
`LoadLibraryA` becomes `dlopen`.

### 3.3 LAN sharing — transport only, the contract is settled

`--lan-share` / `--lan-discover` are no-ops off Apple and say so at boot.
`lan_transport_stub.zig` is unchanged: `sharing()` false, `lookupRemote()`
`.peer_unknown`.

**Verified 2026-08-21 against a live macOS mlx-serve (main, `--lan-share` on),
from this Windows box.** This settles the design questions before any code:

1. **Apple's `dnssd.dll` is not needed.** `tests/probe_mdns_browse.py` — plain
   UDP multicast, no Bonjour, no zeroconf — queried `_mlxserve._tcp.local` and
   got the whole record set back. That is a working miniature of zig-ai's
   `mdns.zig`.
2. **`lan_policy.txtBuild` is byte-correct against a real peer:**
   `03 "v=1"  12 "t=053c24c3f2a2ec00"` — exactly `buf[0]=3`, `"v=1"`,
   `buf[4]=2+token.len`, `"t="`, token.
3. **The keyless gate already agrees across hosts.** `routeClass` × `SharedSet`
   lives in the portable `lan_policy.zig` and already runs on Windows. The
   Mac's peer-facing surface answers exactly what the source says: `/health`,
   `/v1/models`, `/api/version` → 200; `/props`, `/metrics`,
   `/v1/models/rescan` → **403**; `POST /v1/chat/completions` served. The
   security half is done and cross-checked, not remaining work.

So what is left is the **transport only**: advertise, browse, resolve, proxy
tunnel. Two routes, in this order:

1. **Linux may be nearly free** — `avahi-compat-libdns_sd` exposes the
   *identical* `DNSService*` C API, so `lan_bonjour.zig`'s FFI block may link
   unmodified. Prove it with `tests/probe_avahi_dnssd.c` (needs §3.1): it
   registers our real `SERVICE_TYPE` with a real TXT record, browses, resolves,
   and checks the TXT comes back **byte-identical**. Linking only proves the
   ABI exists; the two things that would silently sink the shortcut are
   `DNSServiceRefSockFD` (our whole event loop is a pollable fd — a stub
   returning −1 links fine and never delivers a callback) and the TXT
   round-trip (the self-token that makes proxy loops impossible rides in it).
2. **Windows needs the hand-rolled module** — port `../zig-ai`'s
   `src/server/mdns.zig` (970 lines, 8 tests) + `rawsock.zig` (641, 6). Read
   `../zig-ai/bonjour.md` first: it is a port plan of OUR `lan.zig` written
   against our line numbers. zig-ai copied our pure layer verbatim, so
   `lan_policy.zig` is the shared spec and **must not drift** — any change to
   the id form, TXT format or `X-MLX-*` headers breaks interop with zig-ai and
   with macOS. Change it in both or neither.

Already-paid-for knowledge to carry across:

- **Windows joins the multicast group on only ONE interface for `INADDR_ANY`**,
  and on a box with Hyper-V / WSL / VPN adapters that is frequently the wrong
  one. Join every up interface explicitly.
- DNS name compression must be **parsed** (peers emit it), not emitted.
- Keep the two-tier failure counters (`PEER_DROP_FAILS` 3 / `KNOWN_MAX_FAILS`
  24). One counter reintroduces a real production bug: a single transient
  resolve hiccup evicting a live peer.
- **The gate lands WITH sharing, never after.**

Acceptance test is interop both ways: this build using the Mac's model, and the
Mac using ours. Note this link is cross-subnet (Windows 192.168.0.150, Mac
192.168.2.61, router reflects mDNS) — so it cannot prove the same-subnet-only
case. WSL2's NAT'd networking means the Avahi probe answers "does the API
work", never "does mDNS reach the LAN from WSL"; that needs a real Linux box.

### 3.4 Vision via mtmd

Not started. `mtmd.h` / `mtmd-helper.h` are staged and `mtmd.dll` ships in the
set. mtmd does its OWN preprocessing, so it does not go through
`qwen_vision`/`muse_vision`/`lfm2_vision` and does not resurrect
`vision_stub.zig`.

### 3.5 Draft-model speculative decode

Not started. llama.cpp has its own; partly recovers what was lost with
MTP/DFlash. Shim work plus scheduler wiring. Measure before claiming anything
(`/bench` rules apply).

### 3.6 CI

`.github/workflows/ci.yml` has a `build-test-windows` job (fetch-zig,
fetch-llama, ReleaseFast build, `--version`/`--help` smoke, `zig build test`,
`tests/test_portable_env.sh`). No brew, no mlx, no Xcode. **Never run on a
runner** — the first PR is the test. The shell suite is not run there; it needs
models.

### 3.7 Known issues, deliberately not fixed

- **`"quantization":"0-bit"` for GGUF models** in `/v1/models`. `quant_bits` is
  an affine-safetensors concept never set on the GGUF path, so macOS reports
  the same. Fixing it means plumbing the GGML type through and changes
  macOS-visible API output — do it deliberately, not as a drive-by.
- **11 shell scripts exit 1 on a missing prerequisite** instead of skipping.
  Pre-existing inconsistency, shared with macOS, not a port defect.
- **15 skipped Zig tests**: 2 HF-cache symlink (Windows needs Developer Mode),
  7 `prefillMemoryNeeded`/`resolvePrefillChunk` and 1 `aneGateHeadroom` (MLX/ANE
  behaviour is absent, not untested), 1 `lanShareDenial` (needs a Lan that can
  share — revisit when §3.3 lands).
- **Both hosts stage into the same `.zig-toolchain`** and fight over it (§6.14).
  Use separate clones; making the paths host-specific changes what CI, the docs
  and CLAUDE.md all reference.

### 3.8 Things deliberately left broken

Each is named at its call site; do not "fix" one by making it silently succeed.

- **WebP input** — `lib/webp_stub/` returns NULL from `WebPDecodeRGB`, which the
  single call site already treats as "cannot decode". PNG/JPEG still work via
  stb. Lift by vendoring real libwebp.
- **Media generation, ds4, ANE, MTP/DFlash/PLD/drafter, MLX safetensors** — all
  refuse by name.

---

## 4. Environment

```bash
./scripts/fetch-zig.sh      # -> .zig-toolchain/  (0.17.0-dev nightly, pinned)
./scripts/fetch-llama.sh    # -> lib/llama/{bin,include}  (b10472, CUDA 13.3)
.zig-toolchain/zig.exe build -Doptimize=ReleaseFast
```

Both staged trees are gitignored. `fetch-llama.sh` takes `LLAMA_CUDA_VER`
(default 13.3; upstream also ships 12.4 for x64).

**Run shell tests through the wrapper**, which installs the host shims (§6.10):

```bash
./tests/run.sh test_multi_model_dir.sh
```

Model-gated Zig tests read these at RUN time (see the caching trap in §6.11):

```
LLAMA_TEST_MODEL         a chat .gguf        — prefix reuse, decode, tokenize
LLAMA_TEST_EMBED_MODEL   an embedding .gguf  — /v1/embeddings engine contract
MLX_LLAMA_BACKEND_DIR    lib/llama/bin       — build.zig sets it for `zig build test`
```

Dev box: Windows 11, RTX 5060 Ti 16 GB (sm_120), CUDA 13.3; WSL2 Ubuntu 26.04
with systemd, GPU visible, toolchain incomplete (§3.1).

**Which host to work from.** Not interchangeable, and the split is measured:

| Task | Host | Why |
|---|---|---|
| §3.2 Linux build, `status_linux.zig` | **WSL** | native POSIX shell; the whole §6.10 environment tax disappears and `tests/` runs unmodified |
| §3.3 LAN transport + acceptance | **Windows** | WSL2's default NAT puts it on `172.27.x` behind the host while the LAN is `192.168.x`. The mDNS prober that finds the Mac from Windows returns NOTHING from WSL. `networkingMode=mirrored` in `.wslconfig` would lift this; untested here |
| §3.3 Avahi API probe | either | it asks a question about the local daemon, not about the LAN |
| Anything Windows-shaped | **Windows** | §6.2, §6.10 and §6.11 only exist on Windows, and Windows is a target rather than merely the host |

If you run an agent inside WSL: it needs `node` + the CLI installed (neither is
present), and it must NOT be pointed at the `/mnt/c` checkout — see §6.14.
`~/mlx-serve-linux` already has the branch applied with Zig staged.

**Always `-Doptimize=ReleaseFast` for anything perf-shaped** (project rule:
Debug is 2–4× slower and produces fake regressions). Plain `zig build` is fine
for compile-error iteration.

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
  `homeDir`, `setEnv`, `connectedPair`, `tmpDirPath`, `looksAbsolutePath`,
  `trimTrailingSeps`. Everything here is reached from threads carrying no
  `std.Io` handle; that is *why* it cannot use `std.Io`.
- `src/status_windows.zig` — a real implementation, not a stub. See §6.3.

**Added while widening the surface**
- `tests/lib/portable_env.sh` + `tests/run.sh` — the host shim layer, guarded
  by `tests/test_portable_env.sh`. See §6.10.
- `scripts/build-llama-cuda.sh` — Linux CUDA build (§3.2, never run).
- Shim: `mlx_llama_embed_session_create` / `_session_embed` / `_n_embd` /
  `_is_encoder_only` / `_n_ctx_train`, plus explicit `ggml_backend_load_all`.
- `tests/probe_avahi_dnssd.c`, `tests/probe_mdns_browse.py` — §3.3 probes.
- `.github/workflows/ci.yml` — the `build-test-windows` job.

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
llama+ggml+ggml-metal into one dylib). The whole DLL set must ship beside the
exe even though only a few are linked — ggml dlopens its backends **by
filename** at init, so a missing `ggml-cuda.dll` silently downgrades the backend
instead of failing the link.

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

### 6.9 A return value nobody reads is a contract nobody keeps

The three worst bugs found by actually serving traffic were all a discarded or
never-asked-for answer, and every one produced PLAUSIBLE OUTPUT rather than an
error:

- `llama_memory_seq_rm` returns false when it cannot partially trim. The shim
  ignored it, with a comment asserting the case could not happen. It happens on
  every recurrent/hybrid checkpoint — which is what Qwen3.5/3.8 are — and the
  result was a repeat request answering from the wrong position. The existing
  "prefix reuse is byte-identical" test missed it because it only ever EXTENDS a
  prompt, so the back-off-and-trim branch never runs; the new test re-serves an
  IDENTICAL prompt over a generated tail.
- `llama_model_n_ctx_train` was never asked, so the stub config's 8192 guess
  stood. A 2048-window embedder accepted inputs four times too long.
- The model's pooling type was never asked, so an embedding model advertised
  `chat` and answered chat requests with garbage.

When wrapping a C API, the question is not "did this call succeed" but "what is
this function telling me that I am throwing away". `grep` the shim for calls
whose result is unused; each one is a claim you have not checked.

### 6.10 The suite's environment is a bigger port surface than the suite

Not one of the 166 shell scripts needed rewriting for Windows. What needed
fixing was the environment they assume, and each gap fails SILENTLY:

- `python3` exists, is executable, and does not run (Microsoft Store alias,
  exit 49). `command -v` finds it, so all 645 call sites returned an EMPTY
  string, which reads as the server having returned nothing.
  `test_multi_model_dir` reported "single-dir case returned []" with a perfectly
  good server behind it. The resolver's contract is "does it RUN".
- POSIX paths reach a native binary through argv but not through a request body.
- `script(1)` does not exist, so a pty transcript is empty and the assertion
  "no bad lines in the transcript" passes vacuously.
- `HOME` is not what the server reads, so an isolated fixture is not isolated.
- A model can be a FILE (`.gguf`), not only a directory.
- `-d "$body"` overflows `ARG_MAX` far sooner than on macOS.

Two rules fall out. Run scripts through `tests/run.sh`, and when a check can
pass because its fixture did NOTHING, assert the fixture did something first.

### 6.11 `zig build test` used to require killing your server

The test step depended on `b.getInstallStep()` purely so the DLLs would be
staged, which reinstalls `mlx-serve.exe` — and Windows holds an open image
locked, so the whole suite failed `AccessDenied` whenever a server was running
out of `zig-out/bin`. That is exactly when you want to run it. It now points
PATH at the staged `lib/llama/bin` and depends on nothing.

The related trap that is NOT fixed: `MLX_LLAMA_BACKEND_DIR` and
`LLAMA_TEST_MODEL` are read at RUN time, but `zig build test` caches the run
step and will report `cached` without re-running when only the environment
changed. Invoke the test binary directly (`ls -t .zig-cache/o/*/test.exe |
head -1`) when varying model env vars.

### 6.12 Loading a DLL and registering a backend are two different things

`llama_backend_init()` does not register ggml's compute backends; ggml dlopens
them BY FILENAME and its own auto-load searches the RUNNING EXECUTABLE's
directory. That is right for the installed server (the whole set ships beside
`mlx-serve.exe`) and empty for a `zig build test` binary under `.zig-cache`, so
every model-gated llama test died `no backends are loaded` no matter what was
on PATH. The shim now calls `ggml_backend_load_all()` explicitly, with
`MLX_LLAMA_BACKEND_DIR` overriding the directory (build.zig sets it for the
test run). Those symbols live in `ggml.dll`, not `ggml-base.dll` — hence the
extra `linkSystemLibrary("ggml")`.

### 6.13 The host you are working on is the one that cannot warn you

`fetch-zig.sh` downloaded to `"$TMP/zig.$KIND"` and extracted
`"$TMP/zig.tar.xz"`. On macOS and Linux `KIND` is `tar`, so the two never named
the same file and **neither host could stage a toolchain from a clean tree**.
Windows was fine because its `KIND` spells `zip` and matched by accident — so
the only host anyone was testing on was the only one that could not see the
break, and it survived a whole port.

`tests/test_toolchain_fetch.sh` did not catch it either: it tested the pure
resolver (does it pick the right asset name), while the defect was a *coupling*
between two lines further down. It now checks that the download target and every
extractor input are the same expression, in both directions. When you refactor a
name, the guard belongs on the thing that has to agree, not on the thing you
renamed.

### 6.14 Do not share one checkout between Windows and WSL

Both hosts stage into the same `.zig-toolchain`, and `fetch-zig.sh` does
`rm -rf "$DEST"` before extracting. Its idempotency check keys on `zig.exe` vs
`zig`, so each host sees the other's toolchain as wrong, wipes it and
re-downloads. `lib/llama` is less violent (its staged-check is layout-aware, so
it merely refetches) but still ping-pongs. Use a separate Linux clone — the
staging trees are gitignored anyway, and WSL over `/mnt/c` is slow regardless.

---

## 7. Before this merges to main

**The macOS build has never been compiled since these changes.** Everything is
gated and macOS *should* be unchanged, but that needs a real
`zig build && zig build test` on a Mac. Note §6.13: until that fix, a clean Mac
tree could not even stage its toolchain, so "should be fine" was never testable.

Highest-risk spots, in order:

1. The six file splits (pre-existing) — `model.zig` and `kv_quant.zig`
   re-export into hot paths.
2. `build.zig`'s `makeSharedModule` (pre-existing) — the exe and test modules
   had drifted; unifying them changes the macOS link line.
3. Tests moved out of `format_corpus_test.zig` and the three vision modules —
   check the macOS test COUNT, not just that it is green.
4. The llama shim now calls `ggml_backend_load_all*`, fenced `#ifndef
   __APPLE__` — on Apple the XCFramework merges llama + ggml + ggml-metal into
   one dylib with its backends compiled IN, so there is nothing to dlopen and,
   crucially, nothing new to link. Only Windows and Linux gained `ggml`. If the
   macOS link breaks, look here first.
5. `platform.homeDir` now prefers an explicit `HOME` on every host. A no-op on
   macOS (the POSIX branch always read HOME) but the code moved, so the
   `homeDir` tests are the check.
6. `runEmbedRequest` gained a llama arm ahead of the MLX one, keyed on
   `llama_engine != null`. An MLX model never enters it.
7. `tests/run.sh` and `tests/lib/portable_env.sh` are additive; on macOS
   `run.sh` is a passthrough. Three scripts gained skip guards keyed on
   `uname -s` / the build — verify they still RUN on a Mac rather than skipping
   (a skip reads as a pass).

The Swift app (`app/`) was not touched and is untested against any of this.
