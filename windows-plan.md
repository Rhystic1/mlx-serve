# Windows / Linux port — state and remaining work

Branch: `testing/windows_linux`. Read this whole file before touching anything.

The traps in **§6** are the ones that cost the most time, and none of them are
guessable. §6.9-§6.15 came out of actually serving traffic and running the
shell suite; they are the ones to read first if you are picking this up.

Status at a glance:

| | |
|---|---|
| §3.1 HTTP surface | **done** on Windows AND Linux — 3 bugs found and fixed |
| §3.2 shell suite | **triaged** on both hosts, host shims added |
| §3.3 Linux | **builds, tests and serves** — 4 more bugs found; CUDA compile still unverified |
| §3.4 embeddings | **done**, verified on both hosts; vision + spec-decode not started |
| §3.5 LAN / Bonjour | **works on Linux** (23/23 integration) — hand-rolled mDNS; Mac interop + Windows untested |
| §3.7 CI | windows + linux jobs added, **neither run on a runner** |

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

`zig build` and `zig build test` pass on **both Windows and Linux**.

Windows: **920 tests pass, 15 skipped, 0 fail** with `LLAMA_TEST_MODEL` +
`LLAMA_TEST_EMBED_MODEL` pointed at real GGUFs (915/20 with neither set).

Linux (WSL2 Ubuntu 26.04, 2026-08-21): **926 pass, 13 skipped, 0 fail** with
both models set. The counts differ for two understood reasons and neither is a
regression: `status_linux.zig` adds 4 tests of its own, and the two HF-cache
symlink tests that Windows skips (they need Developer Mode) simply RUN here.

```
$ mlx-serve.exe --version
mlx-serve 26.8.10
mlx unknown                       <- honestly absent
nax unavailable (built without MLX)
ggml 0.20.1 (60eeeb608)
llama.cpp b10472
```

CUDA is live end to end **on Windows** (RTX 5060 Ti, sm_120, CUDA 13.3). The
Linux stage is CPU-only so far — see §3.3 for exactly what that does and does
not prove.

**The whole text HTTP surface has now been exercised against a real model**
(`Qwen3.8-27B-UD-IQ3_XXS`, 11.9 GB IQ3_XXS), non-streaming and streaming:

- `/v1/chat/completions` — incl. tool calling, `continue_final_message`,
  `json_schema` grammar mode, 4-way concurrency, mid-stream cancellation
- `/v1/messages` (Anthropic / Claude Code) — SSE lifecycle + `tool_use` blocks
- `/v1/responses` — incl. the WebSocket upgrade and its error turns
- `/v1/completions`, `/api/chat`, `/api/generate`, `/api/tags`
- `/v1/embeddings` — see §3.4; GGUF embedders now serve it

The same surface has now been re-exercised on **Linux** against
`LFM2.5-2.6B-heretic-Q4_K_M` (a hybrid gated-conv checkpoint — deliberately the
recurrent class that bug 1 below broke) and `nomic-embed-text-v1.5` for
embeddings: chat non-streaming and streaming with `include_usage`, `/v1/messages`,
`/v1/completions`, `/api/tags`, `/v1/embeddings`, and the repeat-request
sequence that first exposed the prefix-cache corruption. All correct, with
`[llama] partial trim refused (recurrent memory); cold prefill` visible in the
log where it should be.

Finding these took serving a real model and reading the responses. Two of the
three bugs below produce plausible output rather than an error, and none was
visible in the diff — see §6.8 and §6.9.

## 3. Remaining work, in priority order

### 3.1 Widen the surface that has actually run  — DONE

Every endpoint listed in §2 has now been exercised against a live GGUF model.
Three bugs came out of it; all three are fixed and covered.

1. **The prefix cache silently corrupted every repeat request** (the big one).
   `llama_memory_seq_rm` returns false when a partial trim is impossible, and
   the shim discarded that value on the belief that removing a whole tail never
   fails. That holds for a KV cache and is FALSE for a recurrent one: a hybrid
   checkpoint (GatedDeltaNet, Mamba, RWKV — i.e. Qwen3.5/3.8, the flagship) has
   no history to rewind, so the refusal left the old state in place while `pos`
   said otherwise. Live, three identical requests returned the right answer,
   then the PROMPT echoed back, then an empty completion with
   `finish_reason: "stop"`. Multi-turn chat was unusable on exactly the models
   people run. `trim` now reports the refusal and `sync` cold-prefills.
   Straight continuations are pure prefix extensions, never trim, and still
   reuse (measured 22 of 40 tokens on turn 2).

   The existing "prefix reuse is byte-identical" test did not catch it: it only
   ever EXTENDS a prompt, so `common < resident.len` and the back-off branch
   never runs. The new test re-serves an IDENTICAL prompt over a generated tail.

2. **Register-by-path was unreachable** — `startsWith(id, "/")` is a POSIX-only
   spelling of "is this an absolute path", so `C:/...` fell through to
   "Unknown model id". Now `platform.looksAbsolutePath`.

3. **A faked `HOME` was ignored** — `platform.homeDir` read only `USERPROFILE`
   on Windows, so the six test scripts that isolate state with
   `HOME=$(mktemp -d)` were reading and WRITING the developer's real
   `~/.mlx-serve`. An explicit `HOME` now wins on every host.

Smaller things fixed while here: the five `[discovery] skip` log lines had no
trailing newline; the open-bind warning said "this Mac"; `--version`'s llama
line claimed "backend: Metal".

Not investigated: `/v1/models` reports `"quantization":"0-bit"` for GGUF — see
§3.6, unchanged and shared with macOS.

### 3.2 Run the shell integration suite — TRIAGED on both hosts

Windows: **54 pass, 88 skip, 11 fail** across the 166 scripts (`tests/run.sh` runs one).
Every one of the 11 is a missing PREREQUISITE that would fail identically on a
Mac without it — no MLX model on disk, no server on the port the script
expects, no numpy, no llmprobe bundle. They exit 1 rather than skipping, which
is a pre-existing inconsistency in those scripts and not a port defect; the
scripts that skip cleanly are the majority.

None of the 11 is a Windows behaviour difference. The four that WERE are fixed
or gated below.

The suite needed a HOST SHIM layer, not 140 edits:

- `tests/lib/portable_env.sh` — sourced helpers, guarded by
  `tests/test_portable_env.sh` (9 hermetic checks).
- `tests/run.sh` — run any script with the shims installed. **Use it on
  Windows.** A plain passthrough on macOS.

What it fixes, and why each is invisible rather than loud:

- **`python3` is the Microsoft Store stub.** It is on PATH, `command -v` finds
  it, and it exits 49 without running anything — so all 645 `python3` call
  sites returned an EMPTY string, which reads as the server having returned
  nothing. `test_multi_model_dir` reported "single-dir case returned []" with a
  perfectly good server behind it; it passes 7/7 through the shim. The
  resolver's contract is "does it RUN", never "is it on PATH".
- **A path in a REQUEST BODY needs the host's own form.** The shell rewrites
  POSIX paths into argv (which is why `--model-dir /tmp/x` works) but not
  inside a curl payload, so `{"model":"/tmp/x"}` arrives verbatim and 404s.
  `mlxserve_host_path` (identity off Windows).
- **`mlxserve_has_mlx`** keys on the BUILD, not the host, so a Mac configured
  with `-Dgguf-only` skips the same tests.

Gated by name rather than deleted: `test_mlx_staged_nax` (MLX is Apple-only),
`test_connection_thread_reaping` (the measurement IS `vmmap`),
`test_media_eviction_gate` (media generation needs MLX).

`test_run_quiet` was passing VACUOUSLY: Git Bash has no `script(1)`, so the pty
transcript was empty and "no skip lines in it" was true because there were no
lines. It now asserts the transcript is non-empty and skips by name without a
pty. Worth assuming other scripts have the same shape.

`test_embeddings` had two portability bugs of its own — a model was assumed to
be a DIRECTORY (a `.gguf` is a file), and the request body went on the command
line, which overflows `ARG_MAX` once the over-limit fixture scales with a real
window. Both fixed; it now passes 10/10 against a GGUF embedder.

**Linux sweep (2026-08-21): 18 pass, 124 skip, 12 fail across 154 `test_*.sh`.**
Do not read the pass/skip split against the Windows row — it is dominated by
what is on that disk, not by the host: 91 of the skips want an MLX checkpoint
under `~/.mlx-serve/models` that a `-Dgguf-only` build could not load anyway,
and 8 more want a server already listening on 8080. All 12 failures are missing
prerequisites that would fail identically on a Mac without them (no MLX model,
no `numpy`, no `jq`, no llmprobe bundle, no TestServer on 8090, no vision model)
— the same finding as the Windows row, and again none is a Linux behaviour
difference. The shims needed nothing new: `python3` works here, so `run.sh` is
the plain passthrough it is on macOS.

Two of the 14 original failures were real defects in the SCRIPTS, and both are
fixed:

- **`test_run_quiet` was failing for a reason the Windows fix created the
  ability to see.** `script(1)` is two different programs: BSD/macOS takes the
  command as trailing argv (`script -q FILE CMD ...`), util-linux requires
  `-c "CMD"` and treats the trailing form as an error — rc 1, and NO transcript
  written. So the same invocation that works on a Mac produced an empty
  transcript here, and the "produced a transcript at all" assertion added for
  Windows (§6.10) is precisely what turned that into a legible failure instead
  of a vacuous pass. It now detects the FLAVOUR (`script --version`, not the OS
  — brew's util-linux on a Mac is the same program) and passes 3/3, which is
  the first time this test has actually exercised a pty on any host but macOS.
- **`test_flux_lowmem` reported its own clean SKIP as a failure.** Its EXIT trap
  dereferences `$SRV_PID`, and it is installed BEFORE the server starts — so
  under `set -u` every early exit dies in the trap, `exit 0` and all. A scan for
  the class found one more latent instance (`test_hybrid_reuse_equivalence`);
  both now use `${VAR:-}`. A third hit was a false positive.

One triaged finding that is NOT a defect: with a server up, `test_anthropic_api`
is 34/36, and both failures are `usage.cache_read_input_tokens == 0` on a repeat
prompt. That is the CORRECT answer on a recurrent checkpoint — §3.1's bug-1 fix
cold-prefills there rather than reusing state it cannot rewind — so the test
encodes a transformer-KV assumption as a universal one (the "an integration
assertion about what a MODEL does is a checkpoint expectation" class). Left
alone deliberately: the branch it needs is on engine internals HTTP cannot see,
which is a design change, not a triage fix. Worth knowing that BOTH chat GGUFs
in reach are recurrent (LFM2.5, and Qwen3.5/3.8 are GatedDeltaNet), so this was
not checked against a plain-transformer GGUF on either host.

### 3.3 Linux — BUILDS, TESTS AND SERVES; the CUDA compile is still unproven

Done 2026-08-21 under WSL2 Ubuntu 26.04 (kernel 6.18, 16 GB, RTX 5060 Ti
visible via `nvidia-smi`). `zig build -Doptimize=ReleaseFast` links, the binary
runs, `zig build test` is **926 pass / 13 skip / 0 fail**, and the server loads
a GGUF and answers on every text endpoint (§2).

`scripts/build-llama-cuda.sh` grew `LLAMA_CPU_ONLY=1` to make that possible on a
box with no CUDA toolkit. It is an opt-in, not a fallback — nothing degrades to
CPU on its own — and it stays honest three ways: it stamps `<tag>-cpu` rather
than the tag, which does NOT satisfy the script's own "already staged" check, so
a later real CUDA run rebuilds instead of skipping; it warns at the end of every
run; and the stamp is what `--version` prints, so a CPU stage announces itself:

```
$ ./zig-out/bin/mlx-serve --version
mlx-serve 26.8.10
nax unavailable (built without MLX)
ggml 0.20.1 (60eeeb6)
llama.cpp b10472-cpu          <- the stage says so
```

**Four bugs came out of actually running it, and the first one means the
"script written, never run" status was hiding a script that could not work:**

1. **The configure was wrong at the pinned tag.** `-DLLAMA_BUILD_SERVER=OFF`
   does not disable llama.cpp's unified `app` target, which hard-links against
   `-lllama-server-impl` and `-lllama-cli-impl`; the build died at 100%, after
   ~6 minutes, at the very last link. Fixed by naming what we actually want:
   `LLAMA_BUILD_TOOLS=OFF`, `LLAMA_BUILD_APP=OFF`, and `LLAMA_BUILD_MTMD=ON` —
   which builds `tools/mtmd` STANDALONE, so we get `libmtmd.so` (parity with the
   Windows asset, which already ships `mtmd.dll`) without the tools tree. This
   is the "a script that has never run is a draft" case: every refusal path in
   it had been tested, and the happy path had not.

2. **The staged shared objects were loadable only by accident** (§6.13). CMake
   bakes the BUILD directory into each `.so`'s RUNPATH, and this script deletes
   that directory on exit. It did not show up as a missing library at link time
   because ELF `RUNPATH` is not inherited by transitive dependencies: the exe's
   rpath found `libllama`/`libggml`/`libggml-base`, and then the loader went
   looking for libggml's OWN `libggml-cpu.so.0` using libggml's runpath, which
   pointed at `/tmp/tmp.QjOYhy5hlP/build/bin`. Fixed by staging with
   `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'`.

3. **`MLX_LLAMA_BACKEND_DIR` was Windows-gated and Windows-spelled** (§6.12, one
   host over). build.zig set it only under `if (os.tag == .windows)` and only as
   `lib/llama/bin`; a CMake build stages to `lib/llama/lib`. The ELF rpath gets
   the libraries LOADED, which is what hides it — the failure would have been
   `no backends are loaded` on the first model-gated test, not at startup.

4. **`platform.setEnv`'s POSIX arm did not compile** (§6.14). It called
   `std.c.setenv`, and this nightly declares neither `setenv` nor `unsetenv` in
   `std.c` on ANY target. Windows takes the other arm and macOS had not been
   rebuilt, so nothing caught it. There was also a dead
   `extern fn posixSetenvShim` declaration next to it naming a symbol that
   exists nowhere. Both replaced with direct `extern "c"` declarations.

`src/status_linux.zig` now exists — a real implementation, not a stub, for the
reason §6.3 gives: these numbers feed the model-load admission gate and the
auto-context sizer. `MemTotal`/`MemAvailable` from `/proc/meminfo` (NOT
`sysinfo(2)`, whose `freeram` counts the page cache as used — the #45 error one
kernel over), `VmRSS`/`VmSwap` for the footprint, `/proc/stat` tick deltas for
CPU, and NVML by `dlopen("libnvidia-ml.so.1")` for GPU%. `getProcAvailableMemBytes`
returns cgroup-v2 headroom when a ceiling actually applies and 0 otherwise,
because `scheduler.effectiveAvailableBytes` lets a nonzero answer REPLACE the
host figure — under a container limit the host's MemAvailable is a number the
process can never reach. Verified live: the status bar reports real values
(`RSS: 2.7G  Mem: 15%  CPU: 9%  GPU: 0%`) while serving.

**What is still NOT verified, and do not let the green build imply otherwise:**

- **The CUDA compile.** No nvcc on this box, so `GGML_CUDA=ON`, the
  `CMAKE_CUDA_ARCHITECTURES` handling, the `libggml-cuda.so` assertion and GPU
  inference have all still never run. Ubuntu's own `nvidia-cuda-toolkit` is
  **CUDA 12.4 and cannot target sm_120** (Blackwell), so a Linux CUDA build on
  this class of GPU needs NVIDIA's own repo or runfile, not the distro package.
  What the CPU stage does prove is everything downstream of it: the Zig link
  line, the rpath, backend registration, and the whole server.
- **The toolchain used here was assembled rootless** (sudo needs a password on
  this box): cmake from Kitware's binary tarball, and `g++-15` +
  `libstdc++-15-dev` + `g++-15-x86-64-linux-gnu` via `apt-get download` and
  `dpkg-deb -x` into `~/.local/toolchain`, with the system's
  `/usr/lib/gcc/...` and `/usr/libexec/gcc/...` entries symlinked in (a
  relocated GCC finds `cc1plus` relative to the driver, but not the sibling
  `libgcc-15-dev` headers or `liblto_plugin.so`, which live in the system tree).
  It compiles correctly, but it is not the toolchain a normal Linux user has, so
  the build's *portability* is proven less than its correctness.
- GPU-side perf, and therefore every number in §2, is Windows-only.

**Do not share one checkout between Windows and WSL.** Both hosts stage into
the same `.zig-toolchain`, and `fetch-zig.sh` does `rm -rf "$DEST"` before
extracting: its idempotency check keys on `zig.exe` vs `zig`, so each host sees
the other's toolchain as "wrong", wipes it and re-downloads. `lib/llama` is
less violent (its staged-check is layout-aware, so it merely refetches) but
still ping-pongs. Use a separate Linux clone — this work was done in one
(`~/mlx-serve-linux`) and that is the reason it was uneventful; the staging
trees are gitignored anyway. Verified 2026-08-21, not fixed — making the paths
host-specific changes what CI, the docs and CLAUDE.md all reference, and that is
not a change to make on a guess.

### 3.4 Claw back the three features the user asked for

- **Embeddings — DONE.** `/v1/embeddings` serves GGUF embedding models.
  The shim gained its own embedding context (`embeddings` and `pooling_type`
  are fixed at context creation, so it can never be a mode of a generation
  session), the scheduler an arm that runs it on the inference thread like
  every other engine call, and the server tokenizes through the ENGINE on this
  path — `lm.tokenizer` is the decode-only stub with an empty vocab, so
  encoding through it would have produced a confident, wrong vector rather than
  an error.

  Two contract details that are not optional:
  - **The result is L2-normalized.** The MLX encoder normalizes every row and
    the `dimensions` option truncates-then-RE-normalizes assuming unit input.
    llama.cpp pools without normalizing (nomic-embed measures ~22), so the same
    server would otherwise hand out two incomparable kinds of embedding.
  - **An encoder-only checkpoint must not advertise `chat`.** It is detected
    from the POOLING type the model declares, probed with a 32-token context.
    NOT `llama_model_has_encoder`, which llama.cpp reserves for true
    encoder-decoder archs (T5) and which answers false for every BERT-family
    embedder. Setting `config.is_encoder_only` drives the capability list, the
    "use /v1/embeddings instead" 400, and `has_embedding` from one field.

  While here: the GGUF stub config's context was a hardcoded 8192 guess. It is
  now corrected DOWN to the model's trained window when `--ctx-size` was not
  given (nomic-embed trains at 2048, so the server was accepting embedding
  inputs four times longer than the checkpoint has ever seen). Only downward —
  that value also sizes the libllama KV allocation, so adopting a 262144-token
  window because a model mentions one turns a working load into an OOM.

  Verified: `tests/test_embeddings.sh` 10/10 against nomic-embed
  (`EMBED_TEST_MODEL=<path to .gguf>`), plus a model-gated unit test
  (`LLAMA_TEST_EMBED_MODEL`) pinning determinism, input-dependence, unit norm,
  and — the real regression risk — that embedding B then A again reproduces A,
  which only holds because the shim clears memory per call.

  Re-verified end to end on **Linux** 2026-08-21 against the same checkpoint:
  768-dim vectors with L2 norms of exactly 1.0, cos 1.0 between identical
  inputs and 0.41 against a different sentence, `capabilities: ["embeddings"]`
  with no `chat`, the named 400 on `/v1/chat/completions`, and the trained-window
  correction logged (`[llama] context: 2048 (model's trained window, below the
  8192 default)`). One inconsistency it exposes is in §3.6, not here.

- **Vision via mtmd** — not started. `mtmd.h` / `mtmd-helper.h` are staged and
  the library ships in the set on BOTH hosts (`mtmd.dll` on Windows;
  `libmtmd.so` on Linux, which is why `build-llama-cuda.sh` passes
  `LLAMA_BUILD_MTMD=ON` — see §3.3). mtmd does its OWN preprocessing, so it does
  not go through `qwen_vision`/`muse_vision`/`lfm2_vision` and does not
  resurrect `vision_stub.zig`.
- **Draft-model speculative decode** — not started. Needs shim work plus
  scheduler wiring; measure before claiming anything (`/bench` rules apply).

### 3.5 LAN sharing / Bonjour — DONE off Apple; Mac interop still untested

`--lan-share` / `--lan-discover` work on Linux (and are built for Windows) over
a hand-rolled mDNS responder. The no-op stub is gone.

Live on WSL2 Ubuntu 26.04, 2026-08-21: two servers on one box, A sharing a GGUF
and B headless-discovering — B mirrored `<id>@node-a`, a chat naming that id
was proxied to A and answered (B serves an ENCODER-ONLY model, so the reply
could not have come from anywhere else), streaming passed SSE + `[DONE]`
through, and the keyless gate answered 403 on `/metrics` and `/v1/load-model`
from the machine's LAN address while `/health` and `/v1/models` stayed open.
`tests/test_lan_share.sh` — written on macOS, against dns_sd — passes **23/23**
against this transport unchanged.

#### What the decision was, and what it cost

> **One hand-rolled mDNS module for Windows AND Linux; macOS keeps dns_sd.
> Two transports, one policy layer — not three, and not one.**

The reasoning is unchanged from when this was a plan: **Windows needs the
module no matter what** (Apple's `dnssd.dll` ships with iTunes / Bonjour Print
Services and cannot be assumed present), so avahi never saved anyone the work —
it only added a THIRD transport covering one OS, and one that needs a running
`avahi-daemon` plus a root-installed D-Bus policy that a container image has
neither of. `tests/probe_avahi_dnssd.c` is what settled it: six of the seven
entry points we use exist, `DNSServiceGetAddrInfo` does not (not exported by
`libdns_sd.so.1`, not even declared in avahi-compat's `dns_sd.h`).

macOS keeps dns_sd because mDNSResponder owns and defends `<host>.local` there,
because sharing UDP 5353 on a BSD stack wants `SO_REUSEPORT` and not merely
`SO_REUSEADDR`, and because an app doing its own multicast is a different Local
Network privacy posture than delegating to the system daemon. None of that
applies off Apple, where nothing else is answering for our host.

#### What was built

```
lan_policy.zig          pure policy + wire spec (unchanged, shared, the interop contract)
lan_peers.zig      NEW  peer table + peer-model fetch + proxy tunnel — PORTABLE
lan_net.zig        NEW  UDP multicast + outbound TCP; POSIX arm moved verbatim
lan_mdns.zig       NEW  RFC 6762 codec + responder/browser, pure Zig, zero deps
lan_bonjour.zig         dns_sd DISCOVERY only, now 943 -> 567 lines
lan_transport_mdns.zig  NEW  discovery over lan_mdns, everywhere else
```

The split that mattered was `lan_peers.zig`: the peer table, the `/v1/models`
fetch and the tunnel were never Bonjour-specific, and a second copy of the
tunnel would have been a second contract. Only DISCOVERY is per-host now, plus
the retry bookkeeping its shape forces (dns_sd re-resolves by name+domain, mDNS
re-queries by name). Their five tests moved with them and **now run on every
host** rather than macOS only — which is also how it became clear they pass
there.

`lan_bonjour.zig` was edited without a Mac. That is less alarming than it
sounds: it compiles AND its tests run under `zig test` on Linux, because the
dns_sd externs are only referenced by functions the tests never call. That was
the check used throughout; it is not a substitute for a real macOS build (§7).

#### The gap list, closed

- **Name conflicts.** dns_sd renames a colliding instance for free; the
  hand-rolled module has no RFC 6762 §8 probing, so two servers on one box (or
  two same-named hosts) would both have answered for one name. `claimName` is a
  one-shot pre-flight — query, and if the name is held by someone whose TXT
  token differs, append `-2`, `-3`… Verified live: a third server launched with
  the SAME `--lan-name` logged `instance name 'node-a' is taken on this network`
  and came up as `node-a-2`, with the browser then listing both peers
  separately. This is deliberately NOT §8: no tie-breaking, no defence of the
  name afterwards. It can be replaced without changing callers.
- **Single announcement.** `announceBurst` sends three, 250 ms apart (§8.3),
  because nothing retransmits a lost first multicast.
- **The cadence lives in the caller**, as it did in zig-ai — `threadMain` owns
  the 10 s refresh, matching the Bonjour path's, which is what the two-tier
  failure counters are calibrated against.
- **IPv4 only**, which is parity with `kDNSServiceProtocol_IPv4` today.

One bug the unit tests could not have found, and the integration script did:
**a peer that dies without a goodbye was never dropped.** The sweep iterates
services it FOUND, and a powered-off peer appears in no sweep at all, so
nothing ever counted a failure against it. `ageUnseen` now counts a missed
sweep against every known service, which is what makes the grace period apply
to a crash as well as to a failed fetch. Regression test verified red on
revert.

#### Still not proven

- **Interop with a real macOS mlx-serve, in either direction.** Everything
  above is this transport talking to itself or to another copy of itself. Two
  responders agreeing proves the code is self-consistent, not that it speaks
  what mDNSResponder speaks. This is the acceptance test and it needs a Mac and
  this build on the same LAN.
- **Windows.** It compiles for it (the ws2_32 arm of `lan_net.zig` is written
  and the interface enumeration goes through `gethostname` + the resolver
  rather than iphlpapi) but has never run there. The per-interface group join
  exists precisely because Windows joins only ONE interface for `INADDR_ANY`
  and picks the wrong one on a box with Hyper-V/WSL/VPN adapters — untested.
- **WSL2 cannot host the Mac interop test**: `eth0` there is a NAT'd virtual
  network (172.27.190.247/20), not the host LAN. It needs a real Linux host, or
  WSL2 in mirrored networking mode.

Carry these across, they are already-paid-for knowledge:

- **Windows joins the multicast group on only ONE interface for `INADDR_ANY`.**
  Join every up interface explicitly (`Responder.init` does).
- DNS name compression must be **parsed** (peers emit it) but need not be
  **emitted**.
- Keep the two-tier failure counters (`PEER_DROP_FAILS` 3 / `KNOWN_MAX_FAILS`
  24). The comment on them records a real production bug: one transient resolve
  hiccup evicting a live peer, producing alternating success/404 on a peer that
  never went down. A single counter reintroduces it.
- **The gate lands WITH sharing, never after.** `routeClass` x `SharedSet` is
  the whole defence, it always ran on every host, and its server-level test
  (`lanShareDenial`) is no longer macOS-only — see §3.6.

**`lan_policy.zig` is the interop spec and must not drift.** zig-ai copied it
verbatim (same `SERVICE_TYPE`, byte-identical `txtBuild`), so any change to the
id form, the TXT layout or the `X-MLX-*` headers breaks interop with it and
with macOS mlx-serve. Change it in all of them or none.


### 3.6 Un-skip what deserves it

15 tests skip with both `LLAMA_TEST_*` models set (20 with neither). Most are pre-existing env-gated live tests. The ones this port
added, each with its reason stated in code:

| Skipped | Why | Worth revisiting? |
|---|---|---|
| 2 × HF-cache symlink tests | Windows needs Developer Mode to create symlinks | Only if CI enables it |
| 7 × `prefillMemoryNeeded` / `resolvePrefillChunk` | MLX billing math; the terms come from `transformer.zig` + `ane.zig` | No — behaviour is absent, not untested |
| 1 × `aneGateHeadroom` | ANE is Apple hardware | No |
| ~~1 × `lanShareDenial`~~ | ~~needs a Lan that can share; stub never does~~ | **DONE** — both transports share now, so it runs everywhere (§3.5) |

Known warts on the GGUF `/v1/models` row, **all pre-existing, shared with
macOS and Windows, and none a port regression**. They are grouped here because
they are one family — the stub config is built before the engine opens, and
whatever it guessed is what gets advertised as fact:

- `"quantization":"0-bit"`. `quant_bits` is an affine-safetensors concept never
  set on the GGUF path (and `gguf_meta.zig` does not parse the ggml file type at
  all).
- **`vocab_size`, `hidden_size` and `num_layers` are CONSTANTS, not the model's.**
  Measured 2026-08-21: a 2.6B LFM2.5 and a 137M nomic-bert both report
  `262208 / 3840 / 48`, while their GGUFs declare 12 layers and 768 embedding
  width for the latter. This is worse than the 0-bit wart — that one is
  obviously a placeholder, these read as real numbers.
- **`context_length` can contradict `model_max_tokens` in the same object.**
  nomic-embed advertises `context_length: 8192` / `max_model_len: 8192`
  alongside `model_max_tokens: 2048` and `embedding_max_length: 2048`. Both are
  true statements about different things — the served KV really is 8192, the
  model's trained window really is 2048 — but §3.4's down-correction lands on
  `config.max_position_embeddings` while the advertised twins come from
  `getEffectiveContextLength`, which returns `server_config.max_context_size`
  unconditionally when it is set. The practical harm is bounded here
  (`embedding_max_length` gates the only endpoint an encoder-only model serves),
  which is why it is recorded rather than fixed: the fix is either a clamp
  inside `getEffectiveContextLength` — macOS-visible, affects auto-context and
  every row — or a second field, and issue #188 is a reminder that this row's
  shape has downstream readers.

Fixing any of these means plumbing real values through and changing
macOS-visible API output — worth doing deliberately, not as a drive-by.

The security-relevant half (`routeClass` × `SharedSet`) still runs everywhere —
that was the point of splitting `lan.zig`.

### 3.7 CI — TWO JOBS ADDED (neither run on a runner)

`.github/workflows/ci.yml` gained a `build-test-windows` job: fetch-zig,
fetch-llama, `zig build -Doptimize=ReleaseFast`, a `--version`/`--help` smoke
test, `zig build test`, and `tests/test_portable_env.sh`. No brew, no mlx, no
Xcode. `defaults.run.shell` is `bash -eo pipefail {0}` (Git Bash ships on the
runner). The shell integration suite is NOT run there — it needs models.

It now also has a `build-test-linux` job on `ubuntu-latest`, same shape, with
one difference that matters: it builds libllama from source with
`LLAMA_CPU_ONLY=1`. A GitHub runner has no NVIDIA GPU and no CUDA toolkit, and
installing one would put a ~4 GB download in front of every push to compile
kernels the runner cannot execute. What the job covers is the part that
actually broke in §3.3 — the Zig link line, the ELF rpath, backend
registration, and the portable test suite. **It does not cover the CUDA
compile**, and the `-cpu` stamp means a green CI run can never be mistaken for
one that did.

Neither job has executed on a runner; the first PR will be the test.

### 3.8 Things deliberately left broken

Each is named at its call site; do not "fix" one by making it silently succeed.

- **WebP input** — `lib/webp_stub/` returns NULL from `WebPDecodeRGB`, which the
  single call site already treats as "cannot decode". PNG/JPEG still work via
  stb. Lift by vendoring real libwebp.
- **Media generation, ds4, ANE, MTP/DFlash/PLD/drafter, MLX safetensors** — all
  refuse by name.

---

## 4. Environment

Windows:

```bash
./scripts/fetch-zig.sh      # -> .zig-toolchain/zig.exe  (0.17.0-dev nightly, pinned)
./scripts/fetch-llama.sh    # -> lib/llama/{bin,include}  (b10472, CUDA 13.3)
.zig-toolchain/zig.exe build -Doptimize=ReleaseFast
```

Linux (upstream ships no prebuilt CUDA asset, so llama.cpp builds from source):

```bash
./scripts/fetch-zig.sh              # -> .zig-toolchain/zig
./scripts/build-llama-cuda.sh       # -> lib/llama/{lib,include}  (needs cmake, nvcc, g++)
LLAMA_CPU_ONLY=1 ./scripts/build-llama-cuda.sh   # no CUDA toolkit: CPU stage, stamps <tag>-cpu
.zig-toolchain/zig build -Doptimize=ReleaseFast
```

Both staged trees are gitignored. `fetch-llama.sh` takes `LLAMA_CUDA_VER`
(default 13.3; upstream also ships 12.4 for x64). On Linux the staged tree is
`lib/llama/lib` (shared objects), not `lib/llama/bin` (DLLs) — several things
key on that difference; see §6.12.

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

Dev boxes: Windows 11, RTX 5060 Ti 16 GB (sm_120), CUDA 13.3; and WSL2 Ubuntu
26.04 on the same machine (kernel 6.18, 16 GB) in a SEPARATE clone at
`~/mlx-serve-linux` — see the end of §3.3 for why that separation is not
optional.

**Always `-Doptimize=ReleaseFast` for anything perf-shaped** (project rule:
Debug is 2-4x slower and produces fake regressions). Plain `zig build` is fine
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
`lib/ds4_metal_sources_stub.zig`, `lib/webp_stub/`
(`lan_transport_stub.zig` was here too, and is gone — §3.5 replaced it with a
real transport rather than a better stub.)

**Host layer**
- `src/platform.zig` — sleep, shutdown handling, `pollSocket`/`peerClosed`,
  `homeDir`, `setEnv`, `connectedPair`, `tmpDirPath`. Everything here is reached
  from threads carrying no `std.Io` handle; that is *why* it cannot use `std.Io`.
- `src/status_windows.zig` / `src/status_linux.zig` — real implementations, not
  stubs. See §6.3. `status.zig` picks one and derives `delegate` FROM that
  choice rather than re-spelling the host list, so a newly delegated host
  cannot silently keep running the Mach bodies.

**Added while widening the surface (§3.1-§3.4)**
- `tests/lib/portable_env.sh` + `tests/run.sh` — the host shim layer, guarded
  by `tests/test_portable_env.sh`. See §6.10.
- `scripts/build-llama-cuda.sh` — Linux llama.cpp build (§3.3; the CPU-only
  path is now proven, the CUDA path still is not).
- `platform.looksAbsolutePath` / `platform.trimTrailingSeps` — host-aware
  path predicates; `homeDir` now honours an explicit `HOME` everywhere.
- Shim: `mlx_llama_embed_session_create` / `_session_embed` / `_n_embd` /
  `_is_encoder_only` / `_n_ctx_train`, plus explicit `ggml_backend_load_all`.
- `.github/workflows/ci.yml` — the `build-test-windows` and `build-test-linux` jobs.

**Added while bringing Linux up (§3.3, §3.5)**
- `src/status_linux.zig` — procfs + cgroup-v2 + NVML-by-dlopen.
- `scripts/build-llama-cuda.sh` — `LLAMA_CPU_ONLY=1`, `$ORIGIN` install rpath,
  and a configure that actually completes at the pinned tag.
- `tests/probe_avahi_dnssd.c` — the §3.5 measurement that ruled avahi out.
  Kept: the semantics half is still worth running on a box with an
  avahi-daemon, if anyone revisits the decision.

**The LAN transport (§3.5)** — `lan_net.zig`, `lan_mdns.zig`, `lan_peers.zig`,
`lan_transport_mdns.zig` added; `lan_transport_stub.zig` deleted;
`lan_bonjour.zig` reduced to discovery. `platform.randomBytes` is new
(`std.c.arc4random_buf` is Darwin + glibc-only, and the LAN identity token has
to be unguessable on Windows too). `tests/lib/portable_env.sh` gained
`mlxserve_lan_ip`, because `ipconfig getifaddr en0` was the only probe
`test_lan_share.sh` had and it skipped everywhere else as "offline Mac".

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
GetSystemTimes), and `status_linux.zig` now is too: `/proc/meminfo`,
`/proc/self/status`, `/proc/stat`.

Two Linux specifics worth not re-deriving. **`sysinfo(2)` is the wrong
syscall**: its `freeram` is FREE memory, counting the page cache as used, which
is the same error the macOS path exists to avoid (#45) one kernel over.
`MemAvailable` is the kernel's own estimate of what an allocation can have, and
it has the property #45 needs — a resident model's anonymous pages are not
reclaimable, so they count as used and a second large load is correctly
refused. And **`getProcAvailableMemBytes` must return 0 unless a cgroup ceiling
really applies**, because `scheduler.effectiveAvailableBytes` lets a nonzero
answer REPLACE the host figure; under a container limit that is exactly right
(the host's MemAvailable is a number the process can never reach), and outside
one it would be a fabricated ceiling. `memory.current` includes the cgroup's
page cache, so the reclaimable set is subtracted back out — the file-cache rule
again, one layer down.

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

### 6.9 A return value nobody reads is a contract nobody keeps

The three worst bugs found by actually serving traffic were all a discarded or
never-asked-for answer, and every one produced PLAUSIBLE OUTPUT rather than an
error:

- `llama_memory_seq_rm` returns false when it cannot partially trim. The shim
  ignored it, with a comment asserting the case could not happen. It happens on
  every recurrent/hybrid checkpoint — which is what Qwen3.5/3.8 are — and the
  result was a repeat request answering from the wrong position.
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

- `python3` exists, is executable, and does not run (Store alias, exit 49).
- POSIX paths reach a native binary through argv but not through a request body.
- `script(1)` does not exist, so a pty transcript is empty and the assertion
  "no bad lines in the transcript" passes.
- `script(1)` DOES exist on Linux and is a DIFFERENT PROGRAM: util-linux wants
  `-c "cmd"` and rejects the BSD/macOS trailing-argv form with rc 1 and no
  transcript at all. Same empty-transcript symptom, opposite cause — and it
  reads as "the server printed nothing", i.e. as a server bug. Detect the
  flavour (`script --version`), never the OS: brew's util-linux on a Mac is the
  same program.
- An EXIT trap that dereferences a pid installed BEFORE the server starts turns
  every early `exit 0` into a failure under `set -u` — so a script's own clean
  SKIP is reported as red.
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

**It bit a second time on Linux**, because the fix was written Windows-only in
two ways at once: the env var was set inside `if (os.tag == .windows)`, AND its
value was the Windows staging directory name. A CMake build stages shared
objects to `lib/llama/lib`, not `lib/llama/bin`. What makes this one nasty is
that the ELF rpath (§6.13) does get the libraries loaded, so nothing fails at
startup — the first symptom would be `no backends are loaded` from a
model-gated test, i.e. only for someone who had a GGUF on disk. It is now
`os.tag != .macos` with the directory chosen per host; macOS needs neither,
since its XCFramework has the backends compiled in and the shim's call is
fenced `#ifndef __APPLE__`.

### 6.13 An ELF RUNPATH is not inherited; a macOS @rpath effectively is

The Linux staged tree was loadable only by accident. Three facts compose:

1. CMake bakes the BUILD directory into every `.so`'s RUNPATH by default, and
   `build-llama-cuda.sh` deletes that directory on exit.
2. `RUNPATH` (what modern linkers emit, unlike the older `RPATH`) applies ONLY
   to the object's own `DT_NEEDED` entries — it is not searched on behalf of a
   dependency's dependency.
3. So the exe's rpath found `libllama`, `libggml` and `libggml-base`, and then
   the loader looked for libggml's own `libggml-cpu.so.0` using LIBGGML's
   runpath — a `/tmp/tmp.XXXX/build/bin` that no longer existed.

The failure is `cannot open shared object file` for a library sitting in the
same directory as the one that wants it, and it appears at EXEC time, after a
completely clean link. macOS does not have it (dyld resolves `@rpath` against
the whole load chain) and Windows does not have it (the DLL set ships beside
the exe). Fixed by staging with `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
-DCMAKE_INSTALL_RPATH='$ORIGIN'`, so each object finds its siblings beside
itself no matter who loaded it. `readelf -d <so> | grep RUNPATH` is the check.

### 6.14 A cross-platform arm nobody compiles is not a cross-platform arm

`platform.setEnv`'s POSIX branch called `std.c.setenv`, which does not exist in
this nightly on ANY target — `std.c` declares neither `setenv` nor `unsetenv`
(its only `*setenv` is OpenBSD's `auth_setenv`). It survived because Windows
takes the other branch and macOS had not been rebuilt since the split (§7). The
first Linux build found it in seconds.

Sitting beside it was `extern "c" fn posixSetenvShim(...)`, declared and never
called, naming a symbol that exists nowhere in the tree — an extern is only
emitted if it is CALLED, so a dead one links fine forever and reads as evidence
that someone solved this. Both are now direct `extern "c"` declarations of the
libc functions.

The general shape: `if (windows) A else B` compiles B nowhere until someone
builds a non-Windows target, and the same applies to every `else` arm added
during this port. Grep for them before trusting them.

### 6.15 A script that has never run is a draft, however carefully reviewed

`build-llama-cuda.sh` had every refusal path tested (wrong host, missing cmake,
missing nvcc, no `libggml-cuda.so`) and shipped as "written, never run". Its
happy path did not work: at the pinned tag `-DLLAMA_BUILD_SERVER=OFF` does not
disable llama.cpp's unified `app` target, which hard-links
`-lllama-server-impl` and `-lllama-cli-impl`, so the build died at 100% after
six minutes at the final link. Nothing about the script looked wrong, and the
refusals passing is what made it look finished.

The same review pass would not have found it, because the bug was in upstream's
option graph, not in the script's logic. Fixed by naming what we want
(`LLAMA_BUILD_TOOLS=OFF`, `LLAMA_BUILD_APP=OFF`, `LLAMA_BUILD_MTMD=ON` — mtmd
standalone, so `libmtmd.so` still gets staged for §3.4's vision work).

### 6.16 A sweep-driven browser cannot see what is not there

The hand-rolled mDNS browser folded each sweep's results into the peer table
and aged failures from them — which silently meant a peer that DIES (power off,
`kill -9`, a crash) was never dropped: it appears in no sweep, so nothing ever
counted a failure against it, and its models stayed listed forever. The
Bonjour path does not have this shape, because it re-attempts every KNOWN
service rather than every found one.

No unit test found it. It needed `tests/test_lan_share.sh`, which kills a peer
and waits for its models to disappear, and the failure then cascaded: with the
peer never dropped, the restart section ran late and every later gate assertion
got `000` (connection refused) against a server that was not up yet — eleven
red lines from one root cause, none of them about the actual bug.

The rule: an aging policy must iterate the set it is aging (`known`), never the
set it just observed. `ageUnseen` does, and the regression test pins it red on
revert.

### 6.17 A responder that only ever talked to itself has proven very little

The hand-rolled mDNS module has codec round-trips, hostile-packet tests, and a
two-responder discovery test that exercises the real socket path. All of that
passes, and none of it proves interop with Apple's mDNSResponder — two copies
of the same code agree with each other by construction. The acceptance test for
§3.5 is still a Mac and this build on one LAN, in both directions.

The same caution applies one level down: `tests/test_lan_share.sh` passing
23/23 is strong evidence, but it is our client talking to our server. It is
what caught the real bug above; it cannot catch a wire-format disagreement
where both ends are equally wrong.

### 6.7 This Zig nightly

No `std.fs`, no `std.posix.open`, no `std.Thread.Mutex`, no
`std.crypto.random`, no `std.time.nanoTimestamp`, no `std.time.Timer`, no
`std.c.setenv`/`unsetenv` (§6.14), no `std.mem.trimLeft`/`trimRight`
(`trimStart`/`trimEnd`), no `std.fmt.bufPrintZ` (`bufPrintSentinel(..., 0)`).
Everything file-shaped is `std.Io.*` and needs an `Io` handle;
`std.Io.Mutex` is the codebase idiom (17 uses). `std.c.open` cannot even be
*declared* under the Windows calling convention.

`mlx_stub.zig` **aliases the real types** (`const real = @import("mlx.zig")`)
and stubs only functions. A type is not a symbol — an extern is emitted only if
called — so this costs no link dependency, while parallel type declarations made
MLX-side modules see two incompatible `mlx_array` types.

---

## 7. Before this merges to main

**The macOS build has never been compiled since these changes.** Everything is
gated and macOS *should* be unchanged, but that needs a real
`zig build && zig build test` on a Mac. Highest-risk spots, in order:

1. The six file splits (pre-existing) — `model.zig` and `kv_quant.zig`
   re-export into hot paths.
2. `build.zig`'s `makeSharedModule` (pre-existing) — the exe and test modules
   had drifted; unifying them changes the macOS link line.
3. Tests moved out of `format_corpus_test.zig` and the three vision modules —
   check the macOS test COUNT, not just that it is green.
4. **New:** the llama shim now calls `ggml_backend_load_all*`. That is fenced
   `#ifndef __APPLE__` — on Apple the XCFramework merges llama + ggml +
   ggml-metal into one dylib with its backends compiled IN, so there is nothing
   to dlopen and, crucially, nothing new to link. Only the Windows and Linux
   link lines gained `ggml`. If the macOS link breaks, this is the first place
   to look.
5. **New:** `platform.homeDir` now prefers an explicit `HOME` on every host.
   That is a no-op on macOS (the POSIX branch always read HOME) but it moved
   the code, so the `homeDir` tests are the check.
6. **New:** `runEmbedRequest` gained a llama arm ahead of the MLX one, keyed on
   `llama_engine != null`. An MLX model never enters it.
7. **New:** `tests/run.sh` and `tests/lib/portable_env.sh` are additive; on
   macOS `run.sh` is a passthrough. Three scripts gained skip guards keyed on
   `uname -s` / the build — verify they still RUN on a Mac rather than skipping
   (a skip reads as a pass).
8. **New (Linux round):** `platform.setEnv`/`unsetEnv` now call `extern "c"`
   declarations instead of `std.c.*` (§6.14). macOS took the same POSIX arm
   before, so it is the same libc call — but it is the arm macOS actually
   compiles, so it is the one to watch.
9. **New (Linux round):** `status.zig`'s `impl` gained a `.linux` arm and
   `delegate` is now DERIVED from it (`impl != @This()`) rather than re-spelled
   as `os.tag == .windows`. On macOS `impl` is still `@This()` and `delegate` is
   still false, so every body it guards is unchanged — but confirm it folds the
   way it reads.
10. **New (Linux round):** `build.zig` sets `MLX_LLAMA_BACKEND_DIR` for the test
    run on every host except macOS (§6.12). macOS is excluded explicitly; if the
    Mac test run starts behaving differently, check that exclusion first.
11. **New (Linux round):** `test_run_quiet.sh` branches on the `script(1)`
    flavour, and `test_flux_lowmem.sh` / `test_hybrid_reuse_equivalence.sh` had
    their EXIT traps guarded with `${VAR:-}`. A Mac with brew's util-linux ahead
    of `/usr/bin` takes the LINUX branch in the first — correct, but a path the
    Mac has never run.
12. **New (LAN round), and the highest-risk item in this list:**
    `lan_bonjour.zig` lost ~380 lines to `lan_peers.zig` (peer table, model
    fetch, tunnel) and now calls through `lan_net.zig`. Its POSIX socket arm is
    the old code moved verbatim, and Linux exercises the same arm — but the
    file was edited without a Mac. It compiles and its (now-relocated) tests
    run under `zig test` on Linux because the dns_sd externs are never called;
    that is a real check, not a substitute for building it. Check
    `tests/test_lan_share.sh` on a Mac FIRST — it covers discovery, the tunnel,
    the gate and self-detection in one run.
13. **New (LAN round):** `lanShareDenial`'s test no longer skips off macOS and
    constructs `lan_mod.Lan` with `.table` instead of `.peers`. Both transports
    have that field; verify the Mac still compiles the literal.
14. **New (LAN round):** `platform.randomBytes` replaces the direct
    `std.c.arc4random_buf` call in the transport. macOS takes the same libc
    function, one call deeper.

The Swift app (`app/`) was not touched and is untested against any of this.
