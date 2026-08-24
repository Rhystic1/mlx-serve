#!/usr/bin/env bash
# Fetch llama.cpp's prebuilt inference library (NOT llama-server) and stage it
# for linking into the mlx-serve Zig binary.
#
# Per host:
#   macOS   — the XCFramework ships libllama as a single self-contained dylib
#             (llama + ggml + ggml-metal merged, Metal shaders embedded). We
#             thin it to arm64, rewrite its install-name to @rpath, and drop the
#             headers next to it.
#   Windows — the prebuilt win-cuda zip ships DLLs ONLY: no headers and no
#             import libraries. So the headers come from the SOURCE tarball at
#             the same tag, and the import libs are generated from the DLLs.
#             This is the CUDA build; ggml-cuda.dll carries the backend.
#   Linux   — llama.cpp publishes NO prebuilt CUDA asset (only cpu, vulkan,
#             sycl, openvino), so Linux must build from source. Silently taking
#             the cpu tarball would ship a build that reports CUDA support it
#             does not have.
#
# This is the single source of truth for the pinned llama.cpp version.
# Bump LLAMA_TAG to upgrade; CI and local builds re-fetch automatically.
#
# `FETCH_LIB_ONLY=1 . scripts/fetch-llama.sh` defines the resolver and returns
# without touching the network — see tests/test_toolchain_fetch.sh.

# `set -e` in a SOURCED script leaks into the caller's shell and would abort a
# test harness on the first intentional failure probe. Only arm it when this
# script is actually being executed.
if [ -z "${FETCH_LIB_ONLY:-}" ]; then set -euo pipefail; fi

LLAMA_TAG="${LLAMA_TAG:-b10472}"
# Which prebuilt CUDA flavour the Windows asset name selects. llama.cpp ships
# 12.4 and 13.3 for x64; 13.3 matches the CUDA toolkit this port is developed
# against. The cudart-* companion zip uses the same version string.
LLAMA_CUDA_VER="${LLAMA_CUDA_VER:-13.3}"

# llama_asset_plan <uname-s> <uname-m> <tag> -> "<asset>|<kind>|<layout>"
#
# `asset` is empty when there is nothing to download (Linux builds from source).
# `layout` states what the staging step must PRODUCE, because the three hosts
# differ in what the upstream artifact actually contains:
#   dylib                  — self-contained lib + headers, ready to link
#   dll+srcheaders         — DLLs only; fetch headers from source. No import
#                            libs: lld links a DLL directly under the gnu ABI.
#   so+srcheaders          — nothing prebuilt; compile, fetch headers
llama_asset_plan() {
  local uname_s="$1" uname_m="$2" tag="$3" arch
  case "$uname_m" in
    arm64|aarch64) arch="arm64" ;;
    x86_64|amd64)  arch="x64" ;;
    *) echo "[fetch-llama] ERROR: unsupported arch $uname_m" >&2; return 1 ;;
  esac
  case "$uname_s" in
    Darwin)
      echo "llama-${tag}-xcframework.zip|xcframework|dylib" ;;
    MINGW*|MSYS*|CYGWIN*)
      if [ "$arch" != "x64" ]; then
        echo "[fetch-llama] ERROR: unsupported Windows arch $uname_m" >&2; return 1
      fi
      echo "llama-${tag}-bin-win-cuda-${LLAMA_CUDA_VER}-x64.zip|win-cuda|dll+srcheaders" ;;
    Linux)
      echo "|build-from-source|so+srcheaders" ;;
    *) echo "[fetch-llama] ERROR: unsupported OS $uname_s" >&2; return 1 ;;
  esac
}

# The CUDA runtime that ggml-cuda.dll dlopens, published as a SEPARATE release
# asset from the main win-cuda zip and versioned in lockstep with it.
llama_cudart_asset() {
  echo "cudart-llama-bin-win-cuda-${LLAMA_CUDA_VER}-x64.zip"
}

# The runtime DLLs that must sit beside ggml-cuda.dll. ggml resolves them by
# FILENAME at init and, when they are absent, SILENTLY FALLS BACK TO THE CPU —
# no error, no missing-DLL dialog, just a build that reports CUDA support and
# prefills at CPU speed. Staging is therefore verified, not assumed.
llama_cuda_runtime_dlls() {
  printf '%s
' cudart64_13.dll cublas64_13.dll cublasLt64_13.dll
}

# Names any required runtime DLL missing from <dir>, one per line; silent when
# the stage is complete. Factored out of the win-cuda branch so the check that
# prevents a silent CPU build is itself testable without a 700 MB download.
llama_missing_cuda_runtime() {
  local dir="$1" dll
  for dll in $(llama_cuda_runtime_dlls); do
    [ -f "$dir/$dll" ] || echo "$dll"
  done
}

# Sourced for its helpers only (tests): stop here, before any network or
# filesystem work.
if [ -n "${FETCH_LIB_ONLY:-}" ]; then return 0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/lib/llama"
DEST_LIB="$DEST/lib"
DEST_INC="$DEST/include"
DEST_BIN="$DEST/bin"
STAMP="$DEST/.version"

RESOLVED="$(llama_asset_plan "$(uname -s)" "$(uname -m)" "$LLAMA_TAG")"
ASSET="$(echo "$RESOLVED" | cut -d'|' -f1)"
KIND="$(echo "$RESOLVED" | cut -d'|' -f2)"
LAYOUT="$(echo "$RESOLVED" | cut -d'|' -f3)"

# Idempotent: skip when the staged copy already matches the pinned tag. What
# counts as "staged" is per-layout — the macOS dylib and the Windows DLL set are
# different files, so a shared check would false-positive across a host switch.
staged_ok() {
  [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$LLAMA_TAG" ] || return 1
  [ -f "$DEST_INC/llama.h" ] || return 1
  case "$LAYOUT" in
    dylib)  [ -f "$DEST_LIB/libllama.dylib" ] ;;
    dll*)   [ -f "$DEST_BIN/llama.dll" ] ;;
    so*)    [ -f "$DEST_LIB/libllama.so" ] ;;
    *) return 1 ;;
  esac
}
if staged_ok; then
  echo "[fetch-llama] lib/llama already at $LLAMA_TAG — nothing to do"
  exit 0
fi
[ -f "$STAMP" ] && echo "[fetch-llama] staged version '$(cat "$STAMP")' != '$LLAMA_TAG' — refetching"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ── headers from the SOURCE tarball ────────────────────────────────────────
# Used by every host that does not get headers with its binary artifact. We
# take the whole include dirs rather than a hand-listed set of files: llama.h
# pulls in the ggml header chain (ggml.h, ggml-alloc.h, ggml-backend.h,
# ggml-cpu.h, ggml-opt.h, gguf.h), and a list would silently rot the next time
# upstream adds one.
fetch_src_headers() {
  local url="https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_TAG}.tar.gz"
  echo "[fetch-llama] downloading headers from $url"
  curl -fSL --retry 3 -o "$TMP/src.tar.gz" "$url"
  tar xzf "$TMP/src.tar.gz" -C "$TMP"
  local src="$TMP/llama.cpp-${LLAMA_TAG}"
  [ -d "$src/include" ] || { echo "[fetch-llama] ERROR: no include/ in source tarball" >&2; exit 1; }
  cp "$src/include"/*.h "$DEST_INC/"
  cp "$src/ggml/include"/*.h "$DEST_INC/"
  # mtmd (multimodal) lives under tools/, not include/. It is what keeps vision
  # available on the GGUF path, and mtmd.dll ships in the same binary zip.
  [ -f "$src/tools/mtmd/mtmd.h" ] && cp "$src/tools/mtmd/mtmd.h" "$DEST_INC/"
  [ -f "$src/tools/mtmd/mtmd-helper.h" ] && cp "$src/tools/mtmd/mtmd-helper.h" "$DEST_INC/"
}

case "$KIND" in

# ── macOS: XCFramework -> single self-contained dylib ──────────────────────
xcframework)
  URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ASSET}"
  echo "[fetch-llama] downloading $URL"
  curl -fSL --retry 3 -o "$TMP/xcf.zip" "$URL"

  echo "[fetch-llama] extracting macOS slice"
  unzip -q "$TMP/xcf.zip" -d "$TMP/xcf"

  FW="$(find "$TMP/xcf" -type d -path '*macos-arm64*/llama.framework' | head -1)"
  if [ -z "$FW" ]; then
    echo "[fetch-llama] ERROR: no macos-arm64 llama.framework in $ASSET" >&2
    exit 1
  fi
  FW_BIN="$FW/Versions/A/llama"
  FW_HEADERS="$FW/Versions/A/Headers"

  rm -rf "$DEST_LIB" "$DEST_INC"
  mkdir -p "$DEST_LIB" "$DEST_INC"

  # Thin the universal framework binary to arm64 (falls back to a copy if it is
  # already single-arch), then expose it as a conventionally-named dylib.
  if lipo -archs "$FW_BIN" 2>/dev/null | grep -q x86_64; then
    lipo -thin arm64 "$FW_BIN" -output "$DEST_LIB/libllama.dylib"
  else
    cp "$FW_BIN" "$DEST_LIB/libllama.dylib"
  fi

  # Rewrite the framework-style install-name to a plain @rpath dylib so the
  # linker and our bundle-time install_name_tool rewrites (release.yml /
  # build.sh) can treat it like any other bundled dylib.
  install_name_tool -id "@rpath/libllama.dylib" "$DEST_LIB/libllama.dylib"

  # Re-sign ad-hoc: install_name_tool invalidates the signature, and dyld refuses
  # to load an arm64 dylib with a stale signature. Bundle steps re-sign with the
  # Developer ID later.
  codesign --remove-signature "$DEST_LIB/libllama.dylib" 2>/dev/null || true
  codesign --force --sign - "$DEST_LIB/libllama.dylib"

  cp "$FW_HEADERS"/*.h "$DEST_INC/"

  echo "$LLAMA_TAG" > "$STAMP"
  echo "[fetch-llama] staged libllama ($LLAMA_TAG):"
  echo "  $DEST_LIB/libllama.dylib ($(du -h "$DEST_LIB/libllama.dylib" | cut -f1))"
  echo "  $(ls "$DEST_INC" | wc -l | tr -d ' ') headers in $DEST_INC"
  ;;

# ── Windows: prebuilt CUDA zip -> DLLs + generated import libs + src headers ─
win-cuda)
  URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ASSET}"
  echo "[fetch-llama] downloading $URL"
  curl -fSL --retry 3 -o "$TMP/win.zip" "$URL"

  echo "[fetch-llama] extracting"
  mkdir -p "$TMP/win"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$TMP/win.zip" -d "$TMP/win"
  else
    powershell.exe -NoProfile -Command \
      "Expand-Archive -Force -LiteralPath '$(cygpath -w "$TMP/win.zip")' -DestinationPath '$(cygpath -w "$TMP/win")'"
  fi

  rm -rf "$DEST_LIB" "$DEST_INC" "$DEST_BIN"
  mkdir -p "$DEST_LIB" "$DEST_INC" "$DEST_BIN"

  # The zip is flat (or one level deep depending on the release). Take every
  # DLL wherever it landed. The whole set is required at RUNTIME even though we
  # only link llama + mtmd: ggml.dll dlopens its backends (ggml-cuda.dll and the
  # per-microarch ggml-cpu-*.dll variants) by filename at init.
  find "$TMP/win" -name '*.dll' -exec cp {} "$DEST_BIN/" \;
  if [ ! -f "$DEST_BIN/llama.dll" ]; then
    echo "[fetch-llama] ERROR: no llama.dll in $ASSET" >&2
    exit 1
  fi

  # ── CUDA runtime companion ────────────────────────────────────────────
  # A separate asset, and the ONLY thing standing between "links against CUDA"
  # and "actually runs on the GPU": ggml dlopens these by filename and silently
  # continues on the CPU when they are missing.
  CUDART_ASSET="$(llama_cudart_asset)"
  CUDART_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${CUDART_ASSET}"
  echo "[fetch-llama] downloading $CUDART_URL"
  curl -fSL --retry 3 -o "$TMP/cudart.zip" "$CUDART_URL"

  mkdir -p "$TMP/cudart"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$TMP/cudart.zip" -d "$TMP/cudart"
  else
    powershell.exe -NoProfile -Command       "Expand-Archive -Force -LiteralPath '$(cygpath -w "$TMP/cudart.zip")' -DestinationPath '$(cygpath -w "$TMP/cudart")'"
  fi
  find "$TMP/cudart" -name '*.dll' -exec cp {} "$DEST_BIN/" \;

  # VERIFY, do not assume. A missing runtime DLL has no symptom at build time
  # and no symptom at boot — only a prefill that runs on the CPU — so this is
  # the one place it can still be caught by name.
  MISSING="$(llama_missing_cuda_runtime "$DEST_BIN" | tr '
' ' ')"
  if [ -n "${MISSING// /}" ]; then
    echo "[fetch-llama] ERROR: $CUDART_ASSET did not provide: $MISSING" >&2
    echo "  ggml-cuda.dll dlopens these by filename; without them ggml falls" >&2
    echo "  back to the CPU SILENTLY and the build only looks CUDA-enabled." >&2
    exit 1
  fi

  fetch_src_headers

  # No import libraries are generated: lld links directly against a DLL under
  # the gnu ABI (it refuses under msvc, which is why build.zig pins
  # x86_64-windows-gnu on Windows). $DEST_BIN doubles as the link-time library
  # path; build.zig also installs these DLLs beside the exe, since Windows
  # resolves them at load time from the executable's own directory.

  echo "$LLAMA_TAG" > "$STAMP"
  echo "[fetch-llama] staged llama.cpp ($LLAMA_TAG, CUDA $LLAMA_CUDA_VER):"
  echo "  $(ls "$DEST_BIN"/*.dll | wc -l | tr -d ' ') DLLs in $DEST_BIN"
  echo "  $(ls "$DEST_INC" | wc -l | tr -d ' ') headers in $DEST_INC"
  echo "  CUDA runtime staged from $CUDART_ASSET:"
  echo "    $(llama_cuda_runtime_dlls | tr '
' ' ')"
  echo ""
  echo "  IMPORTANT: clear the Zig caches before the next build —"
  echo "    rm -rf .zig-cache \"\${LOCALAPPDATA:-\$HOME/.cache}/zig\""
  echo "  build.zig enumerates lib/llama/bin at CONFIGURE time and Zig caches"
  echo "  that build graph, so DLLs staged after an earlier build are NEVER"
  echo "  copied beside mlx-serve.exe — and a missing CUDA runtime downgrades"
  echo "  the backend SILENTLY. Same class as the stale-link-path note in"
  echo "  CLAUDE.md (Building)."
  ;;

# ── Linux: no prebuilt CUDA asset upstream ─────────────────────────────────
build-from-source)
  echo "[fetch-llama] ERROR: llama.cpp publishes no prebuilt Linux CUDA binaries." >&2
  echo "  Upstream ships only cpu / vulkan / sycl / openvino for Linux, and a" >&2
  echo "  cpu build would report CUDA support it does not have." >&2
  echo "  Build from source instead:" >&2
  echo "    scripts/build-llama-cuda.sh" >&2
  echo "  (needs cmake + the CUDA toolkit; it refuses to stage a build that did" >&2
  echo "   not actually produce libggml-cuda.so)" >&2
  exit 1
  ;;

*)
  echo "[fetch-llama] ERROR: unknown plan kind '$KIND'" >&2
  exit 1
  ;;
esac
