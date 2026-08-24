#!/usr/bin/env bash
# Build llama.cpp from source on macOS with Metal AND the ggml RPC backend, and
# stage it into lib/llama in the SEPARATED layout (libllama + libggml* dylibs),
# matching what scripts/build-llama-cuda.sh produces on Linux.
#
# WHY THIS EXISTS: upstream's macOS release asset is an XCFramework -- one
# merged dylib with Metal compiled in and NO ggml-rpc backend (nm-confirmed
# 2026-08-24: `_llama_supports_rpc` present, no `ggml_backend_rpc_*`). A Mac
# that wants to be a --rpc consumer (split a model with an NVIDIA box) or a
# --rpc-serve worker needs this build. scripts/fetch-llama.sh keeps staging the
# XCFramework for everyone else; the two are told apart by build.zig probing
# for lib/llama/lib/libggml-base.dylib at configure time.
#
# Usage:
#   ./scripts/build-llama-macos.sh
#
# Environment:
#   LLAMA_TAG   llama.cpp tag (default: the pin in fetch-llama.sh)
#   JOBS        parallel build jobs (default: sysctl hw.ncpu)
#   FORCE=1     rebuild even when lib/llama is already at "<tag>-rpc"
#
# Produces:
#   lib/llama/lib/libllama.dylib + libggml*.dylib (incl. libggml-rpc.dylib,
#   libggml-metal.dylib) + libmtmd.dylib, install names @rpath/<name>,
#   ad-hoc signed; lib/llama/include/*.h; lib/llama/.version = "<tag>-rpc".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
FETCH_LIB_ONLY=1 . "$SCRIPT_DIR/fetch-llama.sh"

DEST="$REPO_ROOT/lib/llama"
DEST_LIB="$DEST/lib"
DEST_INC="$DEST/include"
STAMP="$DEST/.version"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
WANT="${LLAMA_TAG}-rpc"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "[build-llama-macos] ERROR: this script is for macOS hosts (Linux: build-llama-cuda.sh)." >&2
  exit 1
fi

if [ "${FORCE:-0}" != "1" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] \
   && [ -f "$DEST_LIB/libllama.dylib" ] && [ -f "$DEST_LIB/libggml-rpc.dylib" ] && [ -f "$DEST_INC/llama.h" ]; then
  echo "[build-llama-macos] lib/llama already at $WANT -- nothing to do (FORCE=1 to rebuild)"
  exit 0
fi

for tool in cmake git xcrun; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "[build-llama-macos] ERROR: '$tool' not found on PATH (brew install cmake; xcode-select --install)." >&2
    exit 1
  }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SRC="$TMP/llama.cpp"
echo "[build-llama-macos] fetching llama.cpp $LLAMA_TAG"
git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$SRC"

# @rpath install names + no build-dir rpath baked in: the consumer (build.zig,
# app/build.sh, release.yml) supplies the rpath, exactly as for the XCFramework
# dylib. GGML_METAL_EMBED_LIBRARY so the metallib rides inside libggml-metal
# and nothing has to be staged beside it.
echo "[build-llama-macos] configuring (Metal + RPC)"
cmake -S "$SRC" -B "$TMP/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_NAME_DIR='@rpath' \
  -DCMAKE_MACOSX_RPATH=ON \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_MTMD=ON \
  -DLLAMA_CURL=OFF

echo "[build-llama-macos] building with $JOBS jobs"
cmake --build "$TMP/build" --config Release -j "$JOBS"

# Same refusal class as the CUDA script: a build that did not produce the
# backend must not be staged as if it had. Both are dlopen'd by filename and a
# missing one silently downgrades at runtime.
for need in libggml-rpc.dylib libggml-metal.dylib; do
  find "$TMP/build" -name "$need" | grep -q . || {
    echo "[build-llama-macos] ERROR: the build produced no $need." >&2
    exit 1
  }
done

echo "[build-llama-macos] staging into $DEST"
rm -rf "$DEST_LIB" "$DEST_INC"
mkdir -p "$DEST_LIB" "$DEST_INC"
find "$TMP/build" \( -name 'libllama*.dylib' -o -name 'libggml*.dylib' -o -name 'libmtmd*.dylib' \) \
  -exec cp -P {} "$DEST_LIB/" \;
[ -f "$DEST_LIB/libllama.dylib" ] || { echo "[build-llama-macos] ERROR: no libllama.dylib staged." >&2; exit 1; }

# Normalise install names to @rpath/<name> (CMake honours INSTALL_NAME_DIR for
# most targets; belt and braces for any that kept an absolute path) and re-sign
# ad-hoc, exactly as fetch-llama.sh does for the XCFramework dylib.
for f in "$DEST_LIB"/*.dylib; do
  [ -L "$f" ] && continue
  base="$(basename "$f")"
  install_name_tool -id "@rpath/$base" "$f"
  # Rewrite any absolute dependency on a sibling to @rpath so the set is relocatable.
  otool -L "$f" | awk 'NR>1 {print $1}' | grep -E '/libggml|/libllama' | grep -v '^@rpath' | while read -r dep; do
    install_name_tool -change "$dep" "@rpath/$(basename "$dep")" "$f"
  done
  codesign --remove-signature "$f" 2>/dev/null || true
  codesign --force --sign - "$f"
done

cp "$SRC/include"/*.h "$DEST_INC/"
cp "$SRC/ggml/include"/*.h "$DEST_INC/"
[ -f "$SRC/tools/mtmd/mtmd.h" ] && cp "$SRC/tools/mtmd/mtmd.h" "$DEST_INC/"
[ -f "$SRC/tools/mtmd/mtmd-helper.h" ] && cp "$SRC/tools/mtmd/mtmd-helper.h" "$DEST_INC/"

echo "$WANT" > "$STAMP"
echo "[build-llama-macos] staged ($WANT):"
ls -1 "$DEST_LIB"
echo "  $(ls "$DEST_INC" | wc -l | tr -d ' ') headers in $DEST_INC"
echo "  NOTE: rm -rf .zig-cache before the next zig build (configure-time probes are cached)."
