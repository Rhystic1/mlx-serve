#!/usr/bin/env bash
# Build llama.cpp with CUDA from source and stage it into lib/llama, matching
# the layout scripts/fetch-llama.sh produces on the other hosts and
# build.zig's verifyLlamaStage expects.
#
# WHY THIS EXISTS: llama.cpp publishes prebuilt Linux binaries for cpu, vulkan,
# sycl and openvino only -- there is no Linux CUDA asset to download. Quietly
# staging the cpu tarball would give a server that reports CUDA support it does
# not have, which is worse than refusing, so fetch-llama.sh refuses Linux BY
# NAME and points here.
#
# Usage:
#   ./scripts/build-llama-cuda.sh
#
# Environment:
#   LLAMA_TAG        llama.cpp tag to build (default: the pin in fetch-llama.sh)
#   CMAKE_CUDA_ARCHITECTURES  GPU archs to compile for (default: native)
#   JOBS             parallel build jobs (default: nproc)
#   FORCE=1          rebuild even when lib/llama is already at the pinned tag
#   LLAMA_CPU_ONLY=1 build WITHOUT CUDA (see below) -- for proving the Zig-side
#                    Linux link on a box with no CUDA toolkit
#
# Produces:
#   lib/llama/lib/libllama.so  + libggml*.so (incl. libggml-cuda.so)
#   lib/llama/include/*.h
#   lib/llama/.version
#
# LLAMA_CPU_ONLY=1 exists so the Zig-side Linux link can be proven on a box
# with no CUDA toolkit, and it is deliberately NOT a fallback: nothing here
# ever degrades to CPU on its own, because a server reporting CUDA it does not
# have is the outcome this script's refusals exist to prevent. Two things keep
# the opt-in honest:
#   - it stamps "<tag>-cpu", which does NOT satisfy the "already staged" check
#     at the top, so a later real CUDA run rebuilds instead of being skipped;
#   - it says so loudly at the end, and again every time it is re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Single source of truth for the pin: read it out of fetch-llama.sh rather than
# repeating it here. Two copies of a version pin are two pins, and the one
# nobody edits is the one that ships.
# shellcheck disable=SC1091
FETCH_LIB_ONLY=1 . "$SCRIPT_DIR/fetch-llama.sh"

DEST="$REPO_ROOT/lib/llama"
DEST_LIB="$DEST/lib"
DEST_INC="$DEST/include"
STAMP="$DEST/.version"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-native}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build-llama-cuda] ERROR: this script is for Linux hosts." >&2
  echo "  macOS and Windows have prebuilt assets: ./scripts/fetch-llama.sh" >&2
  exit 1
fi

if [ "${FORCE:-0}" != "1" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$LLAMA_TAG" ] \
   && [ -f "$DEST_LIB/libllama.so" ] && [ -f "$DEST_INC/llama.h" ]; then
  echo "[build-llama-cuda] lib/llama already at $LLAMA_TAG -- nothing to do (FORCE=1 to rebuild)"
  exit 0
fi

# Refuse early and by name rather than failing 20 minutes into a compile.
if [ "${LLAMA_CPU_ONLY:-0}" = "1" ]; then
  REQUIRED_TOOLS="cmake git"
  echo "[build-llama-cuda] WARNING: LLAMA_CPU_ONLY=1 -- building WITHOUT CUDA." >&2
else
  REQUIRED_TOOLS="cmake nvcc git"
fi
for tool in $REQUIRED_TOOLS; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "[build-llama-cuda] ERROR: '$tool' not found on PATH." >&2
    case "$tool" in
      nvcc) echo "  Install the CUDA toolkit; nvcc is what makes this a CUDA build." >&2 ;;
      cmake) echo "  apt install cmake  (or your distro's equivalent)" >&2 ;;
      git) echo "  apt install git" >&2 ;;
    esac
    exit 1
  }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SRC="$TMP/llama.cpp"
echo "[build-llama-cuda] fetching llama.cpp $LLAMA_TAG"
# Shallow single-tag clone: the build needs the ggml submodule tree that the
# release tarball also carries, and --depth 1 keeps this to seconds.
git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$SRC"

if [ "${LLAMA_CPU_ONLY:-0}" = "1" ]; then
  echo "[build-llama-cuda] configuring (CPU ONLY -- no CUDA backend)"
  CUDA_ARGS=(-DGGML_CUDA=OFF)
else
  echo "[build-llama-cuda] configuring (CUDA archs: $CUDA_ARCHS)"
  CUDA_ARGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
fi
# $ORIGIN, not the build tree. ELF RUNPATH is NOT inherited by transitive
# dependencies -- unlike a macOS @rpath, which dyld resolves against the whole
# load chain -- so the exe's rpath finds libllama/libggml and then the loader
# looks for libggml's OWN libggml-cpu.so.0 using libggml's runpath alone. CMake
# bakes the build directory in there by default, and this script deletes that
# directory on exit, so a staged tree without this is loadable only by accident
# (an LD_LIBRARY_PATH someone set) and fails with "cannot open shared object
# file" for a library sitting right next to the one that wants it.
cmake -S "$SRC" -B "$TMP/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  "${CUDA_ARGS[@]}" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_MTMD=ON \
  -DLLAMA_CURL=OFF

# TOOLS/APP off, MTMD on: we link libraries, never upstream's binaries, and the
# unified `app` target hard-links against llama-server-impl/llama-cli-impl, so
# it fails to link the moment SERVER is off (it does not gate itself on it).
# LLAMA_BUILD_MTMD builds tools/mtmd as a STANDALONE library, which is how we
# get libmtmd without the rest of the tools tree -- it keeps parity with the
# Windows prebuilt asset, which already ships mtmd.dll.

echo "[build-llama-cuda] building with $JOBS jobs (this takes a while)"
cmake --build "$TMP/build" --config Release -j "$JOBS"

# Prove it is actually a CUDA build before staging anything. A CMake run that
# silently fell back to CPU produces a working libllama.so with no CUDA
# backend, which is exactly the outcome the Linux refusal exists to prevent --
# and it would only show up as "why is this so slow" much later.
if [ "${LLAMA_CPU_ONLY:-0}" != "1" ] && ! find "$TMP/build" -name 'libggml-cuda.so' | grep -q .; then
  echo "[build-llama-cuda] ERROR: the build produced no libggml-cuda.so." >&2
  echo "  GGML_CUDA=ON did not take. Check that nvcc matches your driver and" >&2
  echo "  that CMake found the CUDA toolkit; do NOT stage this build." >&2
  exit 1
fi

echo "[build-llama-cuda] staging into $DEST"
rm -rf "$DEST_LIB" "$DEST_INC"
mkdir -p "$DEST_LIB" "$DEST_INC"

# The whole ggml family, not just libllama: ggml dlopens its compute backends
# BY FILENAME at init (the shim calls ggml_backend_load_all), so a missing
# libggml-cuda.so downgrades the backend at runtime instead of failing to link.
find "$TMP/build" \( -name 'libllama.so*' -o -name 'libggml*.so*' -o -name 'libmtmd.so*' \) \
  -exec cp -P {} "$DEST_LIB/" \;

[ -f "$DEST_LIB/libllama.so" ] || {
  echo "[build-llama-cuda] ERROR: no libllama.so was staged." >&2
  exit 1
}

cp "$SRC/include"/*.h "$DEST_INC/"
cp "$SRC/ggml/include"/*.h "$DEST_INC/"
# mtmd (multimodal) lives under tools/, not include/ -- same as fetch-llama.sh.
[ -f "$SRC/tools/mtmd/mtmd.h" ] && cp "$SRC/tools/mtmd/mtmd.h" "$DEST_INC/"
[ -f "$SRC/tools/mtmd/mtmd-helper.h" ] && cp "$SRC/tools/mtmd/mtmd-helper.h" "$DEST_INC/"

if [ "${LLAMA_CPU_ONLY:-0}" = "1" ]; then
  # Stamped so it can never be mistaken for -- or skipped in favour of -- the
  # CUDA stage this script exists to produce.
  echo "$LLAMA_TAG-cpu" > "$STAMP"
  echo "[build-llama-cuda] staged llama.cpp ($LLAMA_TAG, CPU ONLY):"
else
  echo "$LLAMA_TAG" > "$STAMP"
  echo "[build-llama-cuda] staged llama.cpp ($LLAMA_TAG, CUDA archs $CUDA_ARCHS):"
fi
echo "  $(ls "$DEST_LIB" | wc -l | tr -d ' ') shared objects in $DEST_LIB"
echo "  $(ls "$DEST_INC" | wc -l | tr -d ' ') headers in $DEST_INC"
if [ "${LLAMA_CPU_ONLY:-0}" = "1" ]; then
  echo "[build-llama-cuda] WARNING: this stage has NO CUDA backend. Inference"
  echo "  runs on the CPU. Re-run without LLAMA_CPU_ONLY once nvcc is installed;"
  echo "  the '-cpu' stamp makes that rebuild rather than skip."
fi
