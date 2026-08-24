#!/bin/bash
# test_toolchain_fetch.sh — pure-helper tests for the cross-platform toolchain
# fetch scripts (scripts/fetch-zig.sh, scripts/fetch-llama.sh).
#
# Neither script may be run for real here: both download hundreds of MB. The
# project rule for build scripts is "factor a pure helper and test that", so
# each script sources cleanly with FETCH_LIB_ONLY=1 set (defining its resolver
# functions and returning before any network/filesystem work), and this script
# exercises the resolvers across every supported host triple.
#
# Covers:
#   1. zig_asset_name: host os/arch -> ziglang.org nightly asset + archive kind
#   2. llama_asset_plan: host os/arch -> llama.cpp release asset(s) + lib layout
#   3. both refuse unsupported hosts BY NAME rather than guessing
#
# Usage: ./tests/test_toolchain_fetch.sh

set -u
cd "$(dirname "$0")/.." || exit 1

PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $1"; }

# eq <desc> <expected> <actual>
eq() {
  if [ "$2" = "$3" ]; then pass; else fail "$1: expected '$2', got '$3'"; fi
}

export FETCH_LIB_ONLY=1
# shellcheck source=/dev/null
. scripts/fetch-zig.sh || { echo "FAIL: scripts/fetch-zig.sh does not source with FETCH_LIB_ONLY=1"; exit 1; }
# shellcheck source=/dev/null
. scripts/fetch-llama.sh || { echo "FAIL: scripts/fetch-llama.sh does not source with FETCH_LIB_ONLY=1"; exit 1; }

V="0.17.0-dev.1818+7051f8e73"

# ── 1. Zig asset resolution ────────────────────────────────────────────────
# The nightly index names assets zig-<arch>-<os>-<version>.<ext>; Windows ships
# a .zip where the unix hosts ship .tar.xz. A wrong extension here 404s at the
# CDN, so the archive kind travels WITH the name.
eq "zig macos arm64"   "zig-aarch64-macos-$V.tar.xz|tar"  "$(zig_asset_name Darwin arm64 "$V")"
eq "zig macos x86_64"  "zig-x86_64-macos-$V.tar.xz|tar"   "$(zig_asset_name Darwin x86_64 "$V")"
eq "zig linux x86_64"  "zig-x86_64-linux-$V.tar.xz|tar"   "$(zig_asset_name Linux x86_64 "$V")"
eq "zig linux arm64"   "zig-aarch64-linux-$V.tar.xz|tar"  "$(zig_asset_name Linux aarch64 "$V")"
eq "zig windows x64"   "zig-x86_64-windows-$V.zip|zip"    "$(zig_asset_name MINGW64_NT-10.0-26200 x86_64 "$V")"
# Git Bash, MSYS2 and Cygwin all report different uname -s strings for the same
# host. All three are the Windows build.
eq "zig msys x64"      "zig-x86_64-windows-$V.zip|zip"    "$(zig_asset_name MSYS_NT-10.0 x86_64 "$V")"
eq "zig cygwin x64"    "zig-x86_64-windows-$V.zip|zip"    "$(zig_asset_name CYGWIN_NT-10.0 x86_64 "$V")"

if zig_asset_name Plan9 x86_64 "$V" >/dev/null 2>&1; then
  fail "zig_asset_name accepted an unsupported OS"
else pass; fi
if zig_asset_name Linux mips "$V" >/dev/null 2>&1; then
  fail "zig_asset_name accepted an unsupported arch"
else pass; fi

# ── 2. llama.cpp asset resolution ──────────────────────────────────────────
# macOS keeps the XCFramework path (a single self-contained dylib). Windows
# takes the prebuilt CUDA zip, which ships DLLs ONLY — no headers, no import
# libraries — so its plan declares that headers come from source. It needs NO
# import libs: lld links a DLL directly under the gnu ABI (it refuses under
# msvc), which is why the Windows build targets x86_64-windows-gnu.
# Fields: asset|kind|libdir-layout.
T="b10472"
eq "llama macos arm64" "llama-$T-xcframework.zip|xcframework|dylib" \
   "$(llama_asset_plan Darwin arm64 "$T")"
eq "llama windows x64" "llama-$T-bin-win-cuda-13.3-x64.zip|win-cuda|dll+srcheaders" \
   "$(llama_asset_plan MINGW64_NT-10.0-26200 x86_64 "$T")"
# There is NO prebuilt Linux CUDA asset in llama.cpp releases (only cpu, vulkan,
# sycl, openvino), so Linux must be built from source rather than silently
# falling back to a CPU-only build that would report CUDA support it lacks.
eq "llama linux x64"   "|build-from-source|so+srcheaders" \
   "$(llama_asset_plan Linux x86_64 "$T")"

if llama_asset_plan Plan9 x86_64 "$T" >/dev/null 2>&1; then
  fail "llama_asset_plan accepted an unsupported OS"
else pass; fi

# ── 3. The CUDA runtime companion (Windows) ────────────────────────────────
# ggml-cuda.dll DLOPENS its CUDA runtime by filename and, when it is missing,
# ggml SILENTLY CONTINUES ON THE CPU. So a Windows stage that ships
# ggml-cuda.dll without cudart/cublas produces a build that reports CUDA
# support, links fine, boots fine, and prefills at CPU speed. The runtime is a
# SEPARATE release asset from the main win-cuda zip, which is why it is easy to
# omit: this is exactly what a fresh checkout hit (2026-08-23), and the tell
# was a missing `ggml_cuda_init: found N CUDA devices` line, not any error.
eq "cudart asset name" "cudart-llama-bin-win-cuda-13.3-x64.zip"    "$(llama_cudart_asset)"

# The version rides LLAMA_CUDA_VER, exactly like the main asset — the two are
# published as a matched pair and a mismatch is a load-time failure.
( export LLAMA_CUDA_VER=12.4
  eq "cudart asset honors LLAMA_CUDA_VER" "cudart-llama-bin-win-cuda-12.4-x64.zip"      "$(llama_cudart_asset)" ) || fail "cudart asset override subshell failed"

# The names the stage must contain afterwards. Asserted as a LIST because the
# verification step greps for exactly these; drift here silently disarms it.
eq "required cuda runtime dlls" "cublas64_13.dll cublasLt64_13.dll cudart64_13.dll"    "$(llama_cuda_runtime_dlls | sort | tr '
' ' ' | sed 's/ $//')"

# The verification itself, which is the whole point of the change: a complete
# stage must be silent, and an incomplete one must NAME what is missing rather
# than let the build proceed to a CPU fallback nobody notices.
CUDA_T="$(mktemp -d)"
for d in $(llama_cuda_runtime_dlls); do : > "$CUDA_T/$d"; done
eq "complete cuda stage passes silently" "" "$(llama_missing_cuda_runtime "$CUDA_T")"

rm -f "$CUDA_T/cublasLt64_13.dll"
eq "missing cuda dll is named" "cublasLt64_13.dll" "$(llama_missing_cuda_runtime "$CUDA_T")"

rm -f "$CUDA_T"/*.dll
eq "empty stage names all three" "cublas64_13.dll cublasLt64_13.dll cudart64_13.dll"    "$(llama_missing_cuda_runtime "$CUDA_T" | sort | tr '
' ' ' | sed 's/ $//')"
rm -rf "$CUDA_T"

# ── 4. The downloaded path and the extracted path must be the SAME path ────
# A pure-helper test proves the resolver picks the right ASSET; it cannot see
# that the script then saves the download under one name and untars another.
# That is exactly what happened: the Windows port changed the download to
# `-o "$TMP/zig.$KIND"` (so "$TMP/zig.tar" on macOS and Linux) and left the
# extract reading "$TMP/zig.tar.xz". Windows was unaffected because its KIND
# spells "zip" and matches by luck, so the break was invisible on the host
# being worked on -- while macOS and Linux could no longer stage a toolchain
# from a clean tree at all.
#
# A source scan rather than a run: fetching the real archive is hundreds of MB,
# and the defect is a coupling between two lines, which is readable.
# Pull the argument curl saves to, and the argument each extractor reads from.
# Generic on purpose: the check is "these name the same thing", not "they spell
# it the way it is spelled today".
DL_PATH=$(grep -oE -- '-o "[^"]+" "\$URL"' scripts/fetch-zig.sh | head -1 | sed -E 's/-o "([^"]+)".*//')
TAR_PATH=$(grep -oE 'tar xf "[^"]+"' scripts/fetch-zig.sh | head -1 | sed -E 's/tar xf "([^"]+)"//')
ZIP_PATH=$(grep -oE 'unzip -q "[^"]+"' scripts/fetch-zig.sh | head -1 | sed -E 's/unzip -q "([^"]+)"//')
if [ -z "$DL_PATH" ] || [ -z "$TAR_PATH" ] || [ -z "$ZIP_PATH" ]; then
  fail "could not find the download / extract paths in fetch-zig.sh (did they move?)"
else
  if [ "$DL_PATH" = "$TAR_PATH" ]; then pass
  else fail "fetch-zig.sh downloads to '$DL_PATH' but untars '$TAR_PATH' — the unix path cannot work"; fi
  if [ "$DL_PATH" = "$ZIP_PATH" ]; then pass
  else fail "fetch-zig.sh downloads to '$DL_PATH' but unzips '$ZIP_PATH' — the Windows path cannot work"; fi
fi

echo ""
echo "toolchain fetch helpers: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
