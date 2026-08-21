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

echo ""
echo "toolchain fetch helpers: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
