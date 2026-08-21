#!/usr/bin/env bash
# Fetch the pinned Zig nightly and stage it at .zig-toolchain/ (stable path,
# independent of the version string in the archive's own top-level dir name).
#
# 0.17.0 isn't tagged stable yet (homebrew's `zig` formula still ships
# 0.16.0), and 0.16.0's bundled libc++ fails to compile against the macOS 27
# SDK (`use of undeclared identifier 'INFINITY'` in its vendored <random> —
# see build.zig's version-gate comptime block). Fixed upstream by 0.17.0-dev;
# this script pins the exact dev snapshot until 0.17.0 stable ships, at which
# point ZIG_VERSION should drop back to a plain "0.17.0" and this script can
# eventually retire in favor of the homebrew formula again.
#
# This is the single source of truth for the pinned Zig version. Bump
# ZIG_VERSION to upgrade; CI and local builds re-fetch automatically.
#
# Hosts: macOS (arm64/x86_64), Linux (x86_64/aarch64), and Windows x86_64 via
# Git Bash / MSYS2 / Cygwin. `FETCH_LIB_ONLY=1 . scripts/fetch-zig.sh` defines
# the resolver and returns without touching the network — see
# tests/test_toolchain_fetch.sh.

# `set -e` in a SOURCED script leaks into the caller's shell and would abort a
# test harness on the first intentional failure probe. Only arm it when this
# script is actually being executed.
if [ -z "${FETCH_LIB_ONLY:-}" ]; then set -euo pipefail; fi

ZIG_VERSION="${ZIG_VERSION:-0.17.0-dev.1818+7051f8e73}"

# zig_asset_name <uname-s> <uname-m> <version> -> "<asset-filename>|<kind>"
#
# The archive KIND travels with the name because it is not derivable from the
# host alone at the call site: ziglang.org ships .tar.xz for the unix hosts and
# .zip for Windows, and guessing wrong is a CDN 404 rather than a local error.
zig_asset_name() {
  local uname_s="$1" uname_m="$2" version="$3" arch os ext kind
  case "$uname_m" in
    arm64|aarch64) arch="aarch64" ;;
    x86_64|amd64)  arch="x86_64" ;;
    *) echo "[fetch-zig] ERROR: unsupported arch $uname_m" >&2; return 1 ;;
  esac
  case "$uname_s" in
    Darwin) os="macos"; ext="tar.xz"; kind="tar" ;;
    Linux)  os="linux"; ext="tar.xz"; kind="tar" ;;
    # Git Bash reports MINGW64_NT-*, MSYS2 reports MSYS_NT-*, Cygwin
    # CYGWIN_NT-*. All three are the same Windows host and take the same build.
    MINGW*|MSYS*|CYGWIN*)
      os="windows"; ext="zip"; kind="zip"
      if [ "$arch" != "x86_64" ]; then
        echo "[fetch-zig] ERROR: unsupported Windows arch $uname_m" >&2; return 1
      fi
      ;;
    *) echo "[fetch-zig] ERROR: unsupported OS $uname_s" >&2; return 1 ;;
  esac
  echo "zig-${arch}-${os}-${version}.${ext}|${kind}"
}

# Sourced for its helpers only (tests): stop here, before any network or
# filesystem work.
if [ -n "${FETCH_LIB_ONLY:-}" ]; then return 0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/.zig-toolchain"
STAMP="$DEST/.version"

# The staged binary is zig.exe on Windows. Resolve once so the idempotency
# check and the final report agree with what actually got extracted.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ZIG_BIN="zig.exe" ;;
  *) ZIG_BIN="zig" ;;
esac

# Idempotent: skip when the staged copy already matches the pinned version.
if [ -f "$STAMP" ] && [ -x "$DEST/$ZIG_BIN" ]; then
  if [ "$(cat "$STAMP")" = "$ZIG_VERSION" ]; then
    echo "[fetch-zig] .zig-toolchain already at $ZIG_VERSION — nothing to do"
    exit 0
  fi
  echo "[fetch-zig] staged version '$(cat "$STAMP")' != '$ZIG_VERSION' — refetching"
fi

RESOLVED="$(zig_asset_name "$(uname -s)" "$(uname -m)" "$ZIG_VERSION")"
ASSET="${RESOLVED%|*}"
KIND="${RESOLVED##*|}"
URL="https://ziglang.org/builds/${ASSET}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch-zig] downloading $URL"
curl -fSL --retry 3 -o "$TMP/zig.$KIND" "$URL"

echo "[fetch-zig] extracting"
if [ "$KIND" = "zip" ]; then
  # Git Bash ships GNU tar (no zip support) but Windows 10+ ships bsdtar as
  # /c/Windows/System32/tar.exe, which does. Try unzip, then bsdtar, then
  # PowerShell — one of the three exists on every Windows dev box.
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$TMP/zig.zip" -d "$TMP"
  elif /c/Windows/System32/tar.exe --version >/dev/null 2>&1; then
    /c/Windows/System32/tar.exe -xf "$(cygpath -w "$TMP/zig.zip")" -C "$(cygpath -w "$TMP")"
  else
    powershell.exe -NoProfile -Command \
      "Expand-Archive -Force -LiteralPath '$(cygpath -w "$TMP/zig.zip")' -DestinationPath '$(cygpath -w "$TMP")'"
  fi
else
  tar xf "$TMP/zig.tar.xz" -C "$TMP"
fi

# The archive's top-level dir is the asset name minus its extension.
EXTRACTED="$TMP/${ASSET%.tar.xz}"
EXTRACTED="${EXTRACTED%.zip}"
if [ ! -f "$EXTRACTED/$ZIG_BIN" ]; then
  echo "[fetch-zig] ERROR: no $ZIG_BIN executable in $ASSET" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$EXTRACTED"/. "$DEST"/
chmod +x "$DEST/$ZIG_BIN" 2>/dev/null || true

echo "$ZIG_VERSION" > "$STAMP"

echo "[fetch-zig] staged Zig ($ZIG_VERSION):"
echo "  $DEST/$ZIG_BIN ($("$DEST/$ZIG_BIN" version))"
echo ""
echo "  Add it to PATH for this shell:"
echo "    export PATH=\"$DEST:\$PATH\""
