#!/usr/bin/env bash
# Hermetic Latent2RGB + JPEG preview tests (issue #208).
# No MLX, no model weights — runs on Linux as well as macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x "$ROOT/.zig-toolchain/zig" ]; then
  ./scripts/fetch-zig.sh
fi
export PATH="$ROOT/.zig-toolchain:$PATH"
zig build preview-test
echo "PASS: zig build preview-test"
