#!/bin/bash
# Run one of the tests/ integration scripts with the host shims installed.
#
#   ./tests/run.sh test_multi_model_dir.sh [args...]
#
# On macOS this is a plain passthrough — nothing here changes what runs. It
# exists for Windows (Git Bash), where the scripts are fine but the environment
# is not: `python3` resolves to the Microsoft Store's App Execution Alias, which
# exits 49 without running anything, so every JSON-parsing helper in the suite
# returns an empty string and the script blames the server.
#
# Prefer this over invoking a script directly on Windows. Scripts stay written
# the way they are on macOS; the difference lives in one place.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/tests/lib/portable_env.sh"

if [ $# -lt 1 ]; then
    echo "usage: $0 <test script> [args...]" >&2
    exit 2
fi

SCRIPT=$1; shift
# Accept both "test_foo.sh" and "tests/test_foo.sh".
case "$SCRIPT" in
    /*) ;;
    tests/*) SCRIPT="$ROOT/$SCRIPT" ;;
    *) SCRIPT="$ROOT/tests/$SCRIPT" ;;
esac
[ -f "$SCRIPT" ] || { echo "no such test script: $SCRIPT" >&2; exit 2; }

SHIM_DIR=$(mktemp -d)
trap 'rm -rf "$SHIM_DIR"' EXIT

# Only shadow python3 when the name is actually broken here; on a healthy host
# the shim resolves straight back to the same interpreter, so this is a no-op
# either way and never hides which interpreter a failure came from.
if ! python3 -c '' >/dev/null 2>&1; then
    # Resolve BEFORE prepending: afterwards the shim itself is the first
    # working `python3` on PATH and the message would just say "python3",
    # naming the shim instead of the interpreter behind it.
    REAL_PY=$(mlxserve_resolve_python)
    mlxserve_make_python_shim "$SHIM_DIR" >/dev/null
    PATH="$SHIM_DIR:$PATH"
    export PATH
    echo "[run.sh] python3 did not run here; shimmed to $REAL_PY" >&2
fi

exec bash "$SCRIPT" "$@"
