#!/bin/bash
# Guard: tests/lib/portable_env.sh — the shim layer that lets the 140 shell
# integration scripts run unmodified on Windows.
#
# Why this exists: `python3` on a stock Windows box resolves to the Microsoft
# Store's App Execution Alias, a stub that prints an install advert and exits
# 49. `command -v python3` finds it, so the usual "is it on PATH" check says
# yes and every one of the 645 `python3` call sites across tests/ silently
# produces an EMPTY string instead of parsed JSON — which reads as the server
# having returned nothing. test_multi_model_dir reported "single-dir case
# returned []" for exactly that reason, with a working server behind it.
#
# So the resolver's contract is not "is it on PATH" but "does it RUN", and the
# shim directory is what makes the answer reachable from scripts that hardcode
# the name `python3`.
#
# Hermetic: no server, no model, no network.
set -u

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  PASS $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL $1"; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=lib/portable_env.sh
. "$ROOT/tests/lib/portable_env.sh"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ── 1. The resolver must reject an interpreter that is present but does not run.
mkdir -p "$WORK/fakebin"
cat > "$WORK/fakebin/python3" <<'STUB'
#!/bin/sh
echo "Python was not found; run without arguments to install from the Microsoft Store" >&2
exit 49
STUB
cat > "$WORK/fakebin/python" <<'REAL'
#!/bin/sh
echo "real-python $*"
exit 0
REAL
chmod +x "$WORK/fakebin/python3" "$WORK/fakebin/python"

GOT=$(PATH="$WORK/fakebin" mlxserve_resolve_python 2>/dev/null)
if [ "$GOT" = "python" ]; then
    ok "resolver skips a present-but-failing python3 and picks python"
else
    bad "resolver picked '$GOT', expected 'python' (a stub that exits non-zero is not an interpreter)"
fi

# ── 2. With a WORKING python3 first, it must be preferred (no needless churn).
mkdir -p "$WORK/goodbin"
cat > "$WORK/goodbin/python3" <<'REAL3'
#!/bin/sh
exit 0
REAL3
chmod +x "$WORK/goodbin/python3"
GOT=$(PATH="$WORK/goodbin" mlxserve_resolve_python 2>/dev/null)
if [ "$GOT" = "python3" ]; then
    ok "a working python3 is preferred"
else
    bad "resolver picked '$GOT', expected 'python3'"
fi

# ── 3. Nothing usable at all must FAIL, never silently emit a name. A resolver
#      that returns a broken interpreter reintroduces the empty-output class.
mkdir -p "$WORK/emptybin"
if PATH="$WORK/emptybin" mlxserve_resolve_python >/dev/null 2>&1; then
    bad "resolver succeeded with no interpreter on PATH"
else
    ok "resolver fails loudly when no interpreter runs"
fi

# ── 4. The shim directory must make the NAME python3 work, since that is what
#      the 140 scripts type. This is the whole point of the layer.
# PREPEND rather than replace: the helper shells out to mkdir/chmod, so a PATH
# holding only the fixture dir would fail for a reason that has nothing to do
# with interpreter resolution. Prepending is also what run.sh actually does.
SHIM=$(PATH="$WORK/fakebin:$PATH" mlxserve_make_python_shim "$WORK/shim" 2>/dev/null)
if [ -n "$SHIM" ] && [ -x "$SHIM/python3" ]; then
    OUT=$(PATH="$SHIM:$WORK/fakebin:$PATH" python3 hello 2>/dev/null)
    case "$OUT" in
        real-python*hello*) ok "shimmed python3 dispatches to the working interpreter" ;;
        *) bad "shimmed python3 produced '$OUT'" ;;
    esac
else
    bad "shim directory or shim binary missing"
fi

# ── 5. mlxserve_host_path: a path that goes into a REQUEST BODY must be in the
#      host's own form. Git Bash mangles POSIX paths into Windows ones when it
#      hands them to a native binary as argv, but a path embedded in a curl JSON
#      payload is just bytes -- so `--model-dir /tmp/x` reaches the server as
#      C:/.../Temp/x while {"model":"/tmp/x"} arrives verbatim and 404s. The
#      helper must be an IDENTITY on a host with no conversion to do, or it
#      would corrupt the macOS runs that already pass.
HP=$(mlxserve_host_path "$WORK")
if [ -n "$HP" ]; then
    ok "host_path returned a path ($HP)"
else
    bad "host_path returned empty"
fi
# Whatever form it returns must still name the same directory.
mkdir -p "$WORK/probe_dir"
HP2=$(mlxserve_host_path "$WORK/probe_dir")
if [ -d "$HP2" ]; then
    ok "host_path still resolves to the same directory"
else
    bad "host_path returned '$HP2', which is not a directory"
fi
case "$(uname -s)" in
    Darwin|Linux)
        if [ "$HP" = "$WORK" ]; then
            ok "host_path is the identity where no conversion is needed"
        else
            bad "host_path rewrote '$WORK' to '$HP' on a POSIX host"
        fi
        ;;
    *)
        case "$HP" in
            "") bad "host_path returned nothing on Windows" ;;
            /tmp/*|/c/*) bad "host_path left a POSIX path '$HP' on Windows" ;;
            *) ok "host_path produced a native path on Windows" ;;
        esac
        ;;
esac

# ── 6. mlxserve_has_mlx reads the BUILD, not the host: a Mac built -Dgguf-only
#      must answer the same as Windows, or the "check the portable config on a
#      Mac" workflow silently runs the MLX assertions.
cat > "$WORK/fakebin/mlx-serve-nomlx" <<'NOMLX'
#!/bin/sh
echo "mlx-serve 26.8.10"
echo "nax unavailable (built without MLX)"
NOMLX
cat > "$WORK/fakebin/mlx-serve-mlx" <<'WITHMLX'
#!/bin/sh
echo "mlx-serve 26.8.10"
echo "mlx 0.32.0"
echo "nax available"
WITHMLX
chmod +x "$WORK/fakebin/mlx-serve-nomlx" "$WORK/fakebin/mlx-serve-mlx"

if mlxserve_has_mlx "$WORK/fakebin/mlx-serve-mlx"; then
    ok "has_mlx: an MLX build is reported as one"
else
    bad "has_mlx said no for a build that reports an mlx version"
fi
if mlxserve_has_mlx "$WORK/fakebin/mlx-serve-nomlx"; then
    bad "has_mlx said yes for a build that reports 'built without MLX'"
else
    ok "has_mlx: a GGUF-only build is reported as one"
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
