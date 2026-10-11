#!/usr/bin/env bash
# Build and run the QiliSim GPU sr_solve microbenchmark.
#
# Usage:
#   ./run_bench.sh [label]            # build + run, tag CSV rows with [label]
#   ./run_bench.sh --nsys [label]     # same, wrapped in `nsys profile --stats`
#
# Results are appended to benchmarks/results.csv. A typical before/after flow:
#   git checkout <baseline>; ./run_bench.sh before
#   git checkout <optimized>; ./run_bench.sh after
#   (or just re-run after each implementation step)
#
# Env passthrough: BENCH_SIZES, BENCH_ITERS, BENCH_WARMUP, BENCH_CPU_ITERS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/build"

# cmake may be a uv-installed tool under ~/.local/bin rather than on the system PATH.
export PATH="$HOME/.local/bin:$PATH"
command -v cmake >/dev/null || { echo "cmake not found (try: uv tool install cmake)"; exit 1; }

USE_NSYS=0
if [[ "${1:-}" == "--nsys" ]]; then
    USE_NSYS=1
    shift
fi
LABEL="${1:-}"

# sr_solve probes $VIRTUAL_ENV/lib/python*/site-packages/nvidia/*/lib for the
# CUDA runtime libraries shipped by the pip `nvidia-*` wheels. Point it at the
# repo venv so the dlopen resolves them without a system CUDA install.
if [[ -z "${VIRTUAL_ENV:-}" && -d "$REPO/.venv" ]]; then
    export VIRTUAL_ENV="$REPO/.venv"
fi

echo "[run_bench] configuring (build type: ${CMAKE_BUILD_TYPE:-RelWithDebInfo})..."
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" >/dev/null
echo "[run_bench] building..."
cmake --build "$BUILD" -j >/dev/null
BIN="$BUILD/bench_sr_solve"

export BENCH_CSV="${BENCH_CSV:-$HERE/results.csv}"
export BENCH_LABEL="$LABEL"

if [[ "$USE_NSYS" == "1" ]]; then
    command -v nsys >/dev/null || { echo "nsys not found on PATH"; exit 1; }
    OUT="$HERE/nsys_${LABEL:-run}"
    echo "[run_bench] profiling with nsys -> ${OUT}.nsys-rep"
    nsys profile --stats=true --force-overwrite=true -o "$OUT" "$BIN"
else
    "$BIN"
fi

echo "[run_bench] done. CSV: $BENCH_CSV"
