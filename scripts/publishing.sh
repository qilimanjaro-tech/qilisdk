#!/usr/bin/env bash
# Prepare publishing tools once, then validate/upload without dependency resolution.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/publishing"
python="$tool_dir/.venv/bin/python"
case "${1:-}" in
  sync)
    if [ "$#" -ne 1 ]; then
      echo 'Usage: publishing.sh sync' >&2
      exit 2
    fi
    if [ ! -x "$python" ]; then
      uv venv --python 3.13 "$tool_dir/.venv"
    fi
    UV_NO_CONFIG=1 uv pip sync --python "$python" --require-hashes --only-binary :all: "$tool_dir/requirements.txt"
    UV_NO_CONFIG=1 uv pip check --python "$python"
    ;;
  check|upload)
    if [ "$#" -lt 2 ]; then
      echo 'Usage: publishing.sh check|upload <distribution>...' >&2
      exit 2
    fi
    if [ ! -x "$python" ]; then
      echo 'Run publishing.sh sync before checking or uploading distributions' >&2
      exit 2
    fi
    exec "$python" -m twine "$@"
    ;;
  *)
    echo 'Usage: publishing.sh sync|check|upload' >&2
    exit 2
    ;;
esac
