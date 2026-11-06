#!/bin/bash

# --------------------------------------------------------------------------
# This script installs qilisdk into the project's uv venv, along with
# all optional dependencies and dependency groups
# --------------------------------------------------------------------------
# Time estimate: 2 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -euo pipefail

# Make sure we're in the project root directory
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# Install everything, rebuilding the C++ in release mode
# (Python 3.13 since CUDA-Q has no 3.14 wheels yet, matching CI)
echo "Installing qilisdk with all optional dependencies..."
uv sync --python 3.13 --all-groups --extra all-cu13 -Ccmake.build-type=Release --reinstall-package qilisdk 2>&1

echo "qilisdk version $(.venv/bin/python -c 'import qilisdk; print(qilisdk.__version__)')" installed successfully!
