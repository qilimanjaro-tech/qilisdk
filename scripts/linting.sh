#!/bin/bash

# --------------------------------------------------------------------------
# Run all linting and formatting checks for the project, including:
# - Ruff for Python linting and formatting
# - Ty for Python type checking
# - clang-format for C++ formatting
# --------------------------------------------------------------------------
# Time estimate: < 1 minute
# --------------------------------------------------------------------------

# Stop if anything fails
set -euo pipefail

# Ruff
echo "Running Ruff checks..."
ruff check --fix 2>&1
ruff format 2>&1

# Ty
echo "Running Ty checks..."
ty check 2>&1

# clang-format every .cpp and .h in ./src/qilisdk_cpp/
echo "Running clang-format on C++ source files..."
find ./src/qilisdk_cpp/ -regex '.*\.\(cpp\|h\)$' -exec clang-format -i {} \;  2>&1
find ./tests/unit_cpp/ -regex '.*\.\(cpp\|h\)$' -exec clang-format -i {} \;  2>&1
