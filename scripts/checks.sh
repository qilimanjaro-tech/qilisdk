#!/bin/bash

# --------------------------------------------------------------------------
# This script runs all the checks for the project, basically
# everything that should be done before pushing
# --------------------------------------------------------------------------
# Time estimate: 3 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -e

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/checks.log
> $LOG_FILE

# Make sure the C++ is up to date
echo "Syncing C++ dependencies with tests enabled..." | tee -a $LOG_FILE
uv -v sync --all-groups --extra all-cu13 --reinstall -Ccmake.define.tests=ON 2>&1 | tee -a $LOG_FILE

# clang-format every .cpp and .h in ./src/qilisdk_cpp/
echo "Running clang-format on C++ source files..." | tee -a $LOG_FILE 
find ./src/qilisdk_cpp/ -regex '.*\.\(cpp\|h\)$' -exec clang-format -i {} \;  2>&1 | tee -a $LOG_FILE
find ./tests/unit_cpp/ -regex '.*\.\(cpp\|h\)$' -exec clang-format -i {} \;  2>&1 | tee -a $LOG_FILE

# Ruff
echo "Running Ruff checks..." | tee -a $LOG_FILE
ruff check --fix 2>&1 | tee -a $LOG_FILE
ruff format 2>&1 | tee -a $LOG_FILE

# Ty
echo "Running Ty checks..." | tee -a $LOG_FILE
ty check 2>&1 | tee -a $LOG_FILE

# Tests
echo "Running Python unit tests..." | tee -a $LOG_FILE
pytest tests/unit_python 2>&1 | tee -a $LOG_FILE
echo "Running C++ unit tests..." | tee -a $LOG_FILE
./tests/unit_cpp/test_cpp --gtest_brief=1 2>&1 | tee -a $LOG_FILE
echo "Running integration tests..." | tee -a $LOG_FILE
pytest tests/integration 2>&1 | tee -a $LOG_FILE