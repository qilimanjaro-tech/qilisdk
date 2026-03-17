#!/bin/bash

# --------------------------------------------------------------------------
# This script runs all the checks for the project, basically
# everything that should be done before pushing
# --------------------------------------------------------------------------
# Time esimate: 3 minutes
# --------------------------------------------------------------------------

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/checks.log
> $LOG_FILE

# Make sure the C++ is up to date
uv -v sync --all-groups --extra all-cu13 --reinstall -Ccmake.define.tidy=ON -Ccmake.define.tests=ON 2>&1 | tee -a $LOG_FILE

# clang-format everty .cpp and .h in ./src/qilisdk_cpp/
echo "Running clang-format on C++ source files..." | tee -a $LOG_FILE 
find ./src/qilisdk_cpp/ -regex '.*\.\(cpp\|h\)$' -exec clang-format -i {} \;  2>&1 | tee -a $LOG_FILE

# Ruff
ruff check --fix 2>&1 | tee -a $LOG_FILE
ruff format 2>&1 | tee -a $LOG_FILE

# Ty
ty check 2>&1 | tee -a $LOG_FILE

# Tests
pytest tests/ 2>&1 | tee -a $LOG_FILE