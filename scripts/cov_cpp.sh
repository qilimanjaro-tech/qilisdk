#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a coverage report 
# for the C++ code
# --------------------------------------------------------------------------
# Time estimate: 1-2 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -e

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/cov_cpp.log
> $LOG_FILE

# Clear any coverage caches
echo "Clearing coverage caches..." | tee -a $LOG_FILE
rm -rf .coverage tests/unit_cpp/coverage 2>&1 | tee -a $LOG_FILE
mkdir -p coverage

# Rebuild the C++ ensuring we build the test suite
echo "Rebuilding C++ with tests enabled..." | tee -a $LOG_FILE
uv -v sync --group dev --extra all-cu13 -Ccmake.build-type=Debug -Ccmake.define.tests=ON -Ccmake.define.coverage=ON --reinstall 2>&1 | tee -a $LOG_FILE

# Stop we if errored
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Build failed, exiting." | tee -a $LOG_FILE
    exit 1
fi

# Run the C++ test suite
echo "Running C++ tests..." | tee -a $LOG_FILE
GCOV_PREFIX=./tests/unit_cpp/coverage GCOV_PREFIX_STRIP=5 ./tests/unit_cpp/test_cpp --gtest_brief=1 2>&1 | tee -a $LOG_FILE

# Generate the C++ coverage XML report
echo "Generating C++ coverage XML..." | tee -a $LOG_FILE
gcovr \
    --exclude '.*googletest.*' \
    --exclude '/usr/.*' \
    --gcov-ignore-errors=all \
    --exclude-unreachable-branches \
    --exclude-throw-branches \
    --xml coverage/coverage_cpp.xml 2>&1 | tee -a $LOG_FILE
echo "C++ coverage XML generated at coverage/coverage_cpp.xml" | tee -a $LOG_FILE

# Generate the report
echo "Generating report..." | tee -a $LOG_FILE
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_cpp.xml 2>&1 | tee -a $LOG_FILE
echo "C++ HTML report generated at coverage/index.html" | tee -a $LOG_FILE


