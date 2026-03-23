#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a combined coverage report 
# for both the C++ and Python code
# --------------------------------------------------------------------------
# Time esimate: 2-3 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -e

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/cov.log
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

# Run Python coverage
echo "Running Python tests with coverage..." | tee -a $LOG_FILE
python3 -m pytest --cov=qilisdk --cov-report=xml --cov-report=term-missing tests/unit_python/ 2>&1 | tee -a $LOG_FILE
mv coverage.xml coverage/coverage_python.xml

# Generate the C++ coverage XML report 
echo "Generating C++ coverage XML..." | tee -a $LOG_FILE
gcovr \
    --exclude '.*googletest.*' \
    --exclude '/usr/.*' \
    --gcov-ignore-errors=all \
    --xml coverage/coverage_cpp.xml 2>&1 | tee -a $LOG_FILE
echo "C++ coverage XML generated at coverage/coverage_cpp.xml" | tee -a $LOG_FILE

# Combine the reports into a single Cobertura XML report for Codecov
echo "Combining coverage reports into Cobertura XML..." | tee -a $LOG_FILE
gcovr \
    --cobertura-add-tracefile coverage/coverage_python.xml \
    --cobertura-add-tracefile coverage/coverage_cpp.xml \
    --xml coverage/coverage_combined.xml 2>&1 | tee -a $LOG_FILE
echo "Combined Cobertura XML generated at coverage/coverage_combined.xml" | tee -a $LOG_FILE

# Combine the Python and C++ coverage reports into a single HTML report
echo "Generating combined HTML report..." | tee -a $LOG_FILE
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_combined.xml
echo "Combined HTML report generated at coverage/index.html" | tee -a $LOG_FILE


