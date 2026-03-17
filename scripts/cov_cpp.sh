#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a coverage report 
# for the C++ code
# --------------------------------------------------------------------------
# Time esimate: 1-2 minutes
# --------------------------------------------------------------------------

# Clear any coverage caches
echo "Clearing coverage caches..."
rm -rf .coverage coverage* tests/unit_cpp/coverage
mkdir -p coverage

# Rebuild the C++ ensuring we build the test suite
echo "Rebuilding C++ with tests enabled..."
uv -v sync --group dev --extra all-cu13 -Ccmake.build-type=Debug -Ccmake.define.tests=ON -Ccmake.define.coverage=ON --reinstall

# Run the C++ test suite
echo "Running C++ tests..."
GCOV_PREFIX=./tests/unit_cpp/coverage GCOV_PREFIX_STRIP=5 ./tests/unit_cpp/test_cpp

# Generate the C++ coverage XML report
echo "Generating C++ coverage XML..."
gcovr \
    --exclude '.*googletest.*' \
    --exclude '/usr/.*' \
    --gcov-ignore-errors=all \
    --xml coverage/coverage_cpp.xml
echo "C++ coverage XML generated at coverage/coverage_cpp.xml"

# Generate the report
echo "Generating report..."
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_cpp.xml
echo "C++ HTML report generated at coverage/index.html"


