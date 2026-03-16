#!/bin/bash

# Clear any coverage caches
echo "Clearing coverage caches..."
rm -rf .coverage coverage* tests/unit_cpp/coverage
mkdir -p coverage

# Rebuild the C++ ensuring we build the test suite
echo "Rebuilding C++ with tests enabled..."
uv -v sync --group dev --extra all-cu13 -Ccmake.build-type=Debug -Ccmake.define.tests=ON --reinstall

# Run the C++ test suite
echo "Running C++ tests..."
GCOV_PREFIX=./tests/unit_cpp/coverage GCOV_PREFIX_STRIP=5 ./tests/unit_cpp/test_cpp

# Run Python coverage
echo "Running Python tests with coverage..."
python3 -m pytest --cov=qilisdk --cov-report=xml --cov-report=term-missing tests/unit_python/
mv coverage.xml coverage/coverage_python.xml

# Convert the filtered lcov report to gcovr XML format
echo "Generating C++ coverage XML..."
gcovr \
    --exclude '.*googletest.*' \
    --exclude '/usr/.*' \
    --gcov-ignore-errors=all \
    --xml coverage/coverage_cpp.xml
echo "C++ coverage XML generated at coverage/coverage_cpp.xml"

# Combine the Python and C++ coverage reports into a single HTML report
echo "Generating combined HTML report..."
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_python.xml \
    --cobertura-add-tracefile coverage/coverage_cpp.xml
echo "Combined HTML report generated at coverage/index.html"


