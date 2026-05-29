#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a combined coverage report 
# for both the C++ and Python code
# --------------------------------------------------------------------------
# Time estimate: 3 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -euo pipefail

# Clear any coverage caches
echo "Clearing coverage caches..."
rm -rf .coverage tests/unit_cpp/coverage
mkdir -p coverage

# Rebuild the C++ ensuring we build the test suite
echo "Rebuilding C++ with tests enabled..."
uv sync --all-groups --extra all-cu13 -Ccmake.build-type=Debug -Ccmake.define.tests=ON -Ccmake.define.coverage=ON --reinstall 2>&1

# Run the C++ test suite
echo "Running C++ tests with coverage..."
GCOV_PREFIX=./tests/unit_cpp/coverage GCOV_PREFIX_STRIP=5 ./tests/unit_cpp/test_cpp --gtest_brief=1

# Run Python coverage
echo "Running Python tests with coverage..."
python3 -m pytest --cov=qilisdk --cov-report=xml --cov-report=term-missing tests/unit_python/
mv coverage.xml coverage/coverage_python.xml

# Generate the C++ coverage XML report 
echo "Generating C++ coverage XML..."
gcovr \
    --exclude '.*googletest.*' \
    --exclude '/usr/.*' \
    --gcov-ignore-errors=all \
    --exclude-unreachable-branches \
    --exclude-throw-branches \
    --xml coverage/coverage_cpp.xml 2>&1
echo "C++ coverage XML generated at coverage/coverage_cpp.xml"

# Combine the reports into a single Cobertura XML report for Codecov
echo "Combining coverage reports into Cobertura XML..."
gcovr \
    --cobertura-add-tracefile coverage/coverage_python.xml \
    --cobertura-add-tracefile coverage/coverage_cpp.xml \
    --xml coverage/coverage_combined.xml
echo "Combined Cobertura XML generated at coverage/coverage_combined.xml"

# Combine the Python and C++ coverage reports into a single HTML report
echo "Generating combined HTML report..."
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_combined.xml
echo "Combined HTML report generated at coverage/index.html"

# Determine the base branch (default: origin/main)
BASE_BRANCH=${BASE_BRANCH:-origin/main}

# Report diff coverage (changed lines only) to terminal
echo "Generating diff coverage report (changed lines only vs $BASE_BRANCH)..."
diff-cover coverage/coverage_combined.xml \
  --compare-branch="$BASE_BRANCH" \
  --diff-range-notation="..." \

# Also generate an HTML version
diff-cover coverage/coverage_combined.xml \
  --compare-branch="$BASE_BRANCH" \
  --diff-range-notation="..." \
  --html-report coverage/coverage_diff.html
echo "Diff HTML report generated at coverage/coverage_diff.html"

# Fail if the diff coverage is below 100%, don't otherwise print
diff-cover coverage/coverage_combined.xml \
  --compare-branch="$BASE_BRANCH" \
  --quiet \
  --diff-range-notation="..." \
  --fail-under=99

echo "Coverage checks passed successfully!"