#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a coverage report 
# for the Python code
# --------------------------------------------------------------------------
# Time esimate: < 1 minute
# --------------------------------------------------------------------------

# Clear any coverage caches
echo "Clearing coverage caches..."
rm -rf .coverage coverage* tests/unit_cpp/coverage
mkdir -p coverage

# Run Python coverage
echo "Running Python tests with coverage..."
python3 -m pytest --cov=qilisdk --cov-report=xml --cov-report=term-missing tests/unit_python/
mv coverage.xml coverage/coverage_python.xml

# Generate the report
echo "Generating report..."
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_python.xml
echo "Python HTML report generated at coverage/index.html"



