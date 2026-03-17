#!/bin/bash

# --------------------------------------------------------------------------
# This script runs everything needed to generate a coverage report 
# for the Python code
# --------------------------------------------------------------------------
# Time esimate: < 1 minute
# --------------------------------------------------------------------------

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/cov.log
> $LOG_FILE

# Clear any coverage caches
echo "Clearing coverage caches..." | tee -a $LOG_FILE
rm -rf .coverage coverage* tests/unit_cpp/coverage 2>&1 | tee -a $LOG_FILE
mkdir -p coverage

# Run Python coverage
echo "Running Python tests with coverage..." | tee -a $LOG_FILE
python3 -m pytest --cov=qilisdk --cov-report=xml --cov-report=term-missing tests/unit_python/ 2>&1 | tee -a $LOG_FILE
mv coverage.xml coverage/coverage_python.xml

# Generate the report
echo "Generating report..." | tee -a $LOG_FILE
gcovr \
    --html-nested coverage/index.html \
    --html-theme github.green \
    --cobertura-add-tracefile coverage/coverage_python.xml 2>&1 | tee -a $LOG_FILE
echo "Python HTML report generated at coverage/index.html" | tee -a $LOG_FILE



