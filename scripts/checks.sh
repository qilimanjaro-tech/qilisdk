#!/bin/bash

# --------------------------------------------------------------------------
# This script runs all the checks for the project, basically
# everything that should be done before pushing
# --------------------------------------------------------------------------
# Time estimate: 3 minutes
# --------------------------------------------------------------------------

# Stop if any command fails
set -euo pipefail

# Keep a log file in same directory as this script
LOG_FILE=$(dirname "$0")/checks.log
> $LOG_FILE

# Run the linting and formatting checks
echo "Running linting and formatting checks..." | tee -a $LOG_FILE
bash scripts/linting.sh 2>&1 | tee -a $LOG_FILE

# Compile and run coverage checks
echo "Running coverage checks..." | tee -a $LOG_FILE
bash scripts/coverage.sh 2>&1 | tee -a $LOG_FILE

# Also run the integration tests
echo "Running integration tests..." | tee -a $LOG_FILE
pytest tests/integration 2>&1 | tee -a $LOG_FILE

# Check the docs
echo "Running docs checks..." | tee -a $LOG_FILE
bash scripts/docs.sh 2>&1 | tee -a $LOG_FILE

# Check for missing translations
echo "Running missing translations checks..." | tee -a $LOG_FILE
bash scripts/missing_translations.sh 2>&1 | tee -a $LOG_FILE

# Check copyrights
echo "Running copyright header checks..." | tee -a $LOG_FILE
bash scripts/copyrights.sh 2>&1 | tee -a $LOG_FILE

echo "All checks passed successfully!" | tee -a $LOG_FILE