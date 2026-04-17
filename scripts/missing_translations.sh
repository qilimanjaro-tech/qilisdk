#!/bin/bash

# --------------------------------------------------------------------------
# This script goes through every .po file in the project and 
# checks for missing translations.
# --------------------------------------------------------------------------
# Time estimate: 1 minute
# --------------------------------------------------------------------------

# Save the current directory so we can return to it later
ORIG_DIR=$(pwd)

# Change to the root directory of the project
cd "$(dirname "$(dirname "$(realpath "$0")")")"

MISSING=0

for file in $(find ./docs/ -name "*.po"); do
    # match non-empty msgid followed immediately by empty msgstr
    if grep -Pzo 'msgid ".+"\nmsgstr ""\n' "$file" > /dev/null 2>&1; then
        echo "Missing translations in $file"
        MISSING=$((MISSING + 1))
    fi
done

echo "Total files with missing translations: $MISSING"
