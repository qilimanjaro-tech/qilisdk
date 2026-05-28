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
FUZZY=0

for file in $(find ./docs/ -name "*.po"); do
    # match non-empty msgid followed immediately by empty msgstr (not a multiline continuation)
    if grep -Pzo 'msgid ".+"\nmsgstr ""\n(?!")' "$file" > /dev/null 2>&1; then
        echo "Missing translations in $file"
        MISSING=$((MISSING + 1))
    fi
    # match fuzzy translations (lines starting with #, followed by a line containing "fuzzy")
    if grep -Pzo '#.*\nfuzzy' "$file" > /dev/null 2>&1; then
        echo "Fuzzy translations in $file"
        FUZZY=$((FUZZY + 1))
    fi
done

echo "Total files with missing translations: $MISSING"
echo "Total files with fuzzy translations: $FUZZY"
