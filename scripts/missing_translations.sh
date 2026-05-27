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
    # match msgid blocks (single or multiline) followed by an empty msgstr
    if grep -Pzo 'msgid ".*"\n(?:".*"\n)*msgstr ""\n(?!")' "$file" > /dev/null 2>&1; then
        echo "Missing translations in $file"
        MISSING=$((MISSING + 1))
    fi
    # match fuzzy translations, excluding the file header (which always has msgid "")
    if grep -Pzo '#, fuzzy\nmsgid ".+"' "$file" > /dev/null 2>&1; then
        echo "Fuzzy translations in $file"
        FUZZY=$((FUZZY + 1))
    fi
done

echo "Total files with missing translations: $MISSING"
echo "Total files with fuzzy translations: $FUZZY"
