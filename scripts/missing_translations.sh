#!/bin/bash

# --------------------------------------------------------------------------
# This script goes through every .po file in the project and 
# checks for missing translations.
# --------------------------------------------------------------------------
# Time estimate: 1 minute
# --------------------------------------------------------------------------

# Assuming we're in the root directory, change to docs directory
cd docs

# Update the .po files from the docs source files
make gettext

# Check for missing translations in the .po files
MISSING=0
FUZZY=0
for file in $(find ./ -name "*.po"); do
    # match non-empty msgid followed immediately by empty msgstr (not a multiline continuation)
    if grep -Pzo 'msgid ".+"\nmsgstr ""\n(?!")' "$file" > /dev/null 2>&1; then
        echo "Missing translations in $file"
        MISSING=$((MISSING + 1))
    fi
    # match fuzzy translations, skipping the header block
    if tail -n +11 "$file" | grep -q '^#, fuzzy'; then
        echo "Fuzzy translations in $file"
        FUZZY=$((FUZZY + 1))
    fi
done

# Output summary
echo "Total files with missing translations: $MISSING"
echo "Total files with fuzzy translations: $FUZZY"

# Return to the root directory
cd ..

# Return non-zero exit code if any missing or fuzzy translations were found
if [[ $MISSING -gt 0 ]] || [[ $FUZZY -gt 0 ]]; then
    echo "Error: Missing or fuzzy translations found. Please update the .po files." >&2
    exit 1
fi
