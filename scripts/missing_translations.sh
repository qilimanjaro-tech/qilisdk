#!/bin/bash

# --------------------------------------------------------------------------
# This script goes through every .po file in the project and 
# checks for missing translations.
# --------------------------------------------------------------------------
# Time estimate: 1 minute
# --------------------------------------------------------------------------

# Assuming we're in the root directory, change to docs directory
cd docs || exit 1

# Update the .po files from the docs source files
make gettext

# Check every catalog with gettext's own parser
MISSING=0
FUZZY=0
MALFORMED=0
while IFS= read -r file; do

    # Get all the statistics
    if ! STATS=$(LC_ALL=C msgfmt --statistics --output-file=/dev/null "$file" 2>&1); then
        echo "Malformed catalog $file:"
        echo "$STATS" | sed 's/^/    /'
        MALFORMED=$((MALFORMED + 1))
        continue
    fi

    # Parse the stats line
    UNTRANSLATED=$(awk '{for (i = 2; i <= NF; i++) if ($i == "untranslated") print $(i - 1)}' <<< "$STATS")
    INEXACT=$(awk '{for (i = 2; i <= NF; i++) if ($i == "fuzzy") print $(i - 1)}' <<< "$STATS")
    if [[ -n $UNTRANSLATED ]]; then
        echo "Missing translations in $file ($UNTRANSLATED untranslated)"
        MISSING=$((MISSING + 1))
    fi
    if [[ -n $INEXACT ]]; then
        echo "Fuzzy translations in $file ($INEXACT fuzzy)"
        FUZZY=$((FUZZY + 1))
    fi

done < <(find ./ -name "*.po")

# Output summary
echo "Total files with missing translations: $MISSING"
echo "Total files with fuzzy translations: $FUZZY"
echo "Total malformed files: $MALFORMED"

# Return to the root directory
cd ..

# Return non-zero exit code if any missing or fuzzy translations were found
if [[ $MISSING -gt 0 ]] || [[ $FUZZY -gt 0 ]] || [[ $MALFORMED -gt 0 ]]; then
    echo "Error: Missing or fuzzy translations found. Please update the .po files." >&2
    exit 1
fi
