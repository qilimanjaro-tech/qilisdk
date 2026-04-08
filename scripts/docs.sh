#!/bin/bash

# --------------------------------------------------------------------------
# This script goes through every file in the docs directory and scrapes all Python code blocks,
# then runs them to make sure they work and are up to date.
# --------------------------------------------------------------------------
# Time estimate: 1 minute
# --------------------------------------------------------------------------

PASS=0
FAIL=0
SKIP=0

# Save the current directory so we can return to it later
ORIG_DIR=$(pwd)

# Change to the root directory of the project
cd "$(dirname "$(dirname "$(realpath "$0")")")"

TMPFILE=$(mktemp /tmp/docs_test_XXXXXX.py)
trap "rm -f $TMPFILE" EXIT

for file in $(find docs/ -name "*.rst" | sort); do

    # Use Python to extract and concatenate all code-block:: python blocks
    python3 - "$file" > "$TMPFILE" <<'PYEOF'
import re, sys

def extract(filename):
    with open(filename) as f:
        lines = f.readlines()
    blocks = []
    i = 0
    while i < len(lines):
        # Match a code-block:: python directive at any indent level
        m = re.match(r'^(\s*).. code-block:: python\s*$', lines[i])
        if m and (i == 0 or lines[i-1].strip() != '.. SKIP'):
            base_indent = len(m.group(1))
            i += 1
            # Skip directive options (e.g. :linenos:) and blank lines before code
            while i < len(lines) and (
                lines[i].strip() == '' or re.match(r'^\s+:\w[\w-]*:', lines[i])
            ):
                i += 1
            # Collect lines that are indented deeper than the directive
            code_lines = []
            while i < len(lines):
                raw = lines[i].rstrip('\n')
                if raw.strip() == '':
                    code_lines.append('')
                    i += 1
                elif len(raw) - len(raw.lstrip()) > base_indent:
                    code_lines.append(raw)
                    i += 1
                else:
                    break
            # Strip trailing blank lines
            while code_lines and code_lines[-1] == '':
                code_lines.pop()
            if code_lines:
                min_ind = min(len(l) - len(l.lstrip()) for l in code_lines if l.strip())
                dedented = [l[min_ind:] if l.strip() else '' for l in code_lines]
                blocks.append('\n'.join(dedented))
        else:
            i += 1

    header = "import matplotlib\nmatplotlib.use('Agg')\n"
    print(header + '\n\n'.join(blocks))

extract(sys.argv[1])
PYEOF

    # Skip files with no Python code blocks
    if [[ ! -s "$TMPFILE" ]]; then
        echo "SKIP $file (no Python code blocks)"
        SKIP=$((SKIP + 1))
        continue
    fi

    # Run the combined script
    output=$(python3 "$TMPFILE" 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "PASS $file"
        PASS=$((PASS + 1))
    else
        echo "FAIL $file"
        echo "$output" | sed 's/^/     /'
        FAIL=$((FAIL + 1))
    fi

done

echo ""
echo "Note: you can skip a code block by adding a '.. SKIP' line immediately before the '.. code-block:: python' directive." 
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"

# Return to the original directory
cd "$ORIG_DIR"

[[ $FAIL -eq 0 ]]

