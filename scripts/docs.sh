#!/bin/bash

# --------------------------------------------------------------------------
# This script goes through every file in the docs directory, plus every docstring in the source,
# and scrapes all Python code blocks, then runs them to make sure they work and are up to date.
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

# Rebuild in release mode for performance
echo "Rebuilding C++ in release mode..."
uv sync --all-groups --extra all-cu13 -Ccmake.build-type=Release --reinstall 2>&1

# Create a temporary directory for the docs test
cd .tmp
mkdir -p .docs_test
cd .docs_test
TMPFILE=$(mktemp /tmp/docs_test_XXXXXX.py)
trap "rm -f $TMPFILE" EXIT

for file in $( { find ../../docs/ -name "*.rst"; find ../../changes/ -name "*.md"; find ../../src/ -name "*.py"; } | sort ); do

    # Use Python to extract and concatenate all Python code blocks, then run them.
    # RST: code-block:: python, skip with '.. SKIP' on the preceding line.
    # MD:  ```python fenced blocks, skip with '<!-- SKIP -->' on the preceding line.
    # PY:  the RST rules applied to each docstring, plus '>>>' doctest lines; each docstring runs in a fresh namespace.
    # Also resolves .. include:: directives recursively for RST files.
    python3 - "$file" > "$TMPFILE" <<'PYEOF'
import re, sys

import os

def extract_blocks_rst(filename, seen=None):
    if seen is None:
        seen = set()
    filename = os.path.realpath(filename)
    if filename in seen:
        return []
    seen.add(filename)

    with open(filename) as f:
        lines = f.readlines()
    return extract_blocks_rst_lines(lines, os.path.dirname(filename), seen)

def extract_blocks_rst_lines(lines, base_dir, seen):
    blocks = []
    i = 0
    while i < len(lines):
        # Resolve .. include:: directives recursively
        inc = re.match(r'^\s*\.\. include::\s+(.+?)\s*$', lines[i])
        if inc:
            inc_path = os.path.join(base_dir, inc.group(1))
            if os.path.isfile(inc_path):
                blocks.extend(extract_blocks_rst(inc_path, seen))
            i += 1
            continue

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

    return blocks

def extract_blocks_md(filename):
    with open(filename) as f:
        lines = f.readlines()
    blocks = []
    i = 0
    while i < len(lines):
        if re.match(r'^```python\s*$', lines[i]):
            # Check if the previous non-empty line is a SKIP marker
            j = i - 1
            while j >= 0 and lines[j].strip() == '':
                j -= 1
            skip = j >= 0 and lines[j].strip() == '<!-- SKIP -->'
            i += 1
            code_lines = []
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                if not skip:
                    code_lines.append(lines[i].rstrip('\n'))
                i += 1
            if not skip:
                while code_lines and code_lines[-1] == '':
                    code_lines.pop()
                if code_lines:
                    blocks.append('\n'.join(code_lines))
        i += 1
    return blocks

def extract_blocks_doctest(lines):
    code_lines = []
    in_example = False
    for line in lines:
        m = re.match(r'^\s*>>>(?: (.*))?$', line)
        cont = re.match(r'^\s*\.\.\.(?: (.*))?$', line)
        if m:
            code_lines.append(m.group(1) or '')
            in_example = True
        elif cont and in_example:
            code_lines.append(cont.group(1) or '')
        else:
            in_example = False
    return ['\n'.join(code_lines)] if code_lines else []

def extract_docstrings(filename):
    import ast
    with open(filename) as f:
        tree = ast.parse(f.read())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        lines = [l + '\n' for l in doc.splitlines()]
        blocks = extract_blocks_rst_lines(lines, os.path.dirname(filename), set()) + extract_blocks_doctest(lines)
        if blocks:
            lineno = node.body[0].lineno
            found.append((f"{getattr(node, 'name', '<module>')} (line {lineno})", '\n\n'.join(blocks)))
    return found

def extract(filename):
    header = "import matplotlib\nmatplotlib.use('Agg')\n"
    if filename.endswith('.py'):
        docstrings = extract_docstrings(filename)
        if docstrings:
            print(header + f"import sys, traceback\nfailed = False\nfor name, code in {docstrings!r}:\n"
                  f"    try:\n        exec(compile(code, {filename!r} + ':' + name, 'exec'), {{'__name__': '__main__'}})\n"
                  "    except BaseException:\n        failed = True\n        print(f'In docstring of {name}:')\n        traceback.print_exc(file=sys.stdout)\n"
                  "if failed:\n    raise SystemExit(1)")
        return
    if filename.endswith('.md'):
        blocks = extract_blocks_md(filename)
    else:
        blocks = extract_blocks_rst(filename)
    print(header + '\n\n'.join(blocks))

extract(sys.argv[1])
PYEOF

    # Skip files with no Python code blocks (silently for source files, most of which have none)
    if [[ ! -s "$TMPFILE" ]]; then
        if [[ "$file" != *.py ]]; then
            echo "SKIP $file (no Python code blocks)"
            SKIP=$((SKIP + 1))
        fi
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
echo "Note: skip a code block by adding '.. SKIP' immediately before a '.. code-block:: python' directive (RST and docstrings), or '<!-- SKIP -->' immediately before a '\`\`\`python' fence (MD)."
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"

cd ../..
rm -rf .tmp/.docs_test

# Return to the original directory
cd "$ORIG_DIR"

# Return non-zero exit code if any tests failed
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

