#!/bin/bash

# --------------------------------------------------------------------------
# Check to make sure every file has a copyright header 
# containing "Qilimanjaro Quantum Tech"
# --------------------------------------------------------------------------
# Time estimate: < 1 minute
# --------------------------------------------------------------------------

# All files in ./src and ./tests, only .py, .cpp and .h files
FAILED=0
echo "Checking for copyright headers..."
find ./src ./tests -type f \( -name "*.py" -o -name "*.cpp" -o -name "*.h" \) | while read file; do
    if ! grep -q "Qilimanjaro Quantum Tech" "$file"; then
        echo "Missing copyright header in $file"
        FAILED=1
    fi
done

if [[ $FAILED -ne 0 ]]; then
    echo "Copyright header check failed. Please add the appropriate header to the files listed above."
    exit 1
fi

echo "All files have a copyright header."

# As a reminder, the copyright notice is the following:
# --------------------------------------------------------------------------
# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------