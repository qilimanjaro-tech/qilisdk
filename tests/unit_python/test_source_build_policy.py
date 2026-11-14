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

# ruff: file-ignore[pytest-unittest-raises-assertion]  # Standard-library tests run before dependencies are installed.

"""Review-gate checks for the existing documentation source exception."""

import copy
import unittest

from scripts.check_source_build_policy import check_source_build_policy

REVIEWED = {
    "package": [
        {
            "name": "sphinx-multiversion",
            "version": "0.2.5",
            "source": {
                "git": "https://github.com/sphinx-contrib/multiversion?branch=main"
                "#cd4c0db66473f04d7e037188f5f13444574b4b0e"
            },
        }
    ]
}


class TestSourceBuildReviewGate(unittest.TestCase):
    def test_reviewed_source_is_accepted(self):
        check_source_build_policy(copy.deepcopy(REVIEWED))

    def test_new_commit_requires_review(self):
        changed = copy.deepcopy(REVIEWED)
        changed["package"][0]["source"]["git"] = (
            "https://github.com/sphinx-contrib/multiversion?branch=main#" + "0" * 40
        )
        with self.assertRaisesRegex(ValueError, "source changed"):
            check_source_build_policy(changed)

    def test_changed_version_requires_review(self):
        changed = copy.deepcopy(REVIEWED)
        changed["package"][0]["version"] = "0.2.6"
        with self.assertRaisesRegex(ValueError, "source changed"):
            check_source_build_policy(changed)

    def test_changed_origin_requires_review(self):
        changed = copy.deepcopy(REVIEWED)
        changed["package"][0]["source"]["git"] = (
            "https://example.invalid/multiversion?branch=main#cd4c0db66473f04d7e037188f5f13444574b4b0e"
        )
        with self.assertRaisesRegex(ValueError, "source changed"):
            check_source_build_policy(changed)

    def test_missing_or_duplicate_exception_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source changed"):
            check_source_build_policy({"package": []})
        with self.assertRaisesRegex(ValueError, "source changed"):
            check_source_build_policy({"package": REVIEWED["package"] * 2})


if __name__ == "__main__":
    unittest.main()
