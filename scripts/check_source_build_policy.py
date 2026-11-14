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

# ruff: file-ignore[implicit-namespace-package]  # Standalone CI entrypoint.

"""Reject changes to the one reviewed third-party source-build exception."""

from __future__ import annotations

import tomllib
from pathlib import Path

SPHINX_SOURCE = "https://github.com/sphinx-contrib/multiversion?branch=main#cd4c0db66473f04d7e037188f5f13444574b4b0e"


def check_source_build_policy(lock: dict) -> None:
    """Require the reviewed Sphinx extension version and immutable Git source.

    Raises:
        ValueError: The documentation source exception needs another review.
    """
    packages = [package for package in lock["package"] if package["name"] == "sphinx-multiversion"]
    if len(packages) != 1 or packages[0]["version"] != "0.2.5" or packages[0]["source"] != {"git": SPHINX_SOURCE}:
        raise ValueError(
            "sphinx-multiversion source changed: review ci/source-build-policy.md before allowing its build"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    check_source_build_policy(tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8")))
