# Copyright 2025 Qilimanjaro Quantum Tech
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

import ast
import importlib
from pathlib import Path

import pytest

import qilisdk

SRC_ROOT = Path(qilisdk.__file__).parent.parent
STUBS = sorted(SRC_ROOT.joinpath("qilisdk").rglob("__init__.pyi"))


def _stub_all(stub: Path) -> list[str]:
    assignments = {
        target.id: node.value
        for node in ast.parse(stub.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    return ast.literal_eval(assignments["__all__"])


def test_stubs_found():
    assert len(STUBS) >= 6


@pytest.mark.parametrize("stub", STUBS, ids=lambda p: str(p.relative_to(SRC_ROOT)))
def test_stub_all_matches_runtime_all(stub: Path):
    module_name = ".".join(stub.parent.relative_to(SRC_ROOT).parts)
    module = importlib.import_module(module_name)
    assert sorted(_stub_all(stub)) == sorted(module.__all__)
