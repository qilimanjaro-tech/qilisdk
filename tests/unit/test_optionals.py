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

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING

import pytest

from qilisdk._optionals import (
    OptionalDependencyError,
    OptionalFeature,
    Symbol,
    _OptionalDependencyStub,
    import_optional_dependencies,
)

if TYPE_CHECKING:
    import SpeQtrum


def test_optional_stub_raises_on_call() -> None:
    feature = OptionalFeature(
        name="speqtrum",
        dependencies=["definitely-not-installed-dist-xyz"],
        symbols=[Symbol(path="unused", name="SpeQtrum")],
    )

    imported = import_optional_dependencies(feature)
    symbol = imported.symbols["SpeQtrum"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol()

    assert "Using SpeQtrum requires installing the 'speqtrum' optional feature" in str(excinfo.value)
    assert "pip install qilisdk[speqtrum]" in str(excinfo.value)


def test_optional_stub_raises_on_attribute_call() -> None:
    feature = OptionalFeature(
        name="speqtrum",
        dependencies=["definitely-not-installed-dist-xyz"],
        symbols=[Symbol(path="unused", name="SpeQtrum")],
    )

    imported = import_optional_dependencies(feature)
    symbol: SpeQtrum = imported.symbols["SpeQtrum"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol.login()

    assert "Using SpeQtrum.login requires installing the 'speqtrum' optional feature" in str(excinfo.value)
    assert "pip install qilisdk[speqtrum]" in str(excinfo.value)


def test_version_not_found(monkeypatch):
    def raise_not_found(name):
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", raise_not_found)

    sys.modules.pop("qilisdk", None)

    import qilisdk  # noqa: PLC0415

    importlib.reload(qilisdk)

    assert qilisdk.__version__ == "0.0.0"


def test_optional_stub():
    stub = _OptionalDependencyStub(symbol_name="SpeQtrum", feature_name="speqtrum", import_error="test")

    with pytest.raises(OptionalDependencyError):
        stub()

    with pytest.raises(AttributeError):
        stub.__magic__

    assert "missing optional" in repr(stub)


def test_import_optional_dependencies(monkeypatch):
    feature = OptionalFeature(
        name="dummy_feature",
        dependencies=[],
        symbols=[
            Symbol(path="qilisdk._optionals", name="Dummy1"),
            Symbol(path="qilisdk.optional_modules.dummy", name="Dummy2"),
        ],
    )

    import_optional_dependencies(feature)
