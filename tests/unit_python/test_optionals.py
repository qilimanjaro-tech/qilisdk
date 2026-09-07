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
import importlib.metadata
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from packaging.requirements import Requirement

from qilisdk._optionals import (
    DependencyGroup,
    OptionalDependencyError,
    OptionalFeature,
    RequirementMode,
    Symbol,
    _OptionalDependencyStub,
    import_optional_dependencies,
)

if TYPE_CHECKING:
    import SpeQtrum


def test_optional_stub_raises_on_call() -> None:
    feature = OptionalFeature(
        name="speqtrum",
        mode=RequirementMode.ALL,
        dependency_groups=[
            DependencyGroup(dists=["definitely-not-installed-dist-xyz"], extra="speqtrum"),
        ],
        symbols=[Symbol(path="unused", name="SpeQtrum")],
    )

    imported = import_optional_dependencies(feature)
    symbol = imported.symbols["SpeQtrum"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol()

    assert "Using SpeQtrum requires installing optional dependencies" in str(excinfo.value)
    assert "pip install qilisdk[speqtrum]" in str(excinfo.value)


def test_optional_stub_raises_on_attribute_call() -> None:
    feature = OptionalFeature(
        name="speqtrum",
        mode=RequirementMode.ALL,
        dependency_groups=[
            DependencyGroup(dists=["definitely-not-installed-dist-xyz"], extra="speqtrum"),
        ],
        symbols=[Symbol(path="unused", name="SpeQtrum")],
    )

    imported = import_optional_dependencies(feature)
    symbol: SpeQtrum = imported.symbols["SpeQtrum"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol.login()

    assert "Using SpeQtrum.login requires installing optional dependencies" in str(excinfo.value)
    assert "pip install qilisdk[speqtrum]" in str(excinfo.value)


def test_optional_stub_any_generates_programmatic_hint() -> None:
    feature = OptionalFeature(
        name="cuda",
        mode=RequirementMode.ANY,
        dependency_groups=[
            DependencyGroup(dists=["definitely-not-installed-cu12"], extra="cuda12"),
            DependencyGroup(dists=["definitely-not-installed-cu13"], extra="cuda13"),
        ],
        symbols=[Symbol(path="unused", name="CudaqBackend")],
    )

    imported = import_optional_dependencies(feature)
    symbol = imported.symbols["CudaqBackend"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol()

    msg = str(excinfo.value)
    assert "pip install qilisdk[cuda12]" in msg
    assert "pip install qilisdk[cuda13]" in msg
    assert " or " in msg


def test_backend_features_point_at_upstream_distributions() -> None:
    from qilisdk._optionals import _default_install_hint  # ruff: ignore[import-outside-top-level]
    from qilisdk.backends import OPTIONAL_FEATURES  # ruff: ignore[import-outside-top-level]

    hints = {feature.name: _default_install_hint(feature) for feature in OPTIONAL_FEATURES}

    assert "cuda-quantum-cu12" in hints["cudaq"]
    assert "cuda-quantum-cu13" in hints["cudaq"]
    assert "qutip" in hints["qutip"]
    assert "qutip-qip" in hints["qutip"]
    assert not any("qilisdk[" in hint for hint in hints.values())
    # Every requirement reaches the hint verbatim, quoted so a shell keeps the specifier.
    for feature in OPTIONAL_FEATURES:
        for group in feature.dependency_groups:
            for dist in group.dists:
                assert f'"{dist}"' in hints[feature.name]


@pytest.mark.parametrize(
    "dependency_groups",
    [
        [],
        [DependencyGroup(dists=[])],
    ],
)
@pytest.mark.parametrize("mode", list(RequirementMode))
def test_hint_falls_back_to_the_feature_name(mode: RequirementMode, dependency_groups: list[DependencyGroup]) -> None:
    """A feature that names nothing installable still points at its own extra."""
    from qilisdk._optionals import _default_install_hint  # ruff: ignore[import-outside-top-level]

    feature = OptionalFeature(
        name="nameless",
        mode=mode,
        dependency_groups=dependency_groups,
        symbols=[Symbol(path="unused", name="Thing")],
    )

    assert _default_install_hint(feature) == "`pip install qilisdk[nameless]`"


def test_outdated_distribution_reports_the_version_it_found() -> None:
    """An installed but too-old distribution says so, instead of "not installed"."""
    installed = importlib.metadata.version("numpy")
    feature = OptionalFeature(
        name="numpy-feature",
        mode=RequirementMode.ALL,
        dependency_groups=[DependencyGroup(dists=["numpy>=999.0.0"])],
        symbols=[Symbol(path="unused", name="Thing")],
    )

    imported = import_optional_dependencies(feature)

    with pytest.raises(OptionalDependencyError) as excinfo:
        imported.symbols["Thing"]()

    message = str(excinfo.value)
    assert f"numpy>=999.0.0 (found {installed})" in message
    assert 'pip install "numpy>=999.0.0"' in message


def test_missing_and_outdated_alternatives_are_both_reported() -> None:
    """In ANY mode every alternative that failed is listed, with its own reason."""
    installed = importlib.metadata.version("numpy")
    feature = OptionalFeature(
        name="either",
        mode=RequirementMode.ANY,
        dependency_groups=[
            DependencyGroup(dists=["numpy>=999.0.0"]),
            DependencyGroup(dists=["definitely-not-installed-dist-xyz"]),
        ],
        symbols=[Symbol(path="unused", name="Thing")],
    )

    imported = import_optional_dependencies(feature)

    with pytest.raises(OptionalDependencyError) as excinfo:
        imported.symbols["Thing"]()

    message = str(excinfo.value)
    assert f"numpy>=999.0.0 (found {installed})" in message
    assert "definitely-not-installed-dist-xyz (not installed)" in message


def test_satisfied_floor_resolves_the_symbol() -> None:
    """A distribution that meets its floor lets the real symbol through."""
    feature = OptionalFeature(
        name="numpy-feature",
        mode=RequirementMode.ALL,
        dependency_groups=[DependencyGroup(dists=["numpy>=1.0.0"])],
        symbols=[Symbol(path="qilisdk._optionals", name="OptionalFeature")],
    )

    imported = import_optional_dependencies(feature)

    assert imported.symbols["OptionalFeature"] is OptionalFeature


def test_runtime_floors_match_the_versions_ci_tests() -> None:
    """The floors enforced at runtime are the ones the backends group installs."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if not pyproject.exists():  # running against an installed wheel
        pytest.skip("pyproject.toml is not available")

    from qilisdk.backends import OPTIONAL_FEATURES  # ruff: ignore[import-outside-top-level]

    with pyproject.open("rb") as file:
        tested = {
            Requirement(spec.split(";")[0]).name: Requirement(spec.split(";")[0]).specifier
            for spec in tomllib.load(file)["dependency-groups"]["backends"]
        }
    enforced = {
        Requirement(dist).name: Requirement(dist).specifier
        for feature in OPTIONAL_FEATURES
        for group in feature.dependency_groups
        for dist in group.dists
    }

    shared = tested.keys() & enforced.keys()
    assert shared, "the backends group and OPTIONAL_FEATURES name no distribution in common"
    for name in shared:
        assert enforced[name] == tested[name], f"{name}: runtime requires {enforced[name]}, CI tests {tested[name]}"


def test_version_not_found(monkeypatch):
    def raise_not_found(name):
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", raise_not_found)

    sys.modules.pop("qilisdk", None)

    import qilisdk  # ruff: ignore[import-outside-top-level]

    importlib.reload(qilisdk)

    assert qilisdk.__version__ == "0.0.0"


def test_optional_stub():
    stub = _OptionalDependencyStub(
        symbol_name="SpeQtrum",
        feature_name="speqtrum",
        import_error=Exception("test"),
        install_hint="`pip install qilisdk[speqtrum]`",
    )

    with pytest.raises(OptionalDependencyError):
        stub()

    with pytest.raises(AttributeError):
        stub.__magic__

    assert "missing optional" in repr(stub)


def test_import_optional_dependencies(monkeypatch):
    feature = OptionalFeature(
        name="dummy_feature",
        mode=RequirementMode.ALL,
        dependency_groups=[
            DependencyGroup(dists=[], extra="dummy_feature"),
        ],
        symbols=[
            Symbol(path="qilisdk._optionals", name="Dummy1"),
            Symbol(path="qilisdk.optional_modules.dummy", name="Dummy2"),
        ],
    )

    import_optional_dependencies(feature)
