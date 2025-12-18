from __future__ import annotations

import pytest

from qilisdk._optionals import OptionalDependencyError, OptionalFeature, Symbol, import_optional_dependencies


def test_optional_stub_raises_on_call() -> None:
    feature = OptionalFeature(
        name="speqtrum", dependencies=["definitely-not-installed-dist-xyz"], symbols=[Symbol(path="unused", name="SpeQtrum")],
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
    symbol = imported.symbols["SpeQtrum"]

    with pytest.raises(OptionalDependencyError) as excinfo:
        symbol.login()

    assert "Using SpeQtrum.login requires installing the 'speqtrum' optional feature" in str(excinfo.value)
    assert "pip install qilisdk[speqtrum]" in str(excinfo.value)
