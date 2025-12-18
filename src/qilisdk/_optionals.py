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

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Callable, NoReturn


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is missing, with a custom message."""


@dataclass(frozen=True)
class Symbol:
    path: str
    name: str


@dataclass(frozen=True)
class OptionalFeature:
    """Holds metadata about an optional feature.

    Attributes:
        name (str):
            The name of the extras group in pyproject.toml
        dependencies (list[str]):
            The dependencies that must be installed
        symbols (list[str]):
            Which symbols (classes, functions, etc.) to re-export
    """

    name: str
    dependencies: list[str]
    symbols: list[Symbol]


@dataclass
class ImportedFeature:
    """Holds the result of an optional import.

    Attributes:
        name (str):
            A label for the feature (e.g. 'qibo-backend').
        symbols (dict[str, Union[Any, Callable]]):
            A mapping from symbol name to the real or stubbed object.
            If the required dependency is missing, the symbol is a stub
            that raises an error when called.
    """

    name: str
    symbols: dict[str, Any | Callable]


class _OptionalDependencyStub:
    """Callable proxy that raises OptionalDependencyError for missing extras.

    It also intercepts attribute access so expressions like ``SpeQtrum.login()``
    raise an informative OptionalDependencyError rather than AttributeError.
    """

    def __init__(self, *, symbol_name: str, feature_name: str, import_error: Exception | None = None) -> None:
        self._symbol_name = symbol_name
        self._feature_name = feature_name
        self._import_error = import_error
        self.__name__ = symbol_name
        self.__qualname__ = symbol_name

    def _raise(self) -> NoReturn:
        message = (
            f"Using {self._symbol_name} requires installing the '{self._feature_name}' optional feature: `pip install qilisdk[{self._feature_name}]`\n"
        )
        if self._import_error is None:
            raise OptionalDependencyError(message)
        detail = f"{type(self._import_error).__name__}: {self._import_error}"
        raise OptionalDependencyError(message + f"Import failed with: {detail}\n") from self._import_error

    def __call__(self, *_: Any, **__: Any) -> NoReturn:
        self._raise()

    def __getattr__(self, name: str) -> "_OptionalDependencyStub":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _OptionalDependencyStub(
            symbol_name=f"{self._symbol_name}.{name}",
            feature_name=self._feature_name,
            import_error=self._import_error,
        )

    def __repr__(self) -> str:
        return f"<missing optional dependency: {self._symbol_name} (extra '{self._feature_name}')>"


def import_optional_dependencies(feature: OptionalFeature) -> ImportedFeature:
    """Tries to import a submodule at `feature.import_path` along with the required
    distributions. If successful, returns a dict mapping each symbol to the
    actual imported symbol. Otherwise returns stubs that raise an error on usage.

    Args:
        feature (OptionalFeature): An OptionalFeature instance describing the optional group

    Returns:
        Dict[str, Union[Any, Callable]]: A dict { symbol_name: symbol_or_stub }
    """
    missing: list[str] = []
    for dist in feature.dependencies:
        try:
            importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            missing.append(dist)

    def make_stub(symbol_name: str, *, import_error: Exception | None = None) -> _OptionalDependencyStub:
        return _OptionalDependencyStub(symbol_name=symbol_name, feature_name=feature.name, import_error=import_error)

    if missing:
        # Build stubs that raise a helpful error
        stubs: dict[str, Any] = {symbol.name: make_stub(symbol.name) for symbol in feature.symbols}
        return ImportedFeature(name=feature.name, symbols=stubs)

    # All dependencies are present => import the real module
    symbols = {}
    for symbol in feature.symbols:
        try:
            module = importlib.import_module(symbol.path)
            symbols[symbol.name] = getattr(module, symbol.name)
        except Exception as exc:  # noqa: BLE001
            symbols[symbol.name] = make_stub(symbol.name, import_error=exc)
    return ImportedFeature(name=feature.name, symbols=symbols)
