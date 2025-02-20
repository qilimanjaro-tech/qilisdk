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
from typing import Any, Callable


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
        feature_name (str):
            The name of the extras group in pyproject.toml
        distributions (list[str]):
            The PyPI distribution names that must be installed
        import_path (str):
            The Python import path to the module containing the real symbols
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
        feature_name (str):
            A label for the feature (e.g. 'qibo-backend').
        symbol_map (dict[str, Union[Any, Callable]]):
            A mapping from symbol name to the real or stubbed object.
            If the required dependency is missing, the symbol is a stub
            that raises an error when called.
    """

    name: str
    symbols: dict[str, Any | Callable]


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

    if missing:
        # Build stubs that raise a helpful error
        def make_stub(symbol_name: str) -> Callable:
            def _stub(*args: Any, **kwargs: Any) -> None:
                raise OptionalDependencyError(
                    f"Using {symbol_name} requires installing the '{feature.name}' optional feature: `pip install qilisdk[{feature.name}]`\n"
                )

            _stub.__name__ = symbol_name
            return _stub

        stubs = {symbol.name: make_stub(symbol.name) for symbol in feature.symbols}
        return ImportedFeature(name=feature.name, symbols=stubs)

    # All dependencies are present => import the real module
    symbols = {}
    for symbol in feature.symbols:
        module = importlib.import_module(symbol.path)
        symbols[symbol.name] = getattr(module, symbol.name)
    return ImportedFeature(name=feature.name, symbols=symbols)
