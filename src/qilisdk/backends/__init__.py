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
from typing import Any

from qilisdk._optionals import (
    DependencyGroup,
    ImportedFeature,
    OptionalFeature,
    RequirementMode,
    Symbol,
    import_optional_dependencies,
)
from qilisdk.backends.backend_config import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig
from qilisdk.backends.qilisim import QiliSim

__all__ = [
    "AnalogMethod",
    "DigitalMethod",
    "ExecutionConfig",
    "MonteCarloConfig",
    "QiliSim",
]

OPTIONAL_FEATURES: list[OptionalFeature] = [
    OptionalFeature(
        name="cuda",
        mode=RequirementMode.ANY,
        dependency_groups=[
            DependencyGroup(dists=["cuda-quantum-cu12"], extra="cuda12"),
            DependencyGroup(dists=["cuda-quantum-cu13"], extra="cuda13"),
        ],
        symbols=[
            Symbol(path="qilisdk.backends.cuda_backend", name="CudaBackend"),
            Symbol(path="qilisdk.backends.cuda_backend", name="CudaSamplingMethod"),
        ],
    ),
    OptionalFeature(
        name="qutip",
        mode=RequirementMode.ALL,
        dependency_groups=[
            DependencyGroup(dists=["qutip", "qutip-qip", "matplotlib"], extra="qutip"),
        ],
        symbols=[Symbol(path="qilisdk.backends.qutip_backend", name="QutipBackend")],
    ),
]


# Map every optional symbol name to the feature that provides it. Symbols are
# advertised in ``__all__`` (so ``from qilisdk.backends import *`` and tooling see
# them) but are NOT imported eagerly: the heavy dependency (``cudaq``, ``qutip``,
# ...) is only imported the first time the symbol is actually accessed, via the
# module-level ``__getattr__`` below (PEP 562). Static type info lives in the
# accompanying ``__init__.pyi`` stub.
_OPTIONAL_FEATURE_BY_SYMBOL: dict[str, OptionalFeature] = {
    symbol.name: feature for feature in OPTIONAL_FEATURES for symbol in feature.symbols
}

__all__ += list(_OPTIONAL_FEATURE_BY_SYMBOL)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazily import optional backend symbols on first access (PEP 562).

    This runs only when normal attribute lookup on the module fails, i.e. the
    first time ``name`` is requested. It resolves the whole owning feature, caches
    every resolved symbol as a real module attribute so subsequent accesses skip
    this hook, and returns the requested symbol (or a stub raising
    ``OptionalDependencyError`` if the optional dependency is not installed).

    Args:
        name: The attribute being accessed on the ``qilisdk.backends`` module.

    Returns:
        The imported symbol, or a stub that raises on use when the optional
        dependency is missing.

    Raises:
        AttributeError: If ``name`` is not an optional backend symbol.
    """
    feature = _OPTIONAL_FEATURE_BY_SYMBOL.get(name)
    if feature is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    imported_feature: ImportedFeature = import_optional_dependencies(feature)
    globals().update(imported_feature.symbols)
    return imported_feature.symbols[name]


def __dir__() -> list[str]:
    """Return the module's public attributes, including lazily-imported symbols."""
    return sorted(__all__)
