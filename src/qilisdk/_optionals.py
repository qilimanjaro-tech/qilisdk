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
from enum import Enum
from typing import Any, Callable, NoReturn

from loguru import logger
from packaging.requirements import Requirement


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is missing, with a custom message."""


@dataclass(frozen=True)
class Symbol:
    path: str
    name: str


class RequirementMode(str, Enum):
    ALL = "all"  # all dependencies required
    ANY = "any"  # any one dependency required


@dataclass(frozen=True)
class DependencyGroup:
    """One alternative group of distributions that satisfies a feature.

    Attributes:
        dists (list[str]):
            PEP 508 requirement strings, so a group can pin a floor as well as a
            name, e.g. ``"cuda-quantum-cu12>=0.14.0"``.
        extra (str):
            The extra the user should install if this group is unsatisfied, if any.
    """

    dists: list[str]
    extra: str = ""


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
    mode: RequirementMode
    dependency_groups: list[DependencyGroup]
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

    def __init__(
        self,
        *,
        symbol_name: str,
        feature_name: str,
        import_error: Exception | None = None,
        install_hint: str | None = None,
        unmet: list[str] | None = None,
    ) -> None:
        self._symbol_name = symbol_name
        self._feature_name = feature_name
        self._import_error = import_error
        self.__name__ = symbol_name
        self.__qualname__ = symbol_name
        self._install_hint = install_hint
        self._unmet = unmet or []

    def _raise(self) -> NoReturn:
        hint = f"`pip install qilisdk[{self._feature_name}]`" if self._install_hint is None else self._install_hint
        message = f"Using {self._symbol_name} requires installing optional dependencies: {hint}\n"
        if self._unmet:
            message += f"Unsatisfied requirements: {', '.join(self._unmet)}\n"
        if self._import_error is None:
            raise OptionalDependencyError(message)
        detail = f"{type(self._import_error).__name__}: {self._import_error}"
        raise OptionalDependencyError(message + f"Import failed with: {detail}\n")

    def __call__(self, *_: Any, **__: Any) -> NoReturn:
        self._raise()

    def __getattr__(self, name: str) -> "_OptionalDependencyStub":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _OptionalDependencyStub(
            symbol_name=f"{self._symbol_name}.{name}",
            feature_name=self._feature_name,
            import_error=self._import_error,
            install_hint=self._install_hint,
            unmet=self._unmet,
        )

    def __repr__(self) -> str:
        return f"<missing optional dependency: {self._symbol_name} (feature {self._feature_name!r})>"


def _group_installed(dists: list[str]) -> tuple[bool, list[str]]:
    """Check a group of requirements against the installed distributions.

    Args:
        dists (list[str]): PEP 508 requirement strings.

    Returns:
        Whether every requirement is satisfied, and a description of each one that
        is not, saying whether it is absent or which version was found instead.
    """
    logger.trace("[Optionals] Checking requirements {}", dists)
    unmet: list[str] = []
    for dist in dists:
        requirement = Requirement(dist)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            logger.debug("[Optionals] Distribution {} not installed", requirement.name)
            unmet.append(f"{dist} (not installed)")
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            logger.debug(
                "[Optionals] Distribution {} is {}, which does not satisfy {}", requirement.name, installed, dist
            )
            unmet.append(f"{dist} (found {installed})")
    return (len(unmet) == 0), unmet


def _install_command(dists: list[str]) -> str:
    """Return the pip command installing ``dists``, quoted so specifiers survive the shell."""
    return "`pip install " + " ".join(f'"{dist}"' for dist in dists) + "`"


def _group_install_hint(group: DependencyGroup, feature: OptionalFeature) -> str:
    """Return the command that satisfies a single dependency group."""
    if group.extra:
        return f"`pip install qilisdk[{group.extra}]`"
    if group.dists:
        # Not shipped under any extra, so point at the upstream distributions.
        return _install_command(group.dists)
    return f"`pip install qilisdk[{feature.name}]`"


def _default_install_hint(feature: OptionalFeature) -> str:
    """Return the command the user should run to satisfy ``feature``."""
    # ANY: every group is an alternative, so show them all.
    if feature.mode is RequirementMode.ANY:
        hints = [_group_install_hint(g, feature) for g in feature.dependency_groups]
        return " or ".join(hints) if hints else f"`pip install qilisdk[{feature.name}]`"

    # ALL: a single command covers every group.
    extra = next((g.extra for g in feature.dependency_groups if g.extra), "")
    if extra:
        return f"`pip install qilisdk[{extra}]`"
    dists = [dist for group in feature.dependency_groups for dist in group.dists]
    return _install_command(dists) if dists else f"`pip install qilisdk[{feature.name}]`"


def import_optional_dependencies(feature: OptionalFeature) -> ImportedFeature:
    """Tries to import a submodule at `feature.import_path` along with the required
    distributions. If successful, returns a dict mapping each symbol to the
    actual imported symbol. Otherwise returns stubs that raise an error on usage.

    Args:
        feature (OptionalFeature): An OptionalFeature instance describing the optional group

    Returns:
        Dict[str, Union[Any, Callable]]: A dict { symbol_name: symbol_or_stub }
    """
    logger.debug("[Optionals] Resolving optional feature {}", feature.name)

    def make_stub(
        symbol_name: str, *, import_error: Exception | None = None, unmet: list[str] | None = None
    ) -> _OptionalDependencyStub:
        return _OptionalDependencyStub(
            symbol_name=symbol_name,
            feature_name=feature.name,
            import_error=import_error,
            install_hint=_default_install_hint(feature),
            unmet=unmet,
        )

    satisfied_group: DependencyGroup | None = None
    missing_by_group: list[tuple[DependencyGroup, list[str]]] = []

    if feature.mode is RequirementMode.ALL:
        # Treat as: must satisfy the (single) group; if multiple provided, require them all.
        all_dists: list[str] = []
        for g in feature.dependency_groups:
            all_dists.extend(g.dists)
        ok, unmet = _group_installed(all_dists)
        if not ok:
            logger.warning(
                "[Optionals] Optional feature {} unavailable, unsatisfied requirements {}", feature.name, unmet
            )
            # stubs
            stubs: dict[str, Any] = {s.name: make_stub(s.name, unmet=unmet) for s in feature.symbols}
            return ImportedFeature(name=feature.name, symbols=stubs)
        satisfied_group = feature.dependency_groups[0] if feature.dependency_groups else None

    else:  # ANY
        for g in feature.dependency_groups:
            ok, unmet = _group_installed(g.dists)
            if ok:
                satisfied_group = g
                break
            missing_by_group.append((g, unmet))

        if satisfied_group is None:
            logger.warning("[Optionals] Optional feature {} unavailable, no dependency group satisfied", feature.name)
            all_unmet = [requirement for _, unmet in missing_by_group for requirement in unmet]
            stubs: dict[str, Any] = {s.name: make_stub(s.name, unmet=all_unmet) for s in feature.symbols}
            return ImportedFeature(name=feature.name, symbols=stubs)

    # All good: import real symbols
    logger.debug("[Optionals] Optional feature {} resolved, importing symbols", feature.name)
    symbols: dict[str, Any | Callable] = {}
    for symbol in feature.symbols:
        try:
            module = importlib.import_module(symbol.path)
            symbols[symbol.name] = getattr(module, symbol.name)
        except Exception as exc:  # ruff: ignore[blind-except]
            logger.warning(
                "[Optionals] Failed to import symbol {} for feature {}, using stub", symbol.name, feature.name
            )
            symbols[symbol.name] = make_stub(symbol.name, import_error=exc)
    return ImportedFeature(name=feature.name, symbols=symbols)
