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

"""Tests for the lazy import of optional backends in :mod:`qilisdk.backends`.

The optional backends (``CudaBackend``/``CudaSamplingMethod`` -> ``cudaq`` and
``QutipBackend`` -> ``qutip``) must only pull in their heavy third-party
dependency the first time the symbol is actually accessed, never merely because
``qilisdk.backends`` (or the default ``QiliSim`` backend) was imported.

The laziness guarantee is checked in a *fresh* subprocess interpreter: the pytest
session itself already imports ``cudaq``/``qutip`` via the other backend test
modules, so an in-process ``sys.modules`` check would be meaningless.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest

import qilisdk.backends as backends


def _run_probe(body: str) -> dict:
    """Run ``body`` in a fresh interpreter and return the JSON dict it prints.

    The probe must print a single ``RESULT:<json>`` line to stdout; any other
    output (logging, warnings) is ignored.
    """
    source = textwrap.dedent(
        """
        import json
        import sys


        def loaded(name):
            return any(mod == name or mod.startswith(name + ".") for mod in sys.modules)

        """
    ) + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise AssertionError(f"probe produced no RESULT line.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")


@pytest.mark.parametrize(
    "import_statement",
    [
        "import qilisdk",
        "import qilisdk.backends",
        "from qilisdk.backends import QiliSim",
        "from qilisdk.backends.qilisim import QiliSim",
    ],
)
def test_importing_qilisdk_does_not_import_optional_backends(import_statement: str) -> None:
    """Neither ``cudaq`` nor ``qutip`` is imported just by importing qilisdk/QiliSim."""
    result = _run_probe(
        f"""
        {import_statement}
        print("RESULT:" + json.dumps({{"cudaq": loaded("cudaq"), "qutip": loaded("qutip")}}))
        """
    )
    assert result["cudaq"] is False
    assert result["qutip"] is False


def test_optional_symbols_advertised_without_importing_dependency() -> None:
    """The lazy symbols are visible via ``__all__``/``dir`` without loading their deps."""
    result = _run_probe(
        """
        import qilisdk.backends as backends

        names = dir(backends)
        print("RESULT:" + json.dumps({
            "all": backends.__all__,
            "in_dir": {name: (name in names) for name in ["CudaBackend", "CudaSamplingMethod", "QutipBackend"]},
            "cudaq": loaded("cudaq"),
            "qutip": loaded("qutip"),
        }))
        """
    )
    for symbol in ("CudaBackend", "CudaSamplingMethod", "QutipBackend"):
        assert symbol in result["all"]
        assert result["in_dir"][symbol] is True
    # Advertising the names must not have triggered the heavy imports.
    assert result["cudaq"] is False
    assert result["qutip"] is False


@pytest.mark.parametrize(
    ("symbol", "dependency"),
    [
        ("QutipBackend", "qutip"),
        ("CudaBackend", "cudaq"),
        ("CudaSamplingMethod", "cudaq"),
    ],
)
def test_accessing_symbol_imports_its_dependency_and_caches(symbol: str, dependency: str) -> None:
    """Accessing an optional symbol triggers its dependency import and caches the symbol."""
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"optional dependency {dependency!r} is not installed")

    result = _run_probe(
        f"""
        import qilisdk.backends as backends

        before = loaded({dependency!r})
        obj = getattr(backends, {symbol!r})
        print("RESULT:" + json.dumps({{
            "before": before,
            "after": loaded({dependency!r}),
            "cached": {symbol!r} in vars(backends),
        }}))
        """
    )
    # It was not loaded before the access, and is loaded right after.
    assert result["before"] is False
    assert result["after"] is True
    # The resolved symbol is cached as a real module attribute (so __getattr__
    # is not consulted again for it).
    assert result["cached"] is True


def test_accessing_qutip_backend_does_not_import_cudaq() -> None:
    """The optional backends are independent: reaching for QuTiP must not load ``cudaq``.

    (The reverse direction is not asserted: ``cudaq`` imports ``qutip`` itself, as
    its own transitive dependency, which is outside qilisdk's control.)
    """
    if importlib.util.find_spec("qutip") is None:
        pytest.skip("optional dependency 'qutip' is not installed")

    result = _run_probe(
        """
        import qilisdk.backends as backends

        _ = backends.QutipBackend
        print("RESULT:" + json.dumps({"qutip": loaded("qutip"), "cudaq": loaded("cudaq")}))
        """
    )
    assert result["qutip"] is True
    assert result["cudaq"] is False


def test_module_getattr_resolves_and_caches_optional_symbol() -> None:
    """The PEP 562 hook returns the symbol and caches it on the module."""
    # Drop any cached value so the lazy hook is guaranteed to run here regardless
    # of whether a previous test already resolved the symbol in this session.
    vars(backends).pop("QutipBackend", None)
    resolved = backends.QutipBackend
    assert resolved is not None
    assert "QutipBackend" in vars(backends)
    # Once cached, normal attribute lookup returns the same object.
    assert backends.QutipBackend is resolved


def test_module_getattr_unknown_attribute_raises() -> None:
    """Non-optional, unknown attributes raise ``AttributeError``."""
    unknown = "TotallyUnknownBackend"
    with pytest.raises(AttributeError, match=unknown):
        getattr(backends, unknown)


def test_module_dir_is_sorted_and_lists_all_symbols() -> None:
    """``dir(qilisdk.backends)`` is sorted and includes eager and lazy symbols."""
    names = dir(backends)
    assert names == sorted(names)
    for symbol in ("QiliSim", "AnalogMethod", "CudaBackend", "CudaSamplingMethod", "QutipBackend"):
        assert symbol in names
