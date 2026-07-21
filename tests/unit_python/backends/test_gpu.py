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
"""Unit tests for the optional CUDA-library preloading shim (``backends/_gpu.py``).

These exercise the pure-Python preload logic on the CPU without needing a GPU,
by faking the nvidia wheel layout and ``ctypes.CDLL`` loading.
"""

import importlib.util

import pytest

from qilisdk.backends import _gpu
from qilisdk.backends import qilisim as qilisim_mod
from qilisdk.backends.backend_config import ExecutionConfig
from qilisdk.backends.qilisim import QiliSim


@pytest.fixture(autouse=True)
def _reset_preloaded():
    """Reset the module-level preload cache around every test."""
    _gpu._preloaded = False
    yield
    _gpu._preloaded = False


# --- _nvidia_lib_dirs() ---


def test_nvidia_lib_dirs_skips_missing_components(monkeypatch):
    # A component whose spec cannot be found (ImportError) is skipped, not fatal.
    def raise_import_error(_name):
        raise ImportError

    monkeypatch.setattr(importlib.util, "find_spec", raise_import_error)
    assert _gpu._nvidia_lib_dirs() == []


def test_nvidia_lib_dirs_skips_specs_without_locations(monkeypatch):
    spec = type("Spec", (), {"submodule_search_locations": None})()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: spec)
    assert _gpu._nvidia_lib_dirs() == []


def test_nvidia_lib_dirs_collects_existing_lib_dir(monkeypatch, tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    spec = type("Spec", (), {"submodule_search_locations": [str(tmp_path)]})()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: spec)
    dirs = _gpu._nvidia_lib_dirs()
    assert str(lib_dir) in dirs


# --- preload_cuda_libraries() ---


def test_idempotent_when_already_preloaded():
    _gpu._preloaded = True
    assert _gpu.preload_cuda_libraries() is True


def test_returns_false_on_non_linux(monkeypatch):
    monkeypatch.setattr(_gpu.sys, "platform", "darwin")
    assert _gpu.preload_cuda_libraries() is False


def test_returns_false_without_lib_dirs(monkeypatch):
    monkeypatch.setattr(_gpu.sys, "platform", "linux")
    monkeypatch.setattr(_gpu, "_nvidia_lib_dirs", list)
    assert _gpu.preload_cuda_libraries() is False


def test_loads_discovered_libraries(monkeypatch, tmp_path):
    # Two fake shared objects that the fake loader accepts.
    (tmp_path / "liba.so").touch()
    (tmp_path / "libb.so.12").touch()
    monkeypatch.setattr(_gpu.sys, "platform", "linux")
    monkeypatch.setattr(_gpu, "_nvidia_lib_dirs", lambda: [str(tmp_path)])
    loaded: list[str] = []
    monkeypatch.setattr(_gpu.ctypes, "CDLL", lambda path, mode: loaded.append(path))

    assert _gpu.preload_cuda_libraries() is True
    assert len(loaded) == 2
    # Cache is set so a second call short-circuits.
    assert _gpu.preload_cuda_libraries() is True


def test_unmet_dependencies_return_false(monkeypatch, tmp_path):
    (tmp_path / "liba.so").touch()
    monkeypatch.setattr(_gpu.sys, "platform", "linux")
    monkeypatch.setattr(_gpu, "_nvidia_lib_dirs", lambda: [str(tmp_path)])

    def always_fail(path, mode):
        raise OSError("unmet dependency")

    monkeypatch.setattr(_gpu.ctypes, "CDLL", always_fail)
    # No library loads -> the no-progress break + debug log path -> False.
    assert _gpu.preload_cuda_libraries() is False


# --- QiliSim gpu=True integration (covers the backend dispatch branch) ---


def test_qilisim_gpu_request_invokes_preload(monkeypatch):

    called = {"n": 0}

    def fake_preload():
        called["n"] += 1
        return False  # exercise the "relying on system CUDA" debug branch

    monkeypatch.setattr(_gpu, "preload_cuda_libraries", fake_preload)
    QiliSim(execution_config=ExecutionConfig(gpu=True))
    assert called["n"] == 1
    assert qilisim_mod is not None
