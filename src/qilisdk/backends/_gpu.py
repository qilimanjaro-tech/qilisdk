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
"""Runtime preloading of the optional CUDA shared libraries.

The C++ backend offloads selected linear algebra to cuBLAS/cuSOLVER through a
``dlopen``-based shim (``qilisdk_cpp/backends/qilisim/gpu/cuda_solver.cpp``), so
the wheel has no link-time CUDA dependency and runs unchanged without a GPU.

The ``nvidia-*-cu12`` / ``nvidia-*-cu13`` wheels (pulled by the ``qilisdk[cuda12]``
/ ``qilisdk[cuda13]`` extras) install their ``.so`` files under
``<site-packages>/nvidia/<component>/lib``, which is not on the default loader
search path. We preload them here with ``RTLD_GLOBAL`` so that the C++ shim can
resolve them by soname (``dlopen`` returns already-loaded objects) regardless of
``LD_LIBRARY_PATH``. A system-wide CUDA install on the loader path needs no
preloading and is picked up by the shim directly.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import sys

from loguru import logger

# nvidia wheel components, ordered so dependencies precede dependents. Loading
# is retried regardless, so the order is only a hint.
_NVIDIA_COMPONENTS: tuple[str, ...] = (
    "nvidia.cuda_runtime",  # libcudart
    "nvidia.nvjitlink",  # libnvJitLink (cusolver dependency on CUDA >= 12.3)
    "nvidia.cusparse",  # libcusparse (cusolver dependency)
    "nvidia.cublas",  # libcublas, libcublasLt
    "nvidia.cusolver",  # libcusolver
)

_preloaded = False


def _nvidia_lib_dirs() -> list[str]:
    """Return existing ``nvidia/<component>/lib`` directories for installed CUDA wheels."""
    dirs: list[str] = []
    for component in _NVIDIA_COMPONENTS:
        try:
            spec = importlib.util.find_spec(component)
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            lib_dir = pathlib.Path(location) / "lib"
            if lib_dir.is_dir():
                dirs.append(str(lib_dir))
    return dirs


def preload_cuda_libraries() -> bool:
    """Best-effort preload of the NVIDIA CUDA shared libraries from the pip wheels.

    Safe to call unconditionally and repeatedly: it is a no-op on non-Linux
    platforms, when the CUDA wheels are not installed, or after a successful
    first call. It never raises.

    Returns:
        bool: ``True`` if at least one CUDA library was loaded into the process.
        ``False`` does not preclude GPU use — a system CUDA install on the
        loader path is discovered by the C++ shim without preloading.
    """
    global _preloaded  # noqa: PLW0603 -- module-level cache so the preload runs at most once
    if _preloaded:
        return True
    # The shim only implements dlopen-based loading; Windows is not targeted.
    if not sys.platform.startswith("linux"):
        return False

    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        return False

    # Sonames vary across CUDA 12/13, so glob every shared object and let the
    # retry loop sort out load ordering.
    pending: list[str] = []
    for lib_dir in lib_dirs:
        pending.extend(sorted(str(p) for p in pathlib.Path(lib_dir).glob("*.so*")))

    any_loaded = False
    # At most one pass per library is ever needed once a pass makes no progress.
    for _ in range(len(pending) + 1):
        if not pending:
            break
        still_pending: list[str] = []
        progressed = False
        for path in pending:
            try:
                ctypes.CDLL(path, mode=os.RTLD_GLOBAL | os.RTLD_NOW)
                any_loaded = True
                progressed = True
            except OSError:
                still_pending.append(path)
        pending = still_pending
        if not progressed:
            break  # remaining libraries have unmet dependencies; leave them

    if pending:
        logger.debug("Some CUDA libraries could not be preloaded (unmet dependencies): {}", pending)

    _preloaded = any_loaded
    return any_loaded
