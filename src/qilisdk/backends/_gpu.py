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
"""
Runtime preloading of the optional CUDA shared libraries.

On the C++ side we try to load various libraries at runtime, better that
they're found here on the Python side first, since it's more likely
that the environment is set up correctly for Python.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import sys

from loguru import logger

_NVIDIA_COMPONENTS: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.nvjitlink",
    "nvidia.cusparse",
    "nvidia.cublas",
    "nvidia.cusolver",
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

    Returns:
        bool: ``True`` if at least one CUDA library was loaded into the process.
        ``False`` does not preclude GPU use — a system CUDA install on the
        loader path is discovered by the C++ shim without preloading.
    """
    global _preloaded  # noqa: PLW0603 -- module-level cache so the preload runs at most once
    if _preloaded:
        return True

    # Windows not supported
    if not sys.platform.startswith("linux"):
        return False

    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        return False

    # Get the list of all .so files in the lib directories
    pending: list[str] = []
    for lib_dir in lib_dirs:
        pending.extend(sorted(str(p) for p in pathlib.Path(lib_dir).glob("*.so*")))

    # Keep trying to load the libraries until no progress is made
    any_loaded = False
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
