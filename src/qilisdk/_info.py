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

import platform
import subprocess  # noqa: S404
import sys
from math import log2

import cpuinfo
import GPUtil
import psutil


def about() -> str:
    """
    Get information about the QiliSDK installation, including details about the user's system.

    Returns:
        str: A formatted string containing the QiliSDK version and relevant system information.
    """
    from . import __version__  # noqa: PLC0415

    cpu_info = ""
    info = cpuinfo.get_cpu_info()
    brand = info.get("brand_raw", "")
    brand = brand[: brand.find("w/")].strip() if "w/" in brand else brand.strip()
    cpu_info += brand
    ram = psutil.virtual_memory().total / (1024**3)
    ram = round(2 ** round(log2(ram)))
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_name = gpu.name
        gpu_mem = gpu.memoryTotal
        gpu_mem = int(gpu_mem // 1024)

    # Python stuff
    info = ""
    info += "------------------------------------------------------\n"
    info += "             Start of QiliSDK Debug Info\n"
    info += "------------------------------------------------------\n"
    info += f"QiliSDK Version: {__version__}\n"
    info += f"Python Version: {sys.version}\n"

    # Check versions of key dependencies
    try:
        import numpy as np  # noqa: PLC0415

        info += f"Numpy Version: {np.__version__}\n"
    except ImportError:
        info += "Numpy Version: Not installed\n"
    try:
        import scipy  # noqa: PLC0415

        info += f"SciPy Version: {scipy.__version__}\n"
    except ImportError:
        info += "SciPy Version: Not installed\n"
    try:
        import qutip  # noqa: PLC0415

        info += f"QuTiP Version: {qutip.__version__}\n"
    except ImportError:
        info += "QuTiP Version: Not installed\n"
    try:
        import cudaq  # noqa: PLC0415

        info += f"cudaQ Version: {cudaq.__version__}\n"
    except ImportError:
        info += "cudaQ Version: Not installed\n"

    # System info
    info += f"Platform: {platform.system()} {platform.release()} ({platform.version()})\n"
    info += f"Processor: {platform.processor()}\n"
    info += f"CPU Info: {cpu_info}\n"
    info += f"Number of CPU Cores: {psutil.cpu_count(logical=False)}\n"
    info += f"Number of Logical Processors: {psutil.cpu_count(logical=True)}\n"
    info += f"Available Memory: {ram} GB\n"
    if gpus:
        info += f"GPU Info: {gpu_name} with {gpu_mem} GB VRAM\n"
    else:
        info += "GPU Info: None detected\n"

    # C++ compiler info
    try:
        gpp_version = subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT).decode()  # noqa: S607
        info += f"g++ Version: {gpp_version.splitlines()[0]}\n"
    except (subprocess.CalledProcessError, FileNotFoundError):
        info += "g++ Version: Not found\n"
    try:
        clang_version = subprocess.check_output(["clang++", "--version"], stderr=subprocess.STDOUT).decode()  # noqa: S607
        info += f"clang++ Version: {clang_version.splitlines()[0]}\n"
    except (subprocess.CalledProcessError, FileNotFoundError):
        info += "clang++ Version: Not found\n"

    # Check OpenMP support
    try:
        subprocess.check_output(
            ["g++", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],  # noqa: S607
            input="#include <omp.h>\nint main() { return 0; }".encode(),
            stderr=subprocess.STDOUT,
        ).decode()
        info += "OpenMP Support: Yes (g++)\n"
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.check_output(
                ["clang++", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],  # noqa: S607
                input="#include <omp.h>\nint main() { return 0; }".encode(),
                stderr=subprocess.STDOUT,
            ).decode()
            info += "OpenMP Support: Yes (clang++)\n"
        except (subprocess.CalledProcessError, FileNotFoundError):
            info += "OpenMP Support: No\n"

    # Try importing QiliSim
    try:
        from .backends.qilisim import QiliSim  # noqa: PLC0415

        _ = QiliSim()
        info += "QiliSim Import: Success\n"
    except Exception as e:  # noqa: BLE001
        info += f"QiliSim Import: Failed with error: {e}\n"

    # Try importing QTensor
    try:
        from .core.qtensor import ket  # noqa: PLC0415

        _ = ket(0)
        info += "QTensor Import: Success\n"
    except Exception as e:  # noqa: BLE001
        info += f"QTensor Import: Failed with error: {e}\n"

    info += "------------------------------------------------------\n"
    info += "              End of QiliSDK Debug Info\n"
    info += "------------------------------------------------------\n"

    info = info.strip()
    return info.strip()
