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
from math import log2, ceil

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

    # Python stuff
    info = ""
    _DIVIDER = "-" * 54
    info += f"{_DIVIDER}\n"
    info += "             Start of QiliSDK Debug Info\n"
    info += f"{_DIVIDER}\n"
    info += f"QiliSDK Version: {__version__}\n"
    info += f"Python Version: {sys.version}\n"

    # Check versions of key dependencies
    try:
        import numpy as np  # noqa: PLC0415

        info += f"Numpy Version: {np.__version__}\n"
    except ImportError:
        info += "Numpy Version: Not Found\n"
    try:
        import scipy  # noqa: PLC0415

        info += f"SciPy Version: {scipy.__version__}\n"
    except ImportError:
        info += "SciPy Version: Not Found\n"
    try:
        import qutip  # noqa: PLC0415

        info += f"QuTiP Version: {qutip.__version__}\n"
    except ImportError:
        info += "QuTiP Version: Not Found\n"
    try:
        import cudaq  # noqa: PLC0415

        info += f"CUDA-Q Version: {cudaq.__version__}\n"
    except ImportError:
        info += "CUDA-Q Version: Not Found\n"

    # System info
    cpu_info = cpuinfo.get_cpu_info()
    ram = round(2 ** ceil(log2(psutil.virtual_memory().total / (1024**3))))
    gpus = GPUtil.getGPUs()
    nvidia_smi_output = None
    cuda_version = None
    nvidia_driver_version = None
    try:
        nvidia_smi_output = subprocess.check_output(  # noqa: S607
            ["nvidia-smi | grep 'Driver'"], shell=True,
            stderr=subprocess.STDOUT
        ).decode()
        cuda_version = nvidia_smi_output.split("CUDA Version:")[-1].split()[0]
        nvidia_driver_version = nvidia_smi_output.split("Driver Version:")[-1].split()[0]
        nvidia_smi_output = nvidia_smi_output.replace("|", "")
        nvidia_smi_output = nvidia_smi_output.strip()
    except (subprocess.CalledProcessError):
        pass
    info += f"Platform: {platform.system()} {platform.release()} ({platform.version()})\n"
    info += f"Processor: {platform.processor()}\n"
    info += f"CPU Info: {cpu_info.get('brand_raw', 'Unknown')}\n"
    info += f"Number of CPU Cores: {psutil.cpu_count(logical=False)}\n"
    info += f"Number of Logical Processors: {psutil.cpu_count(logical=True)}\n"
    info += f"Available Memory: {ram} GB\n"
    if gpus:
        info += f"GPU Info: {gpus[0].name} with {int(gpus[0].memoryTotal // 1024)} GB VRAM\n"
        if nvidia_smi_output:
            info += f"CUDA Version: {cuda_version}\n"
            info += f"NVIDIA Driver Version: {nvidia_driver_version}\n"
    else:
        info += "GPU Info: Not Found\n"

    # Check for g++
    has_gpp = False
    try:
        gpp_version = subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT).decode()  # noqa: S607
        info += f"g++ Version: {gpp_version.splitlines()[0]}\n"
        has_gpp = True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        info += "g++ Version: Not Found\n"
    if has_gpp:
        try:
            subprocess.check_output(
                ["g++", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],  # noqa: S607
                input="#include <omp.h>\nint main() { return 0; }".encode(),
                stderr=subprocess.STDOUT,
            ).decode()
            info += "g++ OpenMP Support: Yes\n"
        except (subprocess.CalledProcessError, FileNotFoundError):
            info += "g++ OpenMP Support: No\n"

    # Check for clang
    has_clang = False
    try:
        clang_command = subprocess.check_output(["ls /usr/bin/clang-[0-9]*"], shell=True, stderr=subprocess.STDOUT).decode().strip()
        clang_version = subprocess.check_output([f"{clang_command} --version"], shell=True, stderr=subprocess.STDOUT).decode()  # noqa: S607
        info += f"clang++ Version: {clang_version.splitlines()[0]}\n"
        has_clang = True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        info += "clang++ Version: Not Found\n"
    if has_clang:
        try:
            subprocess.check_output(
                [f"{clang_command}", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],  # noqa: S607
                input="#include <omp.h>\nint main() { return 0; }".encode(),
                stderr=subprocess.STDOUT,
            ).decode()
            info += "clang++ OpenMP Support: Yes\n"
        except (subprocess.CalledProcessError, FileNotFoundError):
            info += "clang++ OpenMP Support: No\n"

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

    info += f"{_DIVIDER}\n"
    info += "              End of QiliSDK Debug Info\n"
    info += f"{_DIVIDER}\n"

    info = info.strip()
    return info.strip()
