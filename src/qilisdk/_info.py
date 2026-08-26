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

import ctypes
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from math import ceil, log2
from pathlib import Path

import cpuinfo
import GPUtil
import psutil
from loguru import logger

# The AVX2 flags only ever get used when building for x86
_X86_MACHINES = frozenset({"x86", "x86_64", "amd64", "i386", "i486", "i586", "i686"})

# The Windows PF_AVX2_INSTRUCTIONS_AVAILABLE query
_PF_AVX2 = 40

_PROC_CPUINFO = Path("/proc/cpuinfo")


def about() -> str:
    """
    Get information about the QiliSDK installation, including details about the user's system.

    Returns:
        str: A formatted string containing the QiliSDK version and relevant system information.
    """
    from . import __version__  # ruff: ignore[import-outside-top-level]

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
        import numpy as np  # ruff: ignore[import-outside-top-level]

        info += f"Numpy Version: {np.__version__}\n"
    except ImportError:
        info += "Numpy Version: Not Found\n"
    try:
        import scipy  # ruff: ignore[import-outside-top-level]

        info += f"SciPy Version: {scipy.__version__}\n"
    except ImportError:
        info += "SciPy Version: Not Found\n"
    try:
        import qutip  # ruff: ignore[import-outside-top-level]

        info += f"QuTiP Version: {qutip.__version__}\n"
    except ImportError:
        info += "QuTiP Version: Not Found\n"
    try:
        import cudaq  # ruff: ignore[import-outside-top-level]

        info += f"CUDA-Q Version: {cudaq.__version__}\n"
    except ImportError:
        info += "CUDA-Q Version: Not Found\n"

    # System info
    cpu_info = cpuinfo.get_cpu_info()
    ram = round(2 ** ceil(log2(psutil.virtual_memory().total / (1024**3))))
    try:  # This can fail if there are driver issues
        gpus = GPUtil.getGPUs()
    except ValueError:
        gpus = []
    nvidia_smi_output = None
    cuda_version = "Not Found"
    nvidia_driver_version = "Not Found"
    try:
        nvidia_smi_output = subprocess.check_output(  # ruff: ignore[subprocess-popen-with-shell-equals-true]
            ["nvidia-smi | grep 'Driver'"],  # ruff: ignore[start-process-with-partial-path]
            shell=True,
            stderr=subprocess.STDOUT,
        ).decode()
        cuda_version = nvidia_smi_output.split("CUDA Version:")[-1].split()[0]
        nvidia_driver_version = nvidia_smi_output.split("Driver Version:")[-1].split()[0]
        nvidia_smi_output = nvidia_smi_output.replace("|", "")
        nvidia_smi_output = nvidia_smi_output.strip()
    except subprocess.CalledProcessError:
        pass
    info += f"Platform: {platform.system()} {platform.release()} ({platform.version()})\n"
    info += f"Processor: {platform.processor()}\n"
    info += f"CPU Info: {cpu_info.get('brand_raw', 'Unknown')}\n"
    info += f"CPU AVX2 Support: {'Yes' if _cpu_has_avx2() else 'No'}\n"
    info += f"Number of CPU Cores: {psutil.cpu_count(logical=False)}\n"
    info += f"Number of Logical Processors: {psutil.cpu_count(logical=True)}\n"
    info += f"Available Memory: {ram} GB\n"
    if gpus:
        info += f"GPU Info: {gpus[0].name} with {int(gpus[0].memoryTotal // 1024)} GB VRAM\n"
        info += f"CUDA Version: {cuda_version}\n"
        info += f"NVIDIA Driver Version: {nvidia_driver_version}\n"
    else:
        info += "GPU Info: Not Found\n"

    # Check for g++
    has_gpp = False
    try:
        gpp_version = subprocess.check_output(["g++", "--version"], stderr=subprocess.STDOUT).decode()  # ruff: ignore[start-process-with-partial-path]
        info += f"g++ Version: {gpp_version.splitlines()[0]}\n"
        has_gpp = True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        info += "g++ Version: Not Found\n"
    if has_gpp:
        try:
            subprocess.check_output(
                ["g++", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],  # ruff: ignore[start-process-with-partial-path]
                input="#include <omp.h>\nint main() { return 0; }".encode(),
                stderr=subprocess.STDOUT,
            ).decode()
            info += "g++ OpenMP Support: Yes\n"
        except (subprocess.CalledProcessError, FileNotFoundError):
            info += "g++ OpenMP Support: No\n"

    # Check for clang
    has_clang = False
    try:
        clang_command = (
            subprocess.check_output(["ls /usr/bin/clang-[0-9]*"], shell=True, stderr=subprocess.STDOUT).decode().strip()  # ruff: ignore[subprocess-popen-with-shell-equals-true, start-process-with-partial-path]
        )
        clang_version = subprocess.check_output([clang_command, "--version"], stderr=subprocess.STDOUT).decode()  # ruff: ignore[subprocess-without-shell-equals-true]
        info += f"clang++ Version: {clang_version.splitlines()[0]}\n"
        has_clang = True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        info += "clang++ Version: Not Found\n"
    if has_clang:
        try:
            subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true]
                [f"{clang_command}", "-fopenmp", "-x", "c++", "-", "-o", "/dev/null"],
                input="#include <omp.h>\nint main() { return 0; }".encode(),
                stderr=subprocess.STDOUT,
            ).decode()
            info += "clang++ OpenMP Support: Yes\n"
        except (subprocess.CalledProcessError, FileNotFoundError):
            info += "clang++ OpenMP Support: No\n"

    # Try importing QiliSim
    try:
        from .backends.qilisim import QiliSim  # ruff: ignore[import-outside-top-level]

        _ = QiliSim()
        info += "QiliSim Import: Success\n"
    except Exception as e:  # ruff: ignore[blind-except]
        info += f"QiliSim Import: Failed with error: {e}\n"

    # Try importing QTensor
    try:
        from .core.qtensor import ket  # ruff: ignore[import-outside-top-level]

        _ = ket(0)
        info += "QTensor Import: Success\n"
    except Exception as e:  # ruff: ignore[blind-except]
        info += f"QTensor Import: Failed with error: {e}\n"

    info += f"{_DIVIDER}\n"
    info += "              End of QiliSDK Debug Info\n"
    info += f"{_DIVIDER}\n"

    info = info.strip()
    return info.strip()


def _cpu_has_avx2() -> bool:
    """
    Check whether this machine's CPU supports AVX2.

    Returns:
        bool: Whether AVX2 is available, assuming that it is if the platform gives no easy way of asking.
    """
    system = platform.system()
    if system == "Linux":
        try:
            return "avx2" in _PROC_CPUINFO.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            return True
    if system == "Windows":
        try:
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(_PF_AVX2))  # ty:ignore[unresolved-attribute]
        except (AttributeError, OSError):
            return True
    if system == "Darwin":
        try:
            sysctl = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.leaf7_features"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        return "avx2" in sysctl.stdout.lower()
    return True


def warn_if_no_avx() -> None:
    """
    Warn if the C++ modules were probably compiled with instructions this CPU cannot run.

    The modules are compiled for AVX2 + FMA by default, which a pre-2013 CPU has no way of
    executing, so this is worth saying before anything tries to load them.
    """
    if platform.machine().lower() not in _X86_MACHINES or _cpu_has_avx2():
        return
    logger.warning(
        "This CPU does not support AVX2, but the QiliSDK C++ modules are compiled with AVX2 and FMA by default, so "
        "they may be slow or crash outright. If that happens, reinstall QiliSDK from source with those instructions "
        "disabled, for example 'uv sync -Ccmake.define.no_avx=ON' or 'pip install --no-binary qilisdk "
        "-Ccmake.define.no_avx=ON qilisdk'."
    )
