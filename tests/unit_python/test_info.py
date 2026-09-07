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

import importlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

from loguru_caplog import loguru_caplog as caplog  # ruff: ignore[unused-import]

from qilisdk import about


def _fake_nvidia_smi(gpus=(), driver_version="595.84", cuda_version="13.2"):
    """Build a subprocess.check_output stand-in that emulates nvidia-smi.

    ``gpus`` is a sequence of (name, total memory in MiB) pairs; an empty sequence
    emulates a machine with no NVIDIA driver installed, where the binary is missing.
    """

    def fake_check_output(args, **kwargs):
        if args[0] != "nvidia-smi":
            return b"test_output"
        if not gpus:
            raise FileNotFoundError(2, "No such file or directory: 'nvidia-smi'")
        if any(arg.startswith("--query-gpu") for arg in args):
            return "\n".join(f"{name}, {memory}" for name, memory in gpus).encode()
        return (
            f"| NVIDIA-SMI {driver_version}    Driver Version: {driver_version}    CUDA Version: {cuda_version}     |\n"
        ).encode()

    return MagicMock(side_effect=fake_check_output)


def _monkeypatch_all(monkeypatch, gpus=()):
    fake_get_cpu_info = MagicMock(return_value={"brand_raw": "Test CPU"})
    fake_virtual_memory = MagicMock(return_value=type("vmem", (object,), {"total": 8 * 1024**3})())
    fake_check_output = _fake_nvidia_smi(gpus)

    monkeypatch.setattr("cpuinfo.get_cpu_info", fake_get_cpu_info)
    monkeypatch.setattr("psutil.virtual_memory", fake_virtual_memory)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    return [fake_get_cpu_info, fake_virtual_memory, fake_check_output]


def test_about(monkeypatch):
    checks = _monkeypatch_all(monkeypatch)
    about_str = about()
    assert "QiliSDK Version:" in about_str
    for check in checks:
        assert check.called


def test_about_fake_gpu(monkeypatch):
    _monkeypatch_all(monkeypatch, gpus=[("Test GPU", 8 * 1024)])

    about_str = about()
    assert "GPU Info: Test GPU with 8 GB VRAM" in about_str
    assert "CUDA Version: 13.2" in about_str
    assert "NVIDIA Driver Version: 595.84" in about_str


def test_about_no_gpu(monkeypatch):
    _monkeypatch_all(monkeypatch)

    about_str = about()
    assert "GPU Info: Not Found" in about_str


def test_about_gpu_driver_error(monkeypatch):
    _monkeypatch_all(monkeypatch)

    # nvidia-smi is installed but fails, as it does when the driver is broken
    def fake_check_output(args, **kwargs):
        if args[0] == "nvidia-smi":
            raise subprocess.CalledProcessError(1, args)
        return b"test_output"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    about_str = about()
    assert "GPU Info: Not Found" in about_str


def test_about_gpu_without_memory(monkeypatch):
    # A GPU whose reported memory cannot be parsed is still named
    _monkeypatch_all(monkeypatch, gpus=[("Test GPU", "[N/A]")])

    about_str = about()
    assert "GPU Info: Test GPU\n" in about_str


def test_about_subprocess_fails(monkeypatch):
    _monkeypatch_all(monkeypatch)

    fake_check_output = MagicMock(side_effect=subprocess.CalledProcessError(1, "cmd"))
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    about_str = about()
    assert "g++ Version: Not Found" in about_str
    assert "clang++ Version: Not Found" in about_str


def test_about_bad_imports(monkeypatch):
    # Remove cached modules so imports are re-evaluated
    modules_to_remove = ["numpy", "scipy", "qutip", "cudaq"]
    for mod in modules_to_remove:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    # Make each import raise ImportError
    with patch.dict(
        sys.modules,
        {
            "numpy": None,
            "scipy": None,
            "qutip": None,
            "cudaq": None,
        },
    ):
        # Re-import about so it runs with the patched sys.modules
        import qilisdk  # ruff: ignore[import-outside-top-level]

        importlib.reload(qilisdk)
        from qilisdk import about  # ruff: ignore[import-outside-top-level]

        about_str = about()

    assert "Numpy Version: Not Found" in about_str
    assert "SciPy Version: Not Found" in about_str
    assert "QuTiP Version: Not Found" in about_str
    assert "CUDA-Q Version: Not Found" in about_str


def test_about_qilisim_bad_init(monkeypatch):
    _monkeypatch_all(monkeypatch)

    fake_qilisim = MagicMock()
    fake_qilisim.QiliSim.side_effect = Exception("Initialization failed")
    fake_qtensor = MagicMock()
    fake_qtensor.ket.side_effect = Exception("Import failed")

    with patch.dict(
        sys.modules,
        {
            "qilisdk.backends.qilisim": fake_qilisim,
            "qilisdk.core.qtensor": fake_qtensor,
        },
    ):
        import qilisdk  # ruff: ignore[import-outside-top-level]

        importlib.reload(qilisdk)
        from qilisdk import about  # ruff: ignore[import-outside-top-level]

        about_str = about()
    assert "QiliSim Import: Failed with error:" in about_str
    assert "QTensor Import: Failed with error:" in about_str


def test_about_gpp_but_no_openmp(monkeypatch):
    _monkeypatch_all(monkeypatch)

    # Simulate g++ present but no OpenMP support
    def fake_check_output(args, **kwargs):
        if "-fopenmp" in args:
            raise subprocess.CalledProcessError(1, args)
        return b"g++ (GCC) 9.3.0\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    about_str = about()
    assert "g++ OpenMP Support: No" in about_str


def test_gpu_but_no_version_info(monkeypatch):
    _monkeypatch_all(monkeypatch, gpus=[("Test GPU", 8 * 1024)])

    # The GPU query succeeds but the human-readable output carries no versions
    def fake_check_output(args, **kwargs):
        if args[0] != "nvidia-smi":
            return b"test_output"
        if any(arg.startswith("--query-gpu") for arg in args):
            return b"Test GPU, 8192"
        return b""

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    about_str = about()
    assert "GPU Info: Test GPU with 8 GB VRAM" in about_str
    assert "CUDA Version: Not Found" in about_str
    assert "NVIDIA Driver Version: Not Found" in about_str
