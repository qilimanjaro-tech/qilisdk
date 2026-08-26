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
import importlib
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from loguru_caplog import loguru_caplog as caplog  # ruff: ignore[unused-import]

from qilisdk import _info, about


def _monkeypatch_all(monkeypatch):
    fake_getGPUs = MagicMock(return_value=[])
    fake_get_cpu_info = MagicMock(return_value={"brand_raw": "Test CPU"})
    fake_virtual_memory = MagicMock(return_value=type("vmem", (object,), {"total": 8 * 1024**3})())
    fake_check_output = MagicMock(return_value=b"test_output")

    monkeypatch.setattr("GPUtil.getGPUs", fake_getGPUs)
    monkeypatch.setattr("cpuinfo.get_cpu_info", fake_get_cpu_info)
    monkeypatch.setattr("psutil.virtual_memory", fake_virtual_memory)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    return [fake_getGPUs, fake_get_cpu_info, fake_virtual_memory, fake_check_output]


def test_about(monkeypatch):
    checks = _monkeypatch_all(monkeypatch)
    about_str = about()
    assert "QiliSDK Version:" in about_str
    assert "CPU AVX2 Support:" in about_str
    for check in checks:
        assert check.called


def test_about_fake_gpu(monkeypatch):
    _monkeypatch_all(monkeypatch)

    fake_gpu = MagicMock()
    fake_gpu.name = "Test GPU"
    fake_gpu.memoryTotal = 8 * 1024
    fake_get_gpus = MagicMock(return_value=[fake_gpu])
    monkeypatch.setattr("GPUtil.getGPUs", fake_get_gpus)

    about_str = about()
    assert "GPU Info: Test GPU" in about_str


def test_about_bad_gpu(monkeypatch):
    _monkeypatch_all(monkeypatch)

    fake_gpu = MagicMock()
    fake_gpu.name = "Test GPU"
    fake_gpu.memoryTotal = 8 * 1024
    fake_get_gpus = MagicMock(return_value=[fake_gpu])
    fake_get_gpus.side_effect = ValueError("GPU info retrieval failed")
    monkeypatch.setattr("GPUtil.getGPUs", fake_get_gpus)

    about_str = about()
    assert "GPU Info: Not Found" in about_str


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


def test_gpu_but_no_nvidia_smi(monkeypatch):
    _monkeypatch_all(monkeypatch)

    fake_gpu = MagicMock()
    fake_gpu.name = "Test GPU"
    fake_gpu.memoryTotal = 8 * 1024
    fake_getGPUs = MagicMock(return_value=[fake_gpu])
    monkeypatch.setattr("GPUtil.getGPUs", fake_getGPUs)

    # Simulate nvidia-smi command failing
    def fake_check_output(args, **kwargs):
        if "nvidia-smi" in args[0]:
            raise subprocess.CalledProcessError(1, args)
        return b""

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    about_str = about()
    assert "GPU Info: Test GPU" in about_str
    assert "nvidia-smi Output:" not in about_str


class FakeKernel32:
    """Stands in for the Windows kernel32 library, reporting whether a feature is present."""

    def __init__(self, present):
        self.present = present

    def IsProcessorFeaturePresent(self, feature):
        assert feature == _info._PF_AVX2
        return int(self.present)


def fake_sysctl(monkeypatch, output):
    """Make the sysctl call used on macOS return the given feature list."""

    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=output, stderr="")

    monkeypatch.setattr(_info.subprocess, "run", run)


@pytest.mark.parametrize(("flags", "expected"), [("fpu AVX2 fma\n", True), ("fpu sse4_2\n", False)])
def test_cpu_has_avx2_reads_proc_cpuinfo(monkeypatch, tmp_path, flags, expected):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(f"processor\t: 0\nflags\t\t: {flags}")
    monkeypatch.setattr(_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_info, "_PROC_CPUINFO", cpuinfo)

    assert _info._cpu_has_avx2() is expected


def test_cpu_has_avx2_assumes_yes_without_proc_cpuinfo(monkeypatch, tmp_path):
    monkeypatch.setattr(_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_info, "_PROC_CPUINFO", tmp_path / "not_there")

    assert _info._cpu_has_avx2() is True


@pytest.mark.parametrize("present", [True, False])
def test_cpu_has_avx2_asks_windows(monkeypatch, present):
    monkeypatch.setattr(_info.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=FakeKernel32(present)), raising=False)

    assert _info._cpu_has_avx2() is present


def test_cpu_has_avx2_assumes_yes_when_not_really_on_windows(monkeypatch):
    monkeypatch.setattr(_info.platform, "system", lambda: "Windows")
    monkeypatch.delattr(ctypes, "windll", raising=False)

    assert _info._cpu_has_avx2() is True


@pytest.mark.parametrize(("features", "expected"), [("AVX2 BMI2\n", True), ("SMEP\n", False)])
def test_cpu_has_avx2_asks_sysctl_on_macos(monkeypatch, features, expected):
    monkeypatch.setattr(_info.platform, "system", lambda: "Darwin")
    fake_sysctl(monkeypatch, features)

    assert _info._cpu_has_avx2() is expected


def test_cpu_has_avx2_assumes_yes_without_sysctl(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(_info.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_info.subprocess, "run", missing)

    assert _info._cpu_has_avx2() is True


def test_cpu_has_avx2_assumes_yes_on_other_platforms(monkeypatch):
    monkeypatch.setattr(_info.platform, "system", lambda: "FreeBSD")

    assert _info._cpu_has_avx2() is True


def test_warn_if_no_avx_warns_on_an_older_cpu(monkeypatch, caplog):  # ruff: ignore[redefined-while-unused]
    monkeypatch.setattr(_info.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_info, "_cpu_has_avx2", lambda: False)

    _info.warn_if_no_avx()

    assert "does not support AVX2" in caplog.text
    assert "no_avx=ON" in caplog.text


def test_warn_if_no_avx_stays_quiet_with_avx2(monkeypatch, caplog):  # ruff: ignore[redefined-while-unused]
    monkeypatch.setattr(_info.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(_info, "_cpu_has_avx2", lambda: True)

    _info.warn_if_no_avx()

    assert not caplog.text


def test_warn_if_no_avx_stays_quiet_on_other_architectures(monkeypatch, caplog):  # ruff: ignore[redefined-while-unused]
    monkeypatch.setattr(_info.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(_info, "_cpu_has_avx2", lambda: False)

    _info.warn_if_no_avx()

    assert not caplog.text
