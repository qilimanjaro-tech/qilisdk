import importlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

from loguru_caplog import loguru_caplog as caplog  # noqa: F401

from qilisdk import about


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
    for check in checks:
        assert check.called


def test_about_fake_gpu(monkeypatch):

    _monkeypatch_all(monkeypatch)

    fake_gpu = MagicMock()
    fake_gpu.name = "Test GPU"
    fake_gpu.memoryTotal = 8 * 1024
    fake_getGPUs = MagicMock(return_value=[fake_gpu])
    monkeypatch.setattr("GPUtil.getGPUs", fake_getGPUs)

    about_str = about()
    assert "GPU Info: Test GPU" in about_str


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
        import qilisdk  # noqa: PLC0415

        importlib.reload(qilisdk)
        from qilisdk import about  # noqa: PLC0415

        about_str = about()

    assert "Numpy Version: Not Found" in about_str
    assert "SciPy Version: Not Found" in about_str
    assert "QuTiP Version: Not Found" in about_str
    assert "CUDA-Q Version: Not Found" in about_str


# Try importing QiliSim
# try:
#     from .backends.qilisim import QiliSim

#     _ = QiliSim()
#     info += "QiliSim Import: Success\n"
# except Exception as e:
#     info += f"QiliSim Import: Failed with error: {e}\n"

# # Try importing QTensor
# try:
#     from .core.qtensor import ket

#     _ = ket(0)
#     info += "QTensor Import: Success\n"
# except Exception as e:
#     info += f"QTensor Import: Failed with error: {e}\n"


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
        import qilisdk  # noqa: PLC0415

        importlib.reload(qilisdk)
        from qilisdk import about  # noqa: PLC0415

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
