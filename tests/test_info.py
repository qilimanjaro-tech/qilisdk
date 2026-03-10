from unittest.mock import MagicMock

from loguru_caplog import loguru_caplog as caplog  # noqa: F401

from qilisdk import about


def test_about(monkeypatch):
    fake_getGPUs = MagicMock(return_value=[])
    fake_get_cpu_info = MagicMock(return_value={"brand_raw": "Test CPU"})
    fake_virtual_memory = MagicMock(return_value=type("vmem", (object,), {"total": 8 * 1024**3})())
    fake_check_output = MagicMock(return_value=b"test_output")

    monkeypatch.setattr("GPUtil.getGPUs", fake_getGPUs)
    monkeypatch.setattr("cpuinfo.get_cpu_info", fake_get_cpu_info)
    monkeypatch.setattr("psutil.virtual_memory", fake_virtual_memory)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    about_str = about()
    assert "QiliSDK Version:" in about_str
    assert fake_check_output.called
    assert fake_getGPUs.called
    assert fake_get_cpu_info.called
    assert fake_virtual_memory.called
