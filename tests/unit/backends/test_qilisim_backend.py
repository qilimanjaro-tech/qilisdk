import pytest

from qilisdk.backends.qilisim import QiliSim


@pytest.fixture
def backend():
    return QiliSim()


def test_qilisim_init():
    backend = QiliSim()
    assert backend is not None
