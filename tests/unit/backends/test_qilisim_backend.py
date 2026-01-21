from qilisdk.backends.qilisim import QiliSim


def test_qilisim_init():
    backend = QiliSim()
    assert backend is not None
