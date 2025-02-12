import pytest

from qilisdk import QiboBackend


@pytest.mark.qibo_backend
def test_placeholder():
    backend = QiboBackend()
    assert isinstance(backend, QiboBackend)
