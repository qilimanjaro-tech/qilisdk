# Copyright 2025 Qilimanjaro Quantum Tech
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

from unittest.mock import MagicMock

import pytest

import qilisdk
from qilisdk.analog import Hamiltonian, PauliZ, Schedule
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core import ket
from qilisdk.functionals import Sampling, TimeEvolution


def test_qilisim_init():
    backend = QiliSim()
    assert backend is not None

    with pytest.raises(ValueError, match=r"Unknown time evolution method"):
        QiliSim(evolution_method="invalid_method")
    with pytest.raises(ValueError, match=r"arnoldi_dim must be a positive integer"):
        QiliSim(arnoldi_dim=0)
    with pytest.raises(ValueError, match=r"num_arnoldi_substeps must be a positive integer"):
        QiliSim(num_arnoldi_substeps=-3)
    with pytest.raises(ValueError, match=r"num_integrate_substeps must be a positive integer"):
        QiliSim(num_integrate_substeps=0)
    with pytest.raises(ValueError, match=r"num_monte_carlo_trajectories must be a positive integer"):
        QiliSim(num_monte_carlo_trajectories=-10)
    with pytest.raises(ValueError, match=r"max_cache_size cannot be negative"):
        QiliSim(max_cache_size=-5)


class QiliSimMock:
    def __init__(self, *args, **kwargs):
        pass

    def execute_sampling(self, *args, **kwargs):
        return "mocked_sampling_result"

    def execute_time_evolution(self, *args, **kwargs):
        return "mocked_time_evolution_result"


def test_qilisim_sampling_dummy(monkeypatch):
    monkeypatch.setattr(qilisdk.backends.qilisim, "QiliSimCpp", QiliSimMock)
    circuit = MagicMock()
    sampling = Sampling(circuit)
    backend = QiliSim()
    assert backend.execute(sampling) == "mocked_sampling_result"


def test_qilisim_time_evolution_dummy(monkeypatch):
    monkeypatch.setattr(qilisdk.backends.qilisim, "QiliSimCpp", QiliSimMock)
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    ob = PauliZ(0)
    initial_state = ket(0)
    func = TimeEvolution(schedule=schedule, observables=[ob], initial_state=initial_state)
    backend = QiliSim()
    assert backend.execute(func) == "mocked_time_evolution_result"
