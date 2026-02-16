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

import numpy as np
import pytest

import qilisdk
from qilisdk.analog import Hamiltonian, PauliZ, Schedule
from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core import ket
from qilisdk.functionals import Sampling, TimeEvolution
from qilisdk.noise import Dephasing, NoiseModel


def test_qilisim_init():
    backend = QiliSim()
    assert backend.solver_params is not None

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
    with pytest.raises(ValueError, match=r"atol must be a positive float"):
        QiliSim(atol=-1)


class QiliSimMock:
    def __init__(self, *args, **kwargs): ...

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


def test_time_dependent_hamiltonian_bad_observable():
    backend = QiliSim(seed=42, num_threads=1)
    o = 1.0
    dt = 0.5
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    obs = [
        3.5,  # invalid observable
    ]

    with pytest.raises(ValueError, match=r"Observable type not recognized."):
        backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))


def test_qilisim_dephasing_strength_changes_dynamics():
    initial_state = (ket(0) + ket(1)).unit()
    schedule = Schedule(
        dt=0.1,
        hamiltonians={"h": pauli_z(0)},
        coefficients={"h": {(0, 20): 0}},
    )

    weak_noise = NoiseModel()
    weak_noise.add(Dephasing(t_phi=1e6), qubits=[0])
    weak_backend = QiliSim(noise_model=weak_noise, seed=42, num_threads=1)
    weak_result = weak_backend.execute(TimeEvolution(schedule=schedule, initial_state=initial_state, observables=[pauli_x(0)]))

    strong_noise = NoiseModel()
    strong_noise.add(Dephasing(t_phi=10), qubits=[0])
    strong_backend = QiliSim(noise_model=strong_noise, seed=42, num_threads=1)
    strong_result = strong_backend.execute(
        TimeEvolution(schedule=schedule, initial_state=initial_state, observables=[pauli_x(0)])
    )

    weak_exp = float(np.real(weak_result.final_expected_values[0]))
    strong_exp = float(np.real(strong_result.final_expected_values[0]))
    assert strong_exp < weak_exp
