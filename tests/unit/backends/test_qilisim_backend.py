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
from pydantic import ValidationError

import qilisdk
from qilisdk.analog import Hamiltonian, PauliZ, Schedule
from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.backends.qilisim import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig, QiliSim
from qilisdk.core import ket
from qilisdk.functionals import QuantumReservoir, ReservoirLayer, Sampling, TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.noise import Dephasing, NoiseModel


def test_qilisim_init():
    backend = QiliSim()
    config = backend.get_config()

    assert backend.solver_params is not None
    assert config["evolution_method"] == "integrate"
    assert config["num_integrate_substeps"] == 2
    assert config["monte_carlo"] is False
    assert config["num_monte_carlo_trajectories"] == 100
    assert config["max_cache_size"] == 1000
    assert config["num_threads"] >= 1
    assert isinstance(config["seed"], int)
    assert config["atol"] > 0

    copied_config = backend.get_config()
    copied_config["seed"] = -1
    assert backend.get_config()["seed"] != -1


def test_qilisim_config_builders_and_validation():
    analog = AnalogMethod.arnoldi(
        dim=16,
        num_substeps=3,
        monte_carlo=MonteCarloConfig(trajectories=250),
    )
    digital = DigitalMethod.state_vector(max_cache_size=2048)
    execution = ExecutionConfig(seed=42, num_threads=2)

    backend = QiliSim(
        analog_simulation_method=analog,
        digital_simulation_method=digital,
        execution_config=execution,
    )
    config = backend.get_config()

    assert config["evolution_method"] == "arnoldi"
    assert config["arnoldi_dim"] == 16
    assert config["num_arnoldi_substeps"] == 3
    assert config["monte_carlo"] is True
    assert config["num_monte_carlo_trajectories"] == 250
    assert config["max_cache_size"] == 2048
    assert config["num_threads"] == 2
    assert config["seed"] == 42

    with pytest.raises(ValidationError):
        AnalogMethod(evolution_method="invalid_method")
    with pytest.raises(ValidationError):
        AnalogMethod(arnoldi_dim=0)
    with pytest.raises(ValidationError):
        MonteCarloConfig(trajectories=0)
    with pytest.raises(ValidationError):
        DigitalMethod(max_cache_size=-5)
    with pytest.raises(ValidationError):
        ExecutionConfig(seed=-1)
    with pytest.raises(TypeError, match="does not accept positional arguments"):
        ExecutionConfig(1)


def test_qilisim_invalid_config_types():
    with pytest.raises(ValueError, match="not a valid analog simulation method"):
        QiliSim(analog_simulation_method=DigitalMethod())  # type:ignore[arg-type]
    with pytest.raises(ValueError, match="not a valid digital simulation method"):
        QiliSim(digital_simulation_method=AnalogMethod.integrator())  # type:ignore[arg-type]
    with pytest.raises(ValueError, match="not a valid execution configuration"):
        QiliSim(execution_config=AnalogMethod.integrator())  # type:ignore[arg-type]


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
    backend = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1))
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
    weak_backend = QiliSim(noise_model=weak_noise, execution_config=ExecutionConfig(seed=42, num_threads=1))
    weak_result = weak_backend.execute(
        TimeEvolution(schedule=schedule, initial_state=initial_state, observables=[pauli_x(0)])
    )

    strong_noise = NoiseModel()
    strong_noise.add(Dephasing(t_phi=10), qubits=[0])
    strong_backend = QiliSim(noise_model=strong_noise, execution_config=ExecutionConfig(seed=42, num_threads=1))
    strong_result = strong_backend.execute(
        TimeEvolution(schedule=schedule, initial_state=initial_state, observables=[pauli_x(0)])
    )

    weak_exp = float(np.real(weak_result.final_expected_values[0]))
    strong_exp = float(np.real(strong_result.final_expected_values[0]))
    assert strong_exp < weak_exp


def _build_quantum_reservoir_functional() -> QuantumReservoir:
    schedule = Schedule(
        dt=1,
        hamiltonians={"h": pauli_z(0)},
        coefficients={"h": {(0, 10): lambda t: 1 - t / 10}},
    )
    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        observables=[pauli_z(0)],
        qubits_to_reset=[0],
    )
    return QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}, {}],
        store_final_state=True,
        store_intermediate_states=True,
        nshots=10,
    )


def test_execute_quantum_reservoir_qilisim(monkeypatch):
    backend = QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1))
    functional = _build_quantum_reservoir_functional()
    final_density = ket(0).to_density_matrix()
    monkeypatch.setattr(
        "qilisdk.backends.qilisim.QiliSim._execute_time_evolution",
        lambda self, f: TimeEvolutionResult(final_state=final_density),
    )

    result = backend.execute(functional)

    assert result.final_state is not None
    assert len(result.expected_values) == 2
    assert len(result.intermediate_states) == 2
