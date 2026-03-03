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

import numpy as np
import pytest

from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Y as pauli_y
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.qilisim import AnalogMethod, ExecutionConfig, MonteCarloConfig, QiliSim
from qilisdk.core.qtensor import ket
from qilisdk.digital import Circuit, H, X
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

simulation_types = ["direct", "arnoldi", "integrate"]


def _execution(seed: int | None) -> ExecutionConfig:
    return ExecutionConfig(seed=seed, num_threads=1)


def _analog_method(method: str, monte_carlo: bool = False) -> AnalogMethod:
    monte_carlo_config = MonteCarloConfig(trajectories=100) if monte_carlo else None
    if method == "direct":
        return AnalogMethod.direct(monte_carlo=monte_carlo_config)
    if method == "arnoldi":
        return AnalogMethod.arnoldi(monte_carlo=monte_carlo_config)
    return AnalogMethod.integrator(monte_carlo=monte_carlo_config)


def test_seed_same():
    backend1 = QiliSim(execution_config=_execution(seed=42))
    backend2 = QiliSim(execution_config=_execution(seed=42))
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result1 = backend1.execute(Sampling(circuit=circuit, nshots=100))
    result2 = backend2.execute(Sampling(circuit=circuit, nshots=100))
    assert result1.samples == result2.samples


def test_no_seed():
    backend = QiliSim(execution_config=_execution(seed=None))
    seed = backend.get_config()["seed"]
    assert isinstance(seed, int)
    assert 0 <= seed < 2**15


def test_monte_carlo_circuit():
    p = 0.2
    initial_state_1 = ket(0).unit()
    initial_state_2 = ket(1).unit()
    initial_state = (initial_state_1.to_density_matrix() * (1 - p) + initial_state_2.to_density_matrix() * p).unit()
    backend = QiliSim(
        analog_simulation_method=AnalogMethod.integrator(monte_carlo=MonteCarloConfig(trajectories=100)),
        execution_config=_execution(seed=42),
    )
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result = backend._execute_sampling(Sampling(circuit=circuit, nshots=100), initial_state=initial_state)
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "0" in samples
    assert "1" in samples


def test_seed_different():
    backend1 = QiliSim(execution_config=_execution(seed=42))
    backend2 = QiliSim(execution_config=_execution(seed=43))
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result1 = backend1.execute(Sampling(circuit=circuit, nshots=100))
    result2 = backend2.execute(Sampling(circuit=circuit, nshots=100))
    assert result1.samples != result2.samples


@pytest.mark.parametrize("method", simulation_types)
def test_row_vec_ordering(method):
    dt = 0.5
    T = 100

    coeff = (1 + 1j) / np.sqrt(2)
    hamiltonian = coeff * pauli_x(0) + np.conj(coeff) * pauli_y(0)

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": hamiltonian},
        coefficients={"h1": {(0, T): 1}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    psi0 = psi0.to_density_matrix()
    obs = [
        pauli_y(0),  # measure y
    ]

    backend = QiliSim(analog_simulation_method=_analog_method(method), execution_config=_execution(seed=42))
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    assert res.final_state.shape == (2, 2)

    # check that it's hermitian
    final_rho = res.final_state.dense()
    assert np.allclose(final_rho, final_rho.conj().T, rtol=1e-6)


@pytest.mark.parametrize("method", simulation_types)
def test_monte_carlo_time_evolution(method):
    o = 1.0
    dt = 0.1
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = ((ket(0) - ket(1)).unit()).to_density_matrix()
    psi1 = ket(1).to_density_matrix()
    mix = 0.2
    psi0 = ((1 - mix) * psi0 + mix * psi1).unit()
    obs = [
        pauli_z(0),  # measure z
    ]

    backend = QiliSim(
        analog_simulation_method=_analog_method(method, monte_carlo=True),
        execution_config=_execution(seed=42),
    )
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert res.final_state.shape == (2, 2)
    assert np.isclose(expect_z, -0.8, rtol=1e-1)


def test_exponential_gates():
    backend = QiliSim(execution_config=_execution(seed=42))
    circuit = Circuit(nqubits=1)
    circuit.add(X(0).exponential())
    result = backend.execute(Sampling(circuit=circuit, nshots=100))
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "1" in samples
    assert "0" in samples
