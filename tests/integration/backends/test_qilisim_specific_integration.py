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
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core.qtensor import ket
from qilisdk.digital import CNOT, RX, RY, RZ, U1, U2, U3, Circuit, H, X, Y, Z
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

analog_types = ["direct", "arnoldi", "integrate", "integrate_matrix_free"]
digital_types = ["statevector", "statevector_matrix_free"]


def test_seed_same():
    backend1 = QiliSim(seed=42, num_threads=1)
    backend2 = QiliSim(seed=42, num_threads=1)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result1 = backend1.execute(Sampling(circuit=circuit, nshots=100))
    result2 = backend2.execute(Sampling(circuit=circuit, nshots=100))
    assert result1.samples == result2.samples


def test_no_seed():
    backend = QiliSim(seed=None, num_threads=1)
    seed = backend.solver_params["seed"]
    assert isinstance(seed, int)
    assert 0 <= seed <= 2**15


def test_monte_carlo_circuit():
    p = 0.2
    initial_state_1 = ket(0).unit()
    initial_state_2 = ket(1).unit()
    initial_state = (initial_state_1.to_density_matrix() * (1 - p) + initial_state_2.to_density_matrix() * p).unit()
    backend = QiliSim(monte_carlo=True, num_monte_carlo_trajectories=100, seed=42, num_threads=1)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result = backend._execute_sampling(Sampling(circuit=circuit, nshots=100), initial_state=initial_state)
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "0" in samples
    assert "1" in samples


def test_seed_different():
    backend1 = QiliSim(seed=42, num_threads=1)
    backend2 = QiliSim(seed=43, num_threads=1)
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    result1 = backend1.execute(Sampling(circuit=circuit, nshots=100))
    result2 = backend2.execute(Sampling(circuit=circuit, nshots=100))
    assert result1.samples != result2.samples


@pytest.mark.parametrize("method", analog_types)
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

    backend = QiliSim(evolution_method=method, seed=42, num_threads=1)
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    assert res.final_state.shape == (2, 2)

    # check that it's hermitian
    final_rho = res.final_state.dense()
    assert np.allclose(final_rho, final_rho.conj().T, rtol=1e-6)


@pytest.mark.parametrize("method", analog_types)
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

    backend = QiliSim(evolution_method=method, monte_carlo=True, seed=42, num_threads=1)
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert res.final_state.shape == (2, 2)
    assert np.isclose(expect_z, -0.8, rtol=1e-1)


@pytest.mark.parametrize("method", digital_types)
def test_exponential_gates(method):
    backend = QiliSim(seed=42, num_threads=1, sampling_method=method)
    circuit = Circuit(nqubits=1)
    circuit.add(X(0).exponential())
    result = backend.execute(Sampling(circuit=circuit, nshots=100))
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "1" in samples
    assert "0" in samples


def _counts_similar(counts1, counts2, total_shots, tol=0.1):
    all_keys = set(counts1.keys()) | set(counts2.keys())
    for key in all_keys:
        c1 = counts1.get(key, 0)
        c2 = counts2.get(key, 0)
        if abs(c1 - c2) > tol * total_shots:
            return False
    return True


def test_matrix_free_circuit_versus_normal():
    nqubits = 3
    c = Circuit.random(
        nqubits=nqubits, ngates=1000, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_statevector = QiliSim(seed=42, num_threads=1, sampling_method="statevector")
    backend_matrix_free = QiliSim(seed=42, num_threads=1, sampling_method="statevector_matrix_free")
    res_statevector = backend_statevector.execute(Sampling(circuit=c, nshots=1000))
    res_matrix_free = backend_matrix_free.execute(Sampling(circuit=c, nshots=1000))
    assert _counts_similar(res_statevector.samples, res_matrix_free.samples, total_shots=1000, tol=0.1)


@pytest.mark.parametrize("method", digital_types)
def test_combine_single_qubit_gates(method):
    nqubits = 3
    c = Circuit.random(
        nqubits=nqubits, ngates=1000, single_qubit_gates=[H, X, Y, Z, RX, RY, RZ, U1, U2, U3], two_qubit_gates=[CNOT]
    )
    backend_combined = QiliSim(seed=42, num_threads=1, sampling_method=method, combine_single_qubit_gates=True)
    backend_uncombined = QiliSim(seed=42, num_threads=1, sampling_method=method, combine_single_qubit_gates=False)
    res_combined = backend_combined.execute(Sampling(circuit=c, nshots=1000))
    res_uncombined = backend_uncombined.execute(Sampling(circuit=c, nshots=1000))
    assert _counts_similar(res_combined.samples, res_uncombined.samples, total_shots=1000, tol=0.1)


def test_matrix_free_time_evolution_versus_normal():
    dt = 0.1
    T = 10
    hamiltonian = pauli_x(0) + pauli_z(0)
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": hamiltonian},
        coefficients={"h1": {(0, T): 1}},
    )
    psi0 = (ket(0) - ket(1)).unit()
    obs = [pauli_y(0)]
    backend_normal = QiliSim(seed=42, num_threads=1, evolution_method="integrate")
    backend_matrix_free = QiliSim(seed=42, num_threads=1, evolution_method="integrate_matrix_free")
    res_normal = backend_normal.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))
    res_matrix_free = backend_matrix_free.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))
    assert np.isclose(res_normal.final_expected_values[0], res_matrix_free.final_expected_values[0], rtol=0.01)
