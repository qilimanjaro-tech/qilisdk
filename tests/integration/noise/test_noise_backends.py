# Copyright 2026 Qilimanjaro Quantum Tech
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

import random

import numpy as np
import pytest

pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
)

from qilisdk.analog import Schedule
from qilisdk.analog import X as PauliX
from qilisdk.analog import Z as PauliZ
from qilisdk.analog.hamiltonian import PauliY
from qilisdk.backends import CudaBackend, QiliSim
from qilisdk.core import Parameter, ket
from qilisdk.core.interpolator import Interpolation
from qilisdk.core.qtensor import QTensor, tensor_prod
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import RX, H, I, X, Z
from qilisdk.functionals import Sampling, TimeEvolution
from qilisdk.noise import (
    AmplitudeDamping,
    BitFlip,
    Dephasing,
    Depolarizing,
    NoiseModel,
    OffsetPerturbation,
    PauliChannel,
    ReadoutAssignment,
)
from qilisdk.noise.representations import KrausChannel, LindbladGenerator

backends = [QiliSim, CudaBackend]
args_per_backend = {QiliSim: {"seed": 42, "num_threads": 1}, CudaBackend: {}}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_bit_flip_sampling(backend_class):
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0))

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(Sampling(circuit, nshots=100))

    assert result.samples == {"0": 100}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_bit_flip_two_qubits_sampling(backend_class):
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    circuit.add(X(1))

    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), qubits=[0])

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(Sampling(circuit, nshots=100))

    assert result.samples == {"01": 100}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_bit_flip_only_identity(backend_class):
    circuit = Circuit(nqubits=2)
    circuit.add(I(0))
    circuit.add(X(1))

    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), gate=I)

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(Sampling(circuit, nshots=100))

    assert result.samples == {"11": 100}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_bit_flip_gate_and_qubit(backend_class):
    circuit = Circuit(nqubits=2)
    circuit.add(Z(0))
    circuit.add(I(0))
    circuit.add(Z(1))
    circuit.add(I(1))

    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), gate=I, qubits=[1])

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(Sampling(circuit, nshots=100))

    assert result.samples == {"01": 100}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_static_kraus_sampling(backend_class):
    shots = 100
    circuit = Circuit(nqubits=1)
    circuit.add(I(0))
    sampler = Sampling(circuit, nshots=shots)

    noise_model = NoiseModel()
    noise_model.add(PauliChannel(pX=1.0))

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(sampler)

    assert result.samples == {"1": shots}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_gate_parameter_perturbation(backend_class):
    shots = 100
    circuit = Circuit(nqubits=1)
    circuit.add(RX(0, theta=0.0))
    sampler = Sampling(circuit, nshots=shots)

    noise_model = NoiseModel()
    noise_model.add(OffsetPerturbation(offset=np.pi), gate=RX, parameter="theta")

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(sampler)

    assert result.samples == {"1": shots}


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_time_evolution_amplitude_damping(backend_class):
    total_time = 1.0
    schedule = Schedule(
        hamiltonians={"hz": PauliZ(0)},
        coefficients={"hz": {0.0: 1.0, total_time: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=ket(1),
        observables=[PauliZ(0)],
        store_intermediate_results=False,
    )

    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(t1=0.1))

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(time_evolution)

    assert result.final_expected_values[0] > 0.9


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_time_evolution_dephasing(backend_class):
    total_time = 1.0
    schedule = Schedule(
        hamiltonians={"hz": PauliZ(0)},
        coefficients={"hz": {0.0: 1.0, total_time: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    initial_state = (ket(0) + ket(1)).unit()
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[PauliX(0)],
        store_intermediate_results=False,
    )

    noise_model = NoiseModel()
    noise_model.add(Dephasing(t_phi=0.1))

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(time_evolution)

    assert abs(result.final_expected_values[0]) < 0.1


@pytest.mark.parametrize("backend_class", backends)
def test_qilisim_backend_schedule_parameter_perturbation(backend_class):
    total_time = np.pi / 2
    coupling = Parameter("g", 0.0)
    schedule = Schedule(
        hamiltonians={"hz": coupling * PauliZ(0)},
        coefficients={"hz": {0.0: 1.0, total_time: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    initial_state = (ket(0) + ket(1)).unit()
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[PauliX(0)],
        store_intermediate_results=False,
    )

    noise_model = NoiseModel()
    noise_model.add(OffsetPerturbation(offset=1.0), parameter="g")

    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    result = backend.execute(time_evolution)

    assert result.final_expected_values[0] < -0.8


@pytest.mark.parametrize("backend_class", backends)
def test_depolarizing_noise(backend_class):
    # Define the random circuit and sampler
    shots = 1000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(Depolarizing(probability=p), qubits=[0], gate=X)

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    # With a probability p, the |1> state should flip to |0> or |1> with equal chance
    prob_10 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_10, p * 0.5, atol=0.2)


@pytest.mark.parametrize("backend_class", backends)
def test_digital_dephasing_noise(backend_class):
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    t_phi = 0.7
    p = 0.5 * (1 - np.exp(-1.0 / t_phi))
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(H(0))
    c.add(X(0))
    c.add(H(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(Dephasing(t_phi=t_phi), qubits=[0])

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    # With a probability p, the |+> state should flip to |-> (which maps to |1> after the basis change)
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.2)


@pytest.mark.parametrize("backend_class", backends)
def test_amplitude_damping_noise(backend_class):
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    t1 = 0.7
    p = 1 - np.exp(-1.0 / t1)
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(AmplitudeDamping(t1=t1), qubits=[0])

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    # With a probability gamma, the |1> state should decay to |0>
    prob_00 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_00, p, atol=0.2)


@pytest.mark.parametrize("backend_class", backends)
def test_kraus_noise_single_qubit_noise(backend_class):
    # Define the random circuit and sampler
    shots = 1000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    c.add(X(1))
    sampler = Sampling(c, nshots=shots)
    kraus_ops = [np.sqrt(1 - p) * np.array([[1, 0], [0, 1]]), np.sqrt(p) * np.array([[0, 1], [1, 0]])]
    kraus_ops = [QTensor(K) for K in kraus_ops]

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(KrausChannel(operators=kraus_ops), qubits=[1])

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    # With a probability p, the |1> state should flip to |0>
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.2)

@pytest.fixture
def time_evolution():
    T = 10.0
    dt = 0.1
    nqubits = 1
    h_x = sum(PauliX(i) for i in range(nqubits))
    h_z = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": h_x, "problem": h_z},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
        interpolation=Interpolation.LINEAR,
    )
    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[PauliZ(0), PauliX(0), PauliY(0)],
        store_intermediate_results=False,
    )
    return time_evolution

@pytest.mark.parametrize("backend_class", backends)
def test_analog_dissapation_noise(backend_class, time_evolution):

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.1
    op = QTensor(np.array([[0, 1], [0, 0]]))
    rate = gamma
    noise_model.add(LindbladGenerator(jump_operators=[op], rates=[rate]), qubits=[0])

    # Execute with the backend
    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    results = backend.execute(time_evolution)

    assert results.final_expected_values[0] > -0.8


@pytest.mark.parametrize("backend_class", backends)
def test_analog_amplitude_damping_noise(backend_class, time_evolution):

    # Define the noise model
    noise_model = NoiseModel()
    t1 = 0.1
    noise_model.add(AmplitudeDamping(t1=t1), qubits=[0])

    # Execute with the backend
    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    results = backend.execute(time_evolution)

    assert results.final_expected_values[0] > -0.8


@pytest.mark.parametrize("backend_class", backends)
def test_analog_dephasing_noise(backend_class, time_evolution):

    # Define the noise model
    noise_model = NoiseModel()
    t_phi = 0.1
    noise_model.add(Dephasing(t_phi=t_phi), qubits=[0])

    # Execute with the backend
    backend = backend_class(noise_model=noise_model, **args_per_backend[backend_class])
    results = backend.execute(time_evolution)

    assert results.final_expected_values[0] > -0.8


@pytest.mark.parametrize("backend_class", backends)
def test_readout_error_noise_01(backend_class):
    # Define the random circuit and sampler
    shots = 100
    nqubits = 1
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(I(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(ReadoutAssignment(p01=1.0, p10=0.0), qubits=[0])

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    assert res.samples == {"1": shots}


@pytest.mark.parametrize("backend_class", backends)
def test_readout_error_qilisim_10(backend_class):
    # Define the random circuit and sampler
    shots = 100
    nqubits = 1
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(ReadoutAssignment(p01=0.0, p10=1.0), qubits=[0])

    # Execute with QiliSim backend
    backend = backend_class(noise_model=nm, **args_per_backend[backend_class])
    res = backend.execute(sampler)

    assert res.samples == {"0": shots}
