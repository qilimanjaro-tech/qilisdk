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

from qilisdk.analog import Schedule
from qilisdk.analog import X as PauliX
from qilisdk.analog import Y as PauliY
from qilisdk.analog import Z as PauliZ
from qilisdk.backends import CudaBackend
from qilisdk.core import Parameter, QTensor, ket, tensor_prod
from qilisdk.core.interpolator import Interpolation
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import CNOT, RX, H, X
from qilisdk.functionals import Sampling, TimeEvolution
from qilisdk.noise_models import (
    AnalogAmplitudeDampingNoise,
    AnalogDephasingNoise,
    AnalogDepolarizingNoise,
    DigitalAmplitudeDampingNoise,
    DigitalBitFlipNoise,
    DigitalDephasingNoise,
    DigitalDepolarizingNoise,
    DissipationNoise,
    KrausNoise,
    NoiseModel,
    ParameterNoise,
)


def test_bit_flip_cuda():
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    c.add(X(1))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(DigitalBitFlipNoise(probability=p, affected_qubits=[1], affected_gates=[]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |1> state should flip to |0>
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.1)


def test_digital_dephasing_cuda():
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(H(0))
    c.add(X(0))
    c.add(H(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(DigitalDephasingNoise(probability=p, affected_qubits=[0], affected_gates=[X]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |+> state should flip to |-> (which maps to |1> after the basis change)
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.1)


def test_depolarizing_cuda():
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(DigitalDepolarizingNoise(probability=p, affected_qubits=[0], affected_gates=[X]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |1> state should flip to |0> or |1> with equal chance
    prob_10 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_10, p * 0.5, atol=0.1)


def test_amplitude_damping_cuda():
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    gamma = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(DigitalAmplitudeDampingNoise(gamma=gamma, affected_qubits=[0], affected_gates=[]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability gamma, the |1> state should decay to |0>
    prob_00 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_00, gamma, atol=0.1)


def test_kraus_noise_singe_qubit_cuda():
    # Define the random circuit and sampler
    shots = 10000
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
    nm.add(KrausNoise(kraus_operators=kraus_ops, affected_qubits=[1], affected_gates=[]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |1> state should flip to |0>
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.1)


def test_kraus_noise_two_qubit_cuda():
    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    p = 0.3
    c = Circuit(nqubits=nqubits)
    random.seed(42)
    c.add(X(0))
    c.add(CNOT(0, 1))
    sampler = Sampling(c, nshots=shots)
    ops = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    ops = [QTensor(K) for K in ops]
    kraus_ops = [tensor_prod([op, op]) for op in ops]
    kraus_ops = [np.sqrt(1 - p) * kraus_ops[0], np.sqrt(p) * kraus_ops[1]]

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(KrausNoise(kraus_operators=kraus_ops, affected_qubits=[0, 1], affected_gates=[X]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |11> state should flip to |00>
    prob_00 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_00, p, atol=0.1)


def test_analog_depolarizing_zero_cuda():
    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.00
    noise_model.add(AnalogDepolarizingNoise(gamma=gamma, affected_qubits=[0]))

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert np.isclose(results.final_expected_values[0], -1.0, atol=1e-2)


def test_analog_dissapation_cuda():
    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.1
    noise_model.add(
        DissipationNoise(jump_operators=[(gamma) ** 0.5 * QTensor(np.array([[0, 1], [0, 0]]))], affected_qubits=[0])
    )

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.8


def test_analog_depolarizing_cuda():
    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.1
    noise_model.add(AnalogDepolarizingNoise(gamma=gamma, affected_qubits=[0]))

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.8


def test_analog_amplitude_damping_cuda():
    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.1
    noise_model.add(AnalogAmplitudeDampingNoise(gamma=gamma, affected_qubits=[0]))

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.8


def test_analog_dephasing_cuda():
    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(PauliZ(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    gamma = 0.1
    noise_model.add(AnalogDephasingNoise(gamma=gamma, affected_qubits=[0]))

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.8


def test_parameter_noise_analog_all_cuda():
    np.random.Generator = np.random.default_rng(42)
    random.seed(42)

    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    param = Parameter("a", 1.0)
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(param * PauliZ(i) + (1 - param) * PauliX(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    param_noise = ParameterNoise(noise_std=3.0)
    noise_model.add(param_noise)

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.95


def test_parameter_noise_analog_named_cuda():
    np.random.Generator = np.random.default_rng(42)
    random.seed(42)

    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    param = Parameter("a", 1.0)
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(param * PauliZ(i) + (1 - param) * PauliX(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    param_noise = ParameterNoise(noise_std=3.0, affected_parameters=["a"])
    noise_model.add(param_noise)

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] > -0.95


def test_parameter_noise_analog_no_named_cuda():
    np.random.Generator = np.random.default_rng(42)
    random.seed(42)

    # Define the time evolution
    T = 10.0
    dt = 0.1
    nqubits = 1
    param = Parameter("a", 1.0)
    Hx = sum(PauliX(i) for i in range(nqubits))
    Hz = sum(param * PauliZ(i) + (1.0 - param) * PauliX(i) for i in range(nqubits))
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
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

    # Define the noise model
    noise_model = NoiseModel()
    param_noise = ParameterNoise(noise_std=3.0, affected_parameters=["b"])
    noise_model.add(param_noise)

    # Execute with the backend
    backend = CudaBackend()
    results = backend.execute(time_evolution, noise_model=noise_model)

    assert results.final_expected_values[0] < -0.9


def test_parameter_noise_digital_cuda():
    np.random.Generator = np.random.default_rng(42)
    random.seed(42)

    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    param = Parameter("theta", np.pi / 2)
    c = Circuit(nqubits=nqubits)
    c.add(RX(qubit=0, theta=param))
    sampler = Sampling(c, nshots=shots)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(ParameterNoise(noise_std=1.0, affected_parameters=[param.label]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With parameter noise, the rotation angle will vary, leading to different measurement outcomes
    prob_10 = res.samples.get("10", 0) / shots
    assert prob_10 > 0.6
