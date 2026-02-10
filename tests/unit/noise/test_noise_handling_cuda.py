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


import numpy as np
import pytest

from qilisdk.analog import PauliX as pauli_x
from qilisdk.analog import Schedule
from qilisdk.backends.cuda_backend import CudaBackend, _to_cuda_noise, cudaq
from qilisdk.core import Parameter
from qilisdk.core.qtensor import QTensor
from qilisdk.digital import RX, Circuit, X
from qilisdk.noise import (
    AmplitudeDamping,
    BitFlip,
    Dephasing,
    Depolarizing,
    KrausChannel,
    LindbladGenerator,
    NoiseModel,
    OffsetPerturbation,
    PauliChannel,
    PhaseFlip,
    ReadoutAssignment,
)


def test_to_cuda_noise():
    assert isinstance(_to_cuda_noise(noise=BitFlip(probability=0.5), gate_duration=1.0), cudaq.KrausChannel)
    assert isinstance(_to_cuda_noise(noise=PhaseFlip(probability=0.5), gate_duration=1.0), cudaq.KrausChannel)
    assert isinstance(_to_cuda_noise(noise=Depolarizing(probability=0.5), gate_duration=1.0), cudaq.KrausChannel)
    assert isinstance(_to_cuda_noise(noise=PauliChannel(pX=0.2, pY=0.3, pZ=0.1), gate_duration=1.0), cudaq.KrausChannel)
    ops = [
        QTensor(np.array([[1, 0], [0, np.sqrt(0.8)]])),
        QTensor(np.array([[0, np.sqrt(0.2)], [0, 0]])),
    ]
    assert isinstance(_to_cuda_noise(noise=KrausChannel(operators=ops), gate_duration=1.0), cudaq.KrausChannel)
    assert isinstance(_to_cuda_noise(noise=AmplitudeDamping(t1=1.0), gate_duration=1.0), cudaq.KrausChannel)
    assert isinstance(_to_cuda_noise(noise=Dephasing(t_phi=1.0), gate_duration=1.0), cudaq.KrausChannel)
    assert _to_cuda_noise(noise="bad noise", gate_duration=1.0) is None


def test_handle_readout_errors():
    cudaq_results = {"01": 100}
    noise_model = NoiseModel()
    noise_model.add(ReadoutAssignment(p01=0.0, p10=1.0))
    noise_model.add(ReadoutAssignment(p01=1.0, p10=0.0), qubits=[1])
    adjusted_results = CudaBackend._handle_readout_errors(cudaq_results, noise_model, nqubits=2)
    assert adjusted_results != cudaq_results
    total_counts = sum(adjusted_results.values())
    assert total_counts == 100
    assert all(bitstring in ["00", "01", "10", "11"] for bitstring in adjusted_results)
    assert adjusted_results["10"] == 100


def test_no_readout_errors():
    cudaq_results = {"01": 100}
    noise_model = NoiseModel()
    adjusted_results = CudaBackend._handle_readout_errors(cudaq_results, noise_model, nqubits=2)
    assert adjusted_results == cudaq_results


def test_noise_model_to_cudaq():
    backend = CudaBackend()
    noise_model = NoiseModel()
    single_qubit_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0], [0, 1]])),
        ]
    )
    two_qubit_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ]
    )
    noise_model.add(single_qubit_kraus)
    noise_model.add(single_qubit_kraus, qubits=[1])
    noise_model.add(single_qubit_kraus, qubits=[0], gate=X)
    noise_model.add(single_qubit_kraus, gate=X)
    noise_model.add(two_qubit_kraus)
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model, nqubits=2)
    assert len(cuda_noise_model.get_channels("x", [0])) == 3
    assert len(cuda_noise_model.get_channels("x", [1])) == 3


def test_bad_kraus():
    bad_kraus = KrausChannel(
        operators=[
            QTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.5]])),
        ]
    )
    with pytest.raises(RuntimeError, match="are not completely positive"):
        _ = _to_cuda_noise(noise=bad_kraus, gate_duration=1.0)


def test_parameter_perturbations():
    backend = CudaBackend()
    noise_model = NoiseModel()

    circuit = Circuit(1)
    param1 = Parameter("test1", 0.5)
    param2 = Parameter("test2", 0.5)
    circuit.add(RX(0, theta=param1))
    circuit.add(RX(0, theta=param2))

    perturb = OffsetPerturbation(offset=0.1)
    noise_model.add(perturb, parameter="test1")
    backend._handle_gate_parameter_perturbations(circuit, noise_model)
    assert np.isclose(circuit.get_parameters()["test1"], 0.6)

    noise_model.add(perturb, parameter="theta", gate=RX)

    backend._handle_gate_parameter_perturbations(circuit, noise_model)

    assert np.isclose(circuit.get_parameters()["test1"], 0.8)
    assert np.isclose(circuit.get_parameters()["test2"], 0.6)


def test_parameter_perturbations_errors():
    backend = CudaBackend()
    noise_model = NoiseModel()

    circuit = Circuit(1)
    param1 = Parameter("test1", 0.5)
    param2 = Parameter("test2", 0.5)
    circuit.add(RX(0, theta=param1))
    circuit.add(RX(0, theta=param2))

    perturb = OffsetPerturbation(offset=0.1)
    noise_model.add(perturb, parameter="test_1")

    with pytest.raises(ValueError, match=r"Perturbing Parameter test_1 that doesn't exist in the circuit."):
        backend._handle_gate_parameter_perturbations(circuit, noise_model)

    noise_model = NoiseModel()
    noise_model.add(perturb, gate=RX, parameter="test1")

    with pytest.raises(ValueError, match=r"Invalid parameter name passed to gate."):
        backend._handle_gate_parameter_perturbations(circuit, noise_model)

    assert np.isclose(circuit.get_parameters()["test1"], 0.5)
    assert np.isclose(circuit.get_parameters()["test2"], 0.5)


def test_schedule_parameter_perturbations():
    backend = CudaBackend()
    dt = 1
    param1 = Parameter("test1", 0.5)
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": param1 * pauli_x(0)},
    )
    perturb = OffsetPerturbation(offset=0.1)
    noise_model = NoiseModel()
    noise_model.add(perturb, parameter=param1)
    backend._handle_schedule_parameter_perturbations(schedule, noise_model)
    assert np.isclose(schedule.get_parameters()["test1"], 0.6)


def test_noise_model_to_cudaq_dynamics():
    backend = CudaBackend()
    noise_model = NoiseModel()
    ham_noise = 0.1 * pauli_x(0)
    time_derived_lindblad = PauliChannel(pX=0.2, pY=0.0, pZ=0.0)
    single_qubit_jump = LindbladGenerator(
        jump_operators=[
            QTensor(np.array([[0, 1], [0, 0]])),
        ],
        hamiltonian=ham_noise,
    )
    two_qubit_jump = LindbladGenerator(
        jump_operators=[
            QTensor(np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
        ]
    )
    noise_model.add(time_derived_lindblad)  # adds 2 jumps (to qubits 0 and 1)
    noise_model.add(time_derived_lindblad, qubits=[1])  # adds 1 jump (to qubit 1)
    noise_model.add(single_qubit_jump)  # adds 2 jumps and the delta (to qubits 0 and 1)
    noise_model.add(single_qubit_jump, qubits=[1])  # adds 1 jump and the delta (to qubit 1)
    noise_model.add(two_qubit_jump)  # adds 1 jump (to both qubits)
    cuda_noise_model = backend._noise_model_to_cudaq_dynamics(noise_model, nqubits=2, dt=1.0)
    assert len(cuda_noise_model[0]) == 7  # jump operators
    assert len(cuda_noise_model[1]) == 2  # hamiltonian deltas
