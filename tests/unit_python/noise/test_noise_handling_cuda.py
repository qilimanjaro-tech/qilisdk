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
from loguru_caplog import loguru_caplog as caplog  # ruff: ignore[unused-import]

pytest.importorskip("cudaq", reason="CUDA noise tests require the 'cudaq' optional dependency")

from qilisdk.analog import PauliX as pauli_x
from qilisdk.analog import Schedule
from qilisdk.backends.cuda_backend import _SWAP_OP_NAME, CudaBackend, _to_cuda_noise, cudaq
from qilisdk.core import Parameter
from qilisdk.core.qtensor import QTensor
from qilisdk.digital import CNOT, CZ, RX, SWAP, U1, Circuit, X
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
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model, circuit=Circuit(2))
    assert len(cuda_noise_model.get_channels("x", [0])) == 3
    assert len(cuda_noise_model.get_channels("x", [1])) == 3


def test_noise_model_to_cudaq_on_multi_qubit_gates():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(ReadoutAssignment(p01=0.1, p10=0.1))
    noise_model.add(AmplitudeDamping(t1=1.0))
    noise_model.add(KrausChannel(operators=[QTensor(np.eye(4))]), gate=CNOT)
    noise_model.add(BitFlip(probability=1.0), gate=X, qubits=[0])
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model, circuit)

    # CUDA-Q sees the CNOT as an x on the control and the target, and takes a channel on both of them
    # at once: the amplitude damping is embedded on each of them, and the two-qubit Kraus channel
    # attached to the CNOT is applied as it is, neither of which shows up on a plain x. The readout
    # error has no Kraus operators, so it is not applied to a gate at all.
    assert len(cuda_noise_model.get_channels("x", [0])) == 2
    assert len(cuda_noise_model.get_channels("x", [1])) == 1


def test_noise_model_to_cudaq_on_qubit_of_multi_qubit_gates():
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), qubits=[0])
    noise_model.add(PhaseFlip(probability=1.0), gate=CZ)
    circuit = Circuit(3)
    circuit.add(X(1))
    circuit.add(CNOT(0, 1))
    circuit.add(CZ(1, 0))
    circuit.add(SWAP(1, 2))
    cuda_noise_model = backend._noise_model_to_cudaq(noise_model, circuit)

    flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    # On a multi-qubit gate the noise is embedded at the position of the qubit it is attached to,
    # control included, and CUDA-Q orders the qubits of a channel with the gate's first qubit as the
    # last tensor factor. The CNOT has qubit 0 as its control, so first, and the CZ as its target.
    for gate_name, gate_qubits, expected in (
        ("x", [0, 1], np.kron(identity, flip)),
        ("z", [1, 0], np.kron(flip, identity)),
    ):
        channels = cuda_noise_model.get_channels(gate_name, gate_qubits)
        assert len(channels) == 1
        assert np.allclose(np.array(channels[0].get_ops()[0]), expected)

    # The gates that do not act on the noisy qubit are left alone.
    assert cuda_noise_model.get_channels(_SWAP_OP_NAME, [1, 2]) == []

    # A plain z only gets the bit flip of its qubit: the phase flip is attached to the CZ, which
    # CUDA-Q takes as a z with one control qubit.
    plain_z_channels = cuda_noise_model.get_channels("z", [0])
    assert len(plain_z_channels) == 1
    assert plain_z_channels[0].noise_type == cudaq.NoiseModelType.BitFlipChannel


def test_noise_model_to_cudaq_on_gate_without_cuda_operation(caplog):  # ruff: ignore[redefined-while-unused]
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0), gate=U1)
    backend._noise_model_to_cudaq(noise_model, circuit=Circuit(1))

    # CUDA-Q has no u1 operation to attach a channel to, so the noise is dropped and reported.
    assert "Ignoring the noise on gate 'U1'" in caplog.text


def test_noise_model_to_cudaq_warns_when_a_channel_cannot_be_embedded(caplog):  # ruff: ignore[redefined-while-unused]
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(KrausChannel(operators=[QTensor(np.eye(8))]), gate=CNOT)
    circuit = Circuit(3)
    circuit.add(CNOT(0, 1))
    backend._noise_model_to_cudaq(noise_model, circuit)

    # A three-qubit channel fits neither a single qubit of the CNOT nor the gate as a whole, so it is
    # skipped, both when the channel is built and when the gate it was meant for is registered.
    assert "does not act on a single qubit, cannot embed in multi-qubit gate" in caplog.text
    assert "cannot embed in multi-qubit gate X" in caplog.text


def test_noise_model_to_cudaq_warns_when_a_channel_has_no_kraus_operators(caplog):  # ruff: ignore[redefined-while-unused]
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(ReadoutAssignment(p01=0.1, p10=0.1))
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    backend._noise_model_to_cudaq(noise_model, circuit)

    # A readout error defines no Kraus operators, so there is nothing to embed in the CNOT.
    assert "does not define Kraus operators or they do not act on a single qubit" in caplog.text


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
        total_time=10.0,
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


def test_global_single_qubit_lindblad_uses_single_degree_operator(monkeypatch):
    backend = CudaBackend()
    noise_model = NoiseModel()
    noise_model.add(Dephasing(t_phi=1.0))

    expected_dimensions_per_id = {}
    instantiate_calls = []

    def fake_define(id, expected_dimensions, create, override):
        expected_dimensions_per_id[id] = list(expected_dimensions)

    def fake_instantiate(id, degrees):
        degree_list = [degrees] if isinstance(degrees, int) else list(degrees)
        assert len(expected_dimensions_per_id[id]) == len(degree_list)
        instantiate_calls.append((id, degree_list))
        return object()

    monkeypatch.setattr("qilisdk.backends.cuda_backend.operators.define", fake_define)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.operators.instantiate", fake_instantiate)

    jump_operators, hamiltonian_deltas = backend._noise_model_to_cudaq_dynamics(noise_model, nqubits=2, dt=1.0)

    assert len(jump_operators) == 2
    assert len(hamiltonian_deltas) == 0
    assert len(instantiate_calls) == 2
    assert all(expected_dimensions == [2] for expected_dimensions in expected_dimensions_per_id.values())
