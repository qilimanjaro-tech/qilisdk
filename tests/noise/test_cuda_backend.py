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

pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
)

from qilisdk.analog import Schedule, Z
from qilisdk.analog import X as PauliX
from qilisdk.backends.cuda_backend import CudaBackend
from qilisdk.core import Parameter, ket
from qilisdk.core.interpolator import Interpolation
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import RX, I, X
from qilisdk.functionals import Sampling, TimeEvolution
from qilisdk.noise import AmplitudeDamping, BitFlip, Dephasing, NoiseModel, OffsetPerturbation, PauliChannel


def test_cuda_backend_bit_flip_sampling():
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    noise_model = NoiseModel()
    noise_model.add(BitFlip(probability=1.0))

    backend = CudaBackend()
    result = backend.execute(Sampling(circuit), noise_model=noise_model)

    assert result.samples == {"0": 1000}


def test_cuda_backend_static_kraus_sampling():
    shots = 100
    circuit = Circuit(nqubits=1)
    circuit.add(I(0))
    sampler = Sampling(circuit, nshots=shots)

    noise_model = NoiseModel()
    noise_model.add(PauliChannel(pX=1.0))

    backend = CudaBackend()
    result = backend.execute(sampler, noise_model=noise_model)

    assert result.samples == {"1": shots}


def test_cuda_backend_gate_parameter_perturbation():
    shots = 100
    circuit = Circuit(nqubits=1)
    circuit.add(RX(0, theta=0.0))
    sampler = Sampling(circuit, nshots=shots)

    noise_model = NoiseModel()
    noise_model.add(OffsetPerturbation(offset=np.pi), gate=RX, parameter="theta")

    backend = CudaBackend()
    result = backend.execute(sampler, noise_model=noise_model)

    assert result.samples == {"1": shots}


def test_cuda_backend_time_evolution_amplitude_damping():
    total_time = 1.0
    schedule = Schedule(
        hamiltonians={"hz": Z(0)},
        coefficients={"hz": {0.0: 0.0, total_time: 0.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=ket(1),
        observables=[Z(0)],
        store_intermediate_results=False,
    )

    noise_model = NoiseModel()
    noise_model.add(AmplitudeDamping(T1=0.1))

    backend = CudaBackend()
    result = backend.execute(time_evolution, noise_model=noise_model)

    assert result.final_expected_values[0] > 0.9


def test_cuda_backend_time_evolution_dephasing():
    total_time = 1.0
    schedule = Schedule(
        hamiltonians={"hz": Z(0)},
        coefficients={"hz": {0.0: 0.0, total_time: 0.0}},
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
    noise_model.add(Dephasing(Tphi=0.1))

    backend = CudaBackend()
    result = backend.execute(time_evolution, noise_model=noise_model)

    assert abs(result.final_expected_values[0]) < 0.1


def test_cuda_backend_schedule_parameter_perturbation():
    total_time = np.pi / 2
    coupling = Parameter("g", 0.0)
    schedule = Schedule(
        hamiltonians={"hz": coupling * Z(0)},
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

    backend = CudaBackend()
    result = backend.execute(time_evolution, noise_model=noise_model)

    assert result.final_expected_values[0] < -0.8
