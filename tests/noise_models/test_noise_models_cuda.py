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

from qilisdk.backends.cuda_backend import CudaBackend
from qilisdk.core.qtensor import QTensor, tensor_prod
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import CNOT, H, X
from qilisdk.functionals import Sampling
from qilisdk.noise_models import AmplitudeDampingNoise, BitFlipNoise, DephasingNoise, DepolarizingNoise, NoiseModel
from qilisdk.noise_models.digital_noise import KrausNoise


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
    nm.add(BitFlipNoise(qubit=1, probability=p, affected_gates=[]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |1> state should flip to |0>
    prob_10 = res.samples.get("10", 0) / shots
    assert np.isclose(prob_10, p, atol=0.1)

def test_dephasing_cuda():
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
    nm.add(DephasingNoise(qubit=0, probability=p, affected_gates=[X]))

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
    nm.add(DepolarizingNoise(qubit=0, probability=p, affected_gates=[X]))

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
    nm.add(AmplitudeDampingNoise(qubit=0, gamma=gamma, affected_gates=[X]))

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
    nm.add(KrausNoise(kraus_operators=kraus_ops, affected_qubits=[0,1], affected_gates=[CNOT]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)

    # With a probability p, the |11> state should flip to |00>
    prob_00 = res.samples.get("00", 0) / shots
    assert np.isclose(prob_00, p, atol=0.1)
