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

from qilisdk.core import QTensor
from qilisdk.noise import KrausChannel, LindbladGenerator
from qilisdk.noise.protocols import AttachmentScope


def test_kraus_channel_properties():
    operator = QTensor(np.eye(2, dtype=complex))
    channel = KrausChannel([operator])

    assert channel.operators == [operator]
    assert channel.as_kraus() is channel

    scopes = channel.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes


def test_lindblad_generator_properties():
    jump_operator = QTensor(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))
    hamiltonian = QTensor(np.eye(2, dtype=complex))

    generator = LindbladGenerator([jump_operator])
    assert generator.jump_operators == [jump_operator]
    assert generator.jump_operators_with_rates == [jump_operator]
    assert generator.rates is None
    assert generator.hamiltonian is None

    generator = LindbladGenerator([jump_operator], rates=[0.5], hamiltonian=hamiltonian)
    assert generator.jump_operators == [jump_operator]
    assert generator.jump_operators_with_rates == [np.sqrt(0.5) * jump_operator]
    assert generator.rates == [0.5]
    assert generator.hamiltonian is hamiltonian
    assert generator.as_lindblad() is generator

    scopes = generator.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE not in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT not in scopes
