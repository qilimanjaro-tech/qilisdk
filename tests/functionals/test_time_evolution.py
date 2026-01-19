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

import pytest

from qilisdk.analog.hamiltonian import Hamiltonian, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.functionals.time_evolution import TimeEvolution


def test_time_evolution_initial_state_qubits_mismatch_raises():
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = tensor_prod([ket(0), ket(0)]).unit()

    with pytest.raises(ValueError, match=r"The initial state provided acts on 2 qubits"):
        TimeEvolution(schedule=schedule, observables=[PauliZ(0)], initial_state=initial_state)
