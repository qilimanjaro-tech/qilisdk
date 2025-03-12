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
from typing import Callable, ClassVar

import cudaq
import numpy as np
from cudaq import State
from cudaq.operator import OperatorSum, ScalarOperator
from cudaq.operator import Schedule as cuda_schedule
from cudaq.operator import evolve, spin

from qilisdk.analog.analog_result import AnalogResults
from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator, PauliX, PauliY, PauliZ
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule
from qilisdk.common.backend import AnalogBackend


class CudaqBackend(AnalogBackend):
    _PAULI_MAP: ClassVar[dict] = {PauliZ: spin.z, PauliX: spin.x, PauliY: spin.y, PauliI: spin.i}

    def _hamiltonian_to_cuda(self, hamiltonian: Hamiltonian) -> OperatorSum:
        out = None
        for offset, terms in hamiltonian:
            if out is None:
                out = offset * np.prod([self._PAULI_MAP[pauli.__class__](pauli.qubit) for pauli in terms])
            else:
                out += offset * np.prod([self._PAULI_MAP[pauli.__class__](pauli.qubit) for pauli in terms])
        return out

    def evolve(
        self,
        schedule: Schedule,
        initial_state: QuantumObject,
        observables: list[PauliOperator | Hamiltonian],
        **kwargs: dict,
    ) -> AnalogResults:
        cudaq.set_target("dynamics")

        cuda_ham = None
        steps = np.linspace(0, schedule.T, int(schedule.T / schedule.dt))

        def parameter_values(time_steps: np.ndarray) -> cuda_schedule:

            def compute_value(param_name: str, step_idx: int) -> float:

                return schedule.get_coefficient(time_steps[int(step_idx)], param_name)

            return cuda_schedule(list(range(len(time_steps))), list(schedule.hamiltonians), compute_value)

        cuda_sched = parameter_values(steps)

        def get_schedule(key: str) -> Callable:
            return lambda **args: args[key]

        cuda_ham = sum(
            ScalarOperator(get_schedule(key)) * self._hamiltonian_to_cuda(ham)
            for key, ham in schedule.hamiltonians.items()
        )

        cuda_obs = []
        for obs in observables:
            if isinstance(obs, PauliOperator):
                cuda_obs.append(self._PAULI_MAP[obs.__class__](obs.qubit))
            elif isinstance(obs, Hamiltonian):
                cuda_obs.append(self._hamiltonian_to_cuda(obs))
            else:
                raise ValueError("unsupported type")

        evolution_result = evolve(
            cuda_ham,
            dict.fromkeys(range(schedule.nqubits), 2),
            cuda_sched,
            State.from_data(np.array(initial_state.dense, dtype=np.complex128)),
            observables=cuda_obs,
            collapse_operators=[],
            store_intermediate_results=kwargs.get("store_intermediate_results", False),
        )
        return AnalogResults(
            final_expected_values=np.array(
                [exp_val.expectation() for exp_val in evolution_result.final_expectation_values()]
            ),
            expected_values=(
                np.array(
                    [[val.expectation() for val in exp_vals] for exp_vals in evolution_result.expectation_values()]
                )
                if evolution_result.expectation_values() is not None
                else None
            ),
        )
