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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qilisdk.common.model import QUBO, Model
from qilisdk.common.quantum_objects import QuantumObject, expect_val
from qilisdk.cost_functions.cost_function import CostFunction

if TYPE_CHECKING:
    from qilisdk.common.variables import Number
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


class ModelCostFunction(CostFunction):

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def _compute_cost_time_evolution(self, results: TimeEvolutionResult) -> Number:
        if results.final_state is None:
            if len(self.model.variables()) != len(results.expected_values):
                raise ValueError("Mapping samples to the model's variables is ambiguous.")
            variable_map = {v: results.expected_values[i] for i, v in enumerate(self.model.variables())}
            cost = self.model.evaluate(variable_map)
            total_cost = sum(cost.values())
            if total_cost.imag == 0:
                return total_cost.real
            return total_cost

        if isinstance(self.model, QUBO):
            ham = self.model.to_hamiltonian()
            return expect_val(QuantumObject(ham.to_matrix()), results.final_state)

        total_cost = complex(0.0)

        if results.final_state.is_density_matrix():
            rho = results.final_state.dense
            atol = 1e-12
            if not np.allclose(rho, rho.conj().T, atol=1e-10):
                raise ValueError("rho must be Hermitian (within numerical tolerance).")

            vals, vecs = np.linalg.eigh(rho)  # ascending eigenvalues
            vals = np.clip(vals, 0.0, None)
            s = vals.sum()
            if s <= atol:
                raise ValueError("rho has (near-)zero trace after clipping; not a valid density matrix.")
            probs = vals / s

            for vec, prob in zip(vecs, probs):
                variable_map = {v: vec[i] for i, v in enumerate(self.model.variables())}
                evaluate_results = self.model.evaluate(variable_map)
                total_cost += sum(v for v in evaluate_results.values()) * prob
            if total_cost.imag == 0:
                return total_cost.real
            return total_cost

        dense_state = None
        if results.final_state.is_ket():
            dense_state = results.final_state.dense.T[0]
        elif results.final_state.is_bra():
            dense_state = results.final_state.dense[0]

        if dense_state is None:
            raise ValueError("The final state is invalid.")

        n = len(self.model.variables())

        for i, prob in enumerate(dense_state):
            state = [int(b) for b in f"{i:0{n}b}"]
            variable_map = {v: state[i] for i, v in enumerate(self.model.variables())}
            evaluate_results = self.model.evaluate(variable_map)
            total_cost += sum(v for v in evaluate_results.values()) * prob

        if total_cost.imag == 0:
            return total_cost.real
        return total_cost

    def _compute_cost_sampling(self, results: SamplingResult) -> Number:
        total_cost = complex(0.0)
        for sample, prob in results.get_probabilities():
            bit_configuration = [int(i) for i in sample]
            if len(self.model.variables()) != len(bit_configuration):
                raise ValueError("Mapping samples to the model's variables is ambiguous.")
            variable_map = {v: bit_configuration[i] for i, v in enumerate(self.model.variables())}
            evaluate_results = self.model.evaluate(variable_map)
            total_cost += sum(v for v in evaluate_results.values()) * prob

        if total_cost.imag == 0:
            return total_cost.real
        return total_cost
