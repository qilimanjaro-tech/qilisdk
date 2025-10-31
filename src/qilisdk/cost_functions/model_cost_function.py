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

from qilisdk.core.model import QUBO, Model
from qilisdk.core.qtensor import QTensor, expect_val, ket
from qilisdk.cost_functions.cost_function import CostFunction

if TYPE_CHECKING:
    from qilisdk.core.variables import Number
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


class ModelCostFunction(CostFunction):
    """
    Evaluate the cost of functional results with respect to a :class:`~qilisdk.core.model.Model`.

    Example:
        .. code-block:: python

            from qilisdk.core import BinaryVariable, Model, LEQ
            from qilisdk.cost_functions import ModelCostFunction

            model = Model("demo")
            x0, x1 = BinaryVariable("x0"), BinaryVariable("x1")
            model.set_objective(x0 + x1)
            model.add_constraint("limit", LEQ(x0 + x1, 1))
            cost_fn = ModelCostFunction(model)
    """

    def __init__(self, model: Model) -> None:
        """
        Args:
            model (Model): Classical model describing objective and constraints.
        """
        super().__init__()
        self._model = model

    @property
    def model(self) -> Model:
        """Return the underlying optimisation model."""
        return self._model

    def _compute_cost_time_evolution(self, results: TimeEvolutionResult) -> Number:
        """
        Compute the expectation value of the model objective using a time-evolution result.

        Evaluates the model on each computational basis state with probability extracted from the final state.

        Returns:
            Number: Expectation value of the model objective.

        Raises:
            ValueError: If the final state is not provided in the results.
        """
        if results.final_state is None:
            raise ValueError(
                "can't compute cost using Models from time evolution results when the state is not provided."
            )

        if isinstance(self.model, QUBO):
            ham = self.model.to_hamiltonian()
            total_cost = complex(np.real_if_close(expect_val(QTensor(ham.to_matrix()), results.final_state)))
            if total_cost.imag == 0:
                return total_cost.real
            return total_cost

        total_cost = complex(0.0)

        if results.final_state.is_density_matrix(tol=1e-5):
            rho = results.final_state.dense
            n = results.final_state.nqubits
            for i in range(rho.shape[0]):
                state = [int(b) for b in f"{i:0{n}b}"]
                _ket_state = ket(*state)
                _prob = complex(np.real_if_close(np.trace((_ket_state @ _ket_state.adjoint()).dense @ rho)))
                variable_map = {v: int(state[i]) for i, v in enumerate(self.model.variables())}
                evaluate_results = self.model.evaluate(variable_map)
                total_cost += sum(v for v in evaluate_results.values()) * _prob
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
            total_cost += sum(v for v in evaluate_results.values()) * np.abs(prob**2)

        total_cost = complex(np.real_if_close(total_cost, tol=1e-12))
        if total_cost.imag == 0:
            return total_cost.real
        return total_cost

    def _compute_cost_sampling(self, results: SamplingResult) -> Number:
        """
        Compute the model cost by averaging over sampled bitstrings.

        Each sample is mapped onto the model variables and evaluated using ``model.evaluate``.

        Returns:
            Number: Average cost of the model objective over all samples.

        Raises:
            ValueError: If the number of model variables does not match the sample size.
        """
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
