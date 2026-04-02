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
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number
    from qilisdk.functionals.functional_result import FunctionalResult  # type: ignore[type-arg]


class ModelCostFunction(CostFunction):
    """Evaluate the cost of a ``FunctionalResult`` with respect to a :class:`~qilisdk.core.model.Model`.

    The model encodes an objective function (and optional constraints) over
    binary variables. This cost function maps a ``FunctionalResult`` -- obtained
    from a ``DigitalPropagation`` or ``AnalogEvolution`` -- onto a scalar by
    evaluating the model against either the final quantum state or the sampled
    probability distribution.

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
        """Initialise a ``ModelCostFunction``.

        Args:
            model (Model): Classical model describing objective and constraints.
        """
        super().__init__()
        self._model = model

    @property
    def model(self) -> Model:
        """Return the underlying optimisation model.

        Returns:
            Model: The :class:`~qilisdk.core.model.Model` instance passed at
            construction time.
        """
        return self._model

    def compute_cost(self, results: FunctionalResult) -> Number:
        """Compute the cost from a ``FunctionalResult``.

        Uses the final state if available (via ``StateTomography``), otherwise
        falls back to sampling-based estimation.

        Args:
            results (FunctionalResult): The result from executing a functional.

        Returns:
            Number: Cost value computed from the results.

        Raises:
            ValueError: If ``results`` contains neither a ``StateTomography``
                nor a ``Sampling`` readout.
        """
        if results.has_state():
            return self._compute_from_state(results)
        if results.has_samples():
            return self._compute_from_samples(results)
        raise ValueError("ModelCostFunction requires either a StateTomography or Sampling readout in the results.")

    def _compute_from_state(self, results: FunctionalResult) -> Number:
        """Compute the cost using the full final quantum state.

        Handles ket, bra, and density-matrix representations. For
        :class:`~qilisdk.core.model.QUBO` models the Hamiltonian expectation
        value is computed directly; for general models each computational-basis
        component is evaluated individually.

        Args:
            results (FunctionalResult): A result whose ``final_state`` is
                available.

        Returns:
            Number: The computed cost value.

        Raises:
            ValueError: If the final state is neither a ket, bra, nor a valid
                density matrix.
        """
        final_state = results.state

        if isinstance(self.model, QUBO):
            ham = self.model.to_hamiltonian()
            total_cost = complex(
                np.real_if_close(expect_val(QTensor(ham.to_matrix()), final_state), tol=get_settings().atol)
            )
            if abs(total_cost.imag) < get_settings().atol:
                return total_cost.real
            return total_cost

        total_cost = complex(0.0)

        if final_state.is_density_matrix(tol=1e-5):
            rho = final_state.dense()
            n = final_state.nqubits
            for i in range(rho.shape[0]):
                state = [int(b) for b in f"{i:0{n}b}"]
                _ket_state = ket(*state)
                _prob = complex(
                    np.real_if_close(
                        np.trace((_ket_state @ _ket_state.adjoint()).dense() @ rho), tol=get_settings().atol
                    )
                )
                variable_map = {v: int(state[i]) for i, v in enumerate(self.model.variables())}
                evaluate_results = self.model.evaluate(variable_map)
                total_cost += sum(evaluate_results.values()) * _prob
            if abs(total_cost.imag) < get_settings().atol:
                return total_cost.real
            return total_cost

        dense_state = None
        if final_state.is_ket():
            dense_state = final_state.dense().T[0]
        elif final_state.is_bra():
            dense_state = final_state.dense()[0]

        if dense_state is None:
            raise ValueError("The final state is invalid.")

        n = len(self.model.variables())

        for i, prob in enumerate(dense_state):
            state = [int(b) for b in f"{i:0{n}b}"]
            variable_map = {v: state[i] for i, v in enumerate(self.model.variables())}
            evaluate_results = self.model.evaluate(variable_map)
            total_cost += sum(evaluate_results.values()) * np.abs(prob**2)

        total_cost = complex(np.real_if_close(total_cost, tol=get_settings().atol))
        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def _compute_from_samples(self, results: FunctionalResult) -> Number:
        """Compute the cost from sampled probability distributions.

        Each bitstring sample is mapped to the model's variables and the
        model is evaluated; the total cost is the probability-weighted sum
        of those evaluations.

        Args:
            results (FunctionalResult): A result whose ``probabilities``
                are available.

        Returns:
            Number: The probability-weighted cost value.

        Raises:
            ValueError: If the number of qubits in a sample does not match
                the number of model variables.
        """
        total_cost = complex(0.0)
        probabilities = results.probabilities
        for sample, prob in probabilities.items():
            bit_configuration = [int(i) for i in sample]
            if len(self.model.variables()) != len(bit_configuration):
                raise ValueError("Mapping samples to the model's variables is ambiguous.")
            variable_map = {v: bit_configuration[i] for i, v in enumerate(self.model.variables())}
            evaluate_results = self.model.evaluate(variable_map)
            total_cost += sum(v for v in evaluate_results.values()) * prob

        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def __repr__(self) -> str:
        """Return a string representation of this cost function.

        Returns:
            str: A string of the form ``ModelCostFunction(model=...)``.
        """
        return f"ModelCostFunction(model={self.model})"
