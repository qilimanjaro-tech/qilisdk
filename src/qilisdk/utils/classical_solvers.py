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

import itertools

from loguru import logger

from qilisdk.core import Model
from qilisdk.core.variables import BaseVariable, BinaryVariable, Number, RealNumber, Variable
from qilisdk.settings import get_settings


def _assert_real(number: complex) -> float:
    if isinstance(number, complex):
        if abs(number.imag) < get_settings().atol:
            return number.real
        raise ValueError("Complex Number encountered when expecting only real values to be present.")
    return number


class ClassicalSolver:
    """Base class for classical solvers."""

    def solve(self, model: Model) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:
        """Solve the given model."""
        raise NotImplementedError("ClassicalSolver is an abstract base class.")


class BruteForceSolver(ClassicalSolver):
    """Classical solver that uses brute-force search.

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import BruteForceSolver

            model = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
            results, sample = BruteForceSolver().solve(model)
    """

    def solve(self, model: Model) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:  # noqa: PLR6301
        """Solve the given model by brute-force enumeration of all variable assignments.

        Binary variables are assigned values from {0, 1}. Any other ``Variable`` is decomposed
        via its encoding: all bit patterns are enumerated and decoded to their representable
        values, so the search covers every value the encoding can express regardless of domain.

        Args:
            model: The ``Model`` instance to solve.

        Returns:
            tuple[dict[str, Number], dict[BaseVariable, RealNumber]]: a tuple of
            (results dict mapping objective/constraint labels to their evaluated values,
            sample dict mapping each variable to its value in the optimal solution).

        Raises:
            ValueError: if the model contains a variable that has no encoding (i.e. is not a
                BinaryVariable or a bounded Variable).
        """
        variables = model.variables()

        domains = []
        for v in variables:
            if isinstance(v, BinaryVariable):
                domains.append([0, 1])
            elif isinstance(v, Variable):
                n_bits = v.num_binary_equivalent()
                seen = set()
                vals = []
                for bits_int in range(2**n_bits):
                    bits = [(bits_int >> b) & 1 for b in range(n_bits)]
                    val = v.evaluate(bits)
                    if val not in seen:
                        seen.add(val)
                        vals.append(val)
                domains.append(vals)
            else:
                raise ValueError(f"Brute-force enumeration is not supported for variable {v} of domain {v.domain}.")

        total_combinations = 1
        for d in domains:
            total_combinations *= len(d)
        if total_combinations > 1024:  # noqa: PLR2004
            logger.warning(
                f"[ClassicalSolvers] Model has {total_combinations} combinations, brute-force enumeration may take a long time."
            )

        best_sample = {}
        best_objective_value = float("inf")
        for values in itertools.product(*domains):
            sample = dict(zip(variables, values))
            results = model.evaluate(sample)
            objective_value = _assert_real(results[model.objective.label])
            penalty = sum(_assert_real(results[c.label]) for c in model.constraints)
            if objective_value + penalty < best_objective_value:
                best_objective_value = objective_value + penalty
                best_sample = sample
        return model.evaluate(best_sample), best_sample
