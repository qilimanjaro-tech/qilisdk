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

from solvers_module import solve_with_simulated_annealing  # ty:ignore[import-not-found]

from qilisdk.core import QUBO, Model

from .base_solver import ClassicalSolver, ClassicalSolverResult


class SimulatedAnnealingSolver(ClassicalSolver):
    """
    Classical solver that uses simulated annealing, implemented in C++.
    This solves a :class:`~qilisdk.core.model.QUBO` model, converting any other model with ``to_qubo()`` first.

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import SimulatedAnnealingSolver

            model = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
            result = SimulatedAnnealingSolver(num_reads=100).solve(model)
    """

    def __init__(
        self,
        num_reads: int = 10,
        num_sweeps: int = 1000,
        beta_range: tuple[float, float] | None = None,
        seed: int = 0,
        num_threads: int = 0,
    ) -> None:
        """Create a new simulated annealing based classical solver instance.

        Args:
            num_reads (int, optional): The number of independent anneals to run, the best of which is
                returned. Defaults to 10.
            num_sweeps (int, optional): The number of sweeps over all variables in each anneal.
                Defaults to 1000.
            beta_range (tuple[float, float] | None, optional): The (initial, final) inverse
                temperature to anneal over. If not given, a range is derived from the magnitudes of
                the cost function's coefficients. Defaults to None.
            seed (int, optional): The seed of the random number generators, each read deriving its
                own from it. Defaults to 0.
            num_threads (int, optional): The number of threads to distribute the reads over, or zero
                to let OpenMP decide. Defaults to 0.
        """
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.num_threads = num_threads

    def solve(self, model: Model) -> ClassicalSolverResult:
        """Solve the given model by annealing its QUBO form in C++.

        Args:
            model: The model to solve. A model that is not already a ``QUBO`` is converted with
                ``to_qubo()`` using its default settings, so call ``to_qubo()`` yourself to control
                the penalization or Lagrange multipliers.

        Returns:
            ClassicalSolverResult: the results of the optimization, including the objective value and best solution.

        Raises:
            ValueError: if the given model cannot be converted to a QUBO, or if the annealing settings are invalid.
        """
        if not isinstance(model, QUBO):
            model = model.to_qubo()
        beta_min, beta_max = self.beta_range if self.beta_range is not None else (0.0, 0.0)
        results, sample = solve_with_simulated_annealing(
            qubo=model,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=self.seed,
            num_threads=self.num_threads,
        )
        return ClassicalSolverResult(results, sample, model.objective.label)
