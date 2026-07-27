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

from solvers_module import solve_with_simulated_annealing  # ty:ignore[unresolved-import]

from qilisdk.core import Model
from qilisdk.core.variables import BaseVariable, Number, RealNumber

from .base_solver import ClassicalSolver


class SimulatedAnnealingSolver(ClassicalSolver):
    """
    Classical solver that uses simulated annealing, implemented in C++.
    This solves a :class:`~qilisdk.core.model.QUBO` model and rejects others.

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import SimulatedAnnealingSolver

            model = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
            results, sample = SimulatedAnnealingSolver(num_reads=100).solve(model.to_qubo())
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

    def solve(self, model: Model) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:
        """Solve the given QUBO by annealing it in C++.

        Args:
            model: The ``QUBO`` instance to solve. Typed as ``Model`` to keep the
                :class:`ClassicalSolver` interface, but anything other than a ``QUBO`` is rejected, so
                a general ``Model`` must be converted with ``to_qubo()`` first.

        Returns:
            tuple[dict[str, Number], dict[BaseVariable, RealNumber]]: a tuple of
            (results dict mapping objective/constraint labels to their evaluated values,
            sample dict mapping each binary variable to its value in the best solution found).

        Raises:
            ValueError: if the given model is not a QUBO, or if the annealing settings are invalid.
        """
        beta_min, beta_max = self.beta_range if self.beta_range is not None else (0.0, 0.0)
        return solve_with_simulated_annealing(
            qubo=model,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            beta_min=beta_min,
            beta_max=beta_max,
            seed=self.seed,
            num_threads=self.num_threads,
        )
