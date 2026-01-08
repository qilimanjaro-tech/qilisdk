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

import os
from typing import TYPE_CHECKING

from loguru import logger
from qilisim_module import QiliSimCpp

from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator
from qilisdk.backends.backend import Backend
from qilisdk.core.qtensor import QTensor, tensor_prod

if TYPE_CHECKING:
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution import TimeEvolution
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


class QiliSim(Backend):
    """
    Backend based that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom C++ simulator.
    """

    def __init__(
        self,
        evolution_method: str = "integrate",
        arnoldi_dim: int = 10,
        num_arnoldi_substeps: int = 1,
        num_integrate_substeps: int = 1,
        monte_carlo: bool = False,
        num_monte_carlo_trajectories: int = 100,
        max_cache_size: int = 100,
        num_threads: int = 0,
    ) -> None:
        """
        Instantiate a new :class:`QiliSim` backend. This is a CPU-based simulator
        implemented in C++, using pybind11 for bindings.

        Args:
            evolution_method (str): The solver method to use. Options are 'direct', 'arnoldi' and 'integrate'.
            arnoldi_dim (int): The dimension of the Arnoldi subspace to use for the 'arnoldi' method.
            num_arnoldi_substeps (int): The number of substeps to use when using the Arnoldi method.
            num_integrate_substeps (int): The number of substeps to use when using the Integrate method.
            monte_carlo (bool): Whether to use the Monte Carlo method for open systems.
            num_monte_carlo_trajectories (int): The number of trajectories to use when using the Monte Carlo method.
            max_cache_size (int): The maximum size of the internal cache for gate caching.
            num_threads (int): The number of threads to use for parallel execution. If 0, uses all available cores.
        Raises:
            ValueError: If any of the parameters are invalid.

        """

        # Sanity checks on params
        # Note that these are also in the C++ code, so update there as well if changed here for consistency
        if evolution_method not in {"direct", "arnoldi", "integrate"}:
            raise ValueError(f"Unknown time evolution method: {evolution_method}")
        if arnoldi_dim <= 0:
            raise ValueError("arnoldi_dim must be a positive integer")
        if num_arnoldi_substeps <= 0:
            raise ValueError("num_arnoldi_substeps must be a positive integer")
        if num_integrate_substeps <= 0:
            raise ValueError("num_integrate_substeps must be a positive integer")
        if num_monte_carlo_trajectories <= 0:
            raise ValueError("num_monte_carlo_trajectories must be a positive integer")

        # Set number of threads if non-positive
        if num_threads <= 0:
            num_threads = os.cpu_count() or 1

        # Initialize the backend and the class vars
        super().__init__()
        self.qili_sim = QiliSimCpp()
        self.solver_params = {
            "evolution_method": evolution_method,
            "arnoldi_dim": arnoldi_dim,
            "num_arnoldi_substeps": num_arnoldi_substeps,
            "num_integrate_substeps": num_integrate_substeps,
            "monte_carlo": monte_carlo,
            "num_monte_carlo_trajectories": num_monte_carlo_trajectories,
            "max_cache_size": max_cache_size,
            "num_threads": num_threads,
        }

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling with {} shots", functional.nshots)
        result = self.qili_sim.execute_sampling(functional, self.solver_params)
        logger.success("Sampling finished")
        return result

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        """
        Computes the time evolution under of an initial state under the given schedule.

        Args:
            functional (TimeEvolution): The TimeEvolution functional to execute.

        Returns:
            TimeEvolutionResult: The results of the evolution.

        Raises:
            ValueError: If an observable type is unsupported.

        """

        # Get the time steps
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)
        steps = functional.schedule.tlist

        # Get the Hamiltonians and their parameters from the schedule per timestep
        Hs = [functional.schedule.hamiltonians[h] for h in functional.schedule.hamiltonians]
        coeffs = [[functional.schedule.coefficients[h][t] for t in steps] for h in functional.schedule.hamiltonians]

        # Jump operators TODO
        jump_operators: list[QTensor] = []

        # Get the observables
        observables = []
        identity = QTensor(PauliI(0).matrix)
        for obs in functional.observables:
            aux_obs = None
            if isinstance(obs, PauliOperator):
                for i in range(functional.schedule.nqubits):
                    if aux_obs is None:
                        aux_obs = identity if i != obs.qubit else QTensor(obs.matrix)
                    else:
                        aux_obs = (
                            tensor_prod([aux_obs, identity])
                            if i != obs.qubit
                            else tensor_prod([aux_obs, QTensor(obs.matrix)])
                        )
            elif isinstance(obs, Hamiltonian):
                aux_obs = QTensor(obs.to_matrix())
                if obs.nqubits < functional.schedule.nqubits:
                    for _ in range(functional.schedule.nqubits - obs.nqubits):
                        aux_obs = tensor_prod([aux_obs, identity])
            elif isinstance(obs, QTensor):
                aux_obs = obs
            else:
                logger.error("Unsupported observable type {}", obs.__class__.__name__)
                raise ValueError(f"unsupported observable type of {obs.__class__}")
            if aux_obs is not None:
                observables.append(aux_obs)

        # Execute the time evolution
        result = self.qili_sim.execute_time_evolution(
            functional.initial_state,
            Hs,
            coeffs,
            steps,
            observables,
            jump_operators,
            functional.store_intermediate_results,
            self.solver_params,
        )

        logger.success("TimeEvolution finished")
        return result
