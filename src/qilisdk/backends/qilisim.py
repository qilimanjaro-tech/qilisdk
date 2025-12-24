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

    def __init__(self, solver_params: dict = {}) -> None:
        """
        Instantiate a new :class:`QiliSim` backend. This is a CPU-based simulator
        implemented in C++, using pybind11 for bindings.

        Args:
            solver_params: Optional keyword arguments to configure the time-evolution solver. Supported parameters are:

                - `evolution_method` (str): The solver method to use. Options are direct', 'arnoldi' and 'integrate' (default).
                - `arnoldi_dim` (int): The dimension of the Arnoldi subspace to use for the 'arnoldi' method (default: 10).
                - `num_arnoldi_substeps` (int): The number of substeps to use when using the Arnoldi method (default: 1).
                - `num_integrate_substeps` (int): The number of substeps to use when using the Integrate method (default: 1).
                - `monte_carlo` (bool): Whether to use Monte Carlo wave function method for open systems (default: False).

        """
        super().__init__()
        self.qili_sim = QiliSimCpp()
        self.solver_params = solver_params

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling with {} shots", functional.nshots)
        result = self.qili_sim.execute_sampling(functional)
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
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)

        # Get the time steps
        steps = np.linspace(0, functional.schedule.T, int(functional.schedule.T // functional.schedule.dt))
        tlist = np.array(functional.schedule.tlist)
        steps = np.union1d(steps, tlist)

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
        result = self.qili_sim.execute_time_evolution(functional.initial_state,
                                                      Hs,
                                                      coeffs,
                                                      steps,
                                                      observables,
                                                      jump_operators,
                                                      functional.store_intermediate_results,
                                                      self.solver_params)

        logger.success("TimeEvolution finished")
        return result
