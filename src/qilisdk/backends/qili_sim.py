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
from loguru import logger
from qilisdk.backends.backend import Backend
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.core.qtensor import QTensor, tensor_prod
from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliOperator
import numpy as np

# Import the C++ pybind11 wrapper
from qilisdk.backends.qili_sim_c import QiliSimC

class QiliSim(Backend):
    """
    Backend based that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom C++ simulator.
    """

    def __init__(self) -> None:
        """
        Instantiate a new :class:`QiliSim` backend.
        """
        super().__init__()
        self.qili_sim = QiliSimC()

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

        """
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)
        
        # Get the time steps
        steps = np.linspace(0, functional.schedule.T, int(functional.schedule.T // functional.schedule.dt))
        tlist = np.array(functional.schedule.tlist)
        steps = np.union1d(steps, tlist)
        steps = list(np.sort(steps))

        # Get the Hamiltonians and their parameters from the schedule per timestep
        Hs = [functional.schedule.hamiltonians[h] for h in functional.schedule.hamiltonians]
        coeffs = [[functional.schedule.coefficients[h][t] for t in steps] for h in functional.schedule.hamiltonians]

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

        # Set the Arnoldi dimension for the Krylov subspace method
        arnoldi_dim = 20
        
        # Execute the time evolution
        result = self.qili_sim.execute_time_evolution(functional.initial_state, Hs, coeffs, steps, observables, arnoldi_dim)

        logger.success("TimeEvolution finished")
        return result

