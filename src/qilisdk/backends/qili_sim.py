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
        result = self.qili_sim.execute_time_evolution(functional)
        logger.success("TimeEvolution finished")
        return result

