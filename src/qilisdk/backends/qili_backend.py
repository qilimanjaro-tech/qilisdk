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
from ctypes import cdll
from loguru import logger
from qilisdk.backends.backend import Backend
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution

class QiliSim(Backend):
    """
    Backend based that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom c++ simulator.
    """

    def __init__(self) -> None:
        """
        Instantiate a new :class:`QiliSim` backend.
        """

        super().__init__()
        self.c_lib = cdll.LoadLibrary('./libqilisim.so')
        logger.success("QiliSim initialised")

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling (shots={})", functional.nshots)
        functional_c = self._sampling_to_c(functional)
        res_c = self.c_lib._execute_sampling(functional_c)
        print("Recieved from c backend:", res_c)
        res = self._sampling_result_from_c(res_c)
        return res

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        """
        Computes the time evolution under of an initial state under the given schedule.

        Args:
            functional (TimeEvolution): The TimeEvolution functional to execute.

        Returns:
            TimeEvolutionResult: The results of the evolution.

        """
        logger.info("Executing TimeEvolution (time={}, nsteps={})", functional.time, functional.nsteps)
        functional_c = self._time_evolution_to_c(functional)
        res_c = self.c_lib._execute_time_evolution(functional_c)
        res = self._time_evolution_result_from_c(res_c)
        return res

    def _sampling_to_c(self, functional: Sampling) -> str:
        """
        Convert a Sampling functional to a string form for passing to the C++ backend.

        Args:
            functional (Sampling): The Sampling functional to convert.

        Returns:
            str: The C-compatible representation of the Sampling functional.

        """
        # convert to c representation
        functional_string = "converted to c string"
        functional_string_c = functional_string.encode('utf-8')
        return functional_string_c

    def _sampling_result_from_c(self, res_c: str) -> SamplingResult:
        """
        Convert a C++ Sampling result to a SamplingResult object.

        Args:
            res_c (str): The C++ representation of the Sampling result.

        Returns:
            SamplingResult: The converted SamplingResult object.

        """
        return SamplingResult(0, {"000": 0})

    def _time_evolution_to_c(self, functional: TimeEvolution) -> str:
        """
        Convert a TimeEvolution functional to a string form for passing to the C++ backend.

        Args:
            functional (TimeEvolution): The TimeEvolution functional to convert.

        Returns:
            str: The C-compatible representation of the TimeEvolution functional.

        """
        return "converted to c string"

    def _time_evolution_result_from_c(self, res_c: str) -> TimeEvolutionResult:
        """
        Convert a C++ TimeEvolution result to a TimeEvolutionResult object.

        Args:
            res_c (str): The C++ representation of the TimeEvolution result.
        Returns:
            TimeEvolutionResult: The converted TimeEvolutionResult object.
        """
        return TimeEvolutionResult()
