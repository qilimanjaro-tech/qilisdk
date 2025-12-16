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
import ctypes
from loguru import logger
from qilisdk.backends.backend import Backend
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution

from qilisdk.backends.qili_sim_pybind import QiliSimPybindC

class QiliSimCpp(Backend):
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
        self.obj = self.c_lib.qilisim_create()
        logger.success("QiliSim initialised")

    def __del__(self) -> None:
        """
        Destructor for the QiliSim backend.
        """
        if getattr(self, "c_lib", None) is not None and getattr(self, "obj", None) is not None:
            self.c_lib.qilisim_free(self.obj)
        logger.info("QiliSim destroyed")

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling (shots={})", functional.nshots)
        
        # Serialize to the C compatible representation
        functional_c = self._sampling_to_c(functional)
        
        # Run the backend
        self.c_lib.qilisim_execute_sampling(self.obj, functional_c)

        # Get the return size, create a buffer and then fill it
        buffer_size = self.c_lib.qilisim_get_return_size(self.obj)
        res_c = ctypes.create_string_buffer(buffer_size)
        self.c_lib.qilisim_get_return_buffer(self.obj, res_c)

        # Deserialize the result
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

        # Serialize to the C compatible representation
        functional_c = self._time_evolution_to_c(functional)

        # Run the backend
        self.c_lib.qilisim_execute_time_evolution(functional_c)

        # Get the return size, create a buffer and then fill it
        buffer_size = self.c_lib.qilisim_get_return_size(self.obj)
        res_c = ctypes.create_string_buffer(buffer_size)
        self.c_lib.qilisim_get_return_buffer(self.obj, res_c)

        # Deserialize the result
        res = self._time_evolution_result_from_c(res_c)

        return res

    def _sampling_to_c(self, functional: Sampling) -> str:
        """
        Convert a Sampling functional to a string form for passing to the C++ backend.
        This results in a string like "1000 3 3 H 0 1 0 0 CNOT 1 1 0 0 1 CNOT 1 1 0 1 2"
        where the first three numbers are nshots, nqubits, ngates, followed by gate information:
        gate name, number of control qubits, number of target qubits, number of parameters, qubit indices, parameter values.

        Args:
            functional (Sampling): The Sampling functional to convert.

        Returns:
            str: The C-compatible representation of the Sampling functional.

        """

        # Circuit info
        functional_string = ""
        functional_string += str(functional.nshots)
        functional_string += " " + str(functional.circuit.nqubits)
        functional_string += " " + str(len(functional.circuit.gates))
        
        # Gate info
        for gate in functional.circuit.gates:
            functional_string += " " + gate.name
            functional_string += " " + str(len(gate.control_qubits))
            functional_string += " " + str(len(gate.target_qubits))
            functional_string += " " + str(gate.nparameters)
            functional_string += " " + " ".join(map(str, gate.qubits))
            if gate.nparameters > 0:
                functional_string += " " + " ".join(map(str, gate.get_parameter_values()))

        # Convert to c string
        functional_string_c = functional_string.encode('utf-8')
        return functional_string_c

    def _time_evolution_to_c(self, functional: TimeEvolution) -> str:
        """
        Convert a TimeEvolution functional to a string form for passing to the C++ backend.

        Args:
            functional (TimeEvolution): The TimeEvolution functional to convert.

        Returns:
            str: The C-compatible representation of the TimeEvolution functional.

        """
        return "converted to c string"

    def _sampling_result_from_c(self, res_c: str) -> SamplingResult:
        """
        Convert a C++ Sampling result to a SamplingResult object.
        The resulting string should have the form like "4 1000 00 256 01 512 10 128 11 128"
        where the first number is the number of unique measurement outcomes,
        the second number is the total shots,
        followed by pairs of measurement outcome (as bitstring) and counts.

        Args:
            res_c (str): The C++ representation of the Sampling result.

        Returns:
            SamplingResult: The converted SamplingResult object.

        """
        res_c = res_c.value.decode('utf-8')
        counts = {}
        parts = res_c.split()
        if len(parts) < 2:
            return SamplingResult(1, {"000": 1}) # TODO
            raise ValueError("Invalid Sampling result from C++ backend")
        n_unique = int(parts[0])
        nshots = int(parts[1])
        for i in range(n_unique):
            bitstring = parts[2 + i * 2]
            count = int(parts[3 + i * 2])
            counts[bitstring] = count
        return SamplingResult(nshots, counts)

    def _time_evolution_result_from_c(self, res_c: str) -> TimeEvolutionResult:
        """
        Convert a C++ TimeEvolution result to a TimeEvolutionResult object.

        Args:
            res_c (str): The C++ representation of the TimeEvolution result.
        Returns:
            TimeEvolutionResult: The converted TimeEvolutionResult object.
        """
        return TimeEvolutionResult()

class QiliSimPybind(Backend):
    """
    Backend based that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom c++ simulator.
    """

    def __init__(self) -> None:
        """
        Instantiate a new :class:`QiliSim` backend.
        """
        super().__init__()
        self.qili_sim = QiliSimPybindC()

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        return self.qili_sim.execute_sampling(functional)
