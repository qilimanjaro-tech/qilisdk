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
from pprint import pformat
from typing import Callable

from qilisdk.common.optimizer import Optimizer
from qilisdk.common.result import Result
from qilisdk.digital.ansatz import Ansatz
from qilisdk.digital.digital_algorithm import DigitalAlgorithm
from qilisdk.digital.digital_backend import DigitalBackend
from qilisdk.digital.digital_result import DigitalResult
from qilisdk.yaml import yaml


@yaml.register_class
class VQEResult(Result):
    """
    Represents the result of a VQE calculation.

    Attributes:
        optimal_cost (float): The estimated ground state energy (optimal cost).
        optimal_parameters (list[float]): The optimal parameters found during the optimization.
    """

    def __init__(self, optimal_cost: float, optimal_parameters: list[float]) -> None:
        super().__init__()
        self._optimal_cost = optimal_cost
        self._optimal_parameters = optimal_parameters

    @property
    def optimal_parameters(self) -> list[float]:
        """
        Get the optimal ansatz parameters.

        Returns:
            list[float]: The optimal parameters.
        """
        return list(self._optimal_parameters)

    @property
    def optimal_cost(self) -> float:
        """
        Get the optimal cost (estimated ground state energy).

        Returns:
            float: The optimal cost.
        """
        return self._optimal_cost

    def __repr__(self) -> str:
        """
        Return a string representation of the VQEResult for debugging.

        Returns:
            str: A formatted string detailing the optimal cost and parameters.
        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n  Optimal Cost = {self._optimal_cost},"
            + f"\n  Optimal Parameters={pformat(self._optimal_parameters)}\n )"
        )


class VQE(DigitalAlgorithm):
    """
    Implements the Variational Quantum Eigensolver (VQE) algorithm.

    The VQE algorithm is a hybrid quantum-classical method used to approximate the ground
    state energy of a quantum system. It relies on a parameterized quantum circuit (ansatz)
    whose parameters are tuned by a classical optimizer to minimize the cost function—typically
    the expectation value of the system's Hamiltonian.

    The algorithm outputs a VQEResult that includes the optimal cost (estimated ground state energy)
    and the optimal parameters that yield this cost.
    """

    def __init__(
        self, ansatz: Ansatz, initial_params: list[float], cost_function: Callable[[DigitalResult], float]
    ) -> None:
        """
        Initialize the VQE algorithm.

        Args:
            ansatz (Ansatz): The parameterized quantum circuit representing the trial state.
            initial_params (list[float]): The initial set of parameters for the ansatz.
            cost_function (Callable[[DigitalResult], float]): A function that computes the cost from
                a DigitalResult obtained after executing the circuit. The cost generally represents the
                expectation value of the Hamiltonian.
        """
        self._ansatz = ansatz
        self._initial_params = initial_params
        self._cost_function = cost_function
        self._execution_results: list[DigitalResult]

    def obtain_cost(self, params: list[float], backend: DigitalBackend, nshots: int = 1000) -> float:
        """
        Evaluate the cost at a given parameter set by executing the corresponding quantum circuit.

        The process involves:
            1. Generating the quantum circuit using the ansatz with the specified parameters.
            2. Executing the circuit on the provided digital backend with the given number of shots.
            3. Passing the resulting DigitalResult to the cost_function to obtain the cost value.

        Args:
            params (list[float]): The ansatz parameters to evaluate.
            backend (DigitalBackend): The digital backend that executes the quantum circuit.
            nshots (int, optional): The number of shots (circuit executions). Defaults to 1000.

        Returns:
            float: The cost computed from the DigitalResult.
        """
        circuit = self._ansatz.get_circuit(params)
        results = backend.execute(circuit=circuit, nshots=nshots)
        return self._cost_function(results)

    def execute(self, backend: DigitalBackend, optimizer: Optimizer, nshots: int = 1000) -> DigitalResult:
        """
        Run the VQE algorithm to obtain the optimal parameters and the corresponding cost.

        This method leverages a classical optimizer to minimize the cost function by varying the
        ansatz parameters. The optimizer returns a tuple containing the optimal cost and the optimal
        parameters. A VQEResult object is then created using these values.

        Args:
            backend (DigitalBackend): The backend for executing quantum circuits.
            optimizer (Optimizer): The classical optimizer for tuning the ansatz parameters.
            nshots (int, optional): The number of shots for each circuit execution. Defaults to 1000.

        Returns:
            VQEResult: An object containing the optimal cost and the optimal ansatz parameters.
        """
        optimal_cost, optimal_parameters = optimizer.optimize(
            lambda x: self.obtain_cost(x, backend=backend, nshots=nshots), self._initial_params
        )
        return VQEResult(
            optimal_cost=optimal_cost,
            optimal_parameters=optimal_parameters,
        )
