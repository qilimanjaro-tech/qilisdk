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

from qilisdk.common.algorithm import Algorithm
from qilisdk.common.backend import Backend
from qilisdk.common.model import Model
from qilisdk.common.optimizer import Optimizer
from qilisdk.common.optimizer_result import OptimizerIntermediateResult, OptimizerResult
from qilisdk.common.result import Result
from qilisdk.digital.ansatz import Ansatz
from qilisdk.digital.sampling import Sampling
from qilisdk.digital.sampling_result import SamplingResult
from qilisdk.yaml import yaml


@yaml.register_class
class VQEResult(Result):
    """
    Represents the result of a VQE calculation.

    Attributes:
        optimal_cost (float): The estimated ground state energy (optimal cost).
        optimal_parameters (list[float]): The optimal parameters found during the optimization.
        probabilities (list[tuple[str, float]]): the list of samples and their probabilities.
    """

    def __init__(self, optimizer_result: OptimizerResult, digital_result: SamplingResult) -> None:
        super().__init__()
        self._optimizer_result: OptimizerResult = optimizer_result
        self._digital_result: SamplingResult = digital_result

    @property
    def optimal_cost(self) -> float:
        """
        Get the optimal cost (estimated ground state energy).

        Returns:
            float: The optimal cost.
        """
        return self._optimizer_result.optimal_cost

    @property
    def optimal_samples(self) -> dict[str, int]:
        """
        Gets the raw measurement samples.

        Returns:
            dict[str, int]: A dictionary where keys are bitstrings representing measurement outcomes
            and values are the number of times each outcome was observed.
        """
        return self._digital_result.samples

    @property
    def optimal_probabilities(self) -> dict[str, float]:
        """
        Gets the probabilities for each measurement outcome of executing the circuit with the optimal parameters.

        Returns:
            dict[str, float]: A dictionary mapping each bitstring outcome to its corresponding probability.
        """
        return dict(self._digital_result.probabilities)

    def get_optimal_probability(self, bitstring: str) -> float:
        """
        Computes the probability of a specific measurement outcome.

        Args:
            bitstring (str): The bitstring representing the measurement outcome of interest.

        Returns:
            float: The probability of the specified bitstring occurring.
        """
        return self._digital_result.get_probability(bitstring)

    def get_optimal_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """
        Returns the n most probable bitstrings along with their probabilities.

        Parameters:
            n (int): The number of most probable bitstrings to return.

        Returns:
            list[tuple[str, float]]: A list of tuples (bitstring, probability) sorted in descending order by probability.
        """
        return self._digital_result.get_probabilities(n)

    @property
    def optimal_parameters(self) -> list[float]:
        """
        Get the optimal ansatz parameters.

        Returns:
            list[float]: The optimal parameters.
        """
        return self._optimizer_result.optimal_parameters

    @property
    def intermediate_results(self) -> list[OptimizerIntermediateResult]:
        """
        Get the intermediate results.

        Returns:
            list[OptimizerResult]: The intermediate results.
        """
        return self._optimizer_result.intermediate_results

    def __repr__(self) -> str:
        """
        Return a string representation of the VQEResult for debugging.

        Returns:
            str: A formatted string detailing the optimal cost and parameters.
        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n  Optimal Cost = {self.optimal_cost},"
            + f"\n  Optimal Parameters={pformat(self.optimal_parameters)},"
            + f"\n  Intermediate Results={pformat(self.intermediate_results)})"
            + f"\n  Optimal Probabilities={pformat(self.optimal_probabilities)})"
            + f"\n  Optimal Samples={pformat(self.optimal_samples)})"
        )


@yaml.register_class
class VQE(Algorithm):
    """
    Implements the Variational Quantum Eigensolver (VQE) algorithm.

    The VQE algorithm is a hybrid quantum-classical method used to approximate the ground
    state energy of a quantum system. It relies on a parameterized quantum circuit (ansatz)
    whose parameters are tuned by a classical optimizer to minimize the cost function—typically
    the expectation value of the system's Hamiltonian.

    The algorithm outputs a VQEResult that includes the optimal cost (estimated ground state energy)
    and the optimal parameters that yield this cost.
    """

    def __init__(self, ansatz: Ansatz, initial_params: list[float], model: Model) -> None:
        """
        Initialize the VQE algorithm.

        Args:
            ansatz (Ansatz): The parameterized quantum circuit representing the trial state.
            initial_params (list[float]): The initial set of parameters for the ansatz.
            model (Model): The abstract mathematical model representing the cost function. To construct a cost function
                using this model define an objective and a set of constraints that will be evaluated
                during the optimization of the circuit parameters.
        """
        self._ansatz = ansatz
        self._initial_params = initial_params
        self._model = model

    def obtain_cost(self, params: list[float], backend: Backend, nshots: int = 1000) -> float:
        """
        Evaluate the cost at a given parameter set by executing the corresponding quantum circuit.

        The process involves:
            1. Generating the quantum circuit using the ansatz with the specified parameters.
            2. Executing the circuit on the provided digital backend with the given number of shots.
            3. Passing the resulting DigitalResult to the model to obtain the cost value.

        Args:
            params (list[float]): The ansatz parameters to evaluate.
            backend (DigitalBackend): The digital backend that executes the quantum circuit.
            nshots (int, optional): The number of shots (circuit executions). Defaults to 1000.

        Returns:
            float: The cost computed using the model.
        """
        circuit = self._ansatz.get_circuit(params)
        results = backend.execute(Sampling(circuit=circuit, nshots=nshots))
        cost = 0.0
        var_list = self._model.variables()
        for state, prob in results.get_probabilities():
            mapped_state = {v: float(state[i]) for i, v in enumerate(var_list)}
            eval_res = self._model.evaluate(mapped_state)
            aux_cost = eval_res[self._model.objective.label]
            for c in self._model.constraints:
                aux_cost += eval_res[c.label]
            cost += aux_cost * prob
        return cost

    def run(
        self,
        backend: Backend,
        optimizer: Optimizer,
        nshots: int = 1000,
        store_intermediate_results: bool = False,
    ) -> VQEResult:
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
        optimizer_result = optimizer.optimize(
            lambda x: self.obtain_cost(x, backend=backend, nshots=nshots),
            self._initial_params,
            store_intermediate_results=store_intermediate_results,
        )
        circuit = self._ansatz.get_circuit(optimizer_result.optimal_parameters)
        digital_result = backend.execute(Sampling(circuit=circuit, nshots=nshots))
        return VQEResult(optimizer_result=optimizer_result, digital_result=digital_result)
