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
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Union

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import CNOT, CZ, U1, U2, U3, M
from qilisdk.yaml import yaml


class Ansatz(ABC):
    _ONE_QUBIT_GATES: ClassVar[dict[str, type[Union[U1, U2, U3]]]] = {
        "U1": U1,
        "U2": U2,
        "U3": U3,
    }
    _TWO_QUBITS_GATES: ClassVar[dict[str, type[Union[CNOT, CZ]]]] = {
        "CZ": CZ,
        "CNOT": CNOT,
    }

    def __init__(self, nqubits: int, layers: int = 1) -> None:
        self.nqubits = nqubits
        self.layers = layers

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total number of parameters required by all parameterized gates in the ansatz's circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        raise NotImplementedError

    def get_circuit(self, parameters: list[float]) -> Circuit:
        """Get the underlying circuit with the given list of parameters.

        Args:
            params (list[float]): the list of parameters for the unitary gates.

        Raises:
            ValueError: if the number of parameters provided are less than the parameters expected by the ansatz.

        Returns:
            Circuit: The underlying circuit with the updated parameters.
        """
        if len(parameters) != self.nparameters:
            raise ValueError(f"Expecting {self.nparameters} but received {len(parameters)}")

        return self._construct_circuit(parameters=list(parameters))

    @abstractmethod
    def _construct_circuit(self, parameters: list[float]) -> Circuit: ...


@yaml.register_class
class HardwareEfficientAnsatz(Ansatz):
    def __init__(
        self,
        n_qubits: int,
        layers: int = 1,
        connectivity: Literal["Circular", "Linear", "Full"] | list[tuple[int, int]] = "Linear",
        structure: Literal["grouped", "interposed"] = "grouped",
        one_qubit_gate: Literal["U1", "U2", "U3"] = "U1",
        two_qubit_gate: Literal["CZ", "CNOT"] = "CZ",
    ) -> None:
        super().__init__(n_qubits, layers)

        # Define chip topology
        if isinstance(connectivity, list):
            self.connectivity = connectivity
        elif connectivity == "Full":
            self.connectivity = [(i, j) for i in range(self.nqubits) for j in range(i + 1, self.nqubits)]
        elif connectivity == "Circular":
            self.connectivity = [(i, i + 1) for i in range(self.nqubits - 1)] + [(self.nqubits - 1, 0)]
        elif connectivity == "Linear":
            self.connectivity = [(i, i + 1) for i in range(self.nqubits - 1)]
        else:
            raise ValueError(f"Unrecognized connectivity type ({connectivity}).")

        self.gate_types: dict[str, str] = {"one_qubit_gate": one_qubit_gate, "two_qubit_gates": two_qubit_gate}
        self.one_qubit_gate: type[Union[U1, U2, U3]] = self._ONE_QUBIT_GATES[one_qubit_gate]
        self.two_qubit_gate: type[Union[CNOT, CZ]] = self._TWO_QUBITS_GATES[two_qubit_gate]

        if structure not in {"grouped", "interposed"}:
            raise ValueError(f"provided structure {structure} is not supported.")
        self.structure = structure

        self.construct_layer_handlers = {
            "interposed": self._construct_layer_interposed,
            "grouped": self._construct_layer_grouped,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Exclude the mapping that contains bound methods (not serializable).
        state.pop("construct_layer_handlers", None)
        return state

    def __setstate__(self, state) -> None:  # noqa: ANN001
        """
        Restore the object's state after deserialization and reinitialize any attributes that were omitted.
        """
        self.__dict__.update(state)
        # Reconstruct the mapping with the proper bound methods.
        self.construct_layer_handlers = {
            "interposed": self._construct_layer_interposed,
            "grouped": self._construct_layer_grouped,
        }

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total number of parameters required by all parameterized gates in the ansatz's circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return self.nqubits * (self.layers + 1) * len(self.one_qubit_gate.PARAMETER_NAMES)

    def _construct_circuit(self, parameters: list[float]) -> Circuit:
        self._circuit = Circuit(self.nqubits)
        # Add initial layer of unitaries
        for i in range(self.nqubits):
            self._circuit.add(self.one_qubit_gate(i, **dict(zip(self.one_qubit_gate.PARAMETER_NAMES, parameters))))

        construct_layer_handler = self.construct_layer_handlers[self.structure]
        for _ in range(self.layers):
            construct_layer_handler(parameters)
        self._circuit.add(M(*list(range(self.nqubits))))

        return self._circuit

    def _construct_layer_interposed(self, parameters: list[float]) -> None:
        for i in range(self.nqubits):
            self._circuit.add(
                self.one_qubit_gate(i, **{name: parameters.pop() for name in self.one_qubit_gate.PARAMETER_NAMES})
            )
            for p, j in self.connectivity:
                self._circuit.add(self.two_qubit_gate(p, j))

    def _construct_layer_grouped(self, parameters: list[float]) -> None:
        for i in range(self.nqubits):
            self._circuit.add(
                self.one_qubit_gate(i, **{name: parameters.pop() for name in self.one_qubit_gate.PARAMETER_NAMES})
            )
        for i, j in self.connectivity:
            self._circuit.add(self.two_qubit_gate(i, j))
