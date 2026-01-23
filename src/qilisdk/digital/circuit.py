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

import random
from typing import TYPE_CHECKING, Iterable

import numpy as np
from typing_extensions import Self

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import Domain, Parameter
from qilisdk.utils.visualization import CircuitStyle
from qilisdk.yaml import yaml

from .exceptions import ParametersNotEqualError, QubitOutOfRangeError
from .gates import BasicGate, Gate

if TYPE_CHECKING:
    from qilisdk.core.types import RealNumber


@yaml.register_class
class Circuit(Parameterizable):
    def __init__(self, nqubits: int) -> None:
        """
        Initialize a Circuit instance with a specified number of qubits.

        Args:
            nqubits (int): The number of qubits in the circuit.
        """
        super(Circuit, self).__init__()
        self._nqubits: int = nqubits
        self._gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)
        self._parameters: dict[str, Parameter] = {}

    @property
    def nqubits(self) -> int:
        """
        Retrieve the number of qubits in the circuit.

        Returns:
            int: The total number of qubits.
        """
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total number of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return len(self._parameters)

    @property
    def gates(self) -> list[Gate]:
        """
        Retrieve the list of gates in the circuit.

        Returns:
            list[Gate]: A list of gates that have been added to the circuit.
        """
        return self._gates

    def get_parameter_values(self) -> list[float]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return [param.value for param in self._parameters.values()]

    def get_parameter_names(self) -> list[str]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return list(self._parameters.keys())

    def get_parameters(self) -> dict[str, float]:
        """
        Retrieve the parameter names and values from all parameterized gates in the circuit.

        Returns:
            dict[str, float]: A dictionary of the parameters with their current values.
        """
        return {label: param.value for label, param in self._parameters.items()}

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set new parameter values for all parameterized gates in the circuit.

        Args:
            values (list[float]): A list containing new parameter values to assign to the parameterized gates.

        Raises:
            ParametersNotEqualError: If the number of provided values does not match the expected number of parameters.
        """
        if len(values) != self.nparameters:
            raise ParametersNotEqualError
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameters (dict[str, float]): A dictionary with the labels of the parameters to be modified and their new value.

        Raises:
            ValueError: if the label provided doesn't correspond to a parameter defined in this circuit.
        """
        for label, param in parameters.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this circuit.")
            self._parameters[label].set_value(param)

    def get_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {k: v.bounds for k, v in self._parameters.items()}

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        for label, bound in ranges.items():
            if label not in self._parameters:
                raise ValueError(
                    f"The provided parameter label {label} is not defined in the list of parameters in this object."
                )
            self._parameters[label].set_bounds(bound[0], bound[1])

    def _parse_params(self, gate: Gate) -> None:
        """Parse The parameters in the gate

        Args:
            gate (Gate): The gate to be parsed.
        """
        if gate.is_parameterized:
            param_base_label = f"{gate.name}({','.join(map(str, gate.qubits))})"
            for label, parameter in gate.parameters.items():
                if label == parameter.label and label in gate.PARAMETER_NAMES:
                    parameter_label = param_base_label + f"_{label}_{len(self._parameters)}"
                else:
                    parameter_label = parameter.label
                self._parameters[parameter_label] = gate.parameters[label]

    def _add(self, gate: Gate) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate or a list of quantum gates to be added to the circuit.

        Raises:
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError

        self._parse_params(gate)
        self._gates.append(gate)

    def add(self, gates: Gate | Iterable[Gate]) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gates (Gate | list[Gate]): The quantum gate or a list of quantum gates to be added to the circuit.
        """
        if isinstance(gates, Gate):
            self._add(gates)
            return
        for g in gates:
            self._add(g)

    def _insert(self, gate: Gate, index: int = -1) -> None:
        """Insert a quantum gate to the circuit at a given index.

        Args:
            gate (Gate): The gate to be inserted.
            index (int, optional): The index at which the gate is inserted. Defaults to -1.

        Raises:
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError

        self._parse_params(gate)
        self._gates.insert(index, gate)

    def insert(self, gates: Gate | Iterable[Gate], index: int = -1) -> None:
        """Insert a quantum gate to the circuit at a given index.

        Args:
            gates (Gate | list[Gate]): The gate or list of gates to be inserted.
            index (int, optional): The index at which the gate is inserted. Defaults to -1.
        """
        if isinstance(gates, Gate):
            self._insert(gates, index)
            return
        for i, gate in enumerate(gates):
            self._insert(gate, i + index)

    def append(self, circuit: Circuit) -> None:
        """Append circuit elements at the end of the current circuit.

        Args:
            circuit (Circuit): The circuit to be appended.

        Raises:
            QubitOutOfRangeError: If the appended circuit acts on more qubits than the current circuit.
        """
        if circuit.nqubits != self.nqubits:
            raise QubitOutOfRangeError(
                "the appended circuit contains different number of qubits than the current circuit."
            )

        for g in circuit.gates:
            self.add(g)

    def prepend(self, circuit: Circuit) -> None:
        """Prepend circuit elements to the beginning of the current circuit.

        Args:
            circuit (Circuit): The circuit to be prepended.

        Raises:
            QubitOutOfRangeError: If the circuit to be prepended acts on more qubits than the current circuit.
        """
        if circuit.nqubits != self.nqubits:
            raise QubitOutOfRangeError(
                "the prepended circuit contains different number of qubits than the current circuit."
            )

        for i, g in enumerate(circuit.gates):
            self.insert(g, i)

    def __add__(self, other: Circuit | Gate) -> Circuit | NotImplementedError:
        if not isinstance(other, (Circuit, Gate)):
            return NotImplementedError(
                "Addition is only supported between Circuit objects or a Circuit and a Gate objects"
            )
        if isinstance(other, Gate):
            self.add(other)
        else:
            self.append(other)
        return self

    __iadd__ = __add__

    def __radd__(self, other: Circuit | Gate) -> Circuit | NotImplementedError:
        if not isinstance(other, (Circuit, Gate)):
            return NotImplementedError(
                "Addition is only supported between Circuit objects or a Circuit and a Gate objects"
            )
        if isinstance(other, Gate):
            self.insert(other, 0)
        else:
            self.prepend(other)
        return self

    def draw(self, style: CircuitStyle = CircuitStyle(), filepath: str | None = None) -> None:
        """
        Render this circuit with Matplotlib and optionally save it to a file.

        The circuit is rendered using the provided style configuration. If ``filepath`` is
        given, the resulting figure is saved to disk (the output format is inferred
        from the file extension, e.g. ``.png``, ``.pdf``, ``.svg``).

        Args:
            style (CircuitStyle): Visual style configuration applied to the plot.
                If not provided, the default :class:`CircuitStyle` is used.
            filepath (str | None): Destination file path for the rendered figure.
                If ``None``, the figure is not saved.
        """
        from qilisdk.utils.visualization.circuit_renderers import MatplotlibCircuitRenderer  # noqa: PLC0415

        renderer = MatplotlibCircuitRenderer(self, style=style)
        renderer.plot()
        if filepath:
            renderer.save(filepath)

    @classmethod
    def random(
        cls, nqubits: int, single_qubit_gates: set[type[BasicGate]], two_qubit_gates: set[type[BasicGate]], ngates: int
    ) -> Self:
        """
        Generate a random quantum circuit from a given set of gates.

        Args:
            nqubits (int): The number of qubits in the circuit.
            single_qubit_gates (set[Gate]): A set of single-qubit gate classes to choose from.
            two_qubit_gates (set[Gate]): A set of two-qubit gate classes to choose from.
            ngates (int): The number of gates to include in the circuit.

        Returns:
            Circuit: A randomly generated quantum circuit.

        Raises:
            ValueError: If it is not possible to generate a full random circuit with the provided parameters

        """

        # If we only have one gate and one qubit, throw an error
        if nqubits == 1 and len(single_qubit_gates) == 1:
            raise ValueError("Cannot generate a full random circuit with only one qubit and one gate.")

        # Make sure we have given gates
        if len(single_qubit_gates) == 0 and len(two_qubit_gates) == 0:
            raise ValueError("At least one gate must be provided to generate a random circuit.")

        new_circuit = cls(nqubits)
        gate_list: list[type[BasicGate]] = list(single_qubit_gates)
        if nqubits > 1:
            gate_list.extend(list(two_qubit_gates))
        prev_gate_type = None
        prev_qubits = None
        for _ in range(ngates):
            gate_class = random.choice(gate_list)
            gate_nqubits = 1 if gate_class in single_qubit_gates else 2
            qubits = tuple(random.sample(range(nqubits), gate_nqubits))

            # Avoid adding the same gate on the same qubits consecutively
            if gate_class == prev_gate_type and qubits == prev_qubits:
                # If we only have one qubit, pick a different gate
                if nqubits == 1:
                    possible_new_gates = [g for g in single_qubit_gates if g != gate_class]
                    gate_class = random.choice(possible_new_gates)

                # If the gate list does not include all qubits, change the first to be a different qubit
                if len(qubits) < nqubits:
                    possible_new_qubits = [q for q in range(nqubits) if q not in qubits]
                    new_qubit = random.choice(possible_new_qubits)
                    qubits = (new_qubit, *qubits[1:])

                # Otherwise, flip the order of the qubits
                else:
                    qubits = tuple(reversed(qubits))

            # Update previous gate info
            prev_gate_type = gate_class
            prev_qubits = qubits

            # Generate random parameters if needed
            params = {}
            if gate_class.PARAMETER_NAMES:
                for param_name in gate_class.PARAMETER_NAMES:
                    val = random.uniform(-np.pi, np.pi)
                    params[param_name] = Parameter(
                        label=param_name + str(val), value=val, domain=Domain.REAL, bounds=(val, val)
                    )

            # Add the gate to the circuit
            new_circuit.add(gate_class(*qubits, **params))

        return new_circuit
