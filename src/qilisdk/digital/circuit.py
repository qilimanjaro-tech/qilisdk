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

import random

import numpy as np

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.core.variables import Domain, Parameter, RealNumber
from qilisdk.utils.visualization import CircuitStyle
from qilisdk.yaml import yaml

from .exceptions import ParametersNotEqualError, QubitOutOfRangeError
from .gates import CNOT, RX, RY, RZ, U1, U2, U3, BasicGate, Gate, H, S, T, X, Y, Z


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

    def set_parameters(self, parameter_dict: dict[str, RealNumber]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameter_dict (dict[str, RealNumber]): A dictionary with the labels of the parameters to be modified and their new value.

        Raises:
            ValueError: if the label provided doesn't correspond to a parameter defined in this circuit.
        """
        for label, param in parameter_dict.items():
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

    def add(self, gate: Gate) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate to add to the circuit.

        Raises:
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError
        if gate.is_parameterized:
            param_base_label = f"{gate.name}({','.join(map(str, gate.qubits))})"
            for label, parameter in gate.parameters.items():
                if label == parameter.label:
                    parameter_label = param_base_label + f"_{label}_{len(self._parameters)}"
                else:
                    parameter_label = parameter.label
                self._parameters[parameter_label] = gate.parameters[label]
        self._gates.append(gate)

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

    def randomize(self, single_qubit_gates: set[type[BasicGate]], two_qubit_gates: set[type[BasicGate]], ngates: int) -> None:
        """
        Generate a random quantum circuit from a given set of gates.

        Args:
            single_qubit_gates (set[Gate]): A set of single-qubit gate classes to choose from.
            two_qubit_gates (set[Gate]): A set of two-qubit gate classes to choose from.
            ngates (int): The number of gates to include in the circuit.

        """
        gate_list: list[type[BasicGate]] = list(single_qubit_gates) + list(two_qubit_gates)
        for _ in range(ngates):
            gate_class = random.choice(gate_list)
            gate_nqubits = 1 if gate_class in single_qubit_gates else 2
            qubits = tuple(random.sample(range(self.nqubits), gate_nqubits))
            params = {}
            if gate_class.PARAMETER_NAMES:
                for param_name in gate_class.PARAMETER_NAMES:
                    val = random.uniform(-np.pi, np.pi)
                    params[param_name] = Parameter(label=param_name + str(val), value=val, domain=Domain.REAL, bounds=(val, val))

            # If you're reading this and thinking "why don't we just do self.add(gate_class(*qubits, **params))"?
            # It's because mypy gets angry about it, since then we don't know the type of the variable at runtime.
            if gate_class == CNOT:
                self.add(CNOT(control=qubits[0], target=qubits[1]))
            elif gate_class == X:
                self.add(X(qubits[0]))
            elif gate_class == Y:
                self.add(Y(qubits[0]))
            elif gate_class == Z:
                self.add(Z(qubits[0]))
            elif gate_class == H:
                self.add(H(qubits[0]))
            elif gate_class == T:
                self.add(T(qubits[0]))
            elif gate_class == S:
                self.add(S(qubits[0]))
            elif gate_class == RX:
                self.add(RX(qubits[0], **params))
            elif gate_class == RY:
                self.add(RY(qubits[0], **params))
            elif gate_class == RZ:
                self.add(RZ(qubits[0], **params))
            elif gate_class == U1:
                self.add(U1(qubits[0], **params))
            elif gate_class == U2:
                self.add(U2(qubits[0], **params))
            elif gate_class == U3:
                self.add(U3(qubits[0], **params))
            else:
                continue
