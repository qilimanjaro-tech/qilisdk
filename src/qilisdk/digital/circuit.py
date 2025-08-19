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
from typing import Any

import numpy as np

from qilisdk.common.variables import Parameter, RealNumber
from qilisdk.utils.visualization import CircuitStyle
from qilisdk.yaml import yaml

from .exceptions import ParametersNotEqualError, QubitOutOfRangeError
from .gates import Gate


@yaml.register_class
class Circuit:
    def __init__(self, nqubits: int) -> None:
        """
        Initialize a Circuit instance with a specified RealNumber of qubits.

        Args:
            nqubits (int): The RealNumber of qubits in the circuit.
        """
        self._nqubits: int = nqubits
        self._gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)
        self._parameters: dict[str, Parameter] = {}

    @property
    def nqubits(self) -> int:
        """
        Retrieve the RealNumber of qubits in the circuit.

        Returns:
            int: The total RealNumber of qubits.
        """
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total RealNumber of parameters required by all parameterized gates in the circuit.

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
            ParametersNotEqualError: If the RealNumber of provided values does not match the expected RealNumber of parameters.
        """
        if len(values) != self.nparameters:
            raise ParametersNotEqualError
        for i, parameter in enumerate(self._parameters.values()):
            parameter.set_value(values[i])

    def set_parameters(self, parameter_dict: dict[str, RealNumber]) -> None:
        """Set the parameter values by their label. No need to provide the full list of parameters.

        Args:
            parameter_dict (dict[str, RealNumber]): _description_

        Raises:
            ValueError: _description_
        """
        for label, param in parameter_dict.items():
            if label not in self._parameters:
                raise ValueError(f"Parameter {label} is not defined in this circuit.")
            self._parameters[label].set_value(param)

    def add(self, gate: Gate, **kwargs: dict[str, Any]) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate to add to the circuit.

        Raises:
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        override_parameter_name = kwargs.get("override_parameter_name", "")
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError
        if gate.is_parameterized:
            param_base_label = f"{gate.name}({','.join(map(str, gate.qubits))})"
            for p in gate.get_parameter_names():
                parameter_label = str(override_parameter_name) or param_base_label + f"_{p}_{len(self._parameters)}"
                self._parameters[parameter_label] = gate.parameters[p]
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
