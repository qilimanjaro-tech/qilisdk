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

from abc import ABC
from typing import ClassVar

import numpy as np

from .exceptions import GateHasNoParameter, ParametersNotEqualError


class Gate(ABC):
    """
    Represents a quantum gate.

    Class Attributes:
        name: Name of the gate (e.g., "X", "CNOT", etc.).
        nqubits: The number of qubits this gate acts on.
        parameter_names: The symbolic parameter names for gates that require parameters
                        (e.g., ["theta", "phi"]).

    Instance Attributes:
        _control_qubits: A tuple of qubit indices designated as 'control' qubits.
        _target_qubits: A tuple of qubit indices designated as 'target' qubits.
        _matrix: The unitary matrix representation of the gate (if available).
        parameters: A list of numeric values associated with the gate's parameters.

    Methods:
        get_gate: (Optional) Method to implement for retrieving the gate in a
                  specific backend format.
    """

    _NAME: ClassVar[str]
    _NQUBITS: ClassVar[int]
    _PARAMETER_NAMES: ClassVar[list[str]]

    def __init__(self) -> None:
        self._control_qubits: tuple[int, ...] = ()
        self._target_qubits: tuple[int, ...] = ()
        self._matrix: np.ndarray = np.array([])
        self._parameter_values: list[float] = []

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._NAME

    @property
    def control_qubits(self) -> tuple[int, ...]:
        """_summary_

        Returns:
            tuple[int, ...]: _description_
        """
        return self._control_qubits

    @property
    def target_qubits(self) -> tuple[int, ...]:
        """_summary_

        Returns:
            tuple[int, ...]: _description_
        """
        return self._target_qubits

    @property
    def qubits(self) -> tuple[int, ...]:
        """_summary_

        Returns:
            tuple[int, ...]: _description_
        """
        return self._control_qubits + self._target_qubits

    @property
    def nqubits(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._NQUBITS

    @property
    def is_parameterized(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.nparameters > 0

    @property
    def nparameters(self) -> int:
        """_summary_

        Returns:
            bool: _description_
        """
        return len(self._PARAMETER_NAMES)

    @property
    def parameter_names(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        return self._PARAMETER_NAMES

    @property
    def parameter_values(self) -> list[float]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        return self._parameter_values

    @property
    def parameters(self) -> dict[str, float]:
        """_summary_

        Returns:
            dict[str, float]: _description_
        """
        return dict(zip(self._PARAMETER_NAMES, self._parameter_values))

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """_summary_

        Args:
            parameters (dict[str, float]): _description_

        Raises:
            GateHasNoParameter: _description_
        """
        if any(name not in self._PARAMETER_NAMES for name in parameters):
            raise GateHasNoParameter

        for name, value in parameters.items():
            self._parameter_values[self._PARAMETER_NAMES.index(name)] = value

    def set_parameter_values(self, values: list[float]) -> None:
        """_summary_

        Args:
            values (list[float]): _description_

        Raises:
            ParametersNotEqualError: _description_
        """
        if len(values) != len(self._parameter_values):
            raise ParametersNotEqualError

        for i, value in enumerate(values):
            self._parameter_values[i] = value


class X(Gate):
    """
    Represents the Pauli-X gate, a single-qubit gate with no parameters.

    The associated matrix is:
        [[0, 1],
         [1, 0]]
    """

    _NAME: ClassVar[str] = "X"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, 1], [1, 0]], dtype=complex)


class Y(Gate):
    """
    Represents the Pauli-Y gate, a single-qubit gate with no parameters.

    The associated matrix is:
        [[0, -i],
         [i, 0]]
    """

    _NAME: ClassVar[str] = "Y"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)


class Z(Gate):
    """
    Represents the Pauli-Z gate, a single-qubit gate with no parameters.

    The associated matrix is:
        [[1, 0],
         [0, -1]]
    """

    _NAME: ClassVar[str] = "Z"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, -1]], dtype=complex)


class S(Gate):
    """
    Represents the S gate, a single-qubit gate with no parameters.

    The associated matrix is:
        [[1, 0],
         [0, i]]
    """

    _NAME: ClassVar[str] = "S"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, 1j]], dtype=complex)


class T(Gate):
    """
    Represents the T gate, a single-qubit gate with no parameters.

    The associated matrix is:
        [[1,           0],
         [0, exp(i*pi/4)]]
    """

    _NAME: ClassVar[str] = "T"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


class M(Gate):
    """
    Measurement operation on a qubit. The measurement is performed in the computational basis and the
    result is stored in a classical bit with the same label as the measured qubit.
    """

    _NAME: ClassVar[str] = "M"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        super().__init__()
        self._target_qubits = (qubit,)


class RX(Gate):
    """
    Represents the rotation around the X-axis by an angle `theta`.

    The associated matrix is:
        [[cos(theta/2),     -i*sin(theta/2)],
         [-i*sin(theta/2),     cos(theta/2)]]
    """

    _NAME: ClassVar[str] = "RX"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        self._matrix = np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)


class RY(Gate):
    """
    Represents the rotation around the Y-axis by an angle `theta`.

    The associated matrix is:
        [[cos(theta/2), -sin(theta/2)],
         [sin(theta/2),  cos(theta/2)]]
    """

    _NAME: ClassVar[str] = "RY"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        self._matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


class RZ(Gate):
    """
    Represents the rotation around the Z-axis by an angle `theta`.

    The associated matrix is:
        [[exp(-i*theta/2),              0],
         [0,               exp(i*theta/2)]]
    """

    _NAME: ClassVar[str] = "RZ"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        self._matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


class U1(Gate):
    """
    Represents the U1 gate defined by an angle `theta`.

    The associated matrix is:
        [[1,            0],
         [0, exp(i*theta)]]
    """

    _NAME: ClassVar[str] = "U1"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]
        self._matrix = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


class U2(Gate):
    """
    Represents the U2 gate defined by the angles `phi` and `lambda`.

    The associated matrix is:
        1/sqrt(2)*[[exp(-i*(phi+lambda)/2), -exp(-i*(phi-lambda)/2)],
                   [exp(i*(phi-lambda)/2),    exp(i*(phi+lambda)/2)]]
    """

    _NAME: ClassVar[str] = "U2"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["phi", "lam"]

    def __init__(self, qubit: int, *, phi: float, lam: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [phi, lam]
        self._matrix = (1 / np.sqrt(2)) * np.array(
            [
                [np.exp(-1j * (phi + lam) / 2), -np.exp(-1j * (phi - lam) / 2)],
                [np.exp(1j * (phi - lam) / 2), np.exp(1j * (phi + lam) / 2)],
            ],
            dtype=complex,
        )


class U3(Gate):
    """
    Represents the U3 gate defined by the angles `theta`, `phi` and `lambda`.

    The associated matrix is:
        [[cos(theta/2)*exp(-i*(phi+lambda)/2), -sin(theta/2)*exp(-i*(phi-lambda)/2)],
         [sin(theta/2)*exp(i*(phi-lambda)/2),    cos(theta/2)*exp(i*(phi+lambda)/2)]]
    """

    _NAME: ClassVar[str] = "U3"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta", "phi", "lam"]

    def __init__(self, qubit: int, *, theta: float, phi: float, lam: float) -> None:
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta, phi, lam]
        self._matrix = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
            ],
            dtype=complex,
        )


class CNOT(Gate):
    """
    Represents the CNOT gate.

    The associated matrix is:
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]
    """

    _NAME: ClassVar[str] = "CNOT"
    _NQUBITS: ClassVar[int] = 2
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, control: int, target: int) -> None:
        super().__init__()
        self._control_qubits = (control,)
        self._target_qubits = (target,)
        self._matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


class CZ(Gate):
    """
    Represents the CZ gate.

    The associated matrix is:
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, -1]]
    """

    _NAME: ClassVar[str] = "CZ"
    _NQUBITS: ClassVar[int] = 2
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, control: int, target: int) -> None:
        super().__init__()
        self._control_qubits = (control,)
        self._target_qubits = (target,)
        self._matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
