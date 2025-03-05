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

from .exceptions import InvalidParameterNameError, ParametersNotEqualError


class Gate(ABC):
    """
    Represents a quantum gate that can be used in quantum circuits.
    """

    ANY_NUMBER_OF_QUBITS: int = -1

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
        """
        Retrieve the name of the gate.

        Returns:
            str: The name of the gate.
        """
        return self._NAME

    @property
    def control_qubits(self) -> tuple[int, ...]:
        """
        Retrieve the indices of the control qubits.

        Returns:
            tuple[int, ...]: A tuple containing the indices of the control qubits.
        """
        return self._control_qubits

    @property
    def target_qubits(self) -> tuple[int, ...]:
        """
        Retrieve the indices of the target qubits.

        Returns:
            tuple[int, ...]: A tuple containing the indices of the target qubits.
        """
        return self._target_qubits

    @property
    def qubits(self) -> tuple[int, ...]:
        """
        Retrieve all qubits associated with the gate, including both control and target qubits.

        Returns:
            tuple[int, ...]: A tuple of all qubit indices on which the gate operates.
        """
        return self._control_qubits + self._target_qubits

    @property
    def nqubits(self) -> int:
        """
        Retrieve the number of qubits the gate acts upon.

        Returns:
            int: The number of qubits for this gate.
        """
        return self._NQUBITS if self._NQUBITS != self.ANY_NUMBER_OF_QUBITS else len(self.qubits)

    @property
    def is_parameterized(self) -> bool:
        """
        Determine whether the gate requires parameters.

        Returns:
            bool: True if the gate is parameterized; otherwise, False.
        """
        return self.nparameters > 0

    @property
    def nparameters(self) -> int:
        """
        Retrieve the number of parameters for the gate.

        Returns:
            int: The count of parameters needed by the gate.
        """
        return len(self._PARAMETER_NAMES)

    @property
    def parameter_names(self) -> list[str]:
        """
        Retrieve the symbolic names of the gate's parameters.

        Returns:
            list[str]: A list containing the names of the parameters.
        """
        return self._PARAMETER_NAMES

    @property
    def parameter_values(self) -> list[float]:
        """
        Retrieve the numerical values assigned to the gate's parameters.

        Returns:
            list[float]: A list containing the parameter values.
        """
        return self._parameter_values

    @property
    def parameters(self) -> dict[str, float]:
        """
        Retrieve a mapping of parameter names to their corresponding values.

        Returns:
            dict[str, float]: A dictionary mapping each parameter name to its numeric value.
        """
        return dict(zip(self._PARAMETER_NAMES, self._parameter_values))

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Set the parameters for the gate using a dictionary mapping names to values.

        Args:
            parameters (dict[str, float]): A dictionary where keys are parameter names and values are the new parameter values.

        Raises:
            InvalidParameterNameError: If any provided parameter name is not valid for this gate.
        """
        if any(name not in self._PARAMETER_NAMES for name in parameters):
            raise InvalidParameterNameError

        for name, value in parameters.items():
            self._parameter_values[self._PARAMETER_NAMES.index(name)] = value

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set the numerical values for the gate's parameters.

        Args:
            values (list[float]): A list containing new parameter values.

        Raises:
            ParametersNotEqualError: If the number of provided values does not match the expected parameter count.
        """
        if len(values) != len(self._parameter_values):
            raise ParametersNotEqualError

        for i, value in enumerate(values):
            self._parameter_values[i] = value


class M(Gate):
    """
    Measurement operation on a qubit. The measurement is performed in the computational basis and the
    result is stored in a classical bit with the same label as the measured qubit.
    """

    _NAME: ClassVar[str] = "M"
    _NQUBITS: ClassVar[int] = -1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, *qubits: int) -> None:
        """
        Initialize a measurement operation.

        Args:
            qubit (int): The qubit index to be measured.
        """
        super().__init__()
        self._target_qubits = qubits


class X(Gate):
    """
    The Pauli-X gate.

    The associated matrix is:
        [[0, 1],
        [1, 0]]

    This is a pi radians rotation around the X-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "X"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-X gate.

        Args:
            qubit (int): The target qubit index for the X gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, 1], [1, 0]], dtype=complex)


class Y(Gate):
    """
    The Pauli-Y gate.

    The associated matrix is:
        [[0, -i],
         [i, 0]]

    This is a pi radians rotation around the Y-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "Y"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Y gate.

        Args:
            qubit (int): The target qubit index for the Y gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)


class Z(Gate):
    """
    The Pauli-Z gate.

    The associated matrix is:
        [[1, 0],
         [0, -1]]

    This is a pi radians rotation around the Z-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "Z"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Z gate.

        Args:
            qubit (int): The target qubit index for the Z gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, -1]], dtype=complex)


class H(Gate):
    """
    The Hadamard gate.

    The associated matrix is:
        1/sqrt(2) * [[1, 1],
                     [1, -1]]

    This is a pi radians rotation around the (X+Z)-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "H"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Hadamard gate.

        Args:
            qubit (int): The target qubit index for the Hadamard gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


class S(Gate):
    """
    Represents the S gate, which induces a pi/2 phase.

    The associated matrix is:
        [[1, 0],
         [0, i]]

    This gate is also known as the square root of Z gate: `S**2=Z`, or equivalently it is a pi/2 radians rotation around the Z-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "S"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize an S gate.

        Args:
            qubit (int): The target qubit index for the S gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, 1j]], dtype=complex)


class T(Gate):
    """
    Represents the T gate, which induces a pi/4 phase.

    The associated matrix is:
        [[1,           0],
         [0, exp(i*pi/4)]]

    This gate is also known as the fourth-root of Z gate: `T**4=Z`, or equivalently it is a pi/4 radians rotation around the Z-axis in the Bloch sphere.
    """

    _NAME: ClassVar[str] = "T"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a T gate.

        Args:
            qubit (int): The target qubit index for the T gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


class RX(Gate):
    """
    Represents a `theta` angle rotation around the X-axis (polar) in the Bloch sphere.

    The associated matrix is:
        [[cos(theta/2),     -i*sin(theta/2)],
         [-i*sin(theta/2),     cos(theta/2)]]

    This is an exponential of the Pauli-X operator: `RX(theta) = exp(-i*theta*X/2)`.
    """

    _NAME: ClassVar[str] = "RX"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        """
        Initialize an RX gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            theta (float): The rotation angle (polar) in radians.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        self._matrix = np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)


class RY(Gate):
    """
    Represents a `theta` angle rotation around the Y-axis (polar) in the Bloch sphere.

    The associated matrix is:
        [[cos(theta/2), -sin(theta/2)],
         [sin(theta/2),  cos(theta/2)]]

    This is an exponential of the Pauli-Y operator: `RY(theta) = exp(-i*theta*Y/2)`.
    """

    _NAME: ClassVar[str] = "RY"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        """
        Initialize an RY gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            theta (float): The rotation angle (polar) in radians.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta]

        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        self._matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


class RZ(Gate):
    """
    Represents a `phi` angle rotation around the Z-axis (azimuthal) in the Bloch sphere.

    The associated matrix is:
        [[exp(-i*phi/2),              0],
         [0,               exp(i*phi/2)]]

    This is an exponential of the Pauli-Z operator: `RZ(phi) = exp(-i*phi*Z/2)`.

    Which is equivalent to the U1 gate plus a global phase: `RZ(phi) = exp(-i*phi/2)U1(phi)`.

    Other unitaries you can get from this one are:
        - `RZ(phi=pi) = exp(-i*pi/2) Z = -i Z`
        - `RZ(phi=pi/2) = exp(-i*pi/4) S`
        - `RZ(phi=pi/4) = exp(-i*pi/8) T`
    """

    _NAME: ClassVar[str] = "RZ"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

    def __init__(self, qubit: int, *, phi: float) -> None:
        """
        Initialize an RZ gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            phi (float): The rotation angle (azimuthal) in radians.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [phi]

        cos_half = np.cos(phi / 2)
        sin_half = np.sin(phi / 2)
        self._matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


class U1(Gate):
    """
    Represents the U1 gate defined by an azimuthal angle `phi`.

    The associated matrix is:
        [[1,            0],
         [0, exp(i*phi)]]

    Which is equivalent to the RZ gate plus a global phase: `U1(phi) = exp(i*phi/2)RZ(phi)`.

    Other unitaries you can get from this one are:
        - `U1(phi=np.pi) = Z`
        - `U1(phi=np.pi/2) = S`
        - `U1(phi=np.pi/4) = T`
    """

    _NAME: ClassVar[str] = "U1"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

    def __init__(self, qubit: int, *, phi: float) -> None:
        """
        Initialize a U1 gate.

        Args:
            qubit (int): The target qubit index for the U1 gate.
            phi (float): The phase to add, or equivalently the rotation angle (azimuthal) in radians.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [phi]
        self._matrix = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


class U2(Gate):
    """
    Represents the U2 gate defined by the angles `phi` and `lam`.

    The associated matrix is:
        1/sqrt(2)*[[1,                   -exp(i*lam/2)],
                   [exp(i*phi/2),    exp(i*(phi+lam))]]

    Which is equivalent to two azimuthal rotations of `phi` and `lam`, with a pi/2 polar rotation in between:
        `U2(phi, lam) = exp(i*(phi+lam)/2) RZ(phi) RY(pi/2) RZ(lam)`

    This is the same matrix of `qiskit` and `pennylane`, differing from `qibo` implementation on a global phase:
        `U2(phi, lam) = U2_qiskit/pennylane(phi, lam) = exp(i*(phi+lam)/2) U2_qibo(phi, lam)`

    Other unitaries you can get from this one are:
        - `U2(phi=0, lam=np.pi) = H`
        - `U2(phi=0, lam=0) = RY(theta=pi/2)`
        - `U2(phi=-pi/2, lam=pi/2) = RX(theta=pi/2)`
    """

    _NAME: ClassVar[str] = "U2"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["phi", "lam"]

    def __init__(self, qubit: int, *, phi: float, lam: float) -> None:
        """
        Initialize a U2 gate.

        Args:
            qubit (int): The target qubit index for the U2 gate.
            phi (float): The first phase parameter, or equivalently the first rotation angle (azimuthal) in radians.
            lam (float): The second phase parameter, or equivalently the second rotation angle (azimuthal) in radians..
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [phi, lam]
        self._matrix = (1 / np.sqrt(2)) * np.array(
            [
                [1, -np.exp(1j * lam / 2)],
                [np.exp(1j * phi / 2), np.exp(1j * (phi + lam))],
            ],
            dtype=complex,
        )


class U3(Gate):
    """
    Represents the U3 gate defined by the angles `theta`, `phi` and `lambda`.

    The associated matrix is:
        [[cos(theta/2), -exp(i*lambda/2*sin(theta/2))],
         [exp(i*phi/2)*sin(theta/2),    exp(i*(phi+lambda))*cos(theta/2)]]

    Which is equivalent to two azimuthal rotations of `phi` and `lam`, with a 'theta' polar rotation in between:
        `U3(theta, phi, lam) = exp(i*(phi+lam)/2) RZ(phi) RY(theta) RZ(lam)`

    This is the same matrix of `qiskit` and `pennylane`, differing from `QASM` and `qibo` implementation on a global phase:
        `U3(theta, phi, lam) = U3_qiskit/pennylane(theta, phi, lam) = exp(-i*(phi+lam)/2) U3_QASM/qibo(theta, phi, lam)`

    Other unitaries you can get from this one are:
        - `U3(theta=pi/2, phi, lam) = U2(phi, lam)`
        - `U3(theta, phi=0, lam=0) = RY(theta)`
        - `U3(theta, phi=-pi/2, lam=pi/2) = RX(theta)`
    """

    _NAME: ClassVar[str] = "U3"
    _NQUBITS: ClassVar[int] = 1
    _PARAMETER_NAMES: ClassVar[list[str]] = ["theta", "phi", "lam"]

    def __init__(self, qubit: int, *, theta: float, phi: float, lam: float) -> None:
        """
        Initialize a U3 gate.

        Args:
            qubit (int): The target qubit index for the U3 gate.
            theta (float): The rotation angle (polar), in between both phase rotations (azimuthal).
            phi (float): The first phase parameter, or equivalently the first rotation angle (azimuthal) in radians.
            lam (float): The second phase parameter, or equivalently the second rotation angle (azimuthal) in radians.
        """
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

    Which is equivalent to the CZ gate surrounded by two H's gates on the target qubit:
        `CNOT(control, target) = H(target) CZ(control, target) H(target)`
    """

    _NAME: ClassVar[str] = "CNOT"
    _NQUBITS: ClassVar[int] = 2
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, control: int, target: int) -> None:
        """
        Initialize a CNOT gate.

        Args:
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.
        """
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

    This gate is totally symmetric respect control and target, meaning that both are control and target in reality:
        `CZ(control, target) = CZ(target, control)`

    It is also equivalent to the CNOT gate surrounded by two H's gates on the target qubit:
        `CZ(control, target) = H(target) CNOT(control, target) H(target)`
    """

    _NAME: ClassVar[str] = "CZ"
    _NQUBITS: ClassVar[int] = 2
    _PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, control: int, target: int) -> None:
        """
        Initialize a CZ gate.

        Args:
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.
        """
        super().__init__()
        self._control_qubits = (control,)
        self._target_qubits = (target,)
        self._matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
