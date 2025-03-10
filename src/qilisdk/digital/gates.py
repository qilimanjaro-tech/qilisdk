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

from abc import ABC
from typing import ClassVar

import numpy as np
from scipy.linalg import expm

from .exceptions import InvalidParameterNameError, ParametersNotEqualError


class Gate(ABC):
    """
    Represents a quantum gate that can be used in quantum circuits.
    """

    NAME: ClassVar[str]
    PARAMETER_NAMES: ClassVar[list[str]]

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
        return self.NAME

    @property
    def matrix(self) -> np.ndarray:
        """
        Retrieve the matrix of the gate.

        Returns:
            np.ndarray: The matrix of the gate.
        """
        return self._matrix

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
        return len(self.qubits)

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
        return len(self.PARAMETER_NAMES)

    @property
    def parameter_names(self) -> list[str]:
        """
        Retrieve the symbolic names of the gate's parameters.

        Returns:
            list[str]: A list containing the names of the parameters.
        """
        return self.PARAMETER_NAMES

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
        return dict(zip(self.PARAMETER_NAMES, self._parameter_values))

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Set the parameters for the gate using a dictionary mapping names to values.

        Args:
            parameters (dict[str, float]): A dictionary where keys are parameter names and values are the new parameter values.

        Raises:
            InvalidParameterNameError: If any provided parameter name is not valid for this gate.
        """
        if any(name not in self.PARAMETER_NAMES for name in parameters):
            raise InvalidParameterNameError

        for name, value in parameters.items():
            self._parameter_values[self.PARAMETER_NAMES.index(name)] = value

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


class Unitary(Gate):
    def controlled(self, *control_qubits: int) -> Controlled:
        """
        Creates a controlled version of this unitary gate.

        This method returns a new instance of a Controlled gate where the provided qubits serve as the
        control qubits and the current unitary gate is used as the target. The resulting gate operates
        on both the control and target qubits.

        Args:
            *control_qubits (int): One or more integer indices specifying the control qubits.

        Returns:
            Controlled: A new Controlled gate instance that wraps this unitary gate with the specified control qubits.
        """
        return Controlled(*control_qubits, gate=self)

    def adjoint(self) -> Adjoint:
        """
        Returns the adjoint (conjugate transpose) of this unitary gate.

        This method constructs and returns a new gate whose matrix is the conjugate transpose of the current
        gate's matrix. The adjoint (or Hermitian conjugate) is commonly used to reverse the effect of a unitary operation.

        Returns:
            Adjoint: A new Adjoint gate instance representing the adjoint of this gate.
        """
        return Adjoint(self)

    def exponential(self) -> Exponential:
        """
        Returns the exponential of this unitary gate.

        This method computes the matrix exponential of the current gate's matrix, resulting in a new gate.
        The matrix exponential is useful for representing continuous time evolution in quantum systems and
        appears in various quantum mechanics and quantum computing applications.

        Returns:
            Exponential: A new Exponential gate instance whose matrix is the matrix exponential of the current gate's matrix.
        """
        return Exponential(self)


class Controlled(Gate):
    def __init__(self, *control_qubits: int, gate: Unitary) -> None:
        super().__init__()
        self._gate = gate
        self._control_qubits = control_qubits
        self._target_qubits = gate.qubits
        self._matrix = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        I_full = np.eye(1 << self.nqubits, dtype=complex)
        # Construct projector P_control = |1...1><1...1| on the n control qubits.
        P = np.array([[0, 0], [0, 1]], dtype=complex)
        for _ in range(len(self.control_qubits) - 1):
            P = np.kron(P, np.array([[0, 0], [0, 1]], dtype=complex))
        # Extend the projector to the full space (control qubits ⊗ target qubit). It acts as P_control ⊗ I_target.
        I_target = np.eye(1 << len(self.target_qubits), dtype=complex)
        # The controlled gate is then:
        controlled = I_full + np.kron(P, self._gate.matrix - I_target)
        return controlled

    @property
    def name(self) -> str:
        return "C" * len(self.control_qubits) + self._gate.name


class Adjoint(Gate):
    """
    Represents the adjoint (conjugate transpose) of a unitary gate.
    """

    def __init__(self, gate: Unitary) -> None:
        super().__init__()
        self._gate = gate
        self._target_qubits = gate.target_qubits
        self._matrix = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        return self._gate.matrix.conj().T

    @property
    def name(self) -> str:
        """
        Get the name of the adjoint gate.

        The name is constructed by appending an adjoint symbol (†) to the name of the underlying gate.
        For example, if the underlying gate's name is "X", this property returns "X†".

        Returns:
            str: The name of the adjoint gate.
        """
        return self._gate.name + "†"


class Exponential(Gate):
    """
    Represents the exponential of a unitary gate.
    The matrix of this gate is computed as the matrix exponential (e^(gate.matrix))
    of the underlying gate's matrix.
    """

    def __init__(self, gate: Unitary) -> None:
        super().__init__()
        self._gate = gate
        self._target_qubits = gate.target_qubits
        self._matrix = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        return expm(self._gate.matrix)

    @property
    def name(self) -> str:
        """
        Get the name of the exponential gate.

        The name is constructed by wrapping the underlying gate's name within "exp(...)".
        For example, if the underlying gate's name is "X", the exponential gate's name is "exp(X)".

        Returns:
            str: The generated name of the exponential gate.
        """
        return f"e^{self._gate.name}"


class M(Gate):
    """
    Measurement operation on a qubit. The measurement is performed in the computational basis and the
    result is stored in a classical bit with the same label as the measured qubit.
    """

    NAME: ClassVar[str] = "M"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, *qubits: int) -> None:
        """
        Initialize a measurement operation.

        Args:
            qubit (int): The qubit index to be measured.
        """
        super().__init__()
        self._target_qubits = qubits


class X(Unitary):
    """
    The Pauli-X gate.

    The associated matrix is:
        [[0, 1],
        [1, 0]]

    This is a pi radians rotation around the X-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "X"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-X gate.

        Args:
            qubit (int): The target qubit index for the X gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, 1], [1, 0]], dtype=complex)


class Y(Unitary):
    """
    The Pauli-Y gate.

    The associated matrix is:
        [[0, -i],
         [i, 0]]

    This is a pi radians rotation around the Y-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "Y"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Y gate.

        Args:
            qubit (int): The target qubit index for the Y gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)


class Z(Unitary):
    """
    The Pauli-Z gate.

    The associated matrix is:
        [[1, 0],
         [0, -1]]

    This is a pi radians rotation around the Z-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "Z"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Z gate.

        Args:
            qubit (int): The target qubit index for the Z gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, -1]], dtype=complex)


class H(Unitary):
    """
    The Hadamard gate.

    The associated matrix is:
        1/sqrt(2) * [[1, 1],
                     [1, -1]]

    This is a pi radians rotation around the (X+Z)-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "H"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Hadamard gate.

        Args:
            qubit (int): The target qubit index for the Hadamard gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


class S(Unitary):
    """
    Represents the S gate, which induces a pi/2 phase.

    The associated matrix is:
        [[1, 0],
         [0, i]]

    This gate is also known as the square root of Z gate: `S**2=Z`, or equivalently it is a pi/2 radians rotation around the Z-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "S"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize an S gate.

        Args:
            qubit (int): The target qubit index for the S gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, 1j]], dtype=complex)


class T(Unitary):
    """
    Represents the T gate, which induces a pi/4 phase.

    The associated matrix is:
        [[1,           0],
         [0, exp(i*pi/4)]]

    This gate is also known as the fourth-root of Z gate: `T**4=Z`, or equivalently it is a pi/4 radians rotation around the Z-axis in the Bloch sphere.
    """

    NAME: ClassVar[str] = "T"
    PARAMETER_NAMES: ClassVar[list[str]] = []

    def __init__(self, qubit: int) -> None:
        """
        Initialize a T gate.

        Args:
            qubit (int): The target qubit index for the T gate.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


class RX(Unitary):
    """
    Represents a `theta` angle rotation around the X-axis (polar) in the Bloch sphere.

    The associated matrix is:
        [[cos(theta/2),     -i*sin(theta/2)],
         [-i*sin(theta/2),     cos(theta/2)]]

    This is an exponential of the Pauli-X operator: `RX(theta) = exp(-i*theta*X/2)`.
    """

    NAME: ClassVar[str] = "RX"
    PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

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


class RY(Unitary):
    """
    Represents a `theta` angle rotation around the Y-axis (polar) in the Bloch sphere.

    The associated matrix is:
        [[cos(theta/2), -sin(theta/2)],
         [sin(theta/2),  cos(theta/2)]]

    This is an exponential of the Pauli-Y operator: `RY(theta) = exp(-i*theta*Y/2)`.
    """

    NAME: ClassVar[str] = "RY"
    PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

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


class RZ(Unitary):
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

    NAME: ClassVar[str] = "RZ"
    PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

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


class U1(Unitary):
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

    NAME: ClassVar[str] = "U1"
    PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

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


class U2(Unitary):
    """
    Represents the U2 gate defined by the angles `phi` and `gamma`.

    The associated matrix is:
        1/sqrt(2)*[[1,                   -exp(i*gamma)],
                   [exp(i*phi),    exp(i*(phi+gamma))]]

    Which is equivalent to two azimuthal rotations of `phi` and `gamma`, with a pi/2 polar rotation in between:
        `U2(phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(pi/2) RZ(gamma)`

    This is the same matrix of `qiskit` and `pennylane`, differing from `qibo` implementation on a global phase:
        `U2(phi, gamma) = U2_qiskit/pennylane(phi, gamma) = exp(i*(phi+gamma)/2) U2_qibo(phi, gamma)`

    Other unitaries you can get from this one are:
        - `U2(phi=0, gamma=np.pi) = H`
        - `U2(phi=0, gamma=0) = RY(theta=pi/2)`
        - `U2(phi=-pi/2, gamma=pi/2) = RX(theta=pi/2)`
    """

    NAME: ClassVar[str] = "U2"
    PARAMETER_NAMES: ClassVar[list[str]] = ["phi", "gamma"]

    def __init__(self, qubit: int, *, phi: float, gamma: float) -> None:
        """
        Initialize a U2 gate.

        Args:
            qubit (int): The target qubit index for the U2 gate.
            phi (float): The first phase parameter, or equivalently the first rotation angle (azimuthal) in radians.
            gamma (float): The second phase parameter, or equivalently the second rotation angle (azimuthal) in radians..
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [phi, gamma]
        self._matrix = (1 / np.sqrt(2)) * np.array(
            [
                [1, -np.exp(1j * gamma)],
                [np.exp(1j * phi), np.exp(1j * (phi + gamma))],
            ],
            dtype=complex,
        )


class U3(Unitary):
    """
    Represents the U3 gate defined by the angles `theta`, `phi` and `gamma`.

    The associated matrix is:
        [[cos(theta/2), -exp(i*gamma/2*sin(theta/2))],
         [exp(i*phi/2)*sin(theta/2),    exp(i*(phi+gamma))*cos(theta/2)]]

    Which is equivalent to two azimuthal rotations of `phi` and `gamma`, with a 'theta' polar rotation in between:
        `U3(theta, phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(theta) RZ(gamma)`

    This is the same matrix of `qiskit` and `pennylane`, differing from `QASM` and `qibo` implementation on a global phase:
        `U3(theta, phi, gamma) = U3_qiskit/pennylane(theta, phi, gamma) = exp(-i*(phi+gamma)/2) U3_QASM/qibo(theta, phi, gamma)`

    Other unitaries you can get from this one are:
        - `U3(theta=pi/2, phi, gamma) = U2(phi, gamma)`
        - `U3(theta, phi=0, gamma=0) = RY(theta)`
        - `U3(theta, phi=-pi/2, gamma=pi/2) = RX(theta)`
    """

    NAME: ClassVar[str] = "U3"
    PARAMETER_NAMES: ClassVar[list[str]] = ["theta", "phi", "gamma"]

    def __init__(self, qubit: int, *, theta: float, phi: float, gamma: float) -> None:
        """
        Initialize a U3 gate.

        Args:
            qubit (int): The target qubit index for the U3 gate.
            theta (float): The rotation angle (polar), in between both phase rotations (azimuthal).
            phi (float): The first phase parameter, or equivalently the first rotation angle (azimuthal) in radians.
            gamma (float): The second phase parameter, or equivalently the second rotation angle (azimuthal) in radians.
        """
        super().__init__()
        self._target_qubits = (qubit,)
        self._parameter_values = [theta, phi, gamma]
        self._matrix = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * gamma) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + gamma)) * np.cos(theta / 2)],
            ],
            dtype=complex,
        )


# TODO(vyron): we don't need this anymore
# QHC-907
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

    NAME: ClassVar[str] = "CNOT"
    PARAMETER_NAMES: ClassVar[list[str]] = []

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


# TODO(vyron): we don't need this anymore
# QHC-907
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

    NAME: ClassVar[str] = "CZ"
    PARAMETER_NAMES: ClassVar[list[str]] = []

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
