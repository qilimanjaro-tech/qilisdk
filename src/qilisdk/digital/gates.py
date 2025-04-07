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

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import numpy as np
from scipy.linalg import expm
from typing_extensions import Self

from qilisdk.yaml import yaml

from .exceptions import (
    GateHasNoMatrixError,
    GateNotParameterizedError,
    InvalidParameterNameError,
    ParametersNotEqualError,
)

TBasicGate = TypeVar("TBasicGate", bound="BasicGate")


class Gate(ABC):
    """
    Represents a quantum gate that can be used in quantum circuits.
    """

    PARAMETER_NAMES: ClassVar[list[str]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Retrieve the name of the gate.

        Returns:
            str: The name of the gate.
        """

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """
        Retrieve the matrix of the gate.

        Raises:
            GateHasNoMatrixError: If gate has no matrix.

        Returns:
            np.ndarray: The matrix of the gate.
        """

    @property
    def control_qubits(self) -> tuple[int, ...]:
        """
        Retrieve the indices of the control qubits.

        Returns:
            tuple[int, ...]: A tuple containing the indices of the control qubits.
        """
        return ()

    @property
    def target_qubits(self) -> tuple[int, ...]:
        """
        Retrieve the indices of the target qubits.

        Returns:
            tuple[int, ...]: A tuple containing the indices of the target qubits.
        """
        return ()

    @property
    def qubits(self) -> tuple[int, ...]:
        """
        Retrieve all qubits associated with the gate, including both control and target qubits.

        Returns:
            tuple[int, ...]: A tuple of all qubit indices on which the gate operates.
        """
        return self.control_qubits + self.target_qubits

    @property
    def nqubits(self) -> int:
        """
        Retrieve the number of qubits the gate acts upon.

        Returns:
            int: The number of qubits for this gate.
        """
        return len(self.qubits)

    @property
    def parameters(self) -> dict[str, float]:
        """
        Retrieve a mapping of parameter names to their corresponding values.

        Returns:
            dict[str, float]: A dictionary mapping each parameter name to its numeric value.
        """
        return {}

    @property
    def parameter_names(self) -> list[str]:
        """
        Retrieve the symbolic names of the gate's parameters.

        Returns:
            list[str]: A list containing the names of the parameters.
        """
        return list(self.parameters.keys())

    @property
    def parameter_values(self) -> list[float]:
        """
        Retrieve the numerical values assigned to the gate's parameters.

        Returns:
            list[float]: A list containing the parameter values.
        """
        return list(self.parameters.values())

    @property
    def nparameters(self) -> int:
        """
        Retrieve the number of parameters for the gate.

        Returns:
            int: The count of parameters needed by the gate.
        """
        return len(self.parameters)

    @property
    def is_parameterized(self) -> bool:
        """
        Determine whether the gate requires parameters.

        Returns:
            bool: True if the gate is parameterized; otherwise, False.
        """
        return self.nparameters != 0

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Set the parameters for the gate using a dictionary mapping names to values.

        Args:
            parameters (dict[str, float]): A dictionary where keys are parameter names and values are the new parameter values.

        Raises:
            GateNotParameterizedError: If gate is not parameterized.
            InvalidParameterNameError: If any provided parameter name is not valid for this gate.
        """
        if not self.is_parameterized:
            raise GateNotParameterizedError

        if any(name not in self.parameters for name in parameters):
            raise InvalidParameterNameError

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set the numerical values for the gate's parameters.

        Args:
            values (list[float]): A list containing new parameter values.

        Raises:
            GateNotParameterizedError: If gate is not parameterized.
            ParametersNotEqualError: If the number of provided values does not match the expected parameter count.
        """
        if not self.is_parameterized:
            raise GateNotParameterizedError

        if len(values) != len(self.parameters):
            raise ParametersNotEqualError

    def __repr__(self) -> str:
        qubits_str = f"({self.qubits[0]})" if self.nqubits == 1 else str(self.qubits)
        return f"{self.name}{qubits_str}"


class BasicGate(Gate):
    """
    Represents a quantum gate that can be used in quantum circuits.
    """

    def __init__(self, target_qubits: tuple[int, ...], parameters: dict[str, float] = {}) -> None:
        # Check for duplicate integers in target_qubits.
        if len(target_qubits) != len(set(target_qubits)):
            raise ValueError("Duplicate target qubits found.")

        self._target_qubits: tuple[int, ...] = target_qubits
        self._parameters: dict[str, float] = parameters
        self._matrix: np.ndarray = self._generate_matrix()

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def target_qubits(self) -> tuple[int, ...]:
        return self._target_qubits

    @property
    def parameters(self) -> dict[str, float]:
        return dict(self._parameters)

    def set_parameters(self, parameters: dict[str, float]) -> None:
        super().set_parameters(parameters=parameters)
        self._parameters.update(parameters)
        self._matrix = self._generate_matrix()

    def set_parameter_values(self, values: list[float]) -> None:
        super().set_parameter_values(values=values)

        for key, value in zip(self.parameters, values):
            self._parameters[key] = value

        self._matrix = self._generate_matrix()

    def controlled(self: Self, *control_qubits: int) -> Controlled[Self]:
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
        return Controlled(*control_qubits, basic_gate=self)

    def adjoint(self: Self) -> Adjoint[Self]:
        """
        Returns the adjoint (conjugate transpose) of this unitary gate.

        This method constructs and returns a new gate whose matrix is the conjugate transpose of the current
        gate's matrix. The adjoint (or Hermitian conjugate) is commonly used to reverse the effect of a unitary operation.

        Returns:
            Adjoint: A new Adjoint gate instance representing the adjoint of this gate.
        """
        return Adjoint(basic_gate=self)

    def exponential(self: Self) -> Exponential[Self]:
        """
        Returns the exponential of this unitary gate.

        This method computes the matrix exponential of the current gate's matrix, resulting in a new gate.
        The matrix exponential is useful for representing continuous time evolution in quantum systems and
        appears in various quantum mechanics and quantum computing applications.

        Returns:
            Exponential: A new Exponential gate instance whose matrix is the matrix exponential of the current gate's matrix.
        """
        return Exponential(basic_gate=self)

    @abstractmethod
    def _generate_matrix(self) -> np.ndarray: ...


@yaml.register_class
class Modified(Gate, Generic[TBasicGate]):
    def __init__(self, basic_gate: TBasicGate) -> None:
        self._basic_gate: TBasicGate = basic_gate
        self._matrix: np.ndarray

    @property
    def basic_gate(self) -> TBasicGate:
        return self._basic_gate

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def target_qubits(self) -> tuple[int, ...]:
        return self._basic_gate.target_qubits

    @property
    def parameters(self) -> dict[str, float]:
        return self._basic_gate.parameters

    @property
    def parameter_names(self) -> list[str]:
        return self._basic_gate.parameter_names

    @property
    def parameter_values(self) -> list[float]:
        return self._basic_gate.parameter_values

    @property
    def nparameters(self) -> int:
        return self._basic_gate.nparameters

    @property
    def is_parameterized(self) -> bool:
        return self._basic_gate.is_parameterized

    def set_parameters(self, parameters: dict[str, float]) -> None:
        self._basic_gate.set_parameters(parameters=parameters)
        self._matrix = self._generate_matrix()

    def set_parameter_values(self, values: list[float]) -> None:
        self._basic_gate.set_parameter_values(values=values)
        self._matrix = self._generate_matrix()

    @abstractmethod
    def _generate_matrix(self) -> np.ndarray: ...

    def is_modified_from(self, gate_type: type[TBasicGate]) -> bool:
        return isinstance(self.basic_gate, gate_type)


@yaml.register_class
class Controlled(Modified[TBasicGate]):
    def __init__(self, *control_qubits: int, basic_gate: TBasicGate) -> None:
        super().__init__(basic_gate=basic_gate)

        # Check for duplicate integers in control_qubits.
        if len(control_qubits) != len(set(control_qubits)):
            raise ValueError("Duplicate control qubits found.")

        # Check if any integer in control_qubits is also in unitary_gate.target_qubits.
        if set(control_qubits) & set(basic_gate.target_qubits):
            raise ValueError("Some control qubits are the same as unitary gate's target qubits.")

        self._control_qubits = control_qubits
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
        controlled = I_full + np.kron(P, self.basic_gate.matrix - I_target)
        return controlled

    @property
    def control_qubits(self) -> tuple[int, ...]:
        return self._control_qubits

    @property
    def name(self) -> str:
        return "C" * len(self.control_qubits) + self.basic_gate.name


@yaml.register_class
class Adjoint(Modified[TBasicGate]):
    """
    Represents the adjoint (conjugate transpose) of a unitary gate.
    """

    def __init__(self, basic_gate: TBasicGate) -> None:
        super().__init__(basic_gate=basic_gate)
        self._matrix = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        return self.basic_gate.matrix.conj().T

    @property
    def name(self) -> str:
        """
        Get the name of the adjoint gate.

        The name is constructed by appending an adjoint symbol (†) to the name of the underlying gate.
        For example, if the underlying gate's name is "X", this property returns "X†".

        Returns:
            str: The name of the adjoint gate.
        """
        return self.basic_gate.name + "†"


@yaml.register_class
class Exponential(Modified[TBasicGate]):
    """
    Represents the exponential of a unitary gate.
    The matrix of this gate is computed as the matrix exponential (e^(gate.matrix))
    of the underlying gate's matrix.
    """

    def __init__(self, basic_gate: TBasicGate) -> None:
        super().__init__(basic_gate=basic_gate)
        self._matrix = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        return expm(self.basic_gate.matrix)

    @property
    def name(self) -> str:
        """
        Get the name of the exponential gate.

        The name is constructed by prepending the underlying gate's name within "e^".
        For example, if the underlying gate's name is "X", the exponential gate's name is "e^X".

        Returns:
            str: The generated name of the exponential gate.
        """
        return f"e^{self.basic_gate.name}"


@yaml.register_class
class M(Gate):
    """
    Measurement operation on a qubit. The measurement is performed in the computational basis and the
    result is stored in a classical bit with the same label as the measured qubit.
    """

    def __init__(self, *qubits: int) -> None:
        """
        Initialize a measurement operation.

        Args:
            qubit (int): The qubit index to be measured.
        """
        self._target_qubits = qubits

    @property
    def name(self) -> str:
        return "M"

    @property
    def matrix(self) -> np.ndarray:
        raise GateHasNoMatrixError

    @property
    def target_qubits(self) -> tuple[int, ...]:
        return self._target_qubits


@yaml.register_class
class X(BasicGate):
    """
    The Pauli-X gate.

    The associated matrix is:
        [[0, 1],
        [1, 0]]

    This is a pi radians rotation around the X-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-X gate.

        Args:
            qubit (int): The target qubit index for the X gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "X"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[0, 1], [1, 0]], dtype=complex)


@yaml.register_class
class Y(BasicGate):
    """
    The Pauli-Y gate.

    The associated matrix is:

    .. code-block:: text
        [[0, -i],
         [i, 0]]

    This is a pi radians rotation around the Y-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Y gate.

        Args:
            qubit (int): The target qubit index for the Y gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "Y"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[0, -1j], [1j, 0]], dtype=complex)


@yaml.register_class
class Z(BasicGate):
    """
    The Pauli-Z gate.

    The associated matrix is:

    .. code-block:: text
        [[1, 0],
         [0, -1]]

    This is a pi radians rotation around the Z-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Pauli-Z gate.

        Args:
            qubit (int): The target qubit index for the Z gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "Z"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[1, 0], [0, -1]], dtype=complex)


@yaml.register_class
class H(BasicGate):
    """
    The Hadamard gate.

    The associated matrix is:

    .. code-block:: text
        1/sqrt(2) * [[1, 1],
                     [1, -1]]

    This is a pi radians rotation around the (X+Z)-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize a Hadamard gate.

        Args:
            qubit (int): The target qubit index for the Hadamard gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "H"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


@yaml.register_class
class S(BasicGate):
    """
    Represents the S gate, which induces a pi/2 phase.

    The associated matrix is:

    .. code-block:: text
        [[1, 0],
         [0, i]]

    This gate is also known as the square root of Z gate: ``S**2=Z``, or equivalently it is a pi/2 radians rotation around the Z-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize an S gate.

        Args:
            qubit (int): The target qubit index for the S gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "S"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[1, 0], [0, 1j]], dtype=complex)


@yaml.register_class
class T(BasicGate):
    """
    Represents the T gate, which induces a pi/4 phase.

    The associated matrix is:

    .. code-block:: text
        [[1,           0],
         [0, exp(i*pi/4)]]

    This gate is also known as the fourth-root of Z gate: ``T**4=Z``, or equivalently it is a pi/4 radians rotation around the Z-axis in the Bloch sphere.
    """

    def __init__(self, qubit: int) -> None:
        """
        Initialize a T gate.

        Args:
            qubit (int): The target qubit index for the T gate.
        """
        super().__init__(target_qubits=(qubit,))

    @property
    def name(self) -> str:
        return "T"

    def _generate_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


@yaml.register_class
class RX(BasicGate):
    """
    Represents a `theta` angle rotation around the X-axis (polar) in the Bloch sphere.

    The associated matrix is:

    .. code-block:: text
        [[cos(theta/2),     -i*sin(theta/2)],
         [-i*sin(theta/2),     cos(theta/2)]]

    This is an exponential of the Pauli-X operator:
        ``RX(theta) = exp(-i*theta*X/2)``
    """

    PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        """
        Initialize an RX gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            theta (float): The rotation angle (polar) in radians.
        """
        super().__init__(target_qubits=(qubit,), parameters={"theta": theta})

    @property
    def name(self) -> str:
        return "RX"

    @property
    def theta(self) -> float:
        return self.parameters["theta"]

    def _generate_matrix(self) -> np.ndarray:
        theta = self.parameters["theta"]
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)


@yaml.register_class
class RY(BasicGate):
    """
    Represents a `theta` angle rotation around the Y-axis (polar) in the Bloch sphere.

    The associated matrix is:

    .. code-block:: text
        [[cos(theta/2), -sin(theta/2)],
         [sin(theta/2),  cos(theta/2)]]

    This is an exponential of the Pauli-Y operator:
        ``RY(theta) = exp(-i*theta*Y/2)``
    """

    PARAMETER_NAMES: ClassVar[list[str]] = ["theta"]

    def __init__(self, qubit: int, *, theta: float) -> None:
        """
        Initialize an RY gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            theta (float): The rotation angle (polar) in radians.
        """
        super().__init__(target_qubits=(qubit,), parameters={"theta": theta})

    @property
    def name(self) -> str:
        return "RY"

    @property
    def theta(self) -> float:
        return self.parameters["theta"]

    def _generate_matrix(self) -> np.ndarray:
        theta = self.parameters["theta"]
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


@yaml.register_class
class RZ(BasicGate):
    """
    Represents a `phi` angle rotation around the Z-axis (azimuthal) in the Bloch sphere.

    The associated matrix is:

    .. code-block:: text
        [[exp(-i*phi/2),              0],
         [0,               exp(i*phi/2)]]

    This is an exponential of the Pauli-Z operator:
        ``RZ(phi) = exp(-i*phi*Z/2)``

    Which is equivalent to the U1 gate plus a global phase:
        ``RZ(phi) = exp(-i*phi/2)U1(phi)``

    Other unitaries you can get from this one are:
        - ``RZ(phi=pi) = exp(-i*pi/2) Z = -i Z``
        - ``RZ(phi=pi/2) = exp(-i*pi/4) S``
        - ``RZ(phi=pi/4) = exp(-i*pi/8) T``
    """

    PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

    def __init__(self, qubit: int, *, phi: float) -> None:
        """
        Initialize an RZ gate.

        Args:
            qubit (int): The target qubit index for the rotation.
            phi (float): The rotation angle (azimuthal) in radians.
        """
        super().__init__(target_qubits=(qubit,), parameters={"phi": phi})

    @property
    def name(self) -> str:
        return "RZ"

    @property
    def theta(self) -> float:
        return self.parameters["theta"]

    def _generate_matrix(self) -> np.ndarray:
        phi = self.parameters["phi"]
        cos_half = np.cos(phi / 2)
        sin_half = np.sin(phi / 2)
        return np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)


@yaml.register_class
class U1(BasicGate):
    """
    Represents the U1 gate defined by an azimuthal angle `phi`.

    The associated matrix is:

    .. code-block:: text
        [[1,            0],
         [0, exp(i*phi)]]

    Which is equivalent to the RZ gate plus a global phase:
        ``U1(phi) = exp(i*phi/2)RZ(phi)``

    Other unitaries you can get from this one are:
        - ``U1(phi=np.pi) = Z``
        - ``U1(phi=np.pi/2) = S``
        - ``U1(phi=np.pi/4) = T``
    """

    PARAMETER_NAMES: ClassVar[list[str]] = ["phi"]

    def __init__(self, qubit: int, *, phi: float) -> None:
        """
        Initialize a U1 gate.

        Args:
            qubit (int): The target qubit index for the U1 gate.
            phi (float): The phase to add, or equivalently the rotation angle (azimuthal) in radians.
        """
        super().__init__(target_qubits=(qubit,), parameters={"phi": phi})

    @property
    def name(self) -> str:
        return "U1"

    @property
    def phi(self) -> float:
        return self.parameters["phi"]

    def _generate_matrix(self) -> np.ndarray:
        phi = self.parameters["phi"]
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


@yaml.register_class
class U2(BasicGate):
    """
    Represents the U2 gate defined by the angles `phi` and `gamma`.

    The associated matrix is:

    .. code-block:: text
        1/sqrt(2)*[[1,                   -exp(i*gamma)],
                   [exp(i*phi),    exp(i*(phi+gamma))]]

    Which is equivalent to two azimuthal rotations of `phi` and `gamma`, with a pi/2 polar rotation in between:
        ``U2(phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(pi/2) RZ(gamma)``

    This is the same matrix of `qiskit` and `pennylane`, differing from `qibo` implementation on a global phase:
        ``U2(phi, gamma) = U2_qiskit/pennylane(phi, gamma) = exp(i*(phi+gamma)/2) U2_qibo(phi, gamma)``

    Other unitaries you can get from this one are:
        - ``U2(phi=0, gamma=np.pi) = H``
        - ``U2(phi=0, gamma=0) = RY(theta=pi/2)``
        - ``U2(phi=-pi/2, gamma=pi/2) = RX(theta=pi/2)``
    """

    PARAMETER_NAMES: ClassVar[list[str]] = ["phi", "gamma"]

    def __init__(self, qubit: int, *, phi: float, gamma: float) -> None:
        """
        Initialize a U2 gate.

        Args:
            qubit (int): The target qubit index for the U2 gate.
            phi (float): The first phase parameter, or equivalently the first rotation angle (azimuthal) in radians.
            gamma (float): The second phase parameter, or equivalently the second rotation angle (azimuthal) in radians..
        """
        super().__init__(target_qubits=(qubit,), parameters={"phi": phi, "gamma": gamma})

    @property
    def name(self) -> str:
        return "U2"

    @property
    def phi(self) -> float:
        return self.parameters["phi"]

    @property
    def gamma(self) -> float:
        return self.parameters["gamma"]

    def _generate_matrix(self) -> np.ndarray:
        phi = self.parameters["phi"]
        gamma = self.parameters["gamma"]
        return (1 / np.sqrt(2)) * np.array(
            [
                [1, -np.exp(1j * gamma)],
                [np.exp(1j * phi), np.exp(1j * (phi + gamma))],
            ],
            dtype=complex,
        )


@yaml.register_class
class U3(BasicGate):
    """
    Represents the U3 gate defined by the angles `theta`, `phi` and `gamma`.

    The associated matrix is:

    .. code-block:: text
        [[cos(theta/2), -exp(i*gamma/2*sin(theta/2))],
         [exp(i*phi/2)*sin(theta/2),    exp(i*(phi+gamma))*cos(theta/2)]]

    Which is equivalent to two azimuthal rotations of `phi` and `gamma`, with a 'theta' polar rotation in between:
        ``U3(theta, phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(theta) RZ(gamma)``

    This is the same matrix of `qiskit` and `pennylane`, differing from `QASM` and `qibo` implementation on a global phase:
        ``U3(theta, phi, gamma) = U3_qiskit/pennylane(theta, phi, gamma) = exp(-i*(phi+gamma)/2) U3_QASM/qibo(theta, phi, gamma)``

    Other unitaries you can get from this one are:
        - ``U3(theta=pi/2, phi, gamma) = U2(phi, gamma)``
        - ``U3(theta=0, phi=0, gamma) = U1(gamma)`` and ``U3(theta=0, phi, gamma=0) = U1(phi)``
        - ``U3(theta, phi=0, gamma=0) = RY(theta)``
        - ``U3(theta, phi=-pi/2, gamma=pi/2) = RX(theta)``
    """

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
        super().__init__(target_qubits=(qubit,), parameters={"theta": theta, "phi": phi, "gamma": gamma})

    @property
    def name(self) -> str:
        return "U3"

    @property
    def theta(self) -> float:
        return self.parameters["theta"]

    @property
    def phi(self) -> float:
        return self.parameters["phi"]

    @property
    def gamma(self) -> float:
        return self.parameters["gamma"]

    def _generate_matrix(self) -> np.ndarray:
        theta = self.parameters["theta"]
        phi = self.parameters["phi"]
        gamma = self.parameters["gamma"]
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * gamma) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + gamma)) * np.cos(theta / 2)],
            ],
            dtype=complex,
        )


@yaml.register_class
class CNOT(Controlled[X]):
    """
    Represents the CNOT gate.

    The associated matrix is:

    .. code-block:: text
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]

    Which is equivalent to the CZ gate surrounded by two H's gates on the target qubit:
        ``CNOT(control, target) = H(target) CZ(control, target) H(target)``
    """

    def __init__(self, control: int, target: int) -> None:
        super().__init__(control, basic_gate=X(target))

    @property
    def name(self) -> str:
        return "CNOT"


@yaml.register_class
class CZ(Controlled[Z]):
    """
    Represents the CZ gate.

    The associated matrix is:

    .. code-block:: text
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, -1]]

    This gate is totally symmetric respect control and target, meaning that both are control and target in reality:
        ``CZ(control, target) = CZ(target, control)``

    It is also equivalent to the CNOT gate surrounded by two H's gates on the target qubit:
        ``CZ(control, target) = H(target) CNOT(control, target) H(target)``
    """

    def __init__(self, control: int, target: int) -> None:
        super().__init__(control, basic_gate=Z(target))
