# Copyright 2026 Qilimanjaro Quantum Tech
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

from typing import TypeAlias, cast

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.core import QTensor, ket
from qilisdk.core.qtensor import identity, tensor_prod

from .noise import Noise
from .protocols import AttachmentScope, SupportsStaticLindblad
from .representations import LindbladGenerator, Rate

# (strength, operator on a system qubit, operator on an environment qubit). Environment operator
# indices are local to this environment (0 .. n_environment_qubits - 1).
Coupling: TypeAlias = "tuple[float, PauliOperator | Hamiltonian, PauliOperator | Hamiltonian]"


def _qubits(operator: PauliOperator | Hamiltonian) -> set[int]:
    """Return the qubits a Pauli operator or Hamiltonian acts on.

    Args:
        operator (PauliOperator | Hamiltonian): The operator.

    Returns:
        set[int]: The qubit indices.
    """
    if isinstance(operator, PauliOperator):
        return {operator.qubit}
    return {op.qubit for key in operator.elements for op in key}


def _shifted(operator: PauliOperator | Hamiltonian, shift: int) -> Hamiltonian:
    """Return a Pauli operator or Hamiltonian as a Hamiltonian with every qubit index moved by ``shift``.

    Args:
        operator (PauliOperator | Hamiltonian): The operator to move.
        shift (int): The amount added to every qubit index.

    Returns:
        Hamiltonian: The moved operator.
    """
    if isinstance(operator, PauliOperator):
        return Hamiltonian({(type(operator)(operator.qubit + shift),): 1.0})
    return Hamiltonian(
        {tuple(type(op)(op.qubit + shift) for op in key): coefficient for key, coefficient in operator.elements.items()}
    )


class EnvironmentNoise(Noise):
    """Non-Markovian noise from hidden environment qubits coupled to the system.

    The environment qubits (e.g. TLS defects) are appended after the system qubits, coupled to them
    through a static Hamiltonian, given their own Markovian noise, evolved together with the system
    and traced out before results are returned. The memory comes from the environment, so the
    existing Lindblad integrator is reused unchanged.

    Example:
        .. code-block:: python

            nm = NoiseModel()
            nm.add(
                EnvironmentNoise(
                    n_environment_qubits=1,
                    couplings=[(2 * np.pi * 20e3, PauliZ(0), PauliX(0))],
                    environment_noise={0: [AmplitudeDamping(t1=1e-3)]},
                )
            )
    """

    def __init__(
        self,
        *,
        n_environment_qubits: int,
        couplings: list[Coupling],
        environment_hamiltonian: Hamiltonian | None = None,
        environment_noise: dict[int, list[Noise]] | None = None,
        environment_state: QTensor | None = None,
    ) -> None:
        """Args:
            n_environment_qubits (int): Number of environment qubits (must be > 0).
            couplings (list[Coupling]): Interaction terms ``strength * P_sys (x) P_env``.
            environment_hamiltonian (Hamiltonian | None): Free environment Hamiltonian on local indices.
            environment_noise (dict[int, list[Noise]] | None): Markovian noise on each environment qubit,
                reusing the existing noise types. Each must have a static Lindblad form (e.g.
                AmplitudeDamping, Dephasing), since an environment qubit has no gate duration to
                derive rates from.
            environment_state (QTensor | None): Initial environment ket or density matrix. Defaults to ``|0...0>``.

        Raises:
            ValueError: If n_environment_qubits is not positive, any environment index is out of range,
                any environment noise has no static Lindblad form, or the environment state has the
                wrong dimension.
        """
        if n_environment_qubits <= 0:
            raise ValueError(f"n_environment_qubits must be > 0, got {n_environment_qubits}.")
        environment_operators = [env_op for _, _, env_op in couplings]
        if environment_hamiltonian is not None:
            environment_operators.append(environment_hamiltonian)
        indices = set(environment_noise or {}).union(*(_qubits(op) for op in environment_operators))
        if any(not 0 <= index < n_environment_qubits for index in indices):
            raise ValueError(
                f"Environment qubit index out of range for {n_environment_qubits} environment qubits: {sorted(indices)}."
            )
        for noises in (environment_noise or {}).values():
            for noise in noises:
                if not isinstance(noise, SupportsStaticLindblad):
                    raise ValueError(f"Environment noise must have a static Lindblad form, got {type(noise).__name__}.")
        if environment_state is not None and environment_state.shape[0] != 2**n_environment_qubits:
            raise ValueError(
                f"Environment state dimension {environment_state.shape[0]} does not match "
                f"{n_environment_qubits} environment qubits."
            )
        self._n_environment_qubits = n_environment_qubits
        self._couplings = couplings
        self._environment_hamiltonian = environment_hamiltonian
        self._environment_noise = environment_noise or {}
        self._environment_state = environment_state

    @property
    def n_environment_qubits(self) -> int:
        """Return the number of environment qubits.

        Returns:
            int: The number of environment qubits.
        """
        return self._n_environment_qubits

    @property
    def couplings(self) -> list[Coupling]:
        """Return the system-environment coupling terms.

        Returns:
            list[Coupling]: The ``(strength, system operator, environment operator)`` terms.
        """
        return self._couplings

    @property
    def environment_hamiltonian(self) -> Hamiltonian | None:
        """Return the free environment Hamiltonian, on local environment indices.

        Returns:
            Hamiltonian | None: The environment Hamiltonian, if any.
        """
        return self._environment_hamiltonian

    @property
    def environment_noise(self) -> dict[int, list[Noise]]:
        """Return the Markovian noise acting on each environment qubit.

        Returns:
            dict[int, list[Noise]]: Local environment qubit index -> noise sources.
        """
        return self._environment_noise

    @property
    def environment_state(self) -> QTensor:
        """Return the initial environment state.

        Returns:
            QTensor: The initial environment state, ``|0...0>`` if none was given.
        """
        if self._environment_state is not None:
            return self._environment_state
        return ket(*([0] * self._n_environment_qubits))

    def as_hamiltonian_with_environment(self, *, nqubits: int, offset: int = 0) -> Hamiltonian:
        """Return the coupling plus environment Hamiltonian on the enlarged register.

        Environment qubit ``j`` is placed at index ``nqubits + offset + j``, so several environments
        can be stacked by accumulating ``offset``.

        Args:
            nqubits (int): Number of system qubits.
            offset (int): Number of environment qubits already placed by previous environments.

        Returns:
            Hamiltonian: The static Hamiltonian acting on the system and this environment.

        Raises:
            ValueError: If a coupling acts on a system qubit outside the ``nqubits`` system qubits.
        """
        shift = nqubits + offset
        terms = []
        for strength, system_op, environment_op in self._couplings:
            if any(qubit >= nqubits for qubit in _qubits(system_op)):
                raise ValueError(f"Coupling acts on system qubits {sorted(_qubits(system_op))} of only {nqubits}.")
            terms.append(strength * _shifted(system_op, 0) * _shifted(environment_op, shift))
        if self._environment_hamiltonian is not None:
            terms.append(_shifted(self._environment_hamiltonian, shift))
        return Hamiltonian.sum(terms)

    def as_lindblad_with_environment(self, *, nqubits: int, offset: int = 0) -> LindbladGenerator:
        """Return the jump operators of the environment noise on the enlarged register.

        Args:
            nqubits (int): Number of system qubits.
            offset (int): Number of environment qubits already placed by previous environments.

        Returns:
            LindbladGenerator: Jump operators on qubits ``0 .. nqubits + offset + n_environment_qubits - 1``,
                acting on this environment's qubits.

        Raises:
            ValueError: If an environment noise has a jump operator that is not single-qubit.
        """
        jump_operators: list[QTensor] = []
        rates: list[Rate] = []
        for j, noises in self._environment_noise.items():
            before = identity(nqubits + offset + j)
            after = identity(self._n_environment_qubits - 1 - j)
            for noise in noises:
                generator = cast("SupportsStaticLindblad", noise).as_lindblad()
                for k, operator in enumerate(generator.jump_operators):
                    if operator.shape != (2, 2):
                        raise ValueError(
                            f"Environment noise jump operators must be single-qubit, got {operator.shape}."
                        )
                    jump_operators.append(tensor_prod([before, operator, after]))
                    rates.append(1.0 if generator.rates is None else generator.rates[k])
        return LindbladGenerator(jump_operators=jump_operators, rates=rates)

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        # Couplings name system qubits explicitly and the environment evolves continuously, not per gate
        return frozenset({AttachmentScope.GLOBAL})

    def __repr__(self) -> str:
        return (
            f"{type(self).__qualname__}(n_environment_qubits={self._n_environment_qubits}, couplings={self._couplings})"
        )
