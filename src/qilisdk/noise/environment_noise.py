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

from typing import TYPE_CHECKING, TypeAlias

from .noise import Noise
from .protocols import AttachmentScope

if TYPE_CHECKING:
    from qilisdk.analog import Hamiltonian
    from qilisdk.analog.hamiltonian import PauliOperator
    from qilisdk.core import QTensor

    from .representations import LindbladGenerator

# (strength, operator on a system qubit, operator on an environment qubit). Environment operator
# indices are local to this environment (0 .. n_environment_qubits - 1).
Coupling: TypeAlias = "tuple[float, PauliOperator, PauliOperator]"


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
                reusing the existing noise types (must support a Lindblad representation).
            environment_state (QTensor | None): Initial environment ket or density matrix. Defaults to ``|0...0>``.

        Raises:
            ValueError: If n_environment_qubits is not positive or any environment index is out of range.
        """
        # TODO(luke): validate n_environment_qubits > 0, environment indices in couplings / environment_hamiltonian /
        #       environment_noise < n_environment_qubits, environment_state dimension == 2**n_environment_qubits,
        #       environment_noise entries support SupportsStaticLindblad or SupportsTimeDerivedLindblad.
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
        # TODO(luke): build |0...0> on n_environment_qubits when None
        raise NotImplementedError

    def as_hamiltonian_with_environment(self, *, nqubits: int, offset: int = 0) -> Hamiltonian:
        """Return the coupling plus environment Hamiltonian on the enlarged register.

        Environment qubit ``j`` is placed at index ``nqubits + offset + j``, so several environments
        can be stacked by accumulating ``offset``.

        Args:
            nqubits (int): Number of system qubits.
            offset (int): Number of environment qubits already placed by previous environments.

        Returns:
            Hamiltonian: The static Hamiltonian acting on the system and this environment.
        """
        # TODO(luke): sum(strength * P_sys * shift(P_env)) + shift(environment_hamiltonian)
        raise NotImplementedError

    def as_lindblad_with_environment(self, *, nqubits: int, offset: int = 0) -> LindbladGenerator:
        """Return the jump operators of the environment noise on the enlarged register.

        Args:
            nqubits (int): Number of system qubits.
            offset (int): Number of environment qubits already placed by previous environments.

        Returns:
            LindbladGenerator: Full-register jump operators acting on this environment's qubits.
        """
        # TODO(luke): for j, noises in environment_noise: noise.as_lindblad() -> expand onto qubit nqubits + offset + j
        raise NotImplementedError

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        # Couplings name system qubits explicitly and the environment evolves continuously, not per gate
        return frozenset({AttachmentScope.GLOBAL})

    def __repr__(self) -> str:
        return (
            f"{type(self).__qualname__}(n_environment_qubits={self._n_environment_qubits}, couplings={self._couplings})"
        )
