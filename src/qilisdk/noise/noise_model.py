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

from collections import defaultdict
from typing import TypeAlias, TypeVar, overload

from qilisdk.digital import Gate

from .noise import Noise
from .noise_config import NoiseConfig
from .parameter_perturbation import ParameterPerturbation
from .protocols import AttachmentScope

Qubit: TypeAlias = int
GateType: TypeAlias = type[Gate]
Parameter: TypeAlias = str

TNoise = TypeVar("TNoise", bound=Noise)
TParameterPerturbation = TypeVar("TParameterPerturbation", bound=ParameterPerturbation)


class NoiseModel:
    """Container for noise sources and parameter perturbations.

    Stores global, per-qubit, and per-gate noise along with parameter-level
    perturbations for later application in backends.
    """

    def __init__(self, noise_config: NoiseConfig | None = None) -> None:
        # configuration
        if noise_config is not None:
            self._noise_config = noise_config
        else:
            self._noise_config = NoiseConfig()

        # noises
        self._global_noise: list[Noise] = []
        self._per_qubit_noise: dict[Qubit, list[Noise]] = defaultdict(list)
        self._per_gate_noise: dict[GateType, list[Noise]] = defaultdict(list)
        self._per_gate_per_qubit_noise: dict[tuple[GateType, Qubit], list[Noise]] = defaultdict(list)

        # parameter perturbations
        self._global_perturbations: dict[Parameter, list[ParameterPerturbation]] = defaultdict(list)
        self._per_gate_perturbations: dict[tuple[GateType, Parameter], list[ParameterPerturbation]] = defaultdict(list)

    @property
    def noise_config(self) -> NoiseConfig:
        """Return the noise configuration.

        Returns:
            NoiseConfig: The noise configuration instance.
        """
        return self._noise_config

    @property
    def global_noise(self) -> list[Noise]:
        """Return the list of globally applied noise sources.

        Returns:
            list[Noise]: Noise sources applied to all operations.
        """
        return self._global_noise

    @property
    def per_qubit_noise(self) -> dict[Qubit, list[Noise]]:
        """Return the per-qubit noise mapping.

        Returns:
            dict[int, list[Noise]]: Noise sources keyed by qubit index.
        """
        return self._per_qubit_noise

    @property
    def per_gate_noise(self) -> dict[GateType, list[Noise]]:
        """Return the per-gate-type noise mapping.

        Returns:
            dict[type[Gate], list[Noise]]: Noise sources keyed by gate type.
        """
        return self._per_gate_noise

    @property
    def per_gate_per_qubit_noise(self) -> dict[tuple[GateType, Qubit], list[Noise]]:
        """Return the per-gate-type and per-qubit noise mapping.

        Returns:
            dict[tuple[type[Gate], int], list[Noise]]: Noise sources keyed by gate type and qubit index.
        """
        return self._per_gate_per_qubit_noise

    @property
    def global_perturbations(self) -> dict[Parameter, list[ParameterPerturbation]]:
        """Return the global parameter perturbation mapping.

        Returns:
            dict[str, list[ParameterPerturbation]]: Perturbations keyed by parameter name.
        """
        return self._global_perturbations

    @property
    def per_gate_perturbations(self) -> dict[tuple[GateType, Parameter], list[ParameterPerturbation]]:
        """Return the per-gate-type parameter perturbation mapping.

        Returns:
            dict[tuple[type[Gate], str], list[ParameterPerturbation]]: Perturbations keyed by gate type and parameter.
        """
        return self._per_gate_perturbations

    # Overloads (typing)
    # -----------------------

    # Noise: global/per_qubit/per_gate_type
    @overload
    def add(self, noise: TNoise) -> None: ...
    @overload
    def add(self, noise: TNoise, *, qubits: list[Qubit]) -> None: ...
    @overload
    def add(self, noise: TNoise, *, gate: GateType) -> None: ...

    # ParameterPerturbation: global/per_gate_type
    @overload
    def add(self, noise: TParameterPerturbation, *, parameter: Parameter) -> None: ...
    @overload
    def add(self, noise: TParameterPerturbation, *, gate: GateType, parameter: Parameter) -> None: ...

    # -----------------------
    # Implementation
    # -----------------------

    def add(
        self,
        noise: Noise | ParameterPerturbation,
        *,
        qubits: list[Qubit] | None = None,
        gate: GateType | None = None,
        parameter: Parameter | None = None,
    ) -> None:
        """Attach a noise source or parameter perturbation to the model.

        Args:
            noise (Noise | ParameterPerturbation): The noise or parameter perturbation instance to attach.
            qubits (list[int] | None): Target qubit index or indices for per-qubit noise attachments.
            gate (type[Gate] | None): Target gate type for per-gate noise or perturbation attachments.
            parameter (str | None): Target parameter name for perturbation attachments.

        Raises:
            ValueError: If the noise/perturbation does not allow the inferred scope.
        """
        if isinstance(noise, ParameterPerturbation):
            if qubits is not None:
                raise ValueError(f"{noise.__class__.__name__} cannot be applied to specific qubits.")
            self._add_parameter_perturbation(
                noise=noise,
                gate=gate,
                parameter=parameter,
            )
        else:
            self._add_noise(
                noise=noise,
                qubits=qubits,
                gate=gate,
                parameter=parameter,
            )

    @staticmethod
    def _check_scope_allowed(
        noise: Noise | ParameterPerturbation,
        scope: AttachmentScope,
    ) -> None:
        """Check if the noise/perturbation allows the given scope.

        Args:
            noise (Noise | ParameterPerturbation): The noise or parameter perturbation instance to check.
            scope (AttachmentScope): The scope to check.
        Raises:
            ValueError: If the noise/perturbation does not allow the given scope.
        """
        if scope not in noise.allowed_scopes():
            raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")

    def _add_noise(
        self,
        noise: Noise,
        *,
        qubits: list[Qubit] | None = None,
        gate: GateType | None = None,
        parameter: Parameter | None = None,
    ) -> None:
        """Attach a noise source to the model.

        Args:
            noise (Noise): The noise instance to attach.
            qubits (list[int] | None): Target qubit index or indices for per-qubit noise attachments.
            gate (type[Gate] | None): Target gate type for per-gate noise attachments.
            parameter (str | None): Target parameter name for perturbation attachments.

        Raises:
            ValueError: If the noise does not allow the inferred scope.
        """
        if parameter is not None:
            raise ValueError(f"{noise.__class__.__name__} cannot be applied to parameters.")
        if qubits is None and gate is None:
            self._check_scope_allowed(noise, AttachmentScope.GLOBAL)
            self.global_noise.append(noise)
            return
        if qubits is not None and gate is None:
            self._check_scope_allowed(noise, AttachmentScope.PER_QUBIT)
            for q in qubits:
                self.per_qubit_noise[q].append(noise)
            return
        if qubits is None and gate is not None:
            self._check_scope_allowed(noise, AttachmentScope.PER_GATE_TYPE)
            self.per_gate_noise[gate].append(noise)
            return
        if qubits is not None and gate is not None:
            self._check_scope_allowed(noise, AttachmentScope.PER_GATE_TYPE_PER_QUBIT)
            for q in qubits:
                self.per_gate_per_qubit_noise[gate, q].append(noise)

    def _add_parameter_perturbation(
        self,
        noise: ParameterPerturbation,
        *,
        gate: GateType | None = None,
        parameter: Parameter | None = None,
    ) -> None:
        """Attach a parameter perturbation to the model.

        Args:
            noise (ParameterPerturbation): The parameter perturbation instance to attach.
            gate (type[Gate] | None): Target gate type for per-gate perturbation attachments.
            parameter (str | None): Target parameter name for perturbation attachments.

        Raises:
            ValueError: If the perturbation does not allow the inferred scope.
        """
        if parameter is None:
            raise ValueError(f"{noise.__class__.__name__} requires a parameter name.")
        if gate is None:
            self._check_scope_allowed(noise, AttachmentScope.GLOBAL)
            self.global_perturbations[parameter].append(noise)
            return
        self._check_scope_allowed(noise, AttachmentScope.PER_GATE_TYPE)
        self.per_gate_perturbations[gate, parameter].append(noise)
