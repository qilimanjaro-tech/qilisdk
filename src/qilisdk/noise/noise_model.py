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
from typing import TypeVar, overload

from qilisdk.digital import Gate

from .noise import Noise
from .parameter_pertubation import ParameterPerturbation
from .protocols import AttachmentScope

Qubit = int
GateType = type[Gate]
Parameter = str

TNoise = TypeVar("TNoise", bound=Noise)
TParameterPerturbation = TypeVar("TParameterPerturbation", bound=ParameterPerturbation)


class NoiseModel:
    """Container for noise sources and parameter perturbations.

    Stores global, per-qubit, and per-gate noise along with parameter-level
    perturbations for later application in backends.
    """

    def __init__(self) -> None:
        # noises
        self.global_noise: list[Noise] = []
        self.per_qubit_noise: dict[Qubit, list[Noise]] = defaultdict(list)
        self.per_gate_noise: dict[GateType, list[Noise]] = defaultdict(list)

        # parameter pertubations
        self.global_pertubations: dict[Parameter, list[ParameterPerturbation]] = defaultdict(list)
        self.per_gate_pertubations: dict[tuple[GateType, Parameter], list[ParameterPerturbation]] = defaultdict(list)

        # -----------------------
    # Overloads (typing)
    # -----------------------

    # Noise: global/per_qubit/per_gate_type
    @overload
    def add(self, noise: TNoise) -> None: ...
    @overload
    def add(self, noise: TNoise, *, qubit: Qubit) -> None: ...
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
        qubit: Qubit | None = None,
        gate: GateType | None = None,
        parameter: Parameter | None = None,
    ) -> None:
        """Attach a noise source or parameter perturbation to the model.

        Args:
            noise (Noise | ParameterPerturbation): The noise or parameter perturbation instance to attach.
            qubit (int | None): Target qubit index for per-qubit noise attachments.
            gate (type[Gate] | None): Target gate type for per-gate noise or perturbation attachments.
            parameter (str | None): Target parameter name for perturbation attachments.

        Raises:
            ValueError: If the noise/perturbation does not allow the inferred scope.
        """
        if isinstance(noise, ParameterPerturbation):
            if parameter is None:
                raise ValueError(f"{noise.__class__.__name__} requires a parameter name.")
            if qubit is not None:
                raise ValueError(f"{noise.__class__.__name__} cannot be applied to specific qubits.")
            if gate is None:
                scope = AttachmentScope.GLOBAL
                if scope not in noise.allowed_scopes():
                    raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")
                self.global_pertubations[parameter].append(noise)
                return
            scope = AttachmentScope.PER_GATE_TYPE
            if scope not in noise.allowed_scopes():
                raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")
            self.per_gate_pertubations[gate, parameter].append(noise)
        else:
            if parameter is not None:
                raise ValueError(f"{noise.__class__.__name__} requires a parameter name.")
            if qubit is None and gate is None:
                scope = AttachmentScope.GLOBAL
                if scope not in noise.allowed_scopes():
                    raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")
                self.global_noise.append(noise)
                return
            if qubit is not None and gate is None:
                scope = AttachmentScope.PER_QUBIT
                if scope not in noise.allowed_scopes():
                    raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")
                self.per_qubit_noise[qubit].append(noise)
                return
            if qubit is None and gate is not None:
                scope = AttachmentScope.PER_GATE_TYPE
                if scope not in noise.allowed_scopes():
                    raise ValueError(f"{noise.__class__.__name__} cannot be added with scope '{scope.value}'.")
                self.per_gate_noise[gate].append(noise)
                return
