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

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .representations import KrausChannel, LindbladGenerator


class AttachmentScope(str, Enum):
    """Scope describing where a noise or perturbation can be attached.

    Attributes:
        GLOBAL: Applies to all operations or parameters.
        PER_QUBIT: Applies only to a specific qubit.
        PER_GATE_TYPE: Applies only to a specific gate type.
    """

    GLOBAL = "global"
    PER_QUBIT = "per_qubit"
    PER_GATE_TYPE = "per_gate_type"
    PER_GATE_TYPE_PER_QUBIT = "per_gate_type_per_qubit"


@runtime_checkable
class HasAllowedScopes(Protocol):
    """Protocol for types that declare their allowed attachment scopes."""

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """Return the attachment scopes supported by the type.

        Returns:
            The set of scopes where the type can be attached.
        """
        ...


@runtime_checkable
class SupportsTimeDerivedKraus(Protocol):
    """Protocol for types that produce duration-dependent Kraus channels."""

    def as_kraus_from_duration(self, *, duration: float) -> KrausChannel:
        """Return a Kraus channel derived for a specific duration.

        Args:
            duration (float): Duration over which the noise acts.

        Returns:
            The Kraus channel for the given duration.
        """
        ...


@runtime_checkable
class SupportsStaticKraus(Protocol):
    """Protocol for types that expose a fixed Kraus channel."""

    def as_kraus(self) -> KrausChannel:
        """Return the static Kraus channel for this noise.

        Returns:
            The Kraus channel representation.
        """
        ...


@runtime_checkable
class SupportsStaticLindblad(Protocol):
    """Protocol for types that expose a fixed Lindblad generator."""

    def as_lindblad(self) -> LindbladGenerator:
        """Return the Lindblad generator for this noise.

        Returns:
            The Lindblad generator representation.
        """
        ...


@runtime_checkable
class SupportsTimeDerivedLindblad(Protocol):
    """Protocol for types that expose a Lindblad generator derived from time."""

    def as_lindblad_from_duration(self, *, duration: float) -> LindbladGenerator:
        """Return the Lindblad generator for this noise.

        Args:
            duration (float): Duration over which the noise acts.

        Returns:
            The Lindblad generator representation.
        """
        ...
