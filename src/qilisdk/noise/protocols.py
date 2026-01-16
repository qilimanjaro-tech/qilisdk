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
    GLOBAL = "global"
    PER_QUBIT = "per_qubit"
    PER_GATE_TYPE = "per_gate_type"


@runtime_checkable
class HasAllowedScopes(Protocol):
    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]: ...


@runtime_checkable
class SupportsTimeDerivedKraus(Protocol):
    def as_kraus(self, *, duration: float) -> KrausChannel: ...


@runtime_checkable
class SupportsStaticKraus(Protocol):
    def as_kraus(self) -> KrausChannel: ...


@runtime_checkable
class SupportsLindblad(Protocol):
    def as_lindblad(self) -> LindbladGenerator: ...
