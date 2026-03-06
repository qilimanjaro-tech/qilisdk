# Copyright 2023 Qilimanjaro Quantum Tech
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qilisdk.digital import Circuit


@dataclass
class TranspilationContext:
    """Shared, mutable state for passes."""

    # Artifacts produced/consumed by passes:
    initial_layout: list[int] = field(default_factory=list)  # logical -> physical (after layout)
    final_layout: dict[int, int] = field(default_factory=dict)  # logical -> physical (after router)
    metrics: dict[str, Any] = field(default_factory=dict)
    circuits: dict[str, Circuit] = field(default_factory=dict)
