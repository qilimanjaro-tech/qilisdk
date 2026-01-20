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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qilisdk.digital import Gate


class NoiseConfig:
    """Configuration options for noise models."""

    def __init__(self) -> None:
        """
        Initialize a NoiseConfig with default settings.
        """
        self._gate_times: dict[type[Gate], float] = {}

    @property
    def gate_times(self) -> dict[type[Gate], float]:
        """Return the gate times configuration.
        If a gate type is not specified, it is assumed to have an execution time of 1 (unitless).

        Returns:
            dict[type[Gate], float]: A dictionary mapping gate types to their execution times.
        """
        return self._gate_times

    def set_gate_time(self, gate_type: type[Gate], time: float) -> None:
        """Update the execution time for a specific gate type.

        Args:
            gate_type (type[Gate]): The type of the gate to update.
            time (float): The new execution time for the gate.

        Raises:
            ValueError: If the provided time is not positive.
        """
        if time <= 0:
            raise ValueError("Execution time must be positive.")
        self._gate_times[gate_type] = time
