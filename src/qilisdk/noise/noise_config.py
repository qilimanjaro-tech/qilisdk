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
        self._default_gate_time: float = 1.0

    def get_gate_time(self, gate_type: type[Gate]) -> float:
        """Get the execution time for a specific gate type.

        Args:
            gate_type (type[Gate]): The type of the gate to query.

        Returns:
            float: The execution time for the gate. If not specified, returns the default gate time.
        """
        return self._gate_times.get(gate_type, self._default_gate_time)

    @property
    def default_gate_time(self) -> float:
        """Return the default execution time for gates.

        Returns:
            The default gate time.
        """
        return self._default_gate_time

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

    def set_default_gate_time(self, time: float) -> None:
        """Set the default execution time for gates not explicitly specified.

        Args:
            time (float): The new default execution time.

        Raises:
            ValueError: If the provided time is not positive.
        """
        if time <= 0:
            raise ValueError("Default execution time must be positive.")
        self._default_gate_time = time
