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

from abc import ABC


class Result(ABC):
    """Marker base class for results produced by QiliSDK workflows."""

    @property
    def execution_time(self) -> float | None:
        """Wall-clock time, in seconds, spent executing the workflow that produced this result.

        Populated by :meth:`~qilisdk.backends.Backend.execute` with the time taken by the
        backend to run the simulation. ``None`` when the result was not produced through
        ``execute`` (e.g. constructed manually or deserialised from an older result).
        """
        return getattr(self, "_execution_time", None)

    @execution_time.setter
    def execution_time(self, value: float | None) -> None:
        self._execution_time = value
