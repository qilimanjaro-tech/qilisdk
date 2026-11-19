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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qilisdk.functionals.functional import Functional

if TYPE_CHECKING:
    from qilisdk.backends.backend import Backend
    from qilisdk.core.result import Result
    from qilisdk.readout import Readout


class Algorithm(Functional, ABC):
    """Base class for functionals that define their own execution on top of a backend.

    Unlike the primitive functionals in :mod:`qilisdk.functionals`, which each backend simulates natively, an
    algorithm builds its result out of the operations a backend already provides, so any backend can run it.
    """

    @abstractmethod
    def _execute(self, backend: Backend, readout: Readout) -> Result:
        """Run the algorithm on ``backend``.

        Called by :meth:`~qilisdk.backends.backend.Backend.execute` once the readout has been validated.

        Args:
            backend (Backend): The backend used for any primitive functional the algorithm needs to execute.
            readout (Readout): The readout specification requested by the user.

        Returns:
            Result: The result of the algorithm.
        """
