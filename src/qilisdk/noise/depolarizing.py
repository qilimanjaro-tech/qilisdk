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

from .pauli_channel import PauliChannel
from .utils import _check_probability


class Depolarizing(PauliChannel):
    """Single-qubit depolarizing noise channel.

    This channel mixes the state with the maximally mixed state by
    configuring pX = pY = pZ = p / 3, which implies pI = 1 - p.
    """

    def __init__(self, *, p: float) -> None:
        """Args:
            p (float): Depolarizing probability in the range [0, 1].

        Raises:
            ValueError: If p is outside [0, 1].
        """
        self._p = _check_probability(p, "p")
        super().__init__(pX=self._p / 3.0, pY=self._p / 3.0, pZ=self._p / 3.0)
