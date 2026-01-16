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
from .protocols import AttachmentScope, SupportsStaticKraus
from .utils import _check_probability


class PhaseFlip(PauliChannel, SupportsStaticKraus):
    """Single-qubit Pauli channel configured with an X-axis error.

    Applies an X error with the given probability and the identity otherwise.
    """

    def __init__(self, *, probability: float) -> None:
        """Args:
            probability (float): Probability of applying the Pauli-X error.

        Raises:
            ValueError: If probability is outside [0, 1].
        """
        self._probability = _check_probability(probability, "probability")
        super().__init__(pZ=probability)

    @property
    def probability(self) -> float:
        """Return the probability of applying the Pauli-X error.

        Returns:
            float: The error probability.
        """
        return self._probability

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        return frozenset({AttachmentScope.GLOBAL, AttachmentScope.PER_QUBIT, AttachmentScope.PER_GATE_TYPE})
