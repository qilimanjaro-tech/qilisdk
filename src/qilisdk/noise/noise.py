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

from .noise_abc import NoiseABC
from .protocols import AttachmentScope, HasAllowedScopes


class Noise(NoiseABC, HasAllowedScopes):
    """Base class for state noise sources that can be attached to a model."""

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """Return the attachment scopes supported by this noise type.

        Returns:
            The set of scopes where this noise can be attached.
        """
        return frozenset({AttachmentScope.GLOBAL, AttachmentScope.PER_QUBIT})
