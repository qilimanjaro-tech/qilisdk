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
import numpy as np

from qilisdk.common import Results


class DigitalResults(Results):
    def __init__(self, samples: np.ndarray, frequencies: dict[str, int], nshots: int) -> None:
        self._samples = samples
        self._frequencies = frequencies
        self._nshots = nshots

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def frequencies(self) -> dict[str, int]:
        return self._frequencies

    @property
    def samples(self) -> np.ndarray:
        return self._samples


class SimulationDigitalResults(DigitalResults):
    def __init__(
        self,
        state: np.ndarray,
        probabilities: np.ndarray,
        samples: np.ndarray,
        frequencies: dict[str, int],
        nshots: int,
    ) -> None:
        super().__init__(samples=samples, frequencies=frequencies, nshots=nshots)
        self._state = state
        self._probabilities = probabilities

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities
