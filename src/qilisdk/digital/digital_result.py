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
from pprint import pformat

from qilisdk.common import Results


class DigitalResult(Results):
    def __init__(self, nshots: int, samples: dict[str, int]) -> None:
        self._samples = samples
        self._nshots = nshots

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def samples(self) -> dict[str, int]:
        return self._samples

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(\n  nshots={self.nshots},\n  samples={pformat(self.samples)}\n)"


# class SimulationDigitalResults(DigitalResults):
#     def __init__(
#         self,
#         state: np.ndarray,
#         probabilities: np.ndarray,
#         samples: np.ndarray,
#         frequencies: dict[str, int],
#         nshots: int,
#     ) -> None:
#         super().__init__(samples=samples, frequencies=frequencies, nshots=nshots)
#         self._state = state
#         self._probabilities = probabilities

#     @property
#     def state(self) -> np.ndarray:
#         return self._state

#     @property
#     def probabilities(self) -> np.ndarray:
#         return self._probabilities

#     def __repr__(self) -> str:
#         class_name = self.__class__.__name__
#         return (
#             f"{class_name}(\n"
#             f"  nshots={self.nshots},\n"
#             f"  frequencies={pformat(self.frequencies)},\n"
#             f"  samples={pformat(self.samples)},\n"
#             f"  state={pformat(self.state)},\n"
#             f"  probabilities={pformat(self.probabilities)}\n"
#             ")"
#         )
