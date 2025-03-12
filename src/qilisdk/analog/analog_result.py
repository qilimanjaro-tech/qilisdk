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

import numpy as np

from qilisdk.common import Results


class AnalogResults(Results):
    def __init__(
        self, final_expected_values: np.ndarray | None = None, expected_values: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self._final_expected_values = final_expected_values if final_expected_values is not None else np.array([])
        self._expected_values = expected_values if expected_values is not None else np.array([])

    @property
    def final_expected_values(self) -> np.ndarray:
        return self._final_expected_values

    @property
    def expected_values(self) -> np.ndarray:
        return self._expected_values

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n"
            f"  final_expected_values={pformat(self.final_expected_values)},\n"
            f"  expected_values={pformat(self.expected_values)}\n"
            ")"
        )
