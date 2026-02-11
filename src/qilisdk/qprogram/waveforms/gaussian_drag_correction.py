# Copyright 2023 Qilimanjaro Quantum Tech
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

"""DragCorrection waveform."""

import numpy as np

from qilisdk.yaml import yaml

from .gaussian import Gaussian


@yaml.register_class
class GaussianDragCorrection(Gaussian):
    """Calculates the first order drag correction of the imaginary (Ey) channel of a drive pulse. See https://arxiv.org/abs/0901.0534 (10).

    So far only implemented for Gaussian pulses.

    Args:
        drag_coefficient (float): drag coefficient
        waveform (Waveform): waveform on which the drag transformation is calculated
    """

    def __init__(self, amplitude: float, duration: int, num_sigmas: float, drag_coefficient: float) -> None:
        super().__init__(amplitude=amplitude, duration=duration, num_sigmas=num_sigmas)
        self.drag_coefficient = drag_coefficient

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Returns the envelope corresponding to the drag correction.

        Args:
            resolution (int, optional): Pulse resolution. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.
        """
        sigma = self.duration / self.num_sigmas
        mu = self.duration / 2
        x = np.arange(self.duration / resolution) * resolution

        return (-1 * self.drag_coefficient * (x - mu) / sigma**2) * super().envelope(resolution=resolution)

    def get_duration(self) -> int:
        """Get the duration of the waveform.

        Returns:
            int: The duration of the waveform in ns.
        """
        return self.duration
