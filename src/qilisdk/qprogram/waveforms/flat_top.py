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

"""Square waveform."""

import numpy as np
from scipy.special import erf

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class FlatTop(Waveform):
    """Smoothed square pulse with error function on the edges rise waveform.

    Args:
        duration (int): Duration of the pulse (ns).
        amplitude (float): Maximum amplitude of the pulse.
        smooth_duration (float, optional): duration of the smoothing component in ns.
        buffer (float, optional): Buffer of the waveform. Defaults to 0.
    """

    def __init__(self, amplitude: float, duration: int, smooth_duration: int, buffer: int = 0) -> None:
        super().__init__()
        self.amplitude = amplitude
        self.duration = duration
        self.smooth_duration = smooth_duration
        self.buffer = buffer

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Smoothed square pulse with error function on the edges rise envelope.

        Args:
            resolution (int, optional): Resolution of the pulse. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.
        """
        x = np.arange(-self.duration / 2, self.duration / 2 + 1, resolution)
        A = self.amplitude
        sigma = self.smooth_duration
        buf = self.buffer
        dur = self.duration
        return (
            0.5 * A * np.real((erf(4 * (x + (dur / 2 - buf)) / sigma - 2) - erf(4 * (x - (dur / 2 - buf)) / sigma + 2)))
        )

    def get_duration(self) -> int:
        """Get the duration of the waveform.

        Returns:
            int: The duration of the waveform in ns.
        """
        return self.duration
