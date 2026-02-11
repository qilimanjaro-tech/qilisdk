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
from loguru import logger

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class SuddenNetZero(Waveform):
    """Sudden Net Zero waveform. https://arxiv.org/pdf/2008.07411

    Args:
        duration (int): Duration of the pulse (ns).
        amplitude (float): Maximum amplitude of the pulse.
        b (float): Instant stops height when going from the rectangular half-duration to `height = 0`. Coefficient applied to the amplitude.
        t_phi (int): Time at `height = 0`, in the middle of the positive and negative rectangular pulses.
    """

    def __init__(self, amplitude: float, duration: int, b: float, t_phi: int) -> None:
        self.amplitude = amplitude
        self.duration = duration
        self.b = b
        self.t_phi = t_phi

    def envelope(self, resolution: float = 1.0) -> np.ndarray:
        """SuddenNetZero envelope.

        Args:
            resolution (int, optional): Resolution of the pulse. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.
        """
        # calculate the halfpulse duration
        half_pulse_t = (self.duration - 2 - self.t_phi) / 2
        half_pulse_t = int(half_pulse_t / resolution)

        envelope = np.zeros(round(self.duration / resolution))
        # raise warning if we are rounding
        if (self.duration / resolution) % 1 != 0 or (half_pulse_t / resolution) % 1 != 0:
            logger.warning(
                "Envelope length rounded to nearest value %d from division full_snz_duration (%s) / resolution (%s) = %s",
                len(envelope),
                self.duration,
                resolution,
                self.duration / resolution,
            )

        envelope[:half_pulse_t] = self.amplitude * np.ones(half_pulse_t)  # positive square halfpulse
        envelope[half_pulse_t] = self.b * self.amplitude  # impulse b
        envelope[half_pulse_t + 1 + self.t_phi] = -self.b * self.amplitude  # impulse -b
        envelope[half_pulse_t + 2 + self.t_phi :] = -self.amplitude * np.ones(half_pulse_t)  # negative square halfpulse

        return envelope

    def get_duration(self) -> int:
        """Get the duration of the waveform.

        Returns:
            int: The duration of the waveform in ns.
        """
        return self.duration
