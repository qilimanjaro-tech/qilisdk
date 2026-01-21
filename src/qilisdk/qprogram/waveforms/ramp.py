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

"""Ramp waveform."""

import numpy as np

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class Ramp(Waveform):
    """Ramp waveform. The waveform's amplitude changes linearly from ``from_amplitude`` to ``to_amplitude`` in ``duration`` nanoseconds.

    Args:
        from_amplitude (float): The amplitude of the waveform at t=0.
        to_amplitude (float): The amplitude of the waveform at t=duration.
        duration (int): Duration of the waveform (ns).

    Examples:
        In the following example the waveform's amplitude will ramp from 0.0 to 1.0 in 50 ns.

        .. code-block:: python

            import qililab as ql

            # Create the waveform
            ramp_wf = Ramp(from_amplitude=0.0, to_amplitude=1.0, duration=50)

            # Get waveform's envelope
            envelope = ramp_wf.envelope()
    """

    def __init__(self, from_amplitude: float, to_amplitude: float, duration: int) -> None:
        self.from_amplitude = from_amplitude
        self.to_amplitude = to_amplitude
        self.duration = duration

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Constant amplitude envelope.

        Args:
            resolution (int, optional): Resolution of the pulse. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.
        """
        return np.linspace(self.from_amplitude, self.to_amplitude, round(self.duration / resolution))

    def get_duration(self) -> int:
        """Get the duration of the waveform.

        Returns:
            int: The duration of the waveform in ns.
        """
        return self.duration
