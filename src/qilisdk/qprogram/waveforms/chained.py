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

"""Chained waveform."""

import numpy as np

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class Chained(Waveform):
    """Represents a composite waveform consisting of multiple waveforms played in a chained sequence.

    This class allows combining multiple waveforms (such as Ramp, Square, etc.) into a single waveform by playing them one after the other.

    Args:
        waveforms (list[Waveform]): A list of individual waveform objects that will be chained together in sequence.

    Examples:
        The following example creates a Chained waveform composed of a Ramp, a Square, and another Ramp.

        .. code-block:: python

            from qilisdk.qprogram import Chained, Ramp, Square

            # Create the waveform
            chained_wf = Chained(waveforms=[
                Ramp(from_amplitude=0.0, to_amplitude=1.0, duration=50)
                Square(amplitude=1.0, duration=100)
                Ramp(from_amplitude=1.0, to_amplitude=0.0, duration=50)
            ])

            # Get waveform's envelope
            envelope = chained_wf.envelope()

            # Get waveform's duration
            envelope = chained_wf.get_duration()
    """

    def __init__(self, waveforms: list[Waveform]) -> None:
        """
        Initialize the Chained waveform.

        Args:
            waveforms (list[Waveform]): A list of waveform instances to be chained together in sequence.
        """
        super().__init__()
        self.waveforms = waveforms

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Retrieve the envelope of the chained waveform.

        The envelope is obtained by concatenating the envelopes of each individual waveform in the sequence, with the specified resolution.

        Args:
            resolution (int, optional): The resolution of the waveform (number of time steps per unit of duration). Defaults to 1.

        Returns:
            np.ndarray: An array representing the amplitude of the waveform at each time step.
        """
        return np.concatenate([waveform.envelope(resolution=resolution) for waveform in self.waveforms])

    def get_duration(self) -> int:
        """Retrieve the total duration of the chained waveform.

        The total duration is calculated as the sum of the durations of each individual waveform in the sequence.

        Returns:
            int: The total duration of the waveform in nanoseconds.
        """
        return sum(waveform.get_duration() for waveform in self.waveforms)
