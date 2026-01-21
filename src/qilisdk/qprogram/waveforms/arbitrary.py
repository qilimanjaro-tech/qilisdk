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

"""Arbitrary waveform."""

import numpy as np

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class Arbitrary(Waveform):
    """Arbitrary waveform. Creates a waveform with the passed envelope.

    Args:
        samples(np.ndarray): Passed envelope to base the waveform on.

    Examples:
        If you want to create a waveform with an envelope given by:

        .. code-block:: python

            import numpy as np

            samples = np.ones(50)

        You would just need to do:

        .. code-block:: python

            from qilisdk.qprogram.waveforms import Arbitrary

            arbitrary_envelope = Arbitrary(samples=samples)
    """

    def __init__(self, samples: np.ndarray) -> None:
        super().__init__()
        self.samples = samples

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Returns the envelope corresponding to the arbitrary waveform.

        Args:
            resolution (int, optional): Pulse resolution. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.

        Raises:
            ValueError: If resolution is too high compared to the waveform duration.
        """
        if resolution == 1:
            return self.samples

        # Calculate the downsampled length
        downsampled_length = len(self.samples) // resolution

        # Check for valid resolution
        if downsampled_length < 1:
            raise ValueError("Resolution is too high compared to the waveform duration.")

        # Reshape the waveform into blocks and compute mean for each block
        reshaped_samples = self.samples[: downsampled_length * resolution].reshape(-1, resolution)
        downsampled = reshaped_samples.mean(axis=1)

        # Normalize the downsampled waveform
        if downsampled.max() != downsampled.min():  # Avoid division by zero
            normalized = (downsampled - downsampled.min()) / (downsampled.max() - downsampled.min())
            normalized = normalized * (self.samples.max() - self.samples.min()) + self.samples.min()
        else:
            # If all values are the same, just return a constant array
            normalized = np.full_like(downsampled, self.samples.min())

        return normalized
