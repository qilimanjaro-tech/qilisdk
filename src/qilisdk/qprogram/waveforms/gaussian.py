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

"""Gaussian waveform."""

import numpy as np

from qilisdk.yaml import yaml

from .waveform import Waveform


@yaml.register_class
class Gaussian(Waveform):
    r"""Gaussian waveform with peak at duration/2 and spanning for num_sigmas over the pulse duration.

    Standard centered Gaussian pulse shape, symmetrically spanning for ``num_sigmas`` over the pulse duration.

    The normal distribution's parameters :math:`\mu` (mean) and :math:`\sigma` (standard deviation) will be therefore
    defined by :math:`\mu =` ``duration`` :math:`/ 2` and :math:`\sigma =` ``duration`` :math:`/` ``num_sigmas``:

    .. math::

        Gaussian(x) = A * exp(-0.5 * (x - \mu)^2 / \sigma^2)

    You can think of it, as if an infinite expanding gaussian, is symmetrically cut in the given ``num_sigmas``, then that cut is
    expanded to occupy all the pulse duration, and then is shifted down so that it starts at 0.

    Examples:

        To get the envelope of a gaussian waveform, with ``num_sigmas`` equal to ``X``, you need to do:

        .. code-block:: python

            from qilisdk.qprogram.waveforms import Gaussian

            gaussian_envelope = Gaussian(amplitude=1.0, duration=50, num_sigmas=X).envelope()

        which for ``X`` being ``1``, ``4``, ``6`` or ``8``, look respectively like:

        .. image:: /classes_images/gaussians.png
            :width: 800
            :align: center

        This comes from the following steps, parting from a "full" gaussian (big ``num_sigma``) [blue]:
            1. you cut a full gaussian to the given ``num_sigmas``, for example ``1`` [small purple].
            2. in reality this cut gaussian, actually extends all the duration [big purple],
            3. you shift/distortion down the cut gaussian, so it starts at 0 height [red].

        .. image:: /classes_images/gaussian_explanation.png
            :width: 400
            :align: center

    Args:
        amplitude (float): Maximum amplitude of the pulse.
        duration (int): Duration of the pulse (ns).
        num_sigmas (float): Sigma number of the gaussian pulse shape. Defines the width of the gaussian pulse.
    """

    def __init__(self, amplitude: float, duration: int, num_sigmas: float) -> None:
        self.amplitude = amplitude
        self.duration = duration
        self.num_sigmas = num_sigmas

    def envelope(self, resolution: int = 1) -> np.ndarray:
        """Gaussian envelope centered with respect to the pulse.

        The gaussian is symmetrically cut in the given ``num_sigmas``, meaning that it starts and ends at that sigma width.

        And then to avoid introducing noise at time 0, the full gaussian is shifted down so that it starts at 0.

        Args:
            resolution (int, optional): Resolution of the pulse. Defaults to 1.

        Returns:
            np.ndarray: Height of the envelope for each time step.
        """
        sigma = self.duration / self.num_sigmas
        mu = self.duration / 2
        x = np.arange(self.duration / resolution) * resolution

        gaussian = self.amplitude * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
        norm = np.amax(np.real(gaussian))

        gaussian -= gaussian[0]
        corr_norm = np.amax(np.real(gaussian))

        return gaussian * norm / corr_norm if corr_norm != 0 else gaussian

    def get_duration(self) -> int:
        """Get the duration of the waveform.

        Returns:
            int: The duration of the waveform in ns.
        """
        return self.duration
