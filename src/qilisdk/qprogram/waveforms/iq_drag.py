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

"""Waveform protocol class."""

from qilisdk.yaml import yaml

from .gaussian import Gaussian
from .gaussian_drag_correction import GaussianDragCorrection
from .iq_waveform import IQWaveform
from .waveform import Waveform


@yaml.register_class
class IQDrag(IQWaveform):
    def __init__(self, amplitude: float, duration: int, num_sigmas: float, drag_coefficient: float) -> None:
        self.amplitude = amplitude
        self.duration = duration
        self.num_sigmas = num_sigmas
        self.drag_coefficient = drag_coefficient

    def get_I(self) -> Waveform:
        return Gaussian(amplitude=self.amplitude, duration=self.duration, num_sigmas=self.num_sigmas)

    def get_Q(self) -> Waveform:
        return GaussianDragCorrection(
            amplitude=self.amplitude,
            duration=self.duration,
            num_sigmas=self.num_sigmas,
            drag_coefficient=self.drag_coefficient,
        )

    def get_duration(self) -> int:
        return self.duration
