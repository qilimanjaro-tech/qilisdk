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


from qilisdk.qprogram.waveforms import IQWaveform, Waveform
from qilisdk.yaml import yaml

from .operation import Operation


@yaml.register_class
class Play(Operation):
    def __init__(self, bus: str, waveform: Waveform | IQWaveform) -> None:
        super().__init__()
        self.bus: str = bus
        self.waveform: Waveform | IQWaveform = waveform

    def get_waveforms(self) -> tuple[Waveform, Waveform | None]:
        """Get the waveforms.

        Returns:
            tuple[Waveform, Waveform | None]: The waveforms as tuple. The second waveform can be None.
        """
        wf_I: Waveform = self.waveform.get_I() if isinstance(self.waveform, IQWaveform) else self.waveform
        wf_Q: Waveform | None = self.waveform.get_Q() if isinstance(self.waveform, IQWaveform) else None
        return wf_I, wf_Q
