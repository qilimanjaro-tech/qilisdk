# Copyright 2026 Qilimanjaro Quantum Tech
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

import pytest

from qilisdk.digital.gates import X, Y
from qilisdk.noise import NoiseConfig, NoiseModel


def test_noise_config():
    config = NoiseConfig()
    config.set_gate_time(X, 2.0)
    config.set_gate_time(Y, 3.0)

    assert config.gate_times[X] == 2.0
    assert config.gate_times[Y] == 3.0
    assert len(config.gate_times) == 2

    nm = NoiseModel(noise_config=config)
    assert nm.noise_config == config

    with pytest.raises(ValueError, match="Execution time must be positive"):
        config.set_gate_time(X, -1.0)
