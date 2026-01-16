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

from qilisdk.noise import (
    AmplitudeDamping,
    Dephasing,
    PauliChannel,
    SupportsLindblad,
    SupportsStaticKraus,
    SupportsTimeDerivedKraus,
)
from qilisdk.noise.protocols import AttachmentScope


def test_attachment_scope_values():
    assert AttachmentScope.GLOBAL.value == "global"
    assert AttachmentScope.PER_QUBIT.value == "per_qubit"
    assert AttachmentScope.PER_GATE_TYPE.value == "per_gate_type"


def test_protocols_runtime_checks():
    assert isinstance(PauliChannel(), SupportsStaticKraus)
    assert isinstance(Dephasing(Tphi=1.0), SupportsTimeDerivedKraus)
    assert isinstance(Dephasing(Tphi=1.0), SupportsLindblad)
    assert isinstance(AmplitudeDamping(T1=1.0), SupportsTimeDerivedKraus)
    assert isinstance(AmplitudeDamping(T1=1.0), SupportsLindblad)
