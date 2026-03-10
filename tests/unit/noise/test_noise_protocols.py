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
    SupportsStaticKraus,
    SupportsStaticLindblad,
    SupportsTimeDerivedKraus,
    SupportsTimeDerivedLindblad,
)
from qilisdk.noise.bit_flip import BitFlip
from qilisdk.noise.depolarizing import Depolarizing
from qilisdk.noise.protocols import AttachmentScope
from qilisdk.noise.representations import KrausChannel, LindbladGenerator


def test_attachment_scope_values():
    assert AttachmentScope.GLOBAL.value == "global"
    assert AttachmentScope.PER_QUBIT.value == "per_qubit"
    assert AttachmentScope.PER_GATE_TYPE.value == "per_gate_type"
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT.value == "per_gate_type_per_qubit"


def test_protocols_runtime_checks():
    assert isinstance(LindbladGenerator([]), SupportsStaticLindblad)
    assert not isinstance(LindbladGenerator([]), SupportsTimeDerivedLindblad)
    assert not isinstance(LindbladGenerator([]), SupportsStaticKraus)
    assert not isinstance(LindbladGenerator([]), SupportsTimeDerivedKraus)

    assert isinstance(KrausChannel([]), SupportsStaticKraus)
    assert not isinstance(KrausChannel([]), SupportsTimeDerivedKraus)
    assert not isinstance(KrausChannel([]), SupportsStaticLindblad)
    assert not isinstance(KrausChannel([]), SupportsTimeDerivedLindblad)

    assert isinstance(PauliChannel(), SupportsStaticKraus)
    assert isinstance(PauliChannel(), SupportsTimeDerivedLindblad)
    assert not isinstance(PauliChannel(), SupportsStaticLindblad)
    assert not isinstance(PauliChannel(), SupportsTimeDerivedKraus)

    assert isinstance(Dephasing(t_phi=1.0), SupportsTimeDerivedKraus)
    assert isinstance(Dephasing(t_phi=1.0), SupportsStaticLindblad)
    assert not isinstance(Dephasing(t_phi=1.0), SupportsStaticKraus)
    assert not isinstance(Dephasing(t_phi=1.0), SupportsTimeDerivedLindblad)

    assert isinstance(AmplitudeDamping(t1=1.0), SupportsTimeDerivedKraus)
    assert isinstance(AmplitudeDamping(t1=1.0), SupportsStaticLindblad)
    assert not isinstance(AmplitudeDamping(t1=1.0), SupportsStaticKraus)
    assert not isinstance(AmplitudeDamping(t1=1.0), SupportsTimeDerivedLindblad)

    assert isinstance(Depolarizing(probability=0.1), SupportsStaticKraus)
    assert isinstance(Depolarizing(probability=0.1), SupportsTimeDerivedLindblad)
    assert not isinstance(Depolarizing(probability=0.1), SupportsStaticLindblad)
    assert not isinstance(Depolarizing(probability=0.1), SupportsTimeDerivedKraus)

    assert isinstance(BitFlip(probability=0.1), SupportsStaticKraus)
    assert isinstance(BitFlip(probability=0.1), SupportsTimeDerivedLindblad)
    assert not isinstance(BitFlip(probability=0.1), SupportsStaticLindblad)
    assert not isinstance(BitFlip(probability=0.1), SupportsTimeDerivedKraus)
