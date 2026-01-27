# Copyright 2025 Qilimanjaro Quantum Tech
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


from qilisdk.core import Parameter
from qilisdk.digital import RZ, Circuit
from qilisdk.functionals.sampling import Sampling


def test_sampling():
    circuit = Circuit(2)
    param = Parameter("theta", 0.5)
    circuit.add(RZ(0, phi=param))
    nshots = 1024
    sampling = Sampling(circuit=circuit, nshots=nshots)

    sampling.set_parameters({"theta": 0.3})
    sampling.set_parameter_values([1.0])
    sampling.set_parameter_bounds({"theta": (0.0, 2.0)})

    assert sampling.circuit == circuit
    assert sampling.nshots == nshots
    assert sampling.nparameters == 1
    assert sampling.get_parameters() == {"theta": 1.0}
    assert sampling.get_parameter_names() == ["theta"]
    assert sampling.get_parameter_bounds() == {"theta": (0.0, 2.0)}
    assert sampling.get_parameter_values() == [1.0]
    assert sampling.get_constraints() == []
