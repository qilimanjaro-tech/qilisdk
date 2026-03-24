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

from qilisdk.digital import Circuit
from qilisdk.utils.openqasm3 import from_qasm3


def test_quantum_reg():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qreg q[2];
        """)

    expected = Circuit(2)
    assert c == expected


def test_quantum_reg_init_with_classical_constant():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int N = 2;
            qreg q[N];
        """)

    expected = Circuit(2)
    assert c == expected


def test_classical_expression_init():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int N = 2 * (1 + 1);
            qreg q[N];
        """)

    expected = Circuit(4)
    assert c == expected
