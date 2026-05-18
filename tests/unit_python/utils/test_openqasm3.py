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


import math
import pathlib
import tempfile

import pytest

from qilisdk.core import Parameter
from qilisdk.digital import CNOT, CZ, RX, RY, RZ, U1, U2, U3, Adjoint, Circuit, Controlled, H, M, S, T, X, Y, Z
from qilisdk.utils.openqasm import from_qasm3, from_qasm3_file, to_qasm3, to_qasm3_file


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


def test_classical_expression_assignment():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            int N;
            N = 2 * (1 + 1);
            qreg q[N];
        """)
    expected = Circuit(4)
    assert c == expected


def test_rx():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(3.2) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.2))
    assert c == expected


def test_ry():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            ry(1.5) q[0];
        """)
    expected = Circuit(1)
    expected.add(RY(0, theta=1.5))
    assert c == expected


def test_rz():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rz(0.5) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=0.5))
    assert c == expected


def test_x():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            x q[0];
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_y():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            y q[0];
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_z():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            z q[0];
        """)
    expected = Circuit(1)
    expected.add(Z(0))
    assert c == expected


def test_h():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            h q[0];
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_s():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            s q[0];
        """)
    expected = Circuit(1)
    expected.add(S(0))
    assert c == expected


def test_t():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            t q[0];
        """)
    expected = Circuit(1)
    expected.add(T(0))
    assert c == expected


def test_cz():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            cz q[0], q[1];
        """)
    expected = Circuit(2)
    expected.add(CZ(0, 1))
    assert c == expected


def test_cnot():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            cx q[0], q[1];
        """)
    expected = Circuit(2)
    expected.add(CNOT(0, 1))
    assert c == expected


def test_u1():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            u1(0.5) q[0];
        """)
    expected = Circuit(1)
    expected.add(U1(0, phi=0.5))
    assert c == expected


def test_u2():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            u2(0.5, 1.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(U2(0, phi=0.5, gamma=1.0))
    assert c == expected


def test_u3():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            u3(0.5, 1.0, 1.5) q[0];
        """)
    expected = Circuit(1)
    expected.add(U3(0, theta=0.5, phi=1.0, gamma=1.5))
    assert c == expected


def test_x_on_multiple_qubits():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[3] q;
            x q;
        """)
    expected = Circuit(3)
    expected.add(X(0))
    expected.add(X(1))
    expected.add(X(2))
    assert c == expected


def test_x_on_multiple_qubits_partial_register():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[3] q;
            x q[0:2];
        """)
    expected = Circuit(3)
    expected.add(X(0))
    expected.add(X(1))
    assert c == expected


def test_x_on_multiple_qubits_partial_register_with_step():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            x q[0:2:5];
        """)
    expected = Circuit(5)
    expected.add(X(0))
    expected.add(X(2))
    expected.add(X(4))
    assert c == expected


def test_x_on_multiple_qubits_specific_qubits_in_register():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            x q[{0, 2, 4}];
        """)
    expected = Circuit(5)
    expected.add(X(0))
    expected.add(X(2))
    expected.add(X(4))
    assert c == expected


def test_measure():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            measure q[0];
        """)
    expected = Circuit(1)
    expected.add(M(0))
    assert c == expected


def test_measure_to_classical_throws_error():
    with pytest.raises(ValueError, match="statements with targets"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            creg c[1];
            measure q[0] -> c[0];
        """)


def test_int_def():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int N = 5;
            qreg q[N];
        """)
    expected = Circuit(5)
    assert c == expected


def test_float_def():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const float PI = 3.14159;
            qubit[1] q;
            rz(PI) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=math.pi))
    assert c == expected


def test_bool_def():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bool FLAG = true;
            qubit[1] q;
            if (FLAG) x q[0];
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_complex_def():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const complex PHASE = 1.0 + 0.5 im;
            qubit[1] q;
            rz(real(PHASE)) q[0];
            rz(imag(PHASE)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=1.0))
    expected.add(RZ(0, phi=0.5))
    assert c == expected


def test_bit_def():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bit FLAG = 1;
            qubit[1] q;
            if (FLAG) x q[0];
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_aliasing():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            let myqubits = q[0:2];
            x myqubits;
        """)
    expected = Circuit(5)
    expected.add(X(0))
    expected.add(X(1))
    assert c == expected


def test_custom_gate():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            gate myrx(theta) a {
                rx(theta) a;
            }
            myrx(1.5) q;
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.5))
    assert c == expected


def test_custom_gate_indexing():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            gate myrx(theta) a {
                rx(theta) a;
            }
            myrx(1.5) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.5))
    assert c == expected


def test_subroutine_int_return_type():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def add_one(int a) -> int {
                return a + 1;
            }
            qubit[2] q;
            x q[add_one(0)];
        """)
    expected = Circuit(2)
    expected.add(X(1))
    assert c == expected


def test_subroutine_early_return():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def add_one(int a) -> int {
                return;
                return a + 1;
            }
            qubit[2] q;
            x q[add_one(0)];
        """)
    expected = Circuit(2)
    expected.add(X(0))
    assert c == expected


def test_subroutine_float_return_type():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def add_half(float a) -> float {
                return a + 0.5;
            }
            qubit[1] q;
            rz(add_half(1.0)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=1.5))
    assert c == expected


def test_subroutine_bool_return_type():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def is_positive(int a) -> bool {
                return a > 0;
            }
            def is_negative(int a) -> bool {
                return a < 0;
            }
            qubit[1] q;
            if (is_positive(1)) x q[0];
            if (is_negative(1)) x q[0];
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_comparisons():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A < B) {
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_logical_operations():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bool FLAG1 = true;
            const bool FLAG2 = false;
            qubit[1] q;
            if (FLAG1 && !FLAG2) {
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_else_if():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A > B) {
                x q[0];
            } else if (A < B) {
                y q[0];
            } else {
                z q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_nested_if():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            const int C = 4;
            qubit[1] q;
            if (A < B) {
                if (B < C) {
                    x q[0];
                } else {
                    y q[0];
                }
            } else {
                z q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_switch_case():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            qubit[1] q;
            switch (A) {
                case 1 {
                    x q[0];
                }
                case 2 {
                    y q[0];
                }
                case 3 {
                    z q[0];
                }
                default {
                    h q[0];
                }
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_switch_default_case():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 4;
            qubit[1] q;
            switch (A) {
                case 1 {
                    x q[0];
                }
                case 2 {
                    y q[0];
                }
                case 3 {
                    z q[0];
                }
                default {
                    h q[0];
                }
            }
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_pi():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rz(pi) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=math.pi))
    assert c == expected


def test_sin():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(sin(pi / 2)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_cos():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(cos(pi)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=-1.0))
    assert c == expected


def test_tan():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(tan(pi / 4)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_arcsin():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(arcsin(1.0)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.asin(1.0)))
    assert c == expected


def test_arccos():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(arccos(0.0)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.acos(0.0)))
    assert c == expected


def test_arctan():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(arctan(1.0)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.atan(1.0)))
    assert c == expected


def test_rotl():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(rotl(1, 2)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=4))
    assert c == expected


def test_rotr():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(rotr(4, 2)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1))
    assert c == expected


def test_mod_function():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(mod(5, 2)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_floor():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(floor(1.5)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_ceiling():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(ceiling(1.5)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2.0))
    assert c == expected


def test_log():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(log(euler)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_exp():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(exp(1)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.exp(1)))
    assert c == expected


def test_mod():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(5 % 2) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_sqrt():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(sqrt(4)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2.0))
    assert c == expected


def test_euler():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(euler) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.e))
    assert c == expected


def test_tau():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(tau) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2 * math.pi))
    assert c == expected


def test_subroutine_without_return_type():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def apply_hadamard(qubit q) {
                h q;
            }
            qubit[1] q;
            apply_hadamard(q[0]);
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_subroutine_with_empty_return():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def apply_hadamard(qubit q) {
                h q;
                return;
                x q; // This should not be executed
            }
            qubit[1] q;
            apply_hadamard(q[0]);
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_unit_ns():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(1ns) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1e-9))
    assert c == expected


def test_unit_ms():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(1ms) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1e-3))
    assert c == expected


def test_unit_us():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(1us) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1e-6))
    assert c == expected


def test_unit_s():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(1s) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_bad_subroutine_raises_error():
    with pytest.raises(ValueError, match="Unsupported"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bad_subroutine();
        """)


def test_variables_that_doesnt_exist_raises_error():
    with pytest.raises(ValueError, match="Undefined variable"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(UNDEFINED_VARIABLE) q[0];
        """)


def test_operator_addition():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(1.0 + 2.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.0))
    assert c == expected


def test_operator_subtraction():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(5.0 - 2.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.0))
    assert c == expected


def test_operator_power():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(2.0 ** 3.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=8.0))
    assert c == expected


def test_operator_multiplication():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(2.0 * 3.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=6.0))
    assert c == expected


def test_operator_division():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            rx(6.0 / 2.0) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.0))
    assert c == expected


def test_operator_bitwise_and():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bit[4] a = "1010";
            bit[4] b = "0101";
            qubit[1] q;
            rx(a & b) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=0.0))
    assert c == expected


def test_operator_bitwise_or():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bit[4] a = "1010";
            bit[4] b = "0101";
            qubit[1] q;
            rx(a | b) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=15.0))
    assert c == expected


def test_operator_bitwise_left_shift():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bit[4] a = "1010";
            qubit[1] q;
            rx(a << 1) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=20.0))
    assert c == expected


def test_operator_bitwise_left_shift_overflow():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bit[4] a = "1010";
            bit[4] result = a << 1;
            qubit[1] q;
            rx(result) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=4.0))
    assert c == expected


def test_operator_bitwise_right_shift():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            bit[4] a = "1010";
            qubit[1] q;
            rx(a >> 1) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=5.0))
    assert c == expected


def test_operator_equality():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A == B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_operator_inequality():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A != B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_operator_less_than():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A < B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_operator_less_than_or_equal():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A <= B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_operator_greater_than():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A > B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_operator_greater_than_or_equal():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const int A = 2;
            const int B = 3;
            qubit[1] q;
            if (A >= B) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_operator_logical_and():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bool FLAG1 = true;
            const bool FLAG2 = false;
            qubit[1] q;
            if (FLAG1 && FLAG2) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_operator_logical_or():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bool FLAG1 = true;
            const bool FLAG2 = false;
            qubit[1] q;
            if (FLAG1 || FLAG2) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_operator_logical_not():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            const bool FLAG = true;
            qubit[1] q;
            if (!FLAG) {
                x q[0];
            } else {
                y q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(Y(0))
    assert c == expected


def test_operator_self_negation():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 5;
            a = -a;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=-5.0))
    assert c == expected


def test_angle():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            angle theta = pi / 2;
            rx(theta) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.pi / 2))
    assert c == expected


def test_stretch():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            stretch theta = pi;
            rx(theta) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=math.pi))
    assert c == expected


def test_duration():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            duration t = 1ms;
            rx(t) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1e-3))
    assert c == expected


def test_uint():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            uint[4] a = 5;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=5.0))
    assert c == expected


def test_uint_overflow():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            uint[4] a = 16;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=0.0))
    assert c == expected


def test_array_int():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            array[int[32], 5] arr = {0, 1, 2, 3, 4};
            rx(arr[2]) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2.0))
    assert c == expected


def test_array_multi_dim():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            array[int[32], 2, 3] arr = {{0, 1, 2}, {3, 4, 5}};
            rx(arr[1, 2]) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=5.0))
    assert c == expected


def test_in_place_addition():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 1;
            a += 2;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.0))
    assert c == expected


def test_in_place_subtraction():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 5;
            a -= 2;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=3.0))
    assert c == expected


def test_modifier_ctrll():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            ctrl @ h q[0], q[1];
        """)
    expected = Circuit(2)
    expected.add(Controlled(0, basic_gate=H(1)))
    assert c == expected


def test_modifier_negctrl():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            negctrl @ h q[0], q[1];
        """)
    expected = Circuit(2)
    expected.add(X(0))
    expected.add(Controlled(0, basic_gate=H(1)))
    expected.add(X(0))
    assert c == expected


def test_modifier_pow():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            pow(2) @ h q[0];
        """)
    expected = Circuit(1)
    expected.add(H(0))
    expected.add(H(0))
    assert c == expected


def test_modifier_pow_neg():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            pow(-1) @ h q[0];
        """)
    expected = Circuit(1)
    expected.add(Adjoint(H(0)))
    assert c == expected


def test_modifier_pow_one():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            pow(1) @ h q[0];
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_modifier_fractional_power_raises_error():
    with pytest.raises(ValueError, match="Invalid value"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            pow(0.5) @ h q[0];
        """)


def test_modifier_inv():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            inv @ h q[0];
        """)
    expected = Circuit(1)
    expected.add(Adjoint(H(0)))
    assert c == expected


def test_include_raises_on_bad_path():
    with pytest.raises(ValueError, match=r"something.inc"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            include "something.inc";
        """)


def test_for_range_with_step():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            for int i in [0:2:5] {
                x q[i];
            }
        """)
    expected = Circuit(5)
    expected.add(X(0))
    expected.add(X(2))
    expected.add(X(4))
    assert c == expected


def test_for_in_array():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[5] q;
            array[int[32], 3] indices = {0, 2, 4};
            for int[32] i in indices {
                x q[i];
            }
        """)
    expected = Circuit(5)
    expected.add(X(0))
    expected.add(X(2))
    expected.add(X(4))
    assert c == expected


def test_while_loop():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int i = 0;
            while (i < 3) {
                x q[0];
                i += 1;
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_while_loop_with_break():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int i = 0;
            while (i < 5) {
                x q[0];
                if (i == 2) {
                    break;
                }
                i += 1;
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_while_loop_with_continue():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int i = 0;
            while (i < 5) {
                i += 1;
                if (i % 2 == 0) {
                    continue;
                }
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_for_loop_with_break():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in [0:5] {
                if (i != 3) {
                } else {
                    break;
                }
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_for_loop_with_continue():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in [0:5] {
                if (i % 2 == 0) {
                    continue;
                }
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_break_inside_switch_inside_for_loop():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in [0:5] {
                switch (i) {
                    case 2 {
                        break;
                    }
                    default {
                        x q[0];
                    }
                }
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_break_inside_switch_default_case_inside_for_loop():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in [0:5] {
                switch (i) {
                    case 2 {
                        x q[0];
                    }
                    default {
                        break;
                    }
                }
            }
        """)
    expected = Circuit(1)
    assert c == expected


def test_input_creates_parameter():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            input float theta;
            rx(theta) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=Parameter("theta", 0.0)))
    assert c == expected


def test_simple_circuit_roundtrip():
    original_qasm = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
x q[0];
rx(3.0) q[0];
h q[1];
"""
    c = from_qasm3(original_qasm)
    qasm = to_qasm3(c)
    assert qasm.strip() == original_qasm.strip()


def test_to_file():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            x q[0];
        """)
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        to_qasm3_file(c, tmp.name)
        qasm = pathlib.Path(tmp.name).read_text(encoding="utf-8")
    expected_qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
x q[0];
"""
    assert qasm.strip() == expected_qasm.strip()


def test_from_file():
    qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
x q[0];
"""
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(qasm.encode())
        tmp.flush()
        c = from_qasm3_file(tmp.name)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_cast_to_int():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            float a = 1.5;
            int b = int(a);
            rx(b) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=1.0))
    assert c == expected


def test_cast_to_float():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 2;
            float b = float(a);
            rx(b) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2.0))
    assert c == expected


def test_cast_to_bit():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 1;
            bit b = bit(a);
            if (b) {
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_cast_to_bool():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 1;
            bool b = bool(a);
            if (b) {
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_cast_to_complex():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            float a = 1.0;
            complex c = complex(a);
            rz(real(c)) q[0];
            rz(imag(c)) q[0];
        """)
    expected = Circuit(1)
    expected.add(RZ(0, phi=1.0))
    expected.add(RZ(0, phi=0.0))
    assert c == expected


def test_bad_register():
    with pytest.raises(ValueError, match="Undefined register"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            x q[0];
        """)


def test_hardware_qubit_index():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            x $0;
        """)
    expected = Circuit(1)
    expected.add(X(0))
    assert c == expected


def test_index_out_of_range():
    with pytest.raises(ValueError, match="out of bounds"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            x q[1];
        """)


def test_include_file():
    other_file_qasm = """
            OPENQASM 3.0;
            include "stdgates.inc";
            def apply_hadamard(qubit q) {
                h q;
            }
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix=".qasm") as tmp:
        tmp.write(other_file_qasm.encode())
        tmp.flush()
        c = from_qasm3(f"""
            OPENQASM 3.0;
            include "stdgates.inc";
            include "{tmp.name}";
            qubit[1] q;
            apply_hadamard(q[0]);
        """)
    expected = Circuit(1)
    expected.add(H(0))
    assert c == expected


def test_int_with_size():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int[16] a = 42;
            rx(a) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=42.0))
    assert c == expected


def test_for_loop_inside_subroutine():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def test() -> int {
                for int i in [0:5] {
                    return 1;
                }
                return 0;
            }
            qubit[test()] q;
            """)
    expected = Circuit(1)
    assert c == expected


def test_while_loop_inside_subroutine():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            def test() -> int {
                int i = 0;
                while (i < 5) {
                    return 1;
                }
                return 0;
            }
            qubit[test()] q;
            """)
    expected = Circuit(1)
    assert c == expected


def test_for_over_discrete_set():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in {0, 2, 4} {
                x q[0];
            }
        """)
    expected = Circuit(1)
    expected.add(X(0))
    expected.add(X(0))
    expected.add(X(0))
    assert c == expected


def test_init_with_other_variable():
    c = from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 1;
            int b = a + 1;
            rx(b) q[0];
        """)
    expected = Circuit(1)
    expected.add(RX(0, theta=2.0))
    assert c == expected


def test_bad_loop_range():
    with pytest.raises(ValueError, match="Invalid loop setup"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            for int i in [5:1.0] {
                x q[0];
            }
        """)


def test_not_iterable_loop():
    with pytest.raises(ValueError, match="not iterable"):
        from_qasm3("""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            int a = 3;
            for int i in a {
                x q[0];
            }
        """)
