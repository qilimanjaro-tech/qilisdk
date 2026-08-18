// Copyright 2026 Qilimanjaro Quantum Tech
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>
#include "../../../src/qilisdk_cpp/backends/qilisim/noise/noise_model.h"
#include "../../../src/qilisdk_cpp/libs/pybind.h"

TEST(NoiseModel, GetRelevantKrausOperators) {
    NoiseModelCpp model;
    model.add_kraus_operators_global({SparseMatrix(2, 2), SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(0, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(1, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate("H", {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate_qubit("H", 0, {SparseMatrix(2, 2)});

    // Test retrieval for a gate with all types of Kraus operators (0 controls). Global and per-gate
    // noise is applied once per gate qubit, per-qubit noise only on the qubit it is attached to.
    auto result = model.get_relevant_kraus_operators("H", 0, {0, 1}, 2);
    // We expect 2 (global) + 1 (qubit 0) + 1 (qubit 1) + 2 (gate "H") + 1 (gate "H" on qubit 0) = 7 Kraus operators
    EXPECT_EQ(result.size(), 7);

    // Test retrieval for a gate with only global and per-qubit operators
    result = model.get_relevant_kraus_operators("X", 0, {0, 1}, 2);
    // We expect 2 (global) + 1 (qubit 0) + 1 (qubit 1) = 4 Kraus operators
    EXPECT_EQ(result.size(), 4);

    // Test retrieval for a gate with only global operators
    result = model.get_relevant_kraus_operators("Y", 0, {2}, 3);
    // We expect to get 1 (global) = 1 Kraus operator
    EXPECT_EQ(result.size(), 1);
}

TEST(NoiseModel, PerQubitNoiseStaysOnItsOwnQubit) {
    // X as a (trivial) single-qubit Kraus operator attached to qubit 0 only.
    SparseMatrix X(2, 2);
    X.insert(0, 1) = 1.0;
    X.insert(1, 0) = 1.0;
    X.makeCompressed();

    NoiseModelCpp model;
    model.add_kraus_operators_per_qubit(0, {X});

    // A two-qubit gate on qubits {0, 1} must get the operator on qubit 0 alone, i.e. X (x) I.
    auto result = model.get_relevant_kraus_operators("SWAP", 0, {0, 1}, 2);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 1u);
    SparseMatrix identity(2, 2);
    identity.setIdentity();
    SparseMatrix expected = Eigen::kroneckerProduct(X, identity).eval();
    EXPECT_TRUE(result[0][0].isApprox(expected));

    // The noise is still found when its qubit is the control of the gate rather than a target.
    EXPECT_EQ(model.get_relevant_kraus_operators("X", 1, {0, 1}, 2).size(), 1u);

    // A gate that does not touch qubit 0 gets no noise at all.
    EXPECT_EQ(model.get_relevant_kraus_operators("X", 0, {1}, 2).size(), 0u);
}

TEST(NoiseModel, MultiQubitKrausOperatorsMustSpanTheRegister) {
    SparseMatrix two_qubit_op(4, 4);
    two_qubit_op.setIdentity();

    NoiseModelCpp model;
    model.add_kraus_operators_global({two_qubit_op});

    // Spanning the whole register is fine and is applied once, as-is.
    auto result = model.get_relevant_kraus_operators("SWAP", 0, {0, 1}, 2);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0][0].rows(), 4);

    // Acting on more than one qubit but less than the register is ambiguous.
    EXPECT_THROW(model.get_relevant_kraus_operators("SWAP", 0, {0, 1}, 3), py::value_error);
}

TEST(NoiseModel, PerQubitKrausOperatorsMustBeSingleQubit) {
    // A whole-register set attached to a specific qubit is ambiguous: it would be applied once per
    // qubit it was attached to, so it is rejected rather than silently applied several times.
    SparseMatrix two_qubit_op(4, 4);
    two_qubit_op.setIdentity();

    NoiseModelCpp per_qubit_model;
    per_qubit_model.add_kraus_operators_per_qubit(0, {two_qubit_op});
    EXPECT_THROW(per_qubit_model.get_relevant_kraus_operators("SWAP", 0, {0, 1}, 2), py::value_error);

    NoiseModelCpp per_gate_qubit_model;
    per_gate_qubit_model.add_kraus_operators_per_gate_qubit(NoiseModelCpp::make_gate_key("SWAP", 0), 0, {two_qubit_op});
    EXPECT_THROW(per_gate_qubit_model.get_relevant_kraus_operators("SWAP", 0, {0, 1}, 2), py::value_error);
}

TEST(NoiseModel, TimeDependentJumpRateSeries) {
    NoiseModelCpp model;

    // A constant jump operator stores an empty series and is not time-dependent.
    model.add_jump_operator(SparseMatrix(2, 2));
    EXPECT_FALSE(model.has_time_dependent_rates());

    // A time-dependent jump operator stores its per-step sqrt(rate) series and flips the flag.
    std::vector<double> series = {0.5, 1.0, 1.5};
    model.add_jump_operator(SparseMatrix(2, 2), series);
    EXPECT_TRUE(model.has_time_dependent_rates());

    ASSERT_EQ(model.get_jump_operators().size(), 2u);
    ASSERT_EQ(model.get_jump_rate_series().size(), 2u);
    EXPECT_TRUE(model.get_jump_rate_series()[0].empty());
    EXPECT_EQ(model.get_jump_rate_series()[1], series);
}

TEST(NoiseModel, PerGateNoiseDistinguishesControlCount) {
    // A controlled gate (e.g. CZ, base "Z" with 1 control) must not share per-gate noise
    // with its base gate (plain "Z", 0 controls).
    NoiseModelCpp model;
    model.add_kraus_operators_per_gate(NoiseModelCpp::make_gate_key("Z", 1), {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate_qubit(NoiseModelCpp::make_gate_key("Z", 1), 0, {SparseMatrix(2, 2)});

    // The controlled gate (1 control) sees its per-gate and per-gate-per-qubit noise.
    EXPECT_EQ(model.get_relevant_kraus_operators("Z", 1, {0}, 1).size(), 2u);
    // The plain gate (0 controls) sees none of it.
    EXPECT_EQ(model.get_relevant_kraus_operators("Z", 0, {0}, 1).size(), 0u);

    // Keys are distinct and control-count-aware.
    EXPECT_EQ(NoiseModelCpp::make_gate_key("Z", 0), "Z");
    EXPECT_EQ(NoiseModelCpp::make_gate_key("Z", 1), "Z_c1");
    EXPECT_EQ(NoiseModelCpp::make_gate_key("X", 2), "X_c2");
}

// GCOV_EXCL_BR_STOP