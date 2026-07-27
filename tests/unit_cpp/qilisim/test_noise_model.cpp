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

TEST(NoiseModel, GetRelevantKrausOperators) {
    NoiseModelCpp model;
    model.add_kraus_operators_global({SparseMatrix(2, 2), SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(0, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(1, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate("H", {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate_qubit("H", 0, {SparseMatrix(2, 2)});

    // Test retrieval for a gate with all types of Kraus operators (0 controls)
    auto result = model.get_relevant_kraus_operators("H", 0, {0, 1}, 2);
    // We expect to get 1 (global) + 1 (qubit 0) + 1 (qubit 1) + 1 (gate "H") + 1 (gate "H" on qubit 0) = 5 Kraus operators
    EXPECT_EQ(result.size(), 5);

    // Test retrieval for a gate with only global and per-qubit operators
    result = model.get_relevant_kraus_operators("X", 0, {0, 1}, 2);
    // We expect to get 1 (global) + 1 (qubit 0) + 1 (qubit 1) = 3 Kraus operators
    EXPECT_EQ(result.size(), 3);

    // Test retrieval for a gate with only global operators
    result = model.get_relevant_kraus_operators("Y", 0, {2}, 3);
    // We expect to get 1 (global) = 1 Kraus operator
    EXPECT_EQ(result.size(), 1);
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