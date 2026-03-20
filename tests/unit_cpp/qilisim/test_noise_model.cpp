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

#include <gtest/gtest.h>
#include "../../../src/qilisdk_cpp/backends/qilisim/noise/noise_model.h"

TEST(NoiseModel, GetRelevantKrausOperators) {

    NoiseModelCpp model;
    model.add_kraus_operators_global({SparseMatrix(2, 2), SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(0, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_qubit(1, {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate("H", {SparseMatrix(2, 2)});
    model.add_kraus_operators_per_gate_qubit("H", 0, {SparseMatrix(2, 2)});

    // Test retrieval for a gate with all types of Kraus operators
    auto result = model.get_relevant_kraus_operators("H", {0, 1}, 2);
    // We expect to get 1 (global) + 1 (qubit 0) + 1 (qubit 1) + 1 (gate "H") + 1 (gate "H" on qubit 0) = 5 Kraus operators
    EXPECT_EQ(result.size(), 5);

    // Test retrieval for a gate with only global and per-qubit operators
    result = model.get_relevant_kraus_operators("X", {0, 1}, 2);
    // We expect to get 1 (global) + 1 (qubit 0) + 1 (qubit 1) = 3 Kraus operators
    EXPECT_EQ(result.size(), 3);

    // Test retrieval for a gate with only global operators
    result = model.get_relevant_kraus_operators("Y", {2}, 3);
    // We expect to get 1 (global) = 1 Kraus operator
    EXPECT_EQ(result.size(), 1);

}