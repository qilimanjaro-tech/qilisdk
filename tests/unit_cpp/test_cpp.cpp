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

#include <pybind11/embed.h>
#include "../../src/qilisdk_cpp/libs/pybind.h"
#include "gtest/gtest.h"

// One interpreter for the whole test binary
class PybindEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        initialize_all_pybind_types();
        ASSERT_TRUE(dtype.ptr() != nullptr) << "dtype was not initialized";
        ASSERT_TRUE(SupportsStaticKraus.ptr() != nullptr) << "SupportsStaticKraus was not initialized";
    }
    void TearDown() override { finalize_all_pybind_types(); }
};

int main(int argc, char** argv) {
    pybind11::scoped_interpreter guard{};
    ::testing::Environment* pybind_env = ::testing::AddGlobalTestEnvironment(new PybindEnvironment());
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// GCOV_EXCL_BR_STOP