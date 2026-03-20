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
#include "../../../src/qilisdk_cpp/libs/numpy.h"
#include <pybind11/embed.h>

namespace py = pybind11;

TEST(PybindAllTypes, Initialization) {
    initialize_all_pybind_types();
    EXPECT_TRUE(dtype.ptr() != nullptr);
    EXPECT_TRUE(SupportsStaticKraus.ptr() != nullptr);
}

TEST(PybindExternalTypes, Initialization) {
    initialize_external_pybind_types();
    EXPECT_TRUE(numpy_array.ptr() != nullptr);
    EXPECT_TRUE(csrmatrix.ptr() != nullptr);
}




