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
#include <pybind11/embed.h>
#include <functional>
#include <string>
#include "../../../src/qilisdk_cpp/libs/numpy.h"

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

namespace {

// Adds a temporary loguru sink that records every emitted message into `records`,
// runs `emit`, then removes the sink again. The recorded strings are "{level}|{message}".
void captureLogs(py::list records, const std::function<void()>& emit) {
    auto sink = py::cpp_function([records](py::object message) mutable { records.append(py::str(message)); });
    py::object handler_id = logger.attr("add")(sink, py::arg("format") = "{level}|{message}", py::arg("level") = "DEBUG");
    emit();
    logger.attr("remove")(handler_id);
}

bool containsString(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST(PybindLogging, WarningEmitsMessage) {
    py::list records;
    captureLogs(records, []() { warning("a warning message"); });
    ASSERT_EQ(py::len(records), 1u);
    std::string record = records[0].cast<std::string>();
    EXPECT_TRUE(containsString(record, "WARNING"));
    EXPECT_TRUE(containsString(record, "a warning message"));
}

TEST(PybindLogging, InfoEmitsMessage) {
    py::list records;
    captureLogs(records, []() { info("an info message"); });
    ASSERT_EQ(py::len(records), 1u);
    std::string record = records[0].cast<std::string>();
    EXPECT_TRUE(containsString(record, "INFO"));
    EXPECT_TRUE(containsString(record, "an info message"));
}

TEST(PybindLogging, ErrorEmitsMessage) {
    py::list records;
    captureLogs(records, []() { error("an error message"); });
    ASSERT_EQ(py::len(records), 1u);
    std::string record = records[0].cast<std::string>();
    EXPECT_TRUE(containsString(record, "ERROR"));
    EXPECT_TRUE(containsString(record, "an error message"));
}

// GCOV_EXCL_BR_STOP
