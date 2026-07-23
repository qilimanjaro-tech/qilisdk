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
#include "../../../src/qilisdk_cpp/libs/logging.h"
#include "../../../src/qilisdk_cpp/libs/pybind.h"

namespace py = pybind11;

namespace {

py::list captureLogsAt(const char* level, const std::function<void()>& emit) {
    py::list records;
    logger.attr("remove")();
    auto sink = py::cpp_function([records](py::object message) mutable { records.append(py::str(message)); });
    logger.attr("add")(sink, py::arg("format") = "{level}|{message}", py::arg("level") = level);
    qilisdk::refresh_log_level();
    emit();
    logger.attr("remove")();
    qilisdk::refresh_log_level();
    return records;
}

bool containsString(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST(Logging, EmitsEveryLevelWhenEnabled) {
    py::list records = captureLogsAt("TRACE", []() {
        qilisdk::log_trace("trace message");
        qilisdk::log_debug("debug message");
        qilisdk::log_info("info message");
        qilisdk::log_success("success message");
        qilisdk::log_warning("warning message");
    });
    ASSERT_EQ(py::len(records), 5u);
    EXPECT_TRUE(containsString(records[0].cast<std::string>(), "TRACE"));
    EXPECT_TRUE(containsString(records[1].cast<std::string>(), "DEBUG"));
    EXPECT_TRUE(containsString(records[2].cast<std::string>(), "INFO"));
    EXPECT_TRUE(containsString(records[3].cast<std::string>(), "SUCCESS"));
    EXPECT_TRUE(containsString(records[4].cast<std::string>(), "warning message"));
}

TEST(Logging, SkipsLevelsBelowThreshold) {
    py::list records = captureLogsAt("WARNING", []() {
        qilisdk::log_trace("trace message");
        qilisdk::log_debug("debug message");
        qilisdk::log_info("info message");
        qilisdk::log_warning("warning message");
    });
    ASSERT_EQ(py::len(records), 1u);
    EXPECT_TRUE(containsString(records[0].cast<std::string>(), "warning message"));
}

// GCOV_EXCL_BR_STOP
