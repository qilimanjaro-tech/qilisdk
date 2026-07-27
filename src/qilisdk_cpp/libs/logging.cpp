// Copyright 2025 Qilimanjaro Quantum Tech
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
#include "logging.h"

#include <limits>

#include "pybind.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

namespace qilisdk {

namespace {

// Numeric severities of loguru's built-in levels
constexpr int kLevelTrace = 5;
constexpr int kLevelDebug = 10;
constexpr int kLevelInfo = 20;
constexpr int kLevelSuccess = 25;
constexpr int kLevelWarning = 30;

// Cached copy of the minimum enabled level across all active loguru sinks
int min_enabled_level = std::numeric_limits<int>::max();

void log_at(const char* level, int level_no, const std::string& message) {
    /*
    Log a message at the given level, if that level is enabled.

    Args:
        level (const char*): the name of the level, e.g. "info", "debug", etc.
        level_no (int): the numeric severity of the level, e.g. 20 for info, 10 for debug, etc.
        message (const std::string&): the message to log
    */
    if (level_no < min_enabled_level) {
        return;
    }
    if (!logger) {
        return;  // GCOVR_EXCL_LINE
    }
    logger.attr(level)(message);
}
}  // namespace

void refresh_log_level() {
    /*
    Refresh the cached minimum enabled log level from the Python logger.
    */
    if (!logger_core) {
        min_enabled_level = std::numeric_limits<int>::max();
        return;
    }
    const double level = logger_core.attr("min_level").cast<double>();
    min_enabled_level = level >= static_cast<double>(std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max() : static_cast<int>(level);
}

void log_trace(const std::string& message) {
    log_at("trace", kLevelTrace, message);
}
void log_debug(const std::string& message) {
    log_at("debug", kLevelDebug, message);
}
void log_info(const std::string& message) {
    log_at("info", kLevelInfo, message);
}
void log_success(const std::string& message) {
    log_at("success", kLevelSuccess, message);
}
void log_warning(const std::string& message) {
    log_at("warning", kLevelWarning, message);
}

}  // namespace qilisdk

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
