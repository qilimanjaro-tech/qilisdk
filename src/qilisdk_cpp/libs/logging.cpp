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

#include "pybind.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

namespace qilisdk {

namespace {
// Forward a message to the given loguru level. Safe to call before the logger has
// been initialised (or after finalisation): it simply becomes a no-op.
void log_at(const char* level, const std::string& message) {
    if (!logger) {
        return;
    }
    // The loguru filter matches on the nearest Python frame's module name, which is
    // always a `qilisdk.*` frame while a simulator is running, so C++ logs pass it.
    logger.attr(level)(message);
}
}  // namespace

void log_trace(const std::string& message) { log_at("trace", message); }
void log_debug(const std::string& message) { log_at("debug", message); }
void log_info(const std::string& message) { log_at("info", message); }
void log_success(const std::string& message) { log_at("success", message); }
void log_warning(const std::string& message) { log_at("warning", message); }

}  // namespace qilisdk

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
