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
#pragma once

#include <string>

// Thin wrappers around the globally-available Python loguru `logger` (see pybind.h).
//
// IMPORTANT: these call back into Python and therefore require the GIL to be held.
// Only call them from the master thread and NEVER from inside an OpenMP parallel
// region, otherwise concurrent Python calls from worker threads will crash.
//
// The message string is built eagerly at the call site regardless of the active log
// level, so keep these calls coarse (per-execution / per-phase) and out of hot loops.
namespace qilisdk {

void log_trace(const std::string& message);
void log_debug(const std::string& message);
void log_info(const std::string& message);
void log_success(const std::string& message);
void log_warning(const std::string& message);

}  // namespace qilisdk
