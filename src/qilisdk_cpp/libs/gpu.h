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
#pragma once

#include "eigen.h"  // adjust if eigen typedefs live elsewhere

// ---------------------------------------------------------------------------
// Optional GPU acceleration header
//
// This header intentionally exposes NO CUDA types.
// It loads libcudart / libcublas / libcusolver lazily via
// dlopen at runtime, so the wheel has ZERO link-time CUDA dependency and runs
// unchanged on machines without a GPU.
// ---------------------------------------------------------------------------

namespace qilisdk::gpu {

bool cuda_available();
bool sr_solve(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double epsilon, Eigen::VectorXcd& adot);

}  // namespace qilisdk::gpu
