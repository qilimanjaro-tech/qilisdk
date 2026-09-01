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

#include <cstdint>
#include <utility>
#include <vector>
#include "../../../libs/eigen.h"

// GCOV_EXCL_BR_START

SparseMatrix jump_drift_operator(const std::vector<SparseMatrix>& jump_operators);
SparseMatrix effective_hamiltonian(const SparseMatrix& hamiltonian, const SparseMatrix& drift);
double max_jump_rate_bound(const SparseMatrix& drift);
std::pair<double, double> schedule_step_extremes(const std::vector<double>& step_list);
void warn_if_jumps_underresolved(const SparseMatrix& jump_drift, const std::vector<double>& step_list, double max_expected_jumps_per_step);
void reset_jump_resolution_warning();

class TrajectoryUnraveling {
   private:
    int base_seed = 0;
    std::vector<std::uint64_t> streams;

    void ensure_capacity(long num_trajectories);

   public:
    explicit TrajectoryUnraveling(int seed);

    void apply_jumps(DenseMatrix& trajectories, const std::vector<SparseMatrix>& jump_operators);
    void apply_kraus(DenseMatrix& trajectories, const std::vector<SparseMatrix>& kraus_operators);
};

// GCOV_EXCL_BR_STOP
