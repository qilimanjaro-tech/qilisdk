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

#include <cstddef>
#include <utility>
#include <vector>

// GCOV_EXCL_BR_START

struct MonomialCpp {
    std::vector<int> variables;
    double coefficient;
};

struct AnnealingResultCpp {
    std::vector<int> state;
    double energy;
    std::vector<std::vector<int>> states;
    std::vector<double> energies;
};

class SimulatedAnnealingCpp {
   public:
    SimulatedAnnealingCpp(int num_variables, const std::vector<std::pair<std::vector<int>, double>>& monomials, double offset);

    double energy(const std::vector<int>& state) const;
    std::pair<double, double> default_beta_range() const;
    AnnealingResultCpp anneal(int num_reads, int num_sweeps, double beta_min, double beta_max, int seed, int num_threads);

    int get_num_variables() const { return num_variables; }
    double get_offset() const { return offset; }
    std::size_t get_num_monomials() const { return monomials.size(); }

   private:
    double flip_delta(const std::vector<int>& state, int variable) const;
    std::pair<std::vector<int>, double> single_read(int num_sweeps, double beta_min, double beta_max, unsigned long long seed) const;

    int num_variables;
    double offset;
    std::vector<MonomialCpp> monomials;
    std::vector<std::vector<int>> variable_monomials;
};

// GCOV_EXCL_BR_STOP
