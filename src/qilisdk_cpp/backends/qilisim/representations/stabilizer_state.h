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

#include <bitset>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "../digital/gate.h"

// GCOV_EXCL_BR_START

const int MAX_ROWS_STABILIZER = 256;

class StabilizerState {
   private:
    std::vector<std::bitset<MAX_ROWS_STABILIZER>> x_bits;
    std::vector<std::bitset<MAX_ROWS_STABILIZER>> z_bits;
    std::bitset<MAX_ROWS_STABILIZER> phases;
    int nqubits;
    mutable std::mt19937_64 rng{std::random_device{}()};

   public:
    StabilizerState(int nqubits);
    void set_seed(uint64_t seed) const { rng.seed(seed); }
    const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& get_x_bits() const { return x_bits; }
    const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& get_z_bits() const { return z_bits; }
    const std::bitset<MAX_ROWS_STABILIZER>& get_phases() const { return phases; }
    int get_nqubits() const { return nqubits; }
    std::complex<double> apply_gate(const Gate& gate, bool track_global_phase = true);
    std::string sample() const;
    std::string sample(std::mt19937_64& engine) const;
    std::complex<double> amplitude(const std::string& b) const;
    void project_z(int q, bool outcome);
    int find_x_pivot(int q) const;
    void rowsum(int h, int i);
    bool z_eigenvalue(int q) const;
    std::string representative(int* support_rank = nullptr) const;
    std::complex<double> matrix_element(const Gate& gate, const std::string& b) const;
    std::complex<double> dropped_global_phase(const StabilizerState& after, const Gate& gate) const;
};

class StabilizerStateSum {
   private:
    int nqubits = 0;
    int max_terms = 0;
    std::vector<StabilizerState> states = {};
    std::vector<std::complex<double>> coefficients = {};
    mutable std::mt19937_64 rng{std::random_device{}()};

   public:
    StabilizerStateSum(int nqubits) : nqubits(nqubits) {
        states.emplace_back(nqubits);
        coefficients.push_back(1.0);
    }
    StabilizerStateSum(int nqubits, const std::vector<StabilizerState>& states, const std::vector<std::complex<double>>& coefficients) : nqubits(nqubits), states(states), coefficients(coefficients) {}
    const std::vector<StabilizerState>& get_states() const { return states; }
    const std::vector<std::complex<double>>& get_coefficients() const { return coefficients; }
    friend std::ostream& operator<<(std::ostream& os, const StabilizerStateSum& sss);
    std::map<std::string, int> sample(int nshots) const;
    std::complex<double> amplitude(const std::string& b) const;
    int get_nqubits() const { return nqubits; }
    int get_max_terms() const { return max_terms; }
    void set_max_terms(int n) { max_terms = n; }
    void set_seed(uint64_t seed) const {
        rng.seed(seed);
        for (size_t i = 0; i < states.size(); ++i) {
            states[i].set_seed(seed + i + 1);
        }
    }
    void apply_gate(const Gate& gate);
    void truncate();
    void combine_duplicates();
    void normalize();
};

// GCOV_EXCL_BR_STOP