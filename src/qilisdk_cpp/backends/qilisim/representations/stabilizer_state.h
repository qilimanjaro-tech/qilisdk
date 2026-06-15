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
    const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& get_x_bits() const { return x_bits; }
    const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& get_z_bits() const { return z_bits; }
    const std::bitset<MAX_ROWS_STABILIZER>& get_phases() const { return phases; }
    int get_nqubits() const { return nqubits; }
    // track_global_phase controls whether the (relatively expensive) exact global-phase recovery
    // is performed. It is only needed when this state is one term of a StabilizerStateSum with
    // more than one term; for a lone stabilizer state the global phase is unobservable.
    std::complex<double> apply_gate(const Gate& gate, bool track_global_phase = true);
    std::string sample() const;
    std::complex<double> amplitude(const std::string& b) const;
    void project_z(int q, bool outcome);
    int find_x_pivot(int q) const;
    void rowsum(int h, int i);
    bool z_eigenvalue(int q) const;
    // A computational basis bitstring guaranteed to have nonzero amplitude (used as a reference
    // point for recovering the global phase a Clifford tableau update drops).
    std::string representative() const;
    // <b| G |this>, the exact (true) amplitude of basis state b after applying gate G to this state,
    // computed from this state's amplitudes and G's action.
    std::complex<double> matrix_element(const Gate& gate, const std::string& b) const;
    // The global phase the tableau update for `gate` dropped, given the pre-gate state (*this) and
    // the post-gate state `after`. Returns the unit-modulus factor the term's coefficient must be
    // multiplied by so that interference between terms of a StabilizerStateSum stays correct.
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
    StabilizerStateSum(int nqubits, const std::vector<StabilizerState>& states, const std::vector<std::complex<double>>& coefficients) : states(states), coefficients(coefficients), nqubits(nqubits) {}
    const std::vector<StabilizerState>& get_states() const { return states; }
    const std::vector<std::complex<double>>& get_coefficients() const { return coefficients; }
    friend std::ostream& operator<<(std::ostream& os, const StabilizerStateSum& sss);
    std::map<std::string, int> sample(int nshots) const;
    std::complex<double> amplitude(const std::string& b) const;
    int get_nqubits() const { return nqubits; }
    int get_max_terms() const { return max_terms; }
    void set_max_terms(int n) { max_terms = n; }
    void apply_gate(const Gate& gate);
    void truncate();
    void combine_duplicates();
    void normalize();
};

// GCOV_EXCL_BR_STOP