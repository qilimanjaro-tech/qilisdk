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

#include <complex>
#include <map>
#include <set>
#include <vector>
#include <bitset>

// GCOV_EXCL_BR_START

const int MAX_QUBITS_STABILIZER = 512;

class StabilizerState {
   private:
    std::vector<std::bitset<MAX_QUBITS_STABILIZER>> x_stabilizers;
    std::vector<std::bitset<MAX_QUBITS_STABILIZER>> z_stabilizers;
    std::bitset<MAX_QUBITS_STABILIZER> phases;
    int nqubits;

   public:
    StabilizerState(int nqubits) : nqubits(nqubits) {}
    StabilizerState(int nqubits, const std::vector<std::bitset<MAX_QUBITS_STABILIZER>>& x_stabilizers, const std::vector<std::bitset<MAX_QUBITS_STABILIZER>>& z_stabilizers, const std::bitset<MAX_QUBITS_STABILIZER>& phases)
        : x_stabilizers(x_stabilizers), z_stabilizers(z_stabilizers), phases(phases), nqubits(nqubits) {} 
    const std::vector<std::bitset<MAX_QUBITS_STABILIZER>>& get_x_stabilizers() const { return x_stabilizers; }
    const std::vector<std::bitset<MAX_QUBITS_STABILIZER>>& get_z_stabilizers() const { return z_stabilizers; }
    const std::bitset<MAX_QUBITS_STABILIZER>& get_phases() const { return phases; }
    int get_nqubits() const { return nqubits; }
};

class StabilizerStateSum {
    private:
     int nqubits;
     std::vector<StabilizerState> states;
     std::vector<std::complex<double>> coefficients;
    
    public:
     StabilizerStateSum(int nqubits) : nqubits(nqubits) {}
     StabilizerStateSum(const std::vector<StabilizerState>& states, const std::vector<std::complex<double>>& coefficients) : states(states), coefficients(coefficients) {}
     const std::vector<StabilizerState>& get_states() const { return states; }
     const std::vector<std::complex<double>>& get_coefficients() const { return coefficients; }
     friend std::ostream& operator<<(std::ostream& os, const StabilizerStateSum& sss);
     std::map<std::string, int> sample(int nshots) const;
     int get_nqubits() const { return nqubits; }
    };

// GCOV_EXCL_BR_STOP