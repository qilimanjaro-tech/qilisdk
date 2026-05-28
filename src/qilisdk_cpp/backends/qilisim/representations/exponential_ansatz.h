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

#include <vector>
#include "matrix_free_hamiltonian.h"

// GCOV_EXCL_BR_START

struct SampleSet {
    std::vector<Bitset> configs;
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> O_mat;  // (N_s x p) log-derivatives: O_k(x) = P_k(x) ∈ {-1,+1}, stored as int8 for cache efficiency
};

class ExponentialAnsatz {
   private:
    int shots;
    int warmups;
    int order;
    MatrixFreeHamiltonian terms = MatrixFreeHamiltonian(0);
    int num_qubits;

    std::vector<Bitset> build_z_bits() const;

   public:
    ExponentialAnsatz(int num_qubits, int order, int shots, int warmups);
    friend std::ostream& operator<<(std::ostream& os, const ExponentialAnsatz& ansatz);
    void set_shots(int new_shots) { shots = new_shots; }
    void set_warmups(int new_warmups) { warmups = new_warmups; }
    void set_order(int new_order) { order = new_order; }
    int get_order() const { return order; }
    int get_shots() const { return shots; }
    int get_warmups() const { return warmups; }
    MatrixFreeHamiltonian get_terms() const { return terms; }
    MatrixFreeHamiltonian& get_terms() { return terms; }
    SampleSet draw_samples() const;
    SampleSet draw_samples(int N_s, int n_warmup) const;
    Eigen::VectorXcd local_energy(const SampleSet& samples, const MatrixFreeHamiltonian& H) const;
    double expectation_value(const MatrixFreeHamiltonian& observable) const;
    void prune_terms_not_in_hamiltonian(const MatrixFreeHamiltonian& H);
    ExponentialAnsatz operator*(const double& scalar) const;
    ExponentialAnsatz& operator*=(const double& scalar);
    ExponentialAnsatz operator+(const ExponentialAnsatz& other) const;
    ExponentialAnsatz& operator+=(const ExponentialAnsatz& other);
    ExponentialAnsatz zeroed() const;
    DenseMatrix to_dense() const;
};

// GCOV_EXCL_BR_STOP