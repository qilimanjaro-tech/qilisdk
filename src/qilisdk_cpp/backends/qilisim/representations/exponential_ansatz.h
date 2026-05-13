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

#include "matrix_free_hamiltonian.h"
#include <vector>

// GCOV_EXCL_BR_START

struct SampleSet {
    std::vector<long> configs;
    DenseMatrix O_mat;  // (N_s x p) log-derivatives: O_k(x) = P_k(x) ∈ {-1,+1}
};

class ExponentialAnsatz {
   private:
    int shots = 100;
    int shots_warmup = shots;
    MatrixFreeHamiltonian terms = MatrixFreeHamiltonian(0);
    int num_qubits = 0;

    std::vector<long> build_z_bits() const;

   public:
    ExponentialAnsatz(int num_qubits, int max_terms);
    friend std::ostream& operator<<(std::ostream& os, const ExponentialAnsatz& ansatz);
    void set_shots(int new_shots) { shots = new_shots; shots_warmup = new_shots; }
    int get_shots() const { return shots; }
    MatrixFreeHamiltonian get_terms() const { return terms; }
    MatrixFreeHamiltonian& get_terms() { return terms; }
    SampleSet draw_samples() const;
    SampleSet draw_samples(int N_s, int n_warmup) const;
    Eigen::VectorXcd local_energy(const SampleSet& samples, const MatrixFreeHamiltonian& H) const;
    double expectation_value(const MatrixFreeHamiltonian& observable) const;
    ExponentialAnsatz operator*(const double& scalar) const;
    ExponentialAnsatz& operator*=(const double& scalar);
    ExponentialAnsatz operator+(const ExponentialAnsatz& other) const;
    ExponentialAnsatz& operator+=(const ExponentialAnsatz& other);
    DenseMatrix to_dense() const;
};

// GCOV_EXCL_BR_STOP