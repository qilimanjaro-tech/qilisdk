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

// GCOV_EXCL_BR_START

class ExponentialAnsatz {
   private:
    MatrixFreeHamiltonian terms;
    int num_qubits;

   public:
    ExponentialAnsatz(int num_qubits, int max_terms);
    friend std::ostream& operator<<(std::ostream& os, const ExponentialAnsatz& ansatz);
    MatrixFreeHamiltonian get_terms() const { return terms; }
    MatrixFreeHamiltonian& get_terms() { return terms; }
    double expectation_value(const MatrixFreeHamiltonian& observable) const;
    ExponentialAnsatz operator*(const double& scalar) const;
    ExponentialAnsatz& operator*=(const double& scalar);
    ExponentialAnsatz operator+(const ExponentialAnsatz& other) const;
    ExponentialAnsatz& operator+=(const ExponentialAnsatz& other);
    DenseMatrix to_dense() const;
};

// GCOV_EXCL_BR_STOP