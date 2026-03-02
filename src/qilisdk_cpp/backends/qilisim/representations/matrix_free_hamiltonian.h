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

#include "matrix_free_operator.h"

class MatrixFreeHamiltonian {
    private:
        std::vector<std::pair<std::complex<double>, std::vector<MatrixFreeOperator>>> operators;
    public:
        MatrixFreeHamiltonian() {}
        MatrixFreeHamiltonian(const MatrixFreeOperator& op) {
            operators.push_back(std::make_pair(std::complex<double>(1.0, 0.0), std::vector<MatrixFreeOperator>{op}));
        }
        MatrixFreeHamiltonian(const std::vector<std::pair<std::complex<double>, std::vector<MatrixFreeOperator>>>& ops) : operators(ops) {}
        void apply(DenseMatrix& output_state, MatrixFreeApplicationType application_type) const;
        double expectation_value(const DenseMatrix& state) const;
        MatrixFreeHamiltonian& operator*=(const std::complex<double>& scalar);
        MatrixFreeHamiltonian operator*(const std::complex<double>& scalar) const;
        MatrixFreeHamiltonian operator*(const double& scalar) const;
        MatrixFreeHamiltonian& operator+=(const MatrixFreeHamiltonian& other);
        void add(const std::complex<double>& coeff, const MatrixFreeOperator& op);
        void add(const std::complex<double>& coeff, const std::vector<MatrixFreeOperator>& op);
        friend std::ostream& operator<<(std::ostream& os, const MatrixFreeHamiltonian& hamiltonian);

};