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

#include "matrix_free_hamiltonian.h"
#include "../utils/matrix_utils.h"

// void MatrixFreeHamiltonian::apply(DenseMatrix& output_state, MatrixFreeApplicationType application_type) const {
//     DenseMatrix new_state = DenseMatrix::Zero(output_state.rows(), output_state.cols());
//     for (const auto& [coefficient, ops] : operators) {
//         DenseMatrix temp_state = output_state;
//         for (const auto& op : ops) {
//             op.apply(temp_state, application_type);
//         }
//         new_state += coefficient * temp_state;
//     }
//     output_state = new_state;
// }

DenseMatrix m_temp_state;
DenseMatrix m_new_state;

void MatrixFreeHamiltonian::apply(DenseMatrix& output_state, 
                                   MatrixFreeApplicationType application_type) const {
    m_new_state.resizeLike(output_state);
    m_new_state.setZero();
    m_temp_state.resizeLike(output_state);
    for (const auto& [coefficient, ops] : operators) {
        m_temp_state = output_state;
        for (const auto& op : ops) {
            op.apply(m_temp_state, application_type);
        }
        m_new_state.noalias() += coefficient * m_temp_state;
    }
    output_state.swap(m_new_state);
}

double MatrixFreeHamiltonian::expectation_value(const DenseMatrix& state) const {
    double exp_val = 0.0;
    m_temp_state.resizeLike(state);
    for (const auto& [coefficient, ops] : operators) {
        m_temp_state = state;
        for (const auto& op : ops) {
            op.apply(m_temp_state, MatrixFreeApplicationType::Left);
        }
        exp_val += std::real(coefficient * dot(state, m_temp_state));
    }
    return exp_val;
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator*=(const std::complex<double>& scalar) {
    for (auto& [coefficient, ops] : operators) {
        coefficient *= scalar;
    }
    return *this;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const std::complex<double>& scalar) const {
    MatrixFreeHamiltonian result = *this;
    result *= scalar;
    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const double& scalar) const {
    return (*this) * std::complex<double>(scalar, 0.0);
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator+=(const MatrixFreeHamiltonian& other) {
    std::unordered_map<std::string, int> op_map;
    for (const auto& [coefficient, ops] : other.operators) {
        std::string id;
        for (const auto& op : ops) {
            id += op.get_id() + "|";
        }
        if (op_map.find(id) != op_map.end()) {
            operators[op_map[id]].first += coefficient;
        } else {
            operators.push_back({coefficient, ops});
            op_map[id] = operators.size() - 1;
        }
    }
    return *this;
}

std::ostream& operator<<(std::ostream& os, const MatrixFreeHamiltonian& hamiltonian) {
    int count = 0;
    for (const auto& [coefficient, ops] : hamiltonian.operators) {
        os << coefficient << " * ";
        for (const auto& op : ops) {
            os << op;
            if (&op != &ops.back()) {
                os << " ";
            }
        }
        if (count < int(hamiltonian.operators.size()) - 1) {
            os << " + ";
        }
        count++;
    }
    return os;
}

void MatrixFreeHamiltonian::add(const std::complex<double>& coeff, const MatrixFreeOperator& op) {
    operators.push_back({coeff, std::vector<MatrixFreeOperator>{op}});
}
void MatrixFreeHamiltonian::add(const std::complex<double>& coeff, const std::vector<MatrixFreeOperator>& ops) {
    operators.push_back({coeff, ops});
}