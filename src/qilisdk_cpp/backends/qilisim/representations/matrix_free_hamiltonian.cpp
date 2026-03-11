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

#include "matrix_free_hamiltonian.h"
#include <unordered_map>
#include "../utils/matrix_utils.h"

void MatrixFreeHamiltonian::apply(DenseMatrix& output_state, MatrixFreeApplicationType application_type) const {
    /*
    Applies the matrix-free Hamiltonian to the given output state.
    It iterates through each term in the Hamiltonian, applies the corresponding operators
    to the output state, and accumulates the results. The final result is stored back in the output state.

    Args:
        output_state: The state to which the Hamiltonian will be applied. This state will be modified in-place to contain the result.
        application_type: The type of application (Left, Right, or LeftAndRight) that determines how the operators are applied to the state.
    */
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

void MatrixFreeHamiltonian::apply(const DenseMatrix& input_state, MatrixFreeApplicationType application_type, DenseMatrix& output_state) const {
    /*
    Applies the matrix-free Hamiltonian to the given input state and writes the result to a separate output state.

    Args:
        input_state: The state to which the Hamiltonian will be applied.
        application_type: The type of application (Left, Right, or LeftAndRight).
        output_state: The state where the result will be stored.
    */
    m_temp_state.resizeLike(output_state);
    for (const auto& [coefficient, ops] : operators) {
        m_temp_state = input_state;
        for (const auto& op : ops) {
            op.apply(m_temp_state, application_type);
        }
        output_state += m_temp_state * coefficient;
    }
}

double MatrixFreeHamiltonian::expectation_value(const DenseMatrix& state) const {
    /*
    Calculate the expectation value of the Hamiltonian with respect to a given state.

    Args:
        state: The state for which the expectation value will be calculated.
    */
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
    /*
    Scale in-place by a complex scalar.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.

    Returns:
        A reference to the scaled Hamiltonian.
    */
    for (auto& [coefficient, ops] : operators) {
        coefficient *= scalar;
    }
    return *this;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const std::complex<double>& scalar) const {
    /*
    Scale by a complex scalar and return a new Hamiltonian.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.

    Returns:
        A new Hamiltonian that is the result of scaling this Hamiltonian by the given scalar.
    */
    MatrixFreeHamiltonian result = *this;
    result *= scalar;
    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const double& scalar) const {
    /*
    Scale by a real scalar and return a new Hamiltonian.

    Args:
        scalar: The real scalar by which to scale the Hamiltonian.

    Returns:
        A new Hamiltonian that is the result of scaling this Hamiltonian by the given scalar.
    */
    return (*this) * std::complex<double>(scalar, 0.0);
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator+=(const MatrixFreeHamiltonian& other) {
    /*
    Add another Hamiltonian to this one in-place.

    Args:
        other: The Hamiltonian to be added to this one.

    Returns:
        A reference to the resulting Hamiltonian after addition.
    */
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
            op_map[id] = int(operators.size()) - 1;
        }
    }
    return *this;
}

std::ostream& operator<<(std::ostream& os, const MatrixFreeHamiltonian& hamiltonian) {
    /*
    Output stream operator for MatrixFreeHamiltonian.
    Used like std::cout << hamiltonian; to print the Hamiltonian in a human-readable format.

    Args:
        os: The output stream to which the Hamiltonian will be written.
        hamiltonian: The Hamiltonian to be written to the output stream.

    Returns:
        A reference to the output stream after writing the Hamiltonian.
    */
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
    /*
    Add a term to the Hamiltonian with a given coefficient and operator.

    Args:
        coeff: The complex coefficient for the term being added.
        op: The MatrixFreeOperator that defines the term being added to the Hamiltonian.
    */
    operators.push_back({coeff, std::vector<MatrixFreeOperator>{op}});
}
void MatrixFreeHamiltonian::add(const std::complex<double>& coeff, const std::vector<MatrixFreeOperator>& ops) {
    /*
    Add a term to the Hamiltonian with a given coefficient and a sequence of operators.

    Args:
        coeff: The complex coefficient for the term being added.
        ops: A vector of MatrixFreeOperators that define the term being added to the Hamiltonian. The operators will be applied in sequence.
    */
    operators.push_back({coeff, ops});
}