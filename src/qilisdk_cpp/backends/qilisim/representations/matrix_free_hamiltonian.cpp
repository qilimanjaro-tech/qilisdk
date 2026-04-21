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
#include "../../../libs/pybind.h"
#include "../utils/matrix_utils.h"

// GCOV_EXCL_BR_START

void MatrixFreeHamiltonian::apply(const DenseMatrix& input_state, MatrixFreeApplicationType application_type, DenseMatrix& output_state) const {
    /*
    Applies the matrix-free Hamiltonian to the given input state and writes the 
    result to a separate output state.

    Args:
        input_state: The state to which the Hamiltonian will be applied.
        application_type: The type of application (Left, Right, or LeftAndRight).
        output_state: The state where the result will be stored.
    */
    struct Term {
        std::complex<double> base_phase;     // coefficient * (-i)^n_Y, precomputed
        std::complex<double> base_phase_neg; // -coefficient * (-i)^n_Y, precomputed
        long flip_mask;                      // XOR of all X and Y qubit masks
        long sign_mask;                      // OR of all Y and Z qubit masks (popcount parity = sign flip)
    };
    
    // Precompute things
    std::vector<Term> terms;
    terms.reserve(operators.size());
    const int num_qubits = static_cast<int>(std::log2(input_state.rows()));
    for (const auto& [coefficient, ops] : operators) {
        long flip_mask = 0;
        long sign_mask = 0;
        int n_y = 0;
        for (const auto& op : ops) {
            long mask = 1LL << (num_qubits - 1 - op.get_target_qubits()[0]);
            if (op.get_name() == "X") {
                flip_mask ^= mask;
            } else if (op.get_name() == "Z") {
                sign_mask |= mask;
            } else if (op.get_name() == "Y") {
                flip_mask ^= mask;
                sign_mask |= mask;
                ++n_y;
            } else {
                throw std::invalid_argument("Unsupported operator in Hamiltonian: " + op.get_name());
            }
        }
        static const std::complex<double> neg_i_powers[4] = {{1,0},{0,-1},{-1,0},{0,1}};
        std::complex<double> base_phase = coefficient * neg_i_powers[n_y & 3];
        std::complex<double> base_phase_neg = -base_phase;
        terms.push_back({base_phase, base_phase_neg, flip_mask, sign_mask});
    }

    // Cache some pointers
    const std::complex<double>* in_ptr = input_state.data();
    std::complex<double>* out_ptr = output_state.data();
    const Term* t_begin = terms.data();
    const Term* t_end   = t_begin + terms.size();

    // If it's a statevector
    if (input_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < output_state.size(); ++i) {
            std::complex<double> coeff = 0.0;
            for (const Term* t = t_begin; t != t_end; ++t) {
                long index = i ^ t->flip_mask;
                bool neg = __builtin_popcountll((long long)i & t->sign_mask) & 1;
                coeff += (neg ? t->base_phase_neg : t->base_phase) * in_ptr[index];
            }
            out_ptr[i] = coeff;
        }
    } else {
        long N = output_state.rows();
        if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long i = 0; i < N; ++i) {
                for (const Term* t = t_begin; t != t_end; ++t) {
                    long index = i ^ t->flip_mask;
                    bool neg = __builtin_popcountll((long long)i & t->sign_mask) & 1;
                    output_state.row(i) += (neg ? t->base_phase_neg : t->base_phase) * input_state.row(index);
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long j = 0; j < N; ++j) {
                for (const Term* t = t_begin; t != t_end; ++t) {
                    long index = j ^ t->flip_mask;
                    bool neg = __builtin_popcountll((long long)j & t->sign_mask) & 1;
                    output_state.col(j) += std::conj(neg ? t->base_phase_neg : t->base_phase) * input_state.col(index);
                }
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < N; ++i) {
                    for (const Term* t = t_begin; t != t_end; ++t) {
                        long index = i ^ t->flip_mask;
                        bool neg = __builtin_popcountll((long long)i & t->sign_mask) & 1;
                        output_state.row(i) += (neg ? t->base_phase_neg : t->base_phase) * input_state.row(index);
                    }
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long j = 0; j < N; ++j) {
                    for (const Term* t = t_begin; t != t_end; ++t) {
                        long index = j ^ t->flip_mask;
                        bool neg = __builtin_popcountll((long long)j & t->sign_mask) & 1;
                        output_state.col(j) += std::conj(neg ? t->base_phase_neg : t->base_phase) * input_state.col(index);
                    }
                }
            }
        }
    }


}

double MatrixFreeHamiltonian::expectation_value(const DenseMatrix& state) const {
    /*
    Calculate the expectation value of the Hamiltonian with respect to a given state.

    Args:
        state: The state for which the expectation value will be calculated.

    Raises:
        std::invalid_argument: If any operator in the Hamiltonian acts on a qubit that is out of bounds for the given state.
    */
    int num_qubits_in_state = static_cast<int>(std::log2(state.rows()));
    for (const auto& [coefficient, ops] : operators) {
        for (const auto& op : ops) {
            int max_qubit = -1;
            for (int target : op.get_target_qubits()) {
                max_qubit = std::max(max_qubit, target);
            }
            for (int control : op.get_control_qubits()) {
                max_qubit = std::max(max_qubit, control);
            }
            if (max_qubit >= num_qubits_in_state) {
                throw py::value_error("Operator in Hamiltonian acts on a qubit that is out of bounds for the given state.");
            }
        }
    }
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
    for (size_t i = 0; i < operators.size(); ++i) {
        std::string id;
        for (const auto& op : operators[i].second) {
            id += op.get_id() + "|";
        }
        op_map[id] = int(i);
    }
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

bool MatrixFreeHamiltonian::operator==(const MatrixFreeHamiltonian& other) const {
    /*
    Equality operator for MatrixFreeHamiltonian. Two Hamiltonians are considered equal if they have the same terms with the same coefficients.

    Args:
        other: The Hamiltonian to compare with this one.

    Returns:
        True if the Hamiltonians are equal, false otherwise.
    */
    if (operators.size() != other.operators.size()) {
        return false;
    }
    MatrixFreeHamiltonian sorted_this = *this;
    MatrixFreeHamiltonian sorted_other = other;
    auto sort_key = [](const std::pair<std::complex<double>, std::vector<MatrixFreeOperator>>& term) {
        std::string key;
        for (const auto& op : term.second) {
            key += op.get_id() + "|";
        }
        return key;
    };
    std::sort(sorted_this.operators.begin(), sorted_this.operators.end(), [&](const auto& a, const auto& b) { return sort_key(a) < sort_key(b); });
    std::sort(sorted_other.operators.begin(), sorted_other.operators.end(), [&](const auto& a, const auto& b) { return sort_key(a) < sort_key(b); });
    for (size_t i = 0; i < sorted_this.operators.size(); ++i) {
        if (sorted_this.operators[i].first != sorted_other.operators[i].first) {
            return false;
        }
        const auto& ops1 = sorted_this.operators[i].second;
        const auto& ops2 = sorted_other.operators[i].second;
        if (ops1.size() != ops2.size()) {
            return false;
        }
        for (size_t j = 0; j < ops1.size(); ++j) {
            if (!(ops1[j] == ops2[j])) {
                return false;
            }
        }
    }
    return true;
}

// GCOV_EXCL_BR_STOP