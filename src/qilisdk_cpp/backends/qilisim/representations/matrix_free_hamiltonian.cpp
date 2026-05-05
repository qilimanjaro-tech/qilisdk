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
        std::complex<double> base_phase;      // coefficient * (-i)^n_Y, precomputed
        std::complex<double> base_phase_neg;  // -coefficient * (-i)^n_Y, precomputed
        long flip_mask;                       // XOR of all X and Y qubit masks
        long sign_mask;                       // OR of all Y and Z qubit masks (popcount parity = sign flip)
    };

    // Precompute things
    std::vector<Term> terms;
    terms.reserve(operators.size());
    const int num_qubits = static_cast<int>(std::log2(input_state.rows()));
    for (const auto& [pauli, coefficient] : operators) {
        long flip_mask = 0;
        long sign_mask = 0;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            long mask = 1LL << (num_qubits - 1 - i);
            if (pauli.x_mask[i] && !pauli.z_mask[i]) {  // X
                flip_mask ^= mask;
            } else if (!pauli.x_mask[i] && pauli.z_mask[i]) {  // Z
                sign_mask |= mask;
            } else if (pauli.x_mask[i] && pauli.z_mask[i]) {  // Y
                flip_mask ^= mask;
                sign_mask |= mask;
                ++n_y;
            }
        }
        static const std::complex<double> neg_i_powers[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
        std::complex<double> base_phase = coefficient * neg_i_powers[n_y & 3];
        std::complex<double> base_phase_neg = -base_phase;
        terms.push_back({base_phase, base_phase_neg, flip_mask, sign_mask});
    }

    // Make sure output_state has the right shape and is initialized to zero
    output_state.resizeLike(input_state);
    output_state.setZero();

    // Cache some pointers
    const std::complex<double>* in_ptr = input_state.data();
    std::complex<double>* out_ptr = output_state.data();
    const Term* t_begin = terms.data();
    const Term* t_end = t_begin + terms.size();

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
#pragma omp parallel for schedule(static)
#endif
            for (long i = 0; i < N; ++i) {
                for (const Term* t = t_begin; t != t_end; ++t) {
                    long index = i ^ t->flip_mask;
                    bool neg = __builtin_popcountll((long long)i & t->sign_mask) & 1;
                    output_state.row(i) += (neg ? t->base_phase_neg : t->base_phase) * input_state.row(index);
                }
            }
            DenseMatrix hr_temp = output_state;
            output_state.setZero();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long j = 0; j < N; ++j) {
                for (const Term* t = t_begin; t != t_end; ++t) {
                    long index = j ^ t->flip_mask;
                    bool neg = __builtin_popcountll((long long)j & t->sign_mask) & 1;
                    output_state.col(j) += std::conj(neg ? t->base_phase_neg : t->base_phase) * hr_temp.col(index);
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
    int num_qubits_in_hamiltonian = get_nqubits();
    if (num_qubits_in_hamiltonian > num_qubits_in_state) {
        throw std::invalid_argument("Hamiltonian acts on more qubits than the state has.");
    }

    // Do <state|H|state> by applying H to state and then taking the dot product with state.
    m_temp_state.resizeLike(state);
    m_new_state.resizeLike(state);
    apply(state, MatrixFreeApplicationType::Left, m_temp_state);
    double exp_val = std::real(dot(state, m_temp_state));

    return exp_val;

}

#include <iostream>

double MatrixFreeHamiltonian::expectation_value(const MatrixFreeHamiltonian& other) const {
    /*
    Calculate the expectation value of this Hamiltonian with respect to another Hamiltonian.
    The assumption is that the state is this*|+>.

    Args:
        other: The Hamiltonian with respect to which the expectation value will be calculated.

    Returns:
        The expectation value of this Hamiltonian with respect to the other Hamiltonian.
    */

    // need to calculate <+| H_this^dag H_other H_this |+>
    int n_qubits = get_nqubits();

    // first we calculate H_this^dag H_other H_this:
    MatrixFreeHamiltonian temp = (*this).conjugate() * other * (*this);

    std::cout << "State Hamiltonian: " << *this << std::endl;
    std::cout << "Other Hamiltonian: " << other << std::endl;
    std::cout << "Temp Hamiltonian: " << temp << std::endl;

    // Since <+|P|+> is 1 if P is identity or X and 0 otherwise, we just need to sum the coefficients
    std::complex<double> exp_val = 0.0;
    for (const auto& [pauli, coefficient] : temp.operators) {
        bool is_identity_or_x = true;
        for (int i = 0; i < n_qubits; ++i) {
            if (pauli.z_mask[i]) {
                is_identity_or_x = false;
                break;
            }
        }
        if (is_identity_or_x) {
            exp_val += coefficient;
        }
    }
    return std::real(exp_val);
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator*=(const std::complex<double>& scalar) {
    /*
    Scale in-place by a complex scalar.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.

    Returns:
        A reference to the scaled Hamiltonian.
    */
    for (auto& [pauli, coefficient] : operators) {
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
    MatrixFreeHamiltonian result = *this;
    result *= scalar;
    return result;
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator+=(const MatrixFreeHamiltonian& other) {
    /*
    Add another Hamiltonian to this one in-place.

    Args:
        other: The Hamiltonian to be added to this one.

    Returns:
        A reference to the resulting Hamiltonian after addition.
    */
    for (const auto& [pauli, coefficient] : other.operators) {
        operators[pauli] += coefficient;
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
    for (const auto& [pauli, coefficient] : hamiltonian.operators) {
        os << coefficient << " * " << pauli;
        if (count < int(hamiltonian.operators.size()) - 1) {
            os << " + ";
        }
        count++;
    }
    return os;
}

void MatrixFreeHamiltonian::add(const std::complex<double>& coeff, const PauliString& op) {
    /*
    Add a term to the Hamiltonian with a given coefficient and operator.

    Args:
        coeff: The complex coefficient for the term being added.
        op: The PauliString that defines the term being added to the Hamiltonian.
    */
    operators[op] += coeff;
}

void MatrixFreeHamiltonian::add(const std::complex<double>& coeff, const std::vector<MatrixFreeOperator>& ops) {
    /*
    Add a term to the Hamiltonian with a given coefficient and a vector of operators.

    Args:
        coeff: The complex coefficient for the term being added.
        ops: The vector of MatrixFreeOperator that defines the term being added to the Hamiltonian.
    */
    PauliString ps(get_nqubits());  // start with identity
    for (const auto& op : ops) {
        for (int target : op.get_target_qubits()) {
            if (op.get_name() == "X") {
                ps.x_mask[target] = !ps.x_mask[target];
            } else if (op.get_name() == "Z") {
                ps.z_mask[target] = !ps.z_mask[target];
            } else if (op.get_name() == "Y") {
                ps.x_mask[target] = !ps.x_mask[target];
                ps.z_mask[target] = !ps.z_mask[target];
            }
        }
    }
    operators[ps] += coeff;
}

bool MatrixFreeHamiltonian::operator==(const MatrixFreeHamiltonian& other) const {
    /*
    Equality operator for MatrixFreeHamiltonian. Two Hamiltonians are considered equal if they have the same terms with the same coefficients.

    Args:
        other: The Hamiltonian to compare with this one.

    Returns:
        True if the Hamiltonians are equal, false otherwise.
    */
    return operators == other.operators;
}

MatrixFreeHamiltonian operator*(const std::complex<double>& scalar, const MatrixFreeHamiltonian& hamiltonian) {
    /*
    Scale a Hamiltonian by a complex scalar from the left.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.
        hamiltonian: The Hamiltonian to be scaled.

    Returns:
        A new Hamiltonian that is the result of scaling the given Hamiltonian by the given scalar.
    */
    return hamiltonian * scalar;
}

std::pair<PauliString, std::complex<double>> _multiply_pauli_strings(const PauliString& a, const PauliString& b) {
    /*
    Multiply two Pauli strings together, returning the resulting Pauli string and the phase factor.

    Args:
        a: The first Pauli string to be multiplied.
        b: The second Pauli string to be multiplied.

    Returns:
        A pair consisting of the resulting Pauli string from the multiplication and the complex phase factor.
    */

    const size_t n = a.x_mask.size();
    PauliString result(n);

    // XOR gives the result masks directly (same as adding Paulis mod 2)
    // X*X=I, Z*Z=I, X*Z or Z*X = Y (both bits set), etc.
    int phase_exp = 0; // will be taken mod 4; maps to i^phase_exp
    for (size_t q = 0; q < n; ++q) {
        const int ax = a.x_mask[q], az = a.z_mask[q];
        const int bx = b.x_mask[q], bz = b.z_mask[q];
        result.x_mask[q] = ax ^ bx;
        result.z_mask[q] = az ^ bz;
        phase_exp += 2 * (ax & bz);
        phase_exp -= 2 * (az & bx);
    }

    // Normalize to [0, 4)
    phase_exp = ((phase_exp % 4) + 4) % 4;

    // Map i^k to complex
    static const std::complex<double> phase_table[4] = {
        {1.0, 0.0},
        {0.0, 1.0},
        {-1.0, 0.0},
        {0.0, -1.0}
    };

    return {result, phase_table[phase_exp]};
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const MatrixFreeHamiltonian& other) const {
    /*
    Multiply two Hamiltonians together.

    Args:
        other: The Hamiltonian to multiply with this one.

    Returns:
        A new Hamiltonian that is the result of multiplying this Hamiltonian with the other Hamiltonian.
    */
    MatrixFreeHamiltonian result;

    for (const auto& [ps_a, coeff_a] : operators) {
        for (const auto& [ps_b, coeff_b] : other.operators) {
            auto [ps_result, phase] = _multiply_pauli_strings(ps_a, ps_b);
            std::complex<double> new_coeff = coeff_a * coeff_b * phase;

            auto it = result.operators.find(ps_result);
            if (it != result.operators.end()) {
                it->second += new_coeff;
            } else {
                result.operators[ps_result] = new_coeff;
            }
        }
    }

    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator+(const MatrixFreeHamiltonian& other) const {
    /*
    Add this Hamiltonian to another Hamiltonian.

    Args:
        other: The Hamiltonian to add to this one.

    Returns:
        A new Hamiltonian that is the result of adding this Hamiltonian to the other Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    result += other;
    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator-(const MatrixFreeHamiltonian& other) const {
    /*
    Subtract another Hamiltonian from this Hamiltonian.

    Args:
        other: The Hamiltonian to subtract from this one.

    Returns:
        A new Hamiltonian that is the result of subtracting the other Hamiltonian from this Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    result += other * std::complex<double>(-1.0, 0.0);
    return result;
}

void MatrixFreeHamiltonian::prune(double threshold, int max_terms) {
    /*
    Prune the Hamiltonian by removing terms with coefficients below a certain threshold and limiting the total number of terms.

    Args:
        threshold: The minimum absolute value of coefficients for terms to be kept in the Hamiltonian. Terms with coefficients below this value will be removed.
        max_terms: The maximum number of terms to keep in the Hamiltonian. If there are more terms than this after applying the threshold, only the terms with the largest coefficients (in absolute value) will be kept.

    Returns:
        A reference to the pruned Hamiltonian.
    */
    // Create a vector of terms and sort by absolute value of coefficients
    std::vector<std::pair<PauliString, std::complex<double>>> term_vector(operators.begin(), operators.end());
    std::sort(term_vector.begin(), term_vector.end(), [](const auto& a, const auto& b) {
        return std::abs(a.second) > std::abs(b.second);
    });
    if (term_vector.size() > static_cast<size_t>(max_terms)) {
        term_vector.resize(max_terms);
    }
    // Rebuild the operators map from the pruned vector
    operators.clear();
    for (const auto& [ps, coeff] : term_vector) {
        if (std::abs(coeff) >= threshold) {
            operators[ps] = coeff;
        }
    }
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::conjugate() const {
    /*
    Conjugate the Hamiltonian by taking the complex conjugate of all coefficients.

    Returns:
        A new Hamiltonian that is the complex conjugate of this Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    for (auto& [ps, coeff] : result.operators) {
        coeff = std::conj(coeff);
    }
    return result;
}


// GCOV_EXCL_BR_STOP