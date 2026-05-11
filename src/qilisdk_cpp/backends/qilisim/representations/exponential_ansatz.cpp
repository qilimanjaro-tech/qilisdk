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

#include "exponential_ansatz.h"
#include <algorithm>
#include <cmath>
#include <random>

// GCOV_EXCL_BR_START

ExponentialAnsatz::ExponentialAnsatz(int num_qubits, int max_terms) {
    /*
    Construct an ExponentialAnsatz with the given number of qubits and maximum number of terms.

    The format of this ansatz is exp(sum_i c_i P_i) |+> where P_i are Pauli strings and c_i are coefficients.
    For now we restrict the P_i to be Z operators.

    Args:
        num_qubits (int): The number of qubits in the system.
        max_terms (int): The maximum number of terms to keep in the ansatz.

    Returns:
        ExponentialAnsatz: The constructed ExponentialAnsatz object.
    */

    // Set the number of qubits
    this->num_qubits = num_qubits;

    // Keep adding terms until we reach the maximum number
    int terms_so_far = 0;
    
    // Add single body terms
    for (int i = 0; i < num_qubits; ++i) {
        if (terms_so_far >= max_terms) break;
        PauliString ps(num_qubits, 'Z', i);
        terms.add(0.0, ps);
        terms_so_far++;
    }

    // Add two body terms
    for (int i = 0; i < num_qubits; ++i) {
        for (int j = i + 1; j < num_qubits; ++j) {
            if (terms_so_far >= max_terms) break;
            PauliString ps(num_qubits);
            ps.z_mask[i] = true;
            ps.z_mask[j] = true;
            terms.add(0.0, ps);
            terms_so_far++;
        }
        if (terms_so_far >= max_terms) break;
    }

    // Add three body terms
    for (int i = 0; i < num_qubits; ++i) {
        for (int j = i + 1; j < num_qubits; ++j) {
            for (int k = j + 1; k < num_qubits; ++k) {
                if (terms_so_far >= max_terms) break;
                PauliString ps(num_qubits);
                ps.z_mask[i] = true;
                ps.z_mask[j] = true;
                ps.z_mask[k] = true;
                terms.add(0.0, ps);
                terms_so_far++;
            }
            if (terms_so_far >= max_terms) break;
        }
        if (terms_so_far >= max_terms) break;
    }

    // Add four body terms
    for (int i = 0; i < num_qubits; ++i) {
        for (int j = i + 1; j < num_qubits; ++j) {
            for (int k = j + 1; k < num_qubits; ++k) {
                for (int l = k + 1; l < num_qubits; ++l) {
                    if (terms_so_far >= max_terms) break;
                    PauliString ps(num_qubits);
                    ps.z_mask[i] = true;
                    ps.z_mask[j] = true;
                    ps.z_mask[k] = true;
                    ps.z_mask[l] = true;
                    terms.add(0.0, ps);
                    terms_so_far++;
                }
                if (terms_so_far >= max_terms) break;
            }
            if (terms_so_far >= max_terms) break;
        }
        if (terms_so_far >= max_terms) break;
    }

}

std::ostream& operator<<(std::ostream& os, const ExponentialAnsatz& ansatz) {
    /*
    Output the ExponentialAnsatz in a human-readable format.

    Args:
        os (std::ostream&): The output stream to write to.
        ansatz (const ExponentialAnsatz&): The ExponentialAnsatz to output.

    Returns:
        std::ostream&: The output stream after writing the ansatz.
    */
    os << ansatz.get_terms();
    return os;
}

double ExponentialAnsatz::expectation_value(const MatrixFreeHamiltonian& observable) const {
    /*
    Compute the expectation value of the given observable with respect to the state represented by this ansatz.

    Uses variational Monte Carlo: sample x ~ |Ψ(x)|² via Metropolis-Hastings and average the
    local observable E_O(x) = ∑_{x'} O_{x,x'} Ψ(x')/Ψ(x). The result is real for Hermitian O.

    Args:
        observable (const MatrixFreeHamiltonian&): The observable to compute the expectation value of.

    Returns:
        double: The estimated expectation value <Ψ|O|Ψ>.
    */
    using cdouble = std::complex<double>;
    static const cdouble neg_i_powers[4] = {{1,0},{0,-1},{-1,0},{0,1}};

    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    if (p == 0) return 0.0;

    // Index the ansatz terms for stable iteration order
    std::vector<std::pair<PauliString, cdouble>> terms_vec(ops.begin(), ops.end());

    // Precompute bitmask for each Z-type Pauli string P_k
    // P_k(x) = (-1)^{popcount(x & z_bits[k])}
    std::vector<long> z_bits(p);
    for (int k = 0; k < p; ++k) {
        long bits = 0;
        const auto& ps = terms_vec[k].first;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.z_mask[i]) bits |= (1LL << (num_qubits - 1 - i));
        }
        z_bits[k] = bits;
    }

    // Precompute observable terms: flip_mask, sign_mask, base_phase, and which P_k
    // change sign under the flip (needed for the wavefunction ratio Ψ(x')/Ψ(x))
    struct OTerm {
        cdouble           base_phase;
        long              flip_mask;
        long              sign_mask;
        std::vector<bool> flips_Pk;
    };
    const auto& o_ops = observable.get_operators();
    std::vector<OTerm> o_terms;
    o_terms.reserve(o_ops.size());
    for (const auto& [ps, coeff] : o_ops) {
        long flip_mask = 0, sign_mask = 0;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            long mask = 1LL << (num_qubits - 1 - i);
            if ( ps.x_mask[i] && !ps.z_mask[i]) { flip_mask ^= mask; }
            else if (!ps.x_mask[i] &&  ps.z_mask[i]) { sign_mask |= mask; }
            else if ( ps.x_mask[i] &&  ps.z_mask[i]) { flip_mask ^= mask; sign_mask |= mask; ++n_y; }
        }
        cdouble base_phase = coeff * neg_i_powers[n_y & 3];
        std::vector<bool> flips(p);
        for (int k = 0; k < p; ++k) {
            flips[k] = (__builtin_popcountll((long long)(flip_mask & z_bits[k])) & 1) != 0;
        }
        o_terms.push_back({base_phase, flip_mask, sign_mask, std::move(flips)});
    }

    // Metropolis-Hastings: sample x ~ |Ψ(x)|² = exp(2 Re ∑_k a_k P_k(x))
    const int N_s      = 1000;
    const int n_warmup = 200;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int>     rand_qubit(0, num_qubits - 1);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    // Only Re(a_k) contributes since P_k(x) ∈ ℝ
    auto log_prob = [&](long x) -> double {
        double lp = 0.0;
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            lp += 2.0 * terms_vec[k].second.real() * (neg ? -1.0 : 1.0);
        }
        return lp;
    };

    long x = std::uniform_int_distribution<long>(0, (1LL << num_qubits) - 1)(rng);
    double lp = log_prob(x);

    for (int s = 0; s < n_warmup * num_qubits; ++s) {
        int    i      = rand_qubit(rng);
        long   x_new  = x ^ (1LL << (num_qubits - 1 - i));
        double lp_new = log_prob(x_new);
        if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }
    }

    // Accumulate local observable E_O(x) = ∑_{x'} O_{x,x'} Ψ(x')/Ψ(x)
    // log(Ψ(x')/Ψ(x)) = -2 ∑_{k: P_k changes} a_k P_k(x)
    cdouble total = 0.0;
    for (int s = 0; s < N_s; ++s) {
        for (int i = 0; i < num_qubits; ++i) {
            long   x_new  = x ^ (1LL << (num_qubits - 1 - i));
            double lp_new = log_prob(x_new);
            if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }
        }

        cdouble e_local = 0.0;
        for (const auto& ot : o_terms) {
            bool neg_sign = __builtin_popcountll((long long)(x & ot.sign_mask)) & 1;
            cdouble o_elem = neg_sign ? -ot.base_phase : ot.base_phase;

            cdouble log_ratio = 0.0;
            for (int k = 0; k < p; ++k) {
                if (ot.flips_Pk[k]) {
                    bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
                    log_ratio -= 2.0 * terms_vec[k].second * cdouble(neg ? -1.0 : 1.0, 0.0);
                }
            }
            e_local += o_elem * std::exp(log_ratio);
        }
        total += e_local;
    }

    return std::real(total) / static_cast<double>(N_s);
}

ExponentialAnsatz& ExponentialAnsatz::operator*=(const double& scalar) {
    /*
    In-place multiplication of the ExponentialAnsatz by a scalar.

    Args:
        scalar (const double&): The scalar to multiply by.

    Returns:
        ExponentialAnsatz&: The modified ExponentialAnsatz after multiplication.
    */
    terms *= scalar;
    return *this;
}

ExponentialAnsatz ExponentialAnsatz::operator*(const double& scalar) const {
    /*
    Multiplication of the ExponentialAnsatz by a scalar.

    Args:
        scalar (const double&): The scalar to multiply by.

    Returns:
        ExponentialAnsatz: A new ExponentialAnsatz that is the result of the multiplication.
    */
    ExponentialAnsatz result = *this;
    result *= scalar;
    return result;
}

ExponentialAnsatz ExponentialAnsatz::operator+(const ExponentialAnsatz& other) const {
    /*
    Addition of two ExponentialAnsatz objects.

    Args:
        other (const ExponentialAnsatz&): The other ExponentialAnsatz to add.

    Returns:
        ExponentialAnsatz: A new ExponentialAnsatz that is the result of the addition.
    */
    ExponentialAnsatz result = *this;
    result += other;
    return result;
}

ExponentialAnsatz& ExponentialAnsatz::operator+=(const ExponentialAnsatz& other) {
    /*
    In-place addition of another ExponentialAnsatz to this one.

    Args:
        other (const ExponentialAnsatz&): The other ExponentialAnsatz to add.

    Returns:
        ExponentialAnsatz&: The modified ExponentialAnsatz after addition.
    */
    terms += other.terms;
    return *this;
}

DenseMatrix expand_matrix_to_nqubits(const DenseMatrix& mat, int num_qubits) {
    /*
    Expand a 2x2 matrix to act on a specific qubit in an n-qubit system.

    Args:
        mat (const DenseMatrix&): The 2x2 matrix to expand.
        num_qubits (int): The total number of qubits in the system.

    Returns:
        DenseMatrix: The expanded matrix that acts on the specified qubit.
    */
    int dim = 1 << num_qubits;
    DenseMatrix result = DenseMatrix::Identity(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (((i ^ j) & ~(1 << (num_qubits - 1))) == 0) { // Only differ on the target qubit
                int bit_i = (i >> (num_qubits - 1)) & 1;
                int bit_j = (j >> (num_qubits - 1)) & 1;
                result(i, j) = mat(bit_i, bit_j);
            }
        }
    }
    return result;
}

DenseMatrix ExponentialAnsatz::to_dense() const {
    /*
    Convert the ExponentialAnsatz to a dense matrix representation.

    Returns:
        DenseMatrix: The dense matrix representation of the state represented by this ansatz.
    */
    // We have exp(sum_i c_i P_i) |+>, so form the inner total operator and then do the matrix exponential
    int dim = 1 << num_qubits;
    DenseMatrix total_op = DenseMatrix::Zero(dim, dim);
    DenseMatrix pauli_x = DenseMatrix::Zero(2, 2);
    pauli_x(0, 1) = 1.0;
    pauli_x(1, 0) = 1.0;
    DenseMatrix pauli_z = DenseMatrix::Zero(2, 2);
    pauli_z(0, 0) = 1.0;
    pauli_z(1, 1) = -1.0;
    for (const auto& [ps, coeff] : terms.get_operators()) {
        DenseMatrix op = DenseMatrix::Identity(dim, dim);
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.x_mask[i] && !ps.z_mask[i]) {
                op *= expand_matrix_to_nqubits(pauli_x, num_qubits);
            } else if (!ps.x_mask[i] && ps.z_mask[i]) {
                op *= expand_matrix_to_nqubits(pauli_z, num_qubits);
            } else if (ps.x_mask[i] && ps.z_mask[i]) {
                op *= expand_matrix_to_nqubits(pauli_x * pauli_z, num_qubits);
            }
        }
        total_op += coeff * op;
    }

    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(total_op);
    DenseMatrix exp_total_op = es.eigenvectors() * (es.eigenvalues().array().exp().matrix().asDiagonal()) * es.eigenvectors().adjoint();
    DenseMatrix plus_state = DenseMatrix::Ones(dim, 1) / std::sqrt(static_cast<double>(dim));
    return exp_total_op * plus_state;

}

// GCOV_EXCL_BR_STOP