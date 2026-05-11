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
    using OpsMap  = std::unordered_map<PauliString, cdouble, PauliString::HashFunction>;
 
    if (num_qubits <= 0)
        throw std::runtime_error("ExponentialAnsatz: num_qubits must be positive");
 
    // (-i)^n for n = 0,1,2,3 — phase factor from Y = iXZ
    static const cdouble neg_i_powers[4] = {{1,0},{0,-1},{-1,0},{0,1}};
 
    // Own copies — get_operators() returns by value so never take a const ref to it
    const OpsMap ansatz_ops = terms.get_operators();
    const OpsMap obs_ops    = observable.get_operators();
 
    // Flatten ansatz into parallel arrays for fast inner-loop access
    const int p = static_cast<int>(ansatz_ops.size());
    std::vector<long>    z_bits(p);
    std::vector<cdouble> a_k(p);
    {
        int k = 0;
        for (const auto& [ps, coeff] : ansatz_ops) {
            if (static_cast<int>(ps.z_mask.size()) != num_qubits)
                throw std::runtime_error("Ansatz PauliString size mismatch with num_qubits");
            a_k[k] = coeff;
            long bits = 0;
            for (int i = 0; i < num_qubits; ++i)
                if (ps.z_mask[i]) bits |= (1LL << (num_qubits - 1 - i));
            z_bits[k] = bits;
            ++k;
        }
    }
 
    // -----------------------------------------------------------------------
    // Precompute observable terms
    //
    // For each Pauli term O_t = coeff * ∏ σ^{x/y/z}_i:
    //   base_phase = coeff * (-i)^n_Y          (overall scalar phase)
    //   flip_mask  = bits flipped by X and Y   (x' = x ^ flip_mask)
    //   sign_mask  = bits giving a -1 from Z/Y (sign = (-1)^popcount(x & sign_mask))
    //   flips_Pk   = for each ansatz term k, does flip_mask anticommute with P_k?
    //                i.e. does x' = x ^ flip_mask change the sign of P_k?
    // -----------------------------------------------------------------------
    struct ObsTerm {
        cdouble           base_phase;
        long              flip_mask;
        long              sign_mask;
        std::vector<bool> flips_Pk;   // size p
    };
 
    std::vector<ObsTerm> o_terms;
    o_terms.reserve(obs_ops.size());
    for (const auto& [ps, coeff] : obs_ops) {
        if (static_cast<int>(ps.x_mask.size()) != num_qubits ||
            static_cast<int>(ps.z_mask.size()) != num_qubits)
            throw std::runtime_error("Observable PauliString size mismatch with num_qubits");
 
        long flip_mask = 0, sign_mask = 0;
        int  n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            long bit   = 1LL << (num_qubits - 1 - i);
            bool has_x = ps.x_mask[i];
            bool has_z = ps.z_mask[i];
            if      ( has_x && !has_z) { flip_mask ^= bit; }
            else if (!has_x &&  has_z) { sign_mask |= bit; }
            else if ( has_x &&  has_z) { flip_mask ^= bit; sign_mask |= bit; ++n_y; }
        }
        std::vector<bool> flips(p);
        for (int k = 0; k < p; ++k)
            flips[k] = (__builtin_popcountll((long long)(flip_mask & z_bits[k])) & 1) != 0;
 
        o_terms.push_back({coeff * neg_i_powers[n_y & 3], flip_mask, sign_mask, std::move(flips)});
    }
 
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
 
    // log |Ψ(x)|² = 2 Re ∑_k a_k P_k(x),   P_k(x) = (-1)^popcount(x & z_bits[k])
    auto log_prob = [&](long x) -> double {
        double lp = 0.0;
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            lp += 2.0 * a_k[k].real() * (neg ? -1.0 : 1.0);
        }
        return lp;
    };
 
    // log(Ψ(x')/Ψ(x)) for x' = x ^ ot.flip_mask
    // = ∑_k a_k [P_k(x') - P_k(x)]
    // If flips_Pk[k]: P_k(x') = -P_k(x), so contribution = -2 a_k P_k(x)
    auto compute_log_ratio = [&](long x, const ObsTerm& ot) -> cdouble {
        cdouble lr = 0.0;
        for (int k = 0; k < p; ++k) {
            if (!ot.flips_Pk[k]) continue;
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            lr += a_k[k] * (neg ? cdouble(2.0, 0.0) : cdouble(-2.0, 0.0));
        }
        return lr;
    };
 
    // -----------------------------------------------------------------------
    // Metropolis-Hastings
    //
    // Mixes single-qubit flips (90%) with two-qubit flips (10%) to ensure
    // ergodicity across parity sectors and avoid the parity-orbit trap at a_k=0.
    // Initial state built bit-by-bit to avoid shift overflow for large num_qubits.
    // -----------------------------------------------------------------------
    const int N_s      = 1000;
    const int n_warmup = 500;
 
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int>     rand_qubit(0, num_qubits - 1);
    std::uniform_int_distribution<int>     rand_bit(0, 1);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);
 
    auto propose = [&](long x) -> long {
        if (num_qubits > 1 && rand01(rng) < 0.1) {
            int i = rand_qubit(rng), j;
            do { j = rand_qubit(rng); } while (j == i);
            return x ^ (1LL << (num_qubits - 1 - i))
                     ^ (1LL << (num_qubits - 1 - j));
        }
        return x ^ (1LL << (num_qubits - 1 - rand_qubit(rng)));
    };
 
    // Build initial state bit-by-bit — safe for any num_qubits, no shift overflow
    long x = 0;
    for (int i = 0; i < num_qubits; ++i)
        if (rand_bit(rng)) x |= (1LL << (num_qubits - 1 - i));
    double lp = log_prob(x);
 
    for (int s = 0; s < n_warmup * num_qubits; ++s) {
        long   xn  = propose(x);
        double lpn = log_prob(xn);
        if (std::log(rand01(rng)) < lpn - lp) { x = xn; lp = lpn; }
    }
 
    // -----------------------------------------------------------------------
    // Accumulate local energy
    // E_loc(x) = ∑_t base_phase_t * (-1)^popcount(x & sign_mask_t) * exp(log_ratio_t(x))
    // -----------------------------------------------------------------------
    cdouble total = 0.0;
    for (int s = 0; s < N_s; ++s) {
        long   xn  = propose(x);
        double lpn = log_prob(xn);
        if (std::log(rand01(rng)) < lpn - lp) { x = xn; lp = lpn; }
 
        cdouble e_local = 0.0;
        for (const auto& ot : o_terms) {
            bool    neg   = __builtin_popcountll((long long)(x & ot.sign_mask)) & 1;
            cdouble phase = neg ? -ot.base_phase : ot.base_phase;
            e_local      += phase * std::exp(compute_log_ratio(x, ot));
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