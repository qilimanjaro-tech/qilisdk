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
#include "exponential_ansatz.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

// GCOV_EXCL_BR_START

ExponentialAnsatz::ExponentialAnsatz(int num_qubits, int max_terms) : num_qubits(num_qubits), terms(num_qubits) {
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

std::vector<long> ExponentialAnsatz::build_z_bits() const {
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, std::complex<double>>> terms_vec(ops.begin(), ops.end());
    std::vector<long> z_bits(p);
    for (int k = 0; k < p; ++k) {
        long bits = 0;
        const auto& ps = terms_vec[k].first;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.z_mask[i]) bits |= (1LL << (num_qubits - 1 - i));
        }
        z_bits[k] = bits;
    }
    return z_bits;
}

SampleSet ExponentialAnsatz::draw_samples() const {
    return draw_samples(shots, shots_warmup);
}

SampleSet ExponentialAnsatz::draw_samples(int N_s, int n_warmup) const {
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, std::complex<double>>> terms_vec(ops.begin(), ops.end());
    std::vector<long> z_bits = build_z_bits();

    auto log_prob = [&](long x) -> double {
        double lp = 0.0;
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            lp += 2.0 * terms_vec[k].second.real() * (neg ? -1.0 : 1.0);
        }
        return lp;
    };

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> rand_qubit(0, num_qubits - 1);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    long x = std::uniform_int_distribution<long>(0, (1LL << num_qubits) - 1)(rng);
    double lp = log_prob(x);

    for (int s = 0; s < n_warmup * num_qubits; ++s) {
        int i = rand_qubit(rng);
        long x_new = x ^ (1LL << (num_qubits - 1 - i));
        double lp_new = log_prob(x_new);
        if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }
    }

    SampleSet result;
    result.configs.resize(N_s);
    result.O_mat.resize(N_s, p);

    for (int s = 0; s < N_s; ++s) {
        int i = rand_qubit(rng);
        long x_new = x ^ (1LL << (num_qubits - 1 - i));
        double lp_new = log_prob(x_new);
        if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }

        result.configs[s] = x;
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            result.O_mat(s, k) = std::complex<double>(neg ? -1.0 : 1.0, 0.0);
        }
    }

    return result;
}

Eigen::VectorXcd ExponentialAnsatz::local_energy(const SampleSet& samples, const MatrixFreeHamiltonian& H) const {
    static const std::complex<double> neg_i_powers[4] = {{1,0},{0,-1},{-1,0},{0,1}};

    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, std::complex<double>>> terms_vec(ops.begin(), ops.end());
    std::vector<long> z_bits = build_z_bits();
    const int N_s = static_cast<int>(samples.configs.size());

    struct HTerm {
        std::complex<double> base_phase;
        long flip_mask;
        long sign_mask;
        std::vector<bool> flips_Pk;
    };
    const auto& h_ops = H.get_operators();
    std::vector<HTerm> h_terms;
    h_terms.reserve(h_ops.size());
    for (const auto& [ps, coeff] : h_ops) {
        long flip_mask = 0, sign_mask = 0;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            long mask = 1LL << (num_qubits - 1 - i);
            if ( ps.x_mask[i] && !ps.z_mask[i]) { flip_mask ^= mask; }
            else if (!ps.x_mask[i] &&  ps.z_mask[i]) { sign_mask |= mask; }
            else if ( ps.x_mask[i] &&  ps.z_mask[i]) { flip_mask ^= mask; sign_mask |= mask; ++n_y; }
        }
        std::complex<double> base_phase = coeff * neg_i_powers[n_y & 3];
        std::vector<bool> flips(p);
        for (int k = 0; k < p; ++k) {
            flips[k] = (__builtin_popcountll((long long)(flip_mask & z_bits[k])) & 1) != 0;
        }
        h_terms.push_back({base_phase, flip_mask, sign_mask, std::move(flips)});
    }

    Eigen::VectorXcd El(N_s);
    for (int s = 0; s < N_s; ++s) {
        long x = samples.configs[s];
        std::complex<double> el = 0.0;
        for (const auto& ht : h_terms) {
            bool neg_sign = __builtin_popcountll((long long)(x & ht.sign_mask)) & 1;
            std::complex<double> h_elem = neg_sign ? -ht.base_phase : ht.base_phase;
            std::complex<double> log_ratio = 0.0;
            for (int k = 0; k < p; ++k) {
                if (ht.flips_Pk[k]) {
                    bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
                    log_ratio -= 2.0 * terms_vec[k].second * std::complex<double>(neg ? -1.0 : 1.0, 0.0);
                }
            }
            el += h_elem * std::exp(log_ratio);
        }
        El(s) = el;
    }

    return El;
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
    SampleSet samples = draw_samples(shots, shots_warmup);
    return local_energy(samples, observable).mean().real();
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

DenseMatrix ExponentialAnsatz::to_dense() const {
    /*
    Convert the ExponentialAnsatz to a dense matrix representation.

    Returns:
        DenseMatrix: The dense matrix representation of the state represented by this ansatz.
    */
    int dim = 1 << num_qubits;
    DenseMatrix total_op = DenseMatrix::Zero(dim, dim);
    
    // Build Pauli ops
    DenseMatrix pauli_x(2, 2), pauli_y(2, 2), pauli_z(2, 2);
    pauli_x << std::complex<double>(0), std::complex<double>(1), std::complex<double>(1), std::complex<double>(0);
    pauli_y << std::complex<double>(0), std::complex<double>(0,-1), std::complex<double>(0,1), std::complex<double>(0);
    pauli_z << std::complex<double>(1), std::complex<double>(0), std::complex<double>(0), std::complex<double>(-1);

    for (const auto& [ps, coeff] : terms.get_operators()) {
        DenseMatrix op(1, 1);
        op(0, 0) = 1.0;
        for (int i = 0; i < num_qubits; ++i) {
            DenseMatrix single = DenseMatrix::Identity(2, 2);
            if      (ps.x_mask[i] && !ps.z_mask[i]) single = pauli_x;
            else if (!ps.x_mask[i] && ps.z_mask[i]) single = pauli_z;
            else if (ps.x_mask[i] &&  ps.z_mask[i]) single = pauli_y;
            op = Eigen::kroneckerProduct(op, single).eval();
        }
        total_op += coeff * op;
    }

    DenseMatrix plus_state = DenseMatrix::Ones(dim, 1) / std::sqrt(static_cast<double>(dim));
    DenseMatrix result = (total_op.exp() * plus_state).eval();
    result /= result.norm();
    return result;
}

// GCOV_EXCL_BR_STOP