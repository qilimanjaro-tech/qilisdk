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
#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

ExponentialAnsatz::ExponentialAnsatz(int num_qubits, int order, int shots, int warmups) {
    /*
    Construct an ExponentialAnsatz with the given number of qubits and maximum number of terms.

    The format of this ansatz is exp(sum_i c_i P_i) |+> where P_i are Pauli strings and c_i are coefficients.
    For now we restrict the P_i to be Z operators.

    Args:
        num_qubits (int): The number of qubits in the system.
        order (int): The maximum order of terms to include in the ansatz.
        shots (int): The number of shots to use for sampling.
        warmups (int): The number of warmup steps to use for sampling.

    Returns:
        ExponentialAnsatz: The constructed ExponentialAnsatz object.
    */

    // Set the internals
    this->num_qubits = num_qubits;
    this->order = order;
    this->shots = shots;
    this->warmups = warmups;

    // Add single body terms
    if (order >= 1) {
        for (int i = 0; i < num_qubits; ++i) {
            PauliString ps(num_qubits, 'Z', i);
            terms.add(0.0, ps);
        }
    }

    // Add two body terms
    if (order >= 2) {
        for (int i = 0; i < num_qubits; ++i) {
            for (int j = i + 1; j < num_qubits; ++j) {
                PauliString ps(num_qubits);
                ps.z_mask[i] = true;
                ps.z_mask[j] = true;
                terms.add(0.0, ps);
            }
        }
    }

    // Add three body terms
    if (order >= 3) {
        for (int i = 0; i < num_qubits; ++i) {
            for (int j = i + 1; j < num_qubits; ++j) {
                for (int k = j + 1; k < num_qubits; ++k) {
                    PauliString ps(num_qubits);
                    ps.z_mask[i] = true;
                    ps.z_mask[j] = true;
                    ps.z_mask[k] = true;
                    terms.add(0.0, ps);
                }
            }
        }
    }

    // Add four body terms
    if (order >= 4) {
        for (int i = 0; i < num_qubits; ++i) {
            for (int j = i + 1; j < num_qubits; ++j) {
                for (int k = j + 1; k < num_qubits; ++k) {
                    for (int l = k + 1; l < num_qubits; ++l) {
                        PauliString ps(num_qubits);
                        ps.z_mask[i] = true;
                        ps.z_mask[j] = true;
                        ps.z_mask[k] = true;
                        ps.z_mask[l] = true;
                        terms.add(0.0, ps);
                    }
                }
            }
        }
    }
}

ExponentialAnsatz ExponentialAnsatz::zeroed() const {
    ExponentialAnsatz result(num_qubits, 0, shots, warmups);
    for (const auto& [ps, coeff] : terms.get_operators()) {
        result.terms.add(0.0, ps);
    }
    return result;
}

std::vector<Bitset> ExponentialAnsatz::build_z_bits() const {
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, Complex>> terms_vec(ops.begin(), ops.end());
    std::vector<Bitset> z_bits(p, Bitset());
    for (int k = 0; k < p; ++k) {
        const auto& ps = terms_vec[k].first;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.z_mask[i])
                z_bits[k].set(num_qubits - 1 - i);
        }
    }
    return z_bits;
}

SampleSet ExponentialAnsatz::draw_samples() const {
    /*
    Draw samples from the probability distribution defined by the ansatz, using the default number of shots and warmup steps.

    Returns:
        SampleSet: A struct containing the drawn samples and their corresponding log-derivatives.
    */
    return draw_samples(shots, warmups);
}

SampleSet ExponentialAnsatz::draw_samples(int N_s, int n_warmup) const {
    /*
    Draw samples from the probability distribution defined by the ansatz.

    One Markov chain per thread runs independently. Each chain warms up for n_warmup
    sweeps from a random start, then draws its share of the N_s samples with n_warmup
    sweeps of thinning between consecutive samples. This keeps chains well-mixed while
    eliminating autocorrelation across the chains.

    Args:
        N_s (int): The number of samples to draw.
        n_warmup (int): Warmup sweeps at chain start, and thinning sweeps between samples.

    Returns:
        SampleSet: A struct containing the drawn samples and their corresponding log-derivatives.
    */
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, Complex>> terms_vec(ops.begin(), ops.end());
    std::vector<Bitset> z_bits = build_z_bits();

    // For each qubit i, the indices of terms k whose z-support includes qubit i.
    // When bit i is flipped, only these terms change parity (and thus sign in lp).
    std::vector<std::vector<int>> qubit_to_terms(num_qubits);
    for (int k = 0; k < p; ++k) {
        for (int i = 0; i < num_qubits; ++i) {
            if (z_bits[k].test(num_qubits - 1 - i)) {
                qubit_to_terms[i].push_back(k);
            }
        }
    }

    // One RNG per thread to avoid data races under OpenMP.
#if defined(_OPENMP)
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::random_device rd;
    std::vector<std::mt19937> rngs(nthreads);
    for (auto& r : rngs)
        r.seed(rd());

    SampleSet result;
    result.configs.resize(N_s, Bitset());
    result.O_mat.resize(N_s, p);

    // Each thread runs one long chain for its share of the samples.
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
        const int actual_nthreads = omp_get_num_threads();
#else
        const int tid = 0;
        const int actual_nthreads = 1;
#endif
        const int s_start = (tid * N_s) / actual_nthreads;
        const int s_end = ((tid + 1) * N_s) / actual_nthreads;

        std::mt19937& rng = rngs[tid];
        std::uniform_int_distribution<int> rand_qubit(0, num_qubits - 1);
        std::uniform_real_distribution<double> rand01(0.0, 1.0);

        // Start from a random bitstring.
        Bitset x;
        for (int i = 0; i < num_qubits; ++i) {
            if (rand01(rng) < 0.5)
                x.set(i);
        }

        // Per-term parity and weighted contribution: contrib[k] = 2*coeff_k * (-1)^parity_k
        std::vector<bool> parity(p);
        std::vector<double> contrib(p);
        double lp = 0.0;
        for (int k = 0; k < p; ++k) {
            bool neg = ((x & z_bits[k]).count()) & 1;
            parity[k] = neg;
            contrib[k] = 2.0 * terms_vec[k].second.real() * (neg ? -1.0 : 1.0);
            lp += contrib[k];
        }

        // Calculate the change in log-probability if we flip a given qubit
        auto compute_delta = [&](int qubit) -> double {
            double delta = 0.0;
            for (int k : qubit_to_terms[qubit])
                delta -= 2.0 * contrib[k];
            return delta;
        };

        // Accept a proposed flip of a given qubit, updating the state, parity, contrib
        auto accept_flip = [&](int qubit) {
            x.flip(num_qubits - 1 - qubit);
            for (int k : qubit_to_terms[qubit]) {
                contrib[k] = -contrib[k];
                parity[k] = !parity[k];
            }
        };

        // Advance the chain by the given number of full sweeps.
        auto mh_sweep = [&](int nsweeps) {
            for (int t = 0; t < nsweeps * num_qubits; ++t) {
                int i = rand_qubit(rng);
                double lp_new = lp + compute_delta(i);
                if (std::log(rand01(rng)) < lp_new - lp) {
                    accept_flip(i);
                    lp = lp_new;
                }
            }
        };

        // Initial warmup to mix the chain away from its random start
        mh_sweep(n_warmup);

        // Draw the samples, with thinning in between to reduce autocorrelation
        for (int s = s_start; s < s_end; ++s) {
            if (s > s_start) {
                mh_sweep(1);
            }
            result.configs[s] = x;
            for (int k = 0; k < p; ++k) {
                result.O_mat(s, k) = parity[k] ? int8_t(-1) : int8_t(1);
            }
        }
    }

    return result;
}

DenseVector ExponentialAnsatz::local_energy(const SampleSet& samples, const MatrixFreeHamiltonian& H) const {
    /*
    Compute the local energy E_loc(x) = ∑_{x'} H_{x,x'} Ψ(x')/Ψ(x) for each sample x.

    Args:
        samples (const SampleSet&): The samples to compute the local energy for.
        H (const MatrixFreeHamiltonian&): The Hamiltonian to compute the local energy with respect to.

    Returns:
        DenseVector: A vector containing the local energy for each sample.
    */

    // Get the operators and coefficients from the ansatz
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, Complex>> terms_vec(ops.begin(), ops.end());
    std::vector<Bitset> z_bits = build_z_bits();
    const int N_s = static_cast<int>(samples.configs.size());

    // Precompute the effect of each Hamiltonian term on the samples
    static const Complex i_powers[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    struct HTerm {
        Complex base_phase;
        Bitset flip_mask;
        Bitset sign_mask;
        std::vector<bool> flips_Pk;
    };
    const auto& h_ops = H.get_operators();
    std::vector<HTerm> h_terms;
    h_terms.reserve(h_ops.size());
    for (const auto& [ps, coeff] : h_ops) {
        Bitset flip_mask, sign_mask;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.x_mask[i] && !ps.z_mask[i]) {
                flip_mask.flip(num_qubits - 1 - i);
            } else if (!ps.x_mask[i] && ps.z_mask[i]) {
                sign_mask.set(num_qubits - 1 - i);
            } else if (ps.x_mask[i] && ps.z_mask[i]) {
                flip_mask.flip(num_qubits - 1 - i);
                sign_mask.set(num_qubits - 1 - i);
                ++n_y;
            }
        }
        Complex base_phase = coeff * i_powers[n_y & 3];
        std::vector<bool> flips(p);
        for (int k = 0; k < p; ++k) {
            flips[k] = ((flip_mask & z_bits[k]).count() & 1) != 0;
        }
        h_terms.push_back({base_phase, std::move(flip_mask), std::move(sign_mask), std::move(flips)});
    }

    // Compute the local energy for each sample using the precomputed Hamiltonian term effects
    DenseVector El(N_s);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int s = 0; s < N_s; ++s) {
        const Bitset& x = samples.configs[s];
        Complex el = 0.0;
        for (const auto& ht : h_terms) {
            bool neg_sign = ((x & ht.sign_mask).count()) & 1;
            Complex h_elem = neg_sign ? -ht.base_phase : ht.base_phase;
            Complex log_ratio = 0.0;
            for (int k = 0; k < p; ++k) {
                if (ht.flips_Pk[k]) {
                    bool neg = ((x & z_bits[k]).count()) & 1;
                    log_ratio -= static_cast<Real>(2.0) * terms_vec[k].second * Complex(neg ? -1.0 : 1.0, 0.0);
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
    os << "exp(";
    os << ansatz.get_terms();
    os << ") |+>";
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
    SampleSet samples = draw_samples();
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
    pauli_x << Complex(0), Complex(1), Complex(1), Complex(0);
    pauli_y << Complex(0), Complex(0, -1), Complex(0, 1), Complex(0);
    pauli_z << Complex(1), Complex(0), Complex(0), Complex(-1);

    for (const auto& [ps, coeff] : terms.get_operators()) {
        DenseMatrix op(1, 1);
        op(0, 0) = 1.0;
        for (int i = 0; i < num_qubits; ++i) {
            DenseMatrix single = DenseMatrix::Identity(2, 2);
            if (ps.x_mask[i] && !ps.z_mask[i])
                single = pauli_x;
            else if (!ps.x_mask[i] && ps.z_mask[i])
                single = pauli_z;
            else if (ps.x_mask[i] && ps.z_mask[i])
                single = pauli_y;
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