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
#include <unordered_map>
#include <utility>
#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

namespace {
// Below this much work (samples x terms touched) using many threads costs more than it saves
constexpr long long kMinParallelWork = 1 << 15;
}  // namespace

ExponentialAnsatz::ExponentialAnsatz(int num_qubits, int order, int shots, int warmups, uint64_t seed) : rng(std::make_shared<std::mt19937_64>(seed)) {
    /*
    Construct an ExponentialAnsatz with the given number of qubits and maximum number of terms.

    The format of this ansatz is exp(sum_i c_i P_i) |+> where P_i are Pauli strings and c_i are coefficients.
    For now we restrict the P_i to be Z operators.

    Args:
        num_qubits (int): The number of qubits in the system.
        order (int): The maximum order of terms to include in the ansatz.
        shots (int): The number of shots to use for sampling.
        warmups (int): The number of warmup steps to use for sampling.
        seed (uint64_t): Seed of the random stream used when sampling. Copies of the ansatz share this stream.

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
    result.rng = rng;
    for (const auto& [ps, coeff] : terms.get_operators()) {
        result.terms.add(0.0, ps);
    }
    return result;
}

std::vector<Bitset> ExponentialAnsatz::build_z_bits() const {
    std::vector<Bitset> z_bits;
    z_bits.reserve(terms.get_operators().size());
    for (const auto& [ps, coeff] : terms.get_operators()) {
        Bitset bits;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.z_mask[i])
                bits.set(num_qubits - 1 - i);
        }
        z_bits.push_back(bits);
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
    sweeps from a random start, then draws its share of the N_s samples with one
    sweep of thinning between consecutive samples. Small workloads use a single chain
    on the calling thread, since waking the thread pool would cost more than the sampling.
    The chains are seeded from the ansatz's random stream, so results are reproducible
    for a given seed and thread count.

    Args:
        N_s (int): The number of samples to draw.
        n_warmup (int): Warmup sweeps at chain start.

    Returns:
        SampleSet: A struct containing the drawn samples and their corresponding log-derivatives.
    */
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<Bitset> z_bits = build_z_bits();
    std::vector<double> two_coeffs;
    two_coeffs.reserve(p);
    for (const auto& [ps, coeff] : ops) {
        two_coeffs.push_back(2.0 * coeff.real());
    }

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

    // One seed per potential chain, drawn up front so the chains don't race on the shared stream
#if defined(_OPENMP)
    const int max_chains = omp_get_max_threads();
#else
    const int max_chains = 1;
#endif
    const bool parallel = static_cast<long long>(N_s) * (n_warmup + 1) * std::max(p, 1) >= kMinParallelWork;
    std::vector<uint64_t> chain_seeds(max_chains);
    for (auto& chain_seed : chain_seeds)
        chain_seed = (*rng)();

    SampleSet result;
    result.configs.resize(N_s, Bitset());
    result.O_mat.resize(N_s, p);

    // Each thread runs one long chain for its share of the samples.
#if defined(_OPENMP)
#pragma omp parallel if (parallel)
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

        std::mt19937_64 chain_rng(chain_seeds[tid]);
        std::uniform_int_distribution<int> rand_qubit(0, num_qubits - 1);
        std::uniform_real_distribution<double> rand01(0.0, 1.0);

        // Start from a random bitstring.
        Bitset x;
        for (int i = 0; i < num_qubits; ++i) {
            if (rand01(chain_rng) < 0.5) {
                x.set(i);
            }
        }

        // Per-term sign (-1)^parity_k and weighted contribution contrib[k] = 2*coeff_k * sign_k
        std::vector<int8_t> sign(p);
        std::vector<double> contrib(p);
        for (int k = 0; k < p; ++k) {
            sign[k] = ((x & z_bits[k]).count() & 1) ? int8_t(-1) : int8_t(1);
            contrib[k] = two_coeffs[k] * sign[k];
        }

        // Advance the chain by the given number of full sweeps, accepting flips with the Metropolis rule.
        auto mh_sweep = [&](int nsweeps) {
            for (int t = 0; t < nsweeps * num_qubits; ++t) {
                int i = rand_qubit(chain_rng);
                double delta = 0.0;
                for (int k : qubit_to_terms[i])
                    delta -= 2.0 * contrib[k];
                if (delta >= 0.0 || std::log(rand01(chain_rng)) < delta) {
                    x.flip(num_qubits - 1 - i);
                    for (int k : qubit_to_terms[i]) {
                        contrib[k] = -contrib[k];
                        sign[k] = static_cast<int8_t>(-sign[k]);
                    }
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
                result.O_mat(s, k) = sign[k];
            }
        }
    }

    return result;
}

DenseVector ExponentialAnsatz::local_energy(const SampleSet& samples, const MatrixFreeHamiltonian& H) const {
    /*
    Compute the local energy E_loc(x) = ∑_{x'} H_{x,x'} Ψ(x')/Ψ(x) for each sample x.

    A Hamiltonian term flipping the bits in mask f maps x to x' = x ^ f, and the amplitude ratio is
    Ψ(x')/Ψ(x) = exp(-2 ∑_{k : P_k anticommutes with f} a_k O_k(x)), using the log-derivatives O_k(x)
    already stored in the samples. Terms are grouped by flip mask, so all diagonal (Z-type) terms share
    a ratio of one and each off-diagonal mask costs a single exponential per sample.

    Args:
        samples (const SampleSet&): The samples to compute the local energy for, drawn from this ansatz.
        H (const MatrixFreeHamiltonian&): The Hamiltonian to compute the local energy with respect to.

    Returns:
        DenseVector: A vector containing the local energy for each sample.

    Raises:
        std::invalid_argument: If the samples' log-derivatives don't match this ansatz's terms.
    */

    // Get the operators and coefficients from the ansatz
    const auto& ops = terms.get_operators();
    const int p = static_cast<int>(ops.size());
    const int N_s = static_cast<int>(samples.configs.size());
    if (samples.O_mat.rows() != N_s || samples.O_mat.cols() != p) {
        throw std::invalid_argument("Samples do not match the terms of the ansatz.");
    }
    std::vector<Bitset> z_bits = build_z_bits();
    std::vector<Complex> two_coeffs;
    two_coeffs.reserve(p);
    for (const auto& [ps, coeff] : ops) {
        two_coeffs.push_back(static_cast<Real>(2.0) * coeff);
    }

    // Group the Hamiltonian terms by the bits they flip, since terms sharing a flip share the amplitude ratio.
    // Each Y contributes <x|Y|x^1> = -i (-1)^x, hence the powers of -i and the Y qubits in the sign mask.
    static const Complex minus_i_powers[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
    struct SignedPhase {
        Bitset sign_mask;
        Complex phase;
    };
    struct FlipGroup {
        std::vector<int> flipped_terms;
        std::vector<SignedPhase> phases;
    };
    std::vector<FlipGroup> groups;
    std::unordered_map<Bitset, size_t> group_of_mask;
    for (const auto& [ps, coeff] : H.get_operators()) {
        Bitset flip_mask, sign_mask;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            if (ps.x_mask[i]) {
                flip_mask.set(num_qubits - 1 - i);
            }
            if (ps.z_mask[i]) {
                sign_mask.set(num_qubits - 1 - i);
            }
            if (ps.x_mask[i] && ps.z_mask[i]) {
                ++n_y;
            }
        }
        auto [it, inserted] = group_of_mask.try_emplace(flip_mask, groups.size());
        if (inserted) {
            FlipGroup group;
            for (int k = 0; k < p; ++k) {
                if ((flip_mask & z_bits[k]).count() & 1) {
                    group.flipped_terms.push_back(k);
                }
            }
            groups.push_back(std::move(group));
        }
        groups[it->second].phases.push_back({sign_mask, coeff * minus_i_powers[n_y & 3]});
    }

    // Each sample's log-derivatives contiguous in memory
    const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> O_t = samples.O_mat.transpose();
    long long work_per_sample = 0;
    for (const auto& group : groups) {
        work_per_sample += static_cast<long long>(group.phases.size() + group.flipped_terms.size());
    }
    const bool parallel = N_s * work_per_sample >= kMinParallelWork;

    // Compute the local energy for each sample using the grouped Hamiltonian terms
    DenseVector El(N_s);
#if defined(_OPENMP)
#pragma omp parallel for if (parallel)
#endif
    for (int s = 0; s < N_s; ++s) {
        const Bitset& x = samples.configs[s];
        const int8_t* O_s = O_t.data() + static_cast<Eigen::Index>(s) * p;
        Real el_re = 0.0;
        Real el_im = 0.0;
        for (const auto& group : groups) {
            Real h_re = 0.0;
            Real h_im = 0.0;
            for (const auto& term : group.phases) {
                const Real sign = ((x & term.sign_mask).count() & 1) ? Real(-1.0) : Real(1.0);
                h_re += sign * term.phase.real();
                h_im += sign * term.phase.imag();
            }
            if (group.flipped_terms.empty()) {
                el_re += h_re;
                el_im += h_im;
                continue;
            }
            Real log_re = 0.0;
            Real log_im = 0.0;
            for (int k : group.flipped_terms) {
                log_re -= two_coeffs[k].real() * O_s[k];
                log_im -= two_coeffs[k].imag() * O_s[k];
            }
            const Real magnitude = std::exp(log_re);
            const Real ratio_re = magnitude * std::cos(log_im);
            const Real ratio_im = magnitude * std::sin(log_im);
            el_re += h_re * ratio_re - h_im * ratio_im;
            el_im += h_re * ratio_im + h_im * ratio_re;
        }
        El(s) = Complex(el_re, el_im);
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