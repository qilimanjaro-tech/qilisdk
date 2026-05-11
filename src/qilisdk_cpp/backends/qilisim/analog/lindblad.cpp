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

#include "lindblad.h"
#include <random>

// GCOV_EXCL_BR_START

const std::complex<double> imag(0.0, 1.0);

SparseMatrix create_superoperator(const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators) {
    /*
    Form the Lindblad superoperator for the given Hamiltonian and jump operators.

    Args:
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.

    Returns:
        SparseMatrix: The Lindblad superoperator.
    */

    // The superoperator dimension
    long dim = long(currentH.rows());
    long dim_rho = dim * dim;
    SparseMatrix L(dim_rho, dim_rho);

    // The identity
    Triplets iden_entries;
    for (int i = 0; i < dim; ++i) {
        iden_entries.emplace_back(Triplet(i, i, 1.0));
    }
    SparseMatrix iden(dim, dim);
    iden.setFromTriplets(iden_entries.begin(), iden_entries.end());

    // The contribution from the Hamiltonian
    SparseMatrix iden_H = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(iden, currentH);
    SparseMatrix H_T_iden = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(currentH.transpose(), iden);
    L += iden_H * std::complex<double>(0, -1);
    L += H_T_iden * std::complex<double>(0, 1);

    // The contribution from the jump operators
    for (const auto& L_k : jump_operators) {
        SparseMatrix L_k_dag = L_k.adjoint();
        SparseMatrix L_k_L_k_dag = L_k_dag * L_k;

        SparseMatrix term1 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(L_k, L_k.conjugate());
        SparseMatrix term2 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(L_k_L_k_dag, iden) * 0.5;
        SparseMatrix term3 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(iden, L_k_L_k_dag.transpose()) * 0.5;

        L += term1;
        L -= term2;
        L -= term3;
    }
    return L;
}

void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) {
    /*
    Evaluate the right-hand side of the Lindblad master equation.

    Args:
        drho (DenseMatrix&): The output derivative of the density matrix.
        rho (DenseMatrix): The current density matrix.
        H (SparseMatrix): The Hamiltonian.
        jumps (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
    */
    if (is_unitary_on_statevector) {
        drho = H * rho;
        drho *= -imag;
    } else {
        DenseMatrix Hrho = H * rho;
        drho = Hrho;
        drho -= Hrho.adjoint();
        drho *= -imag;
        for (const auto& J : jumps) {
            SparseMatrix Jdag = J.adjoint();
            SparseMatrix JdagJ = Jdag * J;
            drho += J * rho * Jdag;
            drho -= 0.5 * (JdagJ * rho + rho * JdagJ);
        }
    }
}

void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const MatrixFreeHamiltonian& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) {
    /*
    Evaluate the right-hand side of the Lindblad master equation.

    Args:
        drho (DenseMatrix&): The output derivative of the density matrix.
        rho (DenseMatrix): The current density matrix.
        H (MatrixFreeHamiltonian): The Hamiltonian.
        jumps (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
    */
    if (is_unitary_on_statevector) {
        H.apply(rho, MatrixFreeApplicationType::Left, drho);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < drho.size(); ++i) {
            drho(i) *= -imag;
        }
    } else {
        DenseMatrix Hrho(rho.rows(), rho.cols());
        DenseMatrix rhoH(rho.rows(), rho.cols());
        H.apply(rho, MatrixFreeApplicationType::Left, Hrho);
        H.apply(rho, MatrixFreeApplicationType::Right, rhoH);
        drho = -imag * (Hrho - rhoH);
        for (const auto& J : jumps) {
            SparseMatrix Jdag = J.adjoint();
            SparseMatrix JdagJ = Jdag * J;
            drho += J * rho * Jdag;
            drho -= 0.5 * (JdagJ * rho + rho * JdagJ);
        }
    }
}

void lindblad_rhs(MatrixFreeHamiltonian& drho, const MatrixFreeHamiltonian& rho, const MatrixFreeHamiltonian& H) {
    /*
    Evaluate the right-hand side of the Lindblad master equation for the approximate method.

    Args:
        drho (MatrixFreeHamiltonian&): The output derivative of the density matrix.
        rho (MatrixFreeHamiltonian): The current density matrix.
        H (MatrixFreeHamiltonian): The Hamiltonian.
    */
    drho = H * rho;
    drho *= -imag;
}

void lindblad_rhs(ExponentialAnsatz& drho, const ExponentialAnsatz& rho, const MatrixFreeHamiltonian& H) {
    /*
    Evaluate the right-hand side of the variational equations for the ExponentialAnsatz.

    The ansatz is |Ψ⟩ = exp(∑_k a_k P_k)|+⟩ where P_k are Z-type Pauli strings.
    Uses stochastic reconfiguration (VMC) to compute ȧ_k = (M⁻¹ V)_k where:
      M_{kk'} = <O_k* O_k'> - <O_k*><O_k'>        (quantum geometric tensor)
      V_k     = -(<O_k* E_loc> - <O_k*><E_loc>)   (imaginary-time force / SR)
      O_k(σ)  = P_k(σ) = ∏_{i∈S_k} σᵢ             (log-derivative)
    Samples are drawn from |Ψ(σ)|² via Metropolis-Hastings.

    Args:
        drho (ExponentialAnsatz&): Output: coefficients set to ȧ_k.
        rho (ExponentialAnsatz): Current ansatz with parameters a_k.
        H (MatrixFreeHamiltonian): The Hamiltonian.
    */
    using cdouble = std::complex<double>;
    static const cdouble neg_i_powers[4] = {{1,0},{0,-1},{-1,0},{0,1}};

    const auto& ops = rho.get_terms().get_operators();
    const int n_qubits = rho.get_terms().get_nqubits();
    const int p = static_cast<int>(ops.size());
    if (p == 0) return;

    // Index the ansatz terms for stable iteration order
    std::vector<std::pair<PauliString, cdouble>> terms_vec(ops.begin(), ops.end());

    // Precompute bitmask for each Z-type Pauli string P_k
    // P_k(x) = (-1)^{popcount(x & z_bits[k])}
    std::vector<long> z_bits(p);
    for (int k = 0; k < p; ++k) {
        long bits = 0;
        const auto& ps = terms_vec[k].first;
        for (int i = 0; i < n_qubits; ++i) {
            if (ps.z_mask[i]) bits |= (1LL << (n_qubits - 1 - i));
        }
        z_bits[k] = bits;
    }

    // Precompute Hamiltonian terms: flip_mask, sign_mask, and which P_k change
    // when the flip is applied (needed for the wavefunction ratio Ψ(x')/Ψ(x))
    struct HTerm {
        cdouble base_phase;
        long    flip_mask;
        long    sign_mask;
        std::vector<bool> flips_Pk;
    };
    const auto& h_ops = H.get_operators();
    std::vector<HTerm> h_terms;
    h_terms.reserve(h_ops.size());
    for (const auto& [ps, coeff] : h_ops) {
        long flip_mask = 0, sign_mask = 0;
        int n_y = 0;
        for (int i = 0; i < n_qubits; ++i) {
            long mask = 1LL << (n_qubits - 1 - i);
            if ( ps.x_mask[i] && !ps.z_mask[i]) { flip_mask ^= mask; }
            else if (!ps.x_mask[i] &&  ps.z_mask[i]) { sign_mask |= mask; }
            else if ( ps.x_mask[i] &&  ps.z_mask[i]) { flip_mask ^= mask; sign_mask |= mask; ++n_y; }
        }
        cdouble base_phase = coeff * neg_i_powers[n_y & 3];
        std::vector<bool> flips(p);
        for (int k = 0; k < p; ++k) {
            flips[k] = (__builtin_popcountll((long long)(flip_mask & z_bits[k])) & 1) != 0;
        }
        h_terms.push_back({base_phase, flip_mask, sign_mask, std::move(flips)});
    }

    // --- Metropolis-Hastings: sample x ~ |Ψ(x)|² = exp(2 Re ∑_k a_k P_k(x)) ---
    const int N_s      = 1000;
    const int n_warmup = 200;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int>  rand_qubit(0, n_qubits - 1);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    // Compute log|Ψ(x)|²; only Re(a_k) contributes since P_k(x) ∈ ℝ
    auto log_prob = [&](long x) -> double {
        double lp = 0.0;
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            lp += 2.0 * terms_vec[k].second.real() * (neg ? -1.0 : 1.0);
        }
        return lp;
    };

    long x = std::uniform_int_distribution<long>(0, (1LL << n_qubits) - 1)(rng);
    double lp = log_prob(x);

    // Warmup sweeps
    for (int s = 0; s < n_warmup * n_qubits; ++s) {
        int  i      = rand_qubit(rng);
        long x_new  = x ^ (1LL << (n_qubits - 1 - i));
        double lp_new = log_prob(x_new);
        if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }
    }

    // --- Collect samples, O_k, and E_loc ---
    DenseMatrix O_mat(N_s, p);   // O_mat(s,k) = P_k(x_s) ∈ {-1,+1}
    Eigen::VectorXcd El(N_s);   // El(s) = E_loc(x_s)

    for (int s = 0; s < N_s; ++s) {
        // Systematic sweep over all qubits; a random sweep can trap on even-parity states
        for (int i = 0; i < n_qubits; ++i) {
            long x_new  = x ^ (1LL << (n_qubits - 1 - i));
            double lp_new = log_prob(x_new);
            if (std::log(rand01(rng)) < lp_new - lp) { x = x_new; lp = lp_new; }
        }

        // Log-derivatives O_k(x) = P_k(x)
        for (int k = 0; k < p; ++k) {
            bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
            O_mat(s, k) = cdouble(neg ? -1.0 : 1.0, 0.0);
        }

        // Local energy: E_loc(x) = ∑_{x'} H_{x,x'} Ψ(x')/Ψ(x)
        // For each Hamiltonian Pauli term, x' = x XOR flip_mask (one connected state)
        // log(Ψ(x')/Ψ(x)) = -2 ∑_{k: P_k changes} a_k P_k(x)
        cdouble el = 0.0;
        for (const auto& ht : h_terms) {
            bool neg_sign = __builtin_popcountll((long long)(x & ht.sign_mask)) & 1;
            cdouble h_elem = neg_sign ? -ht.base_phase : ht.base_phase;

            cdouble log_ratio = 0.0;
            for (int k = 0; k < p; ++k) {
                if (ht.flips_Pk[k]) {
                    bool neg = __builtin_popcountll((long long)(x & z_bits[k])) & 1;
                    log_ratio -= 2.0 * terms_vec[k].second * cdouble(neg ? -1.0 : 1.0, 0.0);
                }
            }
            el += h_elem * std::exp(log_ratio);
        }
        El(s) = el;
    }

    // --- Monte Carlo estimators ---
    Eigen::VectorXcd O_mean = O_mat.colwise().mean();
    cdouble El_mean = El.mean();

    // M_{kk'} = <O_k* O_k'> - <O_k*><O_k'>
    DenseMatrix O_conj = O_mat.conjugate();
    DenseMatrix M = (O_conj.transpose() * O_mat) / static_cast<double>(N_s)
                    - O_mean.conjugate() * O_mean.transpose();

    // V_k = -(<O_k* E_loc> - <O_k*><E_loc>)  (imaginary-time / stochastic reconfiguration)
    // The real-time Schrödinger form would prepend -i, but that makes adot purely imaginary
    // for a real Hamiltonian and real initial parameters, leaving Re(a_k) frozen and the
    // sampling distribution unchanged.  Imaginary-time evolution gives a real force, so
    // Re(a_k) evolves and the distribution tracks the ground state.
    Eigen::VectorXcd V = -(
        (O_conj.transpose() * El) / static_cast<double>(N_s) - O_mean.conjugate() * El_mean
    );

    // Regularise M to handle near-singular cases, then solve M ȧ = V
    M += 1e-4 * DenseMatrix::Identity(p, p);
    Eigen::VectorXcd adot = M.lu().solve(V);

    // Write ȧ_k into drho: zero existing coefficients, then set each to adot(k)
    drho *= 0.0;
    for (int k = 0; k < p; ++k) {
        drho.get_terms().add(adot(k), terms_vec[k].first);
    }
}

// GCOV_EXCL_BR_STOP