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

    const auto& ops = rho.get_terms().get_operators();
    const int p = static_cast<int>(ops.size());
    if (p == 0) return;

    std::vector<std::pair<PauliString, cdouble>> terms_vec(ops.begin(), ops.end());

    SampleSet samples = rho.draw_samples();
    int N_s = static_cast<int>(samples.configs.size());
    Eigen::VectorXcd El = rho.local_energy(samples, H);

    // --- Monte Carlo estimators ---
    Eigen::VectorXcd O_mean = samples.O_mat.colwise().mean();
    cdouble El_mean = El.mean();

    // M_{kk'} = <O_k* O_k'> - <O_k*><O_k'>
    DenseMatrix O_conj = samples.O_mat.conjugate();
    DenseMatrix M = (O_conj.transpose() * samples.O_mat) / static_cast<double>(N_s)
                    - O_mean.conjugate() * O_mean.transpose();

    // V_k = -i(<O_k* E_loc> - <O_k*><E_loc>)
    Eigen::VectorXcd V = std::complex<double>(0.0, -1.0) * (
        (O_conj.transpose() * El) / static_cast<double>(N_s) - O_mean.conjugate() * El_mean
    );

    // Regularise M to handle near-singular cases, then solve M ȧ = V
    M += 1e-4 * DenseMatrix::Identity(p, p);
    Eigen::VectorXcd adot = M.lu().solve(V);

    // Set the drho
    drho *= 0.0;
    for (int k = 0; k < p; ++k) {
        drho.get_terms().add(adot(k), terms_vec[k].first);
    }

}

// GCOV_EXCL_BR_STOP