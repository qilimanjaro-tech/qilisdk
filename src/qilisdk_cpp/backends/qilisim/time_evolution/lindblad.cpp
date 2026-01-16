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

#include "../qilisim.h"

SparseMatrix QiliSimCpp::create_superoperator(const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators) const {
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

void QiliSimCpp::lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) const {
    /*
    Evaluate the right-hand side of the Lindblad master equation.

    Args:
        drho (DenseMatrix&): The output derivative of the density matrix.
        rho (DenseMatrix): The current density matrix.
        H (SparseMatrix): The Hamiltonian.
        jumps (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
    */
    const std::complex<double> I(0.0, 1.0);
    if (is_unitary_on_statevector) {
        drho = H * rho;
        drho *= -I;
    } else {
        SparseMatrix temp = (H * rho).sparseView();
        drho = -I * temp;
        temp = (rho * H).sparseView();
        drho += I * temp;
        for (const auto& J : jumps) {
            SparseMatrix Jdag = J.adjoint();
            SparseMatrix JdagJ = Jdag * J;
            DenseMatrix JdagJ_rho = JdagJ * rho;
            DenseMatrix rho_JdagJ = rho * JdagJ;
            drho += J * rho * Jdag;
            drho -= 0.5 * (JdagJ_rho + rho_JdagJ);
        }
    }
}