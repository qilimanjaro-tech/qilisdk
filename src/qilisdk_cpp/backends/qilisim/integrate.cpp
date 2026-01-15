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

#include "qilisim.h"

SparseMatrix QiliSimCpp::iter_integrate(const SparseMatrix& rho_0,
                                        double dt,
                                        const SparseMatrix& currentH,
                                        const std::vector<SparseMatrix>& jump_operators,
                                        int num_substeps,
                                        bool is_unitary_on_statevector) const {
    /*
    4th-order Runge–Kutta integration of the Lindblad master equation

    Args:
        rho_0 (SparseMatrix): The initial density matrix.
        dt (double): The total time step.
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
        num_substeps (int): Number of substeps to divide the time step into.

    Returns:
        SparseMatrix: The evolved density matrix after time dt.

    Raises:
        py::value_error: If num_substeps is non-positive.
        py::value_error: If currentH is not square.
        py::value_error: If rho_0 is not square (and evolution is not unitary on state vector).
        py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
        py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
    */

    // Sanity checks
    if (currentH.rows() != currentH.cols()) {
        throw py::value_error("Hamiltonian must be square.");
    }
    if (rho_0.rows() != rho_0.cols() && !is_unitary_on_statevector) {
        throw py::value_error("Initial density matrix must be square.");
    }
    long dim = long(currentH.rows());
    if (rho_0.rows() != dim) {
        throw py::value_error("Dimension mismatch.");
    }
    for (const auto& J : jump_operators) {
        if (J.rows() != dim || J.cols() != dim) {
            throw py::value_error("Jump operator dimension mismatch.");
        }
    }

    int rho_rows = int(rho_0.rows());
    int rho_cols = int(rho_0.cols());

    // Standard RK4 loop
    DenseMatrix rho = rho_0;
    DenseMatrix k1(rho_rows, rho_cols);
    DenseMatrix k2(rho_rows, rho_cols);
    DenseMatrix k3(rho_rows, rho_cols);
    DenseMatrix k4(rho_rows, rho_cols);
    DenseMatrix rho_tmp(rho_rows, rho_cols);
    double dt_sub = dt / static_cast<double>(num_substeps);
    for (int step = 0; step < num_substeps; ++step) {
        lindblad_rhs(k1, rho, currentH, jump_operators, is_unitary_on_statevector);
        rho_tmp = rho;
        rho_tmp += 0.5 * dt_sub * k1;
        lindblad_rhs(k2, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho_tmp = rho;
        rho_tmp += 0.5 * dt_sub * k2;
        lindblad_rhs(k3, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho_tmp = rho;
        rho_tmp += dt_sub * k3;
        lindblad_rhs(k4, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho += (dt_sub / 6.0) * k1;
        rho += (dt_sub / 3.0) * k2;
        rho += (dt_sub / 3.0) * k3;
        rho += (dt_sub / 6.0) * k4;

        // Normalize the density matrix
        if (is_unitary_on_statevector) {
            rho /= rho.norm();
        } else {
            std::complex<double> tr = 0;
            for (int i = 0; i < dim; ++i) {
                tr += rho(i, i);
            }
            rho /= tr;
        }
    }
    return rho.sparseView();
}