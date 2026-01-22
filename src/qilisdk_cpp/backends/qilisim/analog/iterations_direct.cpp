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

#include "iterations.h"
#include "../utils/matrix_utils.h"
#include "lindblad.h"
#include "../libs/pybind.h"

SparseMatrix iter_direct(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, double atol) {
    /*
    Perform time evolution using direct matrix exponentiation.

    Args:
        rho_0 (SparseMatrix): The initial density matrix, which should be vectorized.
        dt (double): The total time step.
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether to treat the Hamiltonian as unitary on a statevector.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        SparseMatrix: The evolved density matrix after time dt.

    Raises:
        py::value_error: If currentH is not square.
        py::value_error: If rho_0 is not square.
        py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
        py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
    */

    // Sanity checks
    if (currentH.rows() != currentH.cols()) {
        throw py::value_error("Hamiltonian must be square.");
    }
    if (rho_0.cols() != rho_0.rows() && !is_unitary_on_statevector) {
        throw py::value_error("Initial density matrix must be square.");
    }
    long dim = long(currentH.rows());
    if (rho_0.rows() != dim) {
        throw py::value_error("Initial density matrix dimension does not match Hamiltonian dimension.");
    }
    for (const auto& J : jump_operators) {
        if (J.rows() != dim || J.cols() != dim) {
            throw py::value_error("Jump operator dimension does not match Hamiltonian dimension.");
        }
    }

    // If we're just doing unitary evolution on a statevector, we can exponentiate the Hamiltonian directly
    if (is_unitary_on_statevector) {
        SparseMatrix U = exp_mat(currentH, std::complex<double>(0, -dt));
        return U * rho_0;

        // If we have jump operators, need to form the full superoperator and act on the vectorized density matrix
    } else if (jump_operators.size() > 0) {
        SparseMatrix rho_t = vectorize(rho_0, atol);
        SparseMatrix L = create_superoperator(currentH, jump_operators);
        rho_t = exp_mat_action(L, dt, rho_t);
        return devectorize(rho_t, atol);

        // Otherwise we just exponentiate the Hamiltonian
    } else {
        SparseMatrix U = exp_mat(currentH, std::complex<double>(0, -dt));
        return U * rho_0 * U.adjoint();
    }
}