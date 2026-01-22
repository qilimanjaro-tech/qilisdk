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

#include "../libs/pybind.h"
#include "../utils/matrix_utils.h"
#include "iterations.h"
#include "lindblad.h"

void arnoldi(const SparseMatrix& L, const SparseMatrix& v0, int m, std::vector<SparseMatrix>& V, SparseMatrix& H, double atol) {
    /*
    Perform the Arnoldi iteration to build the basis.

    Args:
        L (SparseMatrix): The Lindblad superoperator.
        v0 (SparseMatrix): The initial vectorized density matrix.
        m (int): The dimension of the subspace.
        V (SparseMatrix&): Output basis vectors.
        H (SparseMatrix&): Output upper Hessenberg matrix.
        atol (double): Absolute tolerance for numerical operations.
    */

    // Set up the outputs
    V.clear();
    H = SparseMatrix(m + 1, m);

    // Normalize the initial vector
    SparseMatrix v = v0;
    double beta = v.norm();
    v /= beta;

    // Add the first vector to the list
    V.push_back(v);

    // For each Arnoldi iteration
    for (int j = 0; j < m; ++j) {
        // Apply the Lindbladian to the previous vector
        SparseMatrix w = L * V[j];

        // Orthogonalize against previous vectors
        for (int i = 0; i <= j; ++i) {
            std::complex<double> prod = dot(V[i], w);
            H.coeffRef(i, j) = prod;
            w -= V[i] * prod;
        }

        // Update H and check for convergence
        double to_add = w.norm();
        H.coeffRef(j + 1, j) = to_add;
        if (to_add < atol) {
            break;
        }

        // Normalize and add to V
        w /= to_add;
        V.push_back(w);
    }
}

void arnoldi_mat(const SparseMatrix& Hsys, const SparseMatrix& rho0, int m, std::vector<SparseMatrix>& V, SparseMatrix& Hk, double atol) {
    /*
    Arnoldi iteration for the unitary Liouvillian:
        L(rho) = -i (H rho - rho H)

    Args:
        Hsys (SparseMatrix): Hamiltonian (dim x dim)
        rho0 (SparseMatrix): Initial density matrix (dim x dim)
        m (int): Krylov dimension
        V (vector<SparseMatrix>): Orthonormal basis (output)
        Hk (SparseMatrix): Upper Hessenberg matrix (output)
    */

    // Set up outputs
    V.clear();
    Hk = SparseMatrix(m + 1, m);

    // Normalize initial matrix (Frobenius norm)
    SparseMatrix v = rho0;
    double beta = v.norm();
    if (beta < atol) {
        return;
    }
    v /= beta;
    V.push_back(v);

    for (int j = 0; j < m; ++j) {
        // Apply reduced Liouvillian: w = -i (H v - v H)
        SparseMatrix w = -std::complex<double>(0.0, 1.0) * (Hsys * V[j] - V[j] * Hsys);

        // Modified Gram–Schmidt
        for (int i = 0; i <= j; ++i) {
            std::complex<double> hij = dot(V[i], w);  // Tr(V[i]† w)
            Hk.coeffRef(i, j) = hij;
            w -= V[i] * hij;
        }

        // Compute norm and check for convergence
        double norm_w = w.norm();
        Hk.coeffRef(j + 1, j) = norm_w;
        if (norm_w < atol) {
            break;
        }

        // Normalize and add to V
        w /= norm_w;
        V.push_back(w);
    }
}

SparseMatrix iter_arnoldi(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int arnoldi_dim, int num_substeps, bool is_unitary_on_statevector, double atol) {
    /*
    Perform time evolution using the Arnoldi iteration.

    Args:
        rho_0 (SparseMatrix): The initial density matrix.
        dt (double): The total time step.
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        arnoldi_dim (int): Dimension of the subspace.
        num_substeps (int): Number of substeps to divide the time step into.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        SparseMatrix: The evolved density matrix after time dt.

    Raises:
        py::value_error: If arnoldi_dim is non-positive.
        py::value_error: If num_substeps is non-positive.
        py::value_error: If currentH is not square.
        py::value_error: If rho_0 is not square.
        py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
        py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
    */

    // Sanity checks
    if (arnoldi_dim <= 0) {
        throw py::value_error("Arnoldi dimension must be positive.");
    }
    if (num_substeps <= 0) {
        throw py::value_error("Number of substeps must be positive.");
    }
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

    // If we don't have jump operators, we can work directly with the density matrix
    bool is_unitary = (jump_operators.size() == 0);

    // Need to vectorize the density matrix if we're going to use the superoperator
    SparseMatrix rho_t;
    if (!is_unitary && !is_unitary_on_statevector) {
        rho_t = vectorize(rho_0, atol);
    } else {
        rho_t = rho_0;
    }

    // Vars for the Arnoldi iteration
    std::vector<SparseMatrix> V;
    SparseMatrix A;
    int subspace_dim = 0;

    // Form the Lindblad superoperator if needed
    SparseMatrix L;
    if (!is_unitary) {
        L = create_superoperator(currentH, jump_operators);
    }

    // Divide into smaller timesteps if requested
    double dt_sub = dt / static_cast<double>(num_substeps);
    for (int substep_ind = 0; substep_ind < num_substeps; ++substep_ind) {
        // Run the Arnoldi iteration to build the basis
        // After this, we have our operator approximated in the basis as A
        // and the basis vectors in V
        if (is_unitary_on_statevector) {
            arnoldi(std::complex<double>(0, 1) * currentH, rho_t, arnoldi_dim, V, A, atol);
            subspace_dim = int(V.size());
        } else if (!is_unitary) {
            arnoldi(L, rho_t, arnoldi_dim, V, A, atol);
            subspace_dim = int(V.size()) - 1;
        } else {
            arnoldi_mat(currentH, rho_t, arnoldi_dim, V, A, atol);
            subspace_dim = int(V.size());
        }
        A.conservativeResize(subspace_dim, subspace_dim);
        V.resize(subspace_dim);

        // If everything is zero then we're probably in an eigenstate and need to skip until we aren't
        if (subspace_dim == 0) {
            continue;
        }

        // Compute the action of the matrix exponential
        SparseMatrix e1(subspace_dim, 1);
        e1.coeffRef(0, 0) = 1;
        SparseMatrix y = exp_mat_action(A, dt_sub, e1);

        // Reconstruct the final density matrix using the basis vectors
        SparseMatrix rho_t_new(rho_t.rows(), rho_t.cols());
        for (int j = 0; j < int(V.size()); ++j) {
            rho_t_new += V[j] * y.coeff(j, 0);
        }
        rho_t = rho_t_new;

        // Normalize the density matrix
        if (is_unitary_on_statevector) {
            rho_t /= rho_t.norm();
        } else if (is_unitary) {
            rho_t /= trace(rho_t);
            continue;
        } else if (!is_unitary_on_statevector) {
            std::complex<double> tr = 0;
            for (long i = 0; i < dim; ++i) {
                long vec_index = i * dim + i;
                tr += rho_t.coeff(vec_index, 0);
            }
            rho_t /= tr;
        }
    }

    // If we vectorized, need to devectorize before returning
    if (!is_unitary && !is_unitary_on_statevector) {
        rho_t = devectorize(rho_t, atol);
    }

    return rho_t;
}