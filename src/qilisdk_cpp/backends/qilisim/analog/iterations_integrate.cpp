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

#include "../../../libs/pybind.h"
#include "iterations.h"
#include "lindblad.h"

#include <iostream>

// GCOV_EXCL_BR_START

DenseMatrix iter_rk4(const DenseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int num_substeps, bool is_unitary_on_statevector) {
    /*
    4th-order Runge–Kutta integration of the Lindblad master equation

    Args:
        rho_0 (SparseMatrix): The initial density matrix.
        dt (double): The total time step.
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        num_substeps (int): Number of substeps to divide the time step into.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.
        atol (double): Absolute tolerance for numerical operations.

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

    long rho_rows = long(rho_0.rows());
    long rho_cols = long(rho_0.cols());

    // Standard RK4 loop
    DenseMatrix rho = rho_0;
    DenseMatrix k(rho_rows, rho_cols);
    DenseMatrix rho_tmp(rho_rows, rho_cols);
    DenseMatrix rho_old(rho_rows, rho_cols);
    double dt_sub = dt / static_cast<double>(num_substeps);
    for (int step = 0; step < num_substeps; ++step) {
        rho_old = rho;

        lindblad_rhs(k, rho, currentH, jump_operators, is_unitary_on_statevector);
        rho += (dt_sub / 6.0) * k;

        rho_tmp = rho_old;
        rho_tmp += 0.5 * dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho += (dt_sub / 3.0) * k;

        rho_tmp = rho_old;
        rho_tmp += 0.5 * dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho += (dt_sub / 3.0) * k;

        rho_tmp = rho_old;
        rho_tmp += dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho += (dt_sub / 6.0) * k;

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
    return rho;
}

void iter_rk4(DenseMatrix& rho_t, double dt, const MatrixFreeHamiltonian& currentH, const std::vector<SparseMatrix>& jump_operators, int num_substeps, bool is_unitary_on_statevector) {
    /*
    4th-order Runge–Kutta integration of the Lindblad master equation using matrix-free methods.

    Args:
        rho_t (DenseMatrix&): The density matrix to be evolved.
        dt (double): The total time step.
        currentH (MatrixFreeHamiltonian): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        num_substeps (int): Number of substeps to divide the time step into.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.
        atol (double): Absolute tolerance for numerical operations.

    Raises:
        py::value_error: If rho_0 is not square (and evolution is not unitary on state vector).
        py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
    */

    // Sanity checks
    if (rho_t.rows() != rho_t.cols() && !is_unitary_on_statevector) {
        throw py::value_error("Initial density matrix must be square.");
    }
    long dim = long(rho_t.rows());
    for (const auto& J : jump_operators) {
        if (J.rows() != dim || J.cols() != dim) {
            throw py::value_error("Jump operator dimension mismatch.");
        }
    }

    long rho_rows = long(rho_t.rows());
    long rho_cols = long(rho_t.cols());

    // Standard RK4 loop
    DenseMatrix k(rho_rows, rho_cols);
    DenseMatrix rho_tmp(rho_rows, rho_cols);
    DenseMatrix rho_old(rho_rows, rho_cols);
    double dt_sub = dt / static_cast<double>(num_substeps);
    for (int step = 0; step < num_substeps; ++step) {
        rho_old = rho_t;

        lindblad_rhs(k, rho_t, currentH, jump_operators, is_unitary_on_statevector);
        rho_t += (dt_sub / 6.0) * k;

        rho_tmp = rho_old + 0.5 * dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho_t += (dt_sub / 3.0) * k;

        rho_tmp = rho_old + 0.5 * dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho_t += (dt_sub / 3.0) * k;

        rho_tmp = rho_old + dt_sub * k;
        lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
        rho_t += (dt_sub / 6.0) * k;

        // Normalize the density matrix
        if (is_unitary_on_statevector) {
            rho_t /= rho_t.norm();
        } else {
            std::complex<double> tr = 0;
            for (int i = 0; i < dim; ++i) {
                tr += rho_t(i, i);
            }
            rho_t /= tr;
        }
    }
}

double iter_rk45(DenseMatrix& rho_t, double& dt, const MatrixFreeHamiltonian& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector) {
    /*
    Adaptive 4th/5th-order Runge–Kutta integration of the Lindblad master equation using matrix-free methods.

    Args:
        rho_t (DenseMatrix&): The density matrix to be evolved.
        dt (double&): The time step, which will be modified based on the adaptive algorithm.
        currentH (MatrixFreeHamiltonian): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.
        atol (double): Absolute tolerance for numerical operations.

    Raises:
        py::value_error: If rho_0 is not square (and evolution is not unitary on state vector).
        py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
    */

    // Sanity checks
    if (rho_t.rows() != rho_t.cols() && !is_unitary_on_statevector) {
        throw py::value_error("Initial density matrix must be square.");
    }
    long dim = long(rho_t.rows());
    for (const auto& J : jump_operators) {
        if (J.rows() != dim || J.cols() != dim) {
            throw py::value_error("Jump operator dimension mismatch.");
        }
    }

    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    
    // For scaling the k's
    static constexpr double b21 = 2.0/9.0;
    static constexpr double b31 = 1.0/12.0,       b32 = 1.0/4.0;
    static constexpr double b41 = 69.0/128.0,      b42 = -243.0/128.0,      b43 = 135.0/64.0;
    static constexpr double b51 = -17.0/12.0, b52 = 27.0/4.0, b53 = -27.0/4.0, b54 = 16.0/15.0;
    static constexpr double b61 = 65.0/432.0,  b62 = -5.0/16.0,     b63 = 13.0/16.0,  b64 = 4.0/27.0,   b65 = 5.0/144.0;

    // For scaling the t (unused)
    static constexpr double a2 = 2.0/9.0, a3 = 1.0/3.0, a4 = 3.0/4.0, a5 = 1.0, a6 = 5.0/6.0;

    // For calculating the rk4
    static constexpr double c1  = 1.0/9.0,           c3  = 9.0/20.0,         c4  = 16.0/45.0,          c5  = 1.0/12.0;
    
    // For calculating the rk5
    static constexpr double c1_prime  = 47.0/450.0,  c3_prime  = 12.0/25.0,  c4_prime  = 32.0/225.0,   c5_prime  = 1.0/30.0,   c6_prime = 6.0/25.0;

    // For calculating the error
    static constexpr double e1 = c1 - c1_prime;
    static constexpr double e3 = c3 - c3_prime;
    static constexpr double e4 = c4 - c4_prime;
    static constexpr double e5 = c5 - c5_prime;
    static constexpr double e6 = -c6_prime;

    static constexpr double atol       = 1e-5;
    static constexpr double rtol       = 1e-3;
    static constexpr double safety     = 0.9;
    static constexpr double min_factor = 0.2;
    static constexpr double max_factor = 10.0;

    long rho_rows = rho_t.rows();
    long rho_cols = rho_t.cols();

    DenseMatrix k1(rho_rows, rho_cols), k2(rho_rows, rho_cols), k3(rho_rows, rho_cols);
    DenseMatrix k4(rho_rows, rho_cols), k5(rho_rows, rho_cols), k6(rho_rows, rho_cols);
    DenseMatrix k7(rho_rows, rho_cols);
    DenseMatrix rho_tmp(rho_rows, rho_cols);
    DenseMatrix rho_5(rho_rows, rho_cols);

    lindblad_rhs(k1, rho_t, currentH, jump_operators, is_unitary_on_statevector);

    rho_tmp = rho_t + dt * b21 * k1;
    lindblad_rhs(k2, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);

    rho_tmp = rho_t + dt * (b31 * k1 + b32 * k2);
    lindblad_rhs(k3, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);

    rho_tmp = rho_t + dt * (b41 * k1 + b42 * k2 + b43 * k3);
    lindblad_rhs(k4, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);

    rho_tmp = rho_t + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4);
    lindblad_rhs(k5, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);

    rho_tmp = rho_t + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5);
    lindblad_rhs(k6, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);

    // 4th-order solution
    DenseMatrix rho_4 = rho_t + dt * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5);
    rho_4 /= rho_4.norm();

    // 5th-order solution
    rho_5 = rho_t + dt * (c1_prime * k1 + c3_prime * k3 + c4_prime * k4 + c5_prime * k5 + c6_prime * k6);
    rho_5 /= rho_5.norm();

    // The error
    std::complex<double> dot = (rho_4.adjoint() * rho_5).sum();
    double err_norm = std::sqrt(1.0 - std::pow(std::abs(dot), 2));
    err_norm /= (atol + rtol * std::max(rho_4.norm(), rho_5.norm()));

    double factor_up = safety * std::pow(err_norm, -0.2);
    factor_up = std::min(max_factor, std::max(min_factor, factor_up));
    double factor_down = safety * std::pow(err_norm, -0.25);
    factor_down = std::min(max_factor, std::max(min_factor, factor_down));
    double factor = (err_norm <= 1.0) ? factor_up : factor_down;

    std::cout << "Error norm: " << err_norm << ", factor: " << factor << std::endl;

    double dt_taken = 0.0;

    if (err_norm <= 1.0) {
        rho_t = rho_5;
        dt_taken = dt;
    }

    dt *= factor;
    return dt_taken;

}

// GCOV_EXCL_BR_STOP