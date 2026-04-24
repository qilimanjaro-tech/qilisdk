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

#if defined(_OPENMP)
#pragma omp declare reduction(complex_double_reduction : std::complex <double> : omp_out += omp_in) initializer(omp_priv = std::complex <double>(0.0, 0.0))
#endif

// GCOV_EXCL_BR_START

DenseMatrix iter_rk4_matrix(const DenseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector) {
    /*
    4th-order Runge–Kutta integration of the Lindblad master equation

    Args:
        rho_0 (DenseMatrix): The initial density matrix.
        dt (double): The total time step.
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        DenseMatrix: The evolved density matrix after time dt.

    Raises:
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

    rho_old = rho;

    lindblad_rhs(k, rho, currentH, jump_operators, is_unitary_on_statevector);
    rho += (dt / 6.0) * k;

    rho_tmp = rho_old;
    rho_tmp += 0.5 * dt * k;
    lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
    rho += (dt / 3.0) * k;

    rho_tmp = rho_old;
    rho_tmp += 0.5 * dt * k;
    lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
    rho += (dt / 3.0) * k;

    rho_tmp = rho_old;
    rho_tmp += dt * k;
    lindblad_rhs(k, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
    rho += (dt / 6.0) * k;

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

    return rho;
}

MatrixFreeHamiltonian construct_current_hamiltonian(double t, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list) {
    /*
    Use linear interpolation to construct the current Hamiltonian at time t
    based on the provided list of Hamiltonians and their corresponding parameters.

    Args:
        hamiltonians (std::vector<MatrixFreeHamiltonian>): The list of Hamiltonians.
        step_list (std::vector<double>): The list of time points corresponding to the parameters.
        parameters_list (std::vector<std::vector<double>>): The list of parameters for each Hamiltonian.
        t (double): The current time.

    Returns:
        MatrixFreeHamiltonian: The interpolated Hamiltonian at time t.
    */
    size_t ind = 0;
    while (ind < step_list.size() && step_list[ind] < t) {
        ind++;
    }
    ind = std::min(ind, step_list.size() - 1);
    MatrixFreeHamiltonian currentH;
    for (size_t h = 0; h < hamiltonians.size(); ++h) {
        if (ind == 0) {
            currentH += hamiltonians[h] * parameters_list[h][0];
        } else {
            double t1 = step_list[ind - 1];
            double t2 = step_list[ind];
            double p1 = parameters_list[h][ind - 1];
            double p2 = parameters_list[h][ind];
            double c = (t - t1) / (t2 - t1);
            currentH += hamiltonians[h] * ((1.0 - c) * p1 + c * p2);
        }
    }
    return currentH;
}

void iter_rk4(DenseMatrix& rho_t, double t, double dt, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector) {
    /*
    4th-order Runge–Kutta integration of the Lindblad master equation using matrix-free methods.

    Args:
        rho_t (DenseMatrix&): The density matrix to be evolved.
        t (double): The current time.
        dt (double): The total time step.
        step_list (std::vector<double>): The list of time points corresponding to the parameters.
        hamiltonians (std::vector<MatrixFreeHamiltonian>): The list of Hamiltonians.
        parameters_list (std::vector<std::vector<double>>): The list of parameters for each Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.

    Raises:
        py::value_error: If rho_t is not square (and evolution is not unitary on state vector).
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
    double t_step = t;
    rho_old = rho_t;

    lindblad_rhs(k, rho_t, construct_current_hamiltonian(t_step, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);
    rho_t += (dt / 6.0) * k;

    rho_tmp = rho_old + 0.5 * dt * k;
    lindblad_rhs(k, rho_tmp, construct_current_hamiltonian(t_step + 0.5 * dt, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);
    rho_t += (dt / 3.0) * k;

    rho_tmp = rho_old + 0.5 * dt * k;
    lindblad_rhs(k, rho_tmp, construct_current_hamiltonian(t_step + 0.5 * dt, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);
    rho_t += (dt / 3.0) * k;

    rho_tmp = rho_old + dt * k;
    lindblad_rhs(k, rho_tmp, construct_current_hamiltonian(t_step + dt, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);
    rho_t += (dt / 6.0) * k;

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

double iter_rk45(DenseMatrix& rho_t, double t, double& dt, const std::vector<double>& step_list, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector, double tol, DenseMatrix& k_saved) {
    /*
    Adaptive 4th/5th-order Runge–Kutta integration of the Lindblad master equation using matrix-free methods.

    Args:
        rho_t (DenseMatrix&): The density matrix to be evolved.
        t (double): The current time.
        dt (double&): The time step, which will be modified based on the adaptive algorithm.
        step_list (std::vector<double>): The list of time points corresponding to the parameters.
        hamiltonians (std::vector<MatrixFreeHamiltonian>): The list of Hamiltonians.
        parameters_list (std::vector<std::vector<double>>): The list of parameters for each Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): If the evolution should be treated as a unitary acting on a state vector.
        tol (double): Tolerance for the adaptive algorithm.
        k_saved (DenseMatrix&): The first Runge-Kutta matrix, which is the same as the last one from the previous step.

    Raises:
        py::value_error: If rho_t is not square (and evolution is not unitary on state vector).
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

    // https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

    // Coefficients for adjusting the solution
    static constexpr double b21 = 1.0 / 5.0;
    static constexpr double b31 = 3.0 / 40.0, b32 = 9.0 / 40.0;
    static constexpr double b41 = 44.0 / 45.0, b42 = -56.0 / 15.0, b43 = 32.0 / 9.0;
    static constexpr double b51 = 19372.0 / 6561.0, b52 = -25360.0 / 2187.0, b53 = 64448.0 / 6561.0, b54 = -212.0 / 729.0;
    static constexpr double b61 = 9017.0 / 3168.0, b62 = -355.0 / 33.0, b63 = 46732.0 / 5247.0, b64 = 49.0 / 176.0, b65 = -5103.0 / 18656.0;
    static constexpr double b71 = 35.0 / 384.0, b73 = 500.0 / 1113.0, b74 = 125.0 / 192.0, b75 = -2187.0 / 6784.0, b76 = 11.0 / 84.0;

    // Coefficients for adjusting the time step
    static constexpr double a2 = 1.0 / 5.0, a3 = 3.0 / 10.0, a4 = 4.0 / 5.0, a5 = 8.0 / 9.0, a6 = 1.0, a7 = 1.0;

    // Coefficients for forming the 4th order solution
    static constexpr double c41 = 5179.0 / 57600.0, c43 = 7571.0 / 16695.0, c44 = 393.0 / 640.0, c45 = -92097.0 / 339200.0, c46 = 187.0 / 2100.0, c47 = 1.0 / 40.0;

    // For dt scaling
    static constexpr double min_factor = 0.1;
    static constexpr double max_factor = 20.0;

    // Check that everything in the Butcher tableau is consistent
    static_assert(std::abs(a2 - (b21)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(a3 - (b31 + b32)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(a4 - (b41 + b42 + b43)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(a5 - (b51 + b52 + b53 + b54)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(a6 - (b61 + b62 + b63 + b64 + b65)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(a7 - (b71 + b73 + b74 + b75 + b76)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(1.0 - (b71 + b73 + b74 + b75 + b76)) < 1e-12, "Inconsistent Butcher tableau");
    static_assert(std::abs(1.0 - (c41 + c43 + c44 + c45 + c46 + c47)) < 1e-12, "Inconsistent Butcher tableau");

    // Sadly we need to do a lot of matrix allocations :(
    long rho_rows = rho_t.rows();
    long rho_cols = rho_t.cols();
    bool already_have_first_step = (k_saved.rows() != 0);
    DenseMatrix k1 = already_have_first_step ? k_saved : DenseMatrix(rho_rows, rho_cols);
    DenseMatrix k2(rho_rows, rho_cols);
    DenseMatrix k3(rho_rows, rho_cols);
    DenseMatrix k4(rho_rows, rho_cols);
    DenseMatrix k5(rho_rows, rho_cols);
    DenseMatrix k6(rho_rows, rho_cols);
    DenseMatrix k7(rho_rows, rho_cols);
    DenseMatrix rho_tmp(rho_rows, rho_cols);
    DenseMatrix rho_4(rho_rows, rho_cols);
    DenseMatrix rho_5(rho_rows, rho_cols);

    if (!already_have_first_step) {
        double t_1 = t;
        lindblad_rhs(k1, rho_t, construct_current_hamiltonian(t_1, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * b21 * k1(i, j);
        }
    }
    double t_2 = t + a2 * dt;
    lindblad_rhs(k2, rho_tmp, construct_current_hamiltonian(t_2, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * (b31 * k1(i, j) + b32 * k2(i, j));
        }
    }
    double t_3 = t + a3 * dt;
    lindblad_rhs(k3, rho_tmp, construct_current_hamiltonian(t_3, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * (b41 * k1(i, j) + b42 * k2(i, j) + b43 * k3(i, j));
        }
    }
    double t_4 = t + a4 * dt;
    lindblad_rhs(k4, rho_tmp, construct_current_hamiltonian(t_4, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * (b51 * k1(i, j) + b52 * k2(i, j) + b53 * k3(i, j) + b54 * k4(i, j));
        }
    }
    double t_5 = t + a5 * dt;
    lindblad_rhs(k5, rho_tmp, construct_current_hamiltonian(t_5, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * (b61 * k1(i, j) + b62 * k2(i, j) + b63 * k3(i, j) + b64 * k4(i, j) + b65 * k5(i, j));
        }
    }
    double t_6 = t + a6 * dt;
    lindblad_rhs(k6, rho_tmp, construct_current_hamiltonian(t_6, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_tmp(i, j) = rho_t(i, j) + dt * (b71 * k1(i, j) + b73 * k3(i, j) + b74 * k4(i, j) + b75 * k5(i, j) + b76 * k6(i, j));
        }
    }
    double t_7 = t + a7 * dt;
    lindblad_rhs(k7, rho_tmp, construct_current_hamiltonian(t_7, step_list, hamiltonians, parameters_list), jump_operators, is_unitary_on_statevector);

    // All these loops combined into one
    std::complex<double> rho_4_norm = 0.0;
    std::complex<double> rho_5_norm = 0.0;
    std::complex<double> rho_5_trace = 0.0;
    std::complex<double> overlap = 0.0;
    
    // Normalizing statevectors is by the full norm
    if (is_unitary_on_statevector) {
#if defined(_OPENMP)
#pragma omp parallel for reduction(complex_double_reduction : overlap, rho_4_norm, rho_5_norm) schedule(static)
#endif
        for (long i = 0; i < rho_rows; ++i) {
            for (long j = 0; j < rho_cols; ++j) {
                rho_4(i, j) = rho_t(i, j) + dt * (c41 * k1(i, j) + c43 * k3(i, j) + c44 * k4(i, j) + c45 * k5(i, j) + c46 * k6(i, j) + c47 * k7(i, j));
                rho_5(i, j) = rho_tmp(i, j);
                rho_4_norm += std::norm(rho_4(i, j));
                rho_5_norm += std::norm(rho_5(i, j));
                overlap += std::conj(rho_4(i, j)) * rho_5(i, j);
            }
        }
        rho_4_norm = std::sqrt(std::abs(rho_4_norm));
        rho_5_norm = std::sqrt(std::abs(rho_5_norm));
    } else {
    // Whilst for density matrices we just use the trace
    #if defined(_OPENMP)
    #pragma omp parallel for reduction(complex_double_reduction : overlap, rho_4_norm, rho_5_norm, rho_5_trace) schedule(static)
    #endif
        for (long i = 0; i < rho_rows; ++i) {
            for (long j = 0; j < rho_cols; ++j) {
                rho_4(i, j) = rho_t(i, j) + dt * (c41 * k1(i, j) + c43 * k3(i, j) + c44 * k4(i, j) + c45 * k5(i, j) + c46 * k6(i, j) + c47 * k7(i, j));
                rho_5(i, j) = rho_tmp(i, j);
                overlap += std::conj(rho_4(i, j)) * rho_5(i, j);
                rho_4_norm += std::norm(rho_4(i, j));
                rho_5_norm += std::norm(rho_5(i, j));
            }
            rho_5_trace += rho_5(i, i);
        }
        rho_4_norm = std::sqrt(std::abs(rho_4_norm));
        rho_5_norm = std::sqrt(std::abs(rho_5_norm));
    }

    std::cout << "rho4: \n" << rho_4 << "\nrho5: \n" << rho_5 << "\n";
    std::cout << "overlap: " << overlap << ", rho_4_norm: " << rho_4_norm << ", rho_5_norm: " << rho_5_norm << std::endl;

    // Make sure it's not zero
    if (std::abs(rho_4_norm) < 1e-14 || std::abs(rho_5_norm) < 1e-14) {
        throw py::value_error("4th order solution has zero norm, cannot perform adaptive step.");
    }

    overlap /= rho_4_norm;
    overlap /= rho_5_norm;
    double err_norm = std::sqrt(std::abs(1.0 - std::pow(std::abs(overlap), 2)));
    err_norm /= tol;

    // Normalize the 5th order solution
    // if (!is_unitary_on_statevector) {
    //     rho_5_norm *= std::abs(rho_5_trace);
    // }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < rho_rows; ++i) {
        for (long j = 0; j < rho_cols; ++j) {
            rho_5(i, j) /= rho_5_norm;
        }
    }

    // Whether we accept the time step or not
    double dt_taken = 0.0;
    if (err_norm <= 1.0) {
        rho_t = rho_5;
        k_saved.resize(rho_rows, rho_cols);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < rho_rows; ++i) {
            for (long j = 0; j < rho_cols; ++j) {
                k_saved(i, j) = k7(i, j) / rho_5_norm;
            }
        }
        dt_taken = dt;
    }

    // Scale the time step
    double factor = 0.9 * std::pow(err_norm, -0.2);
    factor = std::max(min_factor, std::min(max_factor, factor));
    dt *= factor;

    std::cout << "err_norm: " << err_norm << ", factor: " << factor << ", new dt: " << dt << ", dt taken: " << dt_taken << std::endl;

    return dt_taken;
}

// GCOV_EXCL_BR_STOP