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

#include "environment.h"

#include <algorithm>
#include <cmath>
#include "../../../libs/pybind.h"

// GCOV_EXCL_BR_START

EnvironmentCpp::EnvironmentCpp(int n_system_qubits_, int n_environment_qubits_, const SparseMatrix& hamiltonian_, const MatrixFreeHamiltonian& hamiltonian_matrix_free_, const std::vector<SparseMatrix>& jump_operators_, const SparseMatrix& initial_state_) : n_system_qubits(n_system_qubits_), n_environment_qubits(n_environment_qubits_), hamiltonian(hamiltonian_), hamiltonian_matrix_free(hamiltonian_matrix_free_), jump_operators(jump_operators_), initial_state(initial_state_) {}

int EnvironmentCpp::get_n_system_qubits() const {
    return n_system_qubits;
}

int EnvironmentCpp::get_n_environment_qubits() const {
    return n_environment_qubits;
}

int EnvironmentCpp::get_n_total_qubits() const {
    return n_system_qubits + n_environment_qubits;
}

SparseMatrix EnvironmentCpp::attach_to(const SparseMatrix& rho_system) const {
    /*
    Build the initial state of the full register, with the environment after the system.

    Args:
        rho_system (SparseMatrix): The system statevector or density matrix.

    Returns:
        SparseMatrix: The density matrix rho_system (x) rho_environment.
    */
    SparseMatrix rho = rho_system;
    if (rho.cols() == 1) {
        rho = (rho_system * rho_system.adjoint()).eval();
    }
    SparseMatrix full = Eigen::kroneckerProduct(rho, initial_state).eval();
    full.makeCompressed();
    return full;
}

DenseMatrix EnvironmentCpp::trace_out(const DenseMatrix& rho) const {
    /*
    Trace out the environment qubits, leaving the system state.

    Args:
        rho (DenseMatrix): The statevector or density matrix of the full register.

    Returns:
        DenseMatrix: The reduced density matrix of the system qubits.
    */
    const long dim_system = 1L << n_system_qubits;
    const long dim_environment = 1L << n_environment_qubits;

    // Amplitude (i, k) of system state i and environment state k is at row i * dim_environment + k
    if (rho.cols() == 1) {
        Eigen::Map<const DenseMatrix> amplitudes(rho.data(), dim_environment, dim_system);
        return amplitudes.transpose() * amplitudes.conjugate();
    }

    // Trace out the environment by summing over its degrees of freedom
    DenseMatrix reduced = DenseMatrix::Zero(dim_system, dim_system);
    for (long i = 0; i < dim_system; ++i) {
        for (long j = 0; j < dim_system; ++j) {
            reduced(i, j) = rho.block(i * dim_environment, j * dim_environment, dim_environment, dim_environment).trace();
        }
    }
    return reduced;
}

void EnvironmentCpp::add_to_evolution(std::vector<SparseMatrix>& hamiltonians, std::vector<std::vector<double>>& parameters_list, NoiseModelCpp& noise_model_cpp) const {
    /*
    Add the environment to an evolution on the full register: the coupling Hamiltonian with a
    constant unit coefficient at every step, and the environment jump operators.

    Args:
        hamiltonians (std::vector<SparseMatrix>&): The Hamiltonians of the evolution, on the full register.
        parameters_list (std::vector<std::vector<double>>&): The per-step coefficient of each Hamiltonian.
        noise_model_cpp (NoiseModelCpp&): The Markovian noise of the evolution, on the full register.
    */
    size_t n_steps = parameters_list.empty() ? 0 : parameters_list.front().size();
    hamiltonians.push_back(hamiltonian);
    parameters_list.emplace_back(n_steps, 1.0);
    for (const auto& L : jump_operators) {
        noise_model_cpp.add_jump_operator(L);
    }
}

void EnvironmentCpp::add_to_evolution(std::vector<MatrixFreeHamiltonian>& hamiltonians, std::vector<std::vector<double>>& parameters_list, NoiseModelCpp& noise_model_cpp) const {
    /*
    Add the environment to a matrix-free evolution on the full register: the coupling Hamiltonian
    with a constant unit coefficient at every step, and the environment jump operators.

    Args:
        hamiltonians (std::vector<MatrixFreeHamiltonian>&): The Hamiltonians of the evolution, on the full register.
        parameters_list (std::vector<std::vector<double>>&): The per-step coefficient of each Hamiltonian.
        noise_model_cpp (NoiseModelCpp&): The Markovian noise of the evolution, on the full register.
    */
    size_t n_steps = parameters_list.empty() ? 0 : parameters_list.front().size();
    hamiltonians.push_back(hamiltonian_matrix_free);
    parameters_list.emplace_back(n_steps, 1.0);
    for (const auto& L : jump_operators) {
        noise_model_cpp.add_jump_operator(L);
    }
}

void circuit_to_schedule(const std::vector<Gate>& gates, const std::map<std::string, float>& gate_durations, int n_total_qubits, double dt, std::vector<SparseMatrix>& hamiltonians, std::vector<std::vector<double>>& parameters_list, std::vector<double>& step_list) {
    /*
    Lower a circuit to a piecewise-constant schedule, so it can be evolved together with the environment.
    Gates run one after another; gate k is generated by H_k = i log(U_k) / T_k during its duration T_k, and
    every other qubit idles meanwhile. The logarithm is taken through the Schur form U_k = Q diag(e^{i phi}) Q^dag
    on the gate's own qubits, giving the Hermitian H_k = -Q diag(phi) Q^dag / T_k. The eigenphases phi are taken
    in (-pi, pi], which fixes the direction a gate with an eigenvalue of -1 (X, Z, H, CNOT, ...) rotates in while
    the environment is coupled.

    Args:
        gates (std::vector<Gate>): The gates of the circuit, in order, without measurements.
        gate_durations (std::map<std::string, float>): Per-gate noise key -> execution time (see resolve_gate_durations).
        n_total_qubits (int): The number of qubits of the full register.
        dt (double): The maximum time step within a gate.
        hamiltonians (std::vector<SparseMatrix>&): Output, the generator of each gate on the full register.
        parameters_list (std::vector<std::vector<double>>&): Output, 1 during the gate's duration and 0 elsewhere.
        step_list (std::vector<double>&): Output, the end time of each step of the schedule.

    Raises:
        py::value_error: If a gate is not unitary.
    */

    // For each gate, convert it to a piecewise-constant schedule
    std::vector<std::pair<size_t, size_t>> gate_steps;
    double time = 0.0;
    for (const auto& gate : gates) {
        // Get the local unitary e.g. a CNOT on qubit 2 and 5 needs a 4x4 matrix, not a 64x64 matrix
        std::vector<int> qubits = gate.get_qubits();
        std::sort(qubits.begin(), qubits.end());
        auto local_index = [&](const std::vector<int>& original) {
            std::vector<int> local;
            for (int q : original) {
                local.push_back(int(std::lower_bound(qubits.begin(), qubits.end(), q) - qubits.begin()));
            }
            return local;
        };
        Gate local_gate(gate.get_name(), gate.get_base_matrix(), local_index(gate.get_control_qubits()), local_index(gate.get_target_qubits()), {});
        DenseMatrix unitary = DenseMatrix(local_gate.get_full_matrix(int(qubits.size())));
        if (!(unitary.adjoint() * unitary).isApprox(DenseMatrix::Identity(unitary.rows(), unitary.cols()), 1e-9)) {
            throw py::value_error("Non-Markovian noise only supports unitary gates, but " + gate.get_name() + " is not unitary.");
        }

        // H = -Q diag(phi) Q^dag / T, embedded on the gate's qubits of the full register
        double duration = gate_durations.at(NoiseModelCpp::make_gate_key(gate.get_name(), int(gate.get_control_qubits().size())));
        Eigen::ComplexSchur<DenseMatrix> schur(unitary);
        // Eigenphases in (-pi, pi], so an eigenvalue of -1 always gives +pi instead of flipping sign with rounding
        auto minus_phase = [](const Complex& z) {
            double phi = std::arg(z);
            if (phi <= -M_PI + 1e-9) {
                phi += 2.0 * M_PI;
            }
            return Complex(-phi, 0.0);
        };
        DenseMatrix phases = schur.matrixT().diagonal().unaryExpr(minus_phase).asDiagonal();
        DenseMatrix generator = schur.matrixU() * phases * schur.matrixU().adjoint() / duration;
        SparseMatrix generator_sparse = generator.sparseView();
        hamiltonians.push_back(Gate("generator", generator_sparse, {}, qubits, {}).get_full_matrix(n_total_qubits));

        // Split the duration into equal steps no longer than dt
        size_t n_steps = std::max<size_t>(1, size_t(std::ceil(duration / dt - 1e-9)));
        gate_steps.emplace_back(step_list.size(), n_steps);
        for (size_t j = 1; j <= n_steps; ++j) {
            step_list.push_back(time + duration * double(j) / double(n_steps));
        }
        time += duration;
    }

    // Each gate is on only during its own steps
    for (const auto& [first_step, n_steps] : gate_steps) {
        std::vector<double> parameters(step_list.size(), 0.0);
        std::fill(parameters.begin() + long(first_step), parameters.begin() + long(first_step + n_steps), 1.0);
        parameters_list.push_back(parameters);
    }
}

// GCOV_EXCL_BR_STOP
