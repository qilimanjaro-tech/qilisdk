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

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>
#include "../../../src/qilisdk_cpp/backends/qilisim/environment/environment.h"
#include "../../../src/qilisdk_cpp/libs/pybind.h"

#include <cmath>
#include <complex>
#include <map>
#include <string>
#include <vector>

namespace {

SparseMatrix to_sparse(const DenseMatrix& dense) {
    SparseMatrix sparse = dense.sparseView();
    sparse.makeCompressed();
    return sparse;
}

DenseMatrix projector(int dim, int index) {
    DenseMatrix rho = DenseMatrix::Zero(dim, dim);
    rho(index, index) = 1.0;
    return rho;
}

DenseMatrix plus_ket() {
    DenseMatrix ket(2, 1);
    ket << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
    return ket;
}

DenseMatrix pauli_x() {
    DenseMatrix x(2, 2);
    x << 0.0, 1.0, 1.0, 0.0;
    return x;
}

DenseMatrix pauli_z() {
    DenseMatrix z(2, 2);
    z << 1.0, 0.0, 0.0, -1.0;
    return z;
}

DenseMatrix hadamard() {
    DenseMatrix h(2, 2);
    h << 1.0, 1.0, 1.0, -1.0;
    return h / std::sqrt(2.0);
}

DenseMatrix kron(const DenseMatrix& a, const DenseMatrix& b) {
    return Eigen::kroneckerProduct(a, b).eval();
}

// A generic single-qubit mixed state
DenseMatrix mixed_state() {
    DenseMatrix rho(2, 2);
    rho << 0.7, Complex(0.1, -0.2), Complex(0.1, 0.2), 0.3;
    return rho;
}

// One system qubit, one environment qubit in |1><1|, with a ZZ coupling and a single jump operator
EnvironmentCpp single_qubit_environment() {
    SparseMatrix hamiltonian = to_sparse(kron(pauli_z(), pauli_z()));
    MatrixFreeHamiltonian hamiltonian_matrix_free(2, PauliString(2, {MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)}));
    SparseMatrix jump = to_sparse(kron(DenseMatrix::Identity(2, 2), pauli_x()));
    return EnvironmentCpp(1, 1, hamiltonian, hamiltonian_matrix_free, {jump}, to_sparse(projector(2, 1)));
}

// Equal up to a global phase
bool equal_up_to_phase(const DenseMatrix& a, const DenseMatrix& b, double tol) {
    Complex overlap = (b.adjoint() * a).trace();
    if (std::abs(overlap) < tol) {
        return a.norm() < tol && b.norm() < tol;
    }
    Complex phase = overlap / std::abs(overlap);
    return (a - phase * b).norm() < tol;
}

}  // namespace

TEST(Environment, QubitCounts) {
    SparseMatrix hamiltonian(8, 8);
    EnvironmentCpp environment(1, 2, hamiltonian, MatrixFreeHamiltonian(3), {}, to_sparse(projector(4, 0)));

    EXPECT_EQ(environment.get_n_system_qubits(), 1);
    EXPECT_EQ(environment.get_n_environment_qubits(), 2);
    EXPECT_EQ(environment.get_n_total_qubits(), 3);
}

TEST(Environment, AttachToKetGivesProductDensityMatrix) {
    EnvironmentCpp environment = single_qubit_environment();
    DenseMatrix ket = plus_ket();

    DenseMatrix rho = DenseMatrix(environment.attach_to(to_sparse(ket)));

    DenseMatrix expected = kron(ket * ket.adjoint(), projector(2, 1));
    ASSERT_EQ(rho.rows(), 4);
    ASSERT_EQ(rho.cols(), 4);
    EXPECT_TRUE(rho.isApprox(expected, 1e-12));
}

TEST(Environment, AttachToDensityMatrixGivesProductDensityMatrix) {
    EnvironmentCpp environment = single_qubit_environment();

    DenseMatrix rho = DenseMatrix(environment.attach_to(to_sparse(mixed_state())));

    ASSERT_EQ(rho.rows(), 4);
    ASSERT_EQ(rho.cols(), 4);
    EXPECT_TRUE(rho.isApprox(kron(mixed_state(), projector(2, 1)), 1e-12));
}

TEST(Environment, TraceOutProductState) {
    EnvironmentCpp environment = single_qubit_environment();
    DenseMatrix rho = kron(mixed_state(), projector(2, 1));

    DenseMatrix reduced = environment.trace_out(rho);

    ASSERT_EQ(reduced.rows(), 2);
    ASSERT_EQ(reduced.cols(), 2);
    EXPECT_TRUE(reduced.isApprox(mixed_state(), 1e-12));
}

TEST(Environment, TraceOutEntangledStateIsMixed) {
    EnvironmentCpp environment = single_qubit_environment();
    DenseMatrix bell = DenseMatrix::Zero(4, 1);
    bell(0, 0) = 1.0 / std::sqrt(2.0);
    bell(3, 0) = 1.0 / std::sqrt(2.0);

    DenseMatrix reduced = environment.trace_out(bell * bell.adjoint());

    ASSERT_EQ(reduced.rows(), 2);
    ASSERT_EQ(reduced.cols(), 2);
    EXPECT_TRUE(reduced.isApprox(0.5 * DenseMatrix::Identity(2, 2), 1e-12));
}

TEST(Environment, TraceOutKeepsSystemQubitOrder) {
    // Two system qubits in |0><0| (x) rho, then two environment qubits
    SparseMatrix hamiltonian(16, 16);
    EnvironmentCpp environment(2, 2, hamiltonian, MatrixFreeHamiltonian(4), {}, to_sparse(projector(4, 0)));
    DenseMatrix system = kron(projector(2, 0), mixed_state());
    DenseMatrix rho = kron(system, kron(mixed_state(), projector(2, 1)));

    DenseMatrix reduced = environment.trace_out(rho);

    ASSERT_EQ(reduced.rows(), 4);
    ASSERT_EQ(reduced.cols(), 4);
    EXPECT_TRUE(reduced.isApprox(system, 1e-12));
}

TEST(Environment, TraceOutStatevector) {
    EnvironmentCpp environment = single_qubit_environment();
    DenseMatrix env_one = DenseMatrix::Zero(2, 1);
    env_one(1, 0) = 1.0;

    DenseMatrix reduced = environment.trace_out(kron(plus_ket(), env_one));

    ASSERT_EQ(reduced.rows(), 2);
    ASSERT_EQ(reduced.cols(), 2);
    EXPECT_TRUE(reduced.isApprox(plus_ket() * plus_ket().adjoint(), 1e-12));
}

TEST(Environment, TraceOutEntangledStatevectorIsMixed) {
    EnvironmentCpp environment = single_qubit_environment();
    DenseMatrix bell = DenseMatrix::Zero(4, 1);
    bell(0, 0) = 1.0 / std::sqrt(2.0);
    bell(3, 0) = 1.0 / std::sqrt(2.0);

    DenseMatrix reduced = environment.trace_out(bell);

    ASSERT_EQ(reduced.rows(), 2);
    ASSERT_EQ(reduced.cols(), 2);
    EXPECT_TRUE(reduced.isApprox(0.5 * DenseMatrix::Identity(2, 2), 1e-12));
}

TEST(Environment, AddToEvolutionAppendsConstantHamiltonianAndJumps) {
    EnvironmentCpp environment = single_qubit_environment();
    std::vector<SparseMatrix> hamiltonians = {to_sparse(kron(pauli_x(), DenseMatrix::Identity(2, 2)))};
    std::vector<std::vector<double>> parameters_list = {{0.0, 0.5, 1.0}};
    NoiseModelCpp noise_model;

    environment.add_to_evolution(hamiltonians, parameters_list, noise_model);

    ASSERT_EQ(hamiltonians.size(), 2u);
    ASSERT_EQ(parameters_list.size(), 2u);
    ASSERT_EQ(hamiltonians[1].rows(), 4);
    EXPECT_TRUE(DenseMatrix(hamiltonians[1]).isApprox(kron(pauli_z(), pauli_z()), 1e-12));
    EXPECT_EQ(parameters_list[1], std::vector<double>({1.0, 1.0, 1.0}));
    ASSERT_EQ(noise_model.get_jump_operators().size(), 1u);
    ASSERT_EQ(noise_model.get_jump_operators()[0].rows(), 4);
    EXPECT_TRUE(DenseMatrix(noise_model.get_jump_operators()[0]).isApprox(kron(DenseMatrix::Identity(2, 2), pauli_x()), 1e-12));
}

TEST(Environment, AddToMatrixFreeEvolutionAppendsConstantHamiltonianAndJumps) {
    EnvironmentCpp environment = single_qubit_environment();
    std::vector<MatrixFreeHamiltonian> hamiltonians = {MatrixFreeHamiltonian(2, PauliString(2, 'X', 0))};
    std::vector<std::vector<double>> parameters_list = {{0.0, 0.5, 1.0}};
    NoiseModelCpp noise_model;

    environment.add_to_evolution(hamiltonians, parameters_list, noise_model);

    ASSERT_EQ(hamiltonians.size(), 2u);
    ASSERT_EQ(parameters_list.size(), 2u);
    EXPECT_TRUE(hamiltonians[1] == MatrixFreeHamiltonian(2, PauliString(2, {MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)})));
    EXPECT_EQ(parameters_list[1], std::vector<double>({1.0, 1.0, 1.0}));
    EXPECT_EQ(noise_model.get_jump_operators().size(), 1u);
}

TEST(Environment, AddToEvolutionKeepsExistingJumps) {
    EnvironmentCpp environment = single_qubit_environment();
    std::vector<SparseMatrix> hamiltonians = {to_sparse(kron(pauli_x(), DenseMatrix::Identity(2, 2)))};
    std::vector<std::vector<double>> parameters_list = {{1.0, 1.0}};
    NoiseModelCpp noise_model;
    noise_model.add_jump_operator(to_sparse(kron(pauli_z(), DenseMatrix::Identity(2, 2))));

    environment.add_to_evolution(hamiltonians, parameters_list, noise_model);

    EXPECT_EQ(noise_model.get_jump_operators().size(), 2u);
    EXPECT_EQ(noise_model.get_jump_rate_series().size(), 2u);
}

TEST(NoiseModel, ExtendRegisterAppendsIdentities) {
    NoiseModelCpp noise_model;
    noise_model.add_jump_operator(to_sparse(pauli_x()));
    noise_model.add_jump_operator(to_sparse(pauli_z()), {1.0, 0.5});

    noise_model.extend_register(2);

    const std::vector<SparseMatrix>& jumps = noise_model.get_jump_operators();
    ASSERT_EQ(jumps.size(), 2u);
    ASSERT_EQ(jumps[0].rows(), 8);
    ASSERT_EQ(jumps[1].rows(), 8);
    EXPECT_TRUE(DenseMatrix(jumps[0]).isApprox(kron(pauli_x(), DenseMatrix::Identity(4, 4)), 1e-12));
    EXPECT_TRUE(DenseMatrix(jumps[1]).isApprox(kron(pauli_z(), DenseMatrix::Identity(4, 4)), 1e-12));
    EXPECT_EQ(noise_model.get_jump_rate_series()[1], std::vector<double>({1.0, 0.5}));
}

TEST(NoiseModel, ExtendRegisterByZeroIsNoOp) {
    NoiseModelCpp noise_model;
    noise_model.add_jump_operator(to_sparse(pauli_x()));

    noise_model.extend_register(0);

    EXPECT_TRUE(DenseMatrix(noise_model.get_jump_operators()[0]).isApprox(pauli_x(), 1e-12));
}

TEST(Environment, CircuitToScheduleReproducesGates) {
    // X then H on the system qubit, with one environment qubit idling after it
    std::vector<Gate> gates = {Gate("X", to_sparse(pauli_x()), {}, {0}, {}), Gate("H", to_sparse(hadamard()), {}, {0}, {})};
    std::map<std::string, float> gate_durations = {{NoiseModelCpp::make_gate_key("X", 0), 2.0f}, {NoiseModelCpp::make_gate_key("H", 0), 1.0f}};
    double dt = 0.5;
    std::vector<SparseMatrix> hamiltonians;
    std::vector<std::vector<double>> parameters_list;
    std::vector<double> step_list;

    circuit_to_schedule(gates, gate_durations, 2, dt, hamiltonians, parameters_list, step_list);

    ASSERT_EQ(hamiltonians.size(), 2u);
    ASSERT_EQ(parameters_list.size(), 2u);
    ASSERT_FALSE(step_list.empty());
    EXPECT_NEAR(step_list.back(), 3.0, 1e-9);
    for (size_t i = 1; i < step_list.size(); ++i) {
        EXPECT_GT(step_list[i], step_list[i - 1]);
        EXPECT_LE(step_list[i] - step_list[i - 1], dt + 1e-9);
    }

    // Each generator, switched on for its gate's duration, reproduces the gate on the full register
    std::vector<DenseMatrix> expected = {kron(pauli_x(), DenseMatrix::Identity(2, 2)), kron(hadamard(), DenseMatrix::Identity(2, 2))};
    std::vector<double> durations = {2.0, 1.0};
    for (size_t k = 0; k < hamiltonians.size(); ++k) {
        DenseMatrix generator = DenseMatrix(hamiltonians[k]);
        ASSERT_EQ(generator.rows(), 4);
        EXPECT_TRUE(generator.isApprox(generator.adjoint(), 1e-12));
        DenseMatrix unitary = (Complex(0.0, -durations[k]) * generator).exp();
        EXPECT_TRUE(equal_up_to_phase(unitary, expected[k], 1e-9));
    }

    // Gate k is on strictly inside its own interval and off strictly inside the other one
    ASSERT_EQ(parameters_list[0].size(), step_list.size());
    ASSERT_EQ(parameters_list[1].size(), step_list.size());
    for (size_t i = 0; i < step_list.size(); ++i) {
        double t = step_list[i];
        if (t > 1e-9 && t < 2.0 - 1e-9) {
            EXPECT_DOUBLE_EQ(parameters_list[0][i], 1.0);
            EXPECT_DOUBLE_EQ(parameters_list[1][i], 0.0);
        } else if (t > 2.0 + 1e-9 && t < 3.0 - 1e-9) {
            EXPECT_DOUBLE_EQ(parameters_list[0][i], 0.0);
            EXPECT_DOUBLE_EQ(parameters_list[1][i], 1.0);
        }
    }
}

TEST(Environment, CircuitToScheduleRejectsNonUnitaryGates) {
    std::vector<Gate> gates = {Gate("X", to_sparse(2.0 * pauli_x()), {}, {0}, {})};
    std::map<std::string, float> gate_durations = {{NoiseModelCpp::make_gate_key("X", 0), 1.0f}};
    std::vector<SparseMatrix> hamiltonians;
    std::vector<std::vector<double>> parameters_list;
    std::vector<double> step_list;

    EXPECT_ANY_THROW(circuit_to_schedule(gates, gate_durations, 1, 1.0, hamiltonians, parameters_list, step_list));
}

TEST(Environment, CircuitToScheduleControlledGate) {
    // CNOT controlled on qubit 1 targeting qubit 0, then an idle environment qubit
    std::vector<Gate> gates = {Gate("X", to_sparse(pauli_x()), {1}, {0}, {})};
    std::map<std::string, float> gate_durations = {{NoiseModelCpp::make_gate_key("X", 1), 1.0f}};
    std::vector<SparseMatrix> hamiltonians;
    std::vector<std::vector<double>> parameters_list;
    std::vector<double> step_list;

    circuit_to_schedule(gates, gate_durations, 3, 1.0, hamiltonians, parameters_list, step_list);

    ASSERT_EQ(hamiltonians.size(), 1u);
    DenseMatrix unitary = (Complex(0.0, -1.0) * DenseMatrix(hamiltonians[0])).exp();
    DenseMatrix expected = DenseMatrix(gates[0].get_full_matrix(3));
    EXPECT_TRUE(equal_up_to_phase(unitary, expected, 1e-9));
}

TEST(Environment, CircuitToScheduleEmptyCircuit) {
    std::vector<SparseMatrix> hamiltonians;
    std::vector<std::vector<double>> parameters_list;
    std::vector<double> step_list;

    circuit_to_schedule({}, {}, 2, 0.5, hamiltonians, parameters_list, step_list);

    EXPECT_TRUE(hamiltonians.empty());
    EXPECT_TRUE(parameters_list.empty());
}

// GCOV_EXCL_BR_STOP
