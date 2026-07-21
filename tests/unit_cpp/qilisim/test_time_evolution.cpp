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
#include "../../../src/qilisdk_cpp/backends/qilisim/analog/time_evolution.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/utils/matrix_utils.h"

namespace {

constexpr double kTol = 1e-6;
constexpr double kTolLoose = 1e-3;

SparseMatrix to_sparse(const DenseMatrix& M) {
    SparseMatrix S(M.rows(), M.cols());
    S = M.sparseView();
    return S;
}

DenseMatrix pauli_z() {
    DenseMatrix Z(2, 2);
    Z << 1, 0, 0, -1;
    return Z;
}

DenseMatrix pauli_x() {
    DenseMatrix X(2, 2);
    X << 0, 1, 1, 0;
    return X;
}

SparseMatrix pure_zero_sparse() {
    DenseMatrix r = DenseMatrix::Zero(2, 2);
    r(0, 0) = 1.0;
    return to_sparse(r);
}

SparseMatrix pure_plus_sparse() {
    DenseMatrix r(2, 2);
    r << 0.5, 0.5, 0.5, 0.5;
    return to_sparse(r);
}

SparseMatrix statevector_zero_sparse() {
    DenseMatrix v = DenseMatrix::Zero(2, 1);
    v(0, 0) = 1.0;
    return to_sparse(v);
}

SparseMatrix amp_damp_jump() {
    DenseMatrix j = DenseMatrix::Zero(2, 2);
    j(0, 1) = 1.0;
    return to_sparse(j);
}

// A statevector whose first amplitude is NaN, used to feed an already-diverged state into an
// integrator. Built with insert() rather than sparseView(), which would prune the NaN entry.
SparseMatrix nan_statevector_sparse() {
    SparseMatrix v(2, 1);
    v.insert(0, 0) = std::numeric_limits<double>::quiet_NaN();
    v.makeCompressed();
    return v;
}

// An amplitude-damping jump operator with an enormous rate. Its dissipator (L rho L^dagger) pumps
// the diagonal to +inf, overflowing the trace within a single step. A huge Hamiltonian would not
// do this on its own: the trace of a commutator is zero, so unitary dynamics leaves the trace
// finite even as individual entries diverge.
SparseMatrix huge_amp_damp_jump() {
    DenseMatrix j = DenseMatrix::Zero(2, 2);
    j(0, 1) = 1e300;
    return to_sparse(j);
}

MatrixFreeHamiltonian make_matrix_free_H(std::complex<double> coeff, int target_qubit, std::string name) {
    MatrixFreeOperator op(name, {}, {target_qubit}, DenseMatrix());
    return MatrixFreeHamiltonian(1, op, coeff);
}

struct TimeEvolutionOutputs {
    DenseMatrix rho_t;
    std::vector<DenseMatrix> intermediate_rhos;
    std::vector<double> expectation_values;
    std::vector<std::vector<double>> intermediate_expectation_values;
};

TimeEvolutionOutputs run_time_evolution(SparseMatrix rho_0, const std::vector<SparseMatrix>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model, const std::vector<SparseMatrix>& observables, QiliSimConfig& config) {
    TimeEvolutionOutputs out;
    time_evolution(rho_0, hamiltonians, parameters_list, step_list, noise_model, config, out.rho_t, out.intermediate_rhos);

    for (const auto& O : observables) {
        if (out.rho_t.cols() == 1) {
            out.expectation_values.push_back(std::real(dot(out.rho_t, O * out.rho_t)));
        } else {
            out.expectation_values.push_back(std::real(dot(O, out.rho_t)));
        }
    }

    if (config.get_store_intermediate_results()) {
        for (const auto& rho_intermediate : out.intermediate_rhos) {
            std::vector<double> step_expectation_values;
            for (const auto& O : observables) {
                if (rho_intermediate.cols() == 1) {
                    DenseMatrix rho_dense(rho_intermediate);
                    step_expectation_values.push_back(std::real(dot(rho_dense, O * rho_dense)));
                } else {
                    step_expectation_values.push_back(std::real(dot(O, rho_intermediate)));
                }
            }
            out.intermediate_expectation_values.push_back(step_expectation_values);
        }
    }
    return out;
}

struct MatrixFreeOutputs {
    DenseMatrix rho_t;
    std::vector<DenseMatrix> intermediate_rhos;
    std::vector<double> expectation_values;
    std::vector<std::vector<double>> intermediate_expectation_values;
};

MatrixFreeOutputs run_time_evolution_mf(SparseMatrix rho_0, const std::vector<MatrixFreeHamiltonian>& hamiltonians, const std::vector<std::vector<double>>& parameters_list, const std::vector<double>& step_list, NoiseModelCpp& noise_model, const std::vector<MatrixFreeHamiltonian>& observables, QiliSimConfig& config) {
    MatrixFreeOutputs out;
    time_evolution_matrix_free(rho_0, hamiltonians, parameters_list, step_list, noise_model, config, out.rho_t, out.intermediate_rhos);

    // Apply the operators using the Born rule
    for (const auto& O : observables) {
        out.expectation_values.push_back(O.expectation_value(out.rho_t));
    }

    // If we have intermediates, process them too
    if (config.get_store_intermediate_results()) {
        out.intermediate_expectation_values.resize(out.intermediate_rhos.size());
        for (size_t step = 0; step < out.intermediate_rhos.size(); ++step) {
            const auto& rho_intermediate = out.intermediate_rhos[step];
            std::vector<double> step_expectation_values(observables.size());
            for (size_t i = 0; i < observables.size(); ++i) {
                step_expectation_values[i] = observables[i].expectation_value(rho_intermediate);
            }
            out.intermediate_expectation_values[step] = step_expectation_values;
        }
    }
    return out;
}

}  // namespace

class TimeEvolutionTest : public ::testing::Test {
   protected:
    SparseMatrix H_z = to_sparse(0.5 * pauli_z());
    std::vector<SparseMatrix> hamiltonians = {H_z};
    std::vector<std::vector<double>> params = {{1.0, 1.0, 1.0}};
    std::vector<double> steps = {0.1, 0.2, 0.3};
    NoiseModelCpp empty_noise;
    QiliSimConfig config;
    virtual void SetUp() override { config.set_time_evolution_method("integrate_rk4"); }
};

class TimeEvolutionMatrixFreeTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5, 0, "Z");
    std::vector<MatrixFreeHamiltonian> hamiltonians = {H_mf};
    std::vector<std::vector<double>> params = {{1.0, 1.0, 1.0}};
    std::vector<double> steps = {0.1, 0.2, 0.3};
    NoiseModelCpp empty_noise;
    QiliSimConfig config;
    virtual void SetUp() override { config.set_time_evolution_method("integrate_rk4_matrix_free"); }
};

TEST_F(TimeEvolutionTest, DefaultConfigDoesNotThrow) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    SUCCEED();
}

TEST_F(TimeEvolutionTest, BadMethodThrows) {
    config.set_time_evolution_method("non_existent_method");
    EXPECT_ANY_THROW(run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config));
}

TEST_F(TimeEvolutionMatrixFreeTest, DefaultConfigDoesNotThrow) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    SUCCEED();
}

TEST_F(TimeEvolutionMatrixFreeTest, NonMatrixFreeMethodThrows) {
    // A method that passes validation but is not one of the matrix-free branches
    // must fall through to the invalid-method error.
    config.set_time_evolution_method("direct");
    EXPECT_ANY_THROW(run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config));
}

class TimeEvolutionArnoldiMatrixFreeTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5, 0, "Z");
    std::vector<MatrixFreeHamiltonian> hamiltonians = {H_mf};
    std::vector<std::vector<double>> params = {{1.0, 1.0, 1.0}};
    std::vector<double> steps = {0.1, 0.2, 0.3};
    NoiseModelCpp empty_noise;
    QiliSimConfig config;
    virtual void SetUp() override { config.set_time_evolution_method("arnoldi_matrix_free"); }
};

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, DensityMatrixTracePreserved) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, StatevectorInputEvolves) {
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
}

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, WithJumpOperatorsTracePreserved) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, StoreIntermediateFromPureDensityMatrix) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
}

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, StoreIntermediateFromStatevector) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
}

TEST_F(TimeEvolutionTest, FinalStateDimensionMatchesDensityMatrixInput) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
}

TEST_F(TimeEvolutionTest, FinalStateDimensionMatchesStatevectorInput) {
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
}

TEST_F(TimeEvolutionTest, FinalStateDimensionMatchesStatevectorInputBra) {
    SparseMatrix bra = statevector_zero_sparse().adjoint();
    auto out = run_time_evolution(bra, hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
}

TEST_F(TimeEvolutionMatrixFreeTest, FinalStateDimensionMatchesStatevectorInputBra) {
    SparseMatrix bra = statevector_zero_sparse().adjoint();
    auto out = run_time_evolution_mf(bra, hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
}

TEST_F(TimeEvolutionMatrixFreeTest, FinalStateDimensionMatchesDensityMatrixInput) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
}

TEST_F(TimeEvolutionTest, TracePreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, TracePreservedUnitaryStatevector) {
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, TracePreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, TracePreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, TracePreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, HermiticityPreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionTest, HermiticityPreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, HermiticityPreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, HermiticityPreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionTest, ZeroTimeStepReturnsInitialDensityMatrix) {
    std::vector<double> zero_steps = {0.0};
    std::vector<std::vector<double>> one_param = {{1.0}};
    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, one_param, zero_steps, empty_noise, {}, config);
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, ZeroTimeStepReturnsInitialDensityMatrix) {
    std::vector<double> zero_steps = {0.0};
    std::vector<std::vector<double>> one_param = {{1.0}};
    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, one_param, zero_steps, empty_noise, {}, config);
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionTest, EigenstatePopulationsUnchangedUnderHZ) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::real(out.rho_t(1, 1)), 0.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, EigenstatePopulationsUnchangedUnderHZ) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::real(out.rho_t(1, 1)), 0.0, kTol);
}

TEST_F(TimeEvolutionTest, GroundStateIsFixedPointOfAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    DenseMatrix rho_0_dense = DenseMatrix(pure_zero_sparse());
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionTest, ExcitedStateDecaysTowardGroundStateWithAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    DenseMatrix excited = DenseMatrix::Zero(2, 2);
    excited(1, 1) = 1.0;
    auto out = run_time_evolution(to_sparse(excited), hamiltonians, params, steps, noise, {}, config);
    EXPECT_GT(std::real(out.rho_t(0, 0)), 0.0);
    EXPECT_LT(std::real(out.rho_t(1, 1)), 1.0);
}

TEST_F(TimeEvolutionMatrixFreeTest, GroundStateIsFixedPointOfAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    DenseMatrix rho_0_dense = DenseMatrix(pure_zero_sparse());
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, ExcitedStateDecaysTowardGroundStateWithAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    DenseMatrix excited = DenseMatrix::Zero(2, 2);
    excited(1, 1) = 1.0;
    auto out = run_time_evolution_mf(to_sparse(excited), hamiltonians, params, steps, noise, {}, config);
    EXPECT_GT(std::real(out.rho_t(0, 0)), 0.0);
    EXPECT_LT(std::real(out.rho_t(1, 1)), 1.0);
}

TEST_F(TimeEvolutionTest, StatevectorInputProducesValidStatevectorOutput) {
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, RowVectorInputIsTransposedAndEvolved) {
    DenseMatrix row(1, 2);
    row << 1.0, 0.0;
    auto out = run_time_evolution(to_sparse(row), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, StatevectorInputWithJumpsProducesDensityMatrix) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, RowVectorInputWithJumpsProducesDensityMatrix) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    DenseMatrix row(1, 2);
    row << 1.0, 0.0;
    auto out = run_time_evolution(to_sparse(row), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, StatevectorInputProducesValidStatevectorOutput) {
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, StatevectorInputWithJumpsProducesDensityMatrix) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionTest, NoObservablesProducesEmptyExpectationValues) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.expectation_values.empty());
}

TEST_F(TimeEvolutionTest, ExpectationValueCountMatchesObservableCount) {
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z()), to_sparse(pauli_x())};
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_EQ(int(out.expectation_values.size()), 2);
}

TEST_F(TimeEvolutionTest, PauliZExpectationValueOnEigenstateIsOne) {
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z())};
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_NEAR(out.expectation_values[0], 1.0, kTol);
}

TEST_F(TimeEvolutionTest, PauliZExpectationValueOnExcitedStateIsMinusOne) {
    DenseMatrix excited = DenseMatrix::Zero(2, 2);
    excited(1, 1) = 1.0;
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z())};
    auto out = run_time_evolution(to_sparse(excited), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_NEAR(out.expectation_values[0], -1.0, kTol);
}

TEST_F(TimeEvolutionTest, ExpectationValuePreservedForEigenstateStatevectorInput) {
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z())};
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_NEAR(out.expectation_values[0], 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, NoObservablesProducesEmptyExpectationValues) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.expectation_values.empty());
}

TEST_F(TimeEvolutionMatrixFreeTest, ExpectationValueCountMatchesObservableCount) {
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    MatrixFreeHamiltonian obs_x = make_matrix_free_H(1.0, 0, "X");
    std::vector<MatrixFreeHamiltonian> obs = {obs_z, obs_x};
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_EQ(int(out.expectation_values.size()), 2);
}

TEST_F(TimeEvolutionMatrixFreeTest, PauliZExpectationValueOnEigenstateIsOne) {
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {obs_z}, config);
    EXPECT_NEAR(out.expectation_values[0], 1.0, kTol);
}

TEST_F(TimeEvolutionTest, NoIntermediatesStoredByDefault) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.intermediate_rhos.empty());
    EXPECT_TRUE(out.intermediate_expectation_values.empty());
}

TEST_F(TimeEvolutionTest, IntermediateCountMatchesStepCount) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
}

TEST_F(TimeEvolutionTest, IntermediateWithStatevector) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
}

TEST_F(TimeEvolutionTest, IntermediateExpectationValueCountMatchesStepsAndObservables) {
    config.set_store_intermediate_results(true);
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z()), to_sparse(pauli_x())};
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_EQ(int(out.intermediate_expectation_values.size()), int(steps.size()));
    for (const auto& step_vals : out.intermediate_expectation_values) {
        EXPECT_EQ(int(step_vals.size()), 2);
    }
}

TEST_F(TimeEvolutionTest, IntermediateExpectationValueCountMatchesStepsAndObservablesWithStatevector) {
    config.set_store_intermediate_results(true);
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z()), to_sparse(pauli_x())};
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_EQ(int(out.intermediate_expectation_values.size()), int(steps.size()));
    for (const auto& step_vals : out.intermediate_expectation_values) {
        EXPECT_EQ(int(step_vals.size()), 2);
    }
}

TEST_F(TimeEvolutionTest, EachIntermediateRhoHasCorrectDimension) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_EQ(rho_int.rows(), 2);
        EXPECT_EQ(rho_int.cols(), 2);
    }
}

TEST_F(TimeEvolutionTest, EachIntermediateRhoHasUnitTrace) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_NEAR(std::real(rho_int.trace()), 1.0, kTol);
    }
}

TEST_F(TimeEvolutionMatrixFreeTest, NoIntermediatesStoredByDefault) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.intermediate_rhos.empty());
    EXPECT_TRUE(out.intermediate_expectation_values.empty());
}

TEST_F(TimeEvolutionMatrixFreeTest, IntermediateCountMatchesStepCount) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
}

TEST_F(TimeEvolutionMatrixFreeTest, IntermediateExpectationValueCountMatchesStepsAndObservables) {
    config.set_store_intermediate_results(true);
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {obs_z}, config);
    EXPECT_EQ(int(out.intermediate_expectation_values.size()), int(steps.size()));
    for (const auto& step_vals : out.intermediate_expectation_values) {
        EXPECT_EQ(int(step_vals.size()), 1);
    }
}

TEST_F(TimeEvolutionMatrixFreeTest, EachIntermediateRhoHasUnitTrace) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_NEAR(std::real(rho_int.trace()), 1.0, kTol);
    }
}

class TimeEvolutionMethodTest : public ::testing::Test {
   protected:
    SparseMatrix H_z = to_sparse(0.5 * pauli_z());
    std::vector<SparseMatrix> hamiltonians = {H_z};
    std::vector<std::vector<double>> params = {{1.0}};
    std::vector<double> steps = {0.1};
    NoiseModelCpp empty_noise;
    std::vector<SparseMatrix> obs = {to_sparse(pauli_z())};
};

TEST_F(TimeEvolutionMethodTest, IntegrateMethodProducesValidStateWithMultipleThreads) {
    QiliSimConfig cfg;
    cfg.set_num_threads(2);
    cfg.set_time_evolution_method("integrate_rk4");
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMethodTest, DirectMethodProducesValidState) {
    QiliSimConfig cfg;
    cfg.set_time_evolution_method("direct");
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMethodTest, ArnoldiMethodProducesValidState) {
    QiliSimConfig cfg;
    cfg.set_time_evolution_method("arnoldi");
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMethodTest, AllMethodsAgreeToPauliZExpectationValueOnEigenstate) {
    QiliSimConfig cfg_int, cfg_dir, cfg_arn;
    cfg_int.set_time_evolution_method("integrate_rk4");
    cfg_dir.set_time_evolution_method("direct");
    cfg_arn.set_time_evolution_method("arnoldi");

    auto out_int = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg_int);
    auto out_dir = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg_dir);
    auto out_arn = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg_arn);

    EXPECT_NEAR(out_int.expectation_values[0], out_dir.expectation_values[0], kTolLoose);
    EXPECT_NEAR(out_int.expectation_values[0], out_arn.expectation_values[0], kTolLoose);
}

TEST_F(TimeEvolutionTest, ZeroParameterEquivalentToZeroHamiltonian) {
    std::vector<std::vector<double>> zero_params = {{0.0, 0.0, 0.0}};
    SparseMatrix H_zero = to_sparse(DenseMatrix::Zero(2, 2));
    std::vector<SparseMatrix> zero_hamiltonians = {H_zero};
    std::vector<double> zero_steps = {0.0, 0.0, 0.0};

    auto out_zero_param = run_time_evolution(pure_plus_sparse(), hamiltonians, zero_params, steps, empty_noise, {}, config);
    auto out_zero_H = run_time_evolution(pure_plus_sparse(), zero_hamiltonians, params, zero_steps, empty_noise, {}, config);

    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    EXPECT_TRUE(out_zero_param.rho_t.isApprox(rho_0_dense, kTol));
    EXPECT_TRUE(out_zero_H.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionTest, TwoHamiltonianTermsSumCorrectly) {
    SparseMatrix H_x = to_sparse(0.5 * pauli_x());
    std::vector<SparseMatrix> two_hamiltonians = {H_z, H_x};
    std::vector<std::vector<double>> two_params = {{1.0}, {0.0}};
    std::vector<double> one_step = {0.1};

    auto out_two = run_time_evolution(pure_plus_sparse(), two_hamiltonians, two_params, one_step, empty_noise, {}, config);
    auto out_one = run_time_evolution(pure_plus_sparse(), {H_z}, {{1.0}}, one_step, empty_noise, {}, config);

    EXPECT_TRUE(out_two.rho_t.isApprox(out_one.rho_t, kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, ZeroParameterLeavesStateUnchanged) {
    std::vector<std::vector<double>> zero_params = {{0.0, 0.0, 0.0}};
    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, zero_params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionMatrixFreeTest, IntermediateResultsWithStatevectorInput) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(int(out.intermediate_rhos.size()), int(steps.size()));
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_EQ(rho_int.rows(), 2);
        EXPECT_EQ(rho_int.cols(), 1);
        EXPECT_NEAR(std::real(rho_int.norm()), 1.0, kTol);
    }
}

class TimeEvolutionMonteCarloTest : public ::testing::Test {
   protected:
    SparseMatrix H_z = to_sparse(0.5 * pauli_z());
    std::vector<SparseMatrix> hamiltonians = {H_z};
    std::vector<std::vector<double>> params = {{1.0}};
    std::vector<double> steps = {0.1};
    NoiseModelCpp noise;
    QiliSimConfig config;

    void SetUp() override {
        noise.add_jump_operator(amp_damp_jump());
        config.set_monte_carlo(true);
        config.set_num_monte_carlo_trajectories(200);
        config.set_time_evolution_method("integrate_rk4");
        config.set_seed(0);
    }
};

TEST_F(TimeEvolutionMonteCarloTest, FinalStateIsDensityMatrix) {
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
}

TEST_F(TimeEvolutionMonteCarloTest, TracePreserved) {
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMonteCarloTest, HermiticityPreserved) {
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionMonteCarloTest, UnitaryDynamicsDisablesMonteCarlo) {
    NoiseModelCpp no_noise;
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, params, steps, no_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionMonteCarloTest, GroundStateRemainsGroundState) {
    auto out = run_time_evolution(pure_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t(0, 0)), 1.0, kTolLoose);
}

class TimeEvolutionMonteCarloMatrixFreeTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5, 0, "Z");
    std::vector<MatrixFreeHamiltonian> hamiltonians = {H_mf};
    std::vector<std::vector<double>> params = {{1.0}};
    std::vector<double> steps = {0.1};
    NoiseModelCpp noise;
    QiliSimConfig config;
    void SetUp() override {
        noise.add_jump_operator(amp_damp_jump());
        config.set_monte_carlo(true);
        config.set_num_monte_carlo_trajectories(200);
        config.set_time_evolution_method("integrate_rk4_matrix_free");
        config.set_seed(0);
    }
};

TEST_F(TimeEvolutionMonteCarloMatrixFreeTest, FinalStateIsDensityMatrixMF) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
}

TEST_F(TimeEvolutionMonteCarloMatrixFreeTest, TracePreservedMF) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMonteCarloMatrixFreeTest, HermiticityPreservedMF) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionMonteCarloMatrixFreeTest, UnitaryDynamicsDisablesMonteCarloMF) {
    NoiseModelCpp no_noise;
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, no_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionMonteCarloMatrixFreeTest, GroundStateRemainsGroundStateMF) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t(0, 0)), 1.0, kTolLoose);
}

TEST_F(TimeEvolutionMethodTest, BadMethodThrows) {
    QiliSimConfig cfg;
    cfg.set_time_evolution_method("not_a_real_method");
    EXPECT_ANY_THROW(run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, obs, cfg));
}

class TimeEvolutionAdaptiveTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_mf = make_matrix_free_H(0.5, 0, "Z");
    std::vector<MatrixFreeHamiltonian> hamiltonians = {H_mf};
    std::vector<std::vector<double>> params = {{1.0, 1.0, 1.0}};
    std::vector<double> steps = {0.1, 0.2, 0.3};
    NoiseModelCpp empty_noise;
    QiliSimConfig config;
    virtual void SetUp() override { config.set_time_evolution_method("integrate_rk45_matrix_free"); }
};

TEST_F(TimeEvolutionAdaptiveTest, DefaultConfigDoesNotThrow) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    SUCCEED();
}

TEST_F(TimeEvolutionAdaptiveTest, FinalStateDimensionMatchesDensityMatrixInput) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
}

TEST_F(TimeEvolutionAdaptiveTest, FinalStateDimensionMatchesStatevectorInputBra) {
    SparseMatrix bra = statevector_zero_sparse().adjoint();
    auto out = run_time_evolution_mf(bra, hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
}

TEST_F(TimeEvolutionAdaptiveTest, TracePreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, TracePreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, HermiticityPreservedUnitaryDensityMatrix) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionAdaptiveTest, HermiticityPreservedWithJumpOperators) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_TRUE((out.rho_t - out.rho_t.adjoint()).isZero(kTol));
}

TEST_F(TimeEvolutionAdaptiveTest, ZeroTimeStepReturnsInitialDensityMatrix) {
    std::vector<double> zero_steps = {0.0};
    std::vector<std::vector<double>> one_param = {{1.0}};
    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, one_param, zero_steps, empty_noise, {}, config);
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionAdaptiveTest, EigenstatePopulationsUnchangedUnderHZ) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t(0, 0)), 1.0, kTol);
    EXPECT_NEAR(std::real(out.rho_t(1, 1)), 0.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, GroundStateIsFixedPointOfAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    DenseMatrix rho_0_dense = DenseMatrix(pure_zero_sparse());
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionAdaptiveTest, ExcitedStateDecaysTowardGroundStateWithAmpDamping) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    DenseMatrix excited = DenseMatrix::Zero(2, 2);
    excited(1, 1) = 1.0;
    auto out = run_time_evolution_mf(to_sparse(excited), hamiltonians, params, steps, noise, {}, config);
    EXPECT_GT(std::real(out.rho_t(0, 0)), 0.0);
    EXPECT_LT(std::real(out.rho_t(1, 1)), 1.0);
}

TEST_F(TimeEvolutionAdaptiveTest, StatevectorInputProducesValidStatevectorOutput) {
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(out.rho_t.norm(), 1.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, StatevectorInputWithJumpsProducesDensityMatrix) {
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump());
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_EQ(out.rho_t.rows(), 2);
    EXPECT_EQ(out.rho_t.cols(), 2);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, NoObservablesProducesEmptyExpectationValues) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.expectation_values.empty());
}

TEST_F(TimeEvolutionAdaptiveTest, ExpectationValueCountMatchesObservableCount) {
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    MatrixFreeHamiltonian obs_x = make_matrix_free_H(1.0, 0, "X");
    std::vector<MatrixFreeHamiltonian> obs = {obs_z, obs_x};
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, obs, config);
    EXPECT_EQ(int(out.expectation_values.size()), 2);
}

TEST_F(TimeEvolutionAdaptiveTest, PauliZExpectationValueOnEigenstateIsOne) {
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {obs_z}, config);
    EXPECT_NEAR(out.expectation_values[0], 1.0, kTol);
}

TEST_F(TimeEvolutionAdaptiveTest, NoIntermediatesStoredByDefault) {
    auto out = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.intermediate_rhos.empty());
    EXPECT_TRUE(out.intermediate_expectation_values.empty());
}

TEST_F(TimeEvolutionAdaptiveTest, EachIntermediateRhoHasUnitTrace) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_NEAR(std::real(rho_int.trace()), 1.0, kTol);
    }
}

TEST_F(TimeEvolutionAdaptiveTest, ZeroParameterLeavesStateUnchanged) {
    std::vector<std::vector<double>> zero_params = {{0.0, 0.0, 0.0}};
    DenseMatrix rho_0_dense = DenseMatrix(pure_plus_sparse());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, zero_params, steps, empty_noise, {}, config);
    EXPECT_TRUE(out.rho_t.isApprox(rho_0_dense, kTol));
}

TEST_F(TimeEvolutionAdaptiveTest, IntermediateResultsWithStatevectorInput) {
    config.set_store_intermediate_results(true);
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    for (const auto& rho_int : out.intermediate_rhos) {
        EXPECT_EQ(rho_int.rows(), 2);
        EXPECT_EQ(rho_int.cols(), 1);
        EXPECT_NEAR(std::real(rho_int.norm()), 1.0, kTol);
    }
}

TEST_F(TimeEvolutionAdaptiveTest, AgreesWithRK4MatrixFreeOnEigenstate) {
    QiliSimConfig cfg_rk4;
    cfg_rk4.set_time_evolution_method("integrate_rk4_matrix_free");
    MatrixFreeHamiltonian obs_z = make_matrix_free_H(1.0, 0, "Z");
    auto out_rk45 = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {obs_z}, config);
    auto out_rk4 = run_time_evolution_mf(pure_zero_sparse(), hamiltonians, params, steps, empty_noise, {obs_z}, cfg_rk4);
    EXPECT_NEAR(out_rk45.expectation_values[0], out_rk4.expectation_values[0], kTolLoose);
}

TEST_F(TimeEvolutionAdaptiveTest, TighterAdaptiveTolDoesNotThrow) {
    config.set_adaptive_tol(1e-6);
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

class TimeEvolutionVariationalTest : public ::testing::Test {
   protected:
    MatrixFreeHamiltonian H_X = make_matrix_free_H(1.0, 0, "X");
    MatrixFreeHamiltonian H_Z = make_matrix_free_H(1.0, 0, "Z");
    std::vector<MatrixFreeHamiltonian> hamiltonians = {H_X, H_Z};
    // Coefficients: linear interpolation from X to Z
    std::vector<std::vector<double>> params = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> step_list = {0.5, 1.0};
    QiliSimConfig config;

    void SetUp() override {
        config.set_shots(50);
        config.set_warmups(0);
        config.set_order(1);
    }
};

TEST_F(TimeEvolutionVariationalTest, DoesNotThrowForValidInput) {
    ExponentialAnsatz rho_t(1, 1, 50, 0);
    EXPECT_NO_THROW(time_evolution_variational_exponential(rho_t, hamiltonians, params, step_list, config));
}

TEST_F(TimeEvolutionVariationalTest, TermCountUnchangedAfterEvolution) {
    ExponentialAnsatz rho_t(1, 1, 50, 0);
    size_t initial_terms = rho_t.get_terms().size();
    time_evolution_variational_exponential(rho_t, hamiltonians, params, step_list, config);
    EXPECT_EQ(rho_t.get_terms().size(), initial_terms);
}

TEST_F(TimeEvolutionVariationalTest, EmptyHamiltonianListThrows) {
    ExponentialAnsatz rho_t(1, 1, 50, 0);
    EXPECT_ANY_THROW(time_evolution_variational_exponential(rho_t, {}, {}, {}, config));
}

TEST_F(TimeEvolutionTest, TimeDependentRateScalingDense) {
    NoiseModelCpp noise;
    // Mixed: a constant jump operator (empty series, scaled unchanged) and a time-dependent jump
    // operator whose per-step sqrt(rate) series is applied at each step. Exercises both branches of
    // the per-step rescaling loop in the dense evolution.
    noise.add_jump_operator(amp_damp_jump());
    noise.add_jump_operator(amp_damp_jump(), {0.0, 0.5, 1.0});
    EXPECT_TRUE(noise.has_time_dependent_rates());
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, TimeDependentRateScalingMatrixFree) {
    NoiseModelCpp noise;
    // Mixed: a constant jump (empty series, scaled unchanged) and a time-dependent jump. Exercises
    // both branches of the per-step rescaling loop in the matrix-free fixed-step evolution.
    noise.add_jump_operator(amp_damp_jump());
    noise.add_jump_operator(amp_damp_jump(), {0.0, 0.5, 1.0});
    EXPECT_TRUE(noise.has_time_dependent_rates());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_NEAR(std::real(out.rho_t.trace()), 1.0, kTol);
}

TEST_F(TimeEvolutionMatrixFreeTest, TimeDependentRateAdaptiveThrows) {
    config.set_time_evolution_method("integrate_rk45_matrix_free");
    NoiseModelCpp noise;
    noise.add_jump_operator(amp_damp_jump(), {0.0, 0.5, 1.0});
    EXPECT_ANY_THROW(run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config));
}

TEST_F(TimeEvolutionVariationalTest, AnsatzParametersComeFromVariationalConfig) {
    config.set_shots(37);
    config.set_warmups(3);
    config.set_order(1);
    config.set_num_monte_carlo_trajectories(999);  // must NOT leak into the ansatz shots

    ExponentialAnsatz rho_t(1, 1, 50, 0);
    time_evolution_variational_exponential(rho_t, hamiltonians, params, step_list, config);

    EXPECT_EQ(rho_t.get_shots(), 37);
    EXPECT_EQ(rho_t.get_warmups(), 3);
    EXPECT_EQ(rho_t.get_order(), 1);
}

// Divergence handling: when ||H||*dt exceeds an integrator's stability limit the state overflows
// to a non-finite value. The integrators must detect this, mark the state as NaN (rather than
// silently collapsing it to zeros), and stop stepping early. A huge Hamiltonian coefficient forces
// the overflow within a single step for the fixed-step and Krylov methods.

// A Hamiltonian coefficient large enough that a single RK step overflows to +/-inf.
static const std::vector<std::vector<double>> kHugeParams = {{1e300, 1e300, 1e300}};

TEST_F(TimeEvolutionTest, DenseRK4StatevectorDivergesToNaN) {
    // Unitary-on-statevector path: overflow is caught by the norm guard in iter_rk4_matrix.
    auto out = run_time_evolution(statevector_zero_sparse(), hamiltonians, kHugeParams, steps, empty_noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

TEST_F(TimeEvolutionTest, DenseRK4DensityMatrixDivergesToNaN) {
    // Density-matrix path (jump operator forces non-unitary dynamics): a huge jump rate overflows
    // the trace, which is caught by the trace guard in iter_rk4_matrix.
    NoiseModelCpp noise;
    noise.add_jump_operator(huge_amp_damp_jump());
    auto out = run_time_evolution(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

TEST_F(TimeEvolutionMatrixFreeTest, MatrixFreeRK4StatevectorDivergesToNaN) {
    auto out = run_time_evolution_mf(statevector_zero_sparse(), hamiltonians, kHugeParams, steps, empty_noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

TEST_F(TimeEvolutionMatrixFreeTest, MatrixFreeRK4DensityMatrixDivergesToNaN) {
    // A huge jump rate overflows the trace, caught by the trace guard in the matrix-free iter_rk4.
    NoiseModelCpp noise;
    noise.add_jump_operator(huge_amp_damp_jump());
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, params, steps, noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

TEST_F(TimeEvolutionArnoldiMatrixFreeTest, ArnoldiMatrixFreeDivergesToNaN) {
    auto out = run_time_evolution_mf(pure_plus_sparse(), hamiltonians, kHugeParams, steps, empty_noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

TEST_F(TimeEvolutionAdaptiveTest, AdaptiveRK45StopsOnNonFiniteState) {
    // The adaptive stepper shrinks dt in response to a huge Hamiltonian rather than overflowing, so
    // feed it an already-non-finite state: the divergence guard must catch it and stop the loop
    // (returning the NaN state) instead of iterating.
    auto out = run_time_evolution_mf(nan_statevector_sparse(), hamiltonians, params, steps, empty_noise, {}, config);
    EXPECT_FALSE(out.rho_t.allFinite());
}

// GCOV_EXCL_BR_STOP