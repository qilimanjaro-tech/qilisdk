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

#include "../../../src/qilisdk_cpp/backends/qilisim/utils/random.h"

TEST(SampleFromProbabilitiesTest, SampleFromProbabilitiesReturnsValidSamples) {
    std::vector<std::tuple<int, double>> prob_entries = {{0, 0.5}, {1, 0.5}};
    int n_qubits = 1;
    int n_shots = 1000;
    int seed = 42;
    std::map<std::string, int> samples = sample_from_probabilities(prob_entries, n_qubits, n_shots, seed);
    int total_samples = 0;
    for (const auto& entry : samples) {
        EXPECT_TRUE(entry.first == "0" || entry.first == "1");
        total_samples += entry.second;
    }
    EXPECT_EQ(total_samples, n_shots);
}

TEST(SampleFromProbabilitiesTest, SampleFromProbabilitiesWithExplicitProbabilities) {
    std::vector<double> probabilities = {0.25, 0.75};
    int n_qubits = 1;
    int n_shots = 1000;
    int seed = 42;
    std::map<std::string, int> samples = sample_from_probabilities(probabilities, n_qubits, n_shots, seed);
    int total_samples = 0;
    for (const auto& entry : samples) {
        total_samples += entry.second;
        EXPECT_TRUE(entry.first == "0" || entry.first == "1");
    }
    EXPECT_EQ(total_samples, n_shots);
}

TEST(SampleFromDensityMatrixTest, SampleFromDensityMatrixDense) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    rho(0, 0) = 0.5;
    rho(1, 1) = 0.5;
    int n_qubits = 1;
    int n_trajectories = 1000;
    int seed = 42;
    DenseMatrix trajectories = sample_from_density_matrix(rho, n_trajectories, seed);
    int total_trajectories = trajectories.cols();
    EXPECT_EQ(total_trajectories, n_trajectories);
    for (int i = 0; i < trajectories.rows(); ++i) {
        EXPECT_TRUE(std::abs(trajectories(i, 0) - 0.0) < 1e-6 || std::abs(trajectories(i, 0) - 1.0) < 1e-6);
    }
}

TEST(SampleFromDensityMatrixTest, SampleFromDensityMatrixSparse) {
    SparseMatrix rho(2, 2);
    rho.insert(0, 0) = 0.5;
    rho.insert(1, 1) = 0.5;
    rho.makeCompressed();
    int n_qubits = 1;
    int n_trajectories = 1000;
    int seed = 42;
    SparseMatrix trajectories = sample_from_density_matrix(rho, n_trajectories, seed);
    int total_trajectories = trajectories.cols();
    EXPECT_EQ(total_trajectories, n_trajectories);
    for (int i = 0; i < trajectories.rows(); ++i) {
        EXPECT_TRUE(std::abs(trajectories.coeff(i, 0) - 0.0) < 1e-6 || std::abs(trajectories.coeff(i, 0) - 1.0) < 1e-6);
    }
}

TEST(SampleFromDensityMatrixTest, SampleFromZeroDensityMatrixDense) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    int n_qubits = 1;
    int n_trajectories = 1000;
    int seed = 42;
    EXPECT_ANY_THROW(sample_from_density_matrix(rho, n_trajectories, seed));
}

TEST(SampleFromDensityMatrixTest, SampleFromZeroDensityMatrixSparse) {
    SparseMatrix rho(2, 2);
    int n_qubits = 1;
    int n_trajectories = 1000;
    int seed = 42;
    EXPECT_ANY_THROW(sample_from_density_matrix(rho, n_trajectories, seed));
}

TEST(TrajectoriesToDensityMatrixTest, TrajectoriesToDensityMatrixDense) {
    int n_trajectories = 1000;
    DenseMatrix trajectories = DenseMatrix::Zero(2, n_trajectories);
    for (int i = 0; i < n_trajectories / 2; ++i) {
        trajectories(0, i) = 1.0;
    }
    for (int i = n_trajectories / 2; i < n_trajectories; ++i) {
        trajectories(1, i) = 1.0;
    }
    DenseMatrix rho = trajectories_to_density_matrix(trajectories);
    EXPECT_NEAR(rho(0, 0).real(), 0.5, 1e-2);
    EXPECT_NEAR(rho(0, 1).real(), 0.0, 1e-2);
    EXPECT_NEAR(rho(1, 0).real(), 0.0, 1e-2);
    EXPECT_NEAR(rho(1, 1).real(), 0.5, 1e-2);
}

TEST(TrajectoriesToDensityMatrixTest, TrajectoriesToDensityMatrixSparse) {
    int n_trajectories = 1000;
    SparseMatrix trajectories(2, n_trajectories);
    for (int i = 0; i < n_trajectories / 2; ++i) {
        trajectories.insert(0, i) = 1.0;
    }
    for (int i = n_trajectories / 2; i < n_trajectories; ++i) {
        trajectories.insert(1, i) = 1.0;
    }
    trajectories.makeCompressed();
    SparseMatrix rho = trajectories_to_density_matrix(trajectories);
    EXPECT_NEAR(rho.coeff(0, 0).real(), 0.5, 1e-2);
    EXPECT_NEAR(rho.coeff(0, 1).real(), 0.0, 1e-2);
    EXPECT_NEAR(rho.coeff(1, 0).real(), 0.0, 1e-2);
    EXPECT_NEAR(rho.coeff(1, 1).real(), 0.5, 1e-2);
}

TEST(GetVectorFromDensityMatrixTest, GetVectorFromDensityMatrixDense) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    rho(0, 0) = 1.0;
    DenseMatrix state_vector = get_vector_from_density_matrix(rho, 1e-10);
    EXPECT_EQ(state_vector.rows(), 2);
    EXPECT_EQ(state_vector.cols(), 1);
    EXPECT_NEAR(state_vector(0, 0).real(), 1.0, 1e-4);
    EXPECT_NEAR(state_vector(1, 0).real(), 0.0, 1e-4);
}

TEST(GetVectorFromDensityMatrixTest, GetVectorFromDensityMatrixSparse) {
    SparseMatrix rho(2, 2);
    rho.insert(0, 0) = 1.0;
    rho.makeCompressed();
    SparseMatrix state_vector = get_vector_from_density_matrix(rho, 1e-10);
    EXPECT_EQ(state_vector.rows(), 2);
    EXPECT_EQ(state_vector.cols(), 1);
    EXPECT_NEAR(state_vector.coeff(0, 0).real(), 1.0, 1e-4);
    EXPECT_NEAR(state_vector.coeff(1, 0).real(), 0.0, 1e-4);
}

TEST(GetVectorFromDensityMatrixTest, GetVectorFromZeroMatrixDense) {
    DenseMatrix rho = DenseMatrix::Zero(2, 2);
    EXPECT_ANY_THROW(get_vector_from_density_matrix(rho, 1e-10));
}

TEST(GetVectorFromDensityMatrixTest, GetVectorFromZeroMatrixSparse) {
    SparseMatrix rho(2, 2);
    EXPECT_ANY_THROW(get_vector_from_density_matrix(rho, 1e-10));
}

// GCOV_EXCL_BR_STOP