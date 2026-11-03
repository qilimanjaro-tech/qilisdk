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
#include <cmath>
#include <set>
#include "../../../src/qilisdk_cpp/backends/qilisim/config/qilisim_config.h"
#include "../../../src/qilisdk_cpp/libs/pybind.h"

TEST(QilisimConfig, BadValidateThrows) {
    QiliSimConfig default_config;
    QiliSimConfig config;

    config = default_config;
    config.set_arnoldi_dim(-1);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_num_arnoldi_substeps(0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_time_evolution_method("invalid_method");
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_digital_method("invalid_digital");
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_monte_carlo(true);
    config.set_num_monte_carlo_trajectories(0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_num_threads(-2);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_atol(0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_max_cache_size(-1);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_adaptive_tol(-1.0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_order(0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_shots(0);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_warmups(-1);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_max_fused_qubits(-1);
    EXPECT_ANY_THROW(config.validate());
}

TEST(QilisimConfig, MethodNamesMustMatchExactly) {
    // Substrings of valid names (and of the error message) must not pass validation
    for (const std::string name : {"statevector_matrix_fre", "statevecto", "stat", "", "'statevector'"}) {
        QiliSimConfig config;
        config.set_digital_method(name);
        EXPECT_ANY_THROW(config.validate()) << "digital method '" << name << "'";
    }
    for (const std::string name : {"arnoldi_matrix", "integrate_rk4_matrix", "dir", ""}) {
        QiliSimConfig config;
        config.set_time_evolution_method(name);
        EXPECT_ANY_THROW(config.validate()) << "time evolution method '" << name << "'";
    }
    for (const std::string name : {"statevector", "statevector_matrix_free", "stabilizer", "mps"}) {
        QiliSimConfig config;
        config.set_digital_method(name);
        EXPECT_NO_THROW(config.validate()) << "digital method '" << name << "'";
    }
}

TEST(QilisimConfig, ZeroMaxCacheSizeDisablesCachingAndIsValid) {
    QiliSimConfig config;
    config.set_max_cache_size(0);
    EXPECT_NO_THROW(config.validate());
}

TEST(QilisimConfig, FusionGettersSetters) {
    QiliSimConfig config;
    config.set_fuse_gates(true);
    config.set_max_fused_qubits(3);
    EXPECT_TRUE(config.get_fuse_gates());
    EXPECT_EQ(config.get_max_fused_qubits(), 3);
}

TEST(QilisimConfig, VariationalFieldGettersSetters) {
    QiliSimConfig config;
    config.set_order(3);
    config.set_shots(200);
    config.set_warmups(50);
    EXPECT_EQ(config.get_order(), 3);
    EXPECT_EQ(config.get_shots(), 200);
    EXPECT_EQ(config.get_warmups(), 50);
}

TEST(QilisimConfig, OrderGreaterThan4_ThrowsOnValidate) {
    QiliSimConfig config;
    config.set_order(5);
    EXPECT_ANY_THROW(config.validate());
}

TEST(QilisimConfig, StabilizerMaxStates_GetterSetter) {
    QiliSimConfig config;
    config.set_stabilizer_max_states(200);
    EXPECT_EQ(config.get_stabilizer_max_states(), 200);
}

TEST(QilisimConfig, MPSSettings_GetterSetter) {
    QiliSimConfig config;
    config.set_mps_max_bond_dimension(128);
    config.set_mps_truncation_cutoff(1e-8);
    EXPECT_EQ(config.get_mps_max_bond_dimension(), 128);
    EXPECT_DOUBLE_EQ(config.get_mps_truncation_cutoff(), 1e-8);
    EXPECT_NO_THROW(config.validate());
}

TEST(QilisimConfig, MPSMethod_ValidatesAndRejectsBadTruncation) {
    QiliSimConfig config;
    config.set_digital_method("mps");
    EXPECT_NO_THROW(config.validate());

    QiliSimConfig zero_bond;
    zero_bond.set_mps_max_bond_dimension(0);
    EXPECT_THROW(zero_bond.validate(), py::value_error);

    // A cutoff that is negative, NaN, or 1 or more would collapse every bond, so all are rejected
    for (double cutoff : {-1e-3, 1.0, 2.0, std::nan("")}) {
        QiliSimConfig bad_cutoff;
        bad_cutoff.set_mps_truncation_cutoff(cutoff);
        EXPECT_THROW(bad_cutoff.validate(), py::value_error) << "cutoff " << cutoff;
    }
}

TEST(QilisimConfig, BPSettings_GetterSetter) {
    QiliSimConfig config;
    config.set_bp_max_bond_dimension(32);
    config.set_bp_truncation_cutoff(1e-9);
    config.set_bp_max_iterations(7);
    config.set_bp_tolerance(1e-6);
    EXPECT_EQ(config.get_bp_max_bond_dimension(), 32);
    EXPECT_DOUBLE_EQ(config.get_bp_truncation_cutoff(), 1e-9);
    EXPECT_EQ(config.get_bp_max_iterations(), 7);
    EXPECT_DOUBLE_EQ(config.get_bp_tolerance(), 1e-6);
    EXPECT_NO_THROW(config.validate());
}

TEST(QilisimConfig, BPMethod_ValidatesAndRejectsBadSettings) {
    QiliSimConfig config;
    config.set_digital_method("bp");
    EXPECT_NO_THROW(config.validate());

    QiliSimConfig zero_bond;
    zero_bond.set_bp_max_bond_dimension(0);
    EXPECT_THROW(zero_bond.validate(), py::value_error);

    for (double cutoff : {-1e-3, 1.0, std::nan("")}) {
        QiliSimConfig bad_cutoff;
        bad_cutoff.set_bp_truncation_cutoff(cutoff);
        EXPECT_THROW(bad_cutoff.validate(), py::value_error) << "cutoff " << cutoff;
    }

    QiliSimConfig zero_iterations;
    zero_iterations.set_bp_max_iterations(0);
    EXPECT_THROW(zero_iterations.validate(), py::value_error);

    for (double tolerance : {0.0, -1e-3, std::nan("")}) {
        QiliSimConfig bad_tolerance;
        bad_tolerance.set_bp_tolerance(tolerance);
        EXPECT_THROW(bad_tolerance.validate(), py::value_error) << "tolerance " << tolerance;
    }
}

TEST(QilisimConfig, NextSeedGivesDistinctStreams) {
    QiliSimConfig config;
    config.set_seed(7);

    // Each draw is a distinct, non-negative sub-seed, and none of them is the root seed itself
    std::set<int> seeds;
    for (int i = 0; i < 100; ++i) {
        int seed = config.next_seed();
        EXPECT_GE(seed, 0);
        EXPECT_NE(seed, config.get_seed());
        seeds.insert(seed);
    }
    EXPECT_EQ(seeds.size(), 100u);
}

TEST(QilisimConfig, NextSeedIsReproducibleForARootSeed) {
    QiliSimConfig first;
    QiliSimConfig second;
    first.set_seed(7);
    second.set_seed(7);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(first.next_seed(), second.next_seed());
    }

    // A different root seed gives a different stream
    QiliSimConfig other;
    other.set_seed(8);
    QiliSimConfig seven;
    seven.set_seed(7);
    EXPECT_NE(seven.next_seed(), other.next_seed());
}

TEST(QilisimConfig, SetSeedRestartsTheStream) {
    QiliSimConfig config;
    config.set_seed(7);
    const int first_draw = config.next_seed();
    config.next_seed();
    config.set_seed(7);
    EXPECT_EQ(config.next_seed(), first_draw);
}

// GCOV_EXCL_BR_STOP