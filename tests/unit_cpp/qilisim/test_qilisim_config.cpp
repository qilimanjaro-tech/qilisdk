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
#include <set>
#include "../../../src/qilisdk_cpp/backends/qilisim/config/qilisim_config.h"

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
    config.set_max_cache_size(0);
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