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

#include <gtest/gtest.h>
#include "../../src/qilisdk_cpp/backends/qilisim/config/qilisim_config.h"

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
    config.set_num_integrate_substeps(-5);
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_time_evolution_method("invalid_method");
    EXPECT_ANY_THROW(config.validate());

    config = default_config;
    config.set_sampling_method("invalid_sampling");
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

}
