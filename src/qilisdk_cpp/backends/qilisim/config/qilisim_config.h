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
#pragma once

#include <string>

// Config file
class QiliSimConfig {
public:
    bool monte_carlo = false;
    int num_monte_carlo_trajectories = 1000;
    int arnoldi_dim = 10;
    int num_arnoldi_substeps = 10;
    int num_integrate_substeps = 2;
    std::string method = "integrate";
    bool store_intermediate_results = false;
    int num_threads = 1;
    int seed = 42;
    double atol = 1e-12;
    int max_cache_size = 1000;
    
    // Initialize with default values
    QiliSimConfig() = default;

    // Can be called to validate the config and throw a py error if not
    void validate() const;

};
