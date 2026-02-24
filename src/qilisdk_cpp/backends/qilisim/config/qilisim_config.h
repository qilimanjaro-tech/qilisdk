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
   private:
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

    public:

    // Getters
    bool get_monte_carlo() const { return monte_carlo; }
    int get_num_monte_carlo_trajectories() const { return num_monte_carlo_trajectories; }
    int get_arnoldi_dim() const { return arnoldi_dim; }
    int get_num_arnoldi_substeps() const { return num_arnoldi_substeps; }
    int get_num_integrate_substeps() const { return num_integrate_substeps; }
    std::string get_method() const { return method; }
    bool get_store_intermediate_results() const { return store_intermediate_results; }
    int get_num_threads() const { return num_threads; }
    int get_seed() const { return seed; }
    double get_atol() const { return atol; }
    int get_max_cache_size() const { return max_cache_size; }

    // Setters
    void set_monte_carlo(bool value) { monte_carlo = value; }
    void set_num_monte_carlo_trajectories(int value) { num_monte_carlo_trajectories = value; }
    void set_arnoldi_dim(int value) { arnoldi_dim = value; }
    void set_num_arnoldi_substeps(int value) { num_arnoldi_substeps = value; }
    void set_num_integrate_substeps(int value) { num_integrate_substeps = value; }
    void set_method(const std::string& value) { method = value; }
    void set_store_intermediate_results(bool value) { store_intermediate_results = value; }
    void set_num_threads(int value) { num_threads = value; }
    void set_seed(int value) { seed = value; }
    void set_atol(double value) { atol = value; }
    void set_max_cache_size(int value) { max_cache_size = value; }

    // Initialize with default values
    QiliSimConfig() = default;

    // Can be called to validate the config and throw a py error if not
    void validate() const;
};
