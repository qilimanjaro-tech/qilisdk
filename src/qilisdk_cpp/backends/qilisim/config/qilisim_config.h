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

// GCOV_EXCL_BR_START

// Config file
class QiliSimConfig {
   private:
    bool monte_carlo = false;
    int num_monte_carlo_trajectories = 1000;
    int arnoldi_dim = 10;
    int num_arnoldi_substeps = 10;
    std::string time_evolution_method = "integrate_rk4_matrix_free";
    std::string digital_method = "statevector_matrix_free";
    bool store_intermediate_results = false;
    int num_threads = 1;
    int seed = 42;
    double atol = 1e-12;
    int max_cache_size = 1000;
    bool combine_single_qubit_gates = true;
    bool fuse_gates = true;
    int max_fused_qubits = 4;
    bool normalize_after_each_gate = false;
    double adaptive_tol = 1e-2;
    bool measurement_collapse = false;
    int order = 2;
    int shots = 1000;
    int warmups = 100;
    int stabilizer_max_states = 100;

   public:
    // Getters
    int get_stabilizer_max_states() const { return stabilizer_max_states; }
    bool get_monte_carlo() const { return monte_carlo; }
    int get_num_monte_carlo_trajectories() const { return num_monte_carlo_trajectories; }
    int get_arnoldi_dim() const { return arnoldi_dim; }
    double get_adaptive_tol() const { return adaptive_tol; }
    int get_num_arnoldi_substeps() const { return num_arnoldi_substeps; }
    std::string get_time_evolution_method() const { return time_evolution_method; }
    std::string get_digital_method() const { return digital_method; }
    bool get_store_intermediate_results() const { return store_intermediate_results; }
    int get_num_threads() const { return num_threads; }
    int get_seed() const { return seed; }
    double get_atol() const { return atol; }
    int get_max_cache_size() const { return max_cache_size; }
    bool get_normalize_after_gate() const { return normalize_after_each_gate; }
    bool get_combine_single_qubit_gates() const { return combine_single_qubit_gates; }
    bool get_fuse_gates() const { return fuse_gates; }
    int get_max_fused_qubits() const { return max_fused_qubits; }
    bool get_measurement_collapse() const { return measurement_collapse; }
    int get_order() const { return order; }
    int get_shots() const { return shots; }
    int get_warmups() const { return warmups; }

    // Setters
    void set_monte_carlo(bool value) { monte_carlo = value; }
    void set_num_monte_carlo_trajectories(int value) { num_monte_carlo_trajectories = value; }
    void set_arnoldi_dim(int value) { arnoldi_dim = value; }
    void set_adaptive_tol(double value) { adaptive_tol = value; }
    void set_num_arnoldi_substeps(int value) { num_arnoldi_substeps = value; }
    void set_time_evolution_method(const std::string& value) { time_evolution_method = value; }
    void set_digital_method(const std::string& value) { digital_method = value; }
    void set_store_intermediate_results(bool value) { store_intermediate_results = value; }
    void set_num_threads(int value) { num_threads = value; }
    void set_seed(int value) { seed = value; }
    void set_atol(double value) { atol = value; }
    void set_max_cache_size(int value) { max_cache_size = value; }
    void set_normalize_after_gate(bool value) { normalize_after_each_gate = value; }
    void set_combine_single_qubit_gates(bool value) { combine_single_qubit_gates = value; }
    void set_fuse_gates(bool value) { fuse_gates = value; }
    void set_max_fused_qubits(int value) { max_fused_qubits = value; }
    void set_measurement_collapse(bool value) { measurement_collapse = value; }
    void set_order(int value) { order = value; }
    void set_shots(int value) { shots = value; }
    void set_warmups(int value) { warmups = value; }
    void set_stabilizer_max_states(int value) { stabilizer_max_states = value; }

    // Initialize with default values
    QiliSimConfig() = default;

    // Can be called to validate the config and throw a py error if not
    void validate() const;
};

// GCOV_EXCL_BR_STOP
