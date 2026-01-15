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

#include <iomanip>
#include <sstream>
#include "qilisim.h"

py::object QiliSimCpp::execute_sampling(const py::object& functional, const py::dict& solver_params) {
    /*
    Execute a sampling functional using a simple statevector simulator.

    Args:
        functional (py::object): The Sampling functional to execute.
        solver_params (py::dict): Solver parameters, including 'max_cache_size'.

    Returns:
        SamplingResult: A result object containing the measurement samples and computed probabilities.

    Raises:
        py::value_error: If nqubits is non-positive.
        py::value_error: If shots is non-positive.
    */

    // Get info from the functional
    int n_shots = functional.attr("nshots").cast<int>();
    int n_qubits = functional.attr("circuit").attr("nqubits").cast<int>();

    // Get parameters
    int max_cache_size = 100;
    if (solver_params.contains("max_cache_size")) {
        max_cache_size = solver_params["max_cache_size"].cast<int>();
    }
    int num_threads = 1;
    if (solver_params.contains("num_threads")) {
        num_threads = solver_params["num_threads"].cast<int>();
    }

    // Set the number of threads
    if (num_threads <= 0) {
        num_threads = 1;
    }
    Eigen::setNbThreads(num_threads);

    // Sanity checks
    if (n_qubits <= 0) {
        throw py::value_error("nqubits must be positive.");
    }
    if (n_shots <= 0) {
        throw py::value_error("nshots must be positive.");
    }

    // Get the gate
    std::vector<Gate> gates = parse_gates(functional.attr("circuit"));

    // Determine which qubits to measure
    std::vector<bool> qubits_to_measure = parse_measurements(functional.attr("circuit"));

    // Start with the zero state
    long dim = 1L << n_qubits;
    DenseMatrix state = DenseMatrix::Zero(dim, 1);
    state(0, 0) = 1.0;

    // Determine the start/end use of each gate
    std::map<std::string, std::pair<int, int>> gate_first_last_use;
    for (int i = 0; i < int(gates.size()); ++i) {
        std::string gate_id = gates[i].get_id();
        if (gate_first_last_use.find(gate_id) == gate_first_last_use.end()) {
            gate_first_last_use[gate_id] = std::make_pair(i, i);
        } else {
            gate_first_last_use[gate_id].second = i;
        }
    }

    // Apply each gate (partial cache)
    SparseMatrix gate_matrix;
    std::map<std::string, SparseMatrix> gate_cache;
    std::string gate_id = "";
    int gate_count = 0;
    for (const auto& gate : gates) {
        gate_id = gate.get_id();

        // If we already have it in the cache, use it
        if (gate_cache.find(gate_id) != gate_cache.end()) {
            gate_matrix = gate_cache[gate_id];

            // If it will be used again later and we have space, cache it
        } else if (gate_first_last_use[gate_id].second > gate_count && int(gate_cache.size()) < max_cache_size) {
            gate_cache[gate_id] = gate.get_full_matrix(n_qubits);
            gate_matrix = gate_cache[gate_id];

            // Otherwise just generate it on the fly
        } else {
            gate_matrix = gate.get_full_matrix(n_qubits);
        }

        // Apply the gate (Sparse-Dense multiplication, OpenMP parallel if enabled)
        state = gate_matrix * state;

        // Renormalize the state
        state /= state.norm();

        // Clear the gate from the cache if this was its last use
        if (gate_first_last_use[gate_id].second == gate_count) {
            gate_cache.erase(gate_id);
        }
        gate_count++;
    }

    // Get the probabilities
    std::vector<std::tuple<int, double>> prob_entries;
    double total_prob = 0.0;
    for (int row = 0; row < state.rows(); ++row) {
        std::complex<double> amp = state(row, 0);
        double prob = std::norm(amp);
        if (prob > atol_) {
            prob_entries.emplace_back(row, prob);
            total_prob += prob;
        }
    }

    // Make sure probabilities sum to 1
    // const double probability_tolerance = 1e-5;
    // if (std::abs(total_prob - 1.0) > probability_tolerance) {
    if (std::abs(total_prob - 1.0) > atol_) {
        std::stringstream ss;
        ss << std::setprecision(15) << total_prob;
        throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
    }

    // Sample from these probabilities
    std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_shots);

    // Only keep measured qubits in the counts
    std::map<std::string, int> filtered_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        std::string filtered_bitstring = "";
        for (int i = 0; i < n_qubits; ++i) {
            if (qubits_to_measure[i]) {
                filtered_bitstring += bitstring[i];
            }
        }
        filtered_counts[filtered_bitstring] += pair.second;
    }
    counts = filtered_counts;

    // Convert counts to samples dict
    py::dict samples;
    for (const auto& pair : counts) {
        samples[py::cast(pair.first)] = py::cast(pair.second);
    }

    return SamplingResult("nshots"_a = n_shots, "samples"_a = samples);
}