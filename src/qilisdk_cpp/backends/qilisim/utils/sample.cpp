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
#include <random>
#include <sstream>
#include "sample.h"
#include "../representations/matrix_free_hamiltonian.h"
#include "parsers.h"
#include "random.h"

// GCOV_EXCL_BR_START

std::map<std::string, int> apply_readout_error(const std::map<std::string, int>& counts, const NoiseModelCpp& noise_model_cpp, int n_qubits) {
    /*
    Apply readout error to the measurement counts.

    Args:
        counts (std::map<std::string, int>&): The original measurement counts without readout error.
        noise_model_cpp (NoiseModelCpp&): The noise model containing the readout error probabilities.
        n_qubits (int): The number of qubits in the circuit.

    Returns:
        std::map<std::string, int>: The new measurement counts after applying readout error
    */
    std::default_random_engine generator;
    generator.seed(42);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::map<std::string, int> noisy_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        int count = pair.second;
        // For each shot in the count, apply readout error
        for (int shot = 0; shot < count; ++shot) {
            std::string noisy_bitstring = "";
            for (int q = 0; q < n_qubits; ++q) {
                char bit = bitstring[q];
                auto readout_error = noise_model_cpp.get_relevant_readout_error(q);
                double p01 = readout_error.first;
                double p10 = readout_error.second;
                double rand_val = distribution(generator);
                if (bit == '0') {
                    // 0 -> 1 with probability p01
                    if (rand_val < p01) {
                        noisy_bitstring += '1';
                    } else {
                        noisy_bitstring += '0';
                    }
                } else {
                    // 1 -> 0 with probability p10
                    if (rand_val < p10) {
                        noisy_bitstring += '0';
                    } else {
                        noisy_bitstring += '1';
                    }
                }
            }
            noisy_counts[noisy_bitstring] += 1;
        }
    }
    return noisy_counts;
}

std::map<std::string, int> filter_counts(const std::map<std::string, int>& counts, const std::vector<bool>& qubits_to_measure) {
    /*
    Filter the measurement counts to only include the measured qubits.

    Args:
        counts (std::map<std::string, int>&): The original measurement counts for all qubits.
        qubits_to_measure (std::vector<bool>&): A list indicating which qubits were measured.

    Returns:
        std::map<std::string, int>: The filtered measurement counts only for the measured qubits.
    */
    std::map<std::string, int> filtered_counts;
    for (const auto& pair : counts) {
        std::string bitstring = pair.first;
        int count = pair.second;
        std::string filtered_bitstring = "";
        for (size_t q = 0; q < qubits_to_measure.size(); ++q) {
            if (qubits_to_measure[q]) {
                filtered_bitstring += bitstring[q];
            }
        }
        filtered_counts[filtered_bitstring] += count;
    }
    return filtered_counts;
}

std::map<std::string, int> construct_samples(const DenseMatrix& state, int n_qubits, int n_shots, NoiseModelCpp& noise_model_cpp, const QiliSimConfig& config, const std::vector<bool>& qubits_to_measure){
    /*
    Sample a quantum state, given a noise model and a set of qubits to measure.

    Args:
        state (DenseMatrix&): the state to be sampled.
        nqubits (int): the number of qubits in the quantum state.
        nshots (int): the number of shots used for the sampling.
        noise_model_cpp (NoiseModelCpp&): the noise model to be considered when computing the samples.
        config (QiliSimConfig&): QiliSim configuration.
        qubits_to_measure (vector<boo>&): a list of boolean specifying which qubits to measure. 

    Returns:
        std::map<std::string, int>: a map containing the state and the number of samples obtained of that state. 
    
    */
    std::map<std::string, int> counts;
    bool has_noise = !noise_model_cpp.is_empty();
    long dim = 1L << n_qubits;
    bool is_statevector = (state.cols() == 1 && state.rows() == dim);

    // Get the probabilities
    std::vector<double> probabilities(state.rows());
    double total_prob = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_prob) schedule(static)
#endif
    for (int row = 0; row < state.rows(); ++row) {
        double prob = 0.0;
        if (is_statevector) {
            prob = std::norm(state(row, 0));
        } else {
            prob = state.coeff(row, row).real();
        }
        total_prob += prob;
        probabilities[row] = prob;
    }

    // Make sure probabilities sum to 1
    if (std::abs(total_prob - 1.0) > config.get_atol()) {
        std::stringstream ss;
        ss << std::setprecision(15) << total_prob;
        throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
    }

    // Sample from these probabilities
    counts = sample_from_probabilities(probabilities, n_qubits, n_shots, config.get_seed());

    // Apply readout error to counts
    if (has_noise) {
        counts = apply_readout_error(counts, noise_model_cpp, n_qubits);
    }

    // Only keep measured qubits in the counts
    return  filter_counts(counts, qubits_to_measure);

}

// GCOV_EXCL_BR_STOP
