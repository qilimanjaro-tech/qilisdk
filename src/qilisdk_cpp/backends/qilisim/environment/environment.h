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
#pragma once

#include <map>
#include <string>
#include <vector>
#include "../../../libs/eigen.h"
#include "../digital/gate.h"
#include "../noise/noise_model.h"
#include "../representations/matrix_free_hamiltonian.h"

// GCOV_EXCL_BR_START

// All EnvironmentNoise sources of a noise model, combined on the register [system qubits | environment qubits]
class EnvironmentCpp {
   private:
    int n_system_qubits = 0;
    int n_environment_qubits = 0;

    // Couplings plus environment Hamiltonians, on the full register, for the dense and matrix-free methods
    SparseMatrix hamiltonian;
    MatrixFreeHamiltonian hamiltonian_matrix_free;

    // Environment noise, on the full register, with rates folded in
    std::vector<SparseMatrix> jump_operators;

    // Density matrix of all environment qubits
    SparseMatrix initial_state;

   public:
    EnvironmentCpp(int n_system_qubits, int n_environment_qubits, const SparseMatrix& hamiltonian, const MatrixFreeHamiltonian& hamiltonian_matrix_free, const std::vector<SparseMatrix>& jump_operators, const SparseMatrix& initial_state);

    int get_n_system_qubits() const;
    int get_n_environment_qubits() const;
    int get_n_total_qubits() const;

    SparseMatrix attach_to(const SparseMatrix& rho_system) const;
    DenseMatrix trace_out(const DenseMatrix& rho) const;
    void add_to_evolution(std::vector<SparseMatrix>& hamiltonians, std::vector<std::vector<double>>& parameters_list, NoiseModelCpp& noise_model_cpp) const;
    void add_to_evolution(std::vector<MatrixFreeHamiltonian>& hamiltonians, std::vector<std::vector<double>>& parameters_list, NoiseModelCpp& noise_model_cpp) const;
};

void circuit_to_schedule(const std::vector<Gate>& gates, const std::map<std::string, float>& gate_durations, int n_total_qubits, double dt, std::vector<SparseMatrix>& hamiltonians, std::vector<std::vector<double>>& parameters_list, std::vector<double>& step_list);

// GCOV_EXCL_BR_STOP
