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

#include <array>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "../../../libs/eigen.h"
#include "../digital/gate.h"
#include "matrix_free_hamiltonian.h"
#include "tensor.h"

// GCOV_EXCL_BR_START

class BPState {
   private:
    int nqubits = 0;

    // One tensor per qubit, with legs (physical, then one bond per incident edge in the order of incident_edges)
    std::vector<Tensor> sites;
    std::vector<std::pair<int, int>> edges;
    std::map<std::pair<int, int>, int> edge_lookup;
    std::vector<std::vector<int>> incident_edges;

    // Messages of the norm network <psi|psi>, indexed (edge, direction) with direction 0 running from edges[e].first to edges[e].second
    std::vector<std::array<DenseMatrix, 2>> messages;

    // The square root of each message and its pseudo-inverse, kept in step with the messages so gauging never redoes the eigendecomposition
    std::vector<std::array<DenseMatrix, 2>> message_roots;
    std::vector<std::array<DenseMatrix, 2>> message_inverse_roots;

    // Qubits changed since the last belief propagation run, so a gate on one of them starts a new layer
    std::vector<bool> touched;

    int max_bond_dimension = 16;
    Real truncation_cutoff = 1e-10;
    int max_iterations = 50;
    Real tolerance = 1e-8;
    Real total_truncation_error = 0.0;
    int bp_runs = 0;
    int bp_iterations = 0;
    bool bp_converged = true;

    int leg_of(int v, int e) const;
    int message_index_from(int v, int e) const { return edges[e].first == v ? 0 : 1; }
    const DenseMatrix& message_into(int v, int e) const { return messages[e][1 - message_index_from(v, e)]; }
    const DenseMatrix& root_into(int v, int e) const { return message_roots[e][1 - message_index_from(v, e)]; }
    const DenseMatrix& inverse_root_into(int v, int e) const { return message_inverse_roots[e][1 - message_index_from(v, e)]; }
    Tensor gauge_and_reduce(int v, int e, Tensor& isometry) const;
    Tensor restore_site(int v, int e, const Tensor& isometry, const Tensor& reduced) const;
    void check_messages_current() const;
    std::vector<int> cluster_of(const PauliString& pauli) const;
    std::vector<Complex> cluster_expectations(const std::vector<int>& cluster, const std::vector<const PauliString*>& paulis) const;

   public:
    // Safeguard to make sure we don't accidentally try to materialize a huge network
    static constexpr int MAX_DENSE_QUBITS = 24;

    // Largest tensor, in entries, an exact contraction may build along the way, since open bonds can make it far larger than the result
    static constexpr int64_t MAX_CONTRACTION_ENTRIES = int64_t(1) << 27;

    // Total discarded weight above which a simulation warns that the bond dimension is too small
    static constexpr Real TRUNCATION_WARNING_THRESHOLD = 1e-2;

    // Constructors
    explicit BPState(int nqubits);
    BPState(int nqubits, const std::string& b);

    // Getters and setters
    void set_max_bond_dimension(int d);
    void set_truncation_cutoff(Real c);
    void set_max_iterations(int n);
    void set_tolerance(Real t);
    int get_nqubits() const { return nqubits; }
    int get_num_edges() const { return static_cast<int>(edges.size()); }
    const Tensor& get_site_tensor(int q) const { return sites[q]; }
    int get_max_bond_dimension_used() const;
    Real get_truncation_error() const { return total_truncation_error; }
    int get_bp_runs() const { return bp_runs; }
    int get_bp_iterations() const { return bp_iterations; }
    bool get_bp_converged() const { return bp_converged; }

    // Evolving the state
    Real apply_gate(const Gate& gate);
    Real apply_one_site(const DenseMatrix& u, int q);
    Real apply_two_site(const DenseMatrix& u, int q0, int q1);
    int run_belief_propagation();

    // Outputs
    Complex expectation_value(const PauliString& pauli) const;
    Real expectation_value(const MatrixFreeHamiltonian& H) const;
    DenseMatrix as_dense() const;
};

std::ostream& operator<<(std::ostream& os, const BPState& state);

// GCOV_EXCL_BR_STOP
