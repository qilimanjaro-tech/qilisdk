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
#include <complex>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../../../libs/eigen.h"
#include "../digital/gate.h"
#include "../representations/matrix_free_hamiltonian.h"

// Deliberately included here and NOT in libs/eigen.h: the Tensor module is
// cheap to include (~1s) but every contraction it instantiates is expensive,
// and eigen.h is pulled into every translation unit in the backend.
#include <unsupported/Eigen/CXX11/Tensor>

// GCOV_EXCL_BR_START

typedef int VertexId;
typedef int EdgeId;
typedef int LoopId;

// Hard caps, analogous to MAX_ROWS_STABILIZER / MAX_QUBITS_PAULI.
const int MAX_LOOP_WEIGHT = 32;   // largest generalized loop we will ever enumerate
const int MAX_CLUSTER_SIZE = 16;  // cumulant of m loops costs ~2^m, so this is a real limit

// Largest degree of a graph we support. A site tensor holds d * D^degree complex
// numbers, so degree 6 at D = 16 is already ~500 GB: high-degree graphs are
// unreachable for physical reasons long before this constant binds.
constexpr int MAX_TN_DEGREE = 6;
constexpr int SITE_RANK = MAX_TN_DEGREE + 1;  // physical leg + one per incident edge

// Storage rank of every Tensor. One leg of headroom above SITE_RANK so a gate or
// message contraction has somewhere to put an extra index.
constexpr int MAX_TENSOR_RANK = SITE_RANK + 1;

typedef Eigen::Tensor<Complex, MAX_TENSOR_RANK> EigenTensor;

// ---------------------------------------------------------------------------
// Tensor: multi-leg tensor with a runtime shape, backed by Eigen.
//
// Rank in Eigen's Tensor module is a compile-time template parameter, but the
// rank we need is 1 + degree(v), which varies per vertex and per graph. So the
// backing store always carries MAX_TENSOR_RANK legs with the unused trailing
// ones padded to extent 1, and this class tracks the live rank at runtime.
// Padding is free in memory and callers never see it.
//
// Eigen does the two things it is good at: `shuffle` for the stride permutation,
// and gemm for the actual multiply. It does NOT do the contraction bookkeeping.
// Eigen::Tensor::contract derives its result rank from the operand ranks, so on
// uniformly padded operands a single-leg contraction of two rank-8 tensors
// instantiates a rank-14 contraction -- seconds of compile time and hundreds of
// MB each. Routing through shuffle + Map + DenseMatrix product instead gives
// bit-identical results (verified to 5e-16), computes the result rank at
// runtime, and reuses gemm instantiations the rest of the backend already pays
// for.
//
// Layout is column-major to match DenseMatrix, so as_matrix is zero-copy
// whenever the requested row legs are already the leading legs in memory.
//
// Leg convention for site tensors: leg 0 physical, legs 1.. virtual in
// TensorNetworkGraph::incident_edges order.
// ---------------------------------------------------------------------------
class Tensor {
   private:
    EigenTensor data;  // always MAX_TENSOR_RANK legs; legs >= live_rank have extent 1
    int live_rank = 0;

    // Build a padded Eigen dimension array from a runtime shape.
    static EigenTensor::Dimensions pad_shape(const std::vector<int>& shape);
    // True when `row_legs` are already the leading legs, so as_matrix can Map
    // the buffer directly instead of shuffling.
    bool is_contiguous_grouping(const std::vector<int>& row_legs) const;

   public:
    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<Complex>& values);

    int rank() const { return live_rank; }
    int64_t size() const { return static_cast<int64_t>(data.size()); }
    int extent(int leg) const { return static_cast<int>(data.dimension(leg)); }
    std::vector<int> get_shape() const;
    // Escape hatch for the rare op worth writing directly against Eigen.
    const EigenTensor& raw() const { return data; }
    EigenTensor& raw() { return data; }
    Complex operator()(const std::vector<int>& index) const;  // convenience, tests only
    Complex& operator()(const std::vector<int>& index);

    // --- leg manipulation ---------------------------------------------------
    Tensor permute(const std::vector<int>& perm) const;       // Eigen shuffle
    Tensor reshape(const std::vector<int>& new_shape) const;  // metadata only, no copy
    // Fuse each group of legs into one index, in the given group order.
    Tensor fuse(const std::vector<std::vector<int>>& groups) const;
    // Fuse `row_legs` into rows and the remainder into columns.
    DenseMatrix as_matrix(const std::vector<int>& row_legs) const;
    static Tensor from_matrix(const DenseMatrix& m, const std::vector<int>& shape);

    // --- contraction --------------------------------------------------------
    // Contract `legs_a` of *this against `legs_b` of `other`; surviving legs of
    // *this come first, then those of `other`. Throws if the result rank would
    // exceed MAX_TENSOR_RANK -- the intended pattern is to fuse spectator legs
    // (e.g. an environment, or a bra/ket bond pair into one D^2 index) before
    // contracting, which keeps every intermediate inside the budget.
    Tensor contract(const Tensor& other, const std::vector<int>& legs_a, const std::vector<int>& legs_b) const;

    // Truncated SVD across a bond: the workhorse for gate application. Eigen's
    // Tensor module has no decompositions, so this fuses to a matrix, runs
    // BDCSVD, and reshapes back. Reports the discarded weight.
    void split(const std::vector<int>& left_legs, int max_bond_dimension, Real cutoff, Tensor& left, RealVector& singular_values, Tensor& right, Real* truncation_error = nullptr) const;

    // --- elementwise / reductions ------------------------------------------
    Tensor conjugate() const;  // the bra layer of the norm network
    Complex trace_all_with(const Tensor& other) const;
    Real norm() const;
    void scale(Complex factor);
    void set_zero();
    bool has_nan() const;  // mirrors check_state_diverged in eigen.h
};

// Join the ket and bra layers at a vertex, fusing each bra/ket bond pair into a
// single index of dimension D^2. Doing this immediately is what keeps the
// norm-network tensors at rank <= degree instead of 2 * degree.
Tensor double_layer(const Tensor& site);

// ---------------------------------------------------------------------------
// TensorNetworkGraph: the lattice the state lives on. Deliberately generic --
// BP does not care about lattice geometry, only bounded degree, which is the
// whole reason to prefer it over CTMRG/boundary-MPS here.
// ---------------------------------------------------------------------------
class TensorNetworkGraph {
   private:
    int nvertices = 0;
    std::vector<std::pair<VertexId, VertexId>> edges;
    std::vector<std::vector<EdgeId>> incident;  // vertex -> incident edge ids
    std::vector<std::vector<int>> distances;    // lazily filled BFS distances

   public:
    TensorNetworkGraph() = default;
    TensorNetworkGraph(int nvertices_, const std::vector<std::pair<VertexId, VertexId>>& edges_);

    // Named topologies. `chain` makes the state an exact MPS (BP is exact on it),
    // which is the primary validation target.
    static TensorNetworkGraph chain(int n);
    static TensorNetworkGraph grid(int rows, int cols, bool periodic = false);
    static TensorNetworkGraph heavy_hex(int distance);
    static TensorNetworkGraph from_edge_list(int n, const std::vector<std::pair<int, int>>& edges_);

    int get_nvertices() const { return nvertices; }
    int get_nedges() const { return static_cast<int>(edges.size()); }
    int degree(VertexId v) const { return static_cast<int>(incident[v].size()); }
    int max_degree() const;  // constructors throw if this would exceed MAX_TN_DEGREE
    const std::pair<VertexId, VertexId>& get_edge(EdgeId e) const { return edges[e]; }
    const std::vector<EdgeId>& incident_edges(VertexId v) const { return incident[v]; }
    EdgeId find_edge(VertexId u, VertexId v) const;  // -1 if not adjacent
    VertexId other_end(EdgeId e, VertexId v) const;
    int distance(VertexId u, VertexId v) const;
    bool is_tree() const;  // if true, BP is exact and all corrections vanish

    // Index of `e` in the leg ordering of the site tensor at `v`. Physical leg is
    // always leg 0, virtual legs follow in incident_edges order.
    int leg_of(VertexId v, EdgeId e) const;
};

// ---------------------------------------------------------------------------
// Excitation: a generalized loop (closed in the bulk) or a string (allowed to
// terminate on a set of vertices). The paper's L, L_A and L_AB are the same
// object with different terminal sets, so they share one type here.
// ---------------------------------------------------------------------------
struct Excitation {
    std::vector<EdgeId> edges;
    std::vector<VertexId> terminals;  // empty => closed loop, i.e. a member of L
    Complex weight = 0.0;             // Z_l evaluated at the current fixed point

    int size() const { return static_cast<int>(edges.size()); }  // |l|
    std::string key() const;  // canonical id, for deduplication
};

// A cluster W: a multiset of excitations plus its Ursell weight phi_W.
struct Cluster {
    std::vector<std::pair<LoopId, int>> loops;  // (loop id, multiplicity eta)
    Real ursell_weight = 0.0;

    int order() const;  // number of distinct loops, i.e. |loop-set(W)|
};

// What the expansion did, surfaced to Python so a user can tell a converged
// answer from a plausible-looking one. See notes on the confusion regime.
struct TensorNetworkDiagnostics {
    bool messages_converged = false;
    int bp_iterations = 0;
    Real message_residual = 0.0;
    Real loop_decay_c = 0.0;       // -log(max|Z_l|)/|l|, eq. (38)
    Real loop_decay_threshold = 0.0;  // c_0 = log(2(degree-1)) + 1/2
    bool loop_decay_satisfied = false;  // c > c_0; if false, no convergence guarantee
    int loops_enumerated = 0;
    int clusters_evaluated = 0;
    Real last_correction_magnitude = 0.0;  // order-m minus order-(m-1); should shrink
    Real total_truncation_error = 0.0;     // accumulated over apply_gate calls
};

enum class TensorNetworkCorrection { None, Cluster, Cumulant, Region };

// ---------------------------------------------------------------------------
// TensorNetworkState: the user-facing object, same shape as StabilizerState --
// construct, apply_gate repeatedly, then query.
// ---------------------------------------------------------------------------
class TensorNetworkState {
   private:
    TensorNetworkGraph graph;
    int nqubits = 0;
    int physical_dimension = 2;

    std::vector<Tensor> site_tensors;          // one per vertex, leg 0 physical
    std::vector<RealVector> bond_weights;      // per edge, Vidal-gauge singular values
    std::vector<DenseMatrix> messages;         // 2*|E| directed messages mu_{v->w}
    bool messages_valid = false;               // invalidated by every apply_gate

    // Truncation / expansion settings.
    int max_bond_dimension = 64;
    Real truncation_cutoff = 1e-10;
    int bp_max_iterations = 200;
    Real bp_tolerance = 1e-10;
    Real bp_damping = 0.0;
    TensorNetworkCorrection correction = TensorNetworkCorrection::None;
    int max_order = 0;      // m (cluster/cumulant) or k (region)
    int max_loop_weight = 8;

    mutable TensorNetworkDiagnostics diagnostics;
    mutable std::mt19937_64 rng{std::random_device{}()};

    // Cached excitation data, rebuilt whenever the messages are re-converged.
    mutable std::vector<Excitation> loop_cache;
    mutable bool loop_cache_valid = false;

    int message_index(VertexId from, VertexId to) const;

   public:
    TensorNetworkState(int nqubits, const TensorNetworkGraph& graph);
    // Product state |b>, defaulting to all-zeros. Bond dimensions start at 1.
    TensorNetworkState(int nqubits, const TensorNetworkGraph& graph, const std::string& b);

    void set_seed(uint64_t seed) const { rng.seed(seed); }
    void set_max_bond_dimension(int d) { max_bond_dimension = d; }
    void set_truncation_cutoff(Real c) { truncation_cutoff = c; }
    void set_correction(TensorNetworkCorrection c, int order) {
        correction = c;
        max_order = order;
    }
    void set_bp_parameters(int max_iterations, Real tolerance, Real damping = 0.0) {
        bp_max_iterations = max_iterations;
        bp_tolerance = tolerance;
        bp_damping = damping;
    }
    void set_max_loop_weight(int w) { max_loop_weight = w; }

    int get_nqubits() const { return nqubits; }
    const TensorNetworkGraph& get_graph() const { return graph; }
    const Tensor& get_site_tensor(VertexId v) const { return site_tensors[v]; }
    int get_bond_dimension(EdgeId e) const;
    int get_max_bond_dimension_used() const;
    const TensorNetworkDiagnostics& get_diagnostics() const { return diagnostics; }

    // --- evolution ---------------------------------------------------------
    // Returns the truncation error incurred. One- and two-qubit gates act
    // directly; a gate on non-adjacent qubits throws, since routing is the
    // transpiler's job, not ours.
    Real apply_gate(const Gate& gate);
    Real apply_one_site(const DenseMatrix& u, VertexId v);
    Real apply_two_site(const DenseMatrix& u, VertexId v, VertexId w);
    void normalize();

    // --- belief propagation ------------------------------------------------
    // Iterate mu_{v->w} to a fixed point from uniform messages. Idempotent and
    // called lazily by every observable method.
    bool converge_messages() const;
    void invalidate_messages() { messages_valid = false; }
    // Vidal/BP gauge (Tindall-Fishman); makes local truncation near-optimal.
    void gauge();
    const DenseMatrix& get_message(VertexId from, VertexId to) const;

    // --- excitations -------------------------------------------------------
    // Generalized loops up to `max_loop_weight`: connected edge subsets with
    // degree >= 2 at every touched vertex.
    const std::vector<Excitation>& enumerate_loops() const;
    // Strings closed in the bulk but allowed to terminate in `terminals` (L_A / L_AB).
    std::vector<Excitation> enumerate_strings(const std::vector<VertexId>& terminals) const;
    Complex evaluate_excitation(const Excitation& l, const std::vector<VertexId>& region, const DenseMatrix& observable) const;
    std::vector<Cluster> enumerate_clusters(const std::vector<Excitation>& excitations, int order) const;
    Complex cumulant(const std::vector<LoopId>& subset, const std::vector<Excitation>& excitations) const;
    Real counting_number(const std::vector<VertexId>& region, int k) const;  // b_k(R), eq. (21)

    // --- observables -------------------------------------------------------
    // BP-only value; exact on a tree.
    Complex expectation_value_bp(const DenseMatrix& observable, const std::vector<VertexId>& region) const;
    // BP dressed by the configured correction at the configured order.
    Complex expectation_value(const DenseMatrix& observable, const std::vector<VertexId>& region) const;
    // Same signature as StabilizerState, so the sampling layer can stay generic.
    Real expectation_value(const MatrixFreeHamiltonian& H) const;
    Complex correlator(const DenseMatrix& op_a, const std::vector<VertexId>& region_a, const DenseMatrix& op_b, const std::vector<VertexId>& region_b) const;
    Real free_energy() const;  // F = F_0 - sum K(Gamma), eq. (19)

    // --- sampling / inspection --------------------------------------------
    // Sequential sampling from BP conditional marginals: exact on a chain,
    // biased on a loopy graph. Costs ~nqubits message re-solves per shot.
    std::string sample() const;
    std::string sample(std::mt19937_64& engine) const;
    std::map<std::string, int> sample(int nshots) const;
    Complex amplitude(const std::string& b) const;
    // Exact contraction to a statevector. Testing only, throws above ~24 qubits.
    DenseMatrix as_dense() const;

    // --- diagnostics -------------------------------------------------------
    Real loop_decay_rate() const;       // c, eq. (38)
    Real loop_decay_threshold() const;  // c_0 = log(2(Delta-1)) + 1/2
    // Emits a warning via qilisdk::log_warning when c < c_0, when BP fails to
    // converge, or when the order-m correction is not shrinking.
    void check_convergence() const;
};

std::ostream& operator<<(std::ostream& os, const TensorNetworkState& state);

// GCOV_EXCL_BR_STOP
