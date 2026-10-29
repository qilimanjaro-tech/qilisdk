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

#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>

#include "../../../libs/pybind.h"
#include "bp.h"

// GCOV_EXCL_BR_START

namespace {

// Eigenvalues of a message below this fraction of the largest are treated as zero when inverting its square root
constexpr Real ROOT_CUTOFF = 1e-12;

// A matrix view with both an inner and an outer stride, for one leg of a tensor against the legs after it
using StridedView = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

Tensor apply_on_leg(const Tensor& t, int leg, const DenseMatrix& m) {
    /*
    Contract one leg of a tensor with the rows of a matrix, leaving the new leg where the old one was:

        out(..., c, ...) = sum_a t(..., a, ...) m(a, c)

    With the legs before it fused into one and the legs after it fused into another, this is one matrix product per
    index of whichever of the two is shorter, taken on strided views so no data is permuted.

    Args:
        t (Tensor&): The tensor.
        leg (int): The leg to contract.
        m (DenseMatrix&): The matrix, with as many rows as the leg has extent.

    Returns:
        Tensor: The tensor with the leg replaced.
    */
    std::vector<int> shape = t.get_shape();
    int64_t before = 1;
    for (int i = 0; i < leg; ++i) {
        before *= shape[i];
    }
    int64_t extent = shape[leg];
    int64_t after = t.size() / (before * extent);
    int64_t columns = m.cols();
    shape[leg] = static_cast<int>(columns);
    Tensor out(shape);
    const Complex* source = t.raw().data();
    Complex* target = out.raw().data();
    if (before <= after) {
        for (int64_t row = 0; row < before; ++row) {
            Eigen::Map<const DenseMatrix, 0, StridedView> in_view(source + row, extent, after, StridedView(before * extent, before));
            Eigen::Map<DenseMatrix, 0, StridedView> out_view(target + row, columns, after, StridedView(before * columns, before));
            out_view.noalias() = m.transpose() * in_view;
        }
    } else {
        for (int64_t slice = 0; slice < after; ++slice) {
            Eigen::Map<DenseMatrix>(target + slice * before * columns, before, columns).noalias() = Eigen::Map<const DenseMatrix>(source + slice * before * extent, before, extent) * m;
        }
    }
    return out;
}

void hermitian_roots(const DenseMatrix& m, DenseMatrix& root, DenseMatrix& inverse_root) {
    /*
    The square root of a positive semidefinite message and its pseudo-inverse.

    Args:
        m (DenseMatrix&): The Hermitian positive semidefinite matrix.
        root (DenseMatrix&): Output, the matrix square root.
        inverse_root (DenseMatrix&): Output, its pseudo-inverse, dropping eigenvalues below ROOT_CUTOFF times the largest.
    */
    Eigen::SelfAdjointEigenSolver<DenseMatrix> solver(m);
    const RealVector& values = solver.eigenvalues();
    Real threshold = ROOT_CUTOFF * std::max(values.maxCoeff(), Real(0));
    RealVector roots(values.size());
    RealVector inverse_roots(values.size());
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        roots(i) = std::sqrt(std::max(values(i), Real(0)));
        inverse_roots(i) = values(i) > threshold ? Real(1) / roots(i) : Real(0);
    }
    const DenseMatrix& vectors = solver.eigenvectors();
    root = vectors * roots.asDiagonal() * vectors.adjoint();
    inverse_root = vectors * inverse_roots.asDiagonal() * vectors.adjoint();
}

Tensor contract_network(std::vector<Tensor> tensors, std::vector<std::vector<int>> labels, std::vector<int>& result_labels) {
    /*
    Contract a small labeled network, greedily absorbing whichever tensor shares the most labels with the result so far.

    Args:
        tensors (std::vector<Tensor>): The tensors.
        labels (std::vector<std::vector<int>>): The label of each leg of each tensor. A label shared by two legs is contracted.
        result_labels (std::vector<int>&): Output, the labels of the open legs of the result.

    Returns:
        Tensor: The contracted network.

    Raises:
        std::runtime_error: If a step would build a tensor with more than BPState::MAX_CONTRACTION_ENTRIES entries.
    */
    Tensor result = tensors[0];
    result_labels = labels[0];
    std::vector<bool> used(tensors.size(), false);
    used[0] = true;
    for (size_t step = 1; step < tensors.size(); ++step) {
        size_t best = 0;
        int best_shared = -1;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (used[i]) {
                continue;
            }
            int shared = 0;
            for (int label : labels[i]) {
                shared += static_cast<int>(std::count(result_labels.begin(), result_labels.end(), label));
            }
            if (shared > best_shared) {
                best_shared = shared;
                best = i;
            }
        }
        used[best] = true;

        // Contract every leg the two share a label on, the surviving legs of the result coming first
        std::vector<int> legs_result;
        std::vector<int> legs_best;
        std::vector<int> next_labels;
        int64_t next_size = 1;
        for (int i = 0; i < static_cast<int>(result_labels.size()); ++i) {
            auto found = std::find(labels[best].begin(), labels[best].end(), result_labels[i]);
            if (found != labels[best].end()) {
                legs_result.push_back(i);
                legs_best.push_back(static_cast<int>(found - labels[best].begin()));
            } else {
                next_labels.push_back(result_labels[i]);
                next_size *= result.extent(i);
            }
        }
        for (int i = 0; i < static_cast<int>(labels[best].size()); ++i) {
            if (std::find(legs_best.begin(), legs_best.end(), i) == legs_best.end()) {
                next_labels.push_back(labels[best][i]);
                next_size *= tensors[best].extent(i);
            }
        }
        if (next_size > BPState::MAX_CONTRACTION_ENTRIES) {
            throw std::runtime_error("Contracting the tensor network would build a tensor with " + std::to_string(next_size) + " entries, more than the limit of " + std::to_string(BPState::MAX_CONTRACTION_ENTRIES));
        }
        result = result.contract(tensors[best], legs_result, legs_best);
        result_labels = next_labels;
    }
    return result;
}

}  // namespace

// Constructors
BPState::BPState(int nqubits) : BPState(nqubits, std::string(nqubits > 0 ? nqubits : 0, '0')) {}
BPState::BPState(int nqubits, const std::string& b) : nqubits(nqubits) {
    /*
    Build the product state |b>, with no edges. Edges are added as two-qubit gates first touch each pair.

    Args:
        nqubits (int): The number of qubits.
        b (std::string&): The bitstring, one character per qubit, qubit 0 first.

    Raises:
        std::invalid_argument: If nqubits is not positive, or b does not match it.
    */
    if (nqubits <= 0) {
        throw std::invalid_argument("A tensor network needs at least one qubit, got " + std::to_string(nqubits));
    }
    if (static_cast<int>(b.size()) != nqubits) {
        throw std::invalid_argument("Bitstring '" + b + "' has " + std::to_string(b.size()) + " characters but the state has " + std::to_string(nqubits) + " qubits");
    }
    sites.reserve(nqubits);
    incident_edges.resize(nqubits);
    touched.assign(nqubits, false);
    for (int q = 0; q < nqubits; ++q) {
        if (b[q] != '0' && b[q] != '1') {
            throw std::invalid_argument("Bitstring '" + b + "' must contain only '0' and '1'");
        }
        Tensor site({2});
        site({b[q] == '1' ? 1 : 0}) = 1.0;
        sites.push_back(site);
    }
}

void BPState::set_max_bond_dimension(int d) {
    /*
    Set the largest bond dimension kept after each two-qubit gate.

    Args:
        d (int): The bond dimension cap.

    Raises:
        std::invalid_argument: If d is not positive.
    */
    if (d <= 0) {
        throw std::invalid_argument("The maximum BP bond dimension must be positive, got " + std::to_string(d));
    }
    max_bond_dimension = d;
}

void BPState::set_truncation_cutoff(Real c) {
    /*
    Set the relative cutoff below which singular values are discarded.

    Args:
        c (Real): The cutoff, as a fraction of the largest singular value.

    Raises:
        std::invalid_argument: If c is NaN, negative, or 1 or more.
    */
    if (!(c >= Real(0) && c < Real(1))) {
        throw std::invalid_argument("The BP truncation cutoff must be in [0, 1), got " + std::to_string(c));
    }
    truncation_cutoff = c;
}

void BPState::set_max_iterations(int n) {
    /*
    Set the most message-passing sweeps a single belief propagation run may take.

    Args:
        n (int): The iteration cap.

    Raises:
        std::invalid_argument: If n is not positive.
    */
    if (n <= 0) {
        throw std::invalid_argument("The maximum number of BP iterations must be positive, got " + std::to_string(n));
    }
    max_iterations = n;
}

void BPState::set_tolerance(Real t) {
    /*
    Set the largest change in any message at which belief propagation counts as converged.

    Args:
        t (Real): The tolerance.

    Raises:
        std::invalid_argument: If t is NaN or not positive.
    */
    if (!(t > Real(0))) {
        throw std::invalid_argument("The BP tolerance must be positive, got " + std::to_string(t));
    }
    tolerance = t;
}

int BPState::get_max_bond_dimension_used() const {
    /*
    Get the largest bond dimension on any edge.

    Returns:
        int: The largest bond dimension, 1 if there are no edges.
    */
    int largest = 1;
    for (int e = 0; e < get_num_edges(); ++e) {
        largest = std::max(largest, sites[edges[e].first].extent(leg_of(edges[e].first, e)));
    }
    return largest;
}

int BPState::leg_of(int v, int e) const {
    /*
    The leg of qubit v's tensor that carries edge e.

    Args:
        v (int): The qubit.
        e (int): The edge.

    Returns:
        int: The leg index, counting the physical leg as 0.

    Raises:
        std::invalid_argument: If the edge does not touch the qubit.
    */
    const std::vector<int>& incident = incident_edges[v];
    auto found = std::find(incident.begin(), incident.end(), e);
    // Every caller passes an edge it found in the qubit's own incident list, so this cannot fail
    if (found == incident.end()) {
        throw std::invalid_argument("Edge " + std::to_string(e) + " does not touch qubit " + std::to_string(v));  // GCOV_EXCL_LINE
    }
    return 1 + static_cast<int>(found - incident.begin());
}

int BPState::run_belief_propagation() {
    /*
    Iterate the message updates in parallel until the largest change falls below the tolerance, or the
    iteration cap is reached, then refresh the square roots of the messages that changed. The current messages are the starting point, and each sweep only updates the
    messages sent by qubits that changed: those touched by a gate since the last run, and then those whose
    incoming messages moved by more than the tolerance in the previous sweep. A run after a layer of gates
    therefore costs about as much as the qubits in the layer, and usually converges in a handful of sweeps.

    Each update applies every other incoming message to the sending tensor and closes it against
    its conjugate, leaving the two copies of the bond open, then Hermitizes and normalizes to unit trace:

        m(a, a') = sum T(p, b, a) prod_f m_f(b_f, b'_f) conj(T(p, b', a'))

    Returns:
        int: The number of sweeps taken.

    Raises:
        py::error_already_set: If a Python signal handler raised between sweeps, e.g. KeyboardInterrupt.
    */
    int num_messages = 2 * get_num_edges();
    if (num_messages == 0) {
        touched.assign(nqubits, false);
        return 0;
    }
    std::vector<DenseMatrix> next(num_messages);
    std::vector<bool> changed = touched;
    std::vector<bool> updated(num_messages, false);
    int iteration = 0;
    bool converged = false;
    while (iteration < max_iterations && !converged) {
        ++iteration;

        // Let Ctrl-C interrupt a long run between sweeps
        check_signals();

        // Only the messages sent by qubits that changed can change
        std::vector<int> pending;
        for (int index = 0; index < num_messages; ++index) {
            const std::pair<int, int>& edge = edges[index / 2];
            if (changed[index % 2 == 0 ? edge.first : edge.second]) {
                pending.push_back(index);
            }
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(pending.size()); ++i) {
            int index = pending[i];
            int e = index / 2;
            int from = (index % 2 == 0) ? edges[e].first : edges[e].second;
            const Tensor& site = sites[from];
            Tensor ket = site;
            for (int f : incident_edges[from]) {
                if (f != e) {
                    ket = apply_on_leg(ket, leg_of(from, f), message_into(from, f));
                }
            }

            // Close every leg but the open one, looping over whichever of the legs before or after it is shorter
            int open_leg = leg_of(from, e);
            int64_t before = 1;
            for (int leg = 0; leg < open_leg; ++leg) {
                before *= site.extent(leg);
            }
            int64_t bond = site.extent(open_leg);
            int64_t after = site.size() / (before * bond);
            DenseMatrix message = DenseMatrix::Zero(bond, bond);
            if (before <= after) {
                for (int64_t row = 0; row < before; ++row) {
                    Eigen::Map<const DenseMatrix, 0, StridedView> ket_view(ket.raw().data() + row, bond, after, StridedView(before * bond, before));
                    Eigen::Map<const DenseMatrix, 0, StridedView> bra_view(site.raw().data() + row, bond, after, StridedView(before * bond, before));
                    message.noalias() += ket_view * bra_view.adjoint();
                }
            } else {
                for (int64_t slice = 0; slice < after; ++slice) {
                    Eigen::Map<const DenseMatrix> ket_slice(ket.raw().data() + slice * before * bond, before, bond);
                    Eigen::Map<const DenseMatrix> bra_slice(site.raw().data() + slice * before * bond, before, bond);
                    message.noalias() += ket_slice.transpose() * bra_slice.conjugate();
                }
            }
            message = (message + message.adjoint()) / 2.0;
            Real trace = message.trace().real();
            next[index] = trace > Real(0) ? DenseMatrix(message / trace) : DenseMatrix(DenseMatrix::Identity(bond, bond) / static_cast<Real>(bond));
        }

        // A message that moved changes the qubit it points to for the next sweep
        changed.assign(nqubits, false);
        converged = true;
        for (int index : pending) {
            const std::pair<int, int>& edge = edges[index / 2];
            if ((next[index] - messages[index / 2][index % 2]).norm() >= tolerance) {
                converged = false;
                changed[index % 2 == 0 ? edge.second : edge.first] = true;
            }
            messages[index / 2][index % 2] = next[index];
            updated[index] = true;
        }
    }
#pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < num_messages; ++index) {
        if (updated[index]) {
            hermitian_roots(messages[index / 2][index % 2], message_roots[index / 2][index % 2], message_inverse_roots[index / 2][index % 2]);
        }
    }
    touched.assign(nqubits, false);
    ++bp_runs;
    bp_iterations += iteration;
    bp_converged = bp_converged && converged;
    return iteration;
}

Tensor BPState::gauge_and_reduce(int v, int e, Tensor& isometry) const {
    /*
    Prepare one side of a two-qubit gate on edge e. Every other bond of qubit v is gauged with the square root
    of its incoming message, so that under belief propagation the rest of the network acts as the identity there,
    and then the tensor is QR-decomposed with the physical leg and bond e on the R side. The gate and its
    truncation then only touch R, whose size does not depend on the coordination of the qubit.

    Args:
        v (int): The qubit.
        e (int): The edge the gate acts across.
        isometry (Tensor&): Output, Q with legs (other bonds..., new bond).

    Returns:
        Tensor: R, with legs (new bond, physical, bond e).
    */
    Tensor ket = sites[v];
    std::vector<int> perm;
    std::vector<int> other_shape;
    for (int f : incident_edges[v]) {
        if (f == e) {
            continue;
        }
        int leg = leg_of(v, f);
        ket = apply_on_leg(ket, leg, root_into(v, f));
        perm.push_back(leg);
        other_shape.push_back(ket.extent(leg));
    }
    int num_other = static_cast<int>(perm.size());
    int bond = ket.extent(leg_of(v, e));
    perm.push_back(0);
    perm.push_back(leg_of(v, e));
    const Tensor reordered = ket.permute(perm);
    Eigen::Map<const DenseMatrix> matrix = reordered.matrix_view(num_other);
    Eigen::Index rows = matrix.rows();
    Eigen::Index cols = matrix.cols();

    // Only factorize when it actually shrinks the block the gate acts on
    if (rows <= cols) {
        std::vector<int> identity_shape = other_shape;
        identity_shape.push_back(static_cast<int>(rows));
        isometry = Tensor::from_matrix(DenseMatrix::Identity(rows, rows), identity_shape);
        return Tensor::from_matrix(matrix, {static_cast<int>(rows), 2, bond});
    }
    Eigen::HouseholderQR<DenseMatrix> qr(matrix);
    std::vector<int> isometry_shape = other_shape;
    isometry_shape.push_back(static_cast<int>(cols));
    isometry = Tensor::from_matrix(DenseMatrix(qr.householderQ() * DenseMatrix::Identity(rows, cols)), isometry_shape);
    DenseMatrix r_factor = qr.matrixQR().topRows(cols).triangularView<Eigen::Upper>();
    return Tensor::from_matrix(r_factor, {static_cast<int>(cols), 2, bond});
}

Tensor BPState::restore_site(int v, int e, const Tensor& isometry, const Tensor& reduced) const {
    /*
    Undo gauge_and_reduce once the gate has been applied to R: multiply Q back in, remove the gauge from every
    other bond, and return the legs to their usual order.

    Args:
        v (int): The qubit.
        e (int): The edge the gate acted across.
        isometry (Tensor&): Q from gauge_and_reduce.
        reduced (Tensor&): The updated R, with legs (Q bond, physical, bond e).

    Returns:
        Tensor: The new site tensor, with legs (physical, incident bonds...).
    */
    int num_other = isometry.rank() - 1;
    Tensor site = isometry.contract(reduced, {num_other}, {0});
    std::vector<int> perm = {num_other};
    int other = 0;
    for (int f : incident_edges[v]) {
        if (f == e) {
            perm.push_back(num_other + 1);
        } else {
            site = apply_on_leg(site, other, inverse_root_into(v, f));
            perm.push_back(other++);
        }
    }
    return site.permute(perm);
}

Real BPState::apply_one_site(const DenseMatrix& u, int q) {
    /*
    Apply a single-qubit matrix to qubit q. Exact, and a unitary leaves every message intact.

    Args:
        u (DenseMatrix&): The 2 x 2 matrix to apply.
        q (int): The qubit to apply it to.

    Returns:
        Real: The truncation error, which is always 0.0 for a single-qubit gate.

    Raises:
        std::invalid_argument: If u is not 2 x 2.
        std::out_of_range: If q is not a valid qubit index.
    */
    if (u.rows() != 2 || u.cols() != 2) {
        throw std::invalid_argument("A single-qubit gate matrix must be 2 x 2");
    }
    if (q < 0 || q >= nqubits) {
        throw std::out_of_range("Qubit " + std::to_string(q) + " is out of range for a " + std::to_string(nqubits) + " qubit tensor network");
    }
    Eigen::Map<DenseMatrix> site = sites[q].matrix_view(1);
    DenseMatrix updated = u * site;
    site = updated;
    // Only a non-unitary gate changes the messages this qubit sends
    if (!DenseMatrix(u * u.adjoint()).isApprox(DenseMatrix::Identity(2, 2))) {
        touched[q] = true;
    }
    return 0.0;
}

Real BPState::apply_two_site(const DenseMatrix& u, int q0, int q1) {
    /*
    Apply a two-qubit matrix to qubits (q0, q1), with q0 the most significant index of u, adding an edge between
    them if there is none. This is the simple update in the belief propagation gauge: the two tensors are gauged
    with the square roots of their incoming messages, reduced by QR, the gate applied and the bond truncated by SVD,
    after which the gauge is removed. The truncation is optimal for the global state whenever the environment is
    the product of messages, which holds exactly on a tree. The new singular values become the messages across the
    edge. As in the usual layered scheme, belief propagation is only re-run when a gate touches a qubit already changed
    since the last run, so gates within a layer of disjoint pairs share the messages from the start of the layer.

    Args:
        u (DenseMatrix&): The 4 x 4 matrix to apply.
        q0 (int): The first qubit.
        q1 (int): The second qubit.

    Returns:
        Real: The discarded weight from the SVD.

    Raises:
        std::invalid_argument: If u is not 4 x 4, or the qubits coincide.
        std::out_of_range: If either qubit is outside the register.
    */
    if (u.rows() != 4 || u.cols() != 4) {
        throw std::invalid_argument("A two-qubit gate matrix must be 4 x 4");
    }
    if (q0 < 0 || q0 >= nqubits || q1 < 0 || q1 >= nqubits) {
        throw std::out_of_range("Qubits (" + std::to_string(q0) + ", " + std::to_string(q1) + ") are out of range for a " + std::to_string(nqubits) + " qubit tensor network");
    }
    if (q0 == q1) {
        throw std::invalid_argument("A two-qubit gate needs two different qubits, got " + std::to_string(q0) + " twice");
    }

    // Find the edge, or add one with a bond of dimension 1 appended as the last leg of both tensors
    std::pair<int, int> key = {std::min(q0, q1), std::max(q0, q1)};
    auto found = edge_lookup.find(key);
    int e = found != edge_lookup.end() ? found->second : get_num_edges();
    if (e == get_num_edges()) {
        edges.push_back(key);
        edge_lookup[key] = e;
        for (int q : {q0, q1}) {
            std::vector<int> shape = sites[q].get_shape();
            shape.push_back(1);
            sites[q].reshape(shape);
            incident_edges[q].push_back(e);
        }
        messages.push_back({DenseMatrix::Ones(1, 1), DenseMatrix::Ones(1, 1)});
        message_roots.push_back(messages.back());
        message_inverse_roots.push_back(messages.back());
    }

    // A gate on a qubit already changed in this layer starts a new layer, which needs fresh messages
    if (touched[q0] || touched[q1]) {
        run_belief_propagation();
    }
    touched[q0] = true;
    touched[q1] = true;

    // Reduce both sides to small blocks and join them across the edge, giving legs (r0, p0, r1, p1)
    Tensor isometry0;
    Tensor isometry1;
    Tensor reduced0 = gauge_and_reduce(q0, e, isometry0);
    Tensor reduced1 = gauge_and_reduce(q1, e, isometry1);
    Tensor theta = reduced0.contract(reduced1, {2}, {2});

    // Put (p1, p0) first so the fused row index is 2 p0 + p1, matching the gate, then put the legs back as (r0, p0, p1, r1)
    Tensor gated = theta.permute({3, 1, 0, 2});
    Eigen::Map<DenseMatrix> rows = gated.matrix_view(2);
    DenseMatrix updated = u * rows;
    rows = updated;
    theta = gated.permute({2, 1, 0, 3});

    // Truncate and share the singular values evenly between the two sides
    Tensor left;
    Tensor right;
    RealVector singular_values;
    Real error = 0.0;
    theta.split({0, 1}, max_bond_dimension, truncation_cutoff, left, singular_values, right, &error);
    Real weight = singular_values.norm();
    if (weight > Real(0)) {
        singular_values /= weight;
    }
    int bond = static_cast<int>(singular_values.size());
    Eigen::Map<DenseMatrix> left_matrix = left.matrix_view(2);
    Eigen::Map<DenseMatrix> right_matrix = right.matrix_view(1);
    for (int b = 0; b < bond; ++b) {
        Real root = std::sqrt(singular_values(b));
        left_matrix.col(b) *= root;
        right_matrix.row(b) *= root;
    }

    sites[q0] = restore_site(q0, e, isometry0, left);
    sites[q1] = restore_site(q1, e, isometry1, right.permute({2, 1, 0}));

    // In the gauged frame each side is an isometry times the root of the singular values, so both messages across e are diag(S), or uniform if the state vanished
    Real total = singular_values.sum();
    DenseMatrix diagonal = total > Real(0) ? DenseMatrix((singular_values / total).cast<Complex>().asDiagonal()) : DenseMatrix(DenseMatrix::Identity(bond, bond) / static_cast<Real>(bond));
    messages[e] = {diagonal, diagonal};
    hermitian_roots(diagonal, message_roots[e][0], message_inverse_roots[e][0]);
    message_roots[e][1] = message_roots[e][0];
    message_inverse_roots[e][1] = message_inverse_roots[e][0];

    total_truncation_error += error;
    return error;
}

Real BPState::apply_gate(const Gate& gate) {
    /*
    Apply a gate to the state. One- and two-qubit gates are supported, and a two-qubit gate may act on any pair.

    Args:
        gate (Gate&): The gate to apply.

    Returns:
        Real: The truncation error incurred.

    Raises:
        std::invalid_argument: If the gate acts on more than two qubits.
    */
    std::vector<int> qubits = gate.get_qubits();
    if (qubits.size() == 1) {
        return apply_one_site(gate.get_local_matrix(), qubits[0]);
    }
    if (qubits.size() != 2) {
        throw std::invalid_argument("The BP simulator supports one- and two-qubit gates only, but " + gate.get_id() + " acts on " + std::to_string(qubits.size()) + " qubits");
    }
    // The local matrix runs over the qubits in ascending order
    return apply_two_site(gate.get_local_matrix(), std::min(qubits[0], qubits[1]), std::max(qubits[0], qubits[1]));
}

void BPState::check_messages_current() const {
    /*
    Check that no gate has changed the state since belief propagation last ran.

    Raises:
        std::logic_error: If the messages are stale.
    */
    if (std::find(touched.begin(), touched.end(), true) != touched.end()) {
        throw std::logic_error("The belief propagation messages are out of date, call run_belief_propagation before taking an expectation value");
    }
}

std::vector<int> BPState::cluster_of(const PauliString& pauli) const {
    /*
    Join the qubits a Pauli string acts on into a connected cluster, adding each to the cluster so far along a
    shortest path. Qubits not connected to the rest are added on their own.

    Args:
        pauli (PauliString&): The Pauli string, acting on at least one qubit.

    Returns:
        std::vector<int>: The qubits of the cluster, in ascending order.
    */
    std::vector<int> cluster;
    std::vector<bool> in_cluster(nqubits, false);
    for (int start = 0; start < nqubits; ++start) {
        if (!(pauli.x_mask[start] || pauli.z_mask[start]) || in_cluster[start]) {
            continue;
        }

        // Breadth-first search from the new qubit until it hits the cluster
        std::vector<int> parent(nqubits, -1);
        std::vector<bool> seen(nqubits, false);
        std::deque<int> queue = {start};
        seen[start] = true;
        int reached = -1;
        while (!queue.empty() && !cluster.empty() && reached < 0) {
            int q = queue.front();
            queue.pop_front();
            for (int f : incident_edges[q]) {
                int next = edges[f].first == q ? edges[f].second : edges[f].first;
                if (!seen[next]) {
                    seen[next] = true;
                    parent[next] = q;
                    queue.push_back(next);
                    reached = in_cluster[next] ? next : reached;
                }
            }
        }

        // Add the path back from where it hit, or just the qubit if it is not connected to the cluster
        for (int q = reached >= 0 ? parent[reached] : start; q >= 0; q = parent[q]) {
            in_cluster[q] = true;
            cluster.push_back(q);
        }
    }
    std::sort(cluster.begin(), cluster.end());
    return cluster;
}

std::vector<Complex> BPState::cluster_expectations(const std::vector<int>& cluster, const std::vector<const PauliString*>& paulis) const {
    /*
    The belief propagation estimates of <psi|P|psi> / <psi|psi> for Pauli strings supported on the same cluster.
    Every bond leaving the cluster is closed by its incoming message and the cluster is contracted exactly, the
    norm once and the numerator once per Pauli string, rebuilding only the tensors of the qubits it acts on.

    Args:
        cluster (std::vector<int>&): The qubits of the cluster.
        paulis (std::vector<const PauliString*>&): The Pauli strings, each supported inside the cluster.

    Returns:
        std::vector<Complex>: The estimate for each Pauli string.

    Raises:
        std::runtime_error: If the cluster has zero norm.
    */
    std::vector<bool> in_cluster(nqubits, false);
    for (int q : cluster) {
        in_cluster[q] = true;
    }

    // At each qubit, the ket with its outside bonds gauged, and the double layer with the bra and ket copies of each inside bond fused into one leg labeled by the edge
    std::vector<Tensor> kets(cluster.size());
    std::vector<std::vector<int>> closed(cluster.size(), {0});
    std::vector<std::vector<std::vector<int>>> groups(cluster.size());
    std::vector<Tensor> norm_tensors(cluster.size());
    std::vector<std::vector<int>> labels(cluster.size());
    for (size_t i = 0; i < cluster.size(); ++i) {
        int v = cluster[i];
        kets[i] = sites[v];
        for (int f : incident_edges[v]) {
            int neighbour = edges[f].first == v ? edges[f].second : edges[f].first;
            int leg = leg_of(v, f);
            if (in_cluster[neighbour]) {
                labels[i].push_back(f);
            } else {
                kets[i] = apply_on_leg(kets[i], leg, root_into(v, f));
                closed[i].push_back(leg);
            }
        }
        int num_open = static_cast<int>(labels[i].size());
        for (int leg = 0; leg < num_open; ++leg) {
            groups[i].push_back({leg, num_open + leg});
        }
        norm_tensors[i] = kets[i].conjugate().contract(kets[i], closed[i], closed[i]).fuse(groups[i]);
    }
    std::vector<int> result_labels;
    Complex denominator = contract_network(norm_tensors, labels, result_labels).raw()[0];
    if (denominator == Complex(0.0, 0.0)) {
        throw std::runtime_error("Cannot take an expectation value of a tensor network whose norm is zero");
    }

    // Insert each Pauli string on the ket layer of the qubits it acts on
    std::vector<Complex> values;
    values.reserve(paulis.size());
    for (const PauliString* pauli : paulis) {
        std::vector<Tensor> operator_tensors = norm_tensors;
        for (size_t i = 0; i < cluster.size(); ++i) {
            int v = cluster[i];
            if (!(pauli->x_mask[v] || pauli->z_mask[v])) {
                continue;
            }
            Tensor ket = kets[i];
            Eigen::Map<DenseMatrix> physical = ket.matrix_view(1);
            DenseMatrix updated = pauli_matrix(pauli->x_mask[v], pauli->z_mask[v]) * physical;
            physical = updated;
            operator_tensors[i] = kets[i].conjugate().contract(ket, closed[i], closed[i]).fuse(groups[i]);
        }
        values.push_back(contract_network(operator_tensors, labels, result_labels).raw()[0] / denominator);
    }
    return values;
}

Complex BPState::expectation_value(const PauliString& pauli) const {
    /*
    The belief propagation estimate of <psi|P|psi> / <psi|psi> for a Pauli string. The qubits it acts on are joined
    into a connected cluster, which is contracted exactly with the rest of the network replaced by the messages
    entering the cluster. On a tree this is exact, and on a loopy graph it misses only the loops leaving the cluster.

    Args:
        pauli (PauliString&): The Pauli string.

    Returns:
        Complex: The estimate.

    Raises:
        std::logic_error: If a gate changed the state since belief propagation last ran, so the messages are stale.
        std::runtime_error: If the cluster has zero norm.
    */
    check_messages_current();
    if (pauli.x_mask.none() && pauli.z_mask.none()) {
        return Complex(1.0, 0.0);
    }
    return cluster_expectations(cluster_of(pauli), {&pauli})[0];
}

Real BPState::expectation_value(const MatrixFreeHamiltonian& H) const {
    /*
    The belief propagation estimate of <psi|H|psi> / <psi|psi>. The Pauli strings of H are grouped by the cluster
    they are estimated on, so each cluster's norm is contracted once, and the clusters are handled in parallel.

    Args:
        H (MatrixFreeHamiltonian&): The Hamiltonian.

    Returns:
        Real: The estimate.

    Raises:
        std::invalid_argument: If the Hamiltonian is on a different number of qubits.
        std::logic_error: If a gate changed the state since belief propagation last ran, so the messages are stale.
        std::runtime_error: If a cluster has zero norm.
    */
    if (H.get_nqubits() != nqubits) {
        throw std::invalid_argument("Cannot take the expectation value of a " + std::to_string(H.get_nqubits()) + " qubit Hamiltonian on a " + std::to_string(nqubits) + " qubit tensor network");
    }
    check_messages_current();
    Real total = 0.0;
    std::map<std::vector<int>, std::pair<std::vector<const PauliString*>, std::vector<Complex>>> by_cluster;
    for (const auto& [pauli, coefficient] : H.get_operators()) {
        if (pauli.x_mask.none() && pauli.z_mask.none()) {
            total += coefficient.real();
        } else {
            auto& group = by_cluster[cluster_of(pauli)];
            group.first.push_back(&pauli);
            group.second.push_back(coefficient);
        }
    }
    std::vector<const decltype(by_cluster)::value_type*> groups;
    for (const auto& entry : by_cluster) {
        groups.push_back(&entry);
    }

    // An exception must not escape the parallel region, so the first one is kept and rethrown after it
    std::exception_ptr error = nullptr;
#pragma omp parallel for schedule(dynamic) reduction(+ : total)
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        try {
            const auto& [cluster, terms] = *groups[g];
            std::vector<Complex> values = cluster_expectations(cluster, terms.first);
            for (size_t t = 0; t < values.size(); ++t) {
                total += std::real(terms.second[t] * values[t]);
            }
        } catch (...) {
#pragma omp critical(bp_expectation_error)
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return total;
}

DenseMatrix BPState::as_dense() const {
    /*
    Contract the whole network exactly into a normalized statevector. This is exponential in the qubit count,
    so it is only for small systems and for checking the method.

    Returns:
        DenseMatrix: The 2^n x 1 statevector, qubit 0 the most significant bit.

    Raises:
        std::runtime_error: If the register is too large to materialize.
    */
    if (nqubits > MAX_DENSE_QUBITS) {
        throw std::runtime_error("Cannot materialize the statevector of a " + std::to_string(nqubits) + " qubit tensor network since it would need too much memory");
    }

    // Physical legs are labeled -(q + 1) so they never match a bond
    std::vector<std::vector<int>> labels;
    for (int q = 0; q < nqubits; ++q) {
        std::vector<int> site_labels = {-(q + 1)};
        site_labels.insert(site_labels.end(), incident_edges[q].begin(), incident_edges[q].end());
        labels.push_back(site_labels);
    }
    std::vector<int> result_labels;
    Tensor result = contract_network(sites, labels, result_labels);

    // The first leg runs fastest, so qubit n - 1 goes first to make qubit 0 the most significant
    std::vector<int> perm;
    for (int q = nqubits - 1; q >= 0; --q) {
        perm.push_back(static_cast<int>(std::find(result_labels.begin(), result_labels.end(), -(q + 1)) - result_labels.begin()));
    }
    Tensor ordered = result.permute(perm);
    DenseMatrix state = Eigen::Map<const DenseMatrix>(ordered.raw().data(), ordered.size(), 1);
    Real norm = state.norm();
    if (norm > Real(0)) {
        state /= norm;
    }
    return state;
}

std::ostream& operator<<(std::ostream& os, const BPState& state) {
    /*
    Print a summary of the tensor network to an output stream.

    Args:
        os (std::ostream&): The output stream.
        state (BPState&): The state to print.

    Returns:
        std::ostream&: The output stream.
    */
    os << "BPState(nqubits=" << state.get_nqubits() << ", edges=" << state.get_num_edges() << ", max_bond_dimension_used=" << state.get_max_bond_dimension_used() << ", truncation_error=" << state.get_truncation_error() << ", bp_runs=" << state.get_bp_runs() << ")";
    return os;
}

// GCOV_EXCL_BR_STOP
