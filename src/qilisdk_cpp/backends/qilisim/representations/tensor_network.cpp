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
#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>

#include "tensor_network.h"

// GCOV_EXCL_BR_START

namespace {

int64_t shape_product(const std::vector<int>& shape) {
    /*
    Total number of elements implied by a shape (1 for the empty shape, i.e. a scalar).

    Args:
        shape (std::vector<int>&): The extent of each leg.

    Returns:
        int64_t: The product of all extents.

    Raises:
        std::invalid_argument: If any extent is not positive.
    */
    int64_t total = 1;
    for (int extent : shape) {
        if (extent <= 0) {
            throw std::invalid_argument("Tensor leg extents must be positive, got " + std::to_string(extent));
        }
        total *= extent;
    }
    return total;
}

std::vector<int> remaining_legs(int rank, const std::vector<int>& legs) {
    /*
    The legs of a rank-`rank` tensor that are not in `legs`, in ascending order.

    Args:
        rank (int): The rank of the tensor.
        legs (std::vector<int>&): The legs to exclude.

    Returns:
        std::vector<int>: The remaining legs.
    */
    std::vector<bool> excluded(rank, false);
    for (int leg : legs) {
        if (leg < 0 || leg >= rank) {
            throw std::out_of_range("Leg " + std::to_string(leg) + " is out of range for a rank-" + std::to_string(rank) + " tensor");
        }
        excluded[leg] = true;
    }
    std::vector<int> rest;
    for (int leg = 0; leg < rank; ++leg) {
        if (!excluded[leg]) {
            rest.push_back(leg);
        }
    }
    return rest;
}

DenseMatrix transfer_step(const DenseMatrix& environment, const MPSTensor& site, const DenseMatrix& op) {
    /*
    Run one step of the norm transfer matrix with an operator inserted on the ket layer:

        E -> sum_p A^{p dagger} E (op A)^p

    Passing the identity gives the plain norm transfer.

    Args:
        environment (DenseMatrix&): The current environment matrix.
        site (MPSTensor&): The MPS tensor for the site to transfer through.
        op (DenseMatrix&): The operator to insert on the ket layer.

    Returns:
        DenseMatrix: The updated environment matrix.
    */
    DenseMatrix next = DenseMatrix::Zero(site.right(), site.right());
    for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
        DenseMatrix ket = DenseMatrix::Zero(site.left(), site.right());
        for (int p_ket = 0; p_ket < MPSTensor::PHYSICAL_DIMENSION; ++p_ket) {
            if (op(p, p_ket) != Complex(0.0, 0.0)) {
                ket += op(p, p_ket) * site.physical_slice(p_ket);
            }
        }
        next += site.physical_slice(p).adjoint() * environment * ket;
    }
    return next;
}

DenseMatrix transfer_step(const DenseMatrix& environment, const MPSTensor& site) {
    /*
    Run one step of the norm transfer matrix with no operator inserted:

        E -> sum_p A^{p dagger} E A^p

    Args:
        environment (DenseMatrix&): The current environment matrix.
        site (MPSTensor&): The MPS tensor for the site to transfer through.

    Returns:
        DenseMatrix: The updated environment matrix.
    */
    return transfer_step(environment, site, DenseMatrix::Identity(MPSTensor::PHYSICAL_DIMENSION, MPSTensor::PHYSICAL_DIMENSION));
}

DenseMatrix pauli_matrix(bool x, bool z) {
    /*
    Select the single-qubit Pauli matrix corresponding to a (x, z) mask pair, 
    following the PauliString convention where both bits set means Y (with no extra phase).

    Args:
        x (bool): Whether to include the X component.
        z (bool): Whether to include the Z component.

    Returns:
        DenseMatrix: The 2x2 Pauli matrix.
    */
    DenseMatrix op = DenseMatrix::Zero(2, 2);
    if (x && z) {
        op(0, 1) = Complex(0.0, -1.0);
        op(1, 0) = Complex(0.0, 1.0);
    } else if (x) {
        op(0, 1) = 1.0;
        op(1, 0) = 1.0;
    } else if (z) {
        op(0, 0) = 1.0;
        op(1, 1) = -1.0;
    } else {
        op(0, 0) = 1.0;
        op(1, 1) = 1.0;
    }
    return op;
}

DenseMatrix swap_matrix() {
    /*
    Two-qubit SWAP gate matrix, which permutes the two qubits.

    Returns:
        DenseMatrix: The 4x4 SWAP matrix.
    */
    DenseMatrix swap = DenseMatrix::Zero(4, 4);
    swap(0, 0) = 1.0;
    swap(1, 2) = 1.0;
    swap(2, 1) = 1.0;
    swap(3, 3) = 1.0;
    return swap;
}

DenseMatrix local_gate_matrix(const Gate& gate) {
    /*
    Build the dense matrix of a gate over just the qubits it acts on, ordered by
    ascending qubit index with the lowest index the most significant.

    Gate stores its matrix in its own qubit order (controls first, then targets),
    so we ask it for the matrix on a register of exactly that many qubits and then
    permute the index bits into ascending order.

    Args:
        gate (Gate&): The gate to build the matrix for.

    Returns:
        DenseMatrix: The 2^k x 2^k matrix over the gate's k qubits.

    Raises:
        std::invalid_argument: If the gate names the same qubit more than once.
    */
    std::vector<int> gate_order = gate.get_qubits();
    int k = static_cast<int>(gate_order.size());

    // Ask Gate for the matrix on a k-qubit register, laid out in its own qubit order
    std::vector<int> controls;
    std::vector<int> targets;
    for (size_t i = 0; i < gate.get_control_qubits().size(); ++i) {
        controls.push_back(static_cast<int>(i));
    }
    for (size_t i = 0; i < gate.get_target_qubits().size(); ++i) {
        targets.push_back(static_cast<int>(controls.size() + i));
    }
    Gate local(gate.get_name(), gate.get_base_matrix(), controls, targets, gate.get_parameters());
    DenseMatrix matrix = DenseMatrix(local.get_full_matrix(k));

    // Where each of the gate's own qubit slots ends up once sorted by qubit index
    std::vector<int> sorted = gate_order;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("Gate " + gate.get_name() + " names the same qubit more than once");
    }
    std::vector<int> destination(k);
    bool already_ascending = true;
    for (int i = 0; i < k; ++i) {
        destination[i] = static_cast<int>(std::lower_bound(sorted.begin(), sorted.end(), gate_order[i]) - sorted.begin());
        already_ascending = already_ascending && destination[i] == i;
    }
    if (already_ascending) {
        return matrix;
    }

    // Rewrite every basis index by moving each qubit's bit to its sorted position
    int dim = 1 << k;
    std::vector<int> reordered_index(dim, 0);
    for (int index = 0; index < dim; ++index) {
        int mapped = 0;
        for (int i = 0; i < k; ++i) {
            int bit = (index >> (k - 1 - i)) & 1;
            mapped |= bit << (k - 1 - destination[i]);
        }
        reordered_index[index] = mapped;
    }
    DenseMatrix reordered(dim, dim);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            reordered(reordered_index[row], reordered_index[col]) = matrix(row, col);
        }
    }
    return reordered;
}

}  // namespace

// Constructors
Tensor::Tensor(const std::vector<int>& shape) : shape(shape), data(static_cast<size_t>(shape_product(shape)), Complex(0.0, 0.0)) {}
Tensor::Tensor(const std::vector<int>& shape, const std::vector<Complex>& values) : shape(shape), data(values) {
    if (static_cast<int64_t>(values.size()) != shape_product(shape)) {
        throw std::invalid_argument("Tensor was given " + std::to_string(values.size()) + " values but its shape holds " + std::to_string(shape_product(shape)));
    }
}

int64_t Tensor::flat_index(const std::vector<int>& index) const {
    /*
    Convert a multi-index into an offset into the column-major buffer.

    Args:
        index (std::vector<int>&): One index per leg.

    Returns:
        int64_t: The offset into the flat buffer.

    Raises:
        std::invalid_argument: If the index does not have one entry per leg.
        std::out_of_range: If any index is outside its leg's extent.
    */
    if (index.size() != shape.size()) {
        throw std::invalid_argument("Tensor index has " + std::to_string(index.size()) + " entries but the tensor has rank " + std::to_string(shape.size()));
    }
    int64_t flat = 0;
    int64_t stride = 1;
    for (size_t leg = 0; leg < shape.size(); ++leg) {
        if (index[leg] < 0 || index[leg] >= shape[leg]) {
            throw std::out_of_range("Tensor index " + std::to_string(index[leg]) + " is out of range for leg " + std::to_string(leg) + " of extent " + std::to_string(shape[leg]));
        }
        flat += stride * index[leg];
        stride *= shape[leg];
    }
    return flat;
}

Tensor Tensor::permute(const std::vector<int>& perm) const {
    /*
    Reorder the legs so that leg i of the result is leg perm[i] of *this. This moves
    data, unlike reshape.

    Args:
        perm (std::vector<int>&): The source leg for each destination leg.

    Returns:
        Tensor: The permuted tensor.

    Raises:
        std::invalid_argument: If perm is not a permutation of every leg.
    */
    if (static_cast<int>(perm.size()) != rank()) {
        throw std::invalid_argument("Permutation has " + std::to_string(perm.size()) + " entries but the tensor has rank " + std::to_string(rank()));
    }
    std::vector<bool> seen(rank(), false);
    for (int leg : perm) {
        if (leg < 0 || leg >= rank()) {
            throw std::invalid_argument("Permutation entry " + std::to_string(leg) + " is out of range for a rank-" + std::to_string(rank()) + " tensor");
        }
        if (seen[leg]) {
            throw std::invalid_argument("Permutation names leg " + std::to_string(leg) + " more than once");
        }
        seen[leg] = true;
    }

    // Create the new empty tensor with the permuted shape
    std::vector<int> new_shape(perm.size());
    for (size_t leg = 0; leg < perm.size(); ++leg) {
        new_shape[leg] = shape[perm[leg]];
    }
    Tensor out(new_shape);

    // Walk the destination buffer in memory order
    std::vector<int64_t> source_stride(shape.size());
    int64_t stride = 1;
    for (size_t leg = 0; leg < shape.size(); ++leg) {
        source_stride[leg] = stride;
        stride *= shape[leg];
    }
    std::vector<int> odometer(perm.size(), 0);
    int64_t source = 0;
    for (int64_t flat = 0; flat < size(); ++flat) {
        out.data[static_cast<size_t>(flat)] = data[static_cast<size_t>(source)];
        for (size_t leg = 0; leg < perm.size(); ++leg) {
            int64_t leg_stride = source_stride[perm[leg]];
            if (++odometer[leg] < new_shape[leg]) {
                source += leg_stride;
                break;
            }
            source -= leg_stride * (new_shape[leg] - 1);
            odometer[leg] = 0;
        }
    }
    return out;
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    /*
    Reinterpret the same buffer under a new shape.
    Only changes the metadata, data isn't moved unlike permute.

    For example, a rank-3 tensor with shape [2, 3, 4] can be reshaped into a rank-2 
    tensor with shape [6, 4] or a rank-1 tensor with shape [24].

    Args:
        new_shape (std::vector<int>&): The new extent of each leg.

    Raises:
        std::invalid_argument: If the new shape holds a different number of elements.
    */
    if (shape_product(new_shape) != size()) {
        throw std::invalid_argument("Cannot reshape a tensor of " + std::to_string(size()) + " elements into a shape holding " + std::to_string(shape_product(new_shape)));
    }
    shape = new_shape;
}

Tensor Tensor::fuse(const std::vector<std::vector<int>>& groups) const {
    /*
    Fuse each group of legs into a single leg, in the order the groups are given.
    Every leg must appear in exactly one group.

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be fused into a rank-2
    tensor with shape [6, 35] by passing groups = [[0, 1], [2, 3]].

    Args:
        groups (std::vector<std::vector<int>>&): The legs making up each fused leg.

    Returns:
        Tensor: The fused tensor, of rank groups.size().
    */
    std::vector<int> perm;
    std::vector<int> new_shape;
    for (const auto& group : groups) {
        int64_t extent = 1;
        for (int leg : group) {
            perm.push_back(leg);
            if (leg < 0 || leg >= rank()) {
                throw std::out_of_range("Leg " + std::to_string(leg) + " is out of range for a rank-" + std::to_string(rank()) + " tensor");
            }
            extent *= shape[leg];
        }
        new_shape.push_back(static_cast<int>(extent));
    }
    Tensor out = permute(perm);
    out.reshape(new_shape);
    return out;
}

Eigen::Map<const DenseMatrix> Tensor::matrix_view(int nrow_legs) const {
    /*
    View the tensor as a matrix whose rows are the first `nrow_legs` legs and whose
    columns are the rest. This doesn't copy anything, as this is just a view of the underlying buffer.

    Args:
        nrow_legs (int): How many leading legs make up the rows.

    Returns:
        Eigen::Map<const DenseMatrix>: The matrix view.

    Raises:
        std::out_of_range: If nrow_legs is not between 0 and the rank.
    */
    if (nrow_legs < 0 || nrow_legs > rank()) {
        throw std::out_of_range("Cannot take " + std::to_string(nrow_legs) + " row legs from a rank-" + std::to_string(rank()) + " tensor");
    }
    int64_t rows = 1;
    for (int leg = 0; leg < nrow_legs; ++leg) {
        rows *= shape[leg];
    }
    return Eigen::Map<const DenseMatrix>(data.data(), rows, size() / rows);
}

Eigen::Map<DenseMatrix> Tensor::matrix_view(int nrow_legs) {
    /*
    View the tensor as a matrix whose rows are the first `nrow_legs` legs and whose
    columns are the rest. This doesn't copy anything, as this is just a view of the underlying buffer.

    Args:
        nrow_legs (int): How many leading legs make up the rows.

    Returns:
        Eigen::Map<DenseMatrix>: The matrix view.
    */
    if (nrow_legs < 0 || nrow_legs > rank()) {
        throw std::out_of_range("Cannot take " + std::to_string(nrow_legs) + " row legs from a rank-" + std::to_string(rank()) + " tensor");
    }
    int64_t rows = 1;
    for (int leg = 0; leg < nrow_legs; ++leg) {
        rows *= shape[leg];
    }
    return Eigen::Map<DenseMatrix>(data.data(), rows, size() / rows);
}

DenseMatrix Tensor::as_matrix(const std::vector<int>& row_legs) const {
    /*
    Fuse `row_legs` into the rows and the remaining legs into the columns. Unlike
    matrix_view this permutes when the row legs are not already leading, so prefer
    matrix_view where the leg order is yours to choose.

    Args:
        row_legs (std::vector<int>&): The legs making up the rows, in order.

    Returns:
        DenseMatrix: The matricised tensor.
    */
    Tensor fused = fuse({row_legs, remaining_legs(rank(), row_legs)});
    return fused.matrix_view(1);
}

Tensor Tensor::from_matrix(const DenseMatrix& m, const std::vector<int>& shape) {
    /*
    Reinterpret a column-major matrix as a tensor of the given shape.

    Args:
        m (DenseMatrix&): The matrix holding the values.
        shape (std::vector<int>&): The shape to give the result.

    Returns:
        Tensor: The tensor.

    Raises:
        std::invalid_argument: If the matrix and the shape hold different element counts.
    */
    if (static_cast<int64_t>(m.size()) != shape_product(shape)) {
        throw std::invalid_argument("Cannot read a tensor holding " + std::to_string(shape_product(shape)) + " elements from a matrix of " + std::to_string(m.size()));
    }
    Tensor out(shape);
    std::copy(m.data(), m.data() + m.size(), out.data.begin());
    return out;
}

Tensor Tensor::contract(const Tensor& other, const std::vector<int>& legs_a, const std::vector<int>& legs_b) const {
    /*
    Contract `legs_a` of *this against `legs_b` of `other`. The surviving legs of
    *this come first, then those of `other`, both in ascending order.

    Both operands are permuted so the contracted legs are adjacent and then fed to a
    single gemm, so the cost is one matrix product plus two data movements.

    Args:
        other (Tensor&): The tensor to contract against.
        legs_a (std::vector<int>&): The legs of *this to contract.
        legs_b (std::vector<int>&): The matching legs of `other`.

    Returns:
        Tensor: The contracted tensor.

    Raises:
        std::invalid_argument: If the leg lists differ in length or the extents disagree.
    */
    if (legs_a.size() != legs_b.size()) {
        throw std::invalid_argument("Contraction was given " + std::to_string(legs_a.size()) + " legs on one side and " + std::to_string(legs_b.size()) + " on the other");
    }
    std::vector<int> keep_a = remaining_legs(rank(), legs_a);
    std::vector<int> keep_b = remaining_legs(other.rank(), legs_b);
    for (size_t i = 0; i < legs_a.size(); ++i) {
        if (shape[legs_a[i]] != other.shape[legs_b[i]]) {
            throw std::invalid_argument("Cannot contract leg " + std::to_string(legs_a[i]) + " of extent " + std::to_string(shape[legs_a[i]]) + " against leg " + std::to_string(legs_b[i]) + " of extent " + std::to_string(other.shape[legs_b[i]]));
        }
    }

    // Permute both tensors so the contracted legs are adjacent
    std::vector<int> perm_a = keep_a;
    perm_a.insert(perm_a.end(), legs_a.begin(), legs_a.end());
    std::vector<int> perm_b = legs_b;
    perm_b.insert(perm_b.end(), keep_b.begin(), keep_b.end());
    Tensor a = permute(perm_a);
    Tensor b = other.permute(perm_b);
    
    // Do the contraction as a single matrix product
    DenseMatrix product = a.matrix_view(static_cast<int>(keep_a.size())) * b.matrix_view(static_cast<int>(legs_b.size()));

    // Reshape the result
    std::vector<int> out_shape;
    for (int leg : keep_a) {
        out_shape.push_back(shape[leg]);
    }
    for (int leg : keep_b) {
        out_shape.push_back(other.shape[leg]);
    }
    return from_matrix(product, out_shape);

}

void Tensor::split(const std::vector<int>& left_legs, int max_bond_dimension, Real cutoff, Tensor& left, RealVector& singular_values, Tensor& right, Real* truncation_error) const {
    /*
    Split the tensor across a new bond with a truncated SVD. `left` gets `left_legs`
    plus the new bond as its last leg, `right` gets the new bond as its first leg plus
    the remaining legs, and the singular values are handed back separately so the
    caller can decide which side absorbs them.

    Singular values below `cutoff` times the largest are dropped, then at most
    `max_bond_dimension` are kept (pass 0 or less for no cap).

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be split into a rank-3
    tensor with shape [2, 3, k] and a rank-3 tensor with shape [k, 5, 7] for some k <= 5.

    Args:
        left_legs (std::vector<int>&): The legs that end up on the left of the bond.
        max_bond_dimension (int): The largest bond to keep, or 0 for no limit.
        cutoff (Real): Relative threshold below which singular values are discarded.
        left (Tensor&): Output, the left factor.
        singular_values (RealVector&): Output, the singular values kept.
        right (Tensor&): Output, the right factor.
        truncation_error (Real*): Output if not null, the discarded weight as a
            fraction of the total sum of squared singular values.
    */
    
    // Do the SVD
    std::vector<int> right_legs = remaining_legs(rank(), left_legs);
    DenseMatrix matrix = as_matrix(left_legs);
    Eigen::BDCSVD<DenseMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(matrix);
    const RealVector& values = svd.singularValues();
    int available = static_cast<int>(values.size());

    // Drop everything below the relative cutoff, then apply the hard cap
    int keep = 0;
    Real threshold = cutoff * values(0);
    while (keep < available && values(keep) > threshold) {
        ++keep;
    }
    if (max_bond_dimension > 0 && keep > max_bond_dimension) {
        keep = max_bond_dimension;
    }
    if (keep == 0) {
        keep = 1;
    }

    // Keep track of the total truncation error
    if (truncation_error != nullptr) {
        Real total_weight = values.squaredNorm();
        Real kept_weight = values.head(keep).squaredNorm();
        *truncation_error = (total_weight > Real(0)) ? std::max(Real(0), (total_weight - kept_weight) / total_weight) : Real(0);
    }
    singular_values = values.head(keep);

    // Reshape everything
    std::vector<int> left_shape;
    for (int leg : left_legs) {
        left_shape.push_back(shape[leg]);
    }
    left_shape.push_back(keep);
    std::vector<int> right_shape;
    right_shape.push_back(keep);
    for (int leg : right_legs) {
        right_shape.push_back(shape[leg]);
    }
    left = from_matrix(DenseMatrix(svd.matrixU().leftCols(keep)), left_shape);
    right = from_matrix(DenseMatrix(svd.matrixV().leftCols(keep).adjoint()), right_shape);
}

Tensor Tensor::conjugate() const {
    /*
    Return a new tensor with the same shape and the complex conjugate of every value.

    Returns:
        Tensor: The conjugated tensor.
    */
    Tensor out = *this;
    for (Complex& value : out.data) {
        value = std::conj(value);
    }
    return out;
}

Real Tensor::norm() const {
    /*
    Return the Frobenius norm of the tensor.

    Returns:
        Real: The Frobenius norm.
    */
    Real total = 0.0;
    for (const Complex& value : data) {
        total += std::norm(value);
    }
    return std::sqrt(total);
}

void Tensor::scale(Complex factor) {
    /*
    Scale every element of the tensor by a complex factor.

    Args:
        factor (Complex): The scaling factor.
    */
    for (Complex& value : data) {
        value *= factor;
    }
}

void Tensor::set_zero() {
    /*
    Set every element of the tensor to zero.
    */
    std::fill(data.begin(), data.end(), Complex(0.0, 0.0));
}

bool Tensor::has_nan() const {
    /*
    Check if the tensor contains any NaN values.

    Returns:
        bool: True if any element is NaN, False otherwise.
    */
    for (const Complex& value : data) {
        if (std::isnan(value.real()) || std::isnan(value.imag())) {
            return true;
        }
    }
    return false;
}

// Constructors
MPSTensor::MPSTensor(int left, int right) : Tensor({left, PHYSICAL_DIMENSION, right}) {}
MPSTensor::MPSTensor(int left, int right, const std::vector<Complex>& values) : Tensor({left, PHYSICAL_DIMENSION, right}, values) {}
MPSTensor::MPSTensor(const Tensor& t) : Tensor(t) {
    /*
    Adopt the result of a generic tensor operation as a site tensor.

    Args:
        t (Tensor&): The tensor to adopt.

    Raises:
        std::invalid_argument: If it is not rank 3 with a physical leg in the middle.
    */
    if (rank() != 3 || extent(1) != PHYSICAL_DIMENSION) {
        throw std::invalid_argument("An MPS site tensor must be rank 3 with a physical leg of extent " + std::to_string(PHYSICAL_DIMENSION) + " in the middle");
    }
}

Eigen::Map<const DenseMatrix, 0, Eigen::OuterStride<>> MPSTensor::physical_slice(int p) const {
    /*
    The left x right matrix at one physical index.

    For example, say we have a rank-3 tensor with shape [2, 3, 4]. Then physical_slice(1) is the 2x4 matrix

        [[A_{0,1,0}, A_{0,1,1}, A_{0,1,2}, A_{0,1,3}],
         [A_{1,1,0}, A_{1,1,1}, A_{1,1,2}, A_{1,1,3}]]

    where the first index is the left leg and the second is the right leg.

    Basically it's the view of the tensor where we fix the middle index to p, and the other two indices are free.

    Args:
        p (int): The physical index.

    Returns:
        Eigen::Map: The strided view of that slice.

    Raises:
        std::out_of_range: If p is not a valid physical index.
    */
    if (p < 0 || p >= PHYSICAL_DIMENSION) {
        throw std::out_of_range("Physical index " + std::to_string(p) + " is out of range for an MPS site tensor");
    }
    return Eigen::Map<const DenseMatrix, 0, Eigen::OuterStride<>>(data.data() + static_cast<int64_t>(p) * shape[0], shape[0], shape[2], Eigen::OuterStride<>(shape[0] * PHYSICAL_DIMENSION));
}

// Constructors
MPSState::MPSState(int nqubits) : MPSState(nqubits, std::string(nqubits > 0 ? nqubits : 0, '0')) {}
MPSState::MPSState(int nqubits, const std::string& b) : nqubits(nqubits) {
    /*
    Build the product state |b>, every bond dimension 1.

    Args:
        nqubits (int): The number of qubits.
        b (std::string&): The bitstring, one character per qubit, qubit 0 first.

    Raises:
        std::invalid_argument: If nqubits is not positive, or b does not match it.
    */
    if (nqubits <= 0) {
        throw std::invalid_argument("An MPS needs at least one qubit, got " + std::to_string(nqubits));
    }
    if (static_cast<int>(b.size()) != nqubits) {
        throw std::invalid_argument("Bitstring '" + b + "' has " + std::to_string(b.size()) + " characters but the state has " + std::to_string(nqubits) + " qubits");
    }
    sites.reserve(nqubits);
    for (int q = 0; q < nqubits; ++q) {
        if (b[q] != '0' && b[q] != '1') {
            throw std::invalid_argument("Bitstring '" + b + "' must contain only '0' and '1'");
        }
        MPSTensor site(1, 1);
        site(0, b[q] == '1' ? 1 : 0, 0) = 1.0;
        sites.push_back(site);
    }
}

int MPSState::get_bond_dimension(int bond) const {
    /*
    The dimension of the bond joining qubits `bond` and `bond` + 1.

    Args:
        bond (int): The bond index.

    Returns:
        int: The bond dimension.

    Raises:
        std::out_of_range: If there is no such bond.
    */
    if (bond < 0 || bond >= nqubits - 1) {
        throw std::out_of_range("Bond " + std::to_string(bond) + " is out of range for a " + std::to_string(nqubits) + " qubit MPS");
    }
    return sites[bond].right();
}

int MPSState::get_max_bond_dimension_used() const {
    /*
    Get the maximum bond dimension used in the MPS.

    Returns:
        int: The largest bond dimension.
    */
    int largest = 1;
    for (int bond = 0; bond < nqubits - 1; ++bond) {
        largest = std::max(largest, sites[bond].right());
    }
    return largest;
}

void MPSState::move_centre(int q) {
    /*
    Move the orthogonality centre to qubit `q`, QR sweeping one site at a time. Exact:
    the state is unchanged, only its gauge.

    For example, if the centre is on qubit 0 and we move it to qubit 2, we do

        A_0 A_1 A_2 -> Q R A_2 -> Q (R A_2) = A'_0 A'_1 A'_2

    where the new site tensors are

        A'_0 = Q
        A'_1 = R A_2
        A'_2 = I

    Args:
        q (int): The qubit to centre on.

    Raises:
        std::out_of_range: If q is not a valid qubit index.
    */
    if (q < 0 || q >= nqubits) {
        throw std::out_of_range("Cannot centre a " + std::to_string(nqubits) + " qubit MPS on qubit " + std::to_string(q));
    }

    // Sweeping right: A = Q R, keep Q here and push R into the next site
    while (centre < q) {
        int left = sites[centre].left();
        DenseMatrix matrix = sites[centre].left_fused();
        int bond = static_cast<int>(std::min(matrix.rows(), matrix.cols()));
        Eigen::HouseholderQR<DenseMatrix> qr(matrix);
        DenseMatrix q_factor = qr.householderQ() * DenseMatrix::Identity(matrix.rows(), bond);
        DenseMatrix r_factor = qr.matrixQR().topRows(bond).triangularView<Eigen::Upper>();
        int next_right = sites[centre + 1].right();
        sites[centre] = MPSTensor(Tensor::from_matrix(q_factor, {left, MPSTensor::PHYSICAL_DIMENSION, bond}));
        sites[centre + 1] = MPSTensor(Tensor::from_matrix(DenseMatrix(r_factor * sites[centre + 1].right_fused()), {bond, MPSTensor::PHYSICAL_DIMENSION, next_right}));
        ++centre;
    }

    // Sweeping left: the same QR on the transpose, which is an LQ decomposition
    while (centre > q) {
        int right = sites[centre].right();
        DenseMatrix matrix = sites[centre].right_fused().adjoint();
        int bond = static_cast<int>(std::min(matrix.rows(), matrix.cols()));
        Eigen::HouseholderQR<DenseMatrix> qr(matrix);
        DenseMatrix q_factor = qr.householderQ() * DenseMatrix::Identity(matrix.rows(), bond);
        DenseMatrix r_factor = qr.matrixQR().topRows(bond).triangularView<Eigen::Upper>();
        int previous_left = sites[centre - 1].left();
        sites[centre] = MPSTensor(Tensor::from_matrix(DenseMatrix(q_factor.adjoint()), {bond, MPSTensor::PHYSICAL_DIMENSION, right}));
        sites[centre - 1] = MPSTensor(Tensor::from_matrix(DenseMatrix(sites[centre - 1].left_fused() * r_factor.adjoint()), {previous_left, MPSTensor::PHYSICAL_DIMENSION, bond}));
        --centre;
    }
}

Real MPSState::apply_one_site(const DenseMatrix& u, int q) {
    /*
    Apply a single-qubit matrix to qubit `q`. Exact, so no truncation error, but the
    centre is moved onto `q` first so that a non-unitary u cannot break the canonical
    form of the sites around it.

    Args:
        u (DenseMatrix&): The 2 x 2 matrix to apply.
        q (int): The qubit to apply it to.

    Returns:
        Real: The truncation error, which is always 0.0 for a single-qubit gate.

    Raises:
        std::invalid_argument: If u is not 2 x 2.
        std::out_of_range: If q is not a valid qubit index.
    */
    if (u.rows() != MPSTensor::PHYSICAL_DIMENSION || u.cols() != MPSTensor::PHYSICAL_DIMENSION) {
        throw std::invalid_argument("A single-qubit gate matrix must be 2 x 2");
    }
    if (q < 0 || q >= nqubits) {
        throw std::out_of_range("Qubit " + std::to_string(q) + " is out of range for a " + std::to_string(nqubits) + " qubit MPS");
    }
    move_centre(q);
    MPSTensor& site = sites[q];
    DenseMatrix mixed_zero = u(0, 0) * site.physical_slice(0) + u(0, 1) * site.physical_slice(1);
    DenseMatrix mixed_one = u(1, 0) * site.physical_slice(0) + u(1, 1) * site.physical_slice(1);
    for (int r = 0; r < site.right(); ++r) {
        for (int l = 0; l < site.left(); ++l) {
            site(l, 0, r) = mixed_zero(l, r);
            site(l, 1, r) = mixed_one(l, r);
        }
    }
    return 0.0;
}

Real MPSState::apply_two_site(const DenseMatrix& u, int q) {
    /*
    Apply a two-qubit matrix to the adjacent pair (q, q + 1), with qubit q the most
    significant index of u. The pair is contracted into one block, the gate applied,
    and the block split again with a truncated SVD.

    Args:
        u (DenseMatrix&): The 4 x 4 matrix to apply.
        q (int): The left qubit of the pair.

    Returns:
        Real: The discarded weight from the SVD.

    Raises:
        std::invalid_argument: If u is not 4 x 4.
        std::out_of_range: If (q, q + 1) is not a valid adjacent pair.
    */
    int pair_dimension = MPSTensor::PHYSICAL_DIMENSION * MPSTensor::PHYSICAL_DIMENSION;
    if (u.rows() != pair_dimension || u.cols() != pair_dimension) {
        throw std::invalid_argument("A two-qubit gate matrix must be 4 x 4");
    }
    if (q < 0 || q + 1 >= nqubits) {
        throw std::out_of_range("Qubits (" + std::to_string(q) + ", " + std::to_string(q + 1) + ") are out of range for a " + std::to_string(nqubits) + " qubit MPS");
    }
    move_centre(q);

    // Legs of the block: (left bond, physical q, physical q + 1, right bond)
    Tensor block = sites[q].contract(sites[q + 1], {2}, {0});

    // Need to permute the legs to match the order of the gate matrix
    Tensor fused = block.permute({2, 1, 0, 3});
    Tensor applied = Tensor::from_matrix(DenseMatrix(u * fused.matrix_view(2)), fused.get_shape());
    return split_two_site(q, applied.permute({2, 1, 0, 3}), q + 1);
}

Real MPSState::split_two_site(int q, const Tensor& theta, int keep_centre_on) {
    /*
    Split a two-site block back into sites q and q + 1.

    Args:
        q (int): The left qubit of the pair.
        theta (Tensor&): The block, with legs (left bond, physical q, physical q + 1, right bond).
        keep_centre_on (int): Which of the two qubits should end up as the orthogonality
            centre, and so absorb the singular values.

    Returns:
        Real: The discarded weight from the SVD.
    */
    Tensor left;
    Tensor right;
    RealVector singular_values;
    Real error = 0.0;
    theta.split({0, 1}, max_bond_dimension, truncation_cutoff, left, singular_values, right, &error);

    // Whichever site becomes the centre takes the singular values with it
    int bond = static_cast<int>(singular_values.size());
    if (keep_centre_on == q) {
        Eigen::Map<DenseMatrix> matrix = left.matrix_view(2);
        for (int b = 0; b < bond; ++b) {
            matrix.col(b) *= singular_values(b);
        }
        centre = q;
    } else {
        Eigen::Map<DenseMatrix> matrix = right.matrix_view(1);
        for (int b = 0; b < bond; ++b) {
            matrix.row(b) *= singular_values(b);
        }
        centre = q + 1;
    }
    sites[q] = MPSTensor(left);
    sites[q + 1] = MPSTensor(right);
    total_truncation_error += error;
    return error;
}

Real MPSState::apply_gate(const Gate& gate) {
    /*
    Apply a gate to the state. One- and two-qubit gates are supported. 
    A two-qubit gate on non-adjacent qubits is routed by swapping down the chain,
    applying the gate, and swapping back.

    Args:
        gate (Gate&): The gate to apply.

    Returns:
        Real: The truncation error incurred, summed over every SVD involved.

    Raises:
        std::invalid_argument: If the gate acts on more than two qubits.
        std::out_of_range: If the gate names a qubit outside the register.
    */
    std::vector<int> qubits = gate.get_qubits();
    for (int q : qubits) {
        if (q < 0 || q >= nqubits) {
            throw std::out_of_range("Gate " + gate.get_name() + " acts on qubit " + std::to_string(q) + ", outside a " + std::to_string(nqubits) + " qubit MPS");
        }
    }
    if (qubits.size() == 1) {
        return apply_one_site(local_gate_matrix(gate), qubits[0]);
    }
    if (qubits.size() != 2) {
        throw std::invalid_argument("The MPS simulator supports one- and two-qubit gates only, but " + gate.get_id() + " acts on " + std::to_string(qubits.size()) + " qubits");
    }

    // Get the matrix and see which qubit is further down the chain
    DenseMatrix u = local_gate_matrix(gate);
    int low = std::min(qubits[0], qubits[1]);
    int high = std::max(qubits[0], qubits[1]);
    
    // If they happen to be adjacent, just apply the two-site gate directly
    if (high == low + 1) {
        return apply_two_site(u, low);
    }

    // Walk the far qubit down to sit next to the near one, then put it back
    Real error = 0.0;
    for (int site = high - 1; site > low; --site) {
        error += apply_two_site(swap_matrix(), site);
    }
    error += apply_two_site(u, low);
    for (int site = low + 1; site < high; ++site) {
        error += apply_two_site(swap_matrix(), site);
    }
    return error;
}

void MPSState::normalize() {
    /*
    Scale the state to unit norm. Scaling one site tensor scales the whole state, so
    this holds whatever gauge the state is in.

    Raises:
        std::runtime_error: If the state has zero norm and cannot be normalized.
    */
    Real current = norm();
    if (!(current > Real(0))) {
        throw std::runtime_error("Cannot normalize an MPS whose norm is zero");
    }
    sites[centre].scale(Complex(1.0 / current, 0.0));
}

Real MPSState::norm() const {
    /*
    The Frobenius norm of the state, sqrt(<psi|psi>).

    Returns:
        Real: The norm.
    */
    return std::sqrt(std::max(Real(0), overlap(*this).real()));
}

Complex MPSState::overlap(const MPSState& other) const {
    /*
    The exact overlap <this|other>, by sweeping the transfer matrix along the chain.

    Basically we start with a 1x1 environment, then sweep it along the chain, 
    multiplying by each transfer matrix in turn. The final environment is the overlap.

    The transfer matrix at each site is

        T_q = sum_{p_q} A_q^* B_q

    Args:
        other (MPSState&): The state to overlap with.

    Returns:
        Complex: The overlap.

    Raises:
        std::invalid_argument: If the two states have different qubit counts.
    */
    if (other.nqubits != nqubits) {
        throw std::invalid_argument("Cannot overlap a " + std::to_string(nqubits) + " qubit MPS with a " + std::to_string(other.nqubits) + " qubit one");
    }
    DenseMatrix environment = DenseMatrix::Ones(1, 1);
    for (int q = 0; q < nqubits; ++q) {
        const MPSTensor& bra = sites[q];
        const MPSTensor& ket = other.sites[q];
        DenseMatrix next = DenseMatrix::Zero(bra.right(), ket.right());
        for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
            next += bra.physical_slice(p).adjoint() * environment * ket.physical_slice(p);
        }
        environment = next;
    }
    return environment(0, 0);
}

Complex MPSState::amplitude(const std::string& b) const {
    /*
    The amplitude <b|psi>, the product of one physical slice per site.

    It does this by doing the same thing as overlap(), but for the fixed bitstring b instead of another MPS.

    Args:
        b (std::string&): The bitstring, one character per qubit, qubit 0 first.

    Returns:
        Complex: The amplitude.

    Raises:
        std::invalid_argument: If b does not match the register, or is not binary.
    */
    if (static_cast<int>(b.size()) != nqubits) {
        throw std::invalid_argument("Bitstring '" + b + "' has " + std::to_string(b.size()) + " characters but the state has " + std::to_string(nqubits) + " qubits");
    }
    DenseMatrix product = DenseMatrix::Ones(1, 1);
    for (int q = 0; q < nqubits; ++q) {
        if (b[q] != '0' && b[q] != '1') {
            throw std::invalid_argument("Bitstring '" + b + "' must contain only '0' and '1'");
        }
        DenseMatrix next = product * sites[q].physical_slice(b[q] == '1' ? 1 : 0);
        product = next;
    }
    return product(0, 0);
}

Complex MPSState::expectation_value(const DenseMatrix& observable, const std::vector<int>& qubits) const {
    /*
    The expectation value <psi|O|psi> / <psi|psi> of an observable acting on a
    contiguous block of qubits. The block is contracted into one tensor, the
    observable applied to its physical legs, and the rest of the chain closed with
    plain transfer steps, so no gauge assumption is made.

    Args:
        observable (DenseMatrix&): The 2^k x 2^k matrix, with qubits[0] the most
            significant index.
        qubits (std::vector<int>&): The k qubits it acts on, contiguous and ascending.

    Returns:
        Complex: The normalized expectation value.

    Raises:
        std::invalid_argument: If the qubits are not contiguous and ascending, or the
            observable does not match them.
        std::out_of_range: If the qubits are outside the register.
    */
    if (qubits.empty()) {
        throw std::invalid_argument("An observable must act on at least one qubit");
    }
    for (size_t i = 0; i < qubits.size(); ++i) {
        if (qubits[i] < 0 || qubits[i] >= nqubits) {
            throw std::out_of_range("Qubit " + std::to_string(qubits[i]) + " is out of range for a " + std::to_string(nqubits) + " qubit MPS");
        }
        if (i > 0 && qubits[i] != qubits[i - 1] + 1) {
            throw std::invalid_argument("The MPS expectation value needs a contiguous ascending block of qubits");
        }
    }
    int k = static_cast<int>(qubits.size());
    int dimension = 1 << k;
    if (observable.rows() != dimension || observable.cols() != dimension) {
        throw std::invalid_argument("An observable on " + std::to_string(k) + " qubits must be " + std::to_string(dimension) + " x " + std::to_string(dimension));
    }

    // Sweep the left environment up to the first qubit
    int first = qubits.front();
    int last = qubits.back();
    DenseMatrix environment = DenseMatrix::Ones(1, 1);
    for (int q = 0; q < first; ++q) {
        environment = transfer_step(environment, sites[q]);
    }

    // Contract the block into one tensor
    Tensor block = sites[first];
    for (int q = first + 1; q <= last; ++q) {
        block = block.contract(sites[q], {block.rank() - 1}, {0});
    }

    // Reorder to match the observable, apply it, then reorder back
    std::vector<int> perm;
    for (int leg = k; leg >= 1; --leg) {
        perm.push_back(leg);
    }
    perm.push_back(0);
    perm.push_back(k + 1);
    Tensor fused = block.permute(perm);
    Tensor applied_fused = Tensor::from_matrix(DenseMatrix(observable * fused.matrix_view(k)), fused.get_shape());
    std::vector<int> inverse(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inverse[perm[i]] = static_cast<int>(i);
    }
    Tensor applied = applied_fused.permute(inverse);

    // Close the block against the bra layer through the left environment
    std::vector<int> closed_legs;
    for (int leg = 0; leg <= k; ++leg) {
        closed_legs.push_back(leg);
    }
    Tensor left_environment = Tensor::from_matrix(environment, {static_cast<int>(environment.rows()), static_cast<int>(environment.cols())});
    Tensor half = left_environment.contract(applied, {1}, {0});
    Tensor closed = block.conjugate().contract(half, closed_legs, closed_legs);
    environment = closed.matrix_view(1);

    // Sweep the right environment down to the last qubit
    for (int q = last + 1; q < nqubits; ++q) {
        environment = transfer_step(environment, sites[q]);
    }

    // The final environment is the expectation value, but we need to normalize it by the norm of the state
    Complex denominator = overlap(*this);
    if (denominator == Complex(0.0, 0.0)) {
        throw std::runtime_error("Cannot take an expectation value of an MPS whose norm is zero");
    }
    return environment(0, 0) / denominator;

}

Real MPSState::expectation_value(const MatrixFreeHamiltonian& H) const {
    /*
    The expectation value <psi|H|psi> / <psi|psi>, summed over the Pauli strings of H.
    Each string costs one sweep along the chain with its local Pauli applied to the
    ket layer, so the cost is linear in the number of terms and in the qubit count.

    Args:
        H (MatrixFreeHamiltonian&): The Hamiltonian.

    Returns:
        Real: The normalized expectation value.

    Raises:
        std::invalid_argument: If the Hamiltonian is on a different number of qubits.
        std::runtime_error: If the state has zero norm.
    */
    if (H.get_nqubits() != nqubits) {
        throw std::invalid_argument("Cannot take the expectation value of a " + std::to_string(H.get_nqubits()) + " qubit Hamiltonian on a " + std::to_string(nqubits) + " qubit MPS");
    }
    Real denominator = std::real(overlap(*this));
    if (!(denominator > Real(0))) {
        throw std::runtime_error("Cannot take an expectation value of an MPS whose norm is zero");
    }

    Real total = 0.0;
    for (const auto& term : H.get_operators()) {
        const PauliString& pauli = term.first;
        DenseMatrix environment = DenseMatrix::Ones(1, 1);
        for (int q = 0; q < nqubits; ++q) {
            environment = transfer_step(environment, sites[q], pauli_matrix(pauli.x_mask[q], pauli.z_mask[q]));
        }
        total += std::real(term.second * environment(0, 0));
    }
    return total / denominator;
}

std::string MPSState::sample() const {
    /*
    Overload of sample() that uses the internal random engine.

    Returns:
        std::string: The outcome, one character per qubit, qubit 0 first.
    */
    return sample(rng);
}

std::string MPSState::sample(std::mt19937_64& engine) const {
    /*
    Draw one shot by sampling each qubit in turn from its exact conditional marginal.
    Costs one right-to-left sweep to build the environments plus one left-to-right
    sweep to sample, and is exact regardless of gauge.

    Args:
        engine (std::mt19937_64&): The random engine to draw from.

    Returns:
        std::string: The outcome, one character per qubit, qubit 0 first.

    Raises:
        std::runtime_error: If the state has zero norm.
    */
    
    // Build the right environments for each site
    std::vector<DenseMatrix> right_environment(nqubits + 1);
    right_environment[nqubits] = DenseMatrix::Ones(1, 1);
    for (int q = nqubits - 1; q >= 0; --q) {
        const MPSTensor& site = sites[q];
        DenseMatrix next = DenseMatrix::Zero(site.left(), site.left());
        for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
            next += site.physical_slice(p) * right_environment[q + 1] * site.physical_slice(p).adjoint();
        }
        right_environment[q] = next;
    }

    // Build the left environment and sample each qubit in turn
    std::string outcome(nqubits, '0');
    DenseMatrix left_environment = DenseMatrix::Ones(1, 1);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    for (int q = 0; q < nqubits; ++q) {
        const MPSTensor& site = sites[q];
        
        // Get the proability of each outcome, this is kind of like a partial trace
        std::array<DenseMatrix, MPSTensor::PHYSICAL_DIMENSION> extended;
        std::array<Real, MPSTensor::PHYSICAL_DIMENSION> weight;
        for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
            extended[p] = site.physical_slice(p).adjoint() * left_environment * site.physical_slice(p);
            weight[p] = std::max(Real(0), (extended[p] * right_environment[q + 1]).trace().real());
        }
        
        // Get the norm of the conditional distribution
        Real total = weight[0] + weight[1];
        if (!(total > Real(0))) {
            throw std::runtime_error("Cannot sample an MPS whose norm is zero");
        }

        // Sample from these weights
        int outcome_bit = (uniform(engine) * total < weight[0]) ? 0 : 1;
        outcome[q] = outcome_bit == 1 ? '1' : '0';

        // Divide out the conditional probability so the next weights are conditionals too
        left_environment = extended[outcome_bit] / weight[outcome_bit];
    }
    return outcome;
}

std::map<std::string, int> MPSState::sample(int nshots) const {
    /*
    Draw `nshots` shots and count the outcomes.

    Args:
        nshots (int): The number of shots.

    Returns:
        std::map<std::string, int>: Outcome counts.

    Raises:
        std::invalid_argument: If nshots is not positive.
    */
    if (nshots <= 0) {
        throw std::invalid_argument("Number of shots must be positive, got " + std::to_string(nshots));
    }
    std::map<std::string, int> counts;
    for (int shot = 0; shot < nshots; ++shot) {
        counts[sample(rng)] += 1;
    }
    return counts;
}

DenseMatrix MPSState::as_dense() const {
    /*
    Contract the whole chain into a statevector.

    Note that this should not be done for big systems, as the statevector can be huge whilst the MPS tiny.

    Returns:
        DenseMatrix: The 2^n x 1 statevector, qubit 0 the most significant bit.

    Raises:
        std::runtime_error: If the register is too large to materialise.
    */
    DenseMatrix block = DenseMatrix::Ones(1, 1);
    for (int q = 0; q < nqubits; ++q) {
        int rows = static_cast<int>(block.rows());
        DenseMatrix next(rows * MPSTensor::PHYSICAL_DIMENSION, sites[q].right());
        for (int p = 0; p < MPSTensor::PHYSICAL_DIMENSION; ++p) {
            DenseMatrix contribution = block * sites[q].physical_slice(p);
            for (int row = 0; row < rows; ++row) {
                next.row(row * MPSTensor::PHYSICAL_DIMENSION + p) = contribution.row(row);
            }
        }
        block = next;
    }
    return block;
}

std::ostream& operator<<(std::ostream& os, const MPSState& state) {  
    /*
    Print a summary of the MPS state to an output stream.

    Args:
        os (std::ostream&): The output stream.
        state (MPSState&): The MPS state to print.

    Returns:
        std::ostream&: The output stream.
    */
    os << "MPSState(nqubits=" << state.get_nqubits() << ", bond_dimensions=[";
    for (int bond = 0; bond + 1 < state.get_nqubits(); ++bond) {
        if (bond > 0) {
            os << ", ";
        }
        os << state.get_bond_dimension(bond);
    }
    os << "], truncation_error=" << state.get_truncation_error() << ")";
    return os;
}

// GCOV_EXCL_BR_STOP
