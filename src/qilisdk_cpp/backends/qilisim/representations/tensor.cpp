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
#include <limits>
#include <stdexcept>
#include <string>

#include "tensor.h"

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

std::string shape_text(const std::vector<int>& shape) {
    /*
    Render a shape as a bracketed list, for error messages.

    Args:
        shape (std::vector<int>&): The extent of each leg.

    Returns:
        std::string: The shape, e.g. "[2, 3, 4]".
    */
    std::string text = "[";
    for (size_t leg = 0; leg < shape.size(); ++leg) {
        if (leg > 0) {
            text += ", ";
        }
        text += std::to_string(shape[leg]);
    }
    return text + "]";
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

void orthonormalize(Eigen::Ref<DenseMatrix> columns) {
    /*
    Re-orthonormalise the columns of an already nearly-orthonormal block, in place, by
    modified Gram-Schmidt.

    This is a Gram-Schmidt orthonormalisation of the columns of a matrix, but it
    is done in place and modifies the columns directly. It assumes that the columns are
    already nearly orthonormal, so it does not check for linear dependence or zero-length
    columns. It is used to repair numerical drift in a matrix that should be orthonormal, 
    rather than to orthonormalise an arbitrary set of vectors.

    Args:
        columns (Eigen::Ref<DenseMatrix>): The block to orthonormalise, modified in place.
    */
    for (Eigen::Index i = 0; i < columns.cols(); ++i) {
        for (Eigen::Index j = 0; j < i; ++j) {
            columns.col(i) -= columns.col(j) * columns.col(j).dot(columns.col(i));
        }
        Real length = columns.col(i).norm();
        if (length > Real(0)) {
            columns.col(i) /= length;
        }
    }
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

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be viewed as a matrix with
    shape [6, 35] by passing nrow_legs = 2.

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

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be viewed as a matrix with
    shape [6, 35] by passing nrow_legs = 2.

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

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be viewed as a matrix with
    shape [6, 35] by passing row_legs = [0, 1].

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

    For example, a rank-4 tensor with shape [2, 3, 5, 7] can be contracted against a rank-3 tensor with shape [7, 11, 13]
    by passing legs_a = [3] and legs_b = [0], producing a rank-4 tensor with shape [2, 3, 11, 13].

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

Complex Tensor::contract_all(const Tensor& other) const {
    /*
    Contract every leg of *this against the matching leg of `other`, which is the sum
    over every element of the product of the two tensors.

    Neither side is conjugated, so you need to conjugate the bra 
    side yourself if you want the actual inner product.

    Args:
        other (Tensor&): The tensor to contract against, which must have the same shape.

    Returns:
        Complex: The scalar result.

    Raises:
        std::invalid_argument: If the two shapes differ.
    */
    if (shape != other.shape) {
        throw std::invalid_argument("Cannot contract every leg of a tensor of shape " + shape_text(shape) + " against one of shape " + shape_text(other.shape));
    }
    Complex total(0.0, 0.0);
    for (size_t i = 0; i < data.size(); ++i) {
        total += data[i] * other.data[i];
    }
    return total;
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

    // Check whether we should get singular values from A^H A or A A^H
    std::vector<int> right_legs = remaining_legs(rank(), left_legs);
    DenseMatrix matrix = as_matrix(left_legs);
    const Eigen::Index rows = matrix.rows();
    const Eigen::Index cols = matrix.cols();
    const Eigen::Index available_index = std::min(rows, cols);
    const bool gram_on_the_right = (cols <= rows);
    int available = static_cast<int>(available_index);

    // Get the eigenvalues of the Gram matrix, which are the singular values squared
    DenseMatrix left_factor;
    DenseMatrix right_factor;
    RealVector values(available_index);
    if (gram_on_the_right) {
        Eigen::SelfAdjointEigenSolver<DenseMatrix> gram(DenseMatrix(matrix.adjoint() * matrix));
        right_factor.resize(cols, available_index);
        for (Eigen::Index i = 0; i < available_index; ++i) {
            values(i) = std::sqrt(std::max(Real(0), gram.eigenvalues()(available_index - 1 - i)));
            right_factor.col(i) = gram.eigenvectors().col(available_index - 1 - i);
        }
        left_factor = matrix * right_factor;
    } else {
        Eigen::SelfAdjointEigenSolver<DenseMatrix> gram(DenseMatrix(matrix * matrix.adjoint()));
        left_factor.resize(rows, available_index);
        for (Eigen::Index i = 0; i < available_index; ++i) {
            values(i) = std::sqrt(std::max(Real(0), gram.eigenvalues()(available_index - 1 - i)));
            left_factor.col(i) = gram.eigenvectors().col(available_index - 1 - i);
        }
        right_factor = matrix.adjoint() * left_factor;
    }

    // Pick out the factor that came through the matrix rather than the eigensolver
    DenseMatrix& derived = gram_on_the_right ? left_factor : right_factor;

    // Drop everything below the relative cutoff, then apply the hard cap
    int keep = 0;
    Real resolvable = std::sqrt(std::numeric_limits<Real>::epsilon());
    Real threshold = std::max(cutoff, resolvable) * values(0);
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

    // Turn the kept columns of U S into U, repairing what the Gram matrix cost them
    orthonormalize(derived.leftCols(keep));

    left = from_matrix(DenseMatrix(left_factor.leftCols(keep)), left_shape);
    right = from_matrix(DenseMatrix(right_factor.leftCols(keep).adjoint()), right_shape);
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

// GCOV_EXCL_BR_STOP
