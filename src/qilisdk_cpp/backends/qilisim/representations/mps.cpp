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
#include <ostream>
#include <stdexcept>
#include <string>

#include "mps.h"

// GCOV_EXCL_BR_START

namespace {

bool is_unitary(const DenseMatrix& u) {
    /*
    Whether a gate matrix is unitary, and so leaves the canonical form of every site
    tensor it is applied to intact.

    Args:
        u (DenseMatrix&): The matrix to check.

    Returns:
        bool: True if u times its adjoint is the identity.
    */
    return DenseMatrix(u * u.adjoint()).isApprox(DenseMatrix::Identity(u.rows(), u.cols()));
}

int swap_pair_index(int index) {
    /*
    Swap the two physical indices packed into one, mapping p + 2 q to q + 2 p.

    Args:
        index (int): The packed pair index.

    Returns:
        int: The same pair with the two halves exchanged.
    */
    return (index >> 1) + 2 * (index & 1);
}

}  // namespace

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

DenseMatrix MPSState::transfer_step(const DenseMatrix& environment, const MPSTensor& site, const DenseMatrix& op) {
    /*
    Run one step of the norm transfer matrix with an operator inserted on the ket layer:

        E -> sum_p A^{p dagger} E (op A)^p

    Here A^p is the site tensor at physical index p, 
    and (op A)^p is the contraction of the operator with the site tensor on the ket layer.

    This is used in the expectation value calculation, where the environment 
    is the left or right environment and the operator is the observable.

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

DenseMatrix MPSState::transfer_step(const DenseMatrix& environment, const MPSTensor& site) {
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
    // Only non-unitary gates can break the canonical form, so only move the centre if necessary
    if (!is_unitary(u)) {
        move_centre(q);
    }
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

Real MPSState::apply_two_site(const DenseMatrix& u, int q, bool keep_centre_left) {
    /*
    Apply a two-qubit matrix to the adjacent pair (q, q + 1), with qubit q the most
    significant index of u. The pair is contracted into one block, the gate applied,
    and the block split again with a truncated SVD.

    Args:
        u (DenseMatrix&): The 4 x 4 matrix to apply.
        q (int): The left qubit of the pair.
        keep_centre_left (bool): Leave the orthogonality centre on q rather than on
            q + 1. Either choice leaves the same state.

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

    // The legs of the block are (left bond, physical q, physical q + 1, right bond)
    int left_bond = sites[q].left();
    int right_bond = sites[q + 1].right();
    Tensor block({left_bond, MPSTensor::PHYSICAL_DIMENSION, MPSTensor::PHYSICAL_DIMENSION, right_bond});
    block.matrix_view(2).noalias() = sites[q].left_fused() * sites[q + 1].right_fused();

    // The block runs qubit q's index fastest and the gate runs it slowest, so relabel the gate rather than move the block
    DenseMatrix relabelled(pair_dimension, pair_dimension);
    for (int row = 0; row < pair_dimension; ++row) {
        for (int column = 0; column < pair_dimension; ++column) {
            relabelled(row, column) = u(swap_pair_index(row), swap_pair_index(column));
        }
    }

    // Each right-bond slice is then a contiguous left x 4 matrix, so the gate goes on in place
    Complex* values = block.raw().data();
    for (int r = 0; r < right_bond; ++r) {
        Eigen::Map<DenseMatrix> slice(values + static_cast<int64_t>(pair_dimension) * left_bond * r, left_bond, pair_dimension);
        slice = slice * relabelled.transpose();
    }
    return split_two_site(q, block, keep_centre_left ? q : q + 1);
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
        return apply_one_site(gate.get_local_matrix(), qubits[0]);
    }
    if (qubits.size() != 2) {
        throw std::invalid_argument("The MPS simulator supports one- and two-qubit gates only, but " + gate.get_id() + " acts on " + std::to_string(qubits.size()) + " qubits");
    }

    // Get the matrix and see which qubit is further down the chain
    DenseMatrix u = gate.get_local_matrix();
    int low = std::min(qubits[0], qubits[1]);
    int high = std::max(qubits[0], qubits[1]);

    // If they happen to be adjacent, just apply the two-site gate directly
    if (high == low + 1) {
        return apply_two_site(u, low);
    }

    // Walk the far qubit down to sit next to the near one, then put it back
    Real error = 0.0;
    for (int site = high - 1; site > low; --site) {
        error += apply_two_site(SWAP, site, true);
    }
    error += apply_two_site(u, low);
    for (int site = low + 1; site < high; ++site) {
        error += apply_two_site(SWAP, site);
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
    if (nqubits > MAX_DENSE_QUBITS) {
        throw std::runtime_error("Cannot materialise the statevector of a " + std::to_string(nqubits) + " qubit MPS since it would need too much memory");
    }
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
