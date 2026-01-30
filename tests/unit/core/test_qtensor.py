# Copyright 2025 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from scipy.sparse import csc_array, csr_matrix, issparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import norm as scipy_norm

import qilisdk
from qilisdk.core.qtensor import QTensor, basis_state, bra, expect_val, ket, tensor_prod

# --- Constructor Tests ---


def test_constructor_valid_ndarray():
    """QTensor should accept a valid NumPy array and convert it to sparse."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    assert issparse(qobj.data)


def test_constructor_valid_sparse():
    """QTensor should accept a valid SciPy sparse matrix."""
    sparse_mat = csc_array([[1, 0], [0, 1]])
    qobj = QTensor(sparse_mat)
    # Should be stored as a CSR matrix.
    assert qobj.data.format == "csr"


@pytest.mark.parametrize("invalid_input", [1, "string", [1, 2, 3]])
def test_constructor_invalid_input(invalid_input):
    """QTensor should raise ValueError for inputs that are not arrays or sparse matrices."""
    with pytest.raises(ValueError):  # noqa: PT011
        QTensor(invalid_input)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3),  # Row vector with 3 columns (3 is odd and !=1)
        (3, 1),  # Column vector with 3 rows (3 is odd and !=1)
        (3, 3),  # Square matrix of odd size > 1
        (2, 3),  # Non-square matrix (neither row nor column vector)
        (3, 2),  # Non-square matrix (neither row nor column vector)
    ],
)
def test_constructor_invalid_shape(shape):
    """QTensor should raise ValueError for arrays with invalid shapes."""
    arr = np.zeros(shape)
    with pytest.raises(ValueError):  # noqa: PT011
        QTensor(arr)


# --- Property Tests ---


@pytest.mark.parametrize(
    ("array", "expected_nqubits"),
    [
        (np.eye(1), 0),  # 1x1 matrix -> scalar (log2(1)==0)
        (np.eye(2), 1),  # 2x2 matrix -> 1 qubit
        (np.eye(4), 2),  # 4x4 matrix -> 2 qubits
        (np.array([[1, 0]]), 1),  # Row vector: (1,2) -> 1 qubit
        (np.array([[1], [0]]), 1),  # Column vector: (2,1) -> 1 qubit
    ],
)
def test_nqubits(array, expected_nqubits):
    """Test the nqubits property for various valid input shapes."""
    qobj = QTensor(array)
    assert qobj.nqubits == expected_nqubits


def test_dense_method():
    """The dense method should return a NumPy array equivalent to the original data."""
    arr = np.array([[1, 2], [3, 4]])
    qobj = QTensor(arr)
    np.testing.assert_array_equal(qobj.dense(), arr)


# --- Method Tests ---


def test_dag():
    """Test that the dagger (adjoint) method returns the conjugate transpose."""
    arr = np.array([[1 + 2j, 2], [3, 4 + 5j]])
    qobj = QTensor(arr)
    dagger_qobj = qobj.adjoint()
    np.testing.assert_array_equal(dagger_qobj.dense(), arr.conj().T)


def test_ptrace_valid():
    """Test partial trace on a valid 4-qubit density matrices."""
    qket = ket(0, 1, 1, 0)
    rho = qket.to_density_matrix()

    # Different combinations of partial traces.
    reduced_single_qubit_ground = rho.ptrace(keep=[0], dims=[2, 2, 4])
    reduced_single_qubit_excited = rho.ptrace(keep=[1], dims=[2, 2, 4])
    reduced_double_qubit_1 = rho.ptrace(keep=[2], dims=[2, 2, 4])
    reduced_double_qubit_2 = rho.ptrace(keep=[2, 3], dims=[2, 2, 2, 2])
    reduced_double_qubit_3 = rho.ptrace(keep=[3, 2], dims=[2, 2, 2, 2])

    reduced_ket_qubit_1 = qket.ptrace(keep=[1])
    reduced_ket_qubit_0 = qket.ptrace(keep=[0])

    qket = ket(0, 1, 0, *[0 for _ in range(20)])
    reduced_ket_qubit_1_big = qket.ptrace(keep=[1])
    reduced_ket_qubit_0_big = qket.ptrace(keep=[0])

    qket = bra(0, 1, 0, *[0 for _ in range(20)])
    reduced_ket_qubit_1_big_bra = qket.ptrace(keep=[1])
    reduced_ket_qubit_0_big_bra = qket.ptrace(keep=[0])

    # Expected reduced density matrices:
    expected_single_qubit_ground = ket(0).to_density_matrix()
    expected_single_qubit_excited = ket(1).to_density_matrix()
    expected_double_qubit = ket(1, 0).to_density_matrix()

    # Checks:
    np.testing.assert_allclose(reduced_single_qubit_ground.dense(), expected_single_qubit_ground.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_single_qubit_excited.dense(), expected_single_qubit_excited.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_double_qubit_1.dense(), expected_double_qubit.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_double_qubit_2.dense(), expected_double_qubit.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_double_qubit_3.dense(), expected_double_qubit.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_1.dense(), expected_single_qubit_excited.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_0.dense(), expected_single_qubit_ground.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_1_big.dense(), expected_single_qubit_excited.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_0_big.dense(), expected_single_qubit_ground.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_1_big_bra.dense(), expected_single_qubit_excited.dense(), atol=1e-8)
    np.testing.assert_allclose(reduced_ket_qubit_0_big_bra.dense(), expected_single_qubit_ground.dense(), atol=1e-8)


def test_ptrace_valid_keep_with_automatic_dims_and_density_matrix():
    qket = ket(0, 0, 1, 0)
    reduced_single_qubit = qket.ptrace(keep=[2, 3])
    expected_single_qubit = ket(1, 0).to_density_matrix()
    np.testing.assert_allclose(reduced_single_qubit.dense(), expected_single_qubit.dense(), atol=1e-8)


def test_ptrace_works_for_operators_which_are_not_density_matrices():
    # Build a “diagonal” density matrix whose diagonal entries are 0…7. That way each composite basis |i0,i1,i2⟩ ↦
    # flat index i = 4*i0 + 2*i1 + i2 carries a unique number. And the trace != 1, so not a density operator
    dims = [2, 2, 2]
    full_dim = np.prod(dims)
    rho = np.diag(np.arange(full_dim, dtype=float))
    q_obj = QTensor(rho)

    # Pick an out of order keep list:
    keep = [0, 2]  # subspace 2 *then* subspace 0
    expected_result = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 10, 0], [0, 0, 0, 12]])
    np.testing.assert_allclose(q_obj.ptrace(keep, dims).dense(), expected_result, atol=1e-8)


def test_ptrace_invalid_keep():
    """Partial trace should raise ValueError if keep indices are out of bounds."""
    arr = np.eye(2)
    qobj = QTensor(arr)
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.ptrace(keep=[1], dims=[2])
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.ptrace(keep=[2], dims=[2])  # out of bounds index
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.ptrace(keep=[0, 1], dims=[2])  # too many indices


# --- Arithmetic Operator Tests ---


@pytest.mark.parametrize("other", [0, 0 + 0j])
def test_add_scalar_zero(other):
    """Adding zero (of complex/int type) should return the same QTensor."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    result = qobj + other
    np.testing.assert_array_equal(result.dense(), qobj.dense())


def test_add_QTensor():
    """Test addition between two QTensors."""
    arr = np.array([[1, 0], [0, 1]])
    q1 = QTensor(arr)
    q2 = QTensor(arr)
    result = q1 + q2
    np.testing.assert_array_equal(result.dense(), arr + arr)


def test_add_invalid_type():
    """Adding an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    with pytest.raises(TypeError):
        _ = qobj + "invalid"


def test_sub_QTensor():
    """Test subtraction between two QTensors."""
    arr1 = np.array([[2, 0], [0, 2]])
    arr2 = np.eye(2)
    q1 = QTensor(arr1)
    q2 = QTensor(arr2)
    result = q1 - q2
    np.testing.assert_array_equal(result.dense(), arr1 - arr2)


def test_sub_invalid_type():
    """Subtracting an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    with pytest.raises(TypeError):
        _ = qobj - 1


@pytest.mark.parametrize("scalar", [2, 2.5, 1 + 1j])
def test_mul_scalar(scalar):
    """Test multiplication with scalars."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    result = qobj * scalar
    np.testing.assert_array_equal(result.dense(), arr * scalar)


def test_mul_QTensor():
    """Test multiplication between two QTensors (elementwise multiplication)."""
    arr1 = np.array([[2, 0], [0, 2]])
    arr2 = np.eye(2)
    q1 = QTensor(arr1)
    q2 = QTensor(arr2)
    result = q1 * q2
    np.testing.assert_array_equal(result.dense(), arr1 * arr2)


def test_mul_invalid_type():
    """Multiplication with an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    with pytest.raises(TypeError):
        _ = qobj * "invalid"


def test_rmul():
    """Test right multiplication (scalar * QTensor)."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    result = 3 * qobj
    np.testing.assert_array_equal(result.dense(), arr * 3)


def test_matmul():
    """Test matrix multiplication using the @ operator between QTensors."""
    arr = np.array([[1, 2], [3, 4]])
    q1 = QTensor(arr)
    q2 = QTensor(np.eye(2))
    result = q1 @ q2
    np.testing.assert_array_equal(result.dense(), arr @ np.eye(2))


def test_matmul_invalid_type():
    """Matrix multiplication with an unsupported type should raise a TypeError."""
    arr = np.array([[1, 2], [3, 4]])
    qobj = QTensor(arr)
    with pytest.raises(TypeError):
        _ = qobj @ 3


# --- Norm and Unit Tests ---


def test_norm_scalar():
    """For a scalar QTensor, norm should return the single element."""
    qobj = QTensor(np.array([[5]]))
    assert qobj.norm() == 5


def test_norm_density_matrix():
    """Test norm calculation on a density matrix.

    When order is 'tr', the norm should be the trace.
    """
    qdm = ket(0).to_density_matrix()  # density matrix for |0>
    assert np.isclose(qdm.norm(order="tr"), 1)
    # Also test norm with an integer order via scipy_norm.
    expected_norm = scipy_norm(qdm.data, ord=1)
    assert np.isclose(qdm.norm(order=1), expected_norm)

    s = ket(0) + ket(1)
    qdm = s @ s.adjoint()

    assert qdm.norm("tr") == 2


def test_norm_ket():
    """Test norm for a ket state (should be 1 for a basis vector)."""
    qket = ket(1)
    assert np.isclose(qket.norm(), 1)


def test_unit_normalizes():
    """Test that unit() correctly normalizes a non-unit vector."""
    qket = ket(1)
    non_normalized = 3 * qket
    normalized = non_normalized.unit()
    assert np.isclose(normalized.norm(), 1)


def test_unit_zero_norm():
    """Normalization of a zero-norm QTensor should raise a ValueError."""
    zero_vector = np.zeros((2, 1))
    qobj = QTensor(zero_vector)
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.unit()


# --- Exponential and Type Checks ---


def test_expm():
    """Test the matrix exponential of a simple diagonal matrix."""
    arr = np.array([[0, 0], [0, np.log(2)]])
    qobj = QTensor(arr)
    result = qobj.expm()
    expected = np.array([[1, 0], [0, 2]])
    np.testing.assert_allclose(result.dense(), expected, atol=1e-8)


def test_is_ket():
    """Test is_ket: should be True for a valid ket state and False for a density matrix."""
    qket = ket(0)
    assert qket.is_ket()
    qdm = ket(0).to_density_matrix()
    assert not qdm.is_ket()


def test_is_bra():
    """Test is_bra: should be True for a valid bra state and False for a density matrix."""
    qbra = bra(0)
    assert qbra.is_bra()
    qdm = ket(0).to_density_matrix()
    assert not qdm.is_bra()


def test_is_scalar():
    """Test is_scalar for scalar vs non-scalar QTensors."""
    qscalar = QTensor(np.array([[42]]))
    assert qscalar.is_scalar()
    qket_obj = ket(0)
    assert not qket_obj.is_scalar()


def test_is_dm():
    """Test is_dm: density matrices (from ket) should pass, while non-dm matrices should not."""
    qdm = ket(0).to_density_matrix()
    assert qdm.is_density_matrix()
    non_dm = QTensor(np.array([[1, 2], [3, 4]]))
    assert not non_dm.is_density_matrix()


def test_is_herm():
    """Test is_herm for Hermitian and non-Hermitian matrices."""
    herm_matrix = np.array([[1, 2 + 1j], [2 - 1j, 3]])
    qherm = QTensor(herm_matrix)
    assert qherm.is_hermitian()
    non_herm = np.array([[1, 2], [3, 4]])
    qnonherm = QTensor(non_herm)
    assert not qnonherm.is_hermitian()


def test_to_dm_from_dm():
    """to_dm() called on a density matrix should return a valid density matrix."""
    qdm = ket(0).to_density_matrix()
    dm2 = qdm.to_density_matrix()
    np.testing.assert_allclose(dm2.dense(), qdm.dense(), atol=1e-8)


def test_to_dm_from_ket():
    """to_dm() should convert a ket state to a density matrix with trace 1."""
    qket_obj = ket(0)
    qdm = qket_obj.to_density_matrix()
    assert np.isclose(qdm.data.trace(), 1)


def test_to_dm_from_scalar():
    """Attempting to call to_dm() on a scalar should raise a ValueError."""
    qscalar = QTensor(np.array([[1]]))
    with pytest.raises(ValueError):  # noqa: PT011
        qscalar.to_density_matrix()


# --- Helper Function Tests ---


def test_basis():
    """Test that the basis function returns a vector with a 1 in the correct position."""
    N = 4
    n = 2
    qbasis = basis_state(n, N)
    assert qbasis.shape == (N, 1)
    dense = qbasis.dense().flatten()
    expected = np.zeros(N)
    expected[n] = 1
    np.testing.assert_array_equal(dense, expected)

    with pytest.raises(ValueError, match="must be in"):
        basis_state(5, 4)  # n >= N should raise error


@pytest.mark.parametrize("state", [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)])
def test_ket_valid(state):
    """Test that ket returns a valid quantum state for valid bit strings."""
    qket_obj = ket(*state)
    expected_dim = 2 ** len(state)
    assert qket_obj.shape == (expected_dim, 1)


@pytest.mark.parametrize("state", [(), (2,), (-1,), (0, 2), (1, 3)])
def test_ket_invalid(state):
    """ket should raise ValueError if any qubit state is not 0 or 1."""
    with pytest.raises(ValueError):  # noqa: PT011
        ket(*state)


@pytest.mark.parametrize("state", [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)])
def test_bra_valid(state):
    """Test that bra returns a valid bra state for valid bit strings."""
    qbra_obj = bra(*state)
    expected_dim = 2 ** len(state)
    assert qbra_obj.shape == (1, expected_dim)


@pytest.mark.parametrize("state", [(2,), (-1,), (0, 2), (1, 3)])
def test_bra_invalid(state):
    """bra should raise ValueError if any qubit state is not 0 or 1."""
    with pytest.raises(ValueError):  # noqa: PT011
        bra(*state)


def test_tensor():
    """Test the tensor product function on a list of QTensors."""
    q1 = ket(0)
    q2 = ket(1)
    qt = tensor_prod([q1, q2])
    np.testing.assert_array_equal(qt.dense().shape, (4, 1))


def test_bad_tensor_prod():
    with pytest.raises(ValueError, match="at least one"):
        tensor_prod([])


def test_expect_density():
    """Test the expectation value for a density matrix using the identity operator."""
    qdm = ket(0).to_density_matrix()
    identity = QTensor(np.eye(2))
    exp_val = expect_val(identity, qdm)
    # For a valid density matrix, trace(identity * rho) should be 1.
    assert np.isclose(exp_val, 1)


def test_expect_ket():
    """Test the expectation value for a ket state using the identity operator."""
    qket_obj = ket(0)
    identity = QTensor(np.eye(2))
    exp_val = expect_val(identity, qket_obj)
    # For a normalized ket, ⟨ψ|I|ψ⟩ should equal 1.
    assert np.isclose(exp_val, 1)


def test_expect_bra():
    """Test the expectation value for a bra state using the identity operator."""
    qbra_obj = bra(0)
    identity = QTensor(np.eye(2))
    exp_val = expect_val(identity, qbra_obj)
    # For a normalized bra, ⟨ψ|I|ψ⟩ should equal 1.
    assert np.isclose(exp_val, 1)


def test_to_density_matrix():
    s = ket(0) + ket(1)
    qdm = s @ s.adjoint()

    with pytest.raises(ValueError, match=r"Operator is not a density matrix \(trace\≠1 or not Hermitian\)."):
        qdm.to_density_matrix()

    s = ket(0)
    qdm = s.to_density_matrix()
    np.testing.assert_allclose(qdm.dense(), np.array([[1, 0], [0, 0]]), atol=1e-8)

    s = bra(0)
    qdm = s.to_density_matrix()
    np.testing.assert_allclose(qdm.dense(), np.array([[1, 0], [0, 0]]), atol=1e-8)


def test_3d_init():
    """QTensor should raise ValueError for 3D arrays."""
    arr = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="Input ndarray must be 2D"):
        QTensor(arr)


def test_bad_partial_trace():
    """Partial trace should raise ValueError for invalid keep indices."""

    qket = ket(0, 1)
    rho = qket.to_density_matrix()

    with pytest.raises(ValueError, match="must be positive"):
        rho.ptrace(keep=[], dims=[-2, 2])
    with pytest.raises(ValueError, match="does not match Hilbert"):
        rho.ptrace(keep=[], dims=[1, 1])
    with pytest.raises(ValueError, match="Duplicate indices in keep"):
        rho.ptrace(keep=[0, 0], dims=[2, 2])

    nothing = rho.ptrace(keep=[], dims=[2, 2])
    assert nothing.shape == (1, 1)

    everything = rho.ptrace(keep=[0, 1], dims=[2, 2])
    np.testing.assert_allclose(everything.dense(), rho.dense(), atol=1e-8)

    rho._data = np.array([[1, 2, 3], [3, 4, 5]])
    with pytest.raises(ValueError, match="not a valid state or operator"):
        rho.ptrace(keep=[0], dims=[3, 1])

    big_dim = 2**21
    qket._data = csr_matrix((big_dim, 1))
    assert qket.is_ket()
    qket.ptrace(keep=[], dims=[2 for _ in range(21)])


def test_large_trace_norm():
    qket = ket(*([0 for _ in range(21)]))
    rho = qket.to_density_matrix()
    rho._data[0, 0] = 2.0
    with pytest.raises(ValueError, match="Trace norm for large"):
        rho.norm(order="tr")


def test_non_hermitian_trace_norm():
    arr = np.array([[0, 1], [0, 0]])
    qobj = QTensor(arr)
    norm = qobj.norm(order="tr")
    assert np.isclose(norm, 1.0)


def test_non_hermitian_large_trace_norm():
    arr = np.zeros((2048, 2048))
    arr[0, 1] = 1.0
    qobj = QTensor(arr)
    with pytest.raises(ValueError, match="norm for large non-Hermitian"):
        qobj.norm(order="tr")


def test_trace_norm_ket_bra():
    v_ket = ket(0)
    v_bra = bra(0)
    assert np.isclose(v_ket.norm(order="tr"), 1.0)
    assert np.isclose(v_bra.norm(order="tr"), 1.0)


def test_non_operator_to_density_matrix():
    arr = np.array([[0, 2], [3, 0]])
    qobj = QTensor(arr)

    qobj._data = csr_matrix(np.array([[1, 2, 3], [3, 4, 5]]))
    with pytest.raises(ValueError, match="Invalid object for density matrix conversion"):
        qobj.to_density_matrix()

    v_ket = QTensor(np.array([[0], [0]]))
    with pytest.raises(ValueError, match="zero trace"):
        v_ket.to_density_matrix()


def test_non_hermitian_is_dm():
    arr = np.array([[0, 1], [0, 1]])
    qobj = QTensor(arr)
    assert not qobj.is_density_matrix()
    assert not qobj.is_hermitian()


def test_is_dm_no_eigsh(monkeypatch):
    def mock_eigsh(*args, **kwargs):
        raise ArpackNoConvergence("Simulated failure", [], [])

    monkeypatch.setattr(qilisdk.core.qtensor, "eigsh", mock_eigsh)

    arr = np.array([[1, 1], [1, 0]])
    qobj = QTensor(arr)
    assert not qobj.is_density_matrix()

    big_arr = np.zeros((4096, 4096))
    big_arr[0, 0] = 1.0
    big_arr[1, 0] = 1.0
    big_arr[0, 1] = 1.0
    qobj_big = QTensor(big_arr)
    assert not qobj_big.is_density_matrix()


def test_is_hermitian_not_operator():
    arr = np.array([[1, 2], [3, 4]])
    qobj = QTensor(arr)
    qobj._data = csr_matrix(np.array([[1, 2, 3], [3, 4, 5]]))
    assert not qobj.is_hermitian()


def test_arithmetic():
    a = QTensor(np.array([[1, 0], [0, 1]]))
    d = 0 + 0j
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = a + 3.0
    assert a + d == a
    assert d + a == a


def test_qtensor_output():
    arr = np.array([[1, 0], [0, 1]])
    qobj = QTensor(arr)
    output = repr(qobj)
    assert "QTensor" in output
    assert "shape=2x2" in output
    assert "nnz=2" in output


def test_expect_val_bad_operator():
    qket_obj = ket(0)
    arr = np.array([[0, 1], [3, 4]])
    qbad = QTensor(arr)

    bad_state = qbad
    with pytest.raises(ValueError, match="state is invalid"):
        expect_val(qbad, bad_state)

    qbad._data = csr_matrix(np.array([[1, 2, 3], [3, 4, 5]]))
    with pytest.raises(ValueError, match="must be square"):
        expect_val(qbad, qket_obj)


def test_qtensor_equality():
    arr1 = np.array([[1, 0], [0, 1]])
    arr2 = np.array([[1, 0], [0, 1]])
    arr3 = np.array([[0, 1], [1, 0]])

    q1 = QTensor(arr1)
    q2 = QTensor(arr2)
    q3 = QTensor(arr3)

    assert q1 == q2
    assert q1 != q3


def test_qtensor_hash():
    arr1 = np.array([[1, 0], [0, 1]])
    arr2 = np.array([[1, 0], [0, 1]])
    arr3 = np.array([[0, 1], [1, 0]])

    q1 = QTensor(arr1)
    q2 = QTensor(arr2)
    q3 = QTensor(arr3)

    assert hash(q1) == hash(q2)
    assert hash(q1) != hash(q3)
