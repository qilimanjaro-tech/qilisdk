import numpy as np
import pytest
from scipy.sparse import csc_array, issparse
from scipy.sparse.linalg import norm as scipy_norm

from qilisdk.analog.quantum_objects import QuantumObject, basis_state, bra, expect_val, ket, tensor_prod

# --- Constructor Tests ---


def test_constructor_valid_ndarray():
    """QuantumObject should accept a valid NumPy array and convert it to sparse."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    assert issparse(qobj.data)


def test_constructor_valid_sparse():
    """QuantumObject should accept a valid SciPy sparse matrix."""
    sparse_mat = csc_array([[1, 0], [0, 1]])
    qobj = QuantumObject(sparse_mat)
    # Should be stored as a CSR matrix.
    assert qobj.data.format == "csr"


@pytest.mark.parametrize("invalid_input", [1, "string", [1, 2, 3]])
def test_constructor_invalid_input(invalid_input):
    """QuantumObject should raise ValueError for inputs that are not arrays or sparse matrices."""
    with pytest.raises(ValueError):  # noqa: PT011
        QuantumObject(invalid_input)


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
    """QuantumObject should raise ValueError for arrays with invalid shapes."""
    arr = np.zeros(shape)
    with pytest.raises(ValueError):  # noqa: PT011
        QuantumObject(arr)


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
    qobj = QuantumObject(array)
    assert qobj.nqubits == expected_nqubits


def test_dense_property():
    """The dense property should return a NumPy array equivalent to the original data."""
    arr = np.array([[1, 2], [3, 4]])
    qobj = QuantumObject(arr)
    np.testing.assert_array_equal(qobj.dense, arr)


# --- Method Tests ---


def test_dag():
    """Test that the dagger (adjoint) method returns the conjugate transpose."""
    arr = np.array([[1 + 2j, 2], [3, 4 + 5j]])
    qobj = QuantumObject(arr)
    dagger_qobj = qobj.adjoint()
    np.testing.assert_array_equal(dagger_qobj.dense, arr.conj().T)


def test_ptrace_valid():
    """Test partial trace on a valid 2-qubit density matrix.

    The test creates a 2-qubit state |00⟩, converts it to a density matrix,
    and then traces out the second qubit.
    """
    qket = ket(0, 0)
    rho = qket.to_density_matrix()
    # dims for a 2-qubit system are [2, 2]; keep the first qubit (index 0).
    reduced = rho.ptrace([2, 2], keep=[0])
    # Expected reduced density matrix is the pure state |0⟩.
    expected = ket(0).to_density_matrix()
    np.testing.assert_allclose(reduced.dense, expected.dense, atol=1e-8)


def test_ptrace_invalid_dims():
    """Partial trace should raise ValueError if dims do not match the matrix dimensions."""
    arr = np.eye(2)
    qobj = QuantumObject(arr)
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.ptrace([2, 2], keep=[0])


# --- Arithmetic Operator Tests ---


@pytest.mark.parametrize("other", [0, 0 + 0j])
def test_add_scalar_zero(other):
    """Adding zero (of complex/int type) should return the same QuantumObject."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    result = qobj + other
    np.testing.assert_array_equal(result.dense, qobj.dense)


def test_add_quantumobject():
    """Test addition between two QuantumObjects."""
    arr = np.array([[1, 0], [0, 1]])
    q1 = QuantumObject(arr)
    q2 = QuantumObject(arr)
    result = q1 + q2
    np.testing.assert_array_equal(result.dense, arr + arr)


def test_add_invalid_type():
    """Adding an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    with pytest.raises(TypeError):
        _ = qobj + "invalid"


def test_sub_quantumobject():
    """Test subtraction between two QuantumObjects."""
    arr1 = np.array([[2, 0], [0, 2]])
    arr2 = np.eye(2)
    q1 = QuantumObject(arr1)
    q2 = QuantumObject(arr2)
    result = q1 - q2
    np.testing.assert_array_equal(result.dense, arr1 - arr2)


def test_sub_invalid_type():
    """Subtracting an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    with pytest.raises(TypeError):
        _ = qobj - 1


@pytest.mark.parametrize("scalar", [2, 2.5, 1 + 1j])
def test_mul_scalar(scalar):
    """Test multiplication with scalars."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    result = qobj * scalar
    np.testing.assert_array_equal(result.dense, arr * scalar)


def test_mul_quantumobject():
    """Test multiplication between two QuantumObjects (elementwise multiplication)."""
    arr1 = np.array([[2, 0], [0, 2]])
    arr2 = np.eye(2)
    q1 = QuantumObject(arr1)
    q2 = QuantumObject(arr2)
    result = q1 * q2
    np.testing.assert_array_equal(result.dense, arr1 * arr2)


def test_mul_invalid_type():
    """Multiplication with an unsupported type should raise a TypeError."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    with pytest.raises(TypeError):
        _ = qobj * "invalid"


def test_rmul():
    """Test right multiplication (scalar * QuantumObject)."""
    arr = np.array([[1, 0], [0, 1]])
    qobj = QuantumObject(arr)
    result = 3 * qobj
    np.testing.assert_array_equal(result.dense, arr * 3)


def test_matmul():
    """Test matrix multiplication using the @ operator between QuantumObjects."""
    arr = np.array([[1, 2], [3, 4]])
    q1 = QuantumObject(arr)
    q2 = QuantumObject(np.eye(2))
    result = q1 @ q2
    np.testing.assert_array_equal(result.dense, arr @ np.eye(2))


def test_matmul_invalid_type():
    """Matrix multiplication with an unsupported type should raise a TypeError."""
    arr = np.array([[1, 2], [3, 4]])
    qobj = QuantumObject(arr)
    with pytest.raises(TypeError):
        _ = qobj @ 3


# --- Norm and Unit Tests ---


def test_norm_scalar():
    """For a scalar QuantumObject, norm should return the single element."""
    qobj = QuantumObject(np.array([[5]]))
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
    """Normalization of a zero-norm QuantumObject should raise a ValueError."""
    zero_vector = np.zeros((2, 1))
    qobj = QuantumObject(zero_vector)
    with pytest.raises(ValueError):  # noqa: PT011
        qobj.unit()


# --- Exponential and Type Checks ---


def test_expm():
    """Test the matrix exponential of a simple diagonal matrix."""
    arr = np.array([[0, 0], [0, np.log(2)]])
    qobj = QuantumObject(arr)
    result = qobj.expm()
    expected = np.array([[1, 0], [0, 2]])
    np.testing.assert_allclose(result.dense, expected, atol=1e-8)


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
    """Test is_scalar for scalar vs non-scalar QuantumObjects."""
    qscalar = QuantumObject(np.array([[42]]))
    assert qscalar.is_scalar()
    qket_obj = ket(0)
    assert not qket_obj.is_scalar()


def test_is_dm():
    """Test is_dm: density matrices (from ket) should pass, while non-dm matrices should not."""
    qdm = ket(0).to_density_matrix()
    assert qdm.is_density_matrix()
    non_dm = QuantumObject(np.array([[1, 2], [3, 4]]))
    assert not non_dm.is_density_matrix()


def test_is_herm():
    """Test is_herm for Hermitian and non-Hermitian matrices."""
    herm_matrix = np.array([[1, 2 + 1j], [2 - 1j, 3]])
    qherm = QuantumObject(herm_matrix)
    assert qherm.is_hermitian()
    non_herm = np.array([[1, 2], [3, 4]])
    qnonherm = QuantumObject(non_herm)
    assert not qnonherm.is_hermitian()


def test_to_dm_from_dm():
    """to_dm() called on a density matrix should return a valid density matrix."""
    qdm = ket(0).to_density_matrix()
    dm2 = qdm.to_density_matrix()
    np.testing.assert_allclose(dm2.dense, qdm.dense, atol=1e-8)


def test_to_dm_from_ket():
    """to_dm() should convert a ket state to a density matrix with trace 1."""
    qket_obj = ket(0)
    qdm = qket_obj.to_density_matrix()
    assert np.isclose(qdm.data.trace(), 1)


def test_to_dm_from_scalar():
    """Attempting to call to_dm() on a scalar should raise a ValueError."""
    qscalar = QuantumObject(np.array([[1]]))
    with pytest.raises(ValueError):  # noqa: PT011
        qscalar.to_density_matrix()


# --- Helper Function Tests ---


def test_basis():
    """Test that the basis function returns a vector with a 1 in the correct position."""
    N = 4
    n = 2
    qbasis = basis_state(n, N)
    assert qbasis.shape == (N, 1)
    dense = qbasis.dense.flatten()
    expected = np.zeros(N)
    expected[n] = 1
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.parametrize("state", [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)])
def test_ket_valid(state):
    """Test that ket returns a valid quantum state for valid bit strings."""
    qket_obj = ket(*state)
    expected_dim = 2 ** len(state)
    assert qket_obj.shape == (expected_dim, 1)


@pytest.mark.parametrize("state", [(2,), (-1,), (0, 2), (1, 3)])
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
    """Test the tensor product function on a list of QuantumObjects."""
    q1 = ket(0)
    q2 = ket(1)
    qt = tensor_prod([q1, q2])
    np.testing.assert_array_equal(qt.dense.shape, (4, 1))


def test_expect_density():
    """Test the expectation value for a density matrix using the identity operator."""
    qdm = ket(0).to_density_matrix()
    identity = QuantumObject(np.eye(2))
    exp_val = expect_val(identity, qdm)
    # For a valid density matrix, trace(identity * rho) should be 1.
    assert np.isclose(exp_val, 1)


def test_expect_ket():
    """Test the expectation value for a ket state using the identity operator."""
    qket_obj = ket(0)
    identity = QuantumObject(np.eye(2))
    exp_val = expect_val(identity, qket_obj)
    # For a normalized ket, ⟨ψ|I|ψ⟩ should equal 1.
    assert np.isclose(exp_val, 1)
