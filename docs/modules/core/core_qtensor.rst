QTensor
---------------

The :mod:`~qilisdk.core.qtensor` module defines the :class:`~qilisdk.core.qtensor.QTensor`
class and related helpers for representing and manipulating quantum states and
operators in sparse form.

The :class:`~qilisdk.core.qtensor.QTensor` wraps a dense NumPy array or SciPy sparse matrix into a CSR-format sparse matrix, and can represent:

- **Kets** (column vectors of shape ``(2**N, 1)``)  
- **Bras** (row vectors of shape ``(1, 2**N)``)  
- **Operators / Density Matrices** (square matrices of shape ``(2**N, 2**N)``)  
- **Scalars** (``(1, 1)`` matrices)  

Examples of creating various quantum objects:

.. code-block:: python

    import numpy as np
    from qilisdk.core.qtensor import QTensor

    # 1‑qubit |0> ket
    psi_ket = QTensor(np.array([[1], [0]]))
    print("Ket:", psi_ket.dense(), "is_ket?", psi_ket.is_ket())
    print("-" * 20)

    # 1‑qubit <0| bra
    psi_bra = QTensor(np.array([[1, 0]]))
    print("Bra:", psi_bra.dense(), "is_bra?", psi_bra.is_bra())
    print("-" * 20)

    # Density matrix |0><0|
    rho = QTensor(np.array([[1, 0], [0, 0]]))
    print("Density matrix:\n", rho.dense(), "is_density_matrix?", rho.is_density_matrix())
    print("-" * 20)

    # Scalar 0.5
    scalar = QTensor(np.array([[0.5]]))
    print("Scalar:", scalar.dense(), "is_scalar?", scalar.is_scalar())

**Output**

::

    Ket: [[1]
    [0]] is_ket? True
    --------------------
    Bra: [[1 0]] is_bra? True
    --------------------
    Density matrix:
    [[1 0]
    [0 0]] is_density_matrix? True
    --------------------
    Scalar: [[0.5]] is_scalar? True

Helper constructors
^^^^^^^^^^^^^^^^^^^

There are also several constructors for common quantum objects:

 - :func:`~qilisdk.core.qtensor.ket` for computational basis kets
 - :func:`~qilisdk.core.qtensor.bra` for computational basis bras
 - :func:`~qilisdk.core.qtensor.basis_state` for N-dimensional basis states with a single 1 at the specified index
 - :func:`~qilisdk.core.qtensor.identity` for identity operators of specified dimension
 - :func:`~qilisdk.core.qtensor.zero` for generating zero tensors of specified dimension
 - :func:`~qilisdk.core.qtensor.ghz` for generating GHZ states of specified number of qubits
 - :func:`~qilisdk.core.qtensor.uniform` for generating uniform superposition states of specified number of qubits

.. code-block:: python

    from qilisdk.core.qtensor import QTensor

    # Single‑qubit
    print("ket(0):\n", QTensor.ket(0), "\nis_ket?", QTensor.ket(0).is_ket())
    print("bra(1):\n", QTensor.bra(1), "\nis_bra?", QTensor.bra(1).is_bra())

    # Fock basis in N=4 Hilbert space
    print("basis_state(2,4):\n", QTensor.basis_state(2, 4), "\nshape:", QTensor.basis_state(2, 4).shape)
    
    # GHZ state for 2 qubits
    print("GHZ state for 2 qubits:\n", QTensor.ghz(2))

    # Identity and zero operators
    print("Identity (4x4):\n", QTensor.identity(2))
    print("Zero (2x2):\n", QTensor.zero(2))

**Output**

::

    ket(0):
    [[1.]
    [0.]] 
    is_ket? True
    bra(1):
    [[0. 1.]] 
    is_bra? True
    basis_state(2,4):
    [[0.]
    [0.]
    [1.]
    [0.]] 
    shape: (4, 1)

Quantum Object Properties & Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All data are stored sparsely, but you can retrieve dense or sparse views:

- :attr:`.data<qilisdk.core.qtensor.QTensor.data>`: get the underlying sparse matrix (SciPy CSR format)
- :meth:`.dense()<qilisdk.core.qtensor.QTensor.dense>`: convert to a dense NumPy array (use with caution for large tensors)
- or directly accessing elements by value with ``qtensor[i, j]``

Common matrix operations are also available:

- :meth:`.adjoint()<qilisdk.core.qtensor.QTensor.adjoint>`: conjugate transpose  
- :meth:`.conjugate()<qilisdk.core.qtensor.QTensor.conjugate>`: element-wise complex conjugate
- :meth:`.transpose()<qilisdk.core.qtensor.QTensor.transpose>`: matrix transpose
- :meth:`.exp()<qilisdk.core.qtensor.QTensor.exp>`: matrix exponential
- :meth:`.log()<qilisdk.core.qtensor.QTensor.log>`: matrix logarithm
- :meth:`.sqrt()<qilisdk.core.qtensor.QTensor.sqrt>`: matrix square root
- :meth:`.rank()<qilisdk.core.qtensor.QTensor.rank>`: compute the rank of the operator
- :meth:`.pow(exponent)<qilisdk.core.qtensor.QTensor.pow>`: matrix power
- :meth:`.norm(order="l2")<qilisdk.core.qtensor.QTensor.norm>`: vector or matrix norm
- :meth:`.normalized(order='l2')<qilisdk.core.qtensor.QTensor.normalized>`: normalize to unit norm
- :meth:`.eig()<qilisdk.core.qtensor.QTensor.eig>`: get the eigenvalues and eigenvectors
- :meth:`.trace()<qilisdk.core.qtensor.QTensor.trace>`: compute the trace of an operator
- :meth:`.dot(other)<qilisdk.core.qtensor.QTensor.dot>`: Frobenius inner product with another QTensor

As well as some quantum-specific transformations:

- :meth:`.entropy_von_neumann()<qilisdk.core.qtensor.QTensor.entropy_von_neumann>`: compute the von Neumann entropy of a density matrix
- :meth:`.entropy_renyi(alpha)<qilisdk.core.qtensor.QTensor.entropy_renyi>`: compute the Rényi entropy of a density matrix for a given order alpha
- :meth:`.commutator(other)<qilisdk.core.qtensor.QTensor.commutator>`: compute the commutator with another operator
- :meth:`.anticommutator(other)<qilisdk.core.qtensor.QTensor.anticommutator>`: compute the anticommutator with another operator
- :meth:`.fidelity(other)<qilisdk.core.qtensor.QTensor.fidelity>`: compute the fidelity between two quantum states
- :meth:`.probabilities()<qilisdk.core.qtensor.QTensor.probabilities>`: compute the probability distribution of each state in the computational basis
- :meth:`.partial_trace(keep)<qilisdk.core.qtensor.QTensor.ptrace>`: partial trace
- :meth:`.reset_qubits(qubits)<qilisdk.core.qtensor.QTensor.reset_qubits>`: reset specified qubits to 0
- :meth:`.dagger()<qilisdk.core.qtensor.QTensor.dagger>`: alias for adjoint

Examples:

.. code-block:: python

    import numpy as np
    from qilisdk.core.qtensor import QTensor

    # Adjoint of a non-Hermitian operator
    A = QTensor(np.array([[1+1j, 2], [3, 4]]))
    A_dag = A.adjoint()
    print("A:\n", A.dense())
    print("A†:\n", A_dag.dense())

    # Matrix exponential of Pauli-X
    X = QTensor(np.array([[0, 1], [1, 0]]))
    expX = X.exp()
    print("exp(X):\n", np.round(expX.dense(), 3))

    # Norm of a ket and a density matrix
    ket0 = QTensor(np.array([[1], [0]]))
    dm = ket0.to_density_matrix()
    print("||ket0|| =", ket0.norm())
    print("trace norm(dm) =", dm.norm(order='l2'))

    # Partial trace of a Bell state
    from qilisdk.core.qtensor import ket, tensor_prod
    bell = (tensor_prod([ket(0), ket(0)]) + tensor_prod([ket(1), ket(1)])).unit()
    rho_bell = bell.to_density_matrix()
    print("rho_bell:\n", rho_bell)
    rhoA = rho_bell.ptrace([0])
    print("rho_A:\n", rhoA.dense())

**Output**

::

    A:
    [[1.+1.j 2.+0.j]
    [3.+0.j 4.+0.j]]
    A†:
    [[1.-1.j 3.+0.j]
    [2.+0.j 4.+0.j]]
    exp(X):
    [[1.543 1.175]
    [1.175 1.543]]
    ||ket0|| = 1.0
    trace norm(dm) = 1.0
    rho_bell:
    QTensor(shape=4x4, nnz=4, format='csr')
    [[0.5 0.  0.  0.5]
    [0.  0.  0.  0. ]
    [0.  0.  0.  0. ]
    [0.5 0.  0.  0.5]]
    rho_A:
    [[0.5 0. ]
    [0.  0.5]]


Extra Utilities
^^^^^^^^^^^^^^^

- **Tensor product** with :func:`~qilisdk.core.qtensor.tensor_prod`  
- **Expectation value** with :func:`~qilisdk.core.qtensor.expect_val`  

.. code-block:: python

    from qilisdk.core.qtensor import QTensor, expect_val, ket, tensor_prod
    import numpy as np

    # Two‑qubit Hadamard tensor
    H = QTensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    H2 = tensor_prod([H, H])
    print("H ⊗ H:\n", np.round(H2.dense(), 3))

    # Expectation of Z⊗Z on |00>
    Z = QTensor(np.array([[1, 0], [0, -1]]))
    zz = tensor_prod([Z, Z])
    psi00 = tensor_prod([ket(0), ket(0)])
    rho00 = psi00.to_density_matrix()
    ev = expect_val(zz, rho00)
    print("⟨ZZ⟩ on |00> =", ev)

**Output**

::

    H ⊗ H:
    [[ 0.5  0.5  0.5  0.5]
    [ 0.5 -0.5  0.5 -0.5]
    [ 0.5  0.5 -0.5 -0.5]
    [ 0.5 -0.5 -0.5  0.5]]
    ⟨ZZ⟩ on |00> = 1.0

Visualization of Quantum States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also visualize single-qubit states on the Bloch sphere using :meth:`.draw()<qilisdk.core.qtensor.QTensor.draw>`:

.. code-block:: python

    from qilisdk.core import QTensor

    state = QTensor.ket(0)
    state.draw()

The appearance of this plot can be customized using a :class:`~qilisdk.utils.visualization.style.QTensorStyle` object, 
which allows you to set colors, point density, and other visual elements:

.. code-block:: python

    from qilisdk.core import QTensor
    from qilisdk.utils.visualization import QTensorStyle

    state = QTensor.ket(0)
    style = QTensorStyle(
                sphere_color="blue", 
                arrow_color="lightblue", 
                draw_center_circle=True, 
                sphere_points=100, 
                draw_reference_points=True,
            )
    state.draw(style=style)

