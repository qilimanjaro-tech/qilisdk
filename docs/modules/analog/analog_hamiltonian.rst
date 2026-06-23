Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class represents a symbolic Hamiltonian as a sum of weighted Pauli operators. You can create Hamiltonians using the built-in Pauli operators and combine them with standard arithmetic operations.

Constructing
======================

To construct a Hamiltonian with a single Pauli, you can use the constructors ``X(i)``, ``Y(i)``, ``Z(i)``, ``I(i)``. 
From these single-qubit operators, you can build multi-qubit Hamiltonians using arithmetic operations.
The operations follow Python syntax, for example: ``2 * Z(0) + Z(1)`` and ``Z(0) * Z(1)`` build multi-qubit Hamiltonians.

List of Operations
======================

**Arithmetic operations**:

- Addition: ``H1 + H2``  
- Scalar multiplication: ``5 * H``  
- multiplication: ``H0 * H1``  
- Subtraction: ``H1 - H2``
- Division by scalar: ``H / 5``
- Negation: ``-H``

**Extra Symbolic Operators**:

- commutator: :meth:`H1.commutator(H2)<qilisdk.analog.hamiltonian.Hamiltonian.commutator>`  
- anticommutator: :meth:`H1.anticommutator(H2)<qilisdk.analog.hamiltonian.Hamiltonian.anticommutator>`
- vector_norm: :meth:`H.vector_norm()<qilisdk.analog.hamiltonian.Hamiltonian.vector_norm>`
- frobenius_norm: :meth:`H.frobenius_norm()<qilisdk.analog.hamiltonian.Hamiltonian.frobenius_norm>`
- trace: :meth:`H.trace()<qilisdk.analog.hamiltonian.Hamiltonian.trace>`

**Exporting Hamiltonians**:

- to matrix: :meth:`H.to_matrix(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.to_matrix>`
- to qtensor: :meth:`H.to_qtensor(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.to_qtensor>`

**Importing Hamiltonians**:

- from qtensor: :meth:`Hamiltonian.from_qtensor(qtensor)<qilisdk.analog.hamiltonian.Hamiltonian.from_qtensor>`
- from string: :meth:`Hamiltonian.parse(hamiltonian_string)<qilisdk.analog.hamiltonian.Hamiltonian.parse>`

Example: Ising Hamiltonian
============================

To define an Ising Hamiltonian of the form:

.. math::

    H_{\text{Ising}}  =  - \sum_{\langle i, j \rangle} J_{ij} \sigma^Z_i \sigma^Z_j - \sum_j h_j \sigma^Z_j

you can use the Pauli ``Z`` operators from the library:

.. code-block:: python

    from qilisdk.analog import Z

    nqubits = 3
    J = {(0, 1): 1, (0, 2): 2, (1, 2): 4}
    h = {0: 1, 1: 2, 2: 3}

    coupling = sum(weight * Z(i) * Z(j) for (i, j), weight in J.items())
    fields = sum(weight * Z(i) for i, weight in h.items())

    H = -(coupling + fields)
    print(H)

**Output:**

::

    - Z(0) Z(1) - 2 Z(0) Z(2) - 4 Z(1) Z(2) - Z(0) - 2 Z(1) - 3 Z(2)

