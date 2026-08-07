Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class represents a symbolic Hamiltonian as a sum of weighted Pauli operators. You can create Hamiltonians using the built-in Pauli operators and combine them with standard arithmetic operations.

Constructing
======================

To construct a Hamiltonian with a single Pauli, you can use the constructors ``X(i)``, ``Y(i)``, ``Z(i)``, ``I(i)``. 
From these single-qubit operators, you can build multi-qubit Hamiltonians using arithmetic operations.
The operations follow Python syntax, for example: ``2 * Z(0) + Z(1)`` and ``Z(0) * Z(1)`` build multi-qubit Hamiltonians.

Common Hamiltonians
======================

Alternatively, for the models that come up most often there are named constructors:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Constructor
     - Hamiltonian
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.transverse_field`
     - :math:`\sum_i h_x X_i`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.longitudinal_field`
     - :math:`\sum_i h_z Z_i`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.ising`
     - :math:`\sum_{i<j} J Z_i Z_j + \sum_i h_z Z_i`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.ising_chain`
     - :math:`\sum_i J Z_i Z_{i+1} + \sum_i h_z Z_i`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.ising_grid`
     - :math:`\sum_{\langle i, j \rangle} J Z_i Z_j + \sum_i h_z Z_i` on a square lattice
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.transverse_field_ising`
     - :math:`\sum_{i<j} J Z_i Z_j + \sum_i h_x X_i + \sum_i h_z Z_i`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.xy`
     - :math:`\sum_{i<j} \left( J_x X_i X_j + J_y Y_i Y_j \right)`
   * - :meth:`~qilisdk.analog.hamiltonian.Hamiltonian.heisenberg`
     - :math:`\sum_{i<j} \left( J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j \right) + \sum_i h_z Z_i`

.. code-block:: python

    from qilisdk.analog import Hamiltonian

    H = Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=1.3, zz_coefficient=-2)
    print(H)

**Output:**

::

    1.3 X(0) + 1.3 X(1) - 2 Z(0) Z(1)

Every coefficient argument accepts either a fixed value or a ``(low, high)`` range. Given a range,
each term it weights gets its own value drawn uniformly at random from it, so any of the constructors
above can build a disordered model just by passing a tuple instead of a number. A ``seed`` can
also be passed for reproducibility.

.. code-block:: python

    from qilisdk.analog import Hamiltonian

    H = Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=(-1, 1), zz_coefficient=(-1, 1), seed=1)
    print(H)

**Output:**

::

    0.023643249400513433 X(0) + 0.9009273926518706 X(1) - 0.7116807745607325 Z(0) Z(1)

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

**Common Hamiltonians**:

- transverse field: :meth:`Hamiltonian.transverse_field(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.transverse_field>`
- longitudinal field: :meth:`Hamiltonian.longitudinal_field(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.longitudinal_field>`
- Ising: :meth:`Hamiltonian.ising(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.ising>`
- Ising chain: :meth:`Hamiltonian.ising_chain(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.ising_chain>`
- Ising grid: :meth:`Hamiltonian.ising_grid(rows, columns)<qilisdk.analog.hamiltonian.Hamiltonian.ising_grid>`
- transverse-field Ising: :meth:`Hamiltonian.transverse_field_ising(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.transverse_field_ising>`
- XY: :meth:`Hamiltonian.xy(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.xy>`
- Heisenberg: :meth:`Hamiltonian.heisenberg(nqubits)<qilisdk.analog.hamiltonian.Hamiltonian.heisenberg>`

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

