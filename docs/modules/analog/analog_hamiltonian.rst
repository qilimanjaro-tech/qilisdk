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


Visualizing
============================

:meth:`H.draw()<qilisdk.analog.hamiltonian.Hamiltonian.draw>` renders a Hamiltonian as an interaction graph:

- Every qubit is a node, whose disc is split into one slice per local field acting on it, labelled with its Pauli type.
- Every two-qubit term is an edge between the qubits it couples, drawn with a line style per coupling type (see the legend).
- Every term acting on three or more qubits is a star-shaped hyperedge joined at the centroid of the qubits involved.
- Slice and edge colours encode the coefficient of the corresponding term, as described by the colour bar.
- A constant (identity) term is annotated below the graph as an energy offset.

.. code-block:: python

    from qilisdk.analog import X, Z

    nqubits = 3
    J = {(0, 1): 1, (0, 2): 2, (1, 2): 4}

    H = sum(weight * Z(i) * Z(j) for (i, j), weight in J.items()) + sum(X(i) for i in range(nqubits))
    H.draw()

The appearance is controlled with :class:`~qilisdk.utils.visualization.style.HamiltonianStyle`, which shares the
themes of the circuit and schedule renderers. It selects the ``rustworkx`` layout used to place the qubits
(``"spring"``, ``"circular"``, ``"shell"``, ``"spiral"`` or ``"random"``, or explicit ``positions``), whether local
fields and couplings share a single colour scale, and which annotations are drawn:

.. code-block:: python

    from qilisdk.analog import X, Z
    from qilisdk.utils.visualization.style import HamiltonianStyle
    from qilisdk.utils.visualization.themes import dark

    H = 5 * X(0) - 3 * Z(0) + X(1) + 0.1 * Z(0) * Z(1) + 0.05 * X(0) * X(1)

    H.draw(
        HamiltonianStyle(
            theme=dark,
            layout="circular",
            title="My Hamiltonian",
            # Local fields and couplings live on very different scales here, so give each its own colour bar.
            separate_color_scales=True,
        )
    )

To save the figure instead of showing it, pass a ``filepath`` (the format is inferred from the extension):

.. code-block:: python

    from qilisdk.analog import X, Z

    H = X(0) + Z(0) * Z(1)
    H.draw(filepath="hamiltonian.png")
