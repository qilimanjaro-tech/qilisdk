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
