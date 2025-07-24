Analog
=========================


some introduction about how the analog module is composed of these two modules 
- Hamiltonian 
- Schedule 
- quantum objects


Hamiltonian
---------------------------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class is a data object class that is designed to construct arbitrary Hamiltonians. 
The construction of hamiltonian is done by writing the analytical expression of the hamiltonian. For example, to can express an
Ising hamiltonian that takes the following format:

.. math::

    H_{Ising}  =  - \sum_{<i, j>} J_{ij} \sigma^Z_i \sigma^Z_j - \sum_j h_j \sigma^Z_j

We will use the pauli :class:`~qilisdk.analog.Z` operators. Here is an example of a fully connected Ising Hamiltonian acting on 3:

.. code-block:: python

    from qilisdk.analog import Z, X, Y

    nqubits = 3
    J = [
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
    ]
    h = [1, 2, 3]

    H = -1 * sum( 
            sum(
                J[i][j] * Z(i) * Z(j) 
                for i in range(nqubits)
            ) 
            for j in range(nqubits)
        ) - sum(h[j] * Z(j) for j in range(nqubits))

    print(H)

Output:

::

    -2 Z(0) Z(1) - 3 Z(0) Z(2) - 4 Z(1) Z(2) - Z(0) - 2 Z(1) - 3 Z(2)




Schedule 
----------------------------

