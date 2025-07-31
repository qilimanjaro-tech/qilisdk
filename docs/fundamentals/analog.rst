Analog Module
=============

The :mod:`~qilisdk.analog` module in Qili SDK enables the construction and simulation of analog quantum systems. It is composed of the following core components:

- :class:`~qilisdk.analog.hamiltonian.Hamiltonian`: Tools to build arbitrary Hamiltonians using symbolic Pauli operators.
- :class:`~qilisdk.analog.schedule.Schedule`: A system for defining time-dependent evolution of Hamiltonians.

This module is ideal for expressing analog quantum simulations, such as adiabatic time-evolutions or Hamiltonian-based computation.

Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class represents an analytic Hamiltonian as a sum of weighted Pauli operators. You can combine single- and multi-qubit operators using Python arithmetic.

**Common Pauli operators**:

- ``X(i)`` - Pauli_X acting on qubit *i*  
- ``Y(i)`` - Pauli_Y acting on qubit *i*  
- ``Z(i)`` - Pauli_Z acting on qubit *i*  

**Arithmetic operations**:

- Addition: ``H1 + H2``  
- Scalar multiplication: ``alpha * H``  
- Operator multiplication: ``Z(i) * Z(j)``  

Example: Ising Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^

To define an Ising Hamiltonian of the form:

.. math::

    H_{\text{Ising}}  =  - \sum_{\langle i, j \rangle} J_{ij} \sigma^Z_i \sigma^Z_j - \sum_j h_j \sigma^Z_j

you can use the Pauli ``Z`` operators from the library:

.. code-block:: python

    from qilisdk.analog import Z, X, Y

    nqubits = 3
    J = [
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
    ]
    h = [1, 2, 3]

    # Pairwise ZZ couplings
    coupling = sum(
        J[i][j] * Z(i) * Z(j)
        for i in range(nqubits) for j in range(i + 1, nqubits)
    )

    # Local fields
    fields = sum(h[j] * Z(j) for j in range(nqubits))

    # Ising Hamiltonian
    H = -coupling - fields

    print(H)

**Output:**

::

    -2 Z(0) Z(1) - 3 Z(0) Z(2) - 4 Z(1) Z(2) - Z(0) - 2 Z(1) - 3 Z(2)

This defines a fully connected 3-qubit Ising model using symbolic operators.

Schedule
--------

The :class:`~qilisdk.analog.schedule.Schedule` class defines how one or more Hamiltonians evolve over time. You specify:

- **total_time** (float): Duration of the evolution.  
- **time_step** (float): Time increment between discrete steps.  
- **hamiltonians** (dict[str, Hamiltonian]): Map of labels to Hamiltonian instances.  
- **schedule_map** (dict[int, dict[str, float]]): For each time-step index, map labels to coefficients.


Example 1: Dictionary-Based Schedule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule
    from qilisdk.analog.hamiltonian import X, Z

    dt = 0.1
    T = 1
    steps = np.linspace(0, T, int(T / dt))

    # Define two Hamiltonians
    h1 = X(0) + X(1) + X(2)
    h2 = -1 * Z(0) - 1 * Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

    schedule = Schedule(
        total_time=T,
        time_step=dt,
        hamiltonians={"h1": h1, "h2": h2},
        schedule={
            t: {
                "h1": 1 - steps[t] / T,
                "h2": steps[t] / T
            } for t in range(len(steps))
        },
    )

Example 2: Functional Schedule with :meth:`~qilisdk.analog.schedule.Schedule.add_hamiltonian`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, You can add Hamiltonians one at a time, supplying a callable for the coefficient:

.. code-block:: python

    schedule = Schedule(T, dt)

    # Add h1 with a time‐dependent coefficient function
    schedule.add_hamiltonian(
        label="h1", 
        hamiltonian=h1, 
        schedule=lambda t: 1 - steps[t] / T
    )

    # Add h2 similarly
    schedule.add_hamiltonian(
        label="h2", 
        hamiltonian=h2, 
        schedule=lambda t: steps[t] / T
    )

This provides more flexibility and modularity for dynamic or conditional evolution.

Modifying a Schedule
^^^^^^^^^^^^^^^^^^^^

Once constructed, you can refine or extend the schedule:

**Add a new time step:**

.. code-block:: python

    schedule.add_schedule_step(time_step=11, {
        "h1": 0.3
    })

**Update  an existing coefficient:**

.. code-block:: python

    schedule.update_hamiltonian_coefficient_at_time_step(
        time_step=1, 
        hamiltonian_label="h1", 
        new_coefficient=0.2
    )

This lets you insert or override coefficients without rebuilding the full map.
