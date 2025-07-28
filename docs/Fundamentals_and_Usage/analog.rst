Analog Module
=============

The :mod:`~qilisdk.analog` module in Qili SDK enables the construction and simulation of analog quantum systems. It is composed of the following core components:

- :class:`~qilisdk.analog.hamiltonian.Hamiltonian`: Tools to build arbitrary Hamiltonians using symbolic Pauli operators.
- :class:`~qilisdk.analog.schedule.Schedule`: A system for defining time-dependent evolution of Hamiltonians.

This module is ideal for expressing analog quantum simulations, such as adiabatic time-evolutions or Hamiltonian-based computation.

Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class allows you to construct symbolic representations of quantum Hamiltonians. These are expressed analytically using Pauli operators such as ``Z``, ``X``, and ``Y``.

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

    H = -1 * sum( 
            sum(
                J[i][j] * Z(i) * Z(j) 
                for i in range(nqubits)
            ) 
            for j in range(nqubits)
        ) - sum(h[j] * Z(j) for j in range(nqubits))

    print(H)

**Output:**

::

    -2 Z(0) Z(1) - 3 Z(0) Z(2) - 4 Z(1) Z(2) - Z(0) - 2 Z(1) - 3 Z(2)

This defines a fully connected 3-qubit Ising model using symbolic operators.

Schedule
--------

The :class:`~qilisdk.analog.schedule.Schedule` class represents the time evolution of a quantum system. It defines how different Hamiltonians evolve over discrete time steps using time-dependent coefficients.

You begin by defining your Hamiltonians and labeling them. Then, for each time step, you assign a coefficient to each Hamiltonian to define its contribution over time.

Example 1: Dictionary-Based Schedule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule
    from qilisdk.analog.hamiltonian import X, Z

    dt = 0.1
    T = 1
    steps = np.linspace(0, T, int(T / dt))

    h1 = X(0) + X(1) + X(2)
    h2 = -1 * Z(0) - 1 * Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

    schedule = Schedule(
        T,
        dt,
        hamiltonians={"h1": h1, "h2": h2},
        schedule={
            t: {
                "h1": 1 - steps[t] / T,
                "h2": steps[t] / T
            } for t in range(len(steps))
        },
    )

Example 2: Functional Schedule with :meth:`~qilisdk.analog.schedule.Schedule.add_hamiltonian`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can build the schedule incrementally using functional definitions for coefficients:

.. code-block:: python

    schedule = Schedule(T, dt)

    schedule.add_hamiltonian(
        label="h1", 
        hamiltonian=h1, 
        schedule=lambda t: 1 - steps[t] / T
    )

    schedule.add_hamiltonian(
        label="h2", 
        hamiltonian=h2, 
        schedule=lambda t: steps[t] / T
    )

This provides more flexibility and modularity for dynamic or conditional evolution.

Modifying a Schedule
^^^^^^^^^^^^^^^^^^^^

You can update or insert specific time steps using the provided methods.

**Add a new time step:**

.. code-block:: python

    schedule.add_schedule_step(time_step=11, {
        "h1": 0.3
    })

**Modify an existing coefficient:**

.. code-block:: python

    schedule.update_hamiltonian_coefficient_at_time_step(
        time_step=1, 
        hamiltonian_label="h1", 
        new_coefficient=0.2
    )

Summary
-------

The ``analog`` module in Qili SDK enables expressive and dynamic construction of analog quantum simulations. With flexible Hamiltonian composition and powerful scheduling capabilities, it supports a wide range of use cases from Ising models to time-dependent protocols.
