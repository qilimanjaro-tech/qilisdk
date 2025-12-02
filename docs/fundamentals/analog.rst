Analog
=============

The :mod:`qilisdk.analog` layer lets you build symbolic Hamiltonians and time-dependent schedules that can be simulated or submitted to Qili backends. Its core types are :class:`~qilisdk.analog.hamiltonian.Hamiltonian`, :class:`~qilisdk.analog.schedule.Schedule`, and :class:`~qilisdk.analog.linear_schedule.LinearSchedule`.


- :class:`~qilisdk.analog.hamiltonian.Hamiltonian`: Tools to build arbitrary Hamiltonians using symbolic Pauli operators.
- :class:`~qilisdk.analog.schedule.Schedule`: A system for defining time-dependent evolution of Hamiltonians.

Quick Start
-----------

.. code-block:: python

    from qilisdk.analog import Hamiltonian, LinearSchedule, X, Z

    driver = sum(X(i) for i in range(2))
   problem = Z(0) * Z(1)

    schedule = LinearSchedule(
        T=10,
        dt=1,
        hamiltonians={"driver": driver, "problem": problem},
        schedule={
            0: {"driver": 1.0, "problem": 0.0},
            10: {"driver": 0.0, "problem": 1.0},
        },
    )


Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class represents a symbolic Hamiltonian as a sum of weighted Pauli operators. You can create Hamiltonians using the built-in Pauli operators and combine them with standard arithmetic operations.


**Common constructors**

- ``X(i)``, ``Y(i)``, ``Z(i)``, ``I(i)`` produces a Hamiltonian with a single-qubit Pauli.
- Construction using arithmetic operations. The operations follow Python syntax, for example: ``2 * Z(0) + Z(1)`` and ``Z(0) * Z(1)`` build multi-qubit Hamiltonian.

**Pauli operators**:

- :func:`X(i)<qilisdk.analog.hamiltonian.X>` - Pauli X acting on qubit *i*  
- :func:`Y(i)<qilisdk.analog.hamiltonian.Y>` - Pauli Y acting on qubit *i*
- :func:`Z(i)<qilisdk.analog.hamiltonian.Z>` - Pauli Z acting on qubit *i*
- :func:`I(i)<qilisdk.analog.hamiltonian.I>` - Identity acting on qubit *i*

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Schedule
--------

The :class:`~qilisdk.analog.schedule.Schedule` class maps time steps to :class:`~qilisdk.analog.hamiltonian.Hamiltonian` coefficients.

Parameters
~~~~~~~~~~

- **T** (float): total duration of the schedule in units of nano seconds.
- **dt** (int): resolution of time steps in units of nano seconds. Default is 1. 
- **hamiltonians** (dict[str, Hamiltonian]): Map of labels to :class:`~qilisdk.analog.hamiltonian.Hamiltonian` instances.  
- **schedule_map** (dict[int, dict[str, float]]): time-step index to Hamiltonian coefficient mapping. Each key is a time step index (0 to T/dt), and each value is a dictionary mapping Hamiltonian labels to their coefficients at that time step index.


Example 1: Dictionary-Based Schedule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z

    T, dt = 10, 1
    steps = np.linspace(0, T, int(T / dt))

    h1 = X(0) + X(1) + X(2)
    h2 = -Z(0) - Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

    schedule = Schedule(
        T=T,
        dt=dt,
        hamiltonians={"driver": h1, "problem": h2},
        schedule={
            i: {"driver": 1 - t / T, "problem": t / T}
            for i, t in enumerate(steps)
        },
    )
    schedule.draw()


Example 2: Functional Schedule with :meth:`~qilisdk.analog.schedule.Schedule.add_hamiltonian`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, You can add Hamiltonians one at a time, supplying a callable for the coefficient:

.. code-block:: python

    T, dt = 10, 1
    steps = np.linspace(0, T, int(T / dt))

    h1 = X(0) + X(1) + X(2)
    h2 = -Z(0) - Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)
    schedule = Schedule(T, dt)

    # Add h1 with a time‐dependent coefficient function
    schedule.add_hamiltonian(
        label="driver", 
        hamiltonian=h1, 
        schedule=lambda t: 1 - steps[t] / T
    )

    # Add h2 similarly
    schedule.add_hamiltonian(
        label="problem", 
        hamiltonian=h2, 
        schedule=lambda t: steps[t] / T
    )
    schedule.draw()

This provides more flexibility and modularity for dynamic or conditional evolution.

Note: if you add a Hamiltonian with the same label multiple times, the last one will overwrite the previous ones.

Modifying a Schedule
^^^^^^^^^^^^^^^^^^^^

Once constructed, you can refine or extend the schedule:

**Add a new time step:**

.. code-block:: python

    schedule.add_schedule_step(time_step=11, hamiltonian_coefficient_list={"h1": 0.3})

**Update  an existing coefficient:**

.. code-block:: python

    schedule.update_hamiltonian_coefficient_at_time_step(
        time_step=1, 
        hamiltonian_label="h1", 
        new_coefficient=0.2
    )

This lets you insert or override coefficients without rebuilding the full map.

Parameterized Schedules
^^^^^^^^^^^^^^^^^^^^^^^
:class:`~qilisdk.analog.schedule.Schedule` coefficients can be symbolic, enabling classical optimization loops or
experiments that scan over a family of time profiles. Coefficients can be
instances of :class:`~qilisdk.core.variables.Parameter` or algebraic
expressions (:class:`~qilisdk.core.variables.Term`) built from parameters.
The schedule tracks every parameter it encounters so you can query or set them
later.

.. code-block:: python

    from qilisdk.analog import Schedule, Z
    from qilisdk.core.variables import Parameter
    T, dt = 10, 1
    gamma = Parameter("gamma", value=0.5, bounds=(0.0, 1.0))
    schedule = Schedule(
        T=T,
        dt=dt,
        hamiltonians={"problem": Z(0)},
        schedule={0: {"problem": gamma}},
    )
    schedule.set_parameter_bounds({"gamma": (0.2, 0.8)})
    schedule.set_parameter_values([0.7])
    print(schedule.get_parameter_values())  # [0.7]


When coefficients arise from expressions (for example ``0.5 * gamma`` or
``gamma * (1 - t / T)`` inside :meth:`~qilisdk.analog.schedule.Schedule.add_hamiltonian`),
the resulting :class:`~qilisdk.core.variables.Term` is also recorded. The
helpers below give programmatic access to these symbolic coefficients:

- :meth:`~qilisdk.analog.schedule.Schedule.get_parameter_names` and :meth:`~qilisdk.analog.schedule.Schedule.get_parameters` surface labels and values.
- :meth:`set_parameters({"gamma": 0.6})<qilisdk.analog.schedule.Schedule.set_parameters>` updates selected entries in-place.
- :meth:`~qilisdk.analog.schedule.Schedule.get_parameter_bounds` returns the per-parameter bounds dictionary.

LinearSchedule
--------------
The :class:`~qilisdk.analog.linear_schedule.LinearSchedule` subclass provides
linear interpolation between explicitly defined time steps. You supply the
same inputs as :class:`~qilisdk.analog.schedule.Schedule`, and the class fills
in intermediate coefficients on demand.

.. code-block:: python

    from qilisdk.analog import LinearSchedule, X, Z

    gamma = Parameter("gamma", value=0.1, bounds=(0.0, 1.0))

    linear = LinearSchedule(
        T=10,
        dt=1,
        hamiltonians={
            "driver": X(0) + X(1),
            "problem": Z(0) * Z(1),
        },
        schedule={
            0: {"driver": 1.0, "problem": 0.0},
            9: {"driver": gamma, "problem": 1.0},
        },
    )

    mid = linear[5]
    print("Midpoint schedule:", mid)
    coeff = linear.get_coefficient(time_step=5, hamiltonian_key="driver")
    print("Driver coefficient at midpoint (evaluated parameters):", coeff)
    coeff = linear.get_coefficient_expression(time_step=5, hamiltonian_key="driver")
    print("Driver coefficient at midpoint (unevaluated parameters):", coeff)

**Output:**

::

    Midpoint schedule: 0.5 X(0) + 0.5 X(1) + 0.5555555555555556 Z(0) Z(1)
    Driver coefficient at midpoint (evaluated parameters): 0.5
    Driver coefficient at midpoint (unevaluated parameters): (0.4444444444444444) + (0.5555555555555556) * gamma


:class:`~qilisdk.analog.linear_schedule.LinearSchedule` preserves symbolic parameters when computing interpolation
results. Use :meth:`~qilisdk.analog.linear_schedule.LinearSchedule.get_coefficient_expression`
to obtain the unevaluated expression (with parameters intact) or
:meth:`~qilisdk.analog.linear_schedule.LinearSchedule.get_coefficient` to
retrieve the numeric value under the current parameter assignment. This makes
it convenient to visualize or export schedules with parametric sweeps while
reusing the same workflows as :class:`~qilisdk.analog.schedule.Schedule`.
