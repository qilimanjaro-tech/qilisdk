Analog
=============

The :mod:`qilisdk.analog` layer lets you build symbolic Hamiltonians and time-dependent schedules that can be simulated or submitted to Qili backends. Its core types are :class:`~qilisdk.analog.hamiltonian.Hamiltonian` and :class:`~qilisdk.analog.schedule.Schedule`.


- :class:`~qilisdk.analog.hamiltonian.Hamiltonian`: Tools to build arbitrary Hamiltonians using symbolic Pauli operators.
- :class:`~qilisdk.analog.schedule.Schedule`: Flexible time-dependent evolution with callable/interval coefficients and step/linear interpolation.

Quick Start
-----------

.. code-block:: python

    from qilisdk.analog import Hamiltonian, Schedule, X, Z

    driver = sum(X(i) for i in range(2))
    problem = Z(0) * Z(1)
    T = 10

    schedule = Schedule(
        hamiltonians={"driver": driver, "problem": problem},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=1.0,
    )


Hamiltonian
-----------

The :class:`~qilisdk.analog.hamiltonian.Hamiltonian` class represents a symbolic Hamiltonian as a sum of weighted Pauli operators. You can create Hamiltonians using the built-in Pauli operators and combine them with standard arithmetic operations.


**Common constructors**

- ``X(i)``, ``Y(i)``, ``Z(i)``, ``I(i)`` produces a Hamiltonian with a single-qubit Pauli.
- Construction using arithmetic operations. The operations follow Python syntax, for example: ``2 * Z(0) + Z(1)`` and ``Z(0) * Z(1)`` build multi-qubit Hamiltonian.

**Pauli operators**:

- ``X(i)`` - Pauli X acting on qubit *i*  
- ``Y(i)`` - Pauli Y acting on qubit *i*  
- ``Z(i)`` - Pauli Z acting on qubit *i*  
- ``I(i)`` - Identity acting on qubit *i*

**Arithmetic operations**:

- Addition: ``H1 + H2``  
- Scalar multiplication: ``5 * H``  
- multiplication: ``H0 * H1``  
- Subtraction: ``H1 - H2``
- Division by scalar: ``H / 5``
- Negation: ``-H``

**Extra Symbolic Operators**:

- commutator: ``H1.commutator(H2)``  
- anticommutator: ``H1.anticommutator(H2)``
- vector_norm: ``H.vector_norm()``
- forbenius_norm: ``H.frobenius_norm()``
- trace: ``H.trace()``


**Exporting Hamiltonians**:

- to matrix: ``H.to_matrix(nqubits)``
- to qtensor: ``H.to_qtensor(nqubits)``

**Importing Hamiltonians**:

- from qtensor: ``Hamiltonian.from_qtensor(qtensor)``
- from string: ``Hamiltonian.parse(hamiltonian_string)``

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

The :class:`~qilisdk.analog.schedule.Schedule` class maps time points to :class:`~qilisdk.analog.hamiltonian.Hamiltonian` coefficients. Coefficients can be numbers, parameters/terms, or callables of time, and you can define them at discrete points or over intervals that are sampled automatically.

Key arguments
~~~~~~~~~~~~~

- **dt** (float): resolution of time samples. Default is 0.1.
- **hamiltonians** (dict[str, Hamiltonian]): Map of labels to :class:`~qilisdk.analog.hamiltonian.Hamiltonian` instances.
- **coefficients** (dict[str, dict]): Mapping from Hamiltonian label to a time-definition dictionary. Each key is either a time point (float/parameter/term) or a 2-tuple defining an interval; each value can be:
    - the coefficient of the hamiltonian at that time
    - or callable returning a coefficient. This callable can take a parameter ``t`` that will be replaced by time. Moreover, any other parameters passed to this callable need to have a default value or have their value specified in the ``**kwargs``.
    Interpolation is stepwise or linear.
- **interpolation** (:class:`~qilisdk.analog.schedule.Interpolation`): ``LINEAR`` (default) or ``STEP`` behavior between provided points.
- **total_time** (float | Parameter | Term | None): Optional max time that rescales all time points while preserving relative positions.


.. note::

    When you place parameters on the time axis (e.g., as time points or interval endpoints), the schedule automatically
    constrains those parameters to stay between their neighboring time points so the ordering of the timeline remains valid.


Example 1: Callable coefficients with interval sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.analog.schedule import Interpolation

    h_driver = X(0) + X(1)
    h_problem = Z(0) * Z(1)

    schedule = Schedule(
        hamiltonians={"driver": h_driver, "problem": h_problem},
        coefficients={
            "driver": {(0.0, 10.0): lambda t: 1 - t / 10.0},
            "problem": {(0.0, 10.0): lambda t: t / 10.0},
        },
        dt=0.5,
        interpolation=Interpolation.LINEAR,
    )
    schedule.draw()

Example 2: Step interpolation and max-time rescaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qilisdk.analog import Schedule, Z
    from qilisdk.analog.schedule import Interpolation
    from qilisdk.utils.visualization.style import ScheduleStyle

    h = Z(0)
    schedule = Schedule(
        hamiltonians={"h": h},
        coefficients={"h": {0.0: 1.0, 5.0: 0.2}},
        dt=0.01,
        interpolation=Interpolation.STEP,
    )

    # Later, shorten the experiment to 3s without redefining points
    schedule.draw(ScheduleStyle(title="Before Time Scaling"))
    schedule.set_max_time(3.0)
    schedule.draw(ScheduleStyle(title="After Time Scaling"))
    print("Time grid:", schedule.tlist)
    print("Coeff at t=1.5:", schedule.coefficients["h"][1.5])

.. Note::

    The draw function samples 1/dt points from the schedule, therefore, for an increased resolution in the plot you can reduce dt.


Parameterized Schedules
^^^^^^^^^^^^^^^^^^^^^^^
Schedule coefficients can be symbolic, enabling classical optimization loops or
experiments that scan over a family of time profiles. Coefficients can be
instances of :class:`~qilisdk.core.variables.Parameter` or algebraic
expressions (:class:`~qilisdk.core.variables.Term`) built from parameters.
The schedule tracks every parameter it encounters so you can query or set them
later.

.. code-block:: python

    from qilisdk.analog import Schedule, Z
    from qilisdk.core import Parameter, GreaterThanOrEqual

    gamma = Parameter("gamma", value=0.5, bounds=(0.0, 1.0))
    T = Parameter("T", value=10.0, bounds=(1.0, 20.0))

    schedule = Schedule(
        hamiltonians={"problem": Z(0)},
        coefficients={"problem": {(0.0, T): lambda t: gamma * t}},
        dt=0.1,
        total_time=T,
    )

    schedule.get_parameters()


    schedule.set_parameters({"gamma": 0.7})
    print(schedule.get_parameter_names())   # ['gamma', 'T']
    print(schedule.get_constraints())       # [0 <= T]
