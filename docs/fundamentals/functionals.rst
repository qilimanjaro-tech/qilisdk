Functionals
===========

The :mod:`~qilisdk.functionals` module provides high‑level quantum execution procedures by combining tools from the
:mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.common` modules. Currently, it includes two primary functionals:

- :class:`~qilisdk.functionals.sampling.Sampling` — Executes repeated sampling of a digital quantum circuit.
- :class:`~qilisdk.functionals.time_evolution.TimeEvolution` — Simulates analog time evolution of one or more Hamiltonians according to a time-dependent schedule.
- :class:`~qilisdk.functionals.parameterized_program.ParameterizedProgram` — Builds a parameterized program to be optimized.

Sampling
--------

The **Sampling** functional runs a digital quantum circuit multiple times (shots) and aggregates the measurement outcomes.

**Parameters**

- **circuit** (:class:`~qilisdk.digital.circuit.Circuit`): Circuit to be sampled.
- **nshots** (int): Number of times to execute the circuit and collect measurement results.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import Sampling

    # Create a 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(0, theta=np.pi))
    circuit.add(CNOT(0, 1))

    # Initialize the Sampling functional with 100 shots
    sampling = Sampling(circuit=circuit, nshots=100)

    # Run on Qutip backend and retrieve counts
    backend = QutipBackend()
    results = backend.execute(sampling)
    print(results)

Time Evolution
--------------

The **TimeEvolution** functional simulates analog evolution of a quantum system under one or more Hamiltonians, following a specified time‑dependent schedule.

**Parameters**

- **schedule** (:class:`~qilisdk.analog.schedule.Schedule`): Defines total evolution time, time steps, Hamiltonians, and their time‑dependent coefficients.
- **initial_state** (:class:`~qilisdk.common.quantum_objects.QuantumObject`): Initial state of the system.
- **observables** (List[:class:`~qilisdk.analog.hamiltonian.Hamiltonian` or :class:`~qilisdk.analog.hamiltonian.PauliOperator`]): Operators to measure after evolution.
- **nshots** (int, optional): Number of repetitions for each observable measurement. Default is 1.
- **store_intermediate_results** (bool, optional): If True, records the state at each time step. Default is False.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.common import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import TimeEvolution

    # Define total time and timestep
    T = 10.0
    dt = 0.1
    times = np.arange(0, T + dt, dt)
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time‑dependent schedule
    schedule = Schedule(
        total_time=T,
        time_step=dt,
        hamiltonians={"hx": Hx, "hz": Hz},
        schedule_map={
            t: {"hx": 1.0 - t / T, "hz": t / T}
            for t in times
        },
    )

    # Prepare an equal superposition initial state
    initial_state = tensor_prod([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()

    # Create the TimeEvolution functional
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[Z(0), X(0), Y(0)],
        nshots=100,
        store_intermediate_results=True,
    )

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(time_evolution)
    print(results)


Parameterized Program
--------------

The **Parameterized Program** functional defines the components for a variational quantum algorithm. It takes in a 
parameterized Functional, an optimizer, and a model defining the cost function. Then using the 
:meth:`~qilisdk.backend.backend.optimize` of a backend, you can try to find the optimal parameters for the functional. 

**Parameters**

- **functional** (:class:`~qilisdk.functionals.functional.Functional`): A parameterized Functional to be optimized.
- **optimizer** (:class:`~qilisdk.optimizers.optimizer.Optimizer`): A QiliSDK optimizer, to be used in optimizing the Functional's parameters.
- **cost_model** (:class:`~qilisdk.common.model.Model`): A Model object to evaluate the cost of a given set of parameters. This model is the cost function used by the optimizer.

**Usage Example**

.. code-block:: python

    import numpy as np

    from qilisdk.backends import QutipBackend
    from qilisdk.common.model import Model, ObjectiveSense
    from qilisdk.common.variables import LEQ, BinaryVariable
    from qilisdk.digital import CNOT, U2, HardwareEfficientAnsatz
    from qilisdk.functionals import Sampling
    from qilisdk.functionals.parameterized_program import ParameterizedProgram
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer


    values = [2, 3, 7]
    weights = [1, 3, 3]
    max_weight = 4
    binary_var = [BinaryVariable(f"b{i}") for i in range(len(values))]

    model = Model("Knapsack")

    model.set_objective(sum(binary_var[i] * values[i] for i in range(len(values))), sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("max_weights", LEQ(sum(binary_var[i] * weights[i] for i in range(len(weights))), max_weight))


    n_qubits = 3
    ansatz = HardwareEfficientAnsatz(
        n_qubits=n_qubits, layers=3, connectivity="Linear", structure="grouped", one_qubit_gate="U2", two_qubit_gate="CNOT"
    )
    circuit = ansatz.get_circuit([np.random.uniform(0, np.pi) for _ in range(ansatz.nparameters)])

    optimizer = SciPyOptimizer(method="Powell")

    backend = QutipBackend()
    result = backend.optimize(ParameterizedProgram(functional=Sampling(circuit), optimizer=optimizer, cost_model=model))

    print(result)

**Output** 

::

    ParameterizedProgramResults(
    Optimal Cost = -8.888,
    Optimal Parameters=[1.278949303184732,
    3.292560971153471,
    0.40699926001530307,
    -0.017115910525090056,
    1.8692450641078109,
    4.309091104779081,
    1.2074169689932666,
    1.4646918684520598,
    3.933732474878453,
    2.7505186784884854,
    0.31519530209404434,
    1.5676334459989747,
    -0.11583781713519889,
    2.3767875857247778,
    0.46270242112310445,
    3.2901742389124866,
    2.852111307997826,
    2.809347306543785,
    4.460605406445712,
    0.05677964182801364,
    4.126369235231041,
    3.101452974992417,
    5.639854652770387,
    3.503170455662135],
    Intermediate Results=[])
    Optimal results=SamplingResult(
    nshots=1000,
    samples={'001': 56, '010': 2, '101': 941, '110': 1}
    ))