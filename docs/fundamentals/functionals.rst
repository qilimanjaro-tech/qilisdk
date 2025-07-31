Functionals
===========

The :mod:`~qilisdk.functionals` module provides high‑level quantum execution procedures by combining tools from the
:mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.common` modules. Currently, it includes two primary functionals:

- :class:`~qilisdk.functionals.sampling.Sampling` — Executes repeated sampling of a digital quantum circuit.
- :class:`~qilisdk.functionals.time_evolution.TimeEvolution` — Simulates analog time evolution of one or more Hamiltonians according to a time‑dependent schedule.

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
