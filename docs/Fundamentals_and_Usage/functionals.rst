Functionals
======================

the :mod:`~qilisdk.functionals` module defines a set execution procedures that combines the tools defined in the :mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.common` modules.

The current supported functionals are:

- :mod:`~qilisdk.functionals.sampling`: for sampling quantum circuits.
- :mod:`~qilisdk.functionals.time_evolution`: for analog time evolution of a hamiltonian given a schedule. 

sampling
--------

This functional is used to sample a digital quantum circuit. It is defined using: 

- a :class:`~qilisdk.digital.circuit.Circuit` object
- and a number of shots (the number of samples to be taken from the circuits)

.. code-block:: python

    from qilisdk.digital import CNOT, RX, Circuit, H, M
    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import Sampling

    # Create a circuit with 2 qubits
    circuit = Circuit(2)
    circuit.add(H(0))  # Apply Hadamard on qubit 0
    circuit.add(RX(0, theta=np.pi))  # Apply RX rotation on qubit 0con
    circuit.add(CNOT(0, 1))  # Add a CNOT gate between qubit 0 and 1

    sampling_functional = Sampling(circuit, nshots=100)

Time Evolution 
--------------

This functional is used to define an analog time evolution of a set of :class:`~qilisdk.analog.hamiltonian.Hamiltonian`s and a :class:`~qilisdk.analog.schedule.Schedule`. 
This functional is defined using: 

- :class:`~qilisdk.analog.schedule.Schedule`: a schedule that defines the evolution of the system over time. 
- initial state (:class:`~qilisdk.common.quantum_objects.QuantumObject`): the initial state of the system.
- a list of observables (list of  :class:`~qilisdk.analog.hamiltonian.Hamiltonian` or  :class:`~qilisdk.analog.hamiltonian.PauliOperator`): a list of the observables to be measured at the end of the time evolution. 
- number of shots: the number of times each of the observables is measured after the evolution.

.. code-block:: python

    import numpy as np

    from qilisdk.analog import Schedule, X, Y, Z
    from qilisdk.backends import QutipBackend
    from qilisdk.common import ket, tensor_prod
    from qilisdk.functionals import TimeEvolution

    T = 10  # Total evolution time
    dt = 0.1  # Time-step
    steps = np.linspace(0, T, int(T / dt))
    nqubits = 1

    # Define two Hamiltonians for the simulation
    H1 = sum(X(i) for i in range(nqubits))
    H2 = sum(Z(i) for i in range(nqubits))

    # Create a schedule for the time evolution
    schedule = Schedule(
        T,
        dt,
        hamiltonians={"h1": H1, "h2": H2},
        schedule={
            t: {
                "h1": 1 - steps[t] / T,
                "h2": steps[t] / T
                } for t in range(len(steps))
            },
    )

    # Prepare an initial state (equal superposition)
    state = tensor_prod([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()

    # Perform time evolution on the CUDA backend with observables to monitor
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=state,
        observables=[Z(0), X(0), Y(0)],
        store_intermediate_results=True
    )
