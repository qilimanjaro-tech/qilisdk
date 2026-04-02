
Quickstart
===========

Below are minimal examples to get you running digital circuits and analog evolutions.

Digital Circuit Example
~~~~~~~~~~~~~~~~~~~~~~~

Build a 2-qubit circuit, sample it, and inspect measurement counts:

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import QiliSim
    from qilisdk.functionals import Sampling

    # 1. Define a simple 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(1, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))

    # 2. Wrap it in a Sampling functional
    sampling = Sampling(circuit=circuit, nshots=500)

    # 3. Execute on GPU
    results = QiliSim().execute(sampling)

    print("Counts:", results.probabilities)

Analog Time Evolution Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define two Hamiltonians, build an interval-based interpolation schedule, and simulate:

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket, tensor_prod
    from qilisdk.functionals import TimeEvolution
    from qilisdk.backends import QiliSim

    # Total time and step
    T, dt = 5.0, 0.1

    # Hamiltonians H1 = ∑ X, H2 = ∑ Z
    n = 1
    H1 = sum(X(i) for i in range(n))
    H2 = sum(Z(i) for i in range(n))

    # Define coefficients over the full interval with automatic sampling
    schedule = Schedule(
        hamiltonians={"h1": H1, "h2": H2},
        coefficients={
            "h1": {(0.0, T): lambda t: 1 - t / T},
            "h2": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
    )

    # Initial state |+⟩
    psi0 = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(n)]).unit()

    # TimeEvolution functional
    tevo = TimeEvolution(
        schedule=schedule,
        initial_state=psi0,
        observables=[Z(0)],
        nshots=100,
        store_intermediate_results=True,
    )

    # Execute on CPU
    results = QiliSim().execute(tevo)
    print(results)



Next Steps
~~~~~~~~~~

Once you've confirmed everything works, explore the tutorials section or check out each module for more details:

- **Core primitives** (:doc:`/modules/core`) for state vectors, operators, and shared abstractions.
- **Digital workflows** (:doc:`/modules/digital`) covering circuit construction, parameter sweeps, and QASM export.
- **Analog workflows** (:doc:`/modules/analog`) for Hamiltonian builders, schedules, and time-evolution utilities.
- **Functionals** (:doc:`/modules/functionals`) to see how experiments connect models with execution backends.
- **Execution targets** (:doc:`/modules/backends`) to compare Qutip (CPU) and CUDA (CPU/GPU) runtimes.
- **Noise models** (:doc:`/modules/noise_models`) to simulate hardware effects in both digital and analog contexts.
- **Cost functions** (:doc:`/modules/cost_functions`) for common optimization objectives and metrics.
- **SpeQtrum cloud** (:doc:`/modules/speqtrum`) for account setup, calibration-aware jobs, and result retrieval.

Happy quantum coding!
