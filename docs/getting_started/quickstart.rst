
Quickstart
===========

Below are minimal examples to get you running digital circuits and analog evolutions.

Digital Circuit Example
~~~~~~~~~~~~~~~~~~~~~~~

Build a 2-qubit circuit, propagate it using CUDA backend, and inspect measurement counts:

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import SamplingReadout

    # 1. Define a simple 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(1, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))

    # 2. Wrap it in a DigitalPropagation functional
    propagation = DigitalPropagation(circuit=circuit)

    # 3. Execute on GPU with sampling readout
    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    results = backend.execute(propagation, readout=[SamplingReadout(nshots=500)])

    print("Counts:", results.final_probabilities)


.. NOTE::

    This example requires CUDA backend which can be installed using ``pip install qilisdk[cuda]`` if you have the proper drivers installed.
    for more details check the information on the official CUDA documentation: https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q

Analog Evolution Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define two Hamiltonians, build an interval-based interpolation schedule, and run using Qutip backend:

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket, tensor_prod
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.backends import QutipBackend
    from qilisdk.readout import ExpectationReadout, StateTomographyReadout

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

    # AnalogEvolution functional
    evolution = AnalogEvolution(
        schedule=schedule,
        initial_state=psi0,
        store_intermediate_results=True,
    )

    # Execute on CPU with expectation readout
    results = QutipBackend().execute(
        evolution,
        readout=[ExpectationReadout(observables=[Z(0)]), StateTomographyReadout()],
    )
    print(results)


.. NOTE::

    This example requires QuTiP backend which can be installed using ``pip install qilisdk[qutip]``.


Next Steps
~~~~~~~~~~

Once you've confirmed everything works, explore:

- **Core primitives** (:doc:`/fundamentals/core`) for state vectors, operators, and shared abstractions.
- **Digital workflows** (:doc:`/fundamentals/digital`) covering circuit construction, parameter sweeps, and QASM export.
- **Analog workflows** (:doc:`/fundamentals/analog`) for Hamiltonian builders, schedules, and time-evolution utilities.
- **Functionals** (:doc:`/fundamentals/functionals`) to see how experiments connect models with execution backends.
- **Execution targets** (:doc:`/fundamentals/backends`) to compare Qutip (CPU) and CUDA (CPU/GPU) runtimes.
- **SpeQtrum cloud** (:doc:`/fundamentals/speqtrum`) for account setup, calibration-aware jobs, and result retrieval.
- **Example gallery** (:doc:`/examples/code_examples`) for end-to-end notebooks you can adapt to your workflow.

Happy quantum coding!
