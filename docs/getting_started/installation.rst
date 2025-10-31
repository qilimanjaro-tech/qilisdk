Installation
============

QiliSDK and its optional extras are distributed via PyPI. Install the core package first, then add any backend or feature modules you need.

**Base package**

.. code-block:: bash

    pip install qilisdk

**Optional extras**

- **CUDA acceleration** (NVIDIA GPU support with :mod:`~qilisdk.backends.cuda_backend.CudaBackend`):

  .. code-block:: bash

      pip install qilisdk[cuda]

- **Qutip CPU backend** (CPU simulation with :mod:`~qilisdk.backends.qutip_backend.QutipBackend`):

  .. code-block:: bash

      pip install qilisdk[qutip]

- **SpeQtrum** (cloud submission via :class:`~qilisdk.speqtrum`):

  .. code-block:: bash

      pip install qilisdk[speqtrum]

You can combine extras:

.. code-block:: bash

    pip install qilisdk[cuda,qutip,speqtrum]

Quickstart
----------

Below are minimal examples to get you running digital circuits and analog evolutions.

Digital Circuit Example
~~~~~~~~~~~~~~~~~~~~~~~

Build a 2-qubit circuit, sample it, and inspect measurement counts:

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import Sampling

    # 1. Define a simple 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(1, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))

    # 2. Wrap it in a Sampling functional
    sampling = Sampling(circuit=circuit, nshots=500)

    # 3. Execute on GPU
    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    results = backend.execute(sampling)

    print("Counts:", results.probabilities)

Analog Time Evolution Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define two Hamiltonians, build a linear interpolation schedule, and run on Qutip:

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket, tensor_prod
    from qilisdk.functionals import TimeEvolution
    from qilisdk.backends import QutipBackend

    # Total time and step
    T, dt = 5.0, 0.1
    times = np.linspace(0, T, int(T/dt) + 1)

    # Hamiltonians H1 = ∑ X, H2 = ∑ Z
    n = 1
    H1 = sum(X(i) for i in range(n))
    H2 = sum(Z(i) for i in range(n))

    # Linear schedule: start with H1, end with H2
    sched_map = {
        idx: {"h1": 1 - t/T, "h2": t/T}
        for idx, t in enumerate(times)
    }
    schedule = Schedule(
        T=T,
        dt=dt,
        hamiltonians={"h1": H1, "h2": H2},
        schedule=sched_map,
    )

    # Initial state |+⟩
    psi0 = tensor_prod([(ket(0) + ket(1)).unit() for _ in range(n)]).unit()

    # TimeEvolution functional
    tevo = TimeEvolution(
        schedule=schedule,
        initial_state=psi0,
        observables=[Z(0)],
        nshots=100,
        store_intermediate_results=True,
    )

    # Execute on CPU
    results = QutipBackend().execute(tevo)
    print(results)

Next Steps
----------

Once you've confirmed everything works, explore:

- **Core primitives** (:doc:`/fundamentals/core`) for state vectors, operators, and shared abstractions.
- **Digital workflows** (:doc:`/fundamentals/digital`) covering circuit construction, parameter sweeps, and QASM export.
- **Analog workflows** (:doc:`/fundamentals/analog`) for Hamiltonian builders, schedules, and time-evolution utilities.
- **Functionals** (:doc:`/fundamentals/functionals`) to see how experiments connect models with execution backends.
- **Execution targets** (:doc:`/fundamentals/backends`) to compare Qutip (CPU) and CUDA (CPU/GPU) runtimes.
- **SpeQtrum cloud** (:doc:`/fundamentals/speqtrum`) for account setup, calibration-aware jobs, and result retrieval.
- **Example gallery** (:doc:`/examples/code_examples`) for end-to-end notebooks you can adapt to your workflow.

Happy quantum coding!
