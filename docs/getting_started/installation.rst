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

- **Quantum-as-a-Service (QaaS)** (cloud submission via :class:`~qilisdk.backends.qaas.QaaS`):

  .. code-block:: bash

      pip install qilisdk[qaas]

You can combine extras:

.. code-block:: bash

    pip install qilisdk[cuda,qutip,qaas]

Quickstart
----------

Below are minimal examples to get you running digital circuits and analog evolutions.

Digital Circuit Example
~~~~~~~~~~~~~~~~~~~~~~~

Build a 2‑qubit circuit, sample it, and inspect measurement counts:

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend
    from qilisdk.functionals import Sampling

    # 1. Define a simple 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(1, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))

    # 2. Wrap it in a Sampling functional
    sampling = Sampling(circuit=circuit, nshots=500)

    # 3. Execute on GPU
    backend = CudaBackend(sampling_method=CudaBackend.STATE_VECTOR)
    results = backend.execute(sampling)

    print("Counts:", results.probabilities)

Analog Time Evolution Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define two Hamiltonians, build a linear interpolation schedule, and run on Qutip:

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z
    from qilisdk.common import ket, tensor_prod
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
        total_time=T,
        time_step=dt,
        hamiltonians={"h1": H1, "h2": H2},
        schedule_map=sched_map,
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

Once you’ve confirmed everything works, explore:

- **Digital module** (:doc:`/digital`) for advanced gate sets, parameter sweeps, and QASM 2.0 serialization.
- **Analog module** (:doc:`/analog`) for custom Hamiltonians, functional schedules, and intermediate-state storage.  
- **Backends** (:doc:`/backends`) to compare CPU (Qutip), GPU (CUDA), or cloud (QaaS) performance.  
- **Extras & utilities**: VQE workflows, result handling, optimizer gallery, and serialization tools.

For in‑depth reference, see the module pages in this documentation. Happy quantum coding!
