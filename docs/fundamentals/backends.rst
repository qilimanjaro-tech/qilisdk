Backends
========

The :mod:`~qilisdk.backends` module provides concrete execution engines for running :mod:`~qilisdk.functionals` (quantum processes).  
Currently, two backends are supported:

- :class:`~qilisdk.backends.cuda_backend.CudaBackend`  
- :class:`~qilisdk.backends.qutip_backend.QutipBackend`  

.. NOTE::

    Backends are optional; to install one, include its extra when installing QILISDK:

    .. code-block:: console

        pip install qilisdk[<backend_name>]

    For more information check the :doc:`../getting_started/installation` page.

Once installed, any primitive functional can be executed by passing it to the backend's :meth:`~qilisdk.backends.backend.Backend.execute` method:

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.functionals import Sampling

    # ... build your circuit and Sampling functional ...
    results = CudaBackend().execute(sampling_functional)
    print(results)

Architecture Overview
---------------------

All concrete backends subclass :class:`~qilisdk.backends.backend.Backend`, which centralizes the execution workflow used
across the SDK. The :meth:`~qilisdk.backends.backend.Backend.execute` dispatches a primitive functional (e.g. :class:`~qilisdk.functionals.sampling.Sampling` or :class:`~qilisdk.functionals.time_evolution.TimeEvolution`)
to the appropriate simulation routine and returns the functional-specific result object (see the :doc:`Functionals
<functionals>` chapter). The Execute method is also used to optimize variational programs via repeated calls to 
the underlying parameterized primitive functional.

Backends register handlers for the functionals they support. If a functional is not implemented, :meth:`~qilisdk.backends.backend.Backend.execute` raises
``NotImplementedError`` to surface the mismatch early.

Hardware & Dependencies
-----------------------

Installing a backend extra pulls in the library stack required for that simulator. GPU backends also expect compatible
drivers to be present on the system.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Backend
     - Extra
     - Key dependency
     - Notes
   * - :class:`CudaBackend <qilisdk.backends.cuda_backend.CudaBackend>`
     - ``cuda``
     - `cuda-quantum <https://github.com/NVIDIA/cuda-quantum>`_
     - Requires NVIDIA hardware with recent drivers.
   * - :class:`QutipBackend <qilisdk.backends.qutip_backend.QutipBackend>`
     - ``qutip``
     - `QuTiP <https://qutip.org/>`_
     - CPU based; no special hardware needs.

Functional Support
------------------

The table below summarizes which primitive :mod:`~qilisdk.functionals` each backend can execute.

.. list-table::
   :header-rows: 1
   :widths: 35 20 20 25

   * - Backend
     - Sampling
     - TimeEvolution
     - VariationalProgram
   * - :class:`CudaBackend <qilisdk.backends.cuda_backend.CudaBackend>`
     - ✓
     - ✓
     - ✓
   * - :class:`QutipBackend <qilisdk.backends.qutip_backend.QutipBackend>`
     - ✓
     - ✓
     - ✓

CUDA Backend
------------

The **CUDA** backend leverages NVIDIA GPUs via the :mod:`cuda-quantum` framework for both digital and analog simulations.
When no compatible GPU is detected it automatically falls back to cpu-based targets, so you can prototype on
commodity hardware before moving to accelerated machines.

**Installation**

.. code-block:: console

    pip install qilisdk[cuda]

**Initialization**

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(
        sampling_method=CudaSamplingMethod.STATE_VECTOR
    )

**Capabilities**

- Digital circuits through :class:`~qilisdk.functionals.sampling.Sampling`.
- Analog dynamics for :class:`~qilisdk.functionals.time_evolution.TimeEvolution`, powered by ``cudaq.evolve``.
- Hybrid execution when paired with :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

**Sampling methods**

- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.STATE_VECTOR`: Full state-vector simulation (switches to CPU if a GPU is unavailable).  
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.TENSOR_NETWORK`: Tensor-network contraction, suited for shallow yet wide circuits.  
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.MATRIX_PRODUCT_STATE`: Matrix-product-state simulation for low-entanglement workloads.

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import Sampling

    # Build a simple circuit
    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 4))
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    # Create Sampling functional
    sampling = Sampling(circuit=circuit, nshots=500)

    # Execute with the chosen sampling method (GPU if available)
    cuda_backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    result = cuda_backend.execute(sampling)
    print(result.samples)

**Output**  

::

    {'11': 237, '00': 263}


Qutip Backend
-------------

The **Qutip** backend uses the :mod:`qutip` library for simulation on CPU, supporting both digital and analog functionals.
It is the most lightweight option, ideal for local development or environments without NVIDIA GPUs.

**Installation**

.. code-block:: console

    pip install qilisdk[qutip]

**Initialization**

.. code-block:: python

    from qilisdk.backends import QutipBackend

    backend = QutipBackend()

**Capabilities**

- **Sampling** of digital circuits via QuTiP's state-vector solvers.
- **TimeEvolution** driven by :class:`~qilisdk.analog.schedule.Schedule`.
- Compatible with :class:`~qilisdk.functionals.variational_program.VariationalProgram` for fully classical
  optimization loops.

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend, CudaBackend
    from qilisdk.functionals import TimeEvolution

    # Define total time and timestep
    T = 10.0
    dt = 0.1
    steps = np.linspace(0, T + dt, int(T / dt))
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time‑dependent schedule
    schedule = Schedule(T, dt)

    # Add hx with a time‐dependent coefficient function
    schedule.add_hamiltonian(label="hx", hamiltonian=Hx, schedule=lambda t: 1 - steps[t] / T)

    # Add hz similarly
    schedule.add_hamiltonian(label="hz", hamiltonian=Hz, schedule=lambda t: steps[t] / T)

    # Prepare an equal superposition initial state
    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

    # Create the TimeEvolution functional
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[Z(0), X(0), Y(0)],
        nshots=100,
        store_intermediate_results=False,
    )

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(time_evolution)
    print(results)

**Output**

:: 

    TimeEvolutionResult(
        final_expected_values=array([-0.99388223,  0.0467696 , -0.10005353]),
        final_state=QTensor(shape=2x1, nnz=2, format='csr')
        [[0.05506547-0.00516502j]
        [0.3364973 -0.94005887j]]
    )