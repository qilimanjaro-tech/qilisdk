Backends
========

The :mod:`~qilisdk.backends` module provides concrete execution engines for running :mod:`~qilisdk.functionals` (quantum processes).  
Currently, two backends are supported:

- :class:`~qilisdk.backends.cuda_backend.CudaBackend`  
- :class:`~qilisdk.backends.qutip_backend.QutipBackend`  

Backends are optional; to install one, include its extra when installing QILISDK:

.. code-block:: console

    pip install qilisdk[<backend_name>]

Once installed, any functional can be executed by passing it to the backend’s :meth:`~qilisdk.backends.backend.Backend.execute` method:

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.functionals import Sampling

    # ... build your circuit and Sampling functional ...
    results = CudaBackend().execute(sampling_functional)
    print(results)

CUDA Backend
------------

The **CUDA** backend leverages NVIDIA GPUs via the :mod:`cudaq` framework for both digital and analog simulations.

**Installation**

.. code-block:: console

    pip install qilisdk[cuda]

**Initialization**

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(
        sampling_method=CudaSamplingMethod.STATE_VECTOR
    )

**Sampling methods**

- **STATE_VECTOR**: Pure state-vector simulation.  
- **TENSOR_NETWORK**: Tensor-network contraction.  
- **MATRIX_PRODUCT_STATE**: Matrix-product-state simulation.

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import Sampling

    # Build a simple circuit
    circuit = Circuit(1)
    circuit.add(H(0))
    circuit.add(RX(0, theta=np.pi / 4))

    # Create Sampling functional
    sampling = Sampling(circuit=circuit, nshots=500)

    # Execute on GPU with state‑vector method
    cuda_backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    counts = cuda_backend.execute(sampling)
    print(counts)

Qutip Backend
-------------

The **Qutip** backend uses the :mod:`qutip` library for simulation on CPU, supporting both digital and analog functionals.

**Installation**

.. code-block:: console

    pip install qilisdk[qutip]

**Initialization**

.. code-block:: python

    from qilisdk.backends import QutipBackend

    backend = QutipBackend()

The Qutip backend provides a single simulation method but works for:

- **Sampling** (digital circuits)  
- **TimeEvolution** (analog Hamiltonian schedules)

**Example**

.. code-block:: python

    from qilisdk.digital import Circuit, H
    from qilisdk.functionals import TimeEvolution
    from qilisdk.analog import Schedule, X, Z
    from qilisdk.common import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    import numpy as np

    # Define Hamiltonians and schedule
    T, dt = 5.0, 0.1
    times = np.arange(0, T + dt, dt)
    Hx = X(0)
    Hz = Z(0)
    schedule = Schedule(
        total_time=T,
        time_step=dt,
        hamiltonians={"hx": Hx, "hz": Hz},
        schedule_map={t: {"hx": 1 - t / T, "hz": t / T} for t in times},
    )

    # Prepare initial state
    initial = tensor_prod([(ket(0) + ket(1)).unit()])

    # Create TimeEvolution functional
    tevo = TimeEvolution(
        schedule=schedule,
        initial_state=initial,
        observables=[Z(0), X(0)],
        nshots=50,
        store_intermediate_results=False,
    )

    # Run on Qutip backend
    results = QutipBackend().execute(tevo)
    print(results)
