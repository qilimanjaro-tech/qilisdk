CUDA Backend
------------

The **CUDA** backend leverages NVIDIA GPUs via the :mod:`cuda-quantum` framework for both digital and analog simulations.
When no compatible GPU is detected it automatically falls back to cpu-based targets, so you can prototype on
commodity hardware before moving to accelerated machines.

Installation
================

.. code-block:: console

    pip install qilisdk[cuda]

Initialization
=================

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(
        sampling_method=CudaSamplingMethod.STATE_VECTOR
    )

Capabilities
================

- Digital circuits through :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`.
- Analog dynamics for :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`, powered by ``cudaq.evolve``.
- Hybrid execution when paired with :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

Sampling methods
==================

- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.STATE_VECTOR`: Full state-vector simulation (switches to CPU if a GPU is unavailable).
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.TENSOR_NETWORK`: Tensor-network contraction, suited for shallow yet wide circuits.
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.MATRIX_PRODUCT_STATE`: Matrix-product-state simulation for low-entanglement workloads.

Example
============

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # Build a simple circuit
    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 4))
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    # Create DigitalPropagation functional
    functional = DigitalPropagation(circuit)

    # Execute with the chosen sampling method (GPU if available)
    cuda_backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    result = cuda_backend.execute(functional, Readout().with_sampling(nshots=500))
    print(result.get_samples())

**Output**

::

    {'11': 237, '00': 263}

