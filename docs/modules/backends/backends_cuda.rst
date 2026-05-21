CUDA Backend
------------

The **CUDA** backend leverages NVIDIA GPUs via the
`cuda-quantum <https://github.com/NVIDIA/cuda-quantum>`_ framework. When no compatible GPU is
detected it transparently falls back to a CPU target, so the same code prototyped on a laptop will
also run on an accelerated machine.

Installation
============

.. code-block:: console

    pip install qilisdk[cuda12]   # or [cuda13]

Quick start
===========

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 4))
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    result = backend.execute(DigitalPropagation(circuit), Readout().with_sampling(nshots=500))
    print(result.get_samples())

Functional support
==================

.. list-table::
   :header-rows: 1
   :widths: 35 12 53

   * - Functional
     - Support
     - Notes
   * - :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`
     - |y|
     - Native CUDA-Q kernel. Sampling method selected via :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod`. Intermediate measurements raise ``NotImplementedError``.
   * - :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`
     - |y|
     - Driven by ``cudaq.evolve`` on the ``dynamics`` target (always GPU-accelerated when available, independent of the digital sampling method).
   * - :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir`
     - |p|
     - The CudaBackend does not natively implement :meth:`Backend._execute_quantum_reservoir <qilisdk.backends.backend.Backend._execute_quantum_reservoir>`. Circuit steps inside the reservoir layer fall back to dense ``QTensor`` unitary multiplication on CPU; ``Schedule`` steps still use CUDA-Q's ``evolve``. Any attached noise model is ignored.
   * - :class:`~qilisdk.functionals.variational_program.VariationalProgram`
     - |y|
     - Reuses the digital/analog handlers above for each optimization step.

.. |y| unicode:: U+2705
.. |p| unicode:: U+1F7E1
.. |n| unicode:: U+274C

Configuration
=============

The CUDA backend exposes a single configuration parameter —
:class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod` — that selects the underlying CUDA-Q
target used for **digital** circuits. Analog evolution always runs on the ``dynamics`` target and
ignores this setting.

.. list-table::
   :header-rows: 1
   :widths: 25 60 15

   * - Method
     - CUDA-Q target (and fallback)
     - Default
   * - :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.STATE_VECTOR`
     - ``nvidia`` (GPU) when a GPU is available, otherwise ``qpp-cpu``. Precision matches :class:`~qilisdk.settings.Precision`.
     - |y|
   * - :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.TENSOR_NETWORK`
     - ``tensornet``. Good for shallow, wide circuits.
     -
   * - :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.MATRIX_PRODUCT_STATE`
     - ``tensornet-mps``. Good for low-entanglement, long circuits.
     -

Set the method at construction time:

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(sampling_method=CudaSamplingMethod.MATRIX_PRODUCT_STATE)

Noise model support
===================

A :class:`~qilisdk.noise.NoiseModel` can be passed to ``CudaBackend(noise_model=…)``:

- For :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`, qilisdk noise channels
  are translated into a ``cudaq.NoiseModel`` (Kraus channels for static / time-derived noise,
  parameter perturbations applied to the circuit). With noise enabled, only a single
  :class:`~qilisdk.readout.SamplingReadout` is supported.
- For :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`, Lindblad-compatible noise
  channels become CUDA-Q jump operators and Hamiltonian deltas fed to ``cudaq.evolve``.
- For :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir`, the fallback
  implementation drops the noise model (a warning is logged).

Example: a depolarising channel applied to every gate of a digital circuit:

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.noise import NoiseModel, Depolarizing

    backend = CudaBackend(
        sampling_method=CudaSamplingMethod.STATE_VECTOR,
        noise_model=NoiseModel(global_noise=[Depolarizing(probability=1e-3)]),
    )
