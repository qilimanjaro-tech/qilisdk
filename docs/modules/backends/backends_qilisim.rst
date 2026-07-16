QiliSim
=======

.. image:: ../../_static/QiliSim_wht.svg
   :align: left
   :width: 200px
   :class: title-image only-dark

.. image:: ../../_static/QiliSim_blk.svg
   :align: left
   :width: 200px
   :class: title-image only-light

The **QiliSim** backend is the default CPU simulator developed by Qilimanjaro and written in C++.
It implements every primitive functional natively, supports a noise model on all execution paths,
and is included with the core ``qilisdk`` installation — no extra dependency or hardware is required.

Installation
------------

QiliSim is bundled with the core ``qilisdk`` installation, so no extra package is required.


Quick start
-----------

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, CNOT
    from qilisdk.backends import QiliSim
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # Build a simple circuit
    circuit = Circuit(5)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    # Create DigitalPropagation functional
    functional = DigitalPropagation(circuit)

    # Execute with the QiliSim backend
    backend = QiliSim()
    result = backend.execute(functional, Readout().with_sampling(nshots=500))
    print(result.get_samples())

Functional support
-------------------

QiliSim natively supports all primitive functionals through dedicated C++ routines:

.. list-table::
   :header-rows: 1
   :widths: 35 12 53

   * - Functional
     - Support
     - Notes
   * - :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`
     - |y|
     - Statevector-based simulation, configurable via :class:`~qilisdk.backends.backend_config.DigitalMethod`.
   * - :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`
     - |y|
     - Multiple integration schemes, configurable via :class:`~qilisdk.backends.backend_config.AnalogMethod`.
   * - :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir`
     - |y|
     - Native C++ implementation; supports both Circuit and Schedule reservoir steps.
   * - :class:`~qilisdk.functionals.variational_program.VariationalProgram`
     - |y|
     - Reuses the primitive functional handlers above for each optimization step.

.. |y| unicode:: U+2705
.. |p| unicode:: U+1F7E1
.. |n| unicode:: U+274C

Configuration
-------------

QiliSim is configured at construction time through three orthogonal sections, all defined in
:mod:`qilisdk.backends.backend_config`:

- :class:`~qilisdk.backends.backend_config.AnalogMethod` — chooses the analog time-evolution scheme
  and its hyperparameters.
- :class:`~qilisdk.backends.backend_config.DigitalMethod` — chooses the digital simulation strategy
  and its hyperparameters.
- :class:`~qilisdk.backends.backend_config.ExecutionConfig` — global execution controls (threads,
  random seed, Monte Carlo trajectories, measurement-collapse behaviour).

.. code-block:: python

    from qilisdk.backends import (
        AnalogMethod,
        DigitalMethod,
        ExecutionConfig,
        MonteCarloConfig,
        QiliSim,
    )

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.arnoldi(dim=16, num_substeps=2, matrix_free=True),
        digital_simulation_method=DigitalMethod.statevector(
            matrix_free=True,
            max_cache_size=2_000,
            combine_single_qubit_gates=True,
        ),
        execution_config=ExecutionConfig(
            num_threads=4,
            seed=42,
            monte_carlo=MonteCarloConfig(trajectories=200),
            measurement_collapse=False,
        ),
    )

If any argument is omitted, QiliSim falls back to:

- :meth:`AnalogMethod.integrator() <qilisdk.backends.backend_config.AnalogMethod.integrator>`
  (matrix-free RK4),
- :meth:`DigitalMethod.statevector() <qilisdk.backends.backend_config.DigitalMethod.statevector>`,
- a default :class:`~qilisdk.backends.backend_config.ExecutionConfig` (all cores, random seed,
  Monte Carlo disabled).

Analog simulation methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the classmethods of :class:`~qilisdk.backends.backend_config.AnalogMethod` to choose how the
schedule is integrated:

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Constructor
     - Underlying scheme
     - When to use it
   * - :meth:`AnalogMethod.integrator(matrix_free=True) <qilisdk.backends.backend_config.AnalogMethod.integrator>`
     - RK4
     - Default. Fixed-step Runge-Kutta 4. matrix-free is faster for sparse Hamiltonians.
   * - :meth:`AnalogMethod.variational_annealing() <qilisdk.backends.backend_config.AnalogMethod.variational_annealing>`
     - Variational annealing
     - Only available for X to Z annealing schedules. Can be used to simulate large (i.e. 60+ qubits) systems at the cost of accuracy. Set `order=1` for a quicker but less accurate simulation. Increasing warmups can also improve accuracy at the cost of time.
   * - :meth:`AnalogMethod.adaptive_integrator(tol=1e-2) <qilisdk.backends.backend_config.AnalogMethod.adaptive_integrator>`
     - Dormand-Prince RK4/5
     - Adaptive step size; ``tol`` bounds the fidelity error between the RK4 and RK5 estimates.
   * - :meth:`AnalogMethod.arnoldi(dim=10, num_substeps=1, matrix_free=True) <qilisdk.backends.backend_config.AnalogMethod.arnoldi>`
     - Krylov / Arnoldi
     - Can offer decent scaling for large sparse Hamiltonians; tune ``dim`` for the Krylov subspace size.
   * - :meth:`AnalogMethod.direct() <qilisdk.backends.backend_config.AnalogMethod.direct>`
     - Matrix exponential
     - Reference scheme for small systems; cost grows quickly with qubit count.

Example, using the adaptive integrator:

.. code-block:: python

    from qilisdk.backends import AnalogMethod, QiliSim

    backend = QiliSim(analog_simulation_method=AnalogMethod.adaptive_integrator(tol=1e-2))

Digital simulation methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are currently two digital simulation methods available. The first is a statevector simulator, which can be configured through the options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``matrix_free``
     - Apply gates directly to the statevector instead of building dense matrices. Default ``True``.
   * - ``max_cache_size``
     - Maximum number of precomputed gate matrices cached between executions.
   * - ``combine_single_qubit_gates``
     - Merge adjacent single-qubit gates into a single operation before propagating. Disabled if gate-specific noise is used.
   * - ``normalize_after_each_gate``
     - Renormalize the statevector after each gate to mitigate numerical drift, at a runtime cost.

.. code-block:: python

    from qilisdk.backends import DigitalMethod, QiliSim

    backend = QiliSim(
        digital_simulation_method=DigitalMethod.statevector(
            matrix_free=False,
            normalize_after_each_gate=True,
        ),
    )

The second is a stabilizer simulator, which can efficiently simulate circuits composed of Clifford gates and Pauli measurements,
at the cost of being slower if the circuit contains many non-Clifford gates. It has the following options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``max_states``
     - Maximum number of terms in the stabilizer sum. Increasing this can improve accuracy at the cost of memory and runtime.

.. code-block:: python

    from qilisdk.backends import DigitalMethod, QiliSim

    backend = QiliSim(
        digital_simulation_method=DigitalMethod.stabilizer(max_states=100),
    )

Execution and Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~qilisdk.backends.backend_config.ExecutionConfig` controls threading, randomness, and
optional Monte Carlo trajectory sampling for open-system simulations.

- ``num_threads=0`` (default) lets the simulator use every physical core.
- ``seed=None`` (default) draws a fresh random seed at construction time; pass an integer for
  reproducibility.
- ``monte_carlo=MonteCarloConfig(trajectories=N)`` enables stochastic trajectory sampling for
  noise models that admit a Monte Carlo unraveling; leave it ``None`` for deterministic master-equation
  evolution.
- ``measurement_collapse`` controls whether measurements collapse the statevector in place
  (relevant for mid-circuit measurement and reservoir computing); defaults to ``False``.

.. code-block:: python

    from qilisdk.backends import ExecutionConfig, MonteCarloConfig, QiliSim

    backend = QiliSim(
        execution_config=ExecutionConfig(
            num_threads=8,
            seed=42,
            monte_carlo=MonteCarloConfig(trajectories=500),
            measurement_collapse=False,
        ),
    )

Noise model support
-------------------

Any :class:`~qilisdk.noise.NoiseModel` accepted by the SDK can be passed directly to the constructor;
QiliSim applies it inside the C++ solver, so digital, analog, and reservoir runs all see the same
noise channels:

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.noise import NoiseModel, Depolarizing 

    nm = NoiseModel()
    nm.add(Depolarizing(probability=1e-3))

    backend = QiliSim(noise_model=nm)