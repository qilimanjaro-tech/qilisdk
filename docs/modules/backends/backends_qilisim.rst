QiliSim
----------------

The **QiliSim** backend is a CPU-based simulator developed by Qilimanjaro and written in C++, providing
efficient simulation of both digital and analog quantum functionals.
It is designed for ease of use and does not require any special hardware or dependencies.
There is no need to install QiliSim separately, as it is included with the core QILISDK installation.

Example
==================

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

Configuration
================

QiliSim has a variety of configuration options to customize the simulation methods and performance characteristics.
These can be set at initialization via the ``analog_simulation_method``, ``digital_simulation_method``, and ``execution_config`` parameters:

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.backends import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.integrator(),
        digital_simulation_method=DigitalMethod.statevector(),
        execution_config=ExecutionConfig(
            num_threads=4, 
            seed=42, 
            monte_carlo=MonteCarloConfig(trajectories=200)
        ),
    )

Possible analog simulation methods:

- :class:`~qilisdk.backends.backend_config.AnalogMethod.integrator`: General-purpose RK4 integrator.
- :class:`~qilisdk.backends.backend_config.AnalogMethod.adaptive_integrator`: Adaptive RK45 integrator, faster for some systems.
- :class:`~qilisdk.backends.backend_config.AnalogMethod.direct`: Matrix exponential method for small systems.
- :class:`~qilisdk.backends.backend_config.AnalogMethod.arnoldi`: Krylov subspace method.

Possible digital simulation methods:

- :class:`~qilisdk.backends.backend_config.DigitalMethod.statevector`: Exact state-vector simulation with optional caching.

