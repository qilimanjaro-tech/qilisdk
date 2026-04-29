Analog Evolution
------------------

The :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` functional simulates analog evolution of a quantum system under one or more Hamiltonians, following a specified time-dependent schedule. Its parameter interface mirrors that of
:class:`~qilisdk.analog.schedule.Schedule`, making it straightforward to sweep control waveforms or pulse amplitudes from
classical optimizers.

Observables and shot counts are specified separately via readout objects passed to
:meth:`~qilisdk.backends.backend.Backend.execute`.

**Parameters**

- **schedule** (:class:`~qilisdk.analog.schedule.Schedule`): Defines total evolution time, time steps, Hamiltonians, and their time-dependent coefficients.
- **initial_state** (:class:`~qilisdk.core.qtensor.QTensor`): Initial state of the system.
- **store_intermediate_results** (bool, optional): If True, records the state at each time step. Default is False.

**Returns**

- :class:`~qilisdk.functionals.functional_result.FunctionalResult`: Inspect
  :attr:`expectation_values` for measured observables,
  :attr:`state` for the closing state, and
  :attr:`intermediate_states` and :attr:`intermediate_expectation_values` when ``store_intermediate_results`` is enabled.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import QTensor
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.backends import QiliSim, CudaBackend
    from qilisdk.functionals import AnalogEvolution

    # Define total time and timestep
    T = 10.0
    dt = 0.5
    nqubits = 1

    # Define Hamiltonians
    Hx = -sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Create the AnalogEvolution functional
    analog_evolution = AnalogEvolution(
        schedule=Schedule.linear(Hx, Hz, total_time=T, dt=dt),
        initial_state=QTensor.uniform(nqubits),
        store_intermediate_results=False,
    )

we can execute it on the QiliSim backend:

.. code-block:: python

    # Execute on QiliSim backend and inspect results
    backend = QiliSim()
    results = backend.execute(
        analog_evolution,
        Readout().with_expectation(observables=[Z(0), X(0), Y(0)]),
    )
    print(results)

**Output**

::

    - Functional Results: [

    Expectation Value Results: (
        expectation_values=[-0.99388223, 0.0467696, -0.10005353],
    )

    ]

