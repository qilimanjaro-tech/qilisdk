Qutip Backend
-------------

The **Qutip** backend uses the ``qutip`` library for simulation on CPU, supporting both digital and analog functionals.
It is the most lightweight option, ideal for local development or environments without NVIDIA GPUs.

Installation
===============

.. code-block:: console

    pip install qilisdk[qutip]

Initialization
=================

.. code-block:: python

    from qilisdk.backends import QutipBackend

    backend = QutipBackend()

Capabilities
==============

- **DigitalPropagation** of digital circuits via QuTiP's state-vector solvers.
- **AnalogEvolution** driven by :class:`~qilisdk.analog.schedule.Schedule`.
- Compatible with :class:`~qilisdk.functionals.variational_program.VariationalProgram` for fully classical
  optimization loops.

Example
==========

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout

    # Define total time and timestep
    T = 10.0
    dt = 0.5
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time-dependent schedule
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
        interpolation=Interpolation.LINEAR,
    )

    # Prepare an equal superposition initial state
    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

    # Create the AnalogEvolution functional
    analog_evolution = AnalogEvolution(
        schedule=schedule,
        initial_state=initial_state,
    )

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(
        analog_evolution,
        Readout().with_expectation(observables=[Z(0), X(0), Y(0)]).with_state_tomography(),
    )
    print(results)

**Output**

::

    FunctionalResult(
        expectation_values=array([-0.99388223,  0.0467696 , -0.10005353]),
        final_state=QTensor(shape=2x1, nnz=2, format='csr')
        [[0.05506547-0.00516502j]
        [0.3364973 -0.94005887j]]
    )
