Qutip Backend
-------------

The **Qutip** backend uses the `QuTiP <https://qutip.org/>`_ library for CPU-only simulation. It is
the most lightweight optional backend — no GPU, no compiled extension — and is well suited to local
development, CI pipelines and educational notebooks.

Installation
============

.. code-block:: console

    pip install qilisdk[qutip]

Quick start
===========

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout

    T = 10.0
    dt = 0.5
    nqubits = 1

    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
        interpolation=Interpolation.LINEAR,
    )

    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

    backend = QutipBackend()
    results = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=initial_state),
        Readout().with_expectation(observables=[Z(0), X(0), Y(0)]).with_state_tomography(),
    )
    print(results)

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
     - Statevector simulation via ``qutip_qip``'s :class:`CircuitSimulator`. Mid-circuit measurements raise ``ValueError``; all measurements must be at the end of the circuit.
   * - :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`
     - |y|
     - Master-equation integration via QuTiP's :func:`qutip.mesolve`. ``nsteps`` (default ``10_000``) caps the internal ODE substeps.
   * - :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir`
     - |p|
     - The QutipBackend does not natively implement :meth:`Backend._execute_quantum_reservoir <qilisdk.backends.backend.Backend._execute_quantum_reservoir>`. Circuit steps in the reservoir layer fall back to dense ``QTensor`` unitary multiplication on CPU; ``Schedule`` steps still use QuTiP's :func:`mesolve`.
   * - :class:`~qilisdk.functionals.variational_program.VariationalProgram`
     - |y|
     - Reuses the digital/analog handlers above for each optimization step.

.. |y| unicode:: U+2705
.. |p| unicode:: U+1F7E1
.. |n| unicode:: U+274C

Configuration
=============

QutipBackend has a single constructor option:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Meaning
   * - ``nsteps``
     - Upper bound on the number of internal ODE substeps used by
       :func:`qutip.mesolve` per schedule interval. Increase it if a long or stiff schedule causes
       the QuTiP solver to fail with a "max steps exceeded" warning. Defaults to ``10_000``.

.. code-block:: python

    from qilisdk.backends import QutipBackend

    backend = QutipBackend(nsteps=50_000)

.. note::

   QutipBackend does **not** accept a :class:`~qilisdk.noise.NoiseModel`. For noisy digital or
   analog simulations use :class:`~qilisdk.backends.qilisim.QiliSim` or
   :class:`~qilisdk.backends.cuda_backend.CudaBackend` instead.
