Overview
===========

The :mod:`~qilisdk.functionals` module provides high-level quantum execution procedures by combining tools from the
:mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.core` modules. Currently, it includes the following execution functionals:

- :doc:`functionals_digital`: Propagates a digital quantum circuit through the backend.
- :doc:`functionals_analog`: Simulates analog time evolution of one or more Hamiltonians according to a time-dependent schedule.
- :doc:`functionals_reservoirs`: Runs a quantum reservoir pipeline (pre-processing, reservoir dynamics, post-processing) across multiple input layers.

Moreover, it provides more complex functionals that are used to execute more complex algorithms:

- :doc:`functionals_variational`: Builds parameterized program to be optimized in a hybrid quantum-classical environment.

Architecture Overview
---------------------

Every functional conforms to the abstract :class:`~qilisdk.functionals.functional.Functional` interface. Primitive
functionals such as :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` and
:class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` also inherit from
:class:`~qilisdk.functionals.functional.PrimitiveFunctional`, which mixes in the
:class:`~qilisdk.core.parameterizable.Parameterizable` contract. This lets backends query and update symbolic
parameters consistently before execution.

Readout is decoupled from functionals: measurement details (shots, observables, state tomography) are specified via
:mod:`~qilisdk.readout` objects passed to the backend's :meth:`~qilisdk.backends.backend.Backend.execute` method.
All primitive functionals return a unified :class:`~qilisdk.functionals.functional_result.FunctionalResult`.

Result Objects
--------------

* :class:`~qilisdk.functionals.functional_result.FunctionalResult`
    The unified result type for all primitive functionals. Access results through:
    ``samples`` for shot counts, ``probabilities`` for measurement probabilities,
    ``state`` for the terminal :class:`~qilisdk.core.qtensor.QTensor` state (when using
    :meth:`~qilisdk.readout.readout_spec.Readout.with_state_tomography`), and ``expectation_values`` for
    expectation values (when using :meth:`~qilisdk.readout.readout_spec.Readout.with_expectation`).
    When ``store_intermediate_results=True``, intermediate results are available via
    ``intermediate_states``, ``intermediate_samples``, ``intermediate_probabilities``, and ``intermediate_expectation_values``.
* :class:`~qilisdk.functionals.variational_program_result.VariationalProgramResult`
    bundles the optimizer trajectory (optimal cost, parameters, intermediate steps) together with the functional result
    obtained at convergence.

These objects make post-processing workflows ergonomic. For example, after a digital propagation you can surface the
most likely bitstrings:

.. code-block:: python

    from qilisdk.digital import Circuit
    from qilisdk.backends import QiliSim
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    backend = QiliSim()
    circuit = Circuit(2)
    result = backend.execute(DigitalPropagation(circuit), Readout().with_sampling(nshots=1_000))
    print("Most likely outcomes:", result.get_probabilities())






