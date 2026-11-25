Overview
===========

The :mod:`~qilisdk.algorithms` module provides higher-level procedures built on top of the primitive
:mod:`~qilisdk.functionals`. Currently, it includes the following algorithms:

- :doc:`algorithms_variational`: Builds a parameterized program to be optimized in a hybrid quantum-classical environment.
- :doc:`algorithms_reservoirs`: Runs a quantum reservoir pipeline (pre-processing, reservoir dynamics, post-processing) across multiple input layers.

Architecture Overview
---------------------

Every algorithm inherits from :class:`~qilisdk.algorithms.algorithm.Algorithm`, which is itself a
:class:`~qilisdk.functionals.functional.Functional`, so algorithms are run with the same
:meth:`~qilisdk.backends.backend.Backend.execute` call as any other functional. Unlike the primitive functionals, which
every backend should support natively, an algorithm defines its own execution in terms of primitive functionals. This means any backend
supporting the required primitive functionals can run any algorithm. A backend may still provide its own native
implementation of an algorithm, as :class:`~qilisdk.backends.qilisim.QiliSim` does for quantum reservoirs.

Result Objects
--------------

* :class:`~qilisdk.functionals.functional_result.FunctionalResult`
    Returned by :class:`~qilisdk.algorithms.quantum_reservoirs.QuantumReservoir`, holding the readout after the last
    layer, with the readouts of the previous layers as intermediate results.
* :class:`~qilisdk.algorithms.variational_program_result.VariationalProgramResult`
    bundles the optimizer trajectory (optimal cost, parameters, intermediate steps) together with the functional result
    obtained at convergence.
