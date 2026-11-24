Overview
===========

The :mod:`qilisdk.utils` module collects a variety of tools that are useful in quantum workflows, but are otherwise
somewhat independent:

- :doc:`utils_classical_solvers` provides a common interface to brute force, SciPy, simulated annealing and SCIP, so that any :class:`~qilisdk.core.model.Model` can be solved classically for reference.
- :doc:`utils_interoperability` converts circuits and Hamiltonians to and from OpenQASM, QIR and OpenFermion.
- :doc:`utils_serialization` writes any QiliSDK object to YAML and reads it back, and provides the stable hash used throughout the SDK.

Unlike the other modules, :mod:`qilisdk.utils` has no top-level re-exports, i.e. each submodule is
imported by its full path:

.. code-block:: python

    from qilisdk.utils.classical_solvers import BruteForceSolver
    from qilisdk.utils.openfermion import openfermion_to_qilisdk
    from qilisdk.utils.serialization import serialize, deserialize