Expectation Values
^^^^^^^^^^^^^^^^^^^^^^^

Using :meth:`with_expectation()<qilisdk.readout.readout_spec.Readout.with_expectation>` instructs 
the backend to compute ``⟨ψ|O|ψ⟩`` for each observable in the list.

.. code-block:: python

    from qilisdk.analog import X, Y, Z
    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim
    from qilisdk.digital import Circuit
    from qilisdk.functionals import DigitalPropagation

    backend = QiliSim()
    functional = DigitalPropagation(Circuit(2))

    spec = Readout().with_expectation(observables=[Z(0), X(0), Y(0)], nshots=0)
    result = backend.execute(functional, readout=spec)

    evs = result.get_expectation_values()   # list[float], one entry per observable
    # e.g. [-0.994, 0.047, -0.100]

**Parameters**

- **observables** (``list[Hamiltonian | QTensor]``): The operators whose expectation values are
  requested.  Each entry can be a :class:`~qilisdk.analog.hamiltonian.Hamiltonian` expression (e.g. ``Z(0) + Z(1)``)
  or a :class:`~qilisdk.core.qtensor.QTensor` directly.
- **nshots** (``int``, default ``0``): Number of shots for stochastic estimation.  ``0`` uses the
  exact state-vector inner product, meaning no sampling noise, only available on simulators.

**When to Use It**

Use expectation values when you need a scalar energy or observable readout, for example in
variational algorithms (VQE, QAOA energy evaluation) or analog time-evolution studies.

