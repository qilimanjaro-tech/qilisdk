Full Quantum State
^^^^^^^^^^^^^^^^^^^^^^

Using :meth:`with_state_tomography()<qilisdk.readout.readout_spec.Readout.with_state_tomography>` instructs 
the backend to return the full quantum state vector (or density matrix) after execution.

.. code-block:: python

    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim
    from qilisdk.digital import Circuit
    from qilisdk.functionals import DigitalPropagation

    backend = QiliSim()
    functional = DigitalPropagation(Circuit(2))

    spec = Readout().with_state_tomography()
    result = backend.execute(functional, readout=spec)

    state = result.get_state()           # The full ket or density matrix as a QTensor
    probs = result.get_probabilities()   # A dict[str, float] derived from |amplitudes|²

**Parameters**

- **method** (``Literal["exact"]``, default ``"exact"``): Tomography method.  Currently only
  ``"exact"`` is supported, such that the backend returns the raw state vector.

**When to Use It**

Use state tomography when you need the complete quantum state for post-processing: computing custom
observables offline, visualising the state, checking fidelity, or seeding the next step of a
multi-step algorithm.

.. note::

    State tomography is only available on simulators.  Physical QPUs (accessed via
    :class:`~qilisdk.speqtrum.speqtrum.SpeQtrum`) do not support this readout type.

