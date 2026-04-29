Ansatz
------

The :mod:`~qilisdk.digital.ansatz` submodule provides ready-to-use circuit templates (Ansätze) and a lightweight framework for writing your own. To author a custom template:

1. Subclass :class:`~qilisdk.digital.ansatz.Ansatz` (which already inherits from :class:`~qilisdk.digital.circuit.Circuit`).
2. Call ``super().__init__(nqubits=...)`` inside ``__init__`` to set the circuit width.
3. Add gates in any order with ``self.add(gate)``, loops are fine, and you can keep references to any :class:`~qilisdk.core.variables.Parameter` objects you want to expose later.

**Example**

.. code-block:: python

    from qilisdk.core.variables import Parameter
    from qilisdk.digital import H, RX, CZ
    from qilisdk.digital.ansatz import Ansatz

    class NewAnsatz(Ansatz):
        def __init__(self, nqubits: int, beta: float):
            super().__init__(nqubits=nqubits)
            self.beta = Parameter("beta", value=beta)

            # Layer 1: put each qubit in superposition
            for q in range(self.nqubits):
                self.add(H(q))

            # Layer 2: simple linear entangler
            for q in range(self.nqubits - 1):
                self.add(CZ(q, q + 1))

            # Layer 3: parameterized mixer
            for q in range(self.nqubits):
                self.add(RX(q, theta=self.beta))

Once defined, the subclass behaves exactly like any other circuit: you can draw it, bind parameters, or hand it off to backends.

For a prebuilt option, consider any of:

   - :class:`~qilisdk.digital.ansatz.QAOA` for combinatorial optimization problems.
   - :class:`~qilisdk.digital.ansatz.TrotterizedSchedule` for simulating analog dynamics with Trotterization.
   - :class:`~qilisdk.digital.ansatz.HardwareEfficientAnsatz` for a simple layered structure that can be adapted to various hardware topologies.

