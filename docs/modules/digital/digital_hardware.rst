HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.digital.ansatz.HardwareEfficientAnsatz` is a hardware-efficient ansatz tailored to quantum device topologies. Configuration options:


- **layers**: Number of repeating layers of gates.
- **connectivity**:

  - ``circular``: Qubits form a ring.
  - ``linear``: Qubits are connected linearly.
  - ``full``: All-to-all connectivity.
  - Or a list of tuples explicitly specifying the connectivity.
- **one_qubit_gate**: Choose the parameterized single-qubit gate (e.g., :class:`~qilisdk.digital.gates.U1`, :class:`~qilisdk.digital.gates.U2`, :class:`~qilisdk.digital.gates.U3`).
- **two_qubit_gate**: Choose the two-qubit interaction type (e.g., :class:`~qilisdk.digital.gates.CNOT`, :class:`~qilisdk.digital.gates.CZ`).
- **structure**:

  - ``grouped``: Applies all single-qubit gates first, followed by all two-qubit gates.
  - ``interposed``: Interleaves single and two-qubit gates.


**Example**

.. code-block:: python

    from qilisdk.digital.ansatz import HardwareEfficientAnsatz
    from qilisdk.digital import U3, CNOT

    
    ansatz = HardwareEfficientAnsatz(
        nqubits=4, 
        layers=3, 
        connectivity="Circular", 
        one_qubit_gate=U3, 
        two_qubit_gate=CNOT, 
        structure="Interposed"
    )
    ansatz.draw()

  
This ansatz can then be used as a circuit. For example, we can execute this ansatz using QuTiP backend (need to be installed separately):


.. code-block:: python 

    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    backend = QutipBackend()

    backend.execute(DigitalPropagation(ansatz), readout=Readout().with_sampling(nshots=1000))
