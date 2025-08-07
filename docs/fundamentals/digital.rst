Digital Module
==============

The :mod:`~qilisdk.digital` module in the Qili SDK facilitates the construction and simulation of digital quantum systems. It is composed of the following primary components:

- :mod:`~qilisdk.digital.gates`: A collection of quantum gates and tools for gate construction.
- :class:`~qilisdk.digital.circuit.Circuit`: A class to create and manage digital quantum circuits.
- :mod:`~qilisdk.digital.ansatz`: Predefined circuits that can serve as Ansätze.

Gates
-----

This submodule provides the necessary components to define and manipulate quantum gates for use in digital quantum circuits.

Simple Gates
^^^^^^^^^^^^

Use these constructors to apply standard single- and two-qubit operations:

- ``X(qubit: int)``  
  Pauli X (bit-flip) on the specified qubit.  
- ``Y(qubit: int)``  
  Pauli Y (bit-and-phase-flip).  
- ``Z(qubit: int)``  
  Pauli Z (phase-flip).  
- ``H(qubit: int)``  
  Hadamard: creates superposition.  
- ``S(qubit: int)``  
  Phase gate (π/2 rotation about Z).  
- ``T(qubit: int)``  
  T gate (π/4 rotation about Z).  
- ``RX(qubit: int, theta: float)``  
  Rotation by angle `theta` around X.  
- ``RY(qubit: int, theta: float)``  
  Rotation by angle `theta` around Y.  
- ``RZ(qubit: int, phi: float)``  
  Rotation by angle `phi` around Z.  
- ``U1(qubit: int, *, phi: float)``  
  Phase shift equivalent to RZ plus global phase.  
- ``U2(qubit: int, *, phi: float, gamma: float)``  
  π/2 Y-rotation sandwiched by Z-rotations.  
- ``U3(qubit: int, *, theta: float, phi: float, gamma: float)``  
  General single-qubit unitary: RZ–RY–RZ decomposition.  
- ``CNOT(control: int, target: int)``  
  Controlled-X: flips target if control is |1⟩.  
- ``CZ(control: int, target: int)``  
  Controlled-Z: applies Z on target if control is |1⟩.

Controlled Gates
^^^^^^^^^^^^^^^^

Any basic gate can be turned into a controlled gate using the :class:`~qilisdk.digital.gates.Controlled` class:

.. code-block:: python

    from qilisdk.digital.gates import Controlled, Y

    controlled_y = Controlled(0, basic_gate=Y(1))

Adjoint Gates
^^^^^^^^^^^^^

You can create the Hermitian conjugate (dagger) of a gate using the :class:`~qilisdk.digital.gates.Adjoint` class:

.. code-block:: python

    from qilisdk.digital.gates import Adjoint, Y

    adjoint_y = Adjoint(basic_gate=Y(1))

Exponential Gates
^^^^^^^^^^^^^^^^^

To apply a gate as an exponential operator, use the :class:`~qilisdk.digital.gates.Exponential` class:

.. code-block:: python

    from qilisdk.digital.gates import Exponential, Y

    exp_y = Exponential(basic_gate=Y(1))

Circuits
--------

Quantum circuits can be built using the :class:`~qilisdk.digital.circuit.Circuit` class. You can sequentially add gates to define the circuit:

**Initialization**

.. code-block:: python

    from qilisdk.digital import Circuit

    # Create a 3-qubit circuit
    circuit = Circuit(num_qubits=3)

**Adding Gates**

.. code-block:: python

    from qilisdk.digital import H, CNOT

    circuit.add(H(0))         # Hadamard on qubit 0
    circuit.add(CNOT(0, 2))   # CNOT: control 0 → target 2

Parameterized Circuits
^^^^^^^^^^^^^^^^^^^^^^

Circuits can include parameterized gates. Adding them is similar to regular gates:

.. code-block:: python

    from qilisdk.digital import RX
    import numpy as np

    circuit.add(RX(0, theta=np.pi))

You can retrieve the current parameter values:

.. code-block:: python

    print("Initial parameters:", circuit.get_parameter_values())

**Output:**

::

    Initial parameters: [3.141592653589793]

To update parameter values:

.. code-block:: python

    circuit.set_parameter_values([2 * np.pi])

.. warning::

    The order of parameters in the list passed to ``set_parameter_values`` must match the order in which the gates were added to the circuit.

Ansatz
------

The :mod:`~qilisdk.digital.ansatz` submodule provides ready-to-use circuit templates (Ansätze). For example:

HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.digital.ansatz.HardwareEfficientAnsatz` is a hardware-efficient ansatz tailored to quantum device topologies. Configuration options:


- **layers**: Number of repeating layers of gates.
- **connectivity**:
  - ``Circular``: Qubits form a ring.
  - ``Linear``: Qubits are connected linearly.
  - ``Full``: All-to-all connectivity.
- **on_qubit_gates**: Choose the parameterized single-qubit gates (e.g., :class:`~qilisdk.digital.gates.U1`, :class:`~qilisdk.digital.gates.U2`, :class:`~qilisdk.digital.gates.U3`).
- **two_qubit_gates**: Choose the two-qubit interaction type (e.g., :class:`~qilisdk.digital.gates.CNOT`, :class:`~qilisdk.digital.gates.CZ`).
- **structure**:
  - ``grouped``: Applies all single-qubit gates first, followed by all two-qubit gates.
  - ``interposed``: Interleaves single and two-qubit gates.


**Example**

.. code-block:: python

    from qilisdk.digital.ansatz import HardwareEfficientAnsatz

    ansatz = HardwareEfficientAnsatz(
        num_qubits=4,
        layers=3,
        connectivity="circular",
        on_qubit_gates="U3",
        two_qubit_gates="CNOT",
        structure="interleaved"
    )
