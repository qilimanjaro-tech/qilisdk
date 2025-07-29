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

The core functionality includes a set of basic quantum gates:

.. list-table::
   :class: longtable
   :header-rows: 1
   :widths: 20 50

   * - Signature
     - Description
   * - ``X(qubit: int)``
     - Pauli-X gate; performs a bit-flip on the given qubit.
   * - ``Y(qubit: int)``
     - Pauli-Y gate; applies a bit-and-phase flip.
   * - ``Z(qubit: int)``
     - Pauli-Z gate; applies a phase flip.
   * - ``H(qubit: int)``
     - Hadamard gate; places the qubit in a superposition state.
   * - ``S(qubit: int)``
     - S gate; applies a π/2 rotation around the Z-axis.
   * - ``T(qubit: int)``
     - T gate; applies a π/4 rotation around the Z-axis.
   * - ``RX(qubit: int, theta: float)``
     - Rotation around the X-axis by angle `theta` on the Bloch sphere.
   * - ``RY(qubit: int, theta: float)``
     - Rotation around the Y-axis by angle `theta`.
   * - ``RZ(qubit: int, phi: float)``
     - Rotation around the Z-axis by angle `phi`.
   * - ``U1(qubit: int, *, phi: float)``
     - Equivalent to a phase shift and RZ: ``U1(phi) = exp(i*phi/2) RZ(phi)``.
   * - ``U2(qubit: int, *, phi: float, gamma: float)``
     - Combines Z rotations with a π/2 Y-rotation: ``U2(phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(π/2) RZ(gamma)``.
   * - ``U3(qubit: int, *, theta: float, phi: float, gamma: float)``
     - General single-qubit unitary: ``U3(theta, phi, gamma) = exp(i*(phi+gamma)/2) RZ(phi) RY(theta) RZ(gamma)``.
   * - ``CNOT(control: int, target: int)``
     - Controlled-X gate; flips the target qubit if the control qubit is 1.
   * - ``CZ(control: int, target: int)``
     - Controlled-Z gate; applies a Z rotation to the target if the control qubit is 1.

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

.. code-block:: python

    from qilisdk.digital import Circuit, H, X, CNOT

    # Create a circuit with 2 qubits
    circuit = Circuit(2)
    circuit.add(H(0))         # Apply Hadamard on qubit 0
    circuit.add(X(0))         # Apply X gate on qubit 0
    circuit.add(CNOT(0, 1))   # Apply CNOT between qubit 0 and 1

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

1. :class:`~qilisdk.digital.ansatz.HardwareEfficientAnsatz` 

Builds a hardware-efficient ansatz tailored to quantum device topologies. Configuration options:


- **layers**: Number of repeating layers of gates.
- **connectivity**:
  - ``Circular``: Qubits form a ring.
  - ``Linear``: Qubits are connected linearly.
  - ``Full``: All-to-all connectivity.
- **on_qubit_gates**: Choose one or more parameterized single-qubit gates (e.g., :class:`~qilisdk.digital.gates.U1`, :class:`~qilisdk.digital.gates.U2`, :class:`~qilisdk.digital.gates.U3`).
- **two_qubit_gates**: Choose the two-qubit interaction type (e.g., :class:`~qilisdk.digital.gates.CNOT`, :class:`~qilisdk.digital.gates.CZ`).
- **structure**:
  - ``grouped``: Applies all single-qubit gates first, followed by all two-qubit gates.
  - ``interposed``: Interleaves single and two-qubit gates.

