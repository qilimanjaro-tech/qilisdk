Gates
-----

This submodule provides the necessary components to define and manipulate quantum gates for use in digital quantum circuits.

Simple Gates
^^^^^^^^^^^^

Use these constructors to apply standard single- and two-qubit operations:

- :class:`X(qubit: int)<qilisdk.digital.gates.X>`
  Pauli X (bit-flip) on the specified qubit.  
- :class:`Y(qubit: int)<qilisdk.digital.gates.Y>`
  Pauli Y (bit-and-phase-flip).  
- :class:`Z(qubit: int)<qilisdk.digital.gates.Z>`
  Pauli Z (phase-flip).  
- :class:`H(qubit: int)<qilisdk.digital.gates.H>`
  Hadamard: creates superposition.  
- :class:`I(qubit: int)<qilisdk.digital.gates.I>`
  Identity gate: leaves the qubit unchanged.  
- :class:`S(qubit: int)<qilisdk.digital.gates.S>`
  Phase gate (π/2 rotation about Z).  
- :class:`T(qubit: int)<qilisdk.digital.gates.T>`
  T gate (π/4 rotation about Z).  
- :class:`RX(qubit: int, theta: float | Parameter | Term)<qilisdk.digital.gates.RX>`
  Rotation by angle `theta` around X.  
- :class:`RY(qubit: int, theta: float | Parameter | Term)<qilisdk.digital.gates.RY>`
  Rotation by angle `theta` around Y.  
- :class:`RZ(qubit: int, phi: float | Parameter | Term)<qilisdk.digital.gates.RZ>`
  Rotation by angle `phi` around Z.  
- :class:`U1(qubit: int, *, phi: float | Parameter | Term)<qilisdk.digital.gates.U1>`
  Phase shift equivalent to RZ plus global phase.  
- :class:`U2(qubit: int, *, phi: float | Parameter | Term, gamma: float | Parameter | Term)<qilisdk.digital.gates.U2>`
  π/2 Y-rotation sandwiched by Z-rotations.
- :class:`U3(qubit: int, *, theta: float | Parameter | Term, phi: float | Parameter | Term, gamma: float | Parameter | Term)<qilisdk.digital.gates.U3>`
  General single-qubit unitary: RZ-RY-RZ decomposition.
- :class:`SWAP(a: int, b: int)<qilisdk.digital.gates.SWAP>`
  Exchanges the states of qubits ``a`` and ``b``.
- :class:`CNOT(control: int, target: int)<qilisdk.digital.gates.CNOT>`
  Controlled-X: flips target if control is 1.
- :class:`CZ(control: int, target: int)<qilisdk.digital.gates.CZ>`
  Controlled-Z: applies Z on target if control is 1.
- :class:`M(*qubits: int)<qilisdk.digital.gates.M>`
  Measures the listed qubits in the computational basis.

Controlled Gates
^^^^^^^^^^^^^^^^

Any basic gate can be turned into a controlled gate using the :class:`~qilisdk.digital.gates.Controlled` class:

.. code-block:: python

    from qilisdk.digital.gates import Controlled, Y

    controlled_y = Controlled(0, basic_gate=Y(1))
    multiple_controlled_y = Controlled(0, 1, basic_gate=Y(2))

Or alternatively, you can use the :meth:`.controlled()<qilisdk.digital.gates.BasicGate.controlled>` method on any gate instance:

.. code-block:: python

    from qilisdk.digital.gates import Y

    controlled_y = Y(1).controlled(0)
    multiple_controlled_y = Y(2).controlled(0, 1)


Adjoint Gates
^^^^^^^^^^^^^

You can create the Hermitian conjugate (dagger) of a gate either using the :class:`~qilisdk.digital.gates.Adjoint` class
or using the :meth:`.adjoint()<qilisdk.digital.gates.BasicGate.adjoint>` method on any gate instance:

.. code-block:: python

    from qilisdk.digital.gates import Adjoint, Y

    adjoint_y = Adjoint(basic_gate=Y(1))
    adjoint_y = Y(1).adjoint()

Exponential Gates
^^^^^^^^^^^^^^^^^

To apply a gate as an exponential operator, use either the :class:`~qilisdk.digital.gates.Exponential` class or 
the :meth:`.exponential()<qilisdk.digital.gates.BasicGate.exponential>` method on any gate instance:

.. code-block:: python

    from qilisdk.digital.gates import Exponential, Y

    exp_y = Exponential(basic_gate=Y(1))
    exp_y = Y(1).exponential()

