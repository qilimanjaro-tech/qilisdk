Digital
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
- :class:`RX(qubit: int, theta: float)<qilisdk.digital.gates.RX>`
  Rotation by angle `theta` around X.  
- :class:`RY(qubit: int, theta: float)<qilisdk.digital.gates.RY>`
  Rotation by angle `theta` around Y.  
- :class:`RZ(qubit: int, phi: float)<qilisdk.digital.gates.RZ>`
  Rotation by angle `phi` around Z.  
- :class:`U1(qubit: int, *, phi: float)<qilisdk.digital.gates.U1>`
  Phase shift equivalent to RZ plus global phase.  
- :class:`U2(qubit: int, *, phi: float, gamma: float)<qilisdk.digital.gates.U2>`
  π/2 Y-rotation sandwiched by Z-rotations.
- :class:`U3(qubit: int, *, theta: float, phi: float, gamma: float)<qilisdk.digital.gates.U3>`
  General single-qubit unitary: RZ-RY-RZ decomposition.
- :class:`SWAP(a: int, b: int)<qilisdk.digital.gates.SWAP>`
  Exchanges the states of qubits ``a`` and ``b``.
- :class:`CNOT(control: int, target: int)<qilisdk.digital.gates.CNOT>`
  Controlled-X: flips target if control is |1⟩.
- :class:`CZ(control: int, target: int)<qilisdk.digital.gates.CZ>`
  Controlled-Z: applies Z on target if control is |1⟩.
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
    circuit = Circuit(nqubits=3)

**Adding Gates**

.. code-block:: python

  from qilisdk.digital import H, CNOT

  circuit.add(H(0))  # Hadamard on qubit 0
  circuit.add(CNOT(0, 1))  # CNOT: control 0 → target 2
  circuit.draw()

Parameterized Circuits
^^^^^^^^^^^^^^^^^^^^^^

Circuits can include parameterized gates. Adding them is similar to regular gates:

.. code-block:: python

    from qilisdk.digital import RX
    import numpy as np

    circuit.add(RX(0, theta=np.pi))

You can retrieve the current parameter using:

.. code-block:: python

    print("Initial Parameters:", circuit.get_parameters())


**Output:**

::

    Initial Parameters: {'RX(0)_theta_0': 3.141592653589793}


You can also retrieve a list containing only the current parameter values:

.. code-block:: python

    print("Initial parameter values:", circuit.get_parameter_values())


**Output:**

::

    Initial parameter values: [3.141592653589793]


To update parameters by their keys:

.. code-block:: python

    circuit.set_parameters({"RX(0)_theta_0": 2 * np.pi})



To update all parameters with new values:

.. code-block:: python

    circuit.set_parameter_values([2 * np.pi])

.. warning::

    The order of parameters in the list passed to ``set_parameter_values`` must match the order in which the gates were added to the circuit.

Visualization
-------------

Use :meth:`~qilisdk.digital.circuit.Circuit.draw` to render a circuit with Matplotlib. 
By default, the renderer applies the library's built-in styling (including the bundled default font if available). 
You can **bypass all defaults** by passing a custom :class:`~qilisdk.utils.visualization.style.CircuitStyle`, which 
confines styling to the specific call without modifying global Matplotlib settings.

.. code-block:: python

    from qilisdk.digital import Circuit, H, CNOT

    circuit = Circuit(num_qubits=3)
    circuit.add(H(0))
    circuit.add(CNOT(0, 2))

    # Render in a window
    circuit.draw()


**Output**


.. figure:: ../_static/circuit.png
    :alt: Example digital circuit rendered with Circuit.draw
    :align: center

    Example circuit produced by ``circuit.draw()``.

Saving to a file
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Save as SVG (use .png, .pdf, etc. as needed)
    circuit.draw(filepath="my_circuit.svg")

Custom styling with :class:`~qilisdk.utils.visualization.style.CircuitStyle`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a style object to control theme, fonts, spacing, DPI, labels, and more. Passing this object to :meth:`~qilisdk.digital.circuit.Circuit.draw` overrides the library defaults for this call.
You can also change if the order of the draw follows the order they are added in or if it compacts the layers as much as possible by changing the parameter **layout** to *"normal"* (default) or *"compact"* respectively.

.. code-block:: python

    from qilisdk.digital import Circuit, H, CNOT
    from qilisdk.utils.visualization import CircuitStyle, light, dark

    circuit = Circuit(3)
    circuit.add(H(0))
    circuit.add(CNOT(0, 2))

    # Example 1: dark theme, larger text, higher DPI
    style_dark = CircuitStyle(
        theme=dark,
        dpi=200,
        fontsize=12,
        title="Teleportation (fragment)",
    )
    circuit.draw(style=style_dark)

    # Example 2: use a system font family and bypass the bundled font
    style_font = CircuitStyle(
        theme=light,
        fontfname=None,                         # do not force a specific TTF file
        fontfamily=["DejaVu Sans", "Arial"],    # fallback list
        fontsize=11,
    )
    circuit.draw(style=style_font, filepath="circuit_custom_font.png")

    # Example 3: adjust layout spacing
    compact = CircuitStyle(
        theme=dark,
        wire_sep=0.45,      # vertical distance between wires (inches)
        layer_sep=0.45,     # horizontal distance between layers (inches)
        gate_margin=0.10,   # side margin inside each layer cell (inches)
        label_pad=0.08,     # left padding for wire labels (inches)
        layout="compact",   # compresses the circuit whenever possible
        title="Compact layout",
    )
    circuit.draw(style=compact)

.. note::

    ``CircuitStyle`` fields map directly to the renderer's layout and font configuration. In particular, you can switch fonts in two ways:
    (1) provide a specific font file via ``fontfname="/path/to/MyFont.ttf"``; or
    (2) set ``fontfname=None`` and choose a family list with ``fontfamily=[...]`` to use system-resolved fonts. Both approaches only affect the current draw call.


Ansatz
------

The :mod:`~qilisdk.digital.ansatz` submodule provides ready-to-use circuit templates (Ansätze) and a lightweight framework for writing your own. To author a custom template:

1. Subclass :class:`~qilisdk.digital.ansatz.Ansatz` (which already inherits from :class:`~qilisdk.digital.circuit.Circuit`).
2. Call ``super().__init__(nqubits=...)`` inside ``__init__`` to set the circuit width.
3. Add gates in any order with ``self.add(gate)``—loops are fine, and you can keep references to any :class:`~qilisdk.core.variables.Parameter` objects you want to expose later.

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

Once defined, the subclass behaves exactly like any other circuit—you can draw it, bind parameters, or hand it off to backends.

For a prebuilt option, consider:

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

  
This ansatz can be then used as a circuit. For example, we can execute this ansatz using QuTiP backend (need to be installed separately):


.. code-block:: python 

    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import Sampling

    backend = QutipBackend()

    backend.execute(Sampling(ansatz))


Parameter Utilities
-------------------

Circuits collect the symbolic parameters contributed by each gate. Beyond the
quick examples above, you can query names, current values, and bounds, or
update them selectively:

.. code-block:: python

    import numpy as np

    from qilisdk.core.variables import Parameter
    from qilisdk.digital import Circuit, RX, RZ

    circuit = Circuit(nqubits=2)
    theta = Parameter("theta", value=np.pi / 4, bounds=(0.0, np.pi))

    circuit.add(RX(0, theta=theta))
    circuit.add(RZ(1, phi=np.pi / 2))

    print(circuit.get_parameter_names())   # ['RX(0)_theta_0', 'RZ(1)_phi_1']
    print(circuit.get_parameters())        # {'RX(0)_theta_0': 0.785..., 'RZ(1)_phi_1': 1.570...}
    print(circuit.get_parameter_bounds())  # {'RX(0)_theta_0': (0.0, 3.1415...), 'RZ(1)_phi_1': (None, None)}

    circuit.set_parameter_bounds({"RX(0)_theta_0": (0.1, np.pi / 2)})
    circuit.set_parameters({"RX(0)_theta_0": np.pi / 3})
    circuit.set_parameter_values([np.pi / 3, np.pi / 2])

These helpers make it straightforward to plug the circuit into classical
optimization loops while keeping parameter metadata synchronized.

Gate Modifiers and Measurement
------------------------------

Every base gate inherits convenience methods to produce derived operations
without manually instantiating :class:`~qilisdk.digital.gates.Controlled`,
:class:`~qilisdk.digital.gates.Adjoint`, or
:class:`~qilisdk.digital.gates.Exponential`. The measurement gate
:class:`~qilisdk.digital.gates.M` lets you add classical readout at the end of
the circuit.

.. code-block:: python

    import numpy as np

    from qilisdk.digital import Circuit, H, X, RX, M

    circuit = Circuit(2)
    circuit.add(H(0))

    # Promote a basic gate to a controlled version on the fly
    circuit.add(X(1).controlled(0))

    # Generate adjoint / exponential variants, preserving parameters
    circuit.add(RX(1, theta=np.pi / 4).adjoint())
    circuit.add(RX(0, theta=np.pi / 3).exponential())

    # Record measurement results for both qubits
    circuit.add(M(0, 1))

Controlled gates validate that control and target qubits are disjoint, and all wrapper gates forward parameter accessors to the underlying operation.

.. note::

   - The measurement gate cannot be controlled, conjugated, or exponentiated.
   - If a circuit ends without explicit measurements, the backend assumes all qubits are measured.
   - Measuring only a subset of qubits returns samples for that subset.
