OpenQASM Import and Export
--------------------------

QiliSDK can bridge :class:`~qilisdk.digital.Circuit` and both OpenQASM 2.0 and
OpenQASM 3.0 via the following entry-points in :mod:`qilisdk.utils.openqasm`:

**OpenQASM 2.0**

- :func:`~qilisdk.utils.openqasm.openqasm2.from_qasm2` — parse an OpenQASM 2.0 string into a Circuit.
- :func:`~qilisdk.utils.openqasm.openqasm2.from_qasm2_file` — read a ``.qasm`` file.
- :func:`~qilisdk.utils.openqasm.openqasm2.to_qasm2` — serialize a Circuit to an OpenQASM 2.0 string.
- :func:`~qilisdk.utils.openqasm.openqasm2.to_qasm2_file` — write a ``.qasm`` file.

**OpenQASM 3.0**

- :func:`~qilisdk.utils.openqasm.openqasm3.from_qasm3` — parse an OpenQASM 3.0 string into a Circuit.
- :func:`~qilisdk.utils.openqasm.openqasm3.from_qasm3_file` — read a ``.qasm`` file.
- :func:`~qilisdk.utils.openqasm.openqasm3.to_qasm3` — serialize a Circuit to an OpenQASM 3.0 string.
- :func:`~qilisdk.utils.openqasm.openqasm3.to_qasm3_file` — write a ``.qasm`` file.

The ``openqasm3`` dependency is optional; install it with the ``openqasm`` extra:

.. code-block:: bash

    pip install qilisdk[openqasm]

Quick start
^^^^^^^^^^^

.. code-block:: python

    from qilisdk.digital import CNOT, Circuit, H, M
    from qilisdk.utils.openqasm import to_qasm2, from_qasm2, to_qasm3, from_qasm3

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0, 1))

    # OpenQASM 2.0
    qasm2_text = to_qasm2(circuit)
    print(qasm2_text)
    reparsed = from_qasm2(qasm2_text)

    # OpenQASM 3.0
    qasm3_text = to_qasm3(circuit)
    print(qasm3_text)
    reparsed = from_qasm3(qasm3_text)

Reading and writing files:

.. code-block:: python

    from qilisdk.utils.openqasm import to_qasm2_file, from_qasm2_file
    from qilisdk.utils.openqasm import to_qasm3_file, from_qasm3_file

    to_qasm2_file(circuit, "bell.qasm")
    same_circuit = from_qasm2_file("bell.qasm")

    to_qasm3_file(circuit, "bell3.qasm")
    same_circuit = from_qasm3_file("bell3.qasm")

OpenQASM 3.0 supported features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ✅: The feature is fully supported
- 🟡: The feature is partially supported, see the note for explanation
- ❌: The feature is not supported

============================  ============  =====
OpenQASM 3 feature            Supported     Notes
============================  ============  =====
comments                      ✅
QASM version string           ✅
include                       ✅
unicode names                 ✅
qubit                         ✅
bit                           🟡            1
bool                          🟡            1
int                           🟡            1
uint                          🟡            1
float                         🟡            1
angle                         🟡            1
complex                       🟡            1
const                         ✅
pi/π/tau/τ/euler/ℇ            ✅
Aliasing: ``let``             ✅
register concatenation        ❌
casting (``expr.Cast``)       ✅
duration                      🟡            1
durationof                    ❌
ns/μs/us/ms/s/dt              ✅
stretch (``expr.Stretch``)    🟡            1
delay                         ❌
barrier                       ❌
box                           ❌
built-in ``U``                ✅
gate definition               🟡            1
gphase                        ❌
``ctrl @``                    ✅
``negctrl @``                 🟡            1
``inv @``                     ✅
``pow(k) @``                  🟡            1, 2
reset                         ❌
measure                       🟡            3
bit operations                🟡            1
boolean operations            🟡            1
arithmetic expressions        🟡            1
comparisons                   🟡            1
``if``                        🟡            1
``else``                      🟡            1
``else if``                   🟡            1
for loops                     🟡            1
switch                        🟡            1
while loops                   🟡            1
``continue``                  🟡            1
``break``                     🟡            1
extern                        ❌
def subroutines               🟡            1
return                        🟡            1
input                         🟡            1
output                        ❌
============================  ============  =====

1) Reading these constructs is fully supported, but the expressions are not
   stored in the :class:`~qilisdk.digital.Circuit` object and will not be
   written back out when converting to OpenQASM. For example, declaring
   ``int x = 5;`` causes ``x`` to be evaluated and used during parsing, but
   the variable declaration will not appear in the resulting circuit object or
   in any re-exported QASM string.

2) ``pow(k) @`` is only supported when ``k`` is an integer; it is implemented
   by repeated gate application.

3) Mid-circuit measurements are not supported in QiliSDK.
