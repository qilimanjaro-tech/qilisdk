Interoperability
================

QiliSDK is often not the only tool in a workflow, so :mod:`qilisdk.utils` provides converters between
QiliSDK objects and the formats used by other quantum frameworks. Each converter comes as a pair of
``from_*`` and ``to_*`` functions, and each one depends on a third-party library that is shipped as
an optional extra.

.. table::
   :align: left
   :widths: auto

   ============================================ ============================================================== ==============
   Format                                       QiliSDK object                                                 Extra
   ============================================ ============================================================== ==============
   OpenQASM 2.0 / 3.0                           :class:`~qilisdk.digital.circuit.Circuit`                      ``openqasm``
   -------------------------------------------- -------------------------------------------------------------- --------------
   QIR Base Profile                             :class:`~qilisdk.digital.circuit.Circuit`                      ``qir``
   -------------------------------------------- -------------------------------------------------------------- --------------
   OpenFermion ``QubitOperator``                :class:`~qilisdk.analog.hamiltonian.Hamiltonian`               ``openfermion``
   ============================================ ============================================================== ==============

OpenQASM
--------

:mod:`qilisdk.utils.openqasm` bridges a :class:`~qilisdk.digital.circuit.Circuit` and both
OpenQASM 2.0 and OpenQASM 3.0 through the ``openqasm3`` library:

- :func:`~qilisdk.utils.openqasm.openqasm2.from_qasm2` / :func:`~qilisdk.utils.openqasm.openqasm2.to_qasm2` — parse and serialize an OpenQASM 2.0 string.
- :func:`~qilisdk.utils.openqasm.openqasm2.from_qasm2_file` / :func:`~qilisdk.utils.openqasm.openqasm2.to_qasm2_file` — read and write a ``.qasm`` file.
- :func:`~qilisdk.utils.openqasm.openqasm3.from_qasm3` / :func:`~qilisdk.utils.openqasm.openqasm3.to_qasm3` — parse and serialize an OpenQASM 3.0 string.
- :func:`~qilisdk.utils.openqasm.openqasm3.from_qasm3_file` / :func:`~qilisdk.utils.openqasm.openqasm3.to_qasm3_file` — read and write a ``.qasm`` file.

.. code-block:: python

    from qilisdk.digital import CNOT, Circuit, H, M
    from qilisdk.utils.openqasm import from_qasm2, to_qasm2

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0, 1))

    qasm2_text = to_qasm2(circuit)
    reparsed = from_qasm2(qasm2_text)

See :doc:`../digital/digital_qasm` for the full list of OpenQASM 3.0 features QiliSDK supports and
the caveats that apply to each.

QIR
---

:mod:`qilisdk.utils.qir` bridges a :class:`~qilisdk.digital.circuit.Circuit` and the
`QIR Base Profile <https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md>`_
through Microsoft's `pyqir <https://pypi.org/project/pyqir/>`_ library:

- :func:`~qilisdk.utils.qir.qir.from_qir` / :func:`~qilisdk.utils.qir.qir.to_qir` — parse and serialize QIR textual LLVM IR.
- :func:`~qilisdk.utils.qir.qir.from_qir_file` / :func:`~qilisdk.utils.qir.qir.to_qir_file` — read and write a ``.ll`` or ``.bc`` file, dispatched by extension.

.. code-block:: python

    from qilisdk.digital import CNOT, Circuit, H, M
    from qilisdk.utils.qir import from_qir, to_qir

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0, 1))

    qir_text = to_qir(circuit, name="bell")
    reparsed = from_qir(qir_text)

See :doc:`../digital/digital_qir` for the supported gate mapping and the Base Profile restrictions.

OpenFermion
-----------

:mod:`qilisdk.utils.openfermion` converts between an OpenFermion ``QubitOperator`` and a QiliSDK
:class:`~qilisdk.analog.hamiltonian.Hamiltonian`, which lets you build a problem with OpenFermion's
chemistry tooling and then evolve or optimize it with QiliSDK.

- :func:`~qilisdk.utils.openfermion.openfermion.openfermion_to_qilisdk` — build a Hamiltonian from a ``QubitOperator``.
- :func:`~qilisdk.utils.openfermion.openfermion.qilisdk_to_openfermion` — build a ``QubitOperator`` from a Hamiltonian.

Each term of the ``QubitOperator`` becomes a product of
:class:`~qilisdk.analog.hamiltonian.PauliX`, :class:`~qilisdk.analog.hamiltonian.PauliY` and
:class:`~qilisdk.analog.hamiltonian.PauliZ` operators on the corresponding qubits. OpenFermion's
empty term, which stands for the identity, becomes
:class:`~qilisdk.analog.hamiltonian.PauliI` on qubit 0.

.. code-block:: python

    from openfermion import QubitOperator
    from qilisdk.utils.openfermion import openfermion_to_qilisdk

    operator = QubitOperator("X0 Z1", 1.5) + QubitOperator("Y2", 0.5) + QubitOperator("", 0.25)
    hamiltonian = openfermion_to_qilisdk(operator)
    print(hamiltonian)

**Output**::

    0.25 + 1.5 X(0) Z(1) + 0.5 Y(2)

The reverse direction drops identity factors, since OpenFermion writes the identity as the empty
term, and stores every coefficient as a complex number.

.. code-block:: python

    from qilisdk.analog import X, Z
    from qilisdk.utils.openfermion import qilisdk_to_openfermion

    hamiltonian = 0.5 * Z(0) * Z(1) + 1.2 * X(0)
    print(qilisdk_to_openfermion(hamiltonian))

**Output**::

    (1.2+0j) [X0] +
    (0.5+0j) [Z0 Z1]

.. note::

    Only the qubit-operator representation is converted. An OpenFermion ``FermionOperator`` has to
    be mapped to qubits first (with ``jordan_wigner``, ``bravyi_kitaev`` or any other OpenFermion
    transform) before it can be handed to
    :func:`~qilisdk.utils.openfermion.openfermion.openfermion_to_qilisdk`.
