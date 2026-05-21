QIR Import and Export
---------------------

QiliSDK can bridge :class:`~qilisdk.digital.Circuit` and the
`QIR Base Profile <https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md>`_
via Microsoft's `pyqir <https://pypi.org/project/pyqir/>`_ library. Four
entry-points live in :mod:`qilisdk.utils.qir`:

- :func:`~qilisdk.utils.qir.from_qir` — parse a QIR textual LLVM IR string into a Circuit.
- :func:`~qilisdk.utils.qir.from_qir_file` — read a ``.ll`` or ``.bc`` file (dispatched by extension).
- :func:`~qilisdk.utils.qir.to_qir` — serialize a Circuit to QIR textual IR.
- :func:`~qilisdk.utils.qir.to_qir_file` — write a ``.ll`` or ``.bc`` file (dispatched by extension).

The ``pyqir`` dependency is optional; install it with the ``qir`` extra:

.. code-block:: bash

    pip install qilisdk[qir]

Quick start
^^^^^^^^^^^

.. code-block:: python

    from qilisdk.digital import CNOT, Circuit, H, M
    from qilisdk.utils.qir import to_qir, from_qir

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0, 1))

    qir_text = to_qir(circuit, name="bell")
    print(qir_text)

    reparsed = from_qir(qir_text)

Reading and writing files dispatches on the extension — ``.ll`` is treated as
textual LLVM IR, ``.bc`` as LLVM bitcode:

.. code-block:: python

    from qilisdk.utils.qir import to_qir_file, from_qir_file

    to_qir_file(circuit, "bell.ll")  # textual
    to_qir_file(circuit, "bell.bc")  # bitcode

    same_circuit = from_qir_file("bell.bc")

Supported features
^^^^^^^^^^^^^^^^^^

QiliSDK targets the QIR Base Profile — a flat, statically-allocated subset of
QIR with no classical control flow. The table below maps QIR features to their
support status in QiliSDK:

- ✅: The feature is fully supported
- 🟡: The feature is partially supported, see the note for explanation
- ❌: The feature is not supported

============================================  ============================================   ============  =====
QIR feature                                   QiliSDK feature                                Supported     Notes
============================================  ============================================   ============  =====
Base Profile module layout                    --                                             ✅
Entry-point function                          --                                             ✅
Static qubit allocation                       --                                             ✅
Static result allocation                      --                                             ✅
``required_num_qubits`` / ``_num_results``    --                                             ✅
Textual LLVM IR (``.ll``)                     --                                             ✅
LLVM bitcode (``.bc``)                        --                                             ✅
Single-qubit QIS gates                        --                                             ✅            1
Two-qubit QIS gates                           --                                             ✅            1
Adjoint QIS gates (``s_adj`` / ``t_adj``)     --                                             ✅
Parameterized rotations                       --                                             🟡            2
Measurement (``mz``)                          --                                             🟡            3
Multi-qubit measurement grouping              --                                             🟡            4
``barrier`` / ``reset``                       --                                             ❌
Adaptive Profile / classical control          --                                             ❌
Branching on measurement results              --                                             ❌
Output recording calls (``rt__*``)            --                                             ❌
Dynamic qubit / result management             --                                             ❌
--                                            ``ccx`` and other three-qubit intrinsics       ❌            5
--                                            ``U1`` / ``U2`` / ``U3``                       ❌            5
--                                            Arbitrary ``Controlled`` / ``Exponential``     ❌            5

============================================  ============================================   ============  =====

1) See the "Supported gates" table below for the exact intrinsic mapping.

2) Rotation angles are exported as their currently-resolved numeric values.
   Base Profile does not support symbolic parameters, so rebind any
   :class:`~qilisdk.core.Parameter` values to concrete numbers before calling
   :func:`~qilisdk.utils.qir.to_qir`.

3) Only ``mz`` is emitted; mid-circuit measurement followed by classical
   control is not supported (and is forbidden by the Base Profile).

4) A multi-qubit :class:`~qilisdk.digital.gates.M` is serialized as one ``mz``
   call per target qubit. On reparse it comes back as one
   :class:`~qilisdk.digital.gates.M` per measured qubit — the qubit set is
   preserved but the grouping is not.

5) These must be decomposed into supported primitives before export, otherwise
   :func:`~qilisdk.utils.qir.to_qir` raises :class:`NotImplementedError`.

Supported gates
^^^^^^^^^^^^^^^

The exporter maps the following gates onto the standard ``__quantum__qis__*__body``
intrinsics; everything else raises :class:`NotImplementedError` to surface the
need for decomposition.

============================================  ===============================
QiliSDK gate                                  QIR intrinsic
============================================  ===============================
:class:`~qilisdk.digital.gates.I`             *(no-op, emitted as nothing)*
:class:`~qilisdk.digital.gates.X` /
:class:`~qilisdk.digital.gates.Y` /
:class:`~qilisdk.digital.gates.Z`             ``__quantum__qis__x/y/z__body``
:class:`~qilisdk.digital.gates.H`             ``__quantum__qis__h__body``
:class:`~qilisdk.digital.gates.S` /
:class:`~qilisdk.digital.gates.T`             ``__quantum__qis__s/t__body``
``Adjoint(S)`` / ``Adjoint(T)``               ``__quantum__qis__s/t__adj``
:class:`~qilisdk.digital.gates.RX` /
:class:`~qilisdk.digital.gates.RY` /
:class:`~qilisdk.digital.gates.RZ`            ``__quantum__qis__rx/ry/rz__body``
:class:`~qilisdk.digital.gates.CNOT`          ``__quantum__qis__cnot__body``
:class:`~qilisdk.digital.gates.CZ`            ``__quantum__qis__cz__body``
:class:`~qilisdk.digital.gates.SWAP`          ``__quantum__qis__swap__body``
:class:`~qilisdk.digital.gates.M`             one ``__quantum__qis__mz__body``
                                              call per target qubit
============================================  ===============================

Gates outside this table (``U1`` / ``U2`` / ``U3``, three-qubit unitaries,
arbitrary :class:`~qilisdk.digital.gates.Controlled` /
:class:`~qilisdk.digital.gates.Exponential` wrappers) need to be decomposed
into the supported set before export.

Profile and round-trip notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Modules are emitted with one entry-point function and statically allocated
  qubit and result registers (one result register per qubit), matching the
  Base Profile contract.
- Round-tripping preserves gate types, qubit indices, and rotation angles. A
  multi-qubit :class:`~qilisdk.digital.gates.M` instance is serialized as one
  ``mz`` call per target; on parse it comes back as one
  :class:`~qilisdk.digital.gates.M` per measured qubit (i.e. the qubit set is
  preserved but the grouping is not).
- Parameterized rotations are exported with their currently-resolved numeric
  angles; Base Profile does not support symbolic parameters, so rebind any
  :class:`~qilisdk.core.Parameter` values to concrete numbers before calling
  :func:`~qilisdk.utils.qir.to_qir`.
