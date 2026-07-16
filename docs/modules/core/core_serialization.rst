Serialization and Security
==========================

QiliSDK objects (circuits, Hamiltonians, schedules, experiments, calibrations,
results, ...) can be serialized to a YAML string or file and reconstructed
later.

.. code-block:: python

    from qilisdk.utils.serialization import serialize, deserialize, serialize_to, deserialize_from
    from qilisdk.digital import Circuit, H, CNOT

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    blob = serialize(circuit)            # -> YAML string
    restored = deserialize(blob)         # -> Circuit

    serialize_to(circuit, "circuit.yaml")
    restored = deserialize_from("circuit.yaml")

Safe by default
---------------

``deserialize`` and ``deserialize_from`` use a **data-only loader** that
reconstructs the known QiliSDK and primitive types (waveforms, circuits,
Hamiltonians, numpy arrays, tuples, complex numbers, UUIDs, enums, ...) and
**rejects code-execution tags**. Loading an untrusted string or file therefore
cannot execute arbitrary code, so you can treat these functions like
``json.load``.

The ``trust_code`` opt-in
-------------------------

Both functions accept a keyword-only ``trust_code`` flag (default ``False``).
When set to ``True`` they fall back to the unrestricted loader, which **can
execute arbitrary Python code embedded in the input**.

.. danger::

   Only pass ``trust_code=True`` to data you fully control and trust. A
   malicious YAML string or file can run arbitrary code on your machine when it
   is loaded with ``trust_code=True``. Never use it on input received from
   another user, a network request, an uploaded file, or any other untrusted
   source.

.. code-block:: python

    # Safe: rejects code-execution tags - use this for any untrusted input.
    obj = deserialize(untrusted_blob)

    # Dangerous: only for input you fully control and trust.
    obj = deserialize(trusted_blob, trust_code=True)

If you only need to move data between trusted QiliSDK processes, keep the
default safe loader. Reserve ``trust_code=True`` for the rare case where you
must round-trip an object that genuinely embeds custom callables and the source
is fully trusted.
