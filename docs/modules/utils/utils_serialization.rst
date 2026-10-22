Serialization and Hashing
=========================

Almost every object in QiliSDK can be written to disk and read back, and every object that takes part
in symbolic expressions needs a hash that stays the same across processes. Both of these are provided
by :mod:`qilisdk.utils.serialization`.

Serialization
-------------

:mod:`qilisdk.utils.serialization` offers four entry-points, built on the YAML handler in
:mod:`qilisdk.yaml`:

- :func:`~qilisdk.utils.serialization.serialize` — serialize an object to a YAML string.
- :func:`~qilisdk.utils.serialization.serialize_to` — serialize an object to a YAML file.
- :func:`~qilisdk.utils.serialization.deserialize` — rebuild an object from a YAML string.
- :func:`~qilisdk.utils.serialization.deserialize_from` — rebuild an object from a YAML file.

.. code-block:: python

    from qilisdk.analog import X, Z
    from qilisdk.utils.serialization import serialize

    hamiltonian = 0.5 * Z(0) * Z(1) + 1.2 * X(0)
    print(serialize(hamiltonian))

**Output**::

    !Hamiltonian
    _elements: !defaultdict
      default_factory: builtins.complex
      items:
        ? !tuple
        - !PauliZ {_qubit: 0}
        - !PauliZ {_qubit: 1}
        : !complex {imag: 0.0, real: 0.5}
        ? !tuple
        - !PauliX {_qubit: 0}
        : !complex {imag: 0.0, real: 1.2}
    _parameter_constraints: []
    _parameters: {}
    _prefix: ''

Each ``!Tag`` in the output names the class that produced the node, which allows the object
graph to be rebuilt exactly. 

Writing to and reading from a file works in the same way:

.. code-block:: python

    from qilisdk.digital import CNOT, Circuit, H, M
    from qilisdk.utils.serialization import deserialize_from, serialize_to

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(M(0, 1))

    serialize_to(circuit, "bell.yml")
    same_circuit = deserialize_from("bell.yml")

Checking the type on the way back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both ``deserialize`` functions accept an optional class as their second argument. When given, the
rebuilt object is checked against it and the call is typed as returning that class.

.. code-block:: python

    from qilisdk.digital import Circuit
    from qilisdk.utils.serialization import deserialize_from

    circuit = deserialize_from("bell.yml", Circuit)  # typed as Circuit

Errors
^^^^^^

Failures are wrapped so that callers only have to catch two exceptions:

- :class:`~qilisdk.utils.serialization.SerializationError` — the object could not be written.
- :class:`~qilisdk.utils.serialization.DeserializationError` — the YAML could not be parsed, or the result is not an instance of the requested class.

Registering your own classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A class becomes serializable by registering it with the shared YAML handler. The decorator assigns
the tag automatically from the class name, and every public QiliSDK class that you can serialize uses
it:

.. code-block:: python

    from qilisdk.yaml import yaml

    @yaml.register_class
    class MyCustomAnsatz:
        def __init__(self, depth: int) -> None:
            self.depth = depth

Pass ``shared=True`` to prefix the tag with the top-level package name, which avoids collisions with
tags defined elsewhere.

.. warning::

    The handler is a ruamel ``YAML(typ="unsafe")`` instance, so deserializing constructs arbitrary
    Python objects named in the document. Only load files you trust.

Hashing
-------

:func:`qilisdk.utils.hashing.hash` produces a stable hash for the objects QiliSDK builds expressions
out of. It is used by :class:`~qilisdk.core.variables.BaseVariable`,
:class:`~qilisdk.core.expression.Expression`, :class:`~qilisdk.core.qtensor.QTensor`,
:class:`~qilisdk.analog.hamiltonian.Hamiltonian`, :class:`~qilisdk.digital.circuit.Circuit` and the
gates, so that terms can be collected in dictionaries and compared reliably.

.. code-block:: python

    from qilisdk.utils.hashing import hash as qili_hash

    qili_hash("some label", 1.0, (0, 1))

It differs from Python's built-in ``hash`` in two main ways:

- It is **stable across processes**. Python randomizes string hashing per interpreter run, whereas this one is a `blake2b <https://www.blake2.net/>`_ digest of a canonical encoding, so the same object hashes to the same integer every time.
- It is **consistent across numeric types**. Values are encoded as exact integer ratios, so ``1``, ``1.0`` and ``(1+0j)`` all hash alike, and ``nan``, ``inf`` and NumPy scalars are handled explicitly.

.. note::

    The returned integer is compatible with Python's ``__hash__`` protocol, so it can be returned
    directly from a ``__hash__`` method. It is not a cryptographic commitment and should not be used
    as one.
