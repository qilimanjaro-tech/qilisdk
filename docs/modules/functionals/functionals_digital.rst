Digital Propagation
-------------------

The :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` functional propagates a digital quantum circuit
through the backend. Because it subclasses
:class:`~qilisdk.functionals.functional.PrimitiveFunctional`, any symbolic parameters exposed by the underlying
:class:`~qilisdk.digital.circuit.Circuit` can be queried or updated through helper methods such as
:meth:`~qilisdk.functionals.digital_propagation.DigitalPropagation.get_parameter_names`.

Measurement details such as the number of shots are specified separately via readout objects passed to
:meth:`~qilisdk.backends.backend.Backend.execute`.

**Parameters**

- **circuit** (:class:`~qilisdk.digital.circuit.Circuit`): Circuit to be propagated.

**Returns**

- :class:`~qilisdk.functionals.functional_result.FunctionalResult`: Access shot counts via :attr:`samples`, or
  probabilities through :attr:`probabilities`.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.functionals import DigitalPropagation

    # Create a 2-qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(0, theta=np.pi))
    circuit.add(CNOT(0, 1))

    # Initialize the DigitalPropagation functional
    digital_propagation = DigitalPropagation(circuit)


This functional can be executed on any backend that supports digital circuits. For example, we can execute it on the CUDA backend:


.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.readout import Readout

    # Run on CUDA backend and retrieve counts
    backend = CudaBackend()
    results = backend.execute(digital_propagation, Readout().with_sampling(nshots=100))
    print(results)

**Output**

::

    - Functional Results: [

    Sampling Results: (
        nshots=100,
        samples={'00': 53, '11': 47}
    )

    ]

