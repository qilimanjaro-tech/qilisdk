Quantum Reservoirs
------------------

The :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir` functional executes a reservoir pipeline over a
sequence of inputs. Each layer applies an optional pre-processing circuit, an analog reservoir dynamics block
(:class:`~qilisdk.analog.schedule.Schedule`), and an optional post-processing circuit, followed by measurements of one
or more observables. The optional ``qubits_to_reset`` list can be used to reset selected qubits between layers.

The reservoir inputs are represented using :class:`~qilisdk.functionals.quantum_reservoirs.ReservoirInput` parameters.
These behave like standard :class:`~qilisdk.core.variables.Parameter` objects but are marked as non-trainable so they can
be driven by input data sequences rather than optimization loops.

**Parameters**

- **initial_state** (:class:`~qilisdk.core.qtensor.QTensor`): Initial state of the reservoir.
- **reservoir_layer** (:class:`~qilisdk.functionals.quantum_reservoirs.ReservoirLayer`): Defines pre/post-processing,
  reservoir dynamics, observables, and reset policy.
- **input_per_layer** (List[Dict[str, float]]): Input values to apply at each layer, keyed by input parameter names
  (typically the labels of :class:`~qilisdk.functionals.quantum_reservoirs.ReservoirInput`).

**Returns**

- :class:`~qilisdk.functionals.functional_result.FunctionalResult`: Access per-layer expectation values via
  :attr:`expectation_values`, plus optional states via :attr:`state`.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.backends import CudaBackend
    from qilisdk.core import ket
    from qilisdk.digital import Circuit, U2
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
    from qilisdk.analog import Schedule, X, Z
    from qilisdk.readout import Readout

    pre_processing = Circuit(2)
    pre_processing.add(U2(1, phi=ReservoirInput("phi_1", 0.1), gamma=ReservoirInput("gamma_1", 0.1)))

    res_layer = ReservoirLayer(
        evolution_dynamics=Schedule(
            hamiltonians={"h": Z(0) + Z(1) + Z(0) * Z(1) + 0.5 * (X(0) + X(1))},
            total_time=1.0,
            dt=0.1,
        ),
        input_encoding=pre_processing,
        qubits_to_reset=[1],
    )

    reservoir = QuantumReservoir(
        initial_state=(np.random.rand() * ket(0, 0) + np.random.rand() * ket(1, 1)).unit(),
        reservoir_layer=res_layer,
        input_per_layer=[
            {"phi_1": 0.2, "gamma_1": 0.1},
            {"phi_1": 0.3, "gamma_1": 0.2},
            {"phi_1": 0.4, "gamma_1": 0.3},
        ],
    )

    results = CudaBackend().execute(
        reservoir,
        Readout().with_expectation(observables=[Z(0), Z(1), Z(0) * Z(1)]),
    )
    print(results.get_expectation_values())


**Encoding Input Data**

You can inject classical data into a reservoir layer in multiple places. The only requirement is that the keys in
``input_per_layer`` match the labels of the :class:`~qilisdk.functionals.quantum_reservoirs.ReservoirInput` objects you
place in the layer.

**1. Encode with input and output circuits**

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket
    from qilisdk.digital import Circuit, RX, RY
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer

    theta_in = ReservoirInput("theta_in", 0.0)
    phi_out = ReservoirInput("phi_out", 0.0)

    input_circuit = Circuit(2)
    input_circuit.add(RX(0, theta=theta_in))

    output_circuit = Circuit(2)
    output_circuit.add(RY(1, theta=phi_out))

    layer = ReservoirLayer(
        evolution_dynamics=Schedule(
            hamiltonians={"h": Z(0) * Z(1) + 0.3 * (X(0) + X(1))},
            total_time=1.0,
            dt=0.1,
        ),
        input_encoding=input_circuit,
        output_encoding=output_circuit,
    )

    reservoir = QuantumReservoir(
        initial_state=ket(0, 0),
        reservoir_layer=layer,
        input_per_layer=[
            {"theta_in": 0.1, "phi_out": 0.0},
            {"theta_in": 0.5, "phi_out": 0.4},
        ],
    )

**2. Encode directly in Hamiltonian parameters**

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer

    gain = ReservoirInput("gain", 0.2)
    detuning = ReservoirInput("detuning", -0.1)

    hamiltonian = gain * (X(0) + X(1)) + detuning * (Z(0) + Z(1)) + Z(0) * Z(1)

    layer = ReservoirLayer(
        evolution_dynamics=Schedule(
            hamiltonians={"h": hamiltonian},
            coefficients={"h": {(0.0, 1.0): 1.0}},
            dt=0.05,
        ),
    )

    reservoir = QuantumReservoir(
        initial_state=ket(0, 0),
        reservoir_layer=layer,
        input_per_layer=[
            {"gain": 0.1, "detuning": -0.2},
            {"gain": 0.7, "detuning": 0.0},
            {"gain": 0.3, "detuning": 0.2},
        ],
    )

**3. Encode in the schedule profile and duration**

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.core import ket
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer

    drive_amp = ReservoirInput("drive_amp", 1.0)
    duration = ReservoirInput("duration", 1.0)

    layer = ReservoirLayer(
        evolution_dynamics=Schedule(
            hamiltonians={
                "drive": X(0) + X(1),
                "problem": Z(0) * Z(1),
            },
            coefficients={
                "drive": {(0.0, 1.0): drive_amp},
                "problem": {(0.0, 1.0): lambda t: t},
            },
            total_time=duration,
            dt=0.05,
        ),
    )

    reservoir = QuantumReservoir(
        initial_state=ket(0, 0),
        reservoir_layer=layer,
        input_per_layer=[
            {"drive_amp": 1.0, "duration": 0.6},
            {"drive_amp": 0.3, "duration": 1.4},
        ],
    )

