Functionals
===========

The :mod:`~qilisdk.functionals` module provides high-level quantum execution procedures by combining tools from the
:mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.core` modules. Currently, it includes the following execution functionals:

- :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` — Propagates a digital quantum circuit through the backend.
- :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` — Simulates analog time evolution of one or more Hamiltonians according to a time-dependent schedule.
- :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir` — Runs a quantum reservoir pipeline (pre-processing, reservoir dynamics, post-processing) across multiple input layers.

Moreover, it provides more complex functionals that are used to execute more complex algorithms:

- :class:`~qilisdk.functionals.variational_program.VariationalProgram` — Builds parameterized program to be optimized in a hybrid quantum-classical environment.

Architecture Overview
---------------------

Every functional conforms to the abstract :class:`~qilisdk.functionals.functional.Functional` interface. Primitive
functionals such as :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` and
:class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` also inherit from
:class:`~qilisdk.functionals.functional.PrimitiveFunctional`, which mixes in the
:class:`~qilisdk.core.parameterizable.Parameterizable` contract. This lets backends query and update symbolic
parameters consistently before execution.

Readout is decoupled from functionals: measurement details (shots, observables, state tomography) are specified via
:mod:`~qilisdk.readout` objects passed to the backend's :meth:`~qilisdk.backends.backend.Backend.execute` method.
All primitive functionals return a unified :class:`~qilisdk.functionals.functional_result.FunctionalResult`.

Result Objects
--------------

* :class:`~qilisdk.functionals.functional_result.FunctionalResult`
    The unified result type for all primitive functionals. Access results through:
    ``samples`` for shot counts, ``probabilities`` for measurement probabilities,
    ``state`` for the terminal :class:`~qilisdk.core.qtensor.QTensor` state (when using
    :class:`~qilisdk.readout.StateTomographyReadout`), and ``expected_values`` for expectation values
    (when using :class:`~qilisdk.readout.ExpectationReadout`).
    When ``store_intermediate_results=True``, intermediate results are available via
    ``intermediate_states``, ``intermediate_samples``, ``intermediate_probabilities``, and ``intermediate_expected_values``.
* :class:`~qilisdk.functionals.variational_program_result.VariationalProgramResult`
    bundles the optimizer trajectory (optimal cost, parameters, intermediate steps) together with the functional result
    obtained at convergence.

These objects make post-processing workflows ergonomic. For example, after a digital propagation you can surface the
most likely bitstrings:

.. code-block:: python

    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import SamplingReadout

    backend = QutipBackend()
    result = backend.execute(DigitalPropagation(circuit), readout=[SamplingReadout(nshots=1_000)])
    print("Most likely outcomes:", result.probabilities)

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
    from qilisdk.readout import SamplingReadout

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

    # Run on CUDA backend and retrieve counts
    backend = CudaBackend()
    results = backend.execute(digital_propagation, readout=[SamplingReadout(nshots=100)])
    print(results)

**Output**

::

    - Functional Results: [

    Sampling Results: (
        nshots=100,
        samples={'00': 53, '11': 47}
    )

    ]



Analog Evolution
----------------

The :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` functional simulates analog evolution of a quantum system under one or more Hamiltonians, following a specified time-dependent schedule. Its parameter interface mirrors that of
:class:`~qilisdk.analog.schedule.Schedule`, making it straightforward to sweep control waveforms or pulse amplitudes from
classical optimizers.

Observables and shot counts are specified separately via readout objects passed to
:meth:`~qilisdk.backends.backend.Backend.execute`.

**Parameters**

- **schedule** (:class:`~qilisdk.analog.schedule.Schedule`): Defines total evolution time, time steps, Hamiltonians, and their time-dependent coefficients.
- **initial_state** (:class:`~qilisdk.core.qtensor.QTensor`): Initial state of the system.
- **store_intermediate_results** (bool, optional): If True, records the state at each time step. Default is False.

**Returns**

- :class:`~qilisdk.functionals.functional_result.FunctionalResult`: Inspect
  :attr:`expected_values` for measured observables,
  :attr:`state` for the closing state, and
  :attr:`intermediate_states` and :attr:`intermediate_expected_values` when ``store_intermediate_results`` is enabled.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.backends import QutipBackend, CudaBackend
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import ExpectationReadout

    # Define total time and timestep
    T = 10.0
    dt = 0.5
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time-dependent schedule
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
        interpolation=Interpolation.LINEAR,
    )

    # Prepare an equal superposition initial state
    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

    # Create the AnalogEvolution functional
    analog_evolution = AnalogEvolution(
        schedule=schedule,
        initial_state=initial_state,
        store_intermediate_results=False,
    )

we can execute it on the Qutip backend:

.. code-block:: python

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(
        analog_evolution,
        readout=[ExpectationReadout(observables=[Z(0), X(0), Y(0)])],
    )
    print(results)

**Output**

::

    - Functional Results: [

    Expectation Value Results: (
        expected_values=[-0.99388223, 0.0467696, -0.10005353],
    )

    ]


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
  :attr:`expected_values`, plus optional states via :attr:`state`.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.backends import CudaBackend
    from qilisdk.core import ket
    from qilisdk.digital import Circuit, U2
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
    from qilisdk.analog import Schedule, X, Z
    from qilisdk.readout import ExpectationReadout

    pre_processing = Circuit(2)
    pre_processing.add(U2(1, phi=ReservoirInput("phi_1", 0.1), gamma=ReservoirInput("gamma_1", 0.1)))

    res_layer = ReservoirLayer(
        evolution_dynamics=Schedule(
            hamiltonians={"h": Z(0) + Z(1) + Z(0) * Z(1) + 0.5 * (X(0) + X(1))},
            total_time=1.0,
            dt=0.1,
        ),
        observables=[Z(0), Z(1), Z(0) * Z(1)],
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
        readout=[ExpectationReadout(observables=[Z(0), Z(1), Z(0) * Z(1)])],
    )
    print(results.expected_values)


Encoding Input Data
^^^^^^^^^^^^^^^^^^^

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
        observables=[Z(0), Z(1)],
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
        observables=[Z(0)],
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
        observables=[Z(0) * Z(1)],
    )

    reservoir = QuantumReservoir(
        initial_state=ket(0, 0),
        reservoir_layer=layer,
        input_per_layer=[
            {"drive_amp": 1.0, "duration": 0.6},
            {"drive_amp": 0.3, "duration": 1.4},
        ],
    )



Variational Programs
---------------------

The :class:`~qilisdk.functionals.variational_program.VariationalProgram` functional gathers the pieces required for a
variational quantum algorithm. It accepts a parameterized primitive functional, an optimizer, and a cost function. When
you call :meth:`~qilisdk.backends.backend.Backend.execute` it evaluates the functional repeatedly with updated
parameters, feeds the resulting :class:`~qilisdk.functionals.functional_result.FunctionalResult` into the supplied
cost function, and finally returns a :class:`~qilisdk.functionals.variational_program_result.VariationalProgramResult`.
Parameter constraints (inequalities/equalities on parameters) are attached at this level via the ``parameter_constraints``
argument; this is the place to enforce relations like ``theta >= phi`` or cross-parameter bounds across all QiliSDK
functionals.
Only parameters marked as trainable are optimized during this loop.

**Parameters**

- **functional** (:class:`~qilisdk.functionals.functional.PrimitiveFunctional`): Parameterized primitive to optimize
  (for instance :class:`DigitalPropagation <qilisdk.functionals.digital_propagation.DigitalPropagation>` or
  :class:`AnalogEvolution <qilisdk.functionals.analog_evolution.AnalogEvolution>`).
- **optimizer** (:class:`~qilisdk.optimizers.optimizer.Optimizer`): Classical optimizer that proposes new parameter
  values and optionally stores intermediate iterates.
- **cost_function** (:class:`~qilisdk.cost_functions.cost_function.CostFunction`): Object that maps the functional results
  to a scalar cost; frequently constructed from a :class:`~qilisdk.core.model.Model`.
- **store_intermediate_results** (bool, optional): When True, the optimizer keeps the intermediate steps, which are
  exposed through :attr:`~qilisdk.functionals.variational_program_result.VariationalProgramResult.intermediate_results`.
- **parameter_constraints** (list[:class:`~qilisdk.core.variables.ComparisonTerm`], optional): Constraints on functional
  parameters (e.g., ``theta >= 0.5``) evaluated before each optimizer update. This is the supported entry point for
  enforcing parameter relations in QiliSDK.

**Returns**

- :class:`~qilisdk.functionals.variational_program_result.VariationalProgramResult`: Retrieve
  :attr:`optimal_cost <qilisdk.functionals.variational_program_result.VariationalProgramResult.optimal_cost>`,
  :attr:`optimal_parameters <qilisdk.functionals.variational_program_result.VariationalProgramResult.optimal_parameters>`,
  and the final functional output packaged in
  :attr:`optimal_execution_results <qilisdk.functionals.variational_program_result.VariationalProgramResult.optimal_execution_results>`.

**Usage Example (Using QuTiP Backend)**

.. code-block:: python

    import numpy as np

    from qilisdk.backends import QutipBackend
    from qilisdk.core.model import Model, ObjectiveSense
    from qilisdk.core.variables import LEQ, BinaryVariable
    from qilisdk.cost_functions.model_cost_function import ModelCostFunction
    from qilisdk.digital import CNOT, U3, HardwareEfficientAnsatz
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.functionals.variational_program import VariationalProgram
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.readout import SamplingReadout


    values = [2, 3, 7]
    weights = [1, 3, 3]
    max_weight = 4
    binary_var = [BinaryVariable(f"b{i}") for i in range(len(values))]

    model = Model("Knapsack")

    model.set_objective(sum(binary_var[i] * values[i] for i in range(len(values))), sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("max_weights", LEQ(sum(binary_var[i] * weights[i] for i in range(len(weights))), max_weight))


    ansatz = HardwareEfficientAnsatz(
        nqubits=3, layers=4, connectivity="Circular", one_qubit_gate=U3, two_qubit_gate=CNOT, structure="Interposed"
    )

    optimizer = SciPyOptimizer(method="COBYQA")

    backend = QutipBackend()
    result = backend.execute(
        VariationalProgram(
            functional=DigitalPropagation(ansatz),
            optimizer=optimizer,
            cost_function=ModelCostFunction(model),
        ),
        readout=[SamplingReadout(nshots=1000)],
    )

    print(result)

**Output**

::

    VariationalProgramResult(
      Optimal Cost=-9.0,
      Optimal Parameters=[...],
      Intermediate Results=[...],
      Optimal Results=- Functional Results: [

    Sampling Results: (
        nshots=1000,
        samples={'000': 2, '010': 3, '101': 994, '110': 1}
    )

    ]
    )


**Usage Example 2 (Using QuTiP Backend)**
This example optimizes a variational schedule under some parameter constraints.


.. code-block:: python

    from qilisdk.core.variables import LT, GreaterThan
    from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
    from qilisdk.functionals import VariationalProgram, AnalogEvolution
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.analog import *
    from qilisdk.analog.schedule import Interpolation
    from qilisdk.core.variables import Parameter
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    from qilisdk.readout import ReadoutMethod
    import numpy as np

    from qilisdk.utils.visualization.style import ScheduleStyle


    T = 10
    p = [Parameter(f"p_{i}", (i + 1)*2, bounds=(0, 10)) for i in range(4)]
    p.insert(0, 0)
    p.append(T)
    s = [Parameter(f"s_{i}", (i + 2) * 0.1, bounds=(0, 1)) for i in range(2)]
    h0 = X(0)
    h1 = Z(0)
    max_time = Parameter("max_time", 1.5)

    schedule = Schedule(
        hamiltonians={"h_x": h0, "h_z": h1},
        coefficients={
            "h_x": {p[0]: 1, (p[1], p[2]): 1 - s[0], (p[3], p[4]): 1 - s[1], p[5]: 0},
            "h_z": {p[0]: 0, (p[1], p[2]): s[0], (p[3], p[4]): s[1], p[5]: 1},
        },
        interpolation=Interpolation.LINEAR,
    )

    schedule.draw(ScheduleStyle(title="Schedule Before Optimization"))

    te = AnalogEvolution(
        schedule=schedule,
        initial_state=tensor_prod([ket(0) - ket(1) for _ in range(schedule.nqubits)]).unit(),
    )

    vp = VariationalProgram(
        te,
        SciPyOptimizer(method="COBYQA"),
        cost_function=ObservableCostFunction(h1),
        parameter_constraints=[
            GreaterThan(p[3], 5)
        ]
    )

    print(vp.get_constraints()) # print the constraints of the variational program.

    backend = QutipBackend()
    results = backend.execute(vp, readout=[ReadoutMethod.expectation(observables=[h1]), ReadoutMethod.state_tomography(), ReadoutMethod.sampling(1000)])
    schedule.draw(ScheduleStyle(title="Schedule After Optimization"))
    print(results)
