Functionals
===========

The :mod:`~qilisdk.functionals` module provides high-level quantum execution procedures by combining tools from the
:mod:`~qilisdk.analog`, :mod:`~qilisdk.digital`, and :mod:`~qilisdk.core` modules. Currently, it includes the following execution functionals:

- :class:`~qilisdk.functionals.sampling.Sampling` — Executes repeated sampling of a digital quantum circuit.
- :class:`~qilisdk.functionals.time_evolution.TimeEvolution` — Simulates analog time evolution of one or more Hamiltonians according to a time-dependent schedule.
- :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir` — Runs a quantum reservoir pipeline (pre-processing, reservoir dynamics, post-processing) across multiple input layers.

Moreover, it provides more complex functionals that are used to execute more complex algorithms:

- :class:`~qilisdk.functionals.variational_program.VariationalProgram` — Builds parameterized program to be optimized in a hybrid quantum-classical environment.

Architecture Overview
---------------------

Every functional conforms to the abstract :class:`~qilisdk.functionals.functional.Functional` interface. Primitive
functionals such as :class:`~qilisdk.functionals.sampling.Sampling` and
:class:`~qilisdk.functionals.time_evolution.TimeEvolution` also inherit from
:class:`~qilisdk.functionals.functional.PrimitiveFunctional`, which mixes in the
:class:`~qilisdk.core.parameterizable.Parameterizable` contract. This lets backends query and update symbolic
parameters consistently before execution.

Each functional advertises a matching :attr:`result_type` pointing to a concrete
:class:`~qilisdk.functionals.functional_result.FunctionalResult` subclass. When you call
:meth:`~qilisdk.backends.backend.Backend.execute`, the backend inspects this attribute to decide which result object to
construct and return.

Result Objects
--------------

* :class:`~qilisdk.functionals.sampling_result.SamplingResult`
    stores shot counts, probabilities and convenience helpers such as
    :meth:`~qilisdk.functionals.sampling_result.SamplingResult.get_probabilities`.
* :class:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult`
    contains expectation values, the terminal :class:`~qilisdk.core.qtensor.QTensor` state, and (optionally) the list
    of intermediate states when ``store_intermediate_results=True``.
* :class:`~qilisdk.functionals.quantum_reservoirs_result.QuantumReservoirResult`
    extends :class:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult` with the same data model,
    returning per-layer expectation values and optional intermediate/final states.
* :class:`~qilisdk.functionals.variational_program_result.VariationalProgramResult`
    bundles the optimizer trajectory (optimal cost, parameters, intermediate steps) together with the functional result
    obtained at convergence.

These objects make post-processing workflows ergonomic. For example, :class:`~qilisdk.functionals.sampling_result.SamplingResult` can surface the most likely
bitstrings:

.. code-block:: python

    from qilisdk.backends import QutipBackend
    from qilisdk.functionals import Sampling

    backend = QutipBackend()
    sampling_result = backend.execute(Sampling(circuit, nshots=1_000))
    top = sampling_result.get_probabilities(5)
    print("Most likely outcomes:", top)

Sampling
--------

The :class:`~qilisdk.functionals.sampling.Sampling` functional runs a digital quantum circuit multiple times (shots) and aggregates the measurement outcomes. Because it subclasses
:class:`~qilisdk.functionals.functional.PrimitiveFunctional`, any symbolic parameters exposed by the underlying
:class:`~qilisdk.digital.circuit.Circuit` can be queried or updated through helper methods such as
:meth:`~qilisdk.functionals.sampling.Sampling.get_parameter_names`.

**Parameters**

- **circuit** (:class:`~qilisdk.digital.circuit.Circuit`): Circuit to be sampled.
- **nshots** (int): Number of times to execute the circuit and collect measurement results.

**Returns**

- :class:`~qilisdk.functionals.sampling_result.SamplingResult`: Access shot counts via :attr:`samples`, or probabilities
  through :attr:`probabilities`. Convenience helpers like
  :meth:`~qilisdk.functionals.sampling_result.SamplingResult.get_probability` ease downstream analyses.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.functionals import Sampling

    # Create a 2‑qubit circuit
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(RX(0, theta=np.pi))
    circuit.add(CNOT(0, 1))

    # Initialize the Sampling functional with 100 shots
    sampling = Sampling(circuit=circuit, nshots=100)


This functional can be executed on any backend that supports digital circuits. For we can execute it on the CUDA backend:


.. code-block:: python

    from qilisdk.backends import CudaBackend

    # Run on Qutip backend and retrieve counts
    backend = CudaBackend()
    results = backend.execute(sampling)
    print(results)

**Output**

::

    SamplingResult(
        nshots=100,
        samples={'00': 53, '11': 47}
    )



Time Evolution
--------------

The :class:`~qilisdk.functionals.time_evolution.TimeEvolution` functional simulates analog evolution of a quantum system under one or more Hamiltonians, following a specified time‑dependent schedule. Its parameter interface mirrors that of
:class:`~qilisdk.analog.schedule.Schedule`, making it straightforward to sweep control waveforms or pulse amplitudes from
classical optimizers.

**Parameters**

- **schedule** (:class:`~qilisdk.analog.schedule.Schedule`): Defines total evolution time, time steps, Hamiltonians, and their time‑dependent coefficients.
- **initial_state** (:class:`~qilisdk.core.qtensor.QTensor`): Initial state of the system.
- **observables** (List[:class:`~qilisdk.analog.hamiltonian.Hamiltonian` or :class:`~qilisdk.analog.hamiltonian.PauliOperator`]): Operators to measure after evolution.
- **nshots** (int, optional): Number of repetitions for each observable measurement. Default is 1.
- **store_intermediate_results** (bool, optional): If True, records the state at each time step. Default is False.

**Returns**

- :class:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult`: Inspect
  :attr:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult.final_expected_values` for measured observables,
  :attr:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult.final_state` for the closing state, and
  :attr:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult.intermediate_states` when
  ``store_intermediate_results`` is enabled.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.backends import QutipBackend, CudaBackend
    from qilisdk.functionals import TimeEvolution

    # Define total time and timestep
    T = 10.0
    dt = 0.5
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time‑dependent schedule
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

    # Create the TimeEvolution functional
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[Z(0), X(0), Y(0)],
        nshots=100,
        store_intermediate_results=False,
    )

we can execute it on the Qutip backend:

.. code-block:: python

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(time_evolution)
    print(results)

**Output**

::

    TimeEvolutionResult(
        final_expected_values=array([-0.99388223,  0.0467696 , -0.10005353]),
        final_state=QTensor(shape=2x1, nnz=2, format='csr')
        [[0.05506547-0.00516502j]
        [0.3364973 -0.94005887j]]
    )


Quantum Reservoirs
------------------

The :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir` functional executes a reservoir pipeline over a
sequence of inputs. Each layer applies an optional pre-processing circuit, a reservoir dynamics block (either a
digital :class:`~qilisdk.digital.circuit.Circuit` or analog :class:`~qilisdk.analog.schedule.Schedule`), and an optional
post-processing circuit, followed by measurements of one or more observables. The optional ``qubits_to_reset`` list can
be used to reset selected qubits between layers.

The reservoir inputs are represented using :class:`~qilisdk.functionals.quantum_reservoirs.ReservoirInput` parameters.
These behave like standard :class:`~qilisdk.core.variables.Parameter` objects but are marked as non-trainable so they can
be driven by input data sequences rather than optimization loops.
Quantum reservoir execution is supported by the :class:`~qilisdk.backends.cuda_backend.CudaBackend` and
:class:`~qilisdk.backends.qutip_backend.QutipBackend`.

**Parameters**

- **initial_state** (:class:`~qilisdk.core.qtensor.QTensor`): Initial state of the reservoir.
- **reservoir_pass** (:class:`~qilisdk.functionals.quantum_reservoirs.ReservoirPass`): Defines pre/post-processing,
  reservoir dynamics, observables, and reset policy.
- **input_per_layer** (List[Dict[str, float]]): Input values to apply at each layer, keyed by input parameter names
  (typically the labels of :class:`~qilisdk.functionals.quantum_reservoirs.ReservoirInput`).
- **store_final_state** (bool, optional): Store the final state after the last layer.
- **store_intermideate_states** (bool, optional): Store the state after each layer.
- **nshots** (int, optional): Number of shots to pass through to analog evolution steps within the reservoir.

**Returns**

- :class:`~qilisdk.functionals.quantum_reservoirs_result.QuantumReservoirResult`: Access per-layer expectation values via
  :attr:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult.expected_values`, plus optional states when
  ``store_intermideate_states`` or ``store_final_state`` are enabled.

**Usage Example**

.. code-block:: python

    import numpy as np
    from qilisdk.backends import CudaBackend
    from qilisdk.core import ket
    from qilisdk.digital import Circuit, U2
    from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirPass
    from qilisdk.analog import Schedule, X, Z

    pre_processing = Circuit(2)
    pre_processing.add(U2(1, phi=ReservoirInput("phi_1", 0.1), gamma=ReservoirInput("gamma_1", 0.1)))

    res_pass = ReservoirPass(
        reservoir_dynamics=Schedule(
            hamiltonians={"h": Z(0) + Z(1) + Z(0) * Z(1) + 0.5 * (X(0) + X(1))},
            total_time=1.0,
            dt=0.1,
        ),
        measured_observables=[Z(0), Z(1), Z(0) * Z(1)],
        pre_processing=pre_processing,
        qubits_to_reset=[1],
    )

    reservoir = QuantumReservoir(
        initial_state=(np.random.rand() * ket(0, 0) + np.random.rand() * ket(1, 1)).unit(),
        reservoir_pass=res_pass,
        input_per_layer=[
            {"phi_1": 0.2, "gamma_1": 0.1},
            {"phi_1": 0.3, "gamma_1": 0.2},
            {"phi_1": 0.4, "gamma_1": 0.3},
        ],
        store_final_state=True,
        store_intermideate_states=True,
    )

    results = CudaBackend().execute(reservoir)
    print(results.expected_values)



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

**Parameters**

- **functional** (:class:`~qilisdk.functionals.functional.PrimitiveFunctional`): Parameterized primitive to optimize
  (for instance :class:`Sampling <qilisdk.functionals.sampling.Sampling>` or
  :class:`TimeEvolution <qilisdk.functionals.time_evolution.TimeEvolution>`).
- **optimizer** (:class:`~qilisdk.optimizers.optimizer.Optimizer`): Classical optimizer that proposes new parameter
  values and optionally stores intermediate iterates.
- **cost_model** (:class:`~qilisdk.cost_functions.cost_function.CostFunction`): Object that maps the functional results
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
    from qilisdk.functionals import Sampling
    from qilisdk.functionals.variational_program import VariationalProgram
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer


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
    result = backend.execute(VariationalProgram(functional=Sampling(ansatz), optimizer=optimizer, cost_function=ModelCostFunction(model)))

    print(result)

**Output** 

::

    VariationalProgramResult(
        Optimal Cost=-9.0,
        Optimal Parameters=[-3.0255755774847164,
        0.01887035500903607,
        0.028664963343905215,
        0.013368501054216147,
        -0.08437636531417667,
        -0.029660168083975074,
        0.0025941436125639207,
        -0.008258689086022116,
        -1.0021850138459234,
        0.028289669542409385,
        0.05329311289872701,
        0.012030731723947204,
        -0.0016337384126663568,
        0.03515665488520232,
        -0.0872593072394056,
        0.014530612025796379,
        -0.020256728609650863,
        -0.010675287925157617,
        0.004923341397045154,
        -0.09761909602332684,
        -0.03415470841379532,
        0.01903348186553671,
        0.026432142345385774,
        0.02904121692864865,
        0.003957631926259311,
        0.04161531214200481,
        0.010449668427772792,
        0.06987683584625945,
        -0.008282611237782185,
        -1.001211264562447,
        -0.025240867130997865,
        -0.04703717734032765,
        -0.09092557977061645,
        -0.03600273303052503,
        -0.023439848552130913,
        0.03346875504677889],
        Intermediate Results=[],
        Optimal Results=SamplingResult(
        nshots=1000,
        samples={'000': 2, '010': 3, '101': 994, '110': 1}
        )
        )


**Usage Example 2 (Using QuTiP Backend)**
This example optimizes a variational schedule under some parameter constraints.


.. code-block:: python

    from qilisdk.core.variables import LT, GreaterThan
    from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
    from qilisdk.functionals import VariationalProgram, TimeEvolution
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.analog import *
    from qilisdk.analog.schedule import Interpolation
    from qilisdk.core.variables import Parameter
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend
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

    te = TimeEvolution(
        schedule=schedule,
        observables=[h1],
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
    results = backend.execute(vp)
    schedule.draw(ScheduleStyle(title="Schedule After Optimization"))
    print(results)
