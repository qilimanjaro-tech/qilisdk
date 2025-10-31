Cost Functions
==============

The :mod:`qilisdk.cost_functions` module turns the raw outputs returned by functionals into single scalar values that
optimizers can minimize. Each cost function inspects the :class:`~qilisdk.functionals.functional_result.FunctionalResult`
produced by a backend and evaluates a problem-specific metric such as an observable expectation value or the energy of
an abstract optimization model.

You will typically use cost functions to:

- score intermediate iterations inside a :class:`~qilisdk.functionals.variational_program.VariationalProgram`
- post-process :class:`~qilisdk.functionals.sampling_result.SamplingResult` samples into meaningful costs
- translate :class:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult` states into expectation values


CostFunction Interface
----------------------

The abstract :class:`~qilisdk.cost_functions.cost_function.CostFunction` base class exposes a single public method,
:meth:`~qilisdk.cost_functions.cost_function.CostFunction.compute_cost`. It dispatches on the concrete type of the
functional result and calls the appropriate protected hook:

- :meth:`~qilisdk.cost_functions.cost_function.CostFunction._compute_cost_sampling` receives a
  :class:`~qilisdk.functionals.sampling_result.SamplingResult`
- :meth:`~qilisdk.cost_functions.cost_function.CostFunction._compute_cost_time_evolution` receives a
  :class:`~qilisdk.functionals.time_evolution_result.TimeEvolutionResult`

If your workflow introduces new functional result types you can subclass :class:`CostFunction` and register additional
handlers in ``self._handlers`` or override the protected methods above. Custom implementations must return a real or
complex number that represents the score you want to optimize.


ObservableCostFunction
----------------------

The :class:`~qilisdk.cost_functions.observable_cost_function.ObservableCostFunction` evaluates the expectation value of
an observable against either a final state (from time evolution) or the probability distribution of sampled bitstrings.
It accepts three interchangeable representations for the observable: a
:class:`~qilisdk.core.qtensor.QTensor`, a symbolic
:class:`~qilisdk.analog.hamiltonian.Hamiltonian`, or a single
:class:`~qilisdk.analog.hamiltonian.PauliOperator`. Internally, all inputs are converted to ``QTensor`` so the same
numeric pipeline can be reused for both analog and digital results.

Example: expectation value from time evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.backends import QutipBackend
    from qilisdk.core import ket
    from qilisdk.cost_functions import ObservableCostFunction
    from qilisdk.functionals import TimeEvolution

    # Build a linear interpolation between driver and problem Hamiltonians
    T = 10.0
    dt = 1
    schedule = Schedule(T=T, dt=dt)
    schedule.add_hamiltonian("driver", X(0), lambda t: 1 - t / ((T - dt)/dt))
    schedule.add_hamiltonian("problem", Z(0), lambda t: t / ((T - dt)/dt))

    schedule.draw()

    functional = TimeEvolution(
        schedule=schedule,
        initial_state=(ket(0) - ket(1)).unit(),
        observables=[Z(0)],
    )

    backend = QutipBackend()
    evolution_result = backend.execute(functional)


    cost_fn = ObservableCostFunction(Z(0))
    energy = cost_fn.compute_cost(evolution_result)
    print("Expectation value ⟨Z⟩ =", energy)

For sampling workflows, the cost function iterates through the probability distribution exposed by
:meth:`~qilisdk.functionals.sampling_result.SamplingResult.get_probabilities` and accumulates the expectation value in
the computational basis.


ModelCostFunction
-----------------

The :class:`~qilisdk.cost_functions.model_cost_function.ModelCostFunction` bridges classical optimization models with
quantum result objects. It accepts any :class:`~qilisdk.core.model.Model` (including convenience subclasses such as
:class:`~qilisdk.core.model.QUBO`) and evaluates it against measured bitstrings or the amplitudes of a final quantum
state.

When the provided model is a :class:`~qilisdk.core.model.QUBO`, the cost function automatically converts it into a
Hamiltonian and computes the expectation value. Otherwise, it maps each sample to the model's variables, feeds them
through :meth:`~qilisdk.core.model.Model.evaluate`, and aggregates the resulting objective and constraint values.

Example: scoring samples from a variational circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.core.model import Model
    from qilisdk.core.variables import BinaryVariable, LEQ
    from qilisdk.cost_functions import ModelCostFunction
    from qilisdk.digital import Circuit, RX, RZ, CNOT, M
    from qilisdk.functionals import Sampling
    import numpy as np

    # Simple 2-qubit ansatz
    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))
    circuit.add(RZ(1, phi=np.pi / 3))
    circuit.add(M(0))
    circuit.add(M(1))

    sampling = Sampling(circuit, nshots=1_000)

    # Build a toy knapsack-like model
    b0, b1 = (BinaryVariable("b0"), BinaryVariable("b1"))
    model = Model("toy")
    model.set_objective(2 * b0 + 3 * b1, label="obj")
    model.add_constraint("limit", LEQ(b0 + b1, 1))

    cost_fn = ModelCostFunction(model)

    backend = CudaBackend()
    backend_result = backend.execute(sampling)
    score = cost_fn.compute_cost(backend_result)
    print("Aggregated model evaluation =", score)


Cost Functions in Variational Programs
--------------------------------------

Variational workflows combine a parameterized functional, a classical optimizer, and a cost function. At each optimizer
iteration the backend executes the functional, obtains a result object, and feeds it into the configured cost function
to obtain the scalar that drives the optimization loop.

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.cost_functions import ModelCostFunction
    from qilisdk.functionals import VariationalProgram, Sampling
    from qilisdk.optimizers import SciPyOptimizer

    variational_program = VariationalProgram(
        functional=Sampling(ansatz),          # parameterized circuit or schedule
        optimizer=SciPyOptimizer(method="Powell"),
        cost_function=ModelCostFunction(model),
    )

    backend = CudaBackend()
    result = backend.execute(variational_program)
    print("Optimal parameters:", result.optimal_parameters)
    print("Optimal cost:", result.optimal_cost)

Swapping the cost function lets you explore alternative objective definitions without touching the functional itself.
For example, you can start with :class:`ObservableCostFunction` to reproduce a physics-inspired energy expectation and
later try :class:`ModelCostFunction` to include constraint penalties from a combinatorial problem.
