Overview
==============

The :mod:`~qilisdk.cost_functions` module turns the raw outputs returned by functionals into single scalar values that
optimizers can minimize. Each cost function inspects the :class:`~qilisdk.functionals.functional_result.FunctionalResult`
produced by a backend and evaluates a problem-specific metric such as an observable expectation value or the energy of
an abstract optimization model.

Cost functions are used to:

- score intermediate iterations inside a :class:`~qilisdk.functionals.variational_program.VariationalProgram`
- post-process :class:`~qilisdk.functionals.functional_result.FunctionalResult` samples into meaningful costs
- translate :class:`~qilisdk.functionals.functional_result.FunctionalResult` states into expectation values


CostFunction Interface
----------------------

The abstract :class:`~qilisdk.cost_functions.cost_function.CostFunction` base class exposes a single public method,
:meth:`~qilisdk.cost_functions.cost_function.CostFunction.compute_cost`. It dispatches on the concrete type of the
functional result and calls the appropriate protected hook:

- It dispatches on the content of the :class:`~qilisdk.functionals.functional_result.FunctionalResult`, inspecting
  the available readout results (samples, expectation values, final state) to compute the cost.

If your workflow introduces new functional result types you can subclass :class:`~qilisdk.cost_functions.cost_function.CostFunction` and register additional
handlers in ``self._handlers`` or override the protected methods above. Custom implementations must return a real or
complex number that represents the score you want to optimize.


Cost Functions in Variational Programs
--------------------------------------

Variational workflows combine a parameterized :class:`~qilisdk.functionals.functional.Functional`, a classical optimizer, 
and a :class:`~qilisdk.cost_functions.cost_function.CostFunction`. At each optimizer iteration the backend executes 
the functional, obtains a :class:`~qilisdk.functionals.functional_result.FunctionalResult` 
object, and feeds it into the configured cost function to obtain the scalar that drives the optimization loop.

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.cost_functions import ModelCostFunction
    from qilisdk.functionals import VariationalProgram, DigitalPropagation
    from qilisdk.readout import Readout
    from qilisdk.optimizers import SciPyOptimizer
    from qilisdk.digital import Circuit, RX
    from qilisdk.core import Parameter

    param = Parameter("a", 0.3)
    ansatz = Circuit(1)
    ansatz.add(RX(0, theta=param))

    # Build a toy knapsack-like model
    b0 = BinaryVariable("b0")
    model = Model("toy")
    model.set_objective(2 * b0, label="obj")

    variational_program = VariationalProgram(
        functional=DigitalPropagation(ansatz),
        optimizer=SciPyOptimizer(method="Powell"),
        cost_function=ModelCostFunction(model),
    )

    backend = QiliSim()
    result = backend.execute(variational_program, readout=Readout().with_sampling(nshots=1000))
    print("Optimal parameters:", result.optimal_parameters)
    print("Optimal cost:", result.optimal_cost)

Swapping the cost function lets you explore alternative objective definitions without touching the functional itself.
For example, you can start with :class:`~qilisdk.cost_functions.observable_cost_function.ObservableCostFunction` to reproduce a physics-inspired energy expectation and
later try :class:`~qilisdk.cost_functions.model_cost_function.ModelCostFunction` to include constraint penalties from a combinatorial problem.