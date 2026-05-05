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

**Usage Example (Using QiliSim Backend)**

.. code-block:: python

    import numpy as np

    from qilisdk.backends import QiliSim
    from qilisdk.core.model import Model, ObjectiveSense
    from qilisdk.core.variables import LEQ, BinaryVariable
    from qilisdk.cost_functions.model_cost_function import ModelCostFunction
    from qilisdk.digital import CNOT, U3, HardwareEfficientAnsatz
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.functionals.variational_program import VariationalProgram
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.readout import Readout


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

    backend = QiliSim()
    result = backend.execute(
        VariationalProgram(
            functional=DigitalPropagation(ansatz),
            optimizer=optimizer,
            cost_function=ModelCostFunction(model),
        ),
        Readout().with_sampling(nshots=1000),
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


**Usage Example 2 (Using QiliSim Backend)**
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
    from qilisdk.backends import QiliSim
    from qilisdk.readout import Readout
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

    backend = QiliSim()
    results = backend.execute(vp, Readout().with_expectation(observables=[h1]).with_state_tomography().with_sampling(nshots=1000))
    schedule.draw(ScheduleStyle(title="Schedule After Optimization"))
    print(results)
