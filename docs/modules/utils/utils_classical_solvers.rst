Classical Solvers
=================

:mod:`qilisdk.utils.classical_solvers` wraps a handful of purely classical optimizers.
They exist so that an optimization problem written as a :class:`~qilisdk.core.model.Model`
can be solved without a backend, giving you a reference solution to compare a quantum result against.

Every solver subclasses :class:`~qilisdk.utils.classical_solvers.base_solver.ClassicalSolver` and
exposes a single :meth:`~qilisdk.utils.classical_solvers.base_solver.ClassicalSolver.solve` method
that takes a model and returns a
:class:`~qilisdk.utils.classical_solvers.base_solver.ClassicalSolverResult`.

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import BruteForceSolver

    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 2], max_weight=4)
    result = BruteForceSolver().solve(model)
    print(result)

**Output**::

    ClassicalSolverResult(objective=-7.0, sample={b0: 0, b1: 1, b2: 1}, results={'obj': -7.0, 'weight': 0.0})

Available Solvers
-----------------

.. table::
   :align: left
   :widths: auto

   ============================================================================================== ==================== ================================================
   Solver                                                                                         Accepts              Finds the global optimum
   ============================================================================================== ==================== ================================================
   :class:`~qilisdk.utils.classical_solvers.brute_force_solver.BruteForceSolver`                   any ``Model``        ✔ (exponential cost)
   ---------------------------------------------------------------------------------------------- -------------------- ------------------------------------------------
   :class:`~qilisdk.utils.classical_solvers.scipy_solver.ScipySolver`                              any ``Model``        ✕ (local minimizer by default)
   ---------------------------------------------------------------------------------------------- -------------------- ------------------------------------------------
   :class:`~qilisdk.utils.classical_solvers.simulated_annealing_solver.SimulatedAnnealingSolver`   ``QUBO`` only        ✕ (heuristic)
   ---------------------------------------------------------------------------------------------- -------------------- ------------------------------------------------
   :class:`~qilisdk.utils.classical_solvers.scip_solver.ScipSolver`                                any ``Model``        ✔ (requires the ``scip`` extra)
   ============================================================================================== ==================== ================================================

BruteForceSolver
^^^^^^^^^^^^^^^^

:class:`~qilisdk.utils.classical_solvers.brute_force_solver.BruteForceSolver` enumerates every
assignment of every variable and keeps the best one. A :class:`~qilisdk.core.variables.BinaryVariable`
contributes the values ``{0, 1}``; any other :class:`~qilisdk.core.variables.Variable` is decomposed
through its binary encoding, so the search covers every value that encoding can represent. Constraint
penalties are added to the objective when ranking candidates, so the solution returned is the best
*feasible* one whenever the Lagrange multipliers are large enough.

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import BruteForceSolver

    model = Model.random_ising(4)
    result = BruteForceSolver().solve(model)

.. warning::

    The cost grows exponentially with the number of variables, so use it for small reference problems only. 
    A variable with neither a binary nor a bounded domain has no encoding to enumerate and raises a ``ValueError``.

ScipySolver
^^^^^^^^^^^

:class:`~qilisdk.utils.classical_solvers.scipy_solver.ScipySolver` minimizes the model's objective
with SciPy, reusing the same :class:`~qilisdk.optimizers.scipy_optimizer.SciPyOptimizer`
that drives the variational algorithms.

The ``method`` argument selects the SciPy routine, and any further keyword argument is forwarded to
``scipy.optimize.minimize`` (or to the corresponding global optimizer).

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import ScipySolver

    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 2], max_weight=4)
    result = ScipySolver(method="l-bfgs-b").solve(model)
    print(result.objective, result.sample)

**Output**::

    -0.0 {b0: 0, b1: 0, b2: 0}

SimulatedAnnealingSolver
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.utils.classical_solvers.simulated_annealing_solver.SimulatedAnnealingSolver` runs
simulated annealing implemented in C++. It solves a :class:`~qilisdk.core.model.QUBO` and rejects
anything else, so a general model must be converted with :meth:`~qilisdk.core.model.Model.to_qubo`
first.

Configuration options:

- **num_reads**: The number of independent anneals to run, the best of which is returned. Defaults to 10.
- **num_sweeps**: The number of sweeps over all variables in each anneal. Defaults to 1000.
- **beta_range**: The ``(initial, final)`` inverse temperature to anneal over. If not given, a range is derived from the magnitudes of the cost function's coefficients. Defaults to ``None``.
- **seed**: The seed of the random number generators, each read deriving its own from it. Defaults to 0.
- **num_threads**: The number of threads to distribute the reads over, or zero to let OpenMP decide. Defaults to 0.

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import SimulatedAnnealingSolver

    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 2], max_weight=4)
    result = SimulatedAnnealingSolver(num_reads=100, seed=42).solve(model.to_qubo())
    print(result.objective, result.sample)

**Output**::

    -7.0 {b0: 0, b1: 1, b2: 1, weight_slack(1): 0, weight_slack(0): 0, weight_slack(2): 0}

ScipSolver
^^^^^^^^^^

:class:`~qilisdk.utils.classical_solvers.scip_solver.ScipSolver` hands the model to
`SCIP <https://www.scipopt.org/>`__, a mixed-integer programming solver, through ``pyscipopt``. It
solves to global optimality and is the practical choice whenever brute force becomes too slow.

The ``pyscipopt`` dependency is optional; install it with the ``scip`` extra:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            pip install qilisdk[scip]

    .. group-tab:: Mac OSX

        .. code-block:: bash

            pip install "qilisdk[scip]"

    .. group-tab:: Windows

        .. code-block:: bash

            pip install qilisdk[scip]

Unlike the other solvers, :meth:`~qilisdk.utils.classical_solvers.scip_solver.ScipSolver.solve`
takes two extra keyword arguments: ``verbose`` to show SCIP's own output, and ``params`` to forward
parameters to ``pyscipopt.Model.setParams``.

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import ScipSolver

    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 2], max_weight=4)
    result = ScipSolver().solve(model, params={"limits/time": 60})
    print(result.objective, result.sample)

**Output**::

    -7.0 {b0: 0, b1: 1, b2: 1}

Reading the Result
------------------

Every solver returns a
:class:`~qilisdk.utils.classical_solvers.base_solver.ClassicalSolverResult`, which is the model
evaluated at the solution that was found:

- ``result.objective`` - the value of the model's objective at the solution.
- ``result.constraints`` - the value of each of the model's constraints, keyed by label. A constraint evaluates to zero when it is satisfied, and to its penalty otherwise.
- ``result.results`` - the objective and every constraint together, keyed by label.
- ``result.sample`` - the value each of the model's variables takes in the solution.

.. code-block:: python

    from qilisdk.core import Model
    from qilisdk.utils.classical_solvers import BruteForceSolver

    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 2], max_weight=4)
    result = BruteForceSolver().solve(model)

    print(result.objective)
    print(result.constraints)
    print(result.sample)

**Output**::

    -7.0
    {'weight': 0.0}
    {b0: 0, b1: 1, b2: 1}

A non-zero constraint value means the solution violates that constraint, and the number is the
penalty it incurred.