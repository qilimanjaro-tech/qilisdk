ObservableCostFunction
----------------------

The :class:`~qilisdk.cost_functions.observable_cost_function.ObservableCostFunction` evaluates the expectation value of
an observable against either a final state (from time evolution) or the probability distribution of sampled bitstrings.
It accepts three interchangeable representations for the observable: a
:class:`~qilisdk.core.qtensor.QTensor`, a symbolic
:class:`~qilisdk.analog.hamiltonian.Hamiltonian`, or a single
:class:`~qilisdk.analog.hamiltonian.PauliOperator`. Internally, all inputs are converted to :class:`~qilisdk.core.qtensor.QTensor` so the same
numeric pipeline can be reused for both analog and digital results.

**Example: expectation value from analog evolution**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.backends import QiliSim
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout
    from qilisdk.cost_functions import ObservableCostFunction

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

    functional = AnalogEvolution(
        schedule=schedule,
        initial_state=(ket(0) - ket(1)).unit(),
    )

    backend = QiliSim()
    evolution_result = backend.execute(functional, readout=Readout().with_sampling(nshots=1000))

    cost_fn = ObservableCostFunction(Z(0))
    energy = cost_fn.compute_cost(evolution_result)
    print("Expectation value <Z> =", energy)

For sampling workflows, the cost function iterates through the probability distribution exposed by
:attr:`~qilisdk.functionals.functional_result.FunctionalResult.probabilities` and accumulates the expectation value in
the computational basis.

