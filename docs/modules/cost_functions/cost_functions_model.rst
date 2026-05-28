ModelCostFunction
-----------------

The :class:`~qilisdk.cost_functions.model_cost_function.ModelCostFunction` bridges classical optimization models with
quantum result objects. It accepts any :class:`~qilisdk.core.model.Model` (including convenience subclasses such as
:class:`~qilisdk.core.model.QUBO`) and evaluates it against measured bitstrings or the amplitudes of a final quantum
state.

When the provided model is a :class:`~qilisdk.core.model.QUBO`, the cost function automatically converts it into a
Hamiltonian and computes the expectation value. Otherwise, it maps each sample to the model's variables, feeds them
through :meth:`~qilisdk.core.model.Model.evaluate`, and aggregates the resulting objective and constraint values.

**Example: scoring samples from a variational circuit**

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.core.model import Model
    from qilisdk.core.variables import BinaryVariable, LEQ
    from qilisdk.cost_functions import ModelCostFunction
    from qilisdk.digital import Circuit, RX, RZ, CNOT, M
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout
    import numpy as np

    # Simple 2-qubit ansatz
    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 2))
    circuit.add(CNOT(0, 1))
    circuit.add(RZ(1, phi=np.pi / 3))
    circuit.add(M(0))
    circuit.add(M(1))

    functional = DigitalPropagation(circuit)

    # Build a toy knapsack-like model
    b0, b1 = (BinaryVariable("b0"), BinaryVariable("b1"))
    model = Model("toy")
    model.set_objective(2 * b0 + 3 * b1, label="obj")
    model.add_constraint("limit", LEQ(b0 + b1, 1))

    cost_fn = ModelCostFunction(model)

    backend = QiliSim()
    backend_result = backend.execute(functional, readout=Readout().with_sampling(nshots=1_000))
    score = cost_fn.compute_cost(backend_result)
    print("Aggregated model evaluation =", score)

