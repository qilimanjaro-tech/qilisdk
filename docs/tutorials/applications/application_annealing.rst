Optimization with Quantum Annealing
===================================================

In this tutorial, we will explore how to use Quantum Annealing to solve a simple
optimization problem using QiliSDK.

.. note:: If you haven't had a look at these, it might be useful to check them out first:
    :doc:`Quantum Basics </tutorials/introductions/intro_quantum>`, 
    :doc:`Quantum Circuits </tutorials/introductions/intro_circuits>` and
    :doc:`Quantum Annealing </tutorials/introductions/intro_annealing>`.

The Problem
----------------------

.. include:: ../../shared/team_building.rst

The Solution
----------------------

Now that we have our problem formulated as a QUBO, we can use quantum annealing to find an approximate solution to it.
Quantum annealing is a metaheuristic optimization algorithm that is inspired by the process of annealing
in metallurgy, where a material is heated and then slowly cooled to remove defects and find a low-energy state.
In quantum annealing, we start with a simple Hamiltonian whose ground state is easy to prepare
and then slowly evolve it into a more complex Hamiltonian that encodes the optimization problem we want to solve.
The hope is that if we do this slowly enough, the system will remain in its ground state throughout the evolution, 
and thus end up in the ground state of the final Hamiltonian, which corresponds
to the optimal solution of our problem.

Assuming we have reformulated our problem into a QUBO, we can construct the problem Hamiltonian for quantum annealing,
which is given by:

.. math:: 

    H_{prob} = \sum_{i,j} c_{ij} Z(i) Z(j)

Where :math:`Z(i)` is the Pauli-Z operator acting on qubit :math:`i`, and :math:`c_{ij}` are the coefficients from our QUBO formulation.

Our mixing Hamiltonian is typically chosen to be the transverse field Hamiltonian, which is given by:

.. math:: 

    H_{mix} = - \sum_i X(i)

The overall time-dependent Hamiltonian that we evolve is then given by:

.. math:: 

    H(t) = A(t) H_{mix} + B(t) H_{prob}

Where :math:`A(t)` and :math:`B(t)` are functions that determine how we interpolate 
between the mixing Hamiltonian and the problem Hamiltonian over time.
For simplicity here will just assume that we do a linear interpolation, 
such that :math:`A(t) = 1 - t/T` and :math:`B(t) = t/T`, where :math:`T` is the total annealing time.

The Implementation
----------------------

To simulate the evolution of this time-dependent Hamiltonian, first we need to form our problem:

.. code-block:: python

    from qilisdk.core.model import QUBO, ObjectiveSense
    from qilisdk.core.variables import BinaryVariable, EQ

    num_people = 4
    vars = [BinaryVariable(f"x{i}") for i in range(num_people)]
    preferences = [[0, 1, 3, 4],
                  [1, 0, 5, 2],
                  [3, 5, 0, 6],
                  [4, 2, 6, 0]]
    model = QUBO("team_formation_example")
    team_1 = sum(preferences[i][j] * vars[i] * vars[j] for i in range(num_people) for j in range(i+1, num_people))
    team_0 = sum(preferences[i][j] * (1 - vars[i]) * (1 - vars[j]) for i in range(num_people) for j in range(i+1, num_people))
    model.set_objective(team_0 + team_1, label="obj", sense=ObjectiveSense.MAXIMIZE)
    model.add_constraint("team_size_constraint", EQ(sum(vars[i] for i in range(num_people)), 2), lagrange_multiplier=10)

Then we use the model to form our problem Hamiltonian, and we can also define our mixing Hamiltonian and the schedule for our evolution:


.. code-block:: python

    from qilisdk.analog import X
    from qilisdk.analog import Schedule
    from qilisdk.core import QTensor

    problem_hamiltonian = -model.to_hamiltonian()
    mixer_hamiltonian = sum(-1.0 * X(i) for i in range(num_people))

We also need to define the schedule - how the coefficients of the problem and mixing Hamiltonians change over time:

.. code-block:: python

    T = 100.0
    schedule = Schedule(
                    hamiltonians={"problem": problem_hamiltonian, "mixer": mixer_hamiltonian},
                    coefficients={
                        "mixer": {(0, T): lambda t: 1 - t / T},
                        "problem": {(0, T): lambda t: t / T},
                    },
                    dt=0.01,
                )

We will start in the ground state of our mixing Hamiltonian, which is the equal superposition state over all possible team assignments:

.. code-block:: python

    initial_state = (QTensor.ket(1, 0, 0, 0) + QTensor.ket(0, 1, 0, 0) + QTensor.ket(0, 0, 1, 0) + QTensor.ket(0, 0, 0, 1)).normalized()

Finally, we initialize our quantum simulator, execute the evolution, and read out the results:

.. code-block:: python

    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim

    backend = QiliSim()
    evolution = AnalogEvolution(schedule=schedule, initial_state=initial_state)
    readout = Readout().with_sampling(1000).with_expectation([problem_hamiltonian])
    results = backend.execute(evolution, readout)
    print("Results:", results)

This will print something like the following:

.. code-block:: none

    samples={
        '0000': 316,
        '0100': 49,
        '0110': 129,
        '0111': 51,
        '1000': 91,
        '1001': 162,
        '1011': 48,
        '1110': 5,
        '1111': 149
    }


As you can see from these results, the most common samples are those which satisfy the constraint (i.e. have exactly two 1's), 
and among those, the most common ones are those which have a high objective value. In this case there is some leakage into other
states like 0000 and 1111, but this is expected since quantum annealing is a heuristic algorithm and does not always find the optimal solution.

Further Reading
--------------------

- `QUBO`_
- `Quantum Annealing`_

.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization
.. _Quantum Annealing: https://en.wikipedia.org/wiki/Quantum_annealing
