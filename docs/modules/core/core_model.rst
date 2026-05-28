Models
--------

This minimal example introduces a binary optimization model, converts it to a
QUBO penalty form, and exports the corresponding Hamiltonian:

.. code-block:: python

    from qilisdk.core import BinaryVariable, LEQ, Model, ObjectiveSense
    from qilisdk.core.model import QUBO

    x0, x1 = BinaryVariable("x0"), BinaryVariable("x1")

    model = Model("toy")
    model.set_objective(-2 * x0 - 3 * x1 + 4 * x0 * x1,
                        label="energy",
                        sense=ObjectiveSense.MINIMIZE)
    model.add_constraint("budget", LEQ(x0 + x1, 1), lagrange_multiplier=5)

    qubo = model.to_qubo()
    print(qubo.qubo_objective)

    ham = qubo.to_hamiltonian()
    print(ham)

Objectives and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~qilisdk.core.model.Model` consists of:

- A single :class:`~qilisdk.core.model.Objective`
- Zero or more :class:`~qilisdk.core.model.Constraint` instances

**Objective**

The :class:`~qilisdk.core.model.Objective` defines the function the model aims to minimize or maximize. Example:

.. code-block:: python

    from qilisdk.core.model import Model, ObjectiveSense
    from qilisdk.core.variables import Variable, Domain, Bitwise
    model = Model("example_model")
    x = Variable("x", domain=Domain.REAL, bounds=(1, 2), encoding=Bitwise, precision=1e-1)
    model.set_objective(2*x + 3, label="obj", sense=ObjectiveSense.MINIMIZE)
    print(model)

**Output**:

::

    Model name: example_model 
    objective (obj): 
        minimize : 
        (2) * x + (3) 

    subject to the encoding constraint/s: 
        x_upper_bound_constraint: x <= (2) 
        x_lower_bound_constraint: x >= (1) 
    
    With Lagrange Multiplier/s: 
        x_upper_bound_constraint : 100 
        x_lower_bound_constraint : 100 

Encoding constraints are automatically added for bounded continuous variables. Each constraint has an associated Lagrange multiplier, which determines the penalty for violating it.

You can update the multiplier like so:

.. code-block:: python

    model.set_lagrange_multiplier("x_upper_bound_constraint", 1)
    print(model)

**Output**:

::

    Model name: example_model 
    objective (obj): 
        minimize : 
        (2) * x + (3) 
        
    subject to the encoding constraint/s: 
        x_upper_bound_constraint: x <= (2) 
        x_lower_bound_constraint: x >= (1) 

    With Lagrange Multiplier/s: 
        x_upper_bound_constraint : 1 
        x_lower_bound_constraint : 100 

**Constraints**

Additional :class:`Constraints<qilisdk.core.model.Constraint>` can be added to restrict the solution space:

.. code-block:: python

    from qilisdk.core import LT

    model.add_constraint("test_constraint", LT(x, 1.5), lagrange_multiplier=10)
    print(model)

**Output**:

::

    Model name: example_model 
    objective (obj): 
        minimize : 
        (2) * x + (3) 

    subject to the constraint/s: 
        test_constraint: x < (1.5) 

    subject to the encoding constraint/s: 
        x_upper_bound_constraint: x <= (2) 
        x_lower_bound_constraint: x >= (1) 

    With Lagrange Multiplier/s: 
        x_upper_bound_constraint : 1 
        x_lower_bound_constraint : 100 
        test_constraint : 10 


Evaluating a Model
^^^^^^^^^^^^^^^^^^

To evaluate a :class:`~qilisdk.core.model.Model`, provide values for all involved variables:

.. code-block:: python

    model.evaluate({
        x: 1.4
    })

**Output**:

::

    {'obj': 5.8, 'test_constraint': 0.0}

The evaluation returns a dictionary with values for the objective and constraints. A constraint returns `0.0` if satisfied, or 
its Lagrange multiplier if violated.

For example:

.. code-block:: python

    model.evaluate({
        x: 2
    })

**Output**:

::

    {'obj': 7.0, 'test_constraint': 10.0}

