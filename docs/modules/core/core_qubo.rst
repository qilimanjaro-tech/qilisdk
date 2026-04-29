QUBO
^^^^^^^^^^^

The :class:`~qilisdk.core.model.QUBO` subclass specializes in **Quadratic Unconstrained Binary Optimization** models, 
where every decision variable is binary and the objective function is at most quadratic. 
Unlike general models, “hard” constraints are not maintained separately but are encoded directly into the objective as penalty terms. 
The strength of each penalty is controlled by its associated Lagrange multiplier.

The binary quadratic cost function that defines a QUBO
problem is written as

.. math::

    f(x) = \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} q_{ij} x_i x_j,

with decision variables :math:`x_i \in \{0, 1\}` and symmetric coefficients :math:`q_{ij} = q_{ji} \in \mathbb{R}`. 
Separating the diagonal terms highlights the effective linear
weights:

.. math::

    f(x) = \sum_{i=1}^{n-1} \sum_{j>i} q_{ij} x_i x_j + \sum_{i=1}^{n} q_{ii} x_i.


Adding Constraints as Penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since QUBO is unconstrained, every constraint is converted into the objective via a **penalty term**. 
Linear equality constraints can be described as,

.. math::

    \sum_{i=1}^{n} c_i x_i = C, \quad c_i \in \mathbb{Z},

where :math:`c_i` are integer coefficients and :math:`C` is an integer constant.

Embedded as penalties, these constraints give the penalized objective

.. math::

    \min_{x,\,s} \left(
        \sum_{i=1}^{n-1} \sum_{j>i} c_{ij} x_i x_j
        + \sum_{i=1}^{n} h_i x_i
        + \lambda_0 \left( \sum_{i=1}^{n} q_i x_i - C \right)^{2}  
    \right),

where :math:`\lambda_0 > 0` is a penalty strength parameter.

Ineqaulity constraints can be defined as:

.. math::

    \sum_{i=1}^{n} \ell_i x_i \leq B, \quad \ell_i \in \mathbb{Z}.

To translate these into penalties, two strategies are supported:

- **Slack penalization** (default):  
    Introduce additional binary slack variables to turn inequalities into equalities, then square the residual. 
    Therefore, the penalty term becomes:

    .. math::

        \lambda_1 \left( B - \sum_{i=1}^{n} \ell_i x_i - \sum_{k=0}^{N-1} 2^{k} s_k \right)^{2}

    
    where :math:`s_k` are slack binary variables introduced to encode the inequality constraint (with the
    number of bits :math:`N` chosen so that their binary expansion spans the admissible slack range) and
    :math:`\lambda_{1}` control the penalty strength.

- **Unbalanced penalization**:  
    Directly penalize violation without slack variables using two weights (a, b) to scale positive and negative deviations differently [1]_.
    we define :math:`h(x) = B - \sum_{i=1}^{n} \ell_i x_i` as the signed residual of the constraint, and the penalty term becomes:

    .. math::
        
        - a h(x) + b h(x)^2,

    where :math:`a, b > 0` are parameters that control the penalty strength for violations above and below the bound, respectively.







Why QUBO?
~~~~~~~~~~

- **Unconstrained form**: Many quantum annealers and specialized solvers accept only unconstrained binary quadratic forms, making QUBO the lingua franca of quantum optimization.  
- **Penalty encoding**: Instead of throwing away constraint structure, you transform each constraint into a quadratic penalty, preserving problem fidelity.  
- **Direct mapping**: Once in QUBO form, you can directly translate the problem to Ising/Hamiltonian terms for hardware execution.

Defining a QUBO
~~~~~~~~~~~~~~~

1. **Model creation**  

.. code-block::  python

    from qilisdk.core.model import QUBO, ObjectiveSense  
    model = QUBO("qubo_example")  

2. **Objective**  

.. code-block::  python
    
    # sum of weights x minus risk penalty
    model.set_objective(5 * x1 + 3 * x2 - 2 * x1 * x2,
                        label="return_minus_risk",
                        sense=ObjectiveSense.MAXIMIZE)

.. warning::

   - If **transform_to_qubo=True**, your constraint **must be linear** (no quadratic terms), because it will be rewritten as (lhs-rhs)^2.  
   - If **transform_to_qubo=False**, you assume the constraint is already a valid quadratic penalty, and it will be added verbatim.

Example: Slack Penalization
'''''''''''''''''''''''''''

.. code-block:: python

    from qilisdk.core.model import QUBO, ObjectiveSense
    from qilisdk.core.variables import BinaryVariable, LEQ

    b, b2 = BinaryVariable("b"), BinaryVariable("b2")
    model = QUBO("slack_example")
    model.set_objective(b2 + 2 * b + 1, label="obj", sense=ObjectiveSense.MINIMIZE)

    # Enforce b <= 0.5 via slack, squared penalty in objective
    model.add_constraint("c1", LEQ(b + 2 * b2, 1), lagrange_multiplier=10, penalization="slack", transform_to_qubo=True)

    print(model.qubo_objective)

**Output**::

    obj: (-8.0) * b + b2 + (11.0) + (40.0) * (b2 * b) + (40.0) * (b2 * c1_slack(0)) + (20.0) * (b * c1_slack(0)) + (-10.0) * c1_slack(0)

|

Example: Unbalanced Penalization
''''''''''''''''''''''''''''''''

.. code-block:: python

    from qilisdk.core.model import QUBO, ObjectiveSense
    from qilisdk.core.variables import BinaryVariable, LEQ

    b, b2 = BinaryVariable("b"), BinaryVariable("b2")
    model = QUBO("unbalanced_penalization_example")
    model.set_objective(b2 + 2 * b + 1, label="obj", sense=ObjectiveSense.MINIMIZE)

    # Enforce b <= 0.5 via slack, squared penalty in objective
    model.add_constraint("c1", LEQ(b + 2 * b2, 1), lagrange_multiplier=1, penalization="unbalanced", transform_to_qubo=True)

    print(model.qubo_objective)

**Output**::

    obj: (2) * b + (3.0) * b2 + (1) + (4.0) * (b2 * b)


.. [1] Montañez-Barrera, Jhon Alejandro, et al. "Unbalanced penalization: A new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms." Quantum Science and Technology 9.2 (2024): 025022.

Interoperability
~~~~~~~~~~~~~~~~

- **Convert any Model to QUBO**
    If you have a generic :class:`~qilisdk.core.model.Model` with only linear/quadratic terms, you can automatically produce a QUBO:  

    .. code-block:: python

        qubo_model = model.to_qubo()

- **Export to Hamiltonian**  
    Once in QUBO form, translate directly into an analog Ising :class:`~qilisdk.core.hamiltonian.Hamiltonian` for simulation or hardware:  

    .. code-block:: python

        h = qubo_model.to_hamiltonian()

