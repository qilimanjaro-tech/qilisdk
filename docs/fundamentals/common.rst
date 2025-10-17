Common
======

The :mod:`qilisdk.common` layer underpins both the digital and analog stacks. It
provides symbolic variables, optimization models, sparse quantum tensors, and
the ``Parameterizable`` mixin used throughout the SDK.

Highlights:

- :mod:`~qilisdk.common.variables` supplies binary, spin, continuous, and
  parameter variables plus algebraic helpers (:class:`~qilisdk.common.variables.Term`,
  comparison factories, encodings).
- :mod:`~qilisdk.common.model` builds constrained optimization programs and
  offers tools to automatically convert the model to :class:`~qilisdk.common.model.QUBO` format if the constraints are linear.
- :mod:`~qilisdk.common.qtensor` manages sparse quantum objects and utilities
  such as :func:`~qilisdk.common.qtensor.tensor_prod` and :func:`~qilisdk.common.qtensor.expect_val`.
- :mod:`~qilisdk.common.parameterizable.Parameterizable` standardizes how
  objects expose symbolic parameters (shared by circuits, schedules, etc.).

Quick Start
-----------

This minimal example introduces a binary optimization model, converts it to a
QUBO penalty form, and exports the corresponding Hamiltonian:

.. code-block:: python

    from qilisdk.common import BinaryVariable, LEQ, Model, ObjectiveSense
    from qilisdk.common.model import QUBO

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


Models
------

The core components for model construction are defined in the :mod:`~qilisdk.common.variables` module.

Variables
^^^^^^^^^

This module offers tools to define different types of variables:

- **Binary Variables** (:class:`~qilisdk.common.variables.BinaryVariable`)
- **Spin Variables** (:class:`~qilisdk.common.variables.SpinVariable`)
- **Continuous Variables** (:class:`~qilisdk.common.variables.Variable`) — with the following customizable parameters:

    - **Domain** (:class:`~qilisdk.common.variables.Domain`): Specifies the variable type:

        - ``REAL``
        - ``INTEGER``
        - ``POSITIVE_INTEGER``
        - ``BINARY``
        - ``SPIN``
    - **Bounds**: Defines the allowed value range of the variable.
    - **Encoding**: Specifies how the variable is represented using binary encodings:

        - Bit-wise encoding (:class:`~qilisdk.common.variables.BitWise`)
        - Domain wall encoding (:class:`~qilisdk.common.variables.DomainWall`)
        - One-hot encoding (:class:`~qilisdk.common.variables.OneHot`)
    - **Precision**: Applicable to ``REAL`` domain; defines the resolution (e.g., floating-point precision).

Example: creating different types of variables:

.. code-block:: python

    from qilisdk.common.variables import BinaryVariable, Bitwise, Domain, SpinVariable, Variable

    x = Variable("x", domain=Domain.REAL, bounds=(1, 2), encoding=Bitwise, precision=1e-1)
    s = SpinVariable("s")
    b = BinaryVariable("b")

Continuous variables support indexing, where each index refers to a component of the binary-encoded form of the variable. For example:

.. code-block:: python

    print(x.to_binary())

**Output**:

::

    (0.1) * x(0) + (0.2) * x(1) + (0.4) * x(2) + (0.30000000000000004) * x(3) + (1.0)

To index the first binary variable from the binary representation of x you can write: ``x[0]``.
Each binary variable configuration generates a float within the bounds, based on the defined precision. For instance:

.. code-block:: python

    x.evaluate([0, 1, 0, 0])

**Output**:

::

    1.2

Terms
^^^^^

Variables can be combined algebraically to form expressions known as :class:`~qilisdk.common.variables.Term`. Example:

.. code-block:: python

    t1 = 2 * x + 3
    print("t1:", t1)
    t2 = 3 * x**2 + 2 * x + 4
    print("t2:", t2)
    t3 = 2 * x + b - 1
    print("t3:", t3)
    t4 = t1 - t2
    print("t4:", t4)

**Output**:

::

    t1: (2) * x + (3)
    t2: (3) * (x^2) + (2) * x + (4)
    t3: (2) * x + b + (-1)
    t4: (-1.0) + (-3.0) * (x^2)

Terms can be evaluated by providing values for the involved variables:

.. code-block:: python

    t3.evaluate({
        x: 1.5,
        b: 0
    })

**Output**:

::

    2.0

.. warning::

    To evaluate a term, all participating variables must be assigned valid values within their respective domains and bounds.

Parameters and Parameterizable Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many components in QiliSDK expose symbolic parameters that can be optimized or
re-bound at runtime. The :class:`~qilisdk.common.variables.Parameter` class
represents a scalar symbol with optional bounds, and
:class:`~qilisdk.common.parameterizable.Parameterizable` provides a uniform API
(``get_parameter_names``, ``set_parameter_values``…) implemented by circuits,
schedules, models, and more.

.. code-block:: python

    from qilisdk.common import Parameter

    theta = Parameter("theta", value=0.5, bounds=(0.0, 1.0))
    print(theta.value)     # 0.5
    theta.set_value(0.75)
    print(theta.bounds)    # (0.0, 1.0)

Parameters behave like symbolic variables in algebraic expressions, so you can
combine them with other variables and evaluate terms without having to pass the
parameter explicitly—its stored ``value`` is used automatically.

Objects that inherit from :class:`~qilisdk.common.parameterizable.Parameterizable`
collect all the :class:`Parameter` instances they encounter. For example:

.. code-block:: python

    from qilisdk.digital import Circuit, RX

    circuit = Circuit(nqubits=1)
    circuit.add(RX(0, theta=theta))

    print(circuit.get_parameter_names())   # ['RX(0)_theta_0']
    print(circuit.get_parameter_values())  # [0.75]
    circuit.set_parameters({"RX(0)_theta_0": 0.9})

Whenever you interact with one of these parameterizable objects, the helper
methods let you list, bound, or update the symbolic degrees of freedom in a
consistent way.


Comparison Terms
^^^^^^^^^^^^^^^^

Comparison terms define constraints using mathematical comparisons. Use the following operators to construct them:

.. list-table::
   :class: longtable
   :header-rows: 1
   :widths: 20 20 20

   * - Comparison Operation
     - QiliSDK Method
     - Alias
   * - Equality
     - ``Equal(lhs, rhs)``
     - ``EQ(lhs, rhs)``
   * - Not Equal
     - ``NotEqual(lhs, rhs)``
     - ``NEQ(lhs, rhs)``
   * - Less Than
     - ``LessThan(lhs, rhs)``
     - ``LT(lhs, rhs)``
   * - Less Than or Equal
     - ``LessThanOrEqual(lhs, rhs)``
     - ``LEQ(lhs, rhs)``
   * - Greater Than
     - ``GreaterThan(lhs, rhs)``
     - ``GT(lhs, rhs)``
   * - Greater Than or Equal
     - ``GreaterThanOrEqual(lhs, rhs)``
     - ``GEQ(lhs, rhs)``

*Note*: `lhs` and `rhs` refer to the left-hand side and right-hand side expressions, respectively.

Example:

.. code-block:: python

    from qilisdk.common.variables import LT
    LT(2 * x - 1, 1)

**Output**:

::

    (2) * x < (2.0)

When a comparison term is created, constants are automatically moved to the right-hand side, and variable terms to the left-hand side.

Objectives and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~qilisdk.common.model.Model` consists of:

- A single :class:`~qilisdk.common.model.Objective`
- Zero or more :class:`~qilisdk.common.model.Constraint` instances

**Objective**

The objective defines the function the model aims to minimize or maximize. Example:

.. code-block:: python

    from qilisdk.common.model import Model, ObjectiveSense
    model = Model("example_model")
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

Additional constraints can be added to restrict the solution space:

.. code-block:: python

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

To evaluate a model, provide values for all involved variables:

.. code-block:: python

    model.evaluate({
        x: 1.4
    })

**Output**:

::

    {'obj': 5.8, 'test_constraint': 0.0}

The evaluation returns a dictionary with values for the objective and constraints. A constraint returns `0.0` if satisfied, or its Lagrange multiplier if violated.

For example:

.. code-block:: python

    model.evaluate({
        x: 2
    })

**Output**:

::

    {'obj': 7.0, 'test_constraint': 10.0}

QUBO Models
^^^^^^^^^^^

The :class:`~qilisdk.common.model.QUBO` subclass specializes in **Quadratic Unconstrained Binary Optimization** models, 
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

    from qilisdk.common.model import QUBO, ObjectiveSense  
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

    from qilisdk.common.model import QUBO, ObjectiveSense
    from qilisdk.common.variables import BinaryVariable, LEQ

    b, b2 = BinaryVariable("b"), BinaryVariable("b2")
    model = QUBO("slack_example")
    model.set_objective(b2 + 2 * b + 1, label="obj", sense=ObjectiveSense.MINIMIZE)

    # Enforce b <= 0.5 via slack, squared penalty in objective
    model.add_constraint("c1", LEQ(b + 2 * b2, 1), lagrange_multiplier=10, penalization="slack", transform_to_qubo=True)

    print(model.qubo_objective)

**Output**::

    obj: (-8.0) * b + b2 + (11.0) + (40.0) * (b2 * b) + (40.0) * (b2 * c1_slack(0)) + (20.0) * (b * c1_slack(0)) + (-10.0) * c1_slack(0)



Example: Unbalanced Penalization
''''''''''''''''''''''''''''''''

.. code-block:: python

    from qilisdk.common.model import QUBO, ObjectiveSense
    from qilisdk.common.variables import BinaryVariable, LEQ

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
    If you have a generic :class:`~qilisdk.common.model.Model` with only linear/quadratic terms, you can automatically produce a QUBO:  

    .. code-block:: python

        qubo_model = model.to_qubo()

- **Export to Hamiltonian**  
    Once in QUBO form, translate directly into an analog Ising Hamiltonian for simulation or hardware:  

    .. code-block:: python

        from qilisdk.analog.hamiltonian import Hamiltonian
        h = qubo_model.to_hamiltonian()

Quantum Objects
---------------

The :mod:`~qilisdk.common.qtensor` module defines the :class:`~qilisdk.common.qtensor.QTensor`
class and related helpers for representing and manipulating quantum states and
operators in sparse form.

The :class:`~qilisdk.common.qtensor.QTensor` wraps a dense NumPy array or SciPy sparse matrix into a CSR-format sparse matrix, and can represent:

- **Kets** (column vectors of shape ``(2**N, 1)``)  
- **Bras** (row vectors of shape ``(1, 2**N)``)  
- **Operators / Density Matrices** (square matrices of shape ``(2**N, 2**N)``)  
- **Scalars** (``(1, 1)`` matrices)  

Examples of creating various quantum objects:

.. code-block:: python

    import numpy as np
    from qilisdk.common.qtensor import QTensor

    # 1‑qubit |0> ket
    psi_ket = QTensor(np.array([[1], [0]]))
    print("Ket:", psi_ket.dense, "is_ket?", psi_ket.is_ket())
    print("-" * 20)

    # 1‑qubit <0| bra
    psi_bra = QTensor(np.array([[1, 0]]))
    print("Bra:", psi_bra.dense, "is_bra?", psi_bra.is_bra())
    print("-" * 20)

    # Density matrix |0><0|
    rho = QTensor(np.array([[1, 0], [0, 0]]))
    print("Density matrix:\n", rho.dense, "is_density_matrix?", rho.is_density_matrix())
    print("-" * 20)

    # Scalar 0.5
    scalar = QTensor(np.array([[0.5]]))
    print("Scalar:", scalar.dense, "is_scalar?", scalar.is_scalar())

**Output**

::

    Ket: [[1]
    [0]] is_ket? True
    --------------------
    Bra: [[1 0]] is_bra? True
    --------------------
    Density matrix:
    [[1 0]
    [0 0]] is_density_matrix? True
    --------------------
    Scalar: [[0.5]] is_scalar? True

Helper constructors
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qilisdk.common.qtensor import ket, bra, basis_state

    # Single‑qubit
    print("ket(0):\n", ket(0).dense, "\nis_ket?", ket(0).is_ket())
    print("bra(1):\n", bra(1).dense, "\nis_bra?", bra(1).is_bra())

    # Fock basis in N=4 Hilbert space
    print("basis_state(2,4):\n", basis_state(2, 4).dense, "\nshape:", basis_state(2, 4).shape)

**Output**

::

    ket(0):
    [[1.]
    [0.]] 
    is_ket? True
    bra(1):
    [[0. 1.]] 
    is_bra? True
    basis_state(2,4):
    [[0.]
    [0.]
    [1.]
    [0.]] 
    shape: (4, 1)

Quantum Object Properties & Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All data are stored sparsely, but you can retrieve dense or sparse views:

- ``.data``: sparse :class:`scipy.sparse.csr_matrix`  
- ``.dense``: full NumPy array

Key methods:

- ``.adjoint()``: conjugate transpose  
- ``.expm()``: matrix exponential  
- ``.norm(order=1)``: vector or matrix norm  
- ``.unit(order='tr')``: normalize to unit norm  
- ``.ptrace(keep, dims=None)``: partial trace  


Examples:

.. code-block:: python

    import numpy as np
    from qilisdk.common.qtensor import QTensor

    # Adjoint of a non-Hermitian operator
    A = QTensor(np.array([[1+1j, 2], [3, 4]]))
    A_dag = A.adjoint()
    print("A:\n", A.dense)
    print("A†:\n", A_dag.dense)

    # Matrix exponential of Pauli-X
    X = QTensor(np.array([[0, 1], [1, 0]]))
    expX = X.expm()
    print("exp(X):\n", np.round(expX.dense, 3))

    # Norm of a ket and a density matrix
    ket0 = QTensor(np.array([[1], [0]]))
    dm = ket0.to_density_matrix()
    print("||ket0|| =", ket0.norm())
    print("trace norm(dm) =", dm.norm(order='tr'))

    # Partial trace of a Bell state
    from qilisdk.common.qtensor import ket, tensor_prod
    bell = (tensor_prod([ket(0), ket(0)]) + tensor_prod([ket(1), ket(1)])).unit()
    rho_bell = bell.to_density_matrix()
    print("rho_bell:\n", rho_bell)
    rhoA = rho_bell.ptrace([0])
    print("rho_A:\n", rhoA.dense)

**Output**

::

    A:
    [[1.+1.j 2.+0.j]
    [3.+0.j 4.+0.j]]
    A†:
    [[1.-1.j 3.+0.j]
    [2.+0.j 4.+0.j]]
    exp(X):
    [[1.543 1.175]
    [1.175 1.543]]
    ||ket0|| = 1.0
    trace norm(dm) = 1.0
    rho_bell:
    QTensor(shape=4x4, nnz=4, format='csr')
    [[0.5 0.  0.  0.5]
    [0.  0.  0.  0. ]
    [0.  0.  0.  0. ]
    [0.5 0.  0.  0.5]]
    rho_A:
    [[0.5 0. ]
    [0.  0.5]]


Extra Utilities
^^^^^^^^^^^^^^^

- **Tensor product** with :func:`~qilisdk.common.qtensor.tensor_prod`  
- **Expectation value** with :func:`~qilisdk.common.qtensor.expect_val`  

.. code-block:: python

    from qilisdk.common.qtensor import QTensor, expect_val, ket, tensor_prod
    import numpy as np

    # Two‑qubit Hadamard tensor
    H = QTensor(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    H2 = tensor_prod([H, H])
    print("H ⊗ H:\n", np.round(H2.dense, 3))

    # Expectation of Z⊗Z on |00>
    Z = QTensor(np.array([[1, 0], [0, -1]]))
    zz = tensor_prod([Z, Z])
    psi00 = tensor_prod([ket(0), ket(0)])
    rho00 = psi00.to_density_matrix()
    ev = expect_val(zz, rho00)
    print("⟨ZZ⟩ on |00> =", ev)

**Output**

::

    H ⊗ H:
    [[ 0.5  0.5  0.5  0.5]
    [ 0.5 -0.5  0.5 -0.5]
    [ 0.5  0.5 -0.5 -0.5]
    [ 0.5 -0.5 -0.5  0.5]]
    ⟨ZZ⟩ on |00> = 1.0
