Common Module
=============

The :mod:`~qilisdk.common` module in the Qili SDK provides a collection of utilities useful for building quantum algorithms and performing simulations. It consists of the following primary components:

- :mod:`~qilisdk.common.model`: A toolkit to construct and mathematically represent optimization models.

    - Depends on :mod:`~qilisdk.common.variables`, which provides the fundamental building blocks of models.
- :mod:`~qilisdk.common.quantum_objects`: Tools for creating and managing digital quantum circuits.


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

    x.to_binary()

**Output**:

::

    (0.1) * x(0) + (0.2) * x(1) + (0.4) * x(2) + (0.30000000000000004) * x(3) + (1.0)

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

    t1:  (2) * x + (3)
    t2:  (3) * (x^2) + (2) * x + (4)
    t3:  (2) * x + b + (-1)
    t4:  (-1.0) + (-3.0) * (x^2)

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

The :class:`~qilisdk.common.model.QUBO` subclass specializes in **Quadratic Unconstrained Binary Optimization** models, where every decision variable is binary and the objective function is at most quadratic.  Unlike general models, “hard” constraints are not maintained separately but are encoded directly into the objective as penalty terms.  The strength of each penalty is controlled by its associated Lagrange multiplier.

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
    model = QUBO("portfolio_selection")  

2. **Objective**  

.. code-block::  python
    
    # sum of weights x minus risk penalty
    model.set_objective(5 * x1 + 3 * x2 - 2 * x1 * x2,
                        label="return_minus_risk",
                        sense=ObjectiveSense.MAXIMIZE)

Adding Constraints as Penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since QUBO is unconstrained, every constraint is converted into the objective via a **penalty term**:

- **Slack penalization** (default):  
    Introduce additional binary slack variables to turn inequalities into equalities, then square the residual.  
- **Unbalanced penalization**:  
    Directly penalize violation without slack variables using two weights (a, b) to scale positive and negative deviations differently [1]_.

.. warning::

   - If **transform_to_qubo=True**, your constraint **must be linear** (no quadratic terms), because it will be rewritten as (lhs-rhs)^2.  
   - If **transform_to_qubo=False**, you assume the constraint is already a valid quadratic penalty, and it will be added verbatim.

Example: Slack Penalization
'''''''''''''''''''''''''''

.. code-block:: python

    from qilisdk.common.model import QUBO, ObjectiveSense
    from qilisdk.common.variables import BinaryVariable, LT

    b = BinaryVariable("b")
    model = QUBO("slack_example")
    model.set_objective(2 * b + 1,
                        label="obj",
                        sense=ObjectiveSense.MINIMIZE)

    # Enforce b <= 0.5 via slack, squared penalty in objective
    model.add_constraint("c1",
                        LT(b, 0.5),
                        lagrange_multiplier=10,
                        penalization="slack",
                        transform_to_qubo=True)

    print(model.qubo_objective)

**Output**::

    obj: (2) * b + (1) + 10 * (b - slack_c1(0) - 0.5)**2



Example: Unbalanced Penalization
''''''''''''''''''''''''''''''''

.. code-block:: python

    model = QUBO("unbalanced_example")
    model.set_objective(2 * b + 1,
                        label="obj",
                        sense=ObjectiveSense.MINIMIZE)

    # Penalize only violations above 0.5 more heavily than below
    model.add_constraint("c1",
                        LT(b, 0.5),
                        lagrange_multiplier=1,
                        penalization="unbalanced",
                        parameters=[2.0, 5.0],
                        transform_to_qubo=True)

    print(model.qubo_objective)

**Output**::

    obj: (2) * b + (1) + 2.0 * max(0, b - 0.5) + 5.0 * max(0.5 - b, 0)


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

.. [1] Montañez-Barrera, Jhon Alejandro, et al. "Unbalanced penalization: A new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms." Quantum Science and Technology 9.2 (2024): 025022.

Quantum Objects
---------------

The :mod:`~qilisdk.common.quantum_objects` module defines the :class:`~qilisdk.common.quantum_objects.QuantumObject` class and related helpers for representing and manipulating quantum states and operators in sparse form.

The :class:`~qilisdk.common.quantum_objects.QuantumObject` wraps a dense NumPy array or SciPy sparse matrix into a CSR-format sparse matrix, and can represent:

- **Kets** (column vectors of shape ``(2**N, 1)``)  
- **Bras** (row vectors of shape ``(1, 2**N)`` or ``(2**N,)``)  
- **Operators / Density Matrices** (square matrices of shape ``(2**N, 2**N)``)  
- **Scalars** (``(1, 1)`` matrices)  

Examples of creating various quantum objects:

.. code-block:: python

    import numpy as np
    from qilisdk.common.quantum_objects import QuantumObject

    # 1‑qubit |0> ket
    psi_ket = QuantumObject(np.array([[1], [0]]))
    print("Ket:", psi_ket.dense, "is_ket?", psi_ket.is_ket())
    print("-" * 20)

    # 1‑qubit <0| bra
    psi_bra = QuantumObject(np.array([1, 0]))
    print("Bra:", psi_bra.dense, "is_bra?", psi_bra.is_bra())
    print("-" * 20)

    # Density matrix |0><0|
    rho = QuantumObject(np.array([[1, 0], [0, 0]]))
    print("Density matrix:\n", rho.dense, "is_density_matrix?", rho.is_density_matrix())
    print("-" * 20)

    # Scalar 0.5
    scalar = QuantumObject(np.array([[0.5]]))
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

    from qilisdk.common.quantum_objects import ket, bra, basis_state

    # Single‑qubit
    print("ket(0):", ket(0).dense, "is_ket?", ket(0).is_ket())
    print("bra(1):", bra(1).dense, "is_bra?", bra(1).is_bra())

    # Fock basis in N=4 Hilbert space
    print("basis_state(2,4):", basis_state(2, 4).dense, "shape:", basis_state(2,4).shape)

**Output**

::

    ket(0): [[1]
    [0]] is_ket? True
    bra(1): [[0 1]] is_bra? True
    basis_state(2,4): [[0]
    [0]
    [1]
    [0]] shape: (4, 1)

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
    from qilisdk.common.quantum_objects import QuantumObject

    # Adjoint of a non-Hermitian operator
    A = QuantumObject(np.array([[1+1j, 2], [3, 4]]))
    A_dag = A.adjoint()
    print("A:\n", A.dense)
    print("A†:\n", A_dag.dense)

    # Matrix exponential of Pauli-X
    X = QuantumObject(np.array([[0, 1], [1, 0]]))
    expX = X.expm()
    print("exp(X):\n", np.round(expX.dense, 3))

    # Norm of a ket and a density matrix
    ket0 = QuantumObject(np.array([[1], [0]]))
    dm = ket0.to_density_matrix()
    print("||ket0|| =", ket0.norm())
    print("trace norm(dm) =", dm.norm(order='tr'))

    # Partial trace of a Bell state
    from qilisdk.common.quantum_objects import ket, tensor_prod
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
    [[0.5 0.  0.  0.5]
    [0.  0.  0.  0. ]
    [0.  0.  0.  0. ]
    [0.5 0.  0.  0.5]]
    rho_A:
    [[0.5 0. ]
    [0.  0.5]]


Extra Utilities
^^^^^^^^^^^^^^^

- **Tensor product** with :func:`~qilisdk.common.quantum_objects.tensor_prod`  
- **Expectation value** with :func:`~qilisdk.common.quantum_objects.expect_val`  

.. code-block:: python

    from qilisdk.common.quantum_objects import tensor_prod, expect_val, ket, QuantumObject
    import numpy as np

    # Two‑qubit Hadamard tensor
    H = QuantumObject(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    H2 = tensor_prod([H, H])
    print("H ⊗ H:\n", np.round(H2.dense, 3))

    # Expectation of Z⊗Z on |00>
    Z = QuantumObject(np.array([[1, 0], [0, -1]]))
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
