Terms, Maps and Comparisons
---------------------------------

Terms
==========

:class:`Variables<qilisdk.core.variables.Variable>` can be combined algebraically to form expressions 
known as :class:`Terms<qilisdk.core.variables.Term>`. For example:

.. code-block:: python

    from qilisdk.core.variables import BinaryVariable, Bitwise, Domain, SpinVariable, Variable
    x = Variable("x", domain=Domain.REAL, bounds=(1, 2), encoding=Bitwise, precision=1e-1)
    s = SpinVariable("s")
    b = BinaryVariable("b")

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


Mathematical Maps
===============================

Use :class:`~qilisdk.core.variables.MathematicalMap` helpers to apply common
functions to a parameter or term while keeping expressions symbolic.
Each wraps a :class:`~qilisdk.core.variables.Parameter`, :class:`~qilisdk.core.variables.Term`,
or any other base variable and defers evaluation until values are provided.

.. code-block:: python

    from qilisdk.core.variables import Parameter, Sin, Cos

    theta = Parameter("theta", 0.5)
    expr = Sin(theta) + Cos(2 * theta)

    print(expr) # prints: "sin[theta] + cos[(2) * theta]"
    print(expr.evaluate({})) # prints: "0.5" (using default parameter values)

    # You can also supply a different value at evaluation time:
    print(expr.evaluate({theta: 1.0}))

These maps compose naturally with other terms, so you can include them in
constraints or objectives and rely on the same evaluation and encoding rules
as other symbolic expressions.

The list of possible mathematical maps includes:

- :class:`~qilisdk.core.variables.Abs` for absolute value
- :class:`~qilisdk.core.variables.Exp` for exponential
- :class:`~qilisdk.core.variables.Log` for logarithm
- :class:`~qilisdk.core.variables.Pow` for power functions
- :class:`~qilisdk.core.variables.Sqrt` for square root
- :class:`~qilisdk.core.variables.Inv` for inverse
- :class:`~qilisdk.core.variables.Sin` for sine
- :class:`~qilisdk.core.variables.Cos` for cosine
- :class:`~qilisdk.core.variables.Tan` for tangent

Comparison Terms
=======================

Each :class:`~qilisdk.core.variables.ComparisonTerm` defines a constraint using mathematical comparisons. 
Use the following operators to construct them:

.. list-table::
   :class: longtable
   :header-rows: 1
   :widths: 20 20 20

   * - Comparison Operation
     - QiliSDK Method
     - Alias
   * - Equality
     - :meth:`Equal(lhs, rhs)<qilisdk.core.variables.Equal>`
     - :meth:`EQ(lhs, rhs)<qilisdk.core.variables.EQ>`
   * - Not Equal
     - :meth:`NotEqual(lhs, rhs)<qilisdk.core.variables.NotEqual>`
     - :meth:`NEQ(lhs, rhs)<qilisdk.core.variables.NEQ>`
   * - Less Than
     - :meth:`LessThan(lhs, rhs)<qilisdk.core.variables.LessThan>`
     - :meth:`LT(lhs, rhs)<qilisdk.core.variables.LT>`
   * - Less Than or Equal
     - :meth:`LessThanOrEqual(lhs, rhs)<qilisdk.core.variables.LessThanOrEqual>`
     - :meth:`LEQ(lhs, rhs)<qilisdk.core.variables.LEQ>`
   * - Greater Than
     - :meth:`GreaterThan(lhs, rhs)<qilisdk.core.variables.GreaterThan>`
     - :meth:`GT(lhs, rhs)<qilisdk.core.variables.GT>`
   * - Greater Than or Equal
     - :meth:`GreaterThanOrEqual(lhs, rhs)<qilisdk.core.variables.GreaterThanOrEqual>`
     - :meth:`GEQ(lhs, rhs)<qilisdk.core.variables.GEQ>`

*Note*: `lhs` and `rhs` refer to the left-hand side and right-hand side expressions, respectively.

Example:

.. code-block:: python

    from qilisdk.core.variables import BinaryVariable, LT
    x = BinaryVariable("x")
    LT(2 * x - 1, 1)

**Output**:

::

    (2) * x < (2.0)

When a comparison term is created, constants are automatically moved to the right-hand side, and variable terms to the left-hand side.

