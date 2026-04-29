Maps and Comparisons
---------------------------------

Mathematical Maps
===============================

Use :class:`~qilisdk.core.variables.MathematicalMap` helpers to apply common
functions to a parameter or term while keeping expressions symbolic.
:class:`~qilisdk.core.variables.Sin` and :class:`~qilisdk.core.variables.Cos`
wrap a :class:`~qilisdk.core.variables.Parameter`, :class:`~qilisdk.core.variables.Term`,
or any other base variable and defer evaluation until values are provided.

.. code-block:: python

    from qilisdk.core.variables import Parameter, Sin, Cos

    theta = Parameter("theta", 0.5)
    expr = Sin(theta) + Cos(2 * theta)

    print(expr)                # sin[theta] + cos[(2) * theta]
    print(expr.evaluate({}))   # uses theta.value automatically

    # You can also supply a different value at evaluation time:
    print(expr.evaluate({theta: 1.0}))

These maps compose naturally with other terms, so you can include them in
constraints or objectives and rely on the same evaluation and encoding rules
as other symbolic expressions.

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

    from qilisdk.core.variables import LT
    LT(2 * x - 1, 1)

**Output**:

::

    (2) * x < (2.0)

When a comparison term is created, constants are automatically moved to the right-hand side, and variable terms to the left-hand side.

