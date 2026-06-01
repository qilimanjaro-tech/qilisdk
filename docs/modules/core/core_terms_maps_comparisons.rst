Expressions, Functions and Comparisons
--------------------------------------

Expressions
===========

:class:`Variables<qilisdk.core.variables.Variable>` can be combined algebraically with the usual
Python operators (``+``, ``-``, ``*``, ``/``, ``**``) to build a symbolic
:class:`~qilisdk.core.variables.Expression`. Every leaf (a variable, parameter or numeric constant)
and every operator node (:class:`~qilisdk.core.variables.Add`, :class:`~qilisdk.core.variables.Mul`,
:class:`~qilisdk.core.variables.Pow`) is itself an ``Expression``, so expressions compose freely.
For example:

.. code-block:: python

    from qilisdk.core.variables import BinaryVariable, Bitwise, Domain, SpinVariable, Variable
    x = Variable("x", domain=Domain.REAL, bounds=(1, 2), encoding=Bitwise, precision=1e-1)
    s = SpinVariable("s")
    b = BinaryVariable("b")

    e1 = 2 * x + 3
    print("e1:", e1)
    e2 = 3 * x**2 + 2 * x + 4
    print("e2:", e2)
    e3 = 2 * x + b - 1
    print("e3:", e3)
    e4 = e1 - e2
    print("e4:", e4)

**Output**:

::

    e1: 3 + 2 * x
    e2: 4 + 2 * x + 3 * x**2
    e3: -1 + b + 2 * x
    e4: 3 + -1 * (4 + 2 * x + 3 * x**2) + 2 * x

Construction *canonicalizes* the expression (flattening nested sums/products, combining like terms
and powers, folding constants, ordering operands deterministically), which is why the numeric
constant is printed first and ``x + y`` equals ``y + x``. Canonicalization is intentionally cheap:
products are **not** distributed over sums, so ``e4`` keeps the factored ``-1 * (4 + 2 * x + 3 * x**2)``
sub-expression. Use :meth:`~qilisdk.core.variables.Expression.expand` to distribute, and
:meth:`~qilisdk.core.variables.Expression.simplify` to request a simpler (but semantically equal) form:

.. code-block:: python

    print(e4.expand())          # distribute the product over the sum

**Output**:

::

    -1 + -3 * x**2

Expressions can be evaluated by providing values for the involved variables via
:meth:`~qilisdk.core.variables.Expression.evaluate`:

.. code-block:: python

    e3.evaluate({
        x: 1.5,
        b: 0
    })

**Output**:

::

    2.0

.. warning::

    To evaluate an expression, all participating variables must be assigned valid values within their respective domains and bounds.

Inspecting and differentiating expressions
===========================================

An :class:`~qilisdk.core.variables.Expression` exposes a small introspection API. You can list the
named leaves it depends on, isolate just the free :class:`~qilisdk.core.variables.Parameter` leaves,
read its polynomial :attr:`~qilisdk.core.variables.Expression.degree`, and take a symbolic
derivative with :meth:`~qilisdk.core.variables.Expression.diff`:

.. code-block:: python

    from qilisdk.core.variables import Parameter, Variable, Domain

    a = Parameter("a", value=2.0)
    y = Variable("y", domain=Domain.REAL, bounds=(0, 5))

    expr = a * y**2 + 3 * y

    print(expr.variables())          # named leaves, sorted by label
    print(expr.free_parameters())    # only the Parameter leaves
    print(expr.degree)               # highest polynomial degree (a and y both count)
    print(expr.diff(y))              # symbolic d/dy

**Output**:

::

    [a, y]
    {a}
    3
    3 + 2 * a * y


Mathematical Functions
======================

Non-polynomial operations are represented by :class:`~qilisdk.core.variables.Function`, the abstract
base for the unary maths functions. Its concrete subclasses
:class:`~qilisdk.core.variables.Sin`, :class:`~qilisdk.core.variables.Cos`,
:class:`~qilisdk.core.variables.Exp`, :class:`~qilisdk.core.variables.Log`,
:class:`~qilisdk.core.variables.Tan` and :class:`~qilisdk.core.variables.Sqrt` each wrap a single
:class:`~qilisdk.core.variables.Expression` operand (a :class:`~qilisdk.core.variables.Parameter`,
any other variable, or a compound expression) and defer numeric evaluation until values are provided.

.. code-block:: python

    from qilisdk.core.variables import Parameter, Sin, Cos

    theta = Parameter("theta", 0.5)
    expr = Sin(theta) + Cos(2 * theta)

    print(expr)                # cos(2 * theta) + sin(theta)
    print(expr.evaluate({}))   # uses theta.value automatically

    # You can also supply a different value at evaluation time:
    print(expr.evaluate({theta: 1.0}))

**Output**:

::

    cos(2 * theta) + sin(theta)
    1.0197278444723428
    0.4253241482607541

Because every function is a regular ``Expression`` node, it participates in the same algebra: it can
be added to or multiplied with other expressions, differentiated symbolically (the chain rule is
applied automatically), and evaluated. Wrapping a numeric constant folds eagerly to a
:class:`~qilisdk.core.variables.Constant`:

.. code-block:: python

    from qilisdk.core.variables import Cos, Sin, Exp, Parameter

    theta = Parameter("theta", 0.5)

    print(Sin(theta).diff(theta))   # d/dtheta sin(theta) == cos(theta)
    print(Exp(theta).diff(theta))   # d/dtheta exp(theta) == exp(theta)
    print(Cos(0))                   # folds to a numeric constant

**Output**:

::

    cos(theta)
    exp(theta)
    1.0

These functions compose naturally with the rest of the expression tree, so you can include them in
constraints, objectives, or schedule coefficients and rely on the same evaluation and encoding rules
as any other symbolic expression.

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
    print(LT(2 * x - 1, 1))

**Output**:

::

    2 * x < 2

When a comparison term is created, constants are automatically moved to the right-hand side, and variable terms to the left-hand side.

