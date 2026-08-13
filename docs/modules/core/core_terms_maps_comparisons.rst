Expressions, Functions and Comparisons
--------------------------------------

Expressions
===========

:class:`Variables<qilisdk.core.variables.Variable>` can be combined algebraically with the usual
Python operators (``+``, ``-``, ``*``, ``/``, ``**``) to build a symbolic
:class:`~qilisdk.core.expression.Expression`. Every leaf (a variable, parameter or numeric constant)
and every operator node (:class:`~qilisdk.core.expression.Add`, :class:`~qilisdk.core.expression.Mul`,
:class:`~qilisdk.core.expression.Pow`) is itself an ``Expression``, so expressions compose freely.
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
sub-expression. Use :meth:`~qilisdk.core.expression.Expression.expand` to distribute, and
:meth:`~qilisdk.core.expression.Expression.simplify` to request a simpler (but semantically equal) form:

.. code-block:: python

    print(e4.expand())          # distribute the product over the sum

**Output**:

::

    -1 + -3 * x**2

Expressions can be evaluated by providing values for the involved variables via
:meth:`~qilisdk.core.expression.Expression.evaluate`:

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

An :class:`~qilisdk.core.expression.Expression` exposes a small introspection API. You can list the
named leaves it depends on, isolate just the free :class:`~qilisdk.core.variables.Parameter` leaves,
read its polynomial :attr:`~qilisdk.core.expression.Expression.degree`, and take a symbolic
derivative with :meth:`~qilisdk.core.expression.Expression.diff`:

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

Non-polynomial operations are represented by :class:`~qilisdk.core.expression.Function`, the abstract
base for the unary maths functions. Its concrete subclasses
:class:`~qilisdk.core.expression.Sin`, :class:`~qilisdk.core.expression.Cos`,
:class:`~qilisdk.core.expression.Exp`, :class:`~qilisdk.core.expression.Log`,
:class:`~qilisdk.core.expression.Tan`, :class:`~qilisdk.core.expression.Sqrt` and
:class:`~qilisdk.core.expression.Abs` each wrap a single
:class:`~qilisdk.core.expression.Expression` operand (a :class:`~qilisdk.core.variables.Parameter`,
any other variable, or a compound expression) and defer numeric evaluation until values are provided.

.. code-block:: python

    from qilisdk.core.expression import Cos, Sin
    from qilisdk.core.variables import Parameter

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
:class:`~qilisdk.core.expression.Constant`:

.. code-block:: python

    from qilisdk.core.expression import Cos, Exp, Sin
    from qilisdk.core.variables import Parameter

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

The available function nodes are:

- :class:`~qilisdk.core.expression.Sin` for sine
- :class:`~qilisdk.core.expression.Cos` for cosine
- :class:`~qilisdk.core.expression.Tan` for tangent
- :class:`~qilisdk.core.expression.Exp` for exponential
- :class:`~qilisdk.core.expression.Log` for logarithm
- :class:`~qilisdk.core.expression.Sqrt` for square root
- :class:`~qilisdk.core.expression.Abs` for absolute value

``Abs`` is the one function with no derivative: it is not differentiable at zero and there is no
``sign`` node to write its derivative with, so :meth:`~qilisdk.core.expression.Expression.diff`
raises on it.

Powers are not functions. Use the ``**`` operator, which builds a
:class:`~qilisdk.core.expression.Pow` node and accepts a fractional or symbolic exponent.
:func:`~qilisdk.core.expression.Inv` is a shorthand for ``x ** -1``, so ``Inv(x)``, ``1 / x`` and
``x ** -1`` are all the same expression:

.. code-block:: python

    from qilisdk.core.expression import Inv
    from qilisdk.core.variables import Parameter

    x, y = Parameter("x", 4.0), Parameter("y", 0.5)

    print(x**y)                       # symbolic exponent
    print((x**y).evaluate({}))
    print(Inv(x) == 1 / x == x**-1)

**Output**:

::

    x**y
    2.0
    True

To write your own function, subclass :class:`~qilisdk.core.expression.Function` with a ``NAME``, a
numeric kernel and a derivative. Everything else (canonicalization, equality, ``diff``, ``expand``,
``substitute``, serialization) comes from the base class.

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

