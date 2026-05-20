Linear Programming File Parser and Serializer
---------------------------------------------

QiliSDK ships with a parser and a serializer for the CPLEX-flavor `LP format
<https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-lp-file-format-algebraic-representation>`_.
The parser is exposed as :func:`~qilisdk.utils.lp_parser.from_lp` (parse a
string) and :func:`~qilisdk.utils.lp_parser.from_lp_file` (read and parse a
file); both produce a fully-formed :class:`~qilisdk.core.model.Model`. The
inverse pair, :func:`~qilisdk.utils.lp_parser.to_lp` and
:func:`~qilisdk.utils.lp_parser.to_lp_file`, serializes a
:class:`~qilisdk.core.model.Model` back to LP-format text.

Quick start
^^^^^^^^^^^

.. SKIP
.. code-block:: python

    from qilisdk.utils.lp_parser import from_lp_file

    model = from_lp_file("knapsack.lp")
    print(model)

A small inline example:

.. code-block:: python

    from qilisdk.utils.lp_parser import from_lp

    lp = """
    Maximize
     value: 4 x1 + 5 x2 + 3 x3
    Subject To
     weight: 2 x1 + 3 x2 + x3 <= 5
    Binary
     x1
     x2
     x3
    End
    """

    model = from_lp(lp)
    qubo = model.to_qubo()

Round-tripping a model back to LP text:

.. code-block:: python

    from qilisdk.utils.lp_parser import to_lp, to_lp_file

    print(to_lp(model))          # serialize to a string
    to_lp_file(model, "out.lp")  # or write directly to disk

Supported features
^^^^^^^^^^^^^^^^^^

- **Objective sense**: ``Maximize`` / ``Minimize`` (and the ``Max`` / ``Min`` /
  ``Maximum`` / ``Minimum`` aliases, case-insensitive).
- **Section keywords**: ``Subject To`` (also ``subj to``, ``s.t.``, ``st``),
  ``Bounds``, ``General``, ``Binary``, ``End``.
- **Objective expressions**: linear and quadratic terms, including bilinear
  monomials such as ``x1 * x2`` and parenthesised sub-expressions like
  ``- [x + 1]``. Repeated occurrences of the same variable are aggregated
  automatically (``x + x`` becomes ``2*x``).
- **Constraint senses**: ``<``, ``<=``, ``=``, ``>=``, ``>`` — each maps to the
  corresponding :class:`~qilisdk.core.variables.ComparisonTerm`. Constraint
  labels are optional; missing labels are auto-generated as ``c1``, ``c2``, …
- **Bounds**: two-sided ranges (``-2.5 <= x <= 7.5``), open sides via ``+inf`` /
  ``-infinity``, and the ``free`` shorthand for unbounded variables.
- **Variable domains**:

  - Variables listed in ``Binary`` are :class:`~qilisdk.core.variables.BinaryVariable`.
  - Variables listed in ``General`` are integer
    :class:`~qilisdk.core.variables.Variable` with :attr:`~qilisdk.core.variables.Domain.INTEGER`.
    Bounds declared in the ``Bounds`` section are preserved.
  - Variables that appear only in the ``Bounds`` section are continuous
    (``Domain.REAL``) with the declared bounds.
  - Variables that appear only inside expressions are auto-created as
    continuous with the LP-format default bounds ``[0, +inf)``.

- **Comments**: backslash (``\\``) starts a line comment that runs to the end
  of the line.

Serialization conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~qilisdk.utils.lp_parser.to_lp` produces grammar-compliant output that
:func:`~qilisdk.utils.lp_parser.from_lp` can read back. Notable conventions:

- Quadratic and bilinear monomials are expanded into explicit products
  (``x * x``, ``x * y``) since the LP grammar accepted by the parser has no
  ``^`` operator.
- A unit coefficient is omitted (``x`` rather than ``1 x``); a ``-1`` coefficient
  becomes a leading ``- x``.
- Constraint right-hand sides are emitted as a single constant — any constant
  appearing on the left-hand side is folded into the rhs first.
- Unbounded continuous variables are emitted as ``var free``; one-sided bounds
  use ``var >= lo`` or ``-infinity <= var <= up``. Variables matching the LP
  default of ``[0, +inf)`` are omitted from the ``Bounds`` section entirely.
- :class:`~qilisdk.core.variables.BinaryVariable` variables are listed in the
  ``Binary`` section; integer ones in ``General`` (with their bounds, if any).

Worked example
^^^^^^^^^^^^^^

.. code-block:: text

    \\ Mixed-integer LP with one of every variable kind.
    Maximize
     obj: 2 x1 + 1.5 x2 - 3 x3 + b1 * b2 - [x4 + 1] + 10
    Subject To
     c_le: 2 x1 + x2 <= 100
     c_ge: x3 + x4   >= -5
     c_eq: x4 - 2 y  =  7
    Bounds
        0 <= x1 <= 50
     -2.5 <= x2 <= 7.5
        x3 free
     -infinity <= x4 <= 5
        0 <= y <= 10
    General
     x4
     y
    Binary
     b1
     b2
    End

Parsing it returns a :class:`~qilisdk.core.model.Model` with:

- five real-valued variables (``x1``, ``x2``, ``x3``, ``x4`` plus auto-typed
  ones, plus integer ``y``),
- two binary variables (``b1``, ``b2``),
- a maximization objective with linear, bilinear, and parenthesised terms,
- three labelled constraints covering the ``<=``, ``>=``, and ``=`` senses.

The resulting model can be inspected, evaluated against a sample with
:meth:`Model.evaluate <qilisdk.core.model.Model.evaluate>`, or converted to a
QUBO via :meth:`Model.to_qubo <qilisdk.core.model.Model.to_qubo>` like any
other model built directly with the symbolic API. Passing the model to
:func:`~qilisdk.utils.lp_parser.to_lp` yields an LP-format string that, when
fed back through :func:`~qilisdk.utils.lp_parser.from_lp`, reconstructs an
equivalent model.
