Variables and Terms
======================

Variables
----------------

This module offers tools to define different types of variables:

- :class:`Binary Variables <qilisdk.core.variables.BinaryVariable>`
- :class:`Spin Variables <qilisdk.core.variables.SpinVariable>`
- :class:`Continuous Variables <qilisdk.core.variables.Variable>` with the following customizable parameters:

    - Domain: Specifies the variable type, one of :class:`~qilisdk.core.variables.Domain.REAL`, :class:`~qilisdk.core.variables.Domain.INTEGER`, :class:`~qilisdk.core.variables.Domain.POSITIVE_INTEGER`, :class:`~qilisdk.core.variables.Domain.BINARY`, or :class:`~qilisdk.core.variables.Domain.SPIN`.
    - Bounds: Defines the allowed value range of the variable.
    - Encoding: Specifies how the variable is represented using binary encodings, one of: :class:`~qilisdk.core.variables.Bitwise`, :class:`~qilisdk.core.variables.DomainWall`, or :class:`~qilisdk.core.variables.OneHot`.
    - Precision: Applicable to :class:`Domain.REAL <qilisdk.core.variables.Domain.REAL>` domain; defines the resolution (e.g., floating-point precision).

An example of creating different types of variables:

.. code-block:: python

    from qilisdk.core.variables import BinaryVariable, Bitwise, Domain, SpinVariable, Variable

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
---------------

:class:`Variables<qilisdk.core.variables.Variable>` can be combined algebraically to form expressions 
known as :class:`Terms<qilisdk.core.variables.Term>`. For example:

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

