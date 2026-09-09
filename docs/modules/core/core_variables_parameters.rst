Variables and Parameters
=========================

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

    0.1 * (10.0 + x(0) + 2 * x(1) + 3.0 * x(3) + 4 * x(2))

To index the first binary variable from the binary representation of x you can write: ``x[0]``.
Each binary variable configuration generates a float within the bounds, based on the defined precision. For instance:

.. code-block:: python

    x.evaluate({x: [0, 1, 0, 0]})

**Output**:

::

    1.2000000000000002

Parameters
---------------------------------------------

Many components in QiliSDK expose symbolic parameters that can be optimized or
re-bound at runtime. The :class:`~qilisdk.core.variables.Parameter` class
represents a scalar symbol with optional bounds, and
:class:`~qilisdk.core.parameterizable.Parameterizable` provides a uniform API
(:meth:`~qilisdk.core.parameterizable.Parameterizable.get_parameter_names`, :meth:`~qilisdk.core.parameterizable.Parameterizable.set_parameter_values`…) implemented by circuits,
schedules, models, and more.

.. code-block:: python

    from qilisdk.core import Parameter

    theta = Parameter("theta", value=0.5, bounds=(0.0, 1.0))
    print(theta.value)     # 0.5
    theta.set_value(0.75)
    print(theta.bounds)    # (0.0, 1.0)

Parameters behave like symbolic variables in algebraic expressions, so you can
combine them with other variables and evaluate terms without having to pass the
parameter explicitly - its stored ``value`` is used automatically.

Objects that inherit from :class:`~qilisdk.core.parameterizable.Parameterizable`
collect all the :class:`~qilisdk.core.variables.Parameter` instances they encounter. For example:

.. code-block:: python

    from qilisdk.digital import Circuit, RX

    circuit = Circuit(nqubits=1)
    circuit.add(RX(0, theta=theta))

    print(circuit.get_parameter_names())   # ['RX(0)_theta_0']
    print(circuit.get_parameter_values())  # [0.75]
    circuit.set_parameters({"RX(0)_theta_0": 0.9})

All parameter helper methods accept an optional ``where`` predicate to filter
the exposed parameters. This is useful when you want to update only a subset,
such as trainable parameters.

.. code-block:: python

    trainable_values = circuit.get_parameter_values(where=lambda p: p.is_trainable)
    circuit.set_parameter_values([0.3], where=lambda p: p.is_trainable)

.. note::

    Parameterizable objects expose parameter state through
    :meth:`~qilisdk.core.parameterizable.Parameterizable.get_parameters`,
    :meth:`~qilisdk.core.parameterizable.Parameterizable.get_parameter_names`,
    :meth:`~qilisdk.core.parameterizable.Parameterizable.get_parameter_values`,
    and the corresponding ``set_*`` methods. Accessing ``.parameters`` directly
    is not part of the public API.

