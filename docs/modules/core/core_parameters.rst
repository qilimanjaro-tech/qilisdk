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

