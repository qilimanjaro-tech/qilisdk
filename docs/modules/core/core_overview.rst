Overview
==================

The :mod:`qilisdk.core` layer underpins both the digital and analog stacks. It
provides symbolic variables, optimization models, sparse quantum tensors, and
the :mod:`~qilisdk.core.parameterizable.Parameterizable` mixin used throughout the SDK.

Highlights:

- :mod:`~qilisdk.core.variables` supplies binary, spin, continuous, and
  parameter variables plus algebraic helpers (:class:`~qilisdk.core.variables.Term`,
  comparison factories, encodings).
- :mod:`~qilisdk.core.model` builds constrained optimization programs and
  offers tools to automatically convert the model to :class:`~qilisdk.core.model.QUBO` format if the constraints are linear.
- :mod:`~qilisdk.core.qtensor` manages sparse quantum objects and utilities
  such as :func:`~qilisdk.core.qtensor.tensor_prod` and :func:`~qilisdk.core.qtensor.expect_val`.
- :mod:`~qilisdk.core.parameterizable.Parameterizable` standardizes how
  objects expose symbolic parameters (shared by circuits, schedules, etc.).
