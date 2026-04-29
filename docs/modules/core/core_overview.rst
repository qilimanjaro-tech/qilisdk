Overview
==================

The :mod:`qilisdk.core` layer underpins both the digital and analog stacks. It
provides symbolic variables, optimization models, sparse quantum tensors, and
the :mod:`~qilisdk.core.parameterizable.Parameterizable` mixin used throughout the SDK.

Highlights:

- :doc:`core_variables_terms` supplies binary, spin, continuous, and
  parameter variables plus algebraic helpers (:class:`~qilisdk.core.variables.Term`,
  comparison factories, encodings).
- :doc:`core_model` builds constrained optimization programs and
  offers tools to automatically convert the model to :class:`~qilisdk.core.model.QUBO` format if the constraints are linear.
- :doc:`core_qtensor` manages sparse quantum objects, such as states and operators.
- :doc:`core_parameters` standardizes how
  objects expose symbolic parameters (shared by circuits, schedules, etc.).
