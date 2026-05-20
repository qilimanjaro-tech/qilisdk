Overview
==================

The :mod:`qilisdk.core` layer underpins both the digital and analog stacks. It
provides symbolic variables, optimization models, sparse quantum tensors, and
the :mod:`~qilisdk.core.parameterizable.Parameterizable` mixin used throughout the SDK.

Highlights:

- :doc:`core_variables_parameters` supplies a variety of variables (for constructing optimization models), as well as parameters (for parameterizing circuits, schedules, and more).
- :doc:`core_terms_maps_comparisons` provides ways of combining and manipulating variables and parameters.
- :doc:`core_model` to build constrained optimization programs.
- :doc:`core_lp_parser` to load models from and write models to LP-format files.
- :doc:`core_qubo` to convert models to QUBO format.
- :doc:`core_qtensor` to manage sparse quantum objects, such as states and operators.
