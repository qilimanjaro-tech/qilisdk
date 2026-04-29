Overview
=============

The :mod:`qilisdk.analog` module lets you build symbolic Hamiltonians and time-dependent schedules.
The core types are:

- :doc:`analog_hamiltonian`: Tools to build arbitrary Hamiltonians using symbolic Pauli operators.
- :doc:`analog_schedule`: Flexible time-dependent evolution with callable/interval coefficients and step/linear interpolation.

These schedules can then be put into an :doc:`../functionals/functionals_analog` functional 
and then submitted to a backend for simulation or execution.
