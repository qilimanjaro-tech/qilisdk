Datasets
--------

The :mod:`~qilisdk.ml.datasets` module provides a family of equation-based generators for various
datasets that are commonly used in machine-learning research. 
Every generator derives from :class:`~qilisdk.ml.datasets.dataset.Dataset` and exposes a common
:meth:`~qilisdk.ml.datasets.dataset.Dataset.generate` method:

.. code-block:: python

    from qilisdk.ml.datasets import NARMA

    dataset = NARMA(order=10, seed=0)
    sample = dataset.generate(1000)
    inputs, targets = sample

The returned :class:`~qilisdk.ml.datasets.dataset.DatasetSample` is an ``(inputs, targets)`` pair.

Available generators
====================

- :class:`~qilisdk.ml.datasets.narma.NARMA` --- Nonlinear Auto-Regressive Moving Average system identification.
- :class:`~qilisdk.ml.datasets.mackey_glass.MackeyGlass` --- Mackey--Glass chaotic delay differential equation.
- :class:`~qilisdk.ml.datasets.lorenz.Lorenz` --- Lorenz attractor.
- :class:`~qilisdk.ml.datasets.santa_fe_laser.SantaFeLaser` --- Santa Fe laser intensity (Lorenz--Haken equations).
- :class:`~qilisdk.ml.datasets.henon.HenonMap` --- Hénon map.
- :class:`~qilisdk.ml.datasets.logistic_map.LogisticMap` --- Logistic map.

NARMA
=====

The Nonlinear Auto-Regressive Moving Average (:class:`~qilisdk.ml.datasets.narma.NARMA`) benchmark is a
system-identification task. A random input stream :math:`u(t) \sim \mathcal{U}(0, 0.5)` drives an order-:math:`n`
nonlinear recurrence whose output :math:`y(t)` must be predicted from :math:`u`:

.. math::

    y(t+1) = \alpha\, y(t)
             + \beta\, y(t) \sum_{i=0}^{n-1} y(t-i)
             + \gamma\, u(t-n+1)\, u(t)
             + \delta.

The default coefficients :math:`(\alpha, \beta, \gamma, \delta) = (0.3, 0.05, 1.5, 0.1)` correspond to the ubiquitous
``NARMA10`` task (``order=10``). Unlike the other generators, ``inputs`` are the random drive :math:`u` and ``targets``
are the system response :math:`y`; both are shaped ``(npoints, 1)``. Because the drive is random, a ``seed`` can be specified to ensure reproducibility.

.. code-block:: python

    from qilisdk.ml.datasets import NARMA

    inputs, targets = NARMA(order=10, input_range=(0.0, 0.5), seed=42).generate(2000)
    print(inputs.shape, targets.shape)

MackeyGlass
===========

The :class:`~qilisdk.ml.datasets.mackey_glass.MackeyGlass` system is a nonlinear delay differential equation that
produces a chaotic attractor:

.. math::

    \frac{dx}{dt} = \beta\, \frac{x(t - \tau)}{1 + x(t - \tau)^{n}}
                    - \gamma\, x(t).

With the standard parameters :math:`\beta = 0.2`, :math:`\gamma = 0.1`, :math:`n = 10`, the behaviour is set by the
delay :math:`\tau`: the series is periodic for small :math:`\tau`, mildly chaotic at :math:`\tau = 17`, and increasingly
chaotic beyond. The equation is integrated with a fixed-step RK4 scheme at resolution ``dt`` and sub-sampled every
``sample_every`` steps.

.. code-block:: python

    from qilisdk.ml.datasets import MackeyGlass

    inputs, targets = MackeyGlass(tau=17.0).generate(2000)
    print(inputs.shape, targets.shape)

Lorenz
======

The :class:`~qilisdk.ml.datasets.lorenz.Lorenz` attractor is a three-dimensional chaotic dynamical system:

.. math::

    \dot{x} = \sigma (y - x), \quad
    \dot{y} = x (\rho - z) - y, \quad
    \dot{z} = x y - \beta z.

The trajectory is integrated with RK4 and sub-sampled, yielding a ``horizon``-step-ahead prediction task over the
three-dimensional state, so ``inputs`` and ``targets`` are both shaped ``(npoints, 3)``.

.. code-block:: python

    from qilisdk.ml.datasets import Lorenz

    inputs, targets = Lorenz(sigma=10.0, rho=28.0).generate(2000)
    print(inputs.shape, targets.shape)

SantaFeLaser
============

The original *Santa Fe Time Series Competition* Data Set A is a recording of the chaotic intensity pulsations of a
far-infrared :math:`\mathrm{NH_3}` laser. Rather than shipping the recording,
:class:`~qilisdk.ml.datasets.santa_fe_laser.SantaFeLaser` reproduces the same qualitative dynamics from first
principles using the single-mode **Lorenz--Haken** laser equations:

.. math::

    \dot{E} = \sigma (P - E), \quad
    \dot{P} = E (\rho - N) - P, \quad
    \dot{N} = E P - \beta N,

where :math:`E` is the field amplitude, :math:`P` the polarization and :math:`N` the population inversion. The measured
quantity is the laser **intensity** :math:`I \propto E^2`, which is non-negative and reproduces the behaviour of the
Santa Fe recording. Both ``inputs`` and ``targets`` are shaped ``(npoints, 1)``.

.. code-block:: python

    from qilisdk.ml.datasets import SantaFeLaser

    inputs, targets = SantaFeLaser().generate(2000)
    print(inputs.min() >= 0.0)

HenonMap
========

The :class:`~qilisdk.ml.datasets.henon.HenonMap` is a two-dimensional discrete chaotic system:

.. math::

    x_{n+1} = 1 - a\, x_n^2 + y_n, \qquad
    y_{n+1} = b\, x_n,

which is chaotic for the parameters :math:`a = 1.4`, :math:`b = 0.3`. :meth:`generate` returns a ``horizon``-step-ahead
prediction task over the two-dimensional state, so ``inputs`` and ``targets`` are both shaped ``(npoints, 2)``.

.. code-block:: python

    from qilisdk.ml.datasets import HenonMap

    inputs, targets = HenonMap(a=1.4, b=0.3).generate(2000)
    print(inputs.shape, targets.shape)

LogisticMap
===========

The :class:`~qilisdk.ml.datasets.logistic_map.LogisticMap` is a simple one-dimensional chaotic system:

.. math::

    x_{n+1} = r\, x_n (1 - x_n),

which becomes chaotic as the growth rate :math:`r` approaches 4 (the default :math:`r = 3.9` sits well inside the
chaotic regime). :meth:`generate` returns a ``horizon``-step-ahead prediction task, so ``inputs`` and ``targets`` are
shaped ``(npoints, 1)``.

.. code-block:: python

    from qilisdk.ml.datasets import LogisticMap

    inputs, targets = LogisticMap(r=3.9, horizon=1).generate(2000)
    print(inputs.shape, targets.shape)
