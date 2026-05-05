Overview
===========

Readout is the mechanism that controls **what information is extracted from the quantum backend** after a
functional is executed.  It is specified separately from the functional itself, so the same circuit or
schedule can be measured in different ways without any code change to the functional.

All uses start with an instance of the class :class:`~qilisdk.readout.readout_spec.Readout`.

:class:`~qilisdk.readout.readout_spec.Readout` is a builder that accumulates readout methods by
chaining ``with_*`` calls.  Each call returns a *new* spec with one additional slot filled; the original
object is never modified.

.. code-block:: python

    from qilisdk.readout import Readout
    from qilisdk.analog import Z

    spec = Readout()                                  # no readout selected yet
    spec = spec.with_sampling(nshots=1000)            # add sampling
    spec = spec.with_expectation(observables=[Z(0)])  # add expectation values

    # or in one line:
    spec = Readout().with_sampling(nshots=1000).with_expectation(observables=[Z(0)])

The finished spec is passed to :meth:`~qilisdk.backends.backend.Backend.execute` (or
:meth:`~qilisdk.speqtrum.speqtrum.SpeQtrum.submit` for remote hardware):

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.digital import Circuit
    from qilisdk.functionals import DigitalPropagation

    backend = QiliSim()
    circuit = Circuit(2)
    functional = DigitalPropagation(circuit)
    result = backend.execute(functional, readout=spec)

.. note::

    At least one readout method must be added before calling ``execute``.  Passing an empty
    ``Readout()`` raises a ``ValueError``.

Any combination of the three readout types can be requested in a single execution by chaining the
``with_*`` calls.  The backend performs one circuit run and extracts all requested outputs.

.. code-block:: python

    from qilisdk.analog import Z
    from qilisdk.readout import Readout

    spec = (
        Readout()
        .with_sampling(nshots=500)
        .with_expectation(observables=[Z(0)])
        .with_state_tomography()
    )
    result = backend.execute(functional, readout=spec)

    counts = result.get_samples()              # from sampling
    ev     = result.get_expectation_values()   # from expectation
    state  = result.get_state()                # from state tomography

Accessing Results
--------------------

:meth:`~qilisdk.backends.backend.Backend.execute` returns a
:class:`~qilisdk.functionals.functional_result.FunctionalResult`.  There are two ways to access the readout data.

Convenient Shortcuts
^^^^^^^^^^^^^^^^^^^^^^

The quickest way to access results:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Property
     - Type
     - Requires
   * - :meth:`get_samples()<qilisdk.functionals.functional_result.FunctionalResult.get_samples>`
     - ``dict[str, int]``
     - :meth:`with_sampling()<qilisdk.readout.readout_spec.Readout.with_sampling>`
   * - :meth:`get_probabilities()<qilisdk.functionals.functional_result.FunctionalResult.get_probabilities>`
     - ``dict[str, float]``
     - :meth:`with_sampling()<qilisdk.readout.readout_spec.Readout.with_sampling>` or :meth:`with_state_tomography()<qilisdk.readout.readout_spec.Readout.with_state_tomography>`
   * - :meth:`get_expectation_values()<qilisdk.functionals.functional_result.FunctionalResult.get_expectation_values>`
     - ``list[float]``
     - :meth:`with_expectation()<qilisdk.readout.readout_spec.Readout.with_expectation>`
   * - :meth:`get_state()<qilisdk.functionals.functional_result.FunctionalResult.get_state>`
     - ``QTensor``
     - :meth:`with_state_tomography()<qilisdk.readout.readout_spec.Readout.with_state_tomography>`

These raise ``ValueError`` at runtime if the corresponding readout was not requested.

Typed Forwarding Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production code or when combining multiple readout types, use the typed forwarding properties.
These return the raw result objects and let the type checker verify access without runtime guards:

.. code-block:: python

    from qilisdk.analog import Z
    from qilisdk.readout import Readout

    spec = Readout().with_sampling(nshots=1000).with_expectation(observables=[Z(0)])
    result = backend.execute(functional, readout=spec)

    # result.sampling is SamplingReadoutResult - the type checker knows this
    top2 = result.sampling.get_probabilities(n=2)

    # result.expectation is ExpectationReadoutResult - the type checker knows this
    raw_evs = result.expectation.expectation_values

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Property
     - Returns
     - Requires
   * - ``result.sampling``
     - ``SamplingReadoutResult | None``
     - :meth:`with_sampling()<qilisdk.readout.readout_spec.Readout.with_sampling>`
   * - ``result.expectation``
     - ``ExpectationReadoutResult | None``
     - :meth:`with_expectation()<qilisdk.readout.readout_spec.Readout.with_expectation>`
   * - ``result.state_tomography``
     - ``StateTomographyReadoutResult | None``
     - :meth:`with_state_tomography()<qilisdk.readout.readout_spec.Readout.with_state_tomography>`

Complete Example
-------------------

The following example runs an analog annealing schedule, collects expectation values and the final
state at every time step, then plots the observable trajectory.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.backends import QutipBackend
    from qilisdk.core import ket
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout

    T = 5.0
    schedule = Schedule(
        hamiltonians={"driver": X(0), "problem": Z(0)},
        coefficients={
            "driver": {(0.0, T): lambda t: 1.0 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=0.1,
    )

    functional = AnalogEvolution(
        schedule=schedule,
        initial_state=(ket(0) - ket(1)).unit(),
        store_intermediate_results=True,
    )

    spec = (
        Readout()
        .with_expectation(observables=[Z(0)])
        .with_state_tomography()
    )

    result = QutipBackend().execute(functional, readout=spec)

    # Final results
    print("Final <Z>:", result.get_expectation_values()[0])
    print("Final state:", result.get_state())

    # Time-resolved expectation values
    ev_trajectory = [step[0] for step in result.get_intermediate_expectation_values()]
    plt.plot(np.linspace(0, T, len(ev_trajectory)), ev_trajectory)
    plt.xlabel("Time")
    plt.ylabel("⟨Z⟩")
    plt.title("Expectation value during annealing")
    plt.show()
