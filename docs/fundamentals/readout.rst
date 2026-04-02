Readout
=======

Readout is the mechanism that controls **what information is extracted from the quantum backend** after a
functional is executed.  It is specified separately from the functional itself, so the same circuit or
schedule can be measured in different ways without any code change to the functional.

The entry point is always :class:`~qilisdk.readout.ReadoutSpec`.

.. contents::
   :local:
   :depth: 2

----

ReadoutSpec — the builder
--------------------------

:class:`~qilisdk.readout.ReadoutSpec` is a **immutable builder** that accumulates readout methods by
chaining ``with_*`` calls.  Each call returns a *new* spec with one additional slot filled; the original
object is never modified.

.. code-block:: python

    from qilisdk.readout import ReadoutSpec

    spec = ReadoutSpec()                         # no readout selected yet
    spec = spec.with_sampling(nshots=1000)       # add sampling
    spec = spec.with_expectation(observables=[Z(0)])  # add expectation values
    # or in one line:
    spec = ReadoutSpec().with_sampling(nshots=1000).with_expectation(observables=[Z(0)])

The finished spec is passed to :meth:`~qilisdk.backends.backend.Backend.execute` (or
:meth:`~qilisdk.speqtrum.speqtrum.SpeQtrum.submit` for remote hardware):

.. code-block:: python

    result = backend.execute(functional, readout=spec)

.. note::

    At least one readout method must be added before calling ``execute``.  Passing an empty
    ``ReadoutSpec()`` raises ``ValueError``.

Readout types
-------------

with_sampling — bitstring measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instructs the backend to perform ``nshots`` projective measurements in the computational basis and
collect the bitstring counts.

.. code-block:: python

    from qilisdk.readout import ReadoutSpec

    spec = ReadoutSpec().with_sampling(nshots=1000)
    result = backend.execute(functional, readout=spec)

    # Access the results
    counts = result.samples           # dict[str, int]  e.g. {"00": 512, "11": 488}
    probs  = result.probabilities     # dict[str, float] normalised to 1.0

    # Top-k most probable outcomes
    top3 = result.readout_results.sampling.get_probabilities(n=3)

**Parameters**

- **nshots** (``int``): Number of measurement shots.  Must be a positive integer.

**When to use**

Use sampling when you need the full bitstring distribution — for instance to evaluate a combinatorial
cost function, run QAOA post-processing, or compute error rates.

with_expectation — expectation values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instructs the backend to compute ``⟨ψ|O|ψ⟩`` for each observable in the list.

.. code-block:: python

    from qilisdk.analog import X, Y, Z
    from qilisdk.readout import ReadoutSpec

    spec = ReadoutSpec().with_expectation(observables=[Z(0), X(0), Y(0)], nshots=0)
    result = backend.execute(functional, readout=spec)

    evs = result.expectation_values   # list[float], one entry per observable
    # e.g. [-0.994, 0.047, -0.100]

**Parameters**

- **observables** (``list[Hamiltonian | QTensor]``): The operators whose expectation values are
  requested.  Each entry can be a :class:`~qilisdk.analog.Hamiltonian` expression (e.g. ``Z(0) + Z(1)``)
  or a :class:`~qilisdk.core.QTensor` directly.
- **nshots** (``int``, default ``0``): Number of shots for stochastic estimation.  ``0`` uses the
  exact state-vector inner product — no sampling noise, only available on simulators.

**When to use**

Use expectation values when you need a scalar energy or observable readout — for example in
variational algorithms (VQE, QAOA energy evaluation) or analog time-evolution studies.

with_state_tomography — full quantum state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instructs the backend to return the full quantum state vector (or density matrix) after execution.

.. code-block:: python

    from qilisdk.readout import ReadoutSpec

    spec = ReadoutSpec().with_state_tomography()
    result = backend.execute(functional, readout=spec)

    state = result.state              # QTensor — the full ket or density matrix
    probs = result.probabilities      # dict[str, float] derived from |amplitudes|²

**Parameters**

- **method** (``Literal["exact"]``, default ``"exact"``): Tomography method.  Currently only
  ``"exact"`` is supported — the backend returns the raw state vector.

**When to use**

Use state tomography when you need the complete quantum state for post-processing: computing custom
observables offline, visualising the state, checking fidelity, or seeding the next step of a
multi-step algorithm.

.. note::

    State tomography is only available on simulators.  Physical QPUs (accessed via
    :class:`~qilisdk.speqtrum.speqtrum.SpeQtrum`) do not support this readout type.

----

Combining multiple readout types
----------------------------------

Any combination of the three readout types can be requested in a single execution by chaining the
``with_*`` calls.  The backend performs one circuit run and extracts all requested outputs.

.. code-block:: python

    from qilisdk.analog import Z
    from qilisdk.readout import ReadoutSpec

    spec = (
        ReadoutSpec()
        .with_sampling(nshots=500)
        .with_expectation(observables=[Z(0)])
        .with_state_tomography()
    )
    result = backend.execute(functional, readout=spec)

    counts = result.samples              # from sampling
    ev     = result.expectation_values   # from expectation
    state  = result.state                # from state tomography

----

Accessing results
-----------------

:meth:`~qilisdk.backends.backend.Backend.execute` returns a
:class:`~qilisdk.functionals.FunctionalResult`.  There are two ways to access the readout data.

Convenience shortcuts
^^^^^^^^^^^^^^^^^^^^^

The quickest way to access results for interactive use:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Property
     - Type
     - Requires
   * - ``result.samples``
     - ``dict[str, int]``
     - ``.with_sampling()``
   * - ``result.probabilities``
     - ``dict[str, float]``
     - ``.with_sampling()`` or ``.with_state_tomography()``
   * - ``result.expectation_values``
     - ``list[float]``
     - ``.with_expectation()``
   * - ``result.state``
     - ``QTensor``
     - ``.with_state_tomography()``

These raise ``ValueError`` at runtime if the corresponding readout was not requested.

Typed forwarding properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production code or when combining multiple readout types, use the typed forwarding properties.
These return the raw result objects and let the type checker verify access without runtime guards:

.. code-block:: python

    from qilisdk.analog import Z
    from qilisdk.readout import ReadoutSpec

    spec = ReadoutSpec().with_sampling(nshots=1000).with_expectation(observables=[Z(0)])
    result = backend.execute(functional, readout=spec)

    # result.sampling is SamplingReadoutResult — type checker knows this
    top2 = result.sampling.get_probabilities(n=2)

    # result.expectation is ExpectationReadoutResult — type checker knows this
    raw_evs = result.expectation.expectation_values

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Property
     - Returns
     - Requires
   * - ``result.sampling``
     - ``SamplingReadoutResult | None``
     - ``.with_sampling()``
   * - ``result.expectation``
     - ``ExpectationReadoutResult | None``
     - ``.with_expectation()``
   * - ``result.state_tomography``
     - ``StateTomographyReadoutResult | None``
     - ``.with_state_tomography()``

----

Intermediate results
--------------------

When a functional is constructed with ``store_intermediate_results=True`` (currently supported by
:class:`~qilisdk.functionals.AnalogEvolution` and :class:`~qilisdk.functionals.QuantumReservoir`), the backend stores a 
readout result for every time step.  The same readout methods apply at each step.

.. code-block:: python

    from qilisdk.analog import Schedule, Z
    from qilisdk.core import ket
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import ReadoutSpec

    functional = AnalogEvolution(schedule, initial_state=ket(0), store_intermediate_results=True)
    spec = ReadoutSpec().with_expectation(observables=[Z(0)]).with_state_tomography()
    result = backend.execute(functional, readout=spec)

    # Per-step expectation values (intermediate steps + final step)
    all_evs    = result.intermediate_expectation_values  # list[list[float]]
    all_states = result.intermediate_states              # list[QTensor]

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Property
     - Type
     - Requires
   * - ``result.intermediate_samples``
     - ``list[dict[str, int]]``
     - ``.with_sampling()``
   * - ``result.intermediate_probabilities``
     - ``list[dict[str, float]]``
     - ``.with_sampling()`` or ``.with_state_tomography()``
   * - ``result.intermediate_expectation_values``
     - ``list[list[float]]``
     - ``.with_expectation()``
   * - ``result.intermediate_states``
     - ``list[QTensor]``
     - ``.with_state_tomography()``

All intermediate lists contain one entry per time step in chronological order, with the **final**
step appended last.  ``result[i]`` returns the :class:`~qilisdk.readout.ReadoutCompositeResults`
for step ``i``.

----

Complete example
----------------

The following example runs an analog annealing schedule, collects expectation values and the final
state at every time step, then plots the observable trajectory.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.backends import QutipBackend
    from qilisdk.core import ket
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import ReadoutSpec

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
        ReadoutSpec()
        .with_expectation(observables=[Z(0)])
        .with_state_tomography()
    )

    result = QutipBackend().execute(functional, readout=spec)

    # Final results
    print("Final <Z>:", result.expectation_values[0])
    print("Final state:", result.state)

    # Time-resolved expectation values
    ev_trajectory = [step[0] for step in result.intermediate_expectation_values]
    plt.plot(np.linspace(0, T, len(ev_trajectory)), ev_trajectory)
    plt.xlabel("Time")
    plt.ylabel("⟨Z⟩")
    plt.title("Expectation value during annealing")
    plt.show()
