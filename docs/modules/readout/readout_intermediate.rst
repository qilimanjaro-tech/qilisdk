Intermediate Results
--------------------

When a functional is constructed with ``store_intermediate_results=True`` (currently supported by
:class:`~qilisdk.functionals.analog_evolution.AnalogEvolution` and :class:`~qilisdk.functionals.quantum_reservoirs.QuantumReservoir`), or via 
mid-circuit measurements in a :class:`~qilisdk.digital.circuit.Circuit`,
the backend stores a readout result for every time step.  The same readout methods apply at each step.

.. code-block:: python

    from qilisdk.analog import Schedule, Z, X
    from qilisdk.core import ket
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim

    T = 5.0
    schedule = Schedule(
        hamiltonians={"driver": X(0), "problem": Z(0)},
        coefficients={
            "driver": {(0.0, T): lambda t: 1.0 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=0.1,
    )

    backend = QiliSim()
    functional = AnalogEvolution(schedule, initial_state=ket(0), store_intermediate_results=True)
    spec = Readout().with_expectation(observables=[Z(0)]).with_state_tomography()
    result = backend.execute(functional, readout=spec)

    # Per-step expectation values (intermediate steps + final step)
    all_evs     = result.get_intermediate_expectation_values()  # list[list[float]]
    all_states  = result.get_intermediate_states()              # list[QTensor]
    all_probs   = result.get_intermediate_probabilities()       # list[dict[str, float]]
    all_samples = result.get_intermediate_samples()             # list[dict[str, int]]

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Property
     - Type
     - Requires
   * - ``result.get_intermediate_samples()``
     - ``list[dict[str, int]]``
     - ``.with_sampling()``
   * - ``result.get_intermediate_probabilities()``
     - ``list[dict[str, float]]``
     - ``.with_sampling()`` or ``.with_state_tomography()``
   * - ``result.get_intermediate_expectation_values()``
     - ``list[list[float]]``
     - ``.with_expectation()``
   * - ``result.get_intermediate_states()``
     - ``list[QTensor]``
     - ``.with_state_tomography()``

All intermediate lists contain one entry per time step in chronological order, with the **final**
step appended last.  ``result[i]`` returns the :class:`~qilisdk.readout.readout_result.ReadoutCompositeResults`
for step ``i``.