TrotterizedSchedule
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.digital.ansatz.TrotterizedSchedule` builds a digital circuit that follows a
time-ordered schedule of Hamiltonians. Each schedule slice is evolved using a fixed number of
Trotter steps, and you may optionally prepend a state-initialization circuit or list of gates.

Configuration options:

- **schedule**: A :class:`~qilisdk.analog.schedule.Schedule` of Hamiltonians to trotterize.
- **trotter_steps**: Number of Trotter steps per schedule slice. Defaults to 1.

**Example**

.. code-block:: python

    from qilisdk.analog.hamiltonian import Z as pauli_z
    from qilisdk.analog.schedule import Schedule
    from qilisdk.digital.ansatz import TrotterizedSchedule

    hamiltonian = pauli_z(0)
    schedule = Schedule(
        hamiltonians={"h": hamiltonian},
        dt=0.1,
        total_time=1
    )
    ansatz = TrotterizedSchedule(
        schedule=schedule,
        trotter_steps=1,
    )
    ansatz.draw()

