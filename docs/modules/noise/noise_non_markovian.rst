Non-Markovian Noise
=======================

The noise types in :doc:`noise_types` are known as *Markovian*, meaning they act on the current state
and have no memory of what happened before. Many real imperfections
do have memory. A two-level-system (TLS) defect near a qubit, for example, can absorb 
energy or phase from the qubit and hand it back later. This gives effects that no
Markovian model can reproduce, such as coherence that collapses and then revives.

QiliSDK models this kind of noise with the class :class:`~qilisdk.noise.environment_noise.EnvironmentNoise`.
Here the source of the memory is written out as a small set of hidden **environment qubits**. They
are coupled to the system, can carry their own Hamiltonian and Markovian noise, are evolved together with
the system, and are traced out before results are returned. Every readout result (samples, expectation values,
state tomography) refers to the system qubits only.

How it works
--------------------

For a system with Hamiltonian :math:`H_S(t)`, an :class:`~qilisdk.noise.environment_noise.EnvironmentNoise`
adds a static coupling and environment Hamiltonian,

.. math::

    H(t) = H_S(t) \otimes I_E + \sum_k g_k \, P^{(k)}_S \otimes P^{(k)}_E + I_S \otimes H_E,

where each coupling term has a strength :math:`g_k`, an operator :math:`P^{(k)}_S` on the system and an
operator :math:`P^{(k)}_E` on the environment. The full register is evolved with the Lindblad master equation
(as normally done for dynamics simulations), using both the
system's Markovian noise and the environment's own Markovian noise as jump operators. Then the environment
is traced out:

.. math::

    \rho_S(t) = \mathrm{Tr}_E \left[ \rho(t) \right], \qquad \rho(0) = \rho_S(0) \otimes \rho_E(0).

The reduced dynamics of :math:`\rho_S` are non-Markovian even though the full register follows a Markovian
master equation.

The environment qubits come after the system qubits in the register. Environment qubit indices
are local to each :class:`~qilisdk.noise.environment_noise.EnvironmentNoise` (``0`` to
``n_environment_qubits - 1``), so several environments can be added to the same noise model and are
stacked one after another.

Defining an environment
-------------------------------

:class:`~qilisdk.noise.environment_noise.EnvironmentNoise` takes:

- ``n_environment_qubits``: the number of hidden qubits in this environment.
- ``couplings``: a list of ``(strength, system_operator, environment_operator)`` terms, each adding
  ``strength * system_operator ⊗ environment_operator``. The operators can be Pauli operators or
  Hamiltonians, e.g. ``Z(0)`` or ``X(0) * X(1)``. The system operators use system qubit indices, while environment operators use local environment indices.
- ``environment_hamiltonian`` (optional): the free Hamiltonian of the environment, on local indices.
- ``environment_noise`` (optional): Markovian noise on each environment qubit, as a dictionary
  from local environment index to a list of noise sources. Each must have a static Lindblad form, such as
  :class:`~qilisdk.noise.amplitude_damping.AmplitudeDamping`,
  :class:`~qilisdk.noise.dephasing.Dephasing` or a constant-rate
  :class:`~qilisdk.noise.representations.LindbladGenerator`, since an environment qubit has no gate
  duration to derive rates from.
- ``environment_state`` (optional): the initial environment state, as a ket or a density matrix.
  Defaults to :math:`|0 \dots 0\rangle`.

Coupling strengths, rates and gate/schedule times are combined directly, so they must use consistent units.

An :class:`~qilisdk.noise.environment_noise.EnvironmentNoise` can only be added globally, since its
couplings already name the system qubits it acts on:

.. code-block:: python

    from qilisdk.analog import X, Z
    from qilisdk.core import ket
    from qilisdk.noise import AmplitudeDamping, EnvironmentNoise, NoiseModel

    # One TLS, coupled to qubit 0 through Z (x) X, with its own free Hamiltonian and relaxation
    tls = EnvironmentNoise(
        n_environment_qubits=1,
        couplings=[(0.5, Z(0), X(0))],
        environment_hamiltonian=0.2 * Z(0),
        environment_noise={0: [AmplitudeDamping(t1=20.0)]},
        environment_state=(ket(0) + ket(1)).unit(),
    )

    noise_model = NoiseModel()
    noise_model.add(tls)

Example: coherence revivals
--------------------------------------

In a Ramsey-style experiment a qubit is prepared in :math:`|+\rangle` and left idle. With Markovian dephasing
:math:`\langle X \rangle` only decays. A qubit coupled through :math:`J \, Z \otimes Z` to a TLS in
:math:`|+\rangle` instead follows :math:`\langle X \rangle = \cos(2 J t)`: the coherence collapses at
:math:`t = \pi / 4J` and comes back at :math:`t = \pi / 2J` (with the opposite sign) and :math:`t = \pi / J`.
In a digital circuit the idle time is set through the gate time of an :class:`~qilisdk.digital.gates.I`
gate:

.. code-block:: python

    import numpy as np

    from qilisdk.analog import X, Z
    from qilisdk.backends import QiliSim
    from qilisdk.core import ket
    from qilisdk.digital import Circuit, H, I
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.noise import EnvironmentNoise, NoiseModel
    from qilisdk.readout import Readout

    # Create the circuit: put the qubit in the |+> state and wait
    circuit = Circuit(nqubits=1)
    circuit.add(H(0))
    circuit.add(I(0))

    # Consider different wait times
    for idle_time in [np.pi / 8, np.pi / 4, np.pi / 2]:
        
        # The noise model for this wait time
        noise_model = NoiseModel()
        noise_model.add(
            EnvironmentNoise(
                n_environment_qubits=1,
                couplings=[(1.0, Z(0), Z(0))],
                environment_state=(ket(0) + ket(1)).unit(),
            )
        )
        noise_model.noise_config.set_gate_time(H, 1e-6)
        noise_model.noise_config.set_gate_time(I, idle_time)

        # Run the simulation
        result = QiliSim(noise_model=noise_model).execute(
            DigitalPropagation(circuit), readout=Readout().with_expectation(observables=[X(0)])
        )
        print(f"t = {idle_time:.3f}: <X> = {result.get_expectation_values()[0]:+.3f}")

which prints

.. code-block:: text

    t = 0.393: <X> = +0.707
    t = 0.785: <X> = -0.000
    t = 1.571: <X> = -1.000

Example: analog evolution
---------------------------------

The same environment can be attached to an :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`. The coupling and environment Hamiltonian act at every
time step of the schedule:

.. code-block:: python

    from qilisdk.analog import Schedule, X, Z
    from qilisdk.backends import QiliSim
    from qilisdk.core import QTensor
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.noise import Dephasing, EnvironmentNoise, NoiseModel
    from qilisdk.readout import Readout

    # Create the schedule defining the evolution
    schedule = Schedule.linear(X(0), Z(0), dt=0.01, total_time=5.0)

    # Create the noise model
    noise_model = NoiseModel()
    noise_model.add(
        EnvironmentNoise(
            n_environment_qubits=1,
            couplings=[(0.3, X(0), Z(0))],
            environment_noise={0: [Dephasing(t_phi=2.0)]},
        )
    )

    # Run the simulation
    result = QiliSim(noise_model=noise_model).execute(
        AnalogEvolution(schedule=schedule, initial_state=QTensor.uniform(1)),
        readout=Readout().with_expectation(observables=[Z(0)]),
    )

Several environments
----------------------------

Each :class:`~qilisdk.noise.environment_noise.EnvironmentNoise` is an independent set of environment
qubits, placed after the system qubits in the order they were added. For example, two TLSs coupled to
different qubits of a two-qubit system:

.. code-block:: python

    from qilisdk.analog import X, Z
    from qilisdk.noise import AmplitudeDamping, EnvironmentNoise, NoiseModel

    noise_model = NoiseModel()
    noise_model.add(EnvironmentNoise(n_environment_qubits=1, couplings=[(0.4, Z(0), X(0))]))
    noise_model.add(
        EnvironmentNoise(
            n_environment_qubits=1,
            couplings=[(0.7, Z(1), X(0))],
            environment_noise={0: [AmplitudeDamping(t1=5.0)]},
        )
    )

    # The simulated register is [system 0, system 1, first TLS, second TLS]
    print([environment.n_environment_qubits for environment in noise_model.non_markovian_noise])

Environment noise can be combined with Markovian noise on the system, as long as that noise has a
Lindblad form (for example :class:`~qilisdk.noise.amplitude_damping.AmplitudeDamping` or
:class:`~qilisdk.noise.dephasing.Dephasing`, globally or per qubit), and with
:class:`~qilisdk.noise.readout_assignment.ReadoutAssignment` and parameter perturbations.