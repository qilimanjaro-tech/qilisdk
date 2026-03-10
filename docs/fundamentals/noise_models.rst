Noise Models
==============

The :mod:`qilisdk.noise` module contains classes that represent different types of noise in quantum systems.
These noise models can be integrated into quantum simulations to account for real-world imperfections and
decoherence effects. A :class:`~qilisdk.noise.noise_model.NoiseModel` can contains various types of noise, 
each of which is then applied during simulation.

.. table::
   :align: left
   :widths: auto

   ========================================================================== ============================ 
   Backend                                                                    Noise Models Supported
   ========================================================================== ============================
   :class:`~qilisdk.backends.qilisim.QiliSim`                                 ✔                                
   -------------------------------------------------------------------------- ----------------------------
   :class:`~qilisdk.backends.cuda_backend.CudaBackend`                        ✔                                
   -------------------------------------------------------------------------- ---------------------------- 
   :class:`~qilisdk.backends.qutip_backend.QutipBackend`                      ✕ 
   ========================================================================== ============================ 


Usage
-----------

Noise passes are added to a :class:`~qilisdk.noise.noise_model.NoiseModel`, which should then be included when creating a backend. 
For example, to create a simple circuit and then simulate it with a CUDA backend containing a bit-flip noise model:

.. code-block:: python

    from qilisdk.noise import NoiseModel, BitFlip
    from qilisdk.digital import X, Circuit
    from qilisdk.functionals import Sampling
    from qilisdk.backends import CudaBackend

    # Define the random circuit and sampler
    c = Circuit(nqubits=2)
    c.add(X(0))
    c.add(X(1))
    sampler = Sampling(c, nshots=1000)

    # Apply bit-flip noise with 50% probability to X gates on qubit 1
    nm = NoiseModel()
    nm.add(BitFlip(probability=0.5), gate=X, qubits=[1]) 

    # Execute with CUDA backend
    backend_cuda = CudaBackend(noise_model=nm)
    res = backend_cuda.execute(sampler)
    print(res)

Summary of Noise Types
----------------------

Some of the noise passes available in QiliSDK can be used for both digital circuits 
and analog schedules, while others are specific to one or the other:

.. table::
   :align: left
   :widths: auto

   ========================================================================== ==================== ======================
   Noise Type                                                                 Digital Circuits     Analog Schedules
   ========================================================================== ==================== ======================
   :class:`~qilisdk.noise.representations.KrausChannel`                       ✔                                
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.representations.LindbladGenerator`                                       ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.pauli_channel.PauliChannel`                         ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.bit_flip.BitFlip`                                   ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.phase_flip.PhaseFlip`                               ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.depolarizing.Depolarizing`                          ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.amplitude_damping.AmplitudeDamping`                 ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.dephasing.Dephasing`                                ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.gaussian_perturbation.GaussianPerturbation`         ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.offset_perturbation.OffsetPerturbation`             ✔                    ✔
   -------------------------------------------------------------------------- -------------------- ----------------------
   :class:`~qilisdk.noise.readout_assignment.ReadoutAssignment`               ✔
   ========================================================================== ==================== ======================

Noise Config
----------------------

By default, when converting between different noise representations (e.g. from Kraus operators to Lindblad generators),
certain parameters are assumed, such as gate durations. 
These defaults can be modified using the :class:`~qilisdk.noise.noise_config.NoiseConfig` class:

.. code-block:: python

    from qilisdk.noise import NoiseModel, NoiseConfig
    from qilisdk.digital import X

    # Create a noise configuration, setting the X gate duration to 20 ns
    conf = NoiseConfig()
    conf.set_gate_duration[X] = 20e-9

    # Define a simple noise model using this config
    nm = NoiseModel(noise_config=conf)

Parameters and their defaults are as follows:

.. table::
   :align: left
   :widths: auto
   
   ========================================================================== ==========================
   Parameter                                                                  Default Value                                           
   ========================================================================== ==========================
   :attr:`~qilisdk.noise.noise_config.NoiseConfig.gate_times`                 1.0 for all gates
   ========================================================================== ==========================

Noise Types
----------------------

KrausChannel
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.representations.KrausChannel` is a base class for digital noise models that use Kraus operators 
to represent quantum noise channels. Kraus operators can be used to represent a generic quantum channel :math:`\mathcal{E}` acting on a density matrix :math:`\rho` as:

.. math::

    \mathcal{E}(\rho) = \sum_i K_i \rho K_i^\dagger,

where :math:`K_i` are the Kraus operators satisfying the completeness relation :math:`\sum_i K_i^\dagger K_i = I`.
The class is initialized with a list of Kraus operators given as QTensors.

.. code-block:: python

    from qilisdk.noise import KrausChannel
    from qilisdk.core import QTensor
    import numpy as np
    p = 0.1
    K1 = QTensor(np.array([[1, 0], [0, np.sqrt(1 - p)]]))
    K2 = QTensor(np.array([[0, np.sqrt(p)], [0, 0]]))
    kraus_noise = KrausChannel(operators=[K1, K2])

LindbladGenerator
^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.representations.LindbladGenerator` is a base class for analog noise models that use the Lindblad master equation 
to represent quantum noise processes. The Lindblad master equation describes the time evolution of a density matrix :math:`\rho` under
both unitary dynamics and dissipative processes:

.. math::

    \frac{d\rho}{dt} = -i[H, \rho] + \sum_i \left( J_i \rho J_i^\dagger - \frac{1}{2} \{ J_i^\dagger J_i, \rho \} \right),

where :math:`H` is the system Hamiltonian, and :math:`J_i` are the jump operators representing different dissipation channels.
The class is initialized with a list of jump operators given as QTensors and their corresponding rates (assumed to be 1 if not specified).

.. code-block:: python

    from qilisdk.noise import LindbladGenerator
    from qilisdk.core import QTensor
    import numpy as np
    J1 = QTensor(np.array([[0, 1], [0, 0]]))
    J2 = QTensor(np.array([[1, 0], [0, 0]]))
    lindblad_noise = LindbladGenerator(jump_operators=[J1, J2], rates=[0.1, 0.2])

PauliChannel
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.pauli_channel.PauliChannel` is a base class for noise models that use Pauli operators.
A Pauli channel applies the Pauli operators (I, X, Y, Z) with certain probabilities.
The channel can be represented as:

.. math::

    \mathcal{E}(\rho) = p_I \rho + p_X X \rho X + p_Y Y \rho Y + p_Z Z \rho Z,

where :math:`p_I`, :math:`p_X`, :math:`p_Y`, and :math:`p_Z` are the probabilities for 
applying the respective Pauli operators, satisfying :math:`p_I + p_X + p_Y + p_Z = 1`.
This corresponds to Kraus operators:

.. math::

    K_0 = \sqrt{p_I} I, \quad K_1 = \sqrt{p_X} X, \quad K_2 = \sqrt{p_Y} Y, \quad K_3 = \sqrt{p_Z} Z,

Or jump operators:

.. math::

    J_1 = \sqrt{\gamma_X} X, \quad J_2 = \sqrt{\gamma_Y} Y, \quad J_3 = \sqrt{\gamma_Z} Z,

where :math:`\gamma_i` are rates related to the probabilities.

.. code-block:: python

    from qilisdk.noise import PauliChannel
    pauli_noise = PauliChannel(p_X=0.1, p_Y=0.1, p_Z=0.1)

BitFlip
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.bit_flip.BitFlip` represents a bit-flip error model where each qubit 
has a certain probability of flipping its state from |0⟩ to |1⟩ or from |1⟩ to |0⟩.
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = (1 - p) \rho + p X \rho X,

where :math:`p` is the bit-flip probability. As such, it is a special case of :class:`~qilisdk.noise.pauli_channel.PauliChannel` 
with :math:`p_X = p` and :math:`p_I = 1 - p`. This can therefore be written as Kraus operators:

.. math::

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{p} X,

or similarly as a jump operator:

.. math::

    J = \sqrt{\gamma} X,

where now :math:`\gamma` is the bit-flip rate.

.. code-block:: python

    from qilisdk.noise import BitFlip
    bit_flip_noise = BitFlip(probability=0.1)

PhaseFlip
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.phase_flip.PhaseFlip` represents a phase-flip error model where each qubit 
has a certain probability of flipping its phase, changing the sign of the |1⟩ state.
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = (1 - p) \rho + p Z \rho Z,

where :math:`p` is the phase-flip probability. As such, it is a special case of :class:`~qilisdk.noise.pauli_channel.PauliChannel`
with :math:`p_Z = p` and :math:`p_I = 1 - p`. This can therefore be written as Kraus operators:

.. math::

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{p} Z,

or similarly as a jump operator:

.. math::

    J = \sqrt{\gamma} Z,

where now :math:`\gamma` is the phase-flip rate.

.. code-block:: python

    from qilisdk.noise import PhaseFlip
    phase_flip_noise = PhaseFlip(probability=0.1)

Depolarizing
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.depolarizing.Depolarizing` represents a depolarizing error model 
where each qubit has a certain probability of being replaced by the maximally mixed state.
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = (1 - p) \rho + \frac{p}{3} (X \rho X + Y \rho Y + Z \rho Z),

where :math:`p` is the depolarizing probability. As such, it is a special case of :class:`~qilisdk.noise.pauli_channel.PauliChannel`
with :math:`p_X = p_Y = p_Z = p/3` and :math:`p_I = 1 - p`. This can therefore be written as Kraus operators:

.. math::

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{\frac{p}{3}} X, \quad K_2 = \sqrt{\frac{p}{3}} Y, \quad K_3 = \sqrt{\frac{p}{3}} Z,

or similarly as jump operators:

.. math::

    J_1 = \sqrt{\frac{\gamma}{3}} X, \quad J_2 = \sqrt{\frac{\gamma}{3}} Y, \quad J_3 = \sqrt{\frac{\gamma}{3}} Z,

where :math:`\gamma` is the depolarizing rate.

.. code-block:: python

    from qilisdk.noise import Depolarizing
    depolarizing_noise = Depolarizing(probability=0.1)

AmplitudeDampingNoise
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.amplitude_damping.AmplitudeDamping` represents an amplitude damping error model 
where each qubit has a certain probability of decaying from the excited state |1⟩ to the ground state |0⟩.
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger.

Here the Kraus operators are defined as:

.. math::

    E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - p} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{p} \\ 0 & 0 \end{pmatrix},

where :math:`p` is the amplitude damping probability. The channel can also be represented using the jump operator:

.. math::

    J = \sqrt{\gamma} \sigma_-,

where :math:`\gamma` is the amplitude damping rate and :math:`\sigma_-` is the lowering operator. 
The channel is defined by the :math:`T_1` time constant, with the relations:

.. math::

    p = 1 - e^{-t / T_1},
    \gamma = \frac{1}{T_1},

where :math:`t` is the time duration over which the amplitude damping occurs.

.. code-block:: python

    from qilisdk.noise import AmplitudeDamping
    amplitude_damping_noise = AmplitudeDamping(t1=0.1)

Dephasing
^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.dephasing.Dephasing` represents a dephasing error model where each qubit has a 
certain probability of losing coherence between its |0⟩ and |1⟩ states.
This corresponds to the following channel:

.. math::  

    \mathcal{E}(\rho) = (1 - p) \rho + p Z \rho Z

where :math:`p` is the dephasing probability. As such, it is a special case of :class:`~qilisdk.noise.pauli_channel.PauliChannel`
with :math:`p_Z = p` and :math:`p_I = 1 - p`. This can therefore be written as Kraus operators:

.. math::   

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{p} Z,

or similarly as a jump operator:

.. math::

    J = \sqrt{\gamma} Z,

where now :math:`\gamma` is the dephasing rate. The channel is defined by the :math:`T_\phi` time constant, with the relations:

.. math::

    p = \frac{1}{2} \left( 1 - e^{-t / T_\phi} \right),
    \gamma = \frac{1}{T_\phi},

where :math:`t` is the time duration over which the dephasing occurs.

.. code-block:: python

    from qilisdk.noise import Dephasing
    dephasing_noise = Dephasing(t_phi=0.1)


GaussianPerturbation
^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.gaussian_perturbation.GaussianPerturbation` represents a noise model that adds Gaussian-distributed perturbations
to the parameters of gates or Hamiltonian terms in a quantum circuit or schedule. This simulates
imperfections in control signals that can lead to deviations from the intended operations.
The perturbations are drawn from a Gaussian distribution with a specified standard deviation:

.. math::

    \delta \theta \sim \mathcal{N}(\mu, \sigma^2),

where :math:`\sigma` is the standard deviation of the perturbations and :math:`\mu` is the mean (defaulting to 0).

.. code-block:: python

    from qilisdk.noise import GaussianPerturbation
    gaussian_perturbation_noise = GaussianPerturbation(stddev=0.05, mean=0.0)

OffsetPerturbation
^^^^^^^^^^^^^^^^^^^^^
:class:`~qilisdk.noise.offset_perturbation.OffsetPerturbation` represents a noise model that adds fixed offset perturbations
to the parameters of gates or Hamiltonian terms in a quantum circuit or schedule. This simulates
systematic errors in control signals that consistently shift the intended operations.
The offsets are specified as fixed values for each parameter.

.. code-block:: python

    from qilisdk.noise import OffsetPerturbation
    offset_perturbation_noise = OffsetPerturbation(offset=1.3)

ReadoutAssignment
^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.noise.readout_assignment.ReadoutAssignment` represents a readout error model that simulates
imperfections in the measurement process of qubits. It is defined by two probabilities: 
the probability of misreading a |0⟩ state as |1⟩ (:math:`p_{0 \to 1}`) and the probability of misreading a |1⟩ state as |0⟩ (:math:`p_{1 \to 0}`).

.. code-block:: python

    from qilisdk.noise import ReadoutAssignment
    readout_assignment_noise = ReadoutAssignment(p01=0.05, p10=0.1)