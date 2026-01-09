Noise Models
==============

The :mod:`qilisdk.noise_models` module contains classes that represent different types of noise in quantum systems.
These noise models can be integrated into quantum simulations to account for real-world imperfections and
decoherence effects. A :class:`~qilisdk.noise_models.noise_model.NoiseModel` can contains various types of noise, 
each of which is then applied during simulation.
There are three types of noise in QiliSDK: 
 - digital (used in digital circuits)
 - analog (used in analog schedules)
 - parameter noise (used to model imperfections in parameterized gates or schedules)

Note: For now, only the :class:`~qilisdk.backends.cuda_backend.CudaBackend` supports noise models.

Usage
-----------

Noise passes are added to a :class:`~qilisdk.noise_models.noise_model.NoiseModel`, which is then included when executing a functional. 
For example, to add two types of digital noise to a noise model and then use it to simulate a circuit using the CUDA backend:

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.noise_models import NoiseModel, DigitalBitFlipNoise, DigitalDephasingNoise
    from qilisdk.digital import X, Circuit
    from qilisdk.functionals import Sampling

    # Define the random circuit and sampler
    c = Circuit(nqubits=2)
    c.add(X(0))
    c.add(X(1))
    sampler = Sampling(c, nshots=1000)

    # Define a simple noise model
    nm = NoiseModel()
    nm.add(DigitalBitFlipNoise(probability=0.3, affected_qubits=[1], affected_gates=[X]))
    nm.add(DigitalDephasingNoise(probability=0.2, affected_qubits=[0], affected_gates=[X]))

    # Execute with CUDA backend
    backend_cuda = CudaBackend()
    res = backend_cuda.execute(sampler, noise_model=nm)
    print(res)

Digital Noise Types
----------------------

Each of the digital noise types is applied after the execution of the 
specified gates on the specified qubits in a digital circuit. 
For instance, if a :class:`~qilisdk.noise_models.digital_noise.DigitalBitFlipNoise` is added to a noise model with 
an :class:`~qilisdk.digital.gates.X` gate and qubit 0 specified, then after every application of an :class:`~qilisdk.digital.gates.X` gate on qubit 0, 
there is a chance that a bit-flip error will occur on that qubit.
If no gates or qubits are specified, the noise is applied after every gate on all qubits.

KrausNoise
^^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.digital_noise.KrausNoise` is a base class for digital noise models that use Kraus operators 
to represent quantum noise channels. Kraus operators can be used to represent a generic quantum channel :math:`\mathcal{E}` acting on a density matrix :math:`\rho` as:

.. math::

    \mathcal{E}(\rho) = \sum_i K_i \rho K_i^\dagger

where :math:`K_i` are the Kraus operators satisfying the completeness relation :math:`\sum_i K_i^\dagger K_i = I`.
The class is initialized with a list of Kraus operators given as QTensors, along with the qubits and gates for which the noise should be applied.

.. code-block:: python

    from qilisdk.noise_models import KrausNoise
    from qilisdk.core import QTensor
    import numpy as np
    p = 0.1
    K1 = QTensor(np.array([[1, 0], [0, np.sqrt(1 - p)]]))
    K2 = QTensor(np.array([[0, np.sqrt(p)], [0, 0]]))
    kraus_noise = KrausNoise(kraus_operators=[K1, K2])

DigitalBitFlipNoise
^^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.digital_noise.DigitalBitFlipNoise` represents a bit-flip error model where each qubit 
has a certain probability of flipping its state from |0⟩ to |1⟩ or from |1⟩ to |0⟩ after the application of the specified gates.
The noise is characterized by a probability parameter that defines the likelihood of a bit-flip occurring. 
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = (1 - p) \rho + p X \rho X

where :math:`p` is the bit-flip probability.
This class extends :class:`~qilisdk.noise_models.digital_noise.KrausNoise`, using Kraus operators:

.. math::

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{p} X.

Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import DigitalBitFlipNoise
    bit_flip_noise = DigitalBitFlipNoise(probability=0.1)

DigitalDephasingNoise
^^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.digital_noise.DigitalDephasingNoise` represents a dephasing error model where each qubit has a 
certain probability of losing coherence between its |0⟩ and |1⟩ states after the application of the specified gates.
The noise is characterized by a probability parameter that defines the likelihood of a dephasing event occurring. 
This corresponds to the following channel:

.. math::  

    \mathcal{E}(\rho) = (1 - p) \rho + p Z \rho Z

where :math:`p` is the dephasing probability.
This class extends :class:`~qilisdk.noise_models.digital_noise.KrausNoise`, using Kraus operators:

.. math::   

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{p} Z.

Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import DigitalDephasingNoise
    dephasing_noise = DigitalDephasingNoise(probability=0.1)

DigitalDepolarizingNoise
^^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.digital_noise.DigitalDepolarizingNoise` represents a depolarizing error model 
where each qubit has a certain probability of being replaced by the maximally mixed state after the application of the specified gates.
The noise is characterized by a probability parameter that defines the likelihood of a depolarizing event occurring. 
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = (1 - p) \rho + \frac{p}{3} (X \rho X + Y \rho Y + Z \rho Z)

where :math:`p` is the depolarizing probability.
This class extends :class:`~qilisdk.noise_models.digital_noise.KrausNoise`, using Kraus operators:

.. math::

    K_0 = \sqrt{1 - p} I, \quad K_1 = \sqrt{\frac{p}{3}} X, \quad K_2 = \sqrt{\frac{p}{3}} Y, \quad K_3 = \sqrt{\frac{p}{3}} Z

Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import DigitalDepolarizingNoise
    depolarizing_noise = DigitalDepolarizingNoise(probability=0.1)

DigitalAmplitudeDampingNoise
^^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.digital_noise.DigitalAmplitudeDampingNoise` represents an amplitude damping error model 
where each qubit has a certain probability of decaying from the excited state |1⟩ to the ground state |0⟩ after the application of the specified gates.
The noise is characterized by a probability parameter that defines the likelihood of an amplitude damping event occurring. 
This corresponds to the following channel:

.. math::

    \mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger

where the Kraus operators are defined as:

.. math::

    E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - p} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{p} \\ 0 & 0 \end{pmatrix}

where :math:`p` is the amplitude damping probability.
This class extends :class:`~qilisdk.noise_models.digital_noise.KrausNoise`, using the above Kraus operators.
Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import DigitalAmplitudeDampingNoise
    amplitude_damping_noise = DigitalAmplitudeDampingNoise(probability=0.1)

Analog Noise Types
----------------------

Each of the analog noise types is continuously applied during the time evolution of an analog schedule. 
For instance, if a :class:`~qilisdk.noise_models.analog_noise.AnalogDephasingNoise` is added to a noise model, 
then during the entire duration of the analog schedule
there is a chance that dephasing errors will occur on the specified qubits, with some given rate or probability.
As with the digital noises, if no qubits are specified, the noise is applied to all qubits.

DissipationNoise
^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.analog_noise.DissipationNoise` represents a general dissipation noise model that can be applied 
during the time evolution of an analog schedule.
This noise model is characterized by a set of jump operators that define the dissipation processes affecting the qubits, as per the 
Lindblad master equation:

.. math::

    \frac{d\rho}{dt} = -i[H, \rho] + \sum_i \left( J_i \rho J_i^\dagger - \frac{1}{2} \{ J_i^\dagger J_i, \rho \} \right)

where :math:`J_i` are the jump operators representing different dissipation channels.
The class is initialized with a list of jump operators given as QTensors:

.. code-block:: python

    from qilisdk.noise_models import DissipationNoise
    from qilisdk.core import QTensor
    import numpy as np
    J1 = QTensor(np.array([[0, 1], [0, 0]]))
    J2 = QTensor(np.array([[1, 0], [0, 0]]))
    dissipation_noise = DissipationNoise(jump_operators=[J1, J2])

AnalogDephasingNoise
^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.analog_noise.AnalogDephasingNoise` represents a dephasing noise model that can be applied
during the time evolution of an analog schedule.
This noise model is characterized by a dephasing rate that defines the strength of the dephasing process affecting the qubits. 
This class extends :class:`~qilisdk.noise_models.analog_noise.DissipationNoise`, using
the jump operator:

.. math::

    J = \sqrt{\gamma} Z

where :math:`\gamma` is the dephasing rate.
Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import AnalogDephasingNoise
    dephasing_noise = AnalogDephasingNoise(dephasing_rate=0.1)

AnalogDepolarizingNoise
^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.analog.analog_depolarizing_noise.AnalogDepolarizingNoise` represents a depolarizing noise model that can be applied
during the time evolution of an analog schedule.
This noise model is characterized by a depolarizing rate that defines the strength of the depolarizing process affecting the qubits. 
This class extends :class:`~qilisdk.noise_models.analog_noise.DissipationNoise`, using
the jump operators:

.. math::

    J_1 = \sqrt{\frac{\gamma}{3}} X, \quad J_2 = \sqrt{\frac{\gamma}{3}} Y, \quad J_3 = \sqrt{\frac{\gamma}{3}} Z

where :math:`\gamma` is the depolarizing rate.
Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import AnalogDepolarizingNoise
    depolarizing_noise = AnalogDepolarizingNoise(depolarizing_rate=0.1)

AnalogAmplitudeDampingNoise
^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.analog_noise.AnalogAmplitudeDampingNoise` represents an amplitude damping noise model that can be applied
during the time evolution of an analog schedule.
This noise model is characterized by a damping rate that defines the strength of the amplitude damping process affecting the qubits. 
This class extends :class:`~qilisdk.noise_models.analog_noise.DissipationNoise`, using
the jump operator:

.. math::

    J = \sqrt{\gamma} \sigma_-

where :math:`\gamma` is the damping rate and :math:`\sigma_-` is the lowering operator.
Usage is as follows:

.. code-block:: python

    from qilisdk.noise_models import AnalogAmplitudeDampingNoise
    amplitude_damping_noise = AnalogAmplitudeDampingNoise(damping_rate=0.1)

Parameter Noise Types
----------------------

Parameter noise types model imperfections in parameterized gates or schedules by introducing random fluctuations to the parameters.

ParameterNoise
^^^^^^^^^^^^^^
:class:`~qilisdk.noise_models.common_noise.ParameterNoise` is a class for adding random fluctuations to the parameters of gates or schedules.
The noise is characterized by a standard deviation that defines the magnitude of the fluctuations, such that each affected parameter 
:math:`\theta` is modified as:

.. math::

    \theta' = \mathcal{N}(\theta, \sigma)

where :math:`\mathcal{N}(\theta, \sigma)` denotes a normal distribution centered at :math:`\theta` with standard deviation :math:`\sigma`.
The class is initialized with a standard deviation value, along with the list of parameter names to apply the noise to:

.. code-block:: python

    from qilisdk.noise_models import ParameterNoise
    from qilisdk.core import Parameter
    theta = Parameter("theta", 2.0)
    param_noise = ParameterNoise(stddev=0.1, affected_parameters=["theta"])