Noise Models
==============

The :mod:`qilisdk.noise_models` module contains classes that represent different types of noise in quantum systems.
These noise models can be integrated into quantum simulations to account for real-world imperfections and
decoherence effects. A :class:`~qilisdk.noise_models.noise_model.NoiseModel` can contains various types of noise, 
each of which is then applied during simulation.
There are three types of noise in QiliSDK: digital (used in digital circuits), analog (used in analog schedules),
and parameter noise (used to model imperfections in parameterized gates or schedules).

Note: For now, only the :class:`~qilisdk.backends.cuda_backend.CudaBackend` supports noise models.

Usage
-----------

Noise passes are added to a :class:`~qilisdk.noise_models.noise_model.NoiseModel` and then used when executing a functional. For example, to add
two types of digital noise to a circuit simulation and then simulate it using the CUDA backend:

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.noise_models import NoiseModel, DigitalBitFlipNoise, DigitalDephasingNoise
    from qilisdk.digital import X, Circuit
    from qilisdk.functionals import Sampling

    # Define the random circuit and sampler
    shots = 10000
    nqubits = 2
    c = Circuit(nqubits=nqubits)
    c.add(X(0))
    c.add(X(1))
    sampler = Sampling(c, nshots=shots)

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

KrausNoise
^^^^^^^^^^^^^^^
TODO

DigitalBitFlipNoise
^^^^^^^^^^^^^^^
TODO

DigitalDephasingNoise
^^^^^^^^^^^^^^^
TODO

DigitalDepolarizingNoise
^^^^^^^^^^^^^^^
TODO

DigitalAmplitudeDampingNoise
^^^^^^^^^^^^^^^
TODO

Analog Noise Types
----------------------

DissipationNoise
^^^^^^^^^^^^^^
TODO

AnalogDephasingNoise
^^^^^^^^^^^^^^
TODO

AnalogDephasingNoise
^^^^^^^^^^^^^^
TODO

AnalogAmplitudeDampingNoise
^^^^^^^^^^^^^^
TODO

Parameter Noise Types
----------------------

ParameterNoise
^^^^^^^^^^^^^^
TODO