Overview
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
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout
    from qilisdk.backends import CudaBackend

    # Define the random circuit and functional
    c = Circuit(nqubits=2)
    c.add(X(0))
    c.add(X(1))
    functional = DigitalPropagation(c)

    # Apply bit-flip noise with 50% probability to X gates on qubit 1
    nm = NoiseModel()
    nm.add(BitFlip(probability=0.5), gate=X, qubits=[1])

    # Execute with CUDA backend
    backend_cuda = CudaBackend(noise_model=nm)
    res = backend_cuda.execute(functional, readout=Readout().with_sampling(nshots=1000))
    print(res)

Summary of Noise Types
----------------------

Some of the noise passes available in QiliSDK can be used for both digital circuits 
and analog schedules, while others are specific to one or the other:

.. table::
   :align: left
   :widths: auto

   ========================================================================== ==================== ======================
   Noise Type                                                                 Digital Propogation  Analog Evolution
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

