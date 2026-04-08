Backends
========

The :mod:`~qilisdk.backends` module provides concrete execution engines for running :mod:`~qilisdk.functionals` (quantum processes).
Currently, three backends are supported:

- :class:`~qilisdk.backends.qilisim.QiliSim`
- :class:`~qilisdk.backends.cuda_backend.CudaBackend`
- :class:`~qilisdk.backends.qutip_backend.QutipBackend`

.. NOTE::

    Backends other than QiliSim are optional; to install one, include its extra when installing QILISDK:

    .. code-block:: console

        pip install qilisdk[<backend_name>]

    For more information check the :doc:`../getting_started/installation` page.

Once installed, any primitive functional can be executed by passing it to the backend's :meth:`~qilisdk.backends.backend.Backend.execute` method
along with readout specifications:

.. code-block:: python

    from qilisdk.backends import CudaBackend
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # ... build your circuit and DigitalPropagation functional ...
    results = CudaBackend().execute(propagation, Readout().with_sampling(nshots=1000))
    print(results)

Architecture Overview
---------------------

All concrete backends subclass :class:`~qilisdk.backends.backend.Backend`, which centralizes the execution workflow used
across the SDK. The :meth:`~qilisdk.backends.backend.Backend.execute` method dispatches a primitive functional (e.g. :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` or :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`)
to the appropriate simulation routine and returns a :class:`~qilisdk.functionals.functional_result.FunctionalResult` (see the :doc:`Functionals
<functionals>` chapter). The execute method also accepts readout specifications that define how the quantum state is measured.
The execute method is also used to optimize variational programs via repeated calls to
the underlying parameterized primitive functional.

Backends register handlers for the functionals they support. If a functional is not implemented, :meth:`~qilisdk.backends.backend.Backend.execute` raises
``NotImplementedError`` to surface the mismatch early.

Hardware & Dependencies
-----------------------

Installing a backend extra pulls in the library stack required for that simulator. GPU backends also expect compatible
drivers to be present on the system.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Backend
     - Extra
     - Key dependency
     - Notes
   * - :class:`QiliSim <qilisdk.backends.qilisim.QiliSim>`
     - ``qilisim``
     - None
     - CPU based; no special hardware needs.
   * - :class:`CudaBackend <qilisdk.backends.cuda_backend.CudaBackend>`
     - ``cuda``
     - `cuda-quantum <https://github.com/NVIDIA/cuda-quantum>`_
     - Requires NVIDIA hardware with recent drivers.
   * - :class:`QutipBackend <qilisdk.backends.qutip_backend.QutipBackend>`
     - ``qutip``
     - `QuTiP <https://qutip.org/>`_
     - CPU based; no special hardware needs.

Functional Support
------------------

The table below summarizes which primitive :mod:`~qilisdk.functionals` each backend can execute.

.. list-table::
   :header-rows: 1
   :widths: 25 18 18 18 21

   * - Backend
     - DigitalPropagation
     - AnalogEvolution
     - QuantumReservoir
     - VariationalProgram
   * - :class:`QiliSim <qilisdk.backends.qilisim.QiliSim>`
     - |y|
     - |y|
     - |y|
     - |y|
   * - :class:`CudaBackend <qilisdk.backends.cuda_backend.CudaBackend>`
     - |y|
     - |y|
     - |y|
     - |y|
   * - :class:`QutipBackend <qilisdk.backends.qutip_backend.QutipBackend>`
     - |y|
     - |y|
     - |y|
     - |y|

.. |y| unicode:: U+2713

QiliSim Backend
----------------

The **QiliSim** backend is a CPU-based simulator developed by Qilimanjaro and written in C++, providing
efficient simulation of both digital and analog quantum functionals.
It is designed for ease of use and does not require any special hardware or dependencies.
There is no need to install QiliSim separately, as it is included with the core QILISDK installation.

**Initialization**

.. code-block:: python

    from qilisdk.backends import QiliSim

    backend = QiliSim()

**Capabilities**

- **DigitalPropagation** of digital circuits with efficient state-vector simulation.
- **AnalogEvolution** driven by :class:`~qilisdk.analog.schedule.Schedule`.
- Compatible with :class:`~qilisdk.functionals.variational_program.VariationalProgram` for classical optimization loops.

**Parameters**

- ``noise_model`` (:class:`~qilisdk.noise.noise_model.NoiseModel`, optional): Noise model applied during simulation.
- ``analog_simulation_method`` (:class:`~qilisdk.backends.backend_config.AnalogMethod`, optional): Analog simulation method and method-specific options.
- ``digital_simulation_method`` (:class:`~qilisdk.backends.backend_config.DigitalMethod`, optional): Digital simulation options.
- ``execution_config`` (:class:`~qilisdk.backends.backend_config.ExecutionConfig`, optional): Runtime execution options such as thread count and random seed.

**Configuration example**

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.backends.backend_config import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.arnoldi(
            dim=16,
            num_substeps=2
        ),
        digital_simulation_method=DigitalMethod.statevector(max_cache_size=2_000),
        execution_config=ExecutionConfig(num_threads=4, seed=42, monte_carlo=MonteCarloConfig(trajectories=200)),
    )

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, CNOT
    from qilisdk.backends import QiliSim
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # Build a simple circuit
    circuit = Circuit(5)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    # Create DigitalPropagation functional
    functional = DigitalPropagation(circuit)

    # Execute with the QiliSim backend
    qilisim_backend = QiliSim()
    result = qilisim_backend.execute(functional, Readout().with_sampling(nshots=500))
    print(result.samples)

CUDA Backend
------------

The **CUDA** backend leverages NVIDIA GPUs via the :mod:`cuda-quantum` framework for both digital and analog simulations.
When no compatible GPU is detected it automatically falls back to cpu-based targets, so you can prototype on
commodity hardware before moving to accelerated machines.

**Installation**

.. code-block:: console

    pip install qilisdk[cuda]

**Initialization**

.. code-block:: python

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(
        sampling_method=CudaSamplingMethod.STATE_VECTOR
    )

**Capabilities**

- Digital circuits through :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation`.
- Analog dynamics for :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`, powered by ``cudaq.evolve``.
- Hybrid execution when paired with :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

**Sampling methods**

- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.STATE_VECTOR`: Full state-vector simulation (switches to CPU if a GPU is unavailable).
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.TENSOR_NETWORK`: Tensor-network contraction, suited for shallow yet wide circuits.
- :class:`~qilisdk.backends.cuda_backend.CudaSamplingMethod.MATRIX_PRODUCT_STATE`: Matrix-product-state simulation for low-entanglement workloads.

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.digital import Circuit, H, RX, CNOT
    from qilisdk.backends import CudaBackend, CudaSamplingMethod
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # Build a simple circuit
    circuit = Circuit(2)
    circuit.add(RX(0, theta=np.pi / 4))
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))

    # Create DigitalPropagation functional
    functional = DigitalPropagation(circuit)

    # Execute with the chosen sampling method (GPU if available)
    cuda_backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    result = cuda_backend.execute(functional, Readout().with_sampling(nshots=500))
    print(result.samples)

**Output**

::

    {'11': 237, '00': 263}


Qutip Backend
-------------

The **Qutip** backend uses the :mod:`qutip` library for simulation on CPU, supporting both digital and analog functionals.
It is the most lightweight option, ideal for local development or environments without NVIDIA GPUs.

**Installation**

.. code-block:: console

    pip install qilisdk[qutip]

**Initialization**

.. code-block:: python

    from qilisdk.backends import QutipBackend

    backend = QutipBackend()

**Capabilities**

- **DigitalPropagation** of digital circuits via QuTiP's state-vector solvers.
- **AnalogEvolution** driven by :class:`~qilisdk.analog.schedule.Schedule`.
- Compatible with :class:`~qilisdk.functionals.variational_program.VariationalProgram` for fully classical
  optimization loops.

**Example**

.. code-block:: python

    import numpy as np
    from qilisdk.analog import Schedule, X, Z, Y
    from qilisdk.core import ket, tensor_prod
    from qilisdk.backends import QutipBackend
    from qilisdk.core.interpolator import Interpolation
    from qilisdk.functionals import AnalogEvolution
    from qilisdk.readout import Readout

    # Define total time and timestep
    T = 10.0
    dt = 0.5
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits))

    # Build a time-dependent schedule
    schedule = Schedule(
        hamiltonians={"driver": Hx, "problem": Hz},
        coefficients={
            "driver": {(0.0, T): lambda t: 1 - t / T},
            "problem": {(0.0, T): lambda t: t / T},
        },
        dt=dt,
        interpolation=Interpolation.LINEAR,
    )

    # Prepare an equal superposition initial state
    initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

    # Create the AnalogEvolution functional
    analog_evolution = AnalogEvolution(
        schedule=schedule,
        initial_state=initial_state,
    )

    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(
        analog_evolution,
        Readout().with_expectation(observables=[Z(0), X(0), Y(0)]).with_state_tomography(),
    )
    print(results)

**Output**

::

    FunctionalResult(
        expectation_values=array([-0.99388223,  0.0467696 , -0.10005353]),
        final_state=QTensor(shape=2x1, nnz=2, format='csr')
        [[0.05506547-0.00516502j]
        [0.3364973 -0.94005887j]]
    )
