Overview
=========

The :mod:`~qilisdk.backends` module provides concrete execution engines for running :mod:`~qilisdk.functionals` (quantum processes).  
Currently, three backends are supported:

- :doc:`QiliSim <backends_qilisim>`: A high-performance CPU simulator developed by Qilimanjaro, ideal for local development and testing.
- :doc:`CUDA <backends_cuda>`: A GPU-accelerated backend leveraging NVIDIA hardware for large-scale simulations.
- :doc:`Qutip <backends_qutip>`: A CPU-based backend using the QuTiP library, suitable for lightweight simulations and environments without GPU access.

.. NOTE::

    Backends other than QiliSim are optional; to install one, include its extra when installing QILISDK:

    .. code-block:: console

        pip install qilisdk[<backend_name>]

    For more information check the :doc:`../../getting_started/installation` page.

Once installed, any primitive functional can be executed by passing it to the backend's :meth:`~qilisdk.backends.backend.Backend.execute` method
along with readout specifications:

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout
    from qilisdk.digital import Circuit

    circuit = Circuit(2)
    results = QiliSim().execute(DigitalPropagation(circuit), Readout().with_sampling(nshots=1000))
    print(results)

Architecture Overview
---------------------

All concrete backends subclass :class:`~qilisdk.backends.backend.Backend`, which centralizes the execution workflow used
across the SDK. The :meth:`~qilisdk.backends.backend.Backend.execute` method dispatches a primitive functional (e.g. :class:`~qilisdk.functionals.digital_propagation.DigitalPropagation` or :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`)
to the appropriate simulation routine and returns a :class:`~qilisdk.functionals.functional_result.FunctionalResult` (see the :doc:`Functionals
</modules/functionals/functionals>` chapter). The execute method also accepts readout specifications that define how the quantum state is measured.
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
