Installation
============

QiliSDK and its optional extras are distributed via PyPI. Install the core package first, then add any backend or feature modules you need.

**Base package**

.. code-block:: bash

    pip install qilisdk

**Optional extras**

- **CUDA acceleration** (NVIDIA GPU support with :mod:`~qilisdk.backends.cuda_backend.CudaBackend`):

  .. code-block:: bash

      pip install qilisdk[cuda]

.. NOTE::
    The CUDA backend requires a CUDA accelerated GPU to be available on your system and proper drivers to be installed. 
    More information can be found on their official website: https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q

- **Qutip CPU backend** (CPU simulation with :mod:`~qilisdk.backends.qutip_backend.QutipBackend`):

  .. code-block:: bash

      pip install qilisdk[qutip]

- **SpeQtrum** (cloud submission via :class:`~qilisdk.speqtrum`):

  .. code-block:: bash

      pip install qilisdk[speqtrum]

You can combine extras:

.. code-block:: bash

    pip install qilisdk[cuda,qutip,speqtrum]



.. NOTE::

    QiliSDK requires a python version 3.10 or higher. and has a dependency on the following libraries: 
    :: 

        "dill>=0.3.9"
        "numpy>=1.26.4"
        "ruamel-yaml>=0.18.16"
        "scipy>=1.15.3"
        "pydantic>=2.12.3"
        "pydantic-settings>=2.11.0"
        "loguru>=0.7.3"
        "matplotlib>=3.10.7"