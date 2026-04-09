Installation
============

QiliSDK and its optional extras are distributed via PyPI. Use pip to install the core package, plus any extra modules you need.

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

    QiliSDK requires a python version 3.11 or higher.
    
    Minimum OS requiremnets:
        - Linux: Ubuntu 22.04 or higher
        - Windows: Windows 11 or higher
        - MacOS: MacOS 14 or higher

Compiling From Source
^^^^^^^^^^^^^^^^^^^^^^^^^^

The above method will install the pre-built binaries of QiliSDK of the most recent release. 
If you want to get the newest features (those that we're still working on), you can compile the library from source, 
but be aware that this is not recommended for most users, and we make no promises that the latest code will be stable.

Support for Windows is somewhat limited, so we recommend using WSL, which can be installed as per 
`this guide <https://learn.microsoft.com/en-us/windows/wsl/install>`__.

First, make sure you have a Python, pip, git, uv and build-essentials installed. For Ubuntu/Debian, you can run:

.. code-block:: bash

    sudo apt update
    sudo apt install python3 python3-pip git build-essential
    curl -LsSf https://astral.sh/uv/install.sh | sh

Then, clone (i.e. download) and enter the repository:

.. code-block:: bash

    git clone https://github.com/qilimanjaro-tech/qilisdk
    cd qilisdk

Create a new virtual environment using uv and activate it:

.. code-block:: bash

    uv venv
    source .venv/bin/activate

To then install QiliSDK into this new environment, run:

.. code-block:: bash

    uv sync

If you want to install with extras, you can run the following, adjusting as needed:

.. code-block:: bash

    uv sync --extra cuda13 --extra qutip --extra speqtrum

You then have an environment with the latest version of QiliSDK installed.
If you want to install other things to the environment you'll need to use pip with uv:

.. code-block:: bash

    uv pip install <package_name>

And then to run a Python script within the environment, you can use:

.. code-block:: bash

    uv run python3 <script_name.py>