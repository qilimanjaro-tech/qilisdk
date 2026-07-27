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
    
    Minimum OS requirements:
        - Linux: Ubuntu 22.04 or higher
        - Windows: Windows 11 or higher
        - MacOS: MacOS 14 or higher

Compiling From Source
^^^^^^^^^^^^^^^^^^^^^^^^^^

The above method will install the pre-built binaries of QiliSDK of the most recent release. 
If you want to get the newest features (those that we're still working on), you can compile the library from source, 
but be aware that this is not recommended for most users, and we make no promises that the latest code will be stable.

Support for Windows is limited, so we recommend using WSL, which can be installed as per 
`this guide <https://learn.microsoft.com/en-us/windows/wsl/install>`__. 
With this you should then follow the Linux instructions below.
If you must use pure Windows, the Windows instructions below should work, although they disable certain code features (notably some parallelized loops).

First, make sure you have Python, pip, git, cmake and a C++ compiler installed:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            sudo apt update
            sudo apt install python3 python3-pip git build-essential

    .. group-tab:: Mac OSX

        .. code-block:: bash

            xcode-select --install
            brew install python git ninja cmake

    .. group-tab:: Windows

         - Install Python via the Microsoft Store.

         - Install Git via https://git-scm.com/install/windows.

         - Install C++ build tools by installing the "C/C++ Extension Pack" extension for VSCode.

Install uv globally with:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            curl -LsSf https://astral.sh/uv/install.sh | sh

    .. group-tab:: Mac OSX

        .. code-block:: bash

            curl -LsSf https://astral.sh/uv/install.sh | sh

    .. group-tab:: Windows

        .. code-block:: bash

            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Then, clone (i.e. download) and enter the repository:

.. code-block:: bash

    git clone https://github.com/qilimanjaro-tech/qilisdk
    cd qilisdk

Create a new virtual environment using uv:

.. code-block:: bash

    uv venv

Then activate the environment:

.. tabs::

    .. group-tab:: Linux

        .. code-block:: bash

            source .venv/bin/activate

    .. group-tab:: Mac OSX

        .. code-block:: bash

            source .venv/bin/activate

    .. group-tab:: Windows

        .. code-block:: bash

            .venv\Scripts\activate

.. code-block:: bash

To then install QiliSDK into this new environment, run:

.. code-block:: bash

    uv sync

If you want to install with extras, you can run the following, adjusting as needed:

.. code-block:: bash

    uv sync --extra cuda13 --extra qutip --extra speqtrum
    
There are also a number of extra compile flags that can be set to enable/disable certain features:

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Flag
      - What it does
    * - ``-Ccmake.define.build_native=ON``
      - Build for the architecture of the current machine for increased performance. Only recommended if you are building for your own machine.
    * - ``-Ccmake.define.single_precision=ON``
      - Use single precision (``float``) instead of double precision (``double``). Slightly faster, but less accurate.
    * - ``-Ccmake.build-type=Debug``
      - Enable a debug build (no optimizations, debug symbols).
    * - ``-Ccmake.build-type=RelWithDebInfo``
      - Enable an optimized debug build (optimizations plus debug symbols).
    * - ``-Ccmake.build-type=Release``
      - Enable a release build (optimizations, no debug symbols). This is the default.
    * - ``-Ccmake.define.tidy=ON``
      - Run ``clang-tidy`` on the code during compilation.
    * - ``-Ccmake.define.verbose=ON``
      - Enable (very) verbose output from the C++ code.
    * - ``-Ccmake.define.tests=ON``
      - Build the C++ test suite, which will then be available as a ``test_cpp`` executable in the tests directory.
    * - ``-Ccmake.define.coverage=ON``
      - Enable coverage instrumentation flags.

These can be passed to ``uv sync`` by simply appending to the build command, for example:

.. code-block:: bash

    uv sync -Ccmake.define.build_native=ON -Ccmake.build-type=RelWithDebInfo

You then have an environment with the latest version of QiliSDK installed.
If you want to install other things to the environment you'll need to use pip with uv:

.. code-block:: bash

    uv pip install <package_name>

And then to run a Python script within the environment, you can use:

.. code-block:: bash

    uv run python3 <script_name.py>