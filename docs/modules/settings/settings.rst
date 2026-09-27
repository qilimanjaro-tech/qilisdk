Settings
========

The :mod:`~qilisdk.settings` module can be used to configure some of the values that configure QiliSDK as a whole.
They live in the
:class:`~qilisdk.settings.QiliSDKSettings` class, which can be set via code, read from environment variables or read from
a local ``.env`` file.

Reading the settings
--------------------

:func:`~qilisdk.settings.get_settings` returns the settings object currently in use:

.. code-block:: python

    from qilisdk.settings import get_settings

    settings = get_settings()
    print(settings.complex_precision)
    print(settings.atol)
    print(settings.rtol)

Changing the settings
---------------------

Every setting is read from an environment variable named after it, prefixed with ``QILISDK_``:

.. code-block:: bash

    export QILISDK_COMPLEX_PRECISION=COMPLEX_64
    export QILISDK_ATOL=1e-8
    python my_experiment.py

The same values can be kept in a ``.env`` file in the directory the script is run from, which is
convenient for credentials that should not end up in the shell history:

.. code-block:: bash

    QILISDK_SPEQTRUM_USERNAME=alice
    QILISDK_SPEQTRUM_APIKEY=my-secret-key
    QILISDK_ATOL=1e-8

Entries without the ``QILISDK_`` prefix are ignored, so the same file can be shared with other tools.
If a variable is set both in the environment and in the ``.env`` file, the environment wins.

Settings can also be changed by directly assigning to the fields of the settings object:

.. SKIP
.. code-block:: python

    from qilisdk.settings import Precision, get_settings

    get_settings().complex_precision = Precision.COMPLEX_64

Full list of settings
-----------------------

The full list of settings, with the environment variable each one is read from:

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24

   * - Setting
     - Environment variable
     - Default
     - Meaning
   * - ``complex_precision``
     - ``QILISDK_COMPLEX_PRECISION``
     - ``COMPLEX_128``
     - Precision of the complex numbers used for gates, Hamiltonians and states.
   * - ``atol``
     - ``QILISDK_ATOL``
     - ``1e-10``
     - Absolute tolerance below which a value is treated as zero.
   * - ``rtol``
     - ``QILISDK_RTOL``
     - ``1e-5``
     - Relative counterpart of ``atol``.
   * - ``logging_config_path``
     - ``QILISDK_LOGGING_CONFIG_PATH``
     - the packaged ``logging_config.yaml``
     - YAML file describing the logging sinks.
   * - ``speqtrum_username``
     - ``QILISDK_SPEQTRUM_USERNAME``
     - ``None``
     - SpeQtrum username used for authentication.
   * - ``speqtrum_apikey``
     - ``QILISDK_SPEQTRUM_APIKEY``
     - ``None``
     - SpeQtrum API key associated with the account.
   * - ``speqtrum_api_url``
     - ``QILISDK_SPEQTRUM_API_URL``
     - ``https://qilimanjaro.ddns.net/public-api/api/v1``
     - Base URL of the SpeQtrum API.
   * - ``speqtrum_audience``
     - ``QILISDK_SPEQTRUM_AUDIENCE``
     - ``urn:qilimanjaro.tech:public-api:beren``
     - Audience claim expected in the authentication token.