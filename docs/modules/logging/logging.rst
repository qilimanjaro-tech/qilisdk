Logging
=======

QiliSDK reports what it is doing through `Loguru <https://loguru.readthedocs.io>`_. Logging is configured automatically
when the package is imported, and can be adjusted at any time via
:func:`~qilisdk.logging.configure_logging`.

By default only messages of level ``WARNING`` and above are shown, and they are written to the console (``stderr``).

Changing the verbosity
----------------------

Call :func:`~qilisdk.logging.configure_logging` with a ``level`` to override the verbosity:

.. code-block:: python

    from qilisdk import configure_logging

    configure_logging(level="DEBUG")

All other options of the configured sinks (format, filter, colours, ...) are preserved, so only the threshold changes.
Calling it again replaces the previous configuration, so the last call wins.

Logging to a file
-----------------

Pass a ``filename`` to copy the console output into a text file:

.. SKIP
.. code-block:: python

    from qilisdk import configure_logging

    # Console and file
    configure_logging(level="TRACE", filename="run.log")

    # File only, no console output
    configure_logging(level="TRACE", filename="run.log", stderr=False)

Logging levels
--------------

QiliSDK uses the standard Loguru levels, each with its own icon:

.. list-table::
   :header-rows: 1
   :widths: 18 10 72

   * - Level
     - Icon
     - Meaning
   * - ``TRACE``
     - 🔬
     - Very detailed information, typically only useful when written to a file.
   * - ``DEBUG``
     - 🐞
     - Information meant for developers, designed to not flood the terminal during a normal workflow.
   * - ``INFO``
     - 💡
     - Information meant for users, designed to not flood the terminal during a normal workflow.
   * - ``SUCCESS``
     - ✅
     - Messages indicating a successful operation.
   * - ``WARNING``
     - 🚧
     - Things worth noting, such as a parameter being ignored. This is the default threshold.
   * - ``ERROR``
     - ❌
     - Errors. Rarely used, since QiliSDK raises exceptions instead.
   * - ``CRITICAL``
     - 💀
     - Critical errors. Rarely used, for the same reason.

Customising the configuration
-----------------------------

The sinks themselves are described in a YAML file. The packaged one lives next to the source of the
:mod:`~qilisdk.logging` module, and looks like this:

.. code-block:: yaml

    sinks:
      - sink: stderr
        level: WARNING
        format: "QiliSDK | <green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <lvl>{level: <8}</> | <lvl>{level.icon} {message}</>"
        filter: qilisdk
        colorize: true

    intercept_libraries:
      - name: httpx
        level: ERROR

To use your own file, point the ``QILISDK_LOGGING_CONFIG_PATH`` setting at it before importing QiliSDK.