Noise Config
----------------------

By default, when converting between different noise representations (e.g. from Kraus operators to Lindblad generators),
certain parameters are assumed, such as gate times. 
These defaults can be modified using the :class:`~qilisdk.noise.noise_config.NoiseConfig` class:

.. code-block:: python

    from qilisdk.noise import NoiseModel, NoiseConfig
    from qilisdk.digital import X

    # Create a noise configuration, setting the X gate time to 20 ns
    conf = NoiseConfig()
    conf.set_gate_time(X, 20e-9)

    # Define a simple noise model using this config
    nm = NoiseModel(noise_config=conf)

Parameters and their defaults are as follows:

.. table::
   :align: left
   :widths: auto
   
   ========================================================================== ==========================
   Parameter                                                                  Default Value                                           
   ========================================================================== ==========================
   :attr:`~qilisdk.noise.noise_config.NoiseConfig.gate_times`                 1.0 for all gates
   ========================================================================== ==========================

