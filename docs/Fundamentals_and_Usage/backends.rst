Backends
======================

:mod:`~qilisdk.backends` are the tools used to simulate the :mod:`~qilisdk.functionals` (quantum processes). Currently there are only two supported backends:

- :mod:`~qilisdk.backends.cuda_backend`
- :mod:`~qilisdk.backends.qutip_backend`


backends are optional and in order to install a backend you need to run: 

::

    pip install qilisdk[backend_name]

Moreover, all :mod:`~qilisdk.functionals` can be executed on a backend using the :meth:`~qilisdk.backends.backend.execute` method: 

.. code-block:: python

    backend.execute(functional) 


CUDA Backend
-------------

This backend can be installed using: 

::

    pip install qilisdk[cuda]


This backend is built on top of :mod:`cudaq` and uses the modules from cudaq to simulate the functionals defined. 
A :class:`~qilisdk.backends.cuda_backend.CudaBackend` is simply defined like this: 

.. code-block:: python 

    from qilisdk.backends import CudaBackend, CudaSamplingMethod

    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)

where the ``sampling_method`` could be anything between: 

- STATE_VECTOR: used only for digital quantum computing simulations (:class:`~qilisdk.functionals.sampling.Sampling`)
- TENSOR_NETWORK: used only for digital quantum computing simulations (:class:`~qilisdk.functionals.sampling.Sampling`)
- MATRIX_PRODUCT_STATE: used for both digital and analog quantum computing simulation (:class:`~qilisdk.functionals.sampling.Sampling` and :class:`~qilisdk.functionals.time_evolution.TimeEvolution`)


Qutip Backend
-------------

This backend is built on top of :mod:`qutip` and uses the modules from qutip to simulate the functionals defined. 
This backend can be installed using: 

::

    pip install qilisdk[qutip]

A :class:`~qilisdk.backends.qutip_backend.QutipBackend` is simply defined like this: 

.. code-block:: python 

    from qilisdk.backends import QutipBackend

    backend = QutipBackend()

the Qutip backend doesn't offer multiple method of simulation, and it works for both digital and analog simulations. 