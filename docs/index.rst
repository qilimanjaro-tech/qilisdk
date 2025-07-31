QiliSDK
=========================

.. rst-class:: lead center

Welcome to **QiliSDK**, a unified Python framework for developing, simulating, and running both digital and analog quantum algorithms across a variety of backends (CPU, GPU, and Qilimanjaro's QaaS). Its modular design makes it easy to prototype circuits, build Hamiltonians, design variational workflows, and deploy them on local or remote quantum simulators and hardware.

.. grid:: 2

   .. grid-item-card:: Getting Started
      :link: getting_started/installation.html
      :text-align: center
      :img-top: _static/rocket.png

      New to QiliSDK? Here you will find a description of its main concepts, together
      with some tutorials on how to install and start using QiliSDK!

   .. grid-item-card:: Fundamentals and Usage
      :link: fundamentals/analog.html
      :text-align: center
      :img-top: _static/book.png

      This section contains in-depth information about the key concepts of QiliSDK.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/introduction
   getting_started/installation
   


.. toctree::
   :maxdepth: 2
   :caption: Fundamentals

   fundamentals/analog
   fundamentals/digital
   fundamentals/common
   fundamentals/functionals
   fundamentals/backends
   fundamentals/qaas



.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/circuits.ipynb
   examples/hamiltonians.ipynb
   examples/time_evolution.ipynb
   examples/quantum_objects.ipynb
   examples/models.ipynb

.. toctree::
   :maxdepth: 4
   :caption: API Reference
