QiliSDK
=========================

.. rst-class:: lead center

Welcome to **QiliSDK**, a unified Python framework for developing, simulating, and running both digital and analog quantum algorithms across a variety of backends (QPU, CPU, GPU) and Qilimanjaro's SpeQtrum. Its modular design makes it easy to prototype circuits, build Hamiltonians, design variational workflows, and deploy them on local or remote quantum simulators and hardware.

.. grid:: 2

   .. grid-item-card:: Getting Started
      :link: getting_started/installation.html
      :text-align: center
      :img-top: _static/rocket.png

      New to QiliSDK? Here you will find a description of its main concepts, together
      with some tutorials on how to install and start using QiliSDK!

   .. grid-item-card:: Modules
      :link: modules/analog.html
      :text-align: center
      :img-top: _static/book.png

      This section contains in-depth information about each key module of QiliSDK.

   .. grid-item-card:: Tutorials
      :link: tutorials/intro_circuits.html
      :text-align: center
      :img-top: _static/book.png

      This section contains a series of tutorials that cover the basics of quantum computing and how to use QiliSDK to implement various quantum algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/introduction
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/intro_circuits
   tutorials/intro_annealing
   tutorials/application_grovers
   tutorials/application_annealing
   tutorials/application_qaoa

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :hidden:

   modules/analog
   modules/digital
   modules/core
   modules/functionals
   modules/cost_functions
   modules/backends
   modules/noise_models
   modules/speqtrum

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:
