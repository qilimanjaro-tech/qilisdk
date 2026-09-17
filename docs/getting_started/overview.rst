Overview
===============

Welcome to **QiliSDK**, a unified Python framework for developing 
and running a variety of quantum algorithms (digital, analog and hybrid) across a variety 
of backends (CPU, GPU, and QPU).

QiliSDK is split into several modules, which is easiest to 
visualize via the following diagram:

.. image:: ../../_static/the_diagram.png
   :align: center

First, users define the things from the top row: their Circuit and Gates (for digital jobs)
or their Hamiltonian and Schedule (for analog jobs). Both of these can be combined with the 
things from the Core module, like QTensors, Variables and Expressions to create 
Parameterized Circuits and Hamiltonians. A Readout should also be specified: whether the
user wants samples, expectations values, or even a full statevector.

Then, the user defines the job using one of our Functionals:
 - DigitalPropagation for circuit simulation
 - AnalogEvolution for analog simulation
 - QuantumReservoirs for quantum reservoir computing
 - VariationalProgram, which can be combined with any of the above to create a variational workflow

 Finally, the user chooses the Backend to run the job on, which can be 
 a CPU (QiliSim), GPU (QiliSim or CudaqBackend), or QPU (SpeQtrum). 

 To get started, first check out the :doc:`getting_started/installation` guide to install QiliSDK, 
 then the :doc:`getting_started/quickstart` guide to jump straight in to some examples.