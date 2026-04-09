Quantum Search with Grover's Algorithm
=====================================================

In this tutorial, we will learn about Grover's algorithm, which is a quantum algorithm 
for searching an unsorted database with quadratic speedup compared to classical algorithms.

The Problem
----------------------

Say you have a list of N items, and you want to find a specific item that satisfies a certain condition.
For example, you might have a list of phone numbers and want to find the one that belongs to a specific person.

This is the problem of searching an unsorted database. Classicially this would mean you have some
sort of array and you want to interate through it. The quantum version of this problem is to 
find a specific state in a superposition of states that satisfies a certain condition.

For instance, we might have the state:

:math:`|\psi\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle`

and we want to "find" the state :math:`|x_0\rangle` such that :math:`f(x_0) = 1` for some function :math:`f`.
We therefore want to do some operation to increase the likelyhood of us measuring :math:`|x_0\rangle` when 
we measure our state.

The Solution
----------------------

Classically, you would have to check each item one by one, which would take O(N) time in the worst case.
Grover's algorithm allows you to find the desired item in O(sqrt(N)) time, which is a significant improvement for large N.
It was invented by Lov Grover in 1996 and is one of the most well-known quantum algorithms.

The algorithm is formed of two parts: the **oracle** and the **diffusion operator**. 

The **oracle** is a quantum operation that marks the desired state by flipping its phase, in this
case corresponding to the function :math:`f` that we want to evaluate. It can be represented as a unitary operator :math:`O` such that:

:math:`O |x\rangle = (-1)^{f(x)} |x\rangle`

The **diffusion operator** is a quantum operation that amplifies the amplitude of the marked state, 
increasing the probability of measuring it. It can be represented as a unitary operator :math:`D` such that:

:math:`D = 2|\psi\rangle\langle\psi| - I`

where :math:`|\psi\rangle` is the equal superposition state and :math:`I` is the identity operator.

By doing :math:`O` followed by :math:`D` repeatedly, we can amplify the amplitude of the marked state 
and increase the probability of measuring it. More specifically, after :math:`\frac{\pi}{4} \sqrt{N}` iterations,
the probability of measuring the marked state is close to 1.

The Implementation
----------------------

So what do these operators actually look like? Well, the oracle depends on the specific problem you are trying to solve, but
for cases like searching for a specific bitstring it can be implemented with X and controlled Z gates. For instance, if we 
want to search for the bitstring "01" in a 2-qubit system, we can implement the oracle as follows:

.. code-block:: python

    from qilisdk.digital import Circuit, CZ, X

    oracle = Circuit(2)
    oracle.add(X(0))
    oracle.add(CZ(0, 1))
    oracle.add(X(0))

This flips the bit of the first qubit (so now our target state is "11"), applies a controlled Z gate that flips the phase of the "11" state, and then flips the bit of the first qubit back to its original state.

The diffusion operator is a little more complicated, but it can be implemented as follows by doing a Hadamard and then X on all qubits, 
a controlled Z gate on all qubits, and then undoing the previous operations:

.. code-block:: python

    from qilisdk.digital import Circuit, H, X, CZ

    diffusion = Circuit(2)
    diffusion.add(H(0))
    diffusion.add(H(1))
    diffusion.add(X(0))
    diffusion.add(X(1))
    diffusion.add(CZ(0, 1))
    diffusion.add(X(0))
    diffusion.add(X(1))
    diffusion.add(H(0))
    diffusion.add(H(1))

Our initial state is going to be the equal superposition state, which can be prepared by applying a Hadamard gate to each qubit:

.. code-block:: python

    from qilisdk.digital import Circuit, H

    initial_state = Circuit(2)
    initial_state.add(H(0))
    initial_state.add(H(1))

Now we can combine these operations to implement Grover's algorithm. 
We only need one repeat because we have 4 states and :math:`\text{floor}(\frac{\pi}{4} \sqrt{4}) = 1`.

.. code-block:: python

    from qilisdk.digital import Circuit

    grover = initial_state
    grover += oracle
    grover += diffusion

Simulating the result using QiliSim:

.. code-block:: python

    from qilisdk.backends import QiliSim
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    qilisim_backend = QiliSim()
    result = qilisim_backend.execute(DigitalPropagation(grover), Readout().with_sampling(nshots=100))
    print(result.get_samples())

This gives us:

.. code-block:: none

    {'01': 100}
    
This is the expected result since we were searching for the state "01".

Further Reading
--------------------

- `Grover's Algorithm`_

.. _Grover's Algorithm: https://en.wikipedia.org/wiki/Grover%27s_algorithm

