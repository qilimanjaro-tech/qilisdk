Factoring With Shor's Algorithm
=====================================================

In this tutorial, we will learn about Shor's algorithm, which is a quantum algorithm 
for factoring large integers efficiently, providing an exponential speedup compared to classical algorithms.

The Problem
----------------------

The problem of integer factorization is to find the prime factors of a given composite number.
For example, if we have the number 15, its prime factors are 3 and 5.
This problem is of great importance in cryptography, as many encryption schemes rely on the difficulty of factoring large integers.

Classically, the best-known algorithms for factoring large integers run in sub-exponential time, which
makes it infeasible to factor large numbers in a reasonable time frame.

The Solution
----------------------

Shor's algorithm, developed by Peter Shor in 1994, provides a polynomial-time solution to 
the integer factorization problem using quantum computing.
The algorithm consists of two main parts: a classical part and a quantum part. The classical part involves
reducing the factorization problem to a problem of finding the period of a certain function, 
while the quantum part uses quantum Fourier transform to find this period efficiently.

The algorithm can be summarized as follows, when trying to factor the number :math:`N`:

1. Choose a random integer :math:`a` such that :math:`1 < a < N`
2. Compute the greatest common divisor (GCD) of :math:`a` and :math:`N`. If the GCD is greater than 1, then we have found a non-trivial factor of :math:`N`.
3. If the GCD is 1, then we proceed to find the period :math:`r` of the function :math:`f(x) = a^x \mod N` using the quantum part of the algorithm.
4. If :math:`r` is even and the GCD of :math:`a^{r/2} - 1` and :math:`N` is greater than 1, then we have found a non-trivial factor of :math:`N`.
5. Otherwise, we repeat the process with a different random integer :math:`a`.

The Implementation
----------------------

As an example, let's consider the number 3. Whilst 15 is the smallest composite number, the quantum circuit for factoring it is quite large. 
Instead, we will factor 3 into 3 and 1 to demonstrate the algorithm.

So, going through the steps of the algorithm:

1. We choose a random integer :math:`a` such that :math:`1 < a < 3`. The only choice is :math:`a = 2`.
2. We compute the GCD of :math:`2` and :math:`3`, which is 1, so we proceed to the next step.
3. We need to find the period :math:`r` of the function :math:`f(x) = 2^x \mod 3`. To do this, we use the following quantum circuit:

.. code-block:: python

    from qilisdk.digital import Circuit, X, H, Controlled, CNOT, SWAP, T, Adjoint, M
    from qilisdk.backends import QiliSim, DigitalMethod
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.readout import Readout

    # Initialize the circuit with 4 qubits (2 for the input and 2 for the output)
    shors = Circuit(4)
    shors.add(X(3))
    shors.add(H(0))
    shors.add(H(1))

    # Modulo exponentiation
    shors.add(Controlled(1, basic_gate=SWAP(2, 3)))

    # Inverse Quantum Fourier Transform
    shors.add(H(0))
    shors.add(Adjoint(T(0)))
    shors.add(CNOT(1, 0))
    shors.add(T(0))
    shors.add(CNOT(1, 0))
    shors.add(Adjoint(T(1)))
    shors.add(H(1))
    shors.add(SWAP(0, 1))

    # We measure the first two qubits to find the period r
    shors.add(M(0))
    shors.add(M(1))

    # Run the simulation
    backend = QiliSim(digital_simulation_method=DigitalMethod.statevector(matrix_free=False))
    res = backend.execute(DigitalPropagation(shors), Readout().with_sampling(1000))
    print(res)

This will output a 50/50 mix of 0 and 2, which corresponds to the period :math:`r = 2`.

4. Since :math:`r` is even, we compute the GCD of :math:`a^{r/2} - 1 = 2^1 - 1 = 1` and :math:`3`, which is 1, so we do not find a non-trivial factor.
5. We would then repeat the process with a different random integer :math:`a`, but since there are no other choices, we conclude that the factors of 3 are 3 and 1.

Further Reading
--------------------

- `Shor's Algorithm`_

.. _Shor's Algorithm: https://en.wikipedia.org/wiki/Shor%27s_algorithm