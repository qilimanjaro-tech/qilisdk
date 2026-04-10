Optimization with QAOA
=======================================

In this tutorial, we will explore how to use the Quantum Approximate Optimization Algorithm (QAOA) to solve a simple
optimization problem using QiliSDK.

The Problem
----------------------

The optimization problem we are going to solve is the Knapsack Problem, which is a combinatorial optimization problem. 
It is known to be NP-complete, meaning that the vast majority of problems can be converted into this form without much overhead.

The goal is to maximize the total value of items that can be put in a knapsack with a weight limit.
So, imagine you have a backpack (i.e. a knapsack) that you know can only carry 4 kg, 
and you have the following things you want to bring back from your trip to Barcelona:

+----+-----------------------+-------------+------------+
| ID | Item                  | Weight (g)  | Importance |
+----+-----------------------+-------------+------------+
| 0  | Flower Tile           | 2000        | 4          |
| 1  | "I Heart ..." Shirt   | 150         | 5          |
| 2  | Bottle of Sangria     | 750         | 3          |
| 3  | Bottle of Ratafia     | 750         | 4          |
| 4  | Fuet                  | 200         | 1          |
| 5  | Jamon Iberico         | 500         | 1          |
| 6  | Case of Estrella Damm | 1980        | 2          |
+----+-----------------------+-------------+------------+

The task is then to choose whether to bring back each item, such that the total importance is maximized, 
while the total weight does not exceed 4 kg.


The Solution
----------------------

The Implementation
----------------------

Further Reading
--------------------

- `Knapsack Problem`_
- `QUBO`_
- `QAOA`_

.. _Knapsack Problem: https://en.wikipedia.org/wiki/Knapsack_problem
.. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization
.. _QAOA: https://en.wikipedia.org/wiki/Quantum_approximate_optimization_algorithm
