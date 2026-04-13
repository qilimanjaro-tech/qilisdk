The optimization problem we are going to solve is the Knapsack Problem, which is a combinatorial optimization problem. 
It is known to be "NP-complete", meaning that the a large number of problems can be converted into this form without much overhead.

The goal is to maximize the total value of items that can be put in a knapsack with a weight limit.
So, imagine you have a backpack (i.e. a knapsack) that you know can only carry 4 kg, 
and you have the following things you want to bring back from your trip to `Barcelona`_:

.. _Barcelona: https://maps.app.goo.gl/PfKbJtrDQaGTZHyVA

+----+-----------------------+-------------+------------+
| ID | Item                  | Weight (g)  | Importance |
+====+=======================+=============+============+
| 0  | Flower Tile           | 2000        | 4          |
+----+-----------------------+-------------+------------+
| 1  | "I Heart ..." Shirt   | 150         | 5          |
+----+-----------------------+-------------+------------+
| 2  | Bottle of Sangria     | 750         | 3          |
+----+-----------------------+-------------+------------+
| 3  | Bottle of Ratafia     | 750         | 4          |
+----+-----------------------+-------------+------------+
| 4  | Fuet                  | 200         | 1          |
+----+-----------------------+-------------+------------+
| 5  | Jamon Iberico         | 500         | 1          |
+----+-----------------------+-------------+------------+
| 6  | Case of Estrella Damm | 1980        | 2          |
+----+-----------------------+-------------+------------+

The task is then to choose whether to bring back each item, such that the total importance is maximized, 
while the total weight does not exceed 4 kg.

There are many different algorithms to solve this problem, but in general it is known to be a hard problem, 
and the best known algorithms have exponential time complexity.

Before we turn this into a quantum optimization problem, let's first write the problem mathematically. 
We can represent the decision of whether to take an item or not as a binary variable, 
where 1 means we take the item and 0 means we don't. Let's denote these variables by their ID as :math:`x_i` for :math:`i` in :math:`\{0, 1, ..., 6\}`.

The total weight of the items we take can then be expressed as:

:math:`W = 2000x_0 + 150x_1 + 750x_2 + 750x_3 + 200x_4 + 500x_5 + 1980x_6`

And the total importance can be expressed as:

:math:`I = 4x_0 + 5x_1 + 3x_2 + 4x_3 + 1x_4 + 1x_5 + 2x_6`

We want to maximize the total importance :math:`I`, subject to the constraint that the total weight :math:`W` does not exceed 4000 g:

.. math:: 

    \max_{x_i \in \{0, 1\}} I
    \text{subject to} \quad W \leq 4000

One issue with this formulation is that it has a constraint, which makes it more difficult to solve using quantum optimization algorithms.
To fix this, we''' use some techniques to turn this into a QUBO (Quadratic Unconstrained Binary Optimization) problem.
For more details on this process, check out the :doc:`QUBO reference </modules/core/core_qubo>`.

Now we can assume that the problem has been reformulated into the following unconstrained optimization problem:

.. math:: 

    \min_{x_i \in \{0, 1\}} \sum_{i=0,j=0}^6 c_{ij} x_i x_j

Where :math:`c_{ij}` are coefficients that encode the importance and weight of the items, as well as the constraint.

