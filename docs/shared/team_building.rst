Say we have four people and we need to form two teams of two people each. 
We have some information about how well each pair of people work together, 
and we want to form the teams such that the total "compatibility" of the teams is maximized.
The information about who likes who is as follows:

+-------+-------+-------+-------+-------+
|       | Alice | Bob   | Carol | Dave  |
+=======+=======+=======+=======+=======+
| Alice | N/A   | 1     | 3     | 4     |
+-------+-------+-------+-------+-------+
| Bob   | 1     | N/A   | 5     | 2     |
+-------+-------+-------+-------+-------+
| Carol | 3     | 5     | N/A   | 6     |
+-------+-------+-------+-------+-------+
| Dave  | 4     | 2     | 6     | N/A   |
+-------+-------+-------+-------+-------+

So this means that Alice would prefer to work with Dave, while Bob would prefer to work with Carol, and so on.

Mathematically, we can represent our two teams as binary variables, where :math:`x_i` is 0 if person :math:`i` is in team 0, and 1 if they are in team 1.
The compatibility of team 0 can then be expressed as:

.. math::

    C_0 = \sum_{i=0,j=0}^3 p_{ij} (1 - x_i) (1 - x_j)

Where :math:`p_{ij}` is the compatibility score between person :math:`i` and person :math:`j`.
Note that here each term is only non-zero if both :math:`x_i` and :math:`x_j` are 0, which means that both people are in team 0.
Meanwhile the compatibility of team 1 can be expressed as:

.. math::

    C_1 = \sum_{i=0,j=0}^3 p_{ij} x_i x_j

Thus we want to maximize the total compatibility :math:`C = C_0 + C_1`, subject to the constraint 
that team 1 (and thus team 0) has exactly two people:

.. math:: 

    \max_{x_i \in \{0, 1\}} C

.. math:: 

    \text{subject to} \quad x_{Alice} + x_{Bob} + x_{Carol} + x_{Dave} = 2

One issue with this formulation is that it has a constraint, which makes it more difficult to solve using quantum optimization algorithms.
To fix this, we can use some techniques to turn this into a QUBO (Quadratic Unconstrained Binary Optimization) problem.
In this case, since we just have a single equality constraint, we can add a squared version of it to the objective function 
as a penalty term, such that the penalty term is zero when the constraint is satisfied, and positive otherwise.
For more details on this process, check out the :doc:`QUBO reference </modules/core/core_qubo>`.
Now we can assume that the problem has been reformulated into the following unconstrained optimization problem:

.. math:: 

    \min_{x_i \in \{0, 1\}} \sum_{i=0,j=0}^4 c_{ij} x_i x_j

Where :math:`c_{ij}` are coefficients that encode the objective function as well as the constraint.

