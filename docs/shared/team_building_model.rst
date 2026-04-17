.. code-block:: python

    from qilisdk.core.model import QUBO, ObjectiveSense
    from qilisdk.core.variables import BinaryVariable, EQ

    num_people = 4
    vars = [BinaryVariable(f"x{i}") for i in range(num_people)]
    preferences = [[0, 1, 3, 4],
                  [1, 0, 5, 2],
                  [3, 5, 0, 6],
                  [4, 2, 6, 0]]

    model = QUBO("team_formation_example")
    
    team_1 = sum(
        preferences[i][j] * vars[i] * vars[j] 
        for i in range(num_people) 
        for j in range(i+1, num_people)
    )
    
    team_0 = sum(
        preferences[i][j] * (1 - vars[i]) * (1 - vars[j]) 
        for i in range(num_people) 
        for j in range(i+1, num_people)
    )
    
    model.set_objective(
        team_0 + team_1, 
        label="obj", 
        sense=ObjectiveSense.MAXIMIZE
    )
    
    model.add_constraint(
        "team_size_constraint", 
        EQ(sum(vars[i] for i in range(num_people)), 2), 
        lagrange_multiplier=10
    )