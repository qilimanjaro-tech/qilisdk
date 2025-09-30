from qilisdk.analog import LinearSchedule
from qilisdk.analog.hamiltonian import PauliZ
from qilisdk.common.variables import Parameter


def test_linear_schedule_interpolation():
    T = 10
    dt = 1
    H1 = PauliZ(0).to_hamiltonian()
    sch = {0: {"H1": 0.0}, 5: {"H1": 1.0}, 10: {"H1": 2.0}}
    sched = LinearSchedule(T=T, dt=dt, hamiltonians={"H1": H1}, schedule=sch)

    # At t=0, should be 0.0
    assert sched.get_coefficient(0, "H1") == 0.0

    # At t=5, should be 1.0
    assert sched.get_coefficient(5, "H1") == 1.0
    # At t=10, should be 2.0
    assert sched.get_coefficient(10, "H1") == 2.0

    # At t=2, should interpolate between 0 and 1

    assert sched.get_coefficient(2, "H1") == 0.4
    # At t=7, should interpolate between 1 and 2

    assert sched.get_coefficient(7, "H1") == 1.4


def test_linear_schedule_edge_cases():
    T = 10
    dt = 1
    H1 = PauliZ(0).to_hamiltonian()
    # Only one time step defined
    sch = {0: {"H1": 3.0}}
    sched = LinearSchedule(T=T, dt=dt, hamiltonians={"H1": H1}, schedule=sch)
    # All times should return 3.0
    for t in range(0, T + 1, dt):
        assert sched.get_coefficient(t, "H1") == 3.0
    # Undefined Hamiltonian returns 0
    assert sched.get_coefficient(5, "H_unknown") == 0


def test_linear_schedule_expression():
    T = 10
    dt = 1
    p = Parameter("p", 2.0)
    H1 = PauliZ(0).to_hamiltonian()
    sch = {0: {"H1": p}, 10: {"H1": 4.0}}
    sched = LinearSchedule(T=T, dt=dt, hamiltonians={"H1": H1}, schedule=sch)
    # At t=0, should be p
    assert sched.get_coefficient_expression(0, "H1") == p
    # At t=10, should be 4.0
    assert sched.get_coefficient_expression(10, "H1") == 4.0
    # At t=5, should interpolate between p and 4.0
    expr = sched.get_coefficient_expression(5, "H1")
    # Should be a linear combination of p and 4.0
    alpha = (5 - 0) / (10 - 0)
    expected = (1 - alpha) * p + alpha * 4.0
    assert expr == expected
