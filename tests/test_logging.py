from loguru_caplog import loguru_caplog as caplog  # noqa: F401

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import LT, BinaryVariable, Domain, OneHot, Variable


def test_log_output(caplog):  # noqa: F811
    N = 2
    b = [BinaryVariable(f"b({i})") for i in range(N)]
    x = Variable("x", Domain.POSITIVE_INTEGER, bounds=(0, 10), encoding=OneHot)

    m = Model("test")

    m.set_objective(x + 1, sense=ObjectiveSense.MAXIMIZE)

    m.add_constraint("con2", LT(b[1], 2))

    m.to_qubo()

    test_message = " because it is always feasible."
    assert test_message in caplog.text
