from pathlib import Path

import numpy as np
import pytest

from qilisdk.analog.algorithms import TimeEvolution
from qilisdk.analog.hamiltonian import PauliY, X, Y, Z
from qilisdk.analog.quantum_objects import ket, tensor_prod
from qilisdk.analog.schedule import Schedule
from qilisdk.utils.serialization import (
    DeserializationError,
    deserialize,
    deserialize_from,
    serialize,
    serialize_to,
)


def test_time_evolution_algorithm_serialization():
    T = 10
    dt = 0.1
    steps = np.linspace(0, T, int(T / dt))

    nqubits = 1

    H1 = sum(X(i) for i in range(nqubits))
    H2 = sum(Z(i) for i in range(nqubits))

    schedule = Schedule(
        T,
        dt,
        hamiltonians={"h1": H1, "h2": H2},
        schedule={
            t: {
                "h1": 1 - steps[t] / T,
                "h2": steps[t] / T,
            }
            for t in range(len(steps))
        },
    )

    state = tensor_prod([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()

    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=state,
        observables=[Z(0), X(0), Y(0), Z(nqubits - 1), X(nqubits - 1), Y(nqubits - 1)],
    )

    serialized_time_evolution = serialize(time_evolution)
    deserialized_time_evolution = deserialize(serialized_time_evolution, TimeEvolution)
    assert isinstance(deserialized_time_evolution, TimeEvolution)

    serialize_to(time_evolution, "time_evolution.yml")
    deserialized_from_time_evolution = deserialize_from("time_evolution.yml", TimeEvolution)
    assert isinstance(deserialized_from_time_evolution, TimeEvolution)

    Path("time_evolution.yml").unlink()


def test_deserialization_with_wrong_yaml_raises_error():
    not_valid_yaml = "!SomeClass _property: 100"

    with pytest.raises(DeserializationError):
        _ = deserialize(not_valid_yaml)

    Path("not_valid_yaml.yml").write_text(not_valid_yaml, encoding="utf-8")

    with pytest.raises(DeserializationError):
        _ = deserialize_from("not_valid_yaml.yml")

    Path("not_valid_yaml.yml").unlink()


def test_deserialization_with_wrong_cls_raises_error():
    operator = X(0)

    serialized_operator = serialize(operator)

    with pytest.raises(DeserializationError):
        _ = deserialize(serialized_operator, PauliY)

    serialize_to(operator, "pauli_x.yml")

    with pytest.raises(DeserializationError):
        _ = deserialize_from("pauli_x.yml", PauliY)

    Path("pauli_x.yml").unlink()
