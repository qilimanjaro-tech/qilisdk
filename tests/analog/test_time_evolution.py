from unittest.mock import MagicMock

import numpy as np
import pytest

from qilisdk.analog.algorithms import TimeEvolution
from qilisdk.analog.analog_backend import AnalogBackend
from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.hamiltonian import X, Z
from qilisdk.analog.quantum_objects import QuantumObject
from qilisdk.analog.schedule import Schedule


@pytest.fixture
def schedule():
    # Create a minimal schedule with T=1, dt=1 (empty schedule is acceptable here).
    return Schedule(T=1, dt=1)


@pytest.fixture
def initial_state():
    # For a valid ket state, provide a 2x1 array.
    arr = np.array([[1], [0]])
    return QuantumObject(arr)


@pytest.fixture
def observables():
    # Create a list of observables:
    # One observable as a Pauli operator (Z on qubit 0) and one as a Hamiltonian (X on qubit 0).
    return [Z(0), X(0).to_hamiltonian()]


@pytest.fixture
def fake_backend(initial_state):
    # Create a fake backend whose evolve() method returns a known AnalogResult.
    backend = MagicMock(spec=AnalogBackend)
    dummy_result = AnalogResult(
        final_expected_values=np.array([1.0]),
        expected_values=np.array([0.5]),
        final_state=initial_state,
        intermediate_states=[initial_state],
    )
    backend.evolve.return_value = dummy_result
    return backend


def test_properties_assignment(schedule, observables, initial_state):
    """Test that the TimeEvolution instance assigns properties correctly."""
    n_shots = 500
    te = TimeEvolution(schedule=schedule, observables=observables, initial_state=initial_state, n_shots=n_shots)
    assert te.schedule == schedule
    assert te.observables == observables
    assert te.initial_state == initial_state
    assert te.n_shots == n_shots


@pytest.mark.parametrize("store_intermediate_results", [True, False])
def test_evolve_calls_backend(fake_backend, schedule, observables, initial_state, store_intermediate_results):
    """
    Test that the evolve() method correctly calls the backend.evolve() method with the
    expected parameters and returns its result.
    """
    te = TimeEvolution(schedule=schedule, observables=observables, initial_state=initial_state)
    result = te.evolve(backend=fake_backend, store_intermediate_results=store_intermediate_results)
    fake_backend.evolve.assert_called_once_with(
        schedule=schedule,
        initial_state=initial_state,
        observables=observables,
        store_intermediate_results=store_intermediate_results,
    )
    assert result == fake_backend.evolve.return_value


def test_evolve_default_store_intermediate(fake_backend, schedule, initial_state, observables):
    """
    Test that evolve() without explicit store_intermediate_results uses the default value (False).
    """
    te = TimeEvolution(schedule=schedule, observables=observables, initial_state=initial_state)
    te.evolve(backend=fake_backend)  # No argument provided.
    fake_backend.evolve.assert_called_once_with(
        schedule=schedule,
        initial_state=initial_state,
        observables=observables,
        store_intermediate_results=False,
    )
