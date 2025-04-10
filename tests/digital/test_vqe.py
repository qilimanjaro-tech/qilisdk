from unittest.mock import MagicMock

import pytest

from qilisdk.common.optimizer_result import OptimizerResult
from qilisdk.digital.vqe import VQE, VQEResult


@pytest.fixture
def dummy_ansatz():
    """
    Create a dummy ansatz with a get_circuit method.
    The get_circuit method returns a string representing the quantum circuit,
    constructed with the given parameters.
    """
    ansatz = MagicMock()
    ansatz.get_circuit.side_effect = lambda params: f"circuit_for_{params}"
    return ansatz


@pytest.fixture
def initial_params():
    """Initial ansatz parameters for testing."""
    return [0.1, 0.2, 0.3]


@pytest.fixture
def dummy_cost_function():
    """
    A dummy cost function that accepts a DigitalResult and returns a fixed cost.
    For testing, we assume the cost is always 0.7.
    """
    return lambda digital_result: 0.7


@pytest.fixture
def dummy_backend():
    """
    Create a dummy digital backend that simulates execution of a circuit.
    The execute method returns a dummy DigitalResult (here, a MagicMock).
    """
    backend = MagicMock()
    dummy_result = MagicMock(name="DigitalResult")
    backend.execute.return_value = dummy_result
    return backend


@pytest.fixture
def dummy_optimizer():
    """
    Create a dummy optimizer that, upon optimization, returns a tuple of
    (optimal_cost, optimal_parameters). For testing, we use (0.2, [0.9, 0.1]).
    """
    optimizer = MagicMock()
    optimizer.optimize.side_effect = lambda func, init_params, store_intermediate_results: OptimizerResult(
        0.2, [0.9, 0.1]
    )
    return optimizer


def test_vqe_properties_assignment(dummy_ansatz, initial_params, dummy_cost_function):
    """
    Test that the VQE instance correctly stores its initial properties.

    Verifies that the ansatz, initial parameters, and cost function are assigned properly.
    """
    vqe = VQE(dummy_ansatz, initial_params, dummy_cost_function)
    assert vqe._ansatz == dummy_ansatz
    assert vqe._initial_params == initial_params
    assert vqe._cost_function == dummy_cost_function


def test_obtain_cost_calls_backend(dummy_ansatz, initial_params, dummy_cost_function, dummy_backend):
    """
    Test that the obtain_cost method correctly generates the circuit, calls the backend,
    and applies the cost function.

    This ensures:
      - ansatz.get_circuit is called with the provided parameters.
      - backend.execute is called with the generated circuit and specified number of shots.
      - The returned cost is as defined by the dummy cost function.
    """
    vqe = VQE(dummy_ansatz, initial_params, dummy_cost_function)
    test_params = [0.1, 0.2, 0.3]
    # Call obtain_cost with a custom number of shots.
    cost = vqe.obtain_cost(test_params, backend=dummy_backend, nshots=500)

    # Verify get_circuit and execute were called with expected arguments.
    expected_circuit = f"circuit_for_{test_params}"
    dummy_ansatz.get_circuit.assert_called_once_with(test_params)
    dummy_backend.execute.assert_called_once_with(circuit=expected_circuit, nshots=500)

    # The dummy_cost_function returns 0.7 regardless of input.
    assert cost == 0.7


def test_obtain_cost_default_nshots(dummy_ansatz, initial_params, dummy_cost_function, dummy_backend):
    """
    Test that obtain_cost uses the default nshots value (1000) when not explicitly provided.

    Verifies backend.execute is called with nshots=1000.
    """
    vqe = VQE(dummy_ansatz, initial_params, dummy_cost_function)
    test_params = [0.1, 0.2, 0.3]
    _ = vqe.obtain_cost(test_params, backend=dummy_backend)  # nshots not specified

    expected_circuit = f"circuit_for_{test_params}"
    dummy_backend.execute.assert_called_once_with(circuit=expected_circuit, nshots=1000)


def test_execute_calls_optimizer_and_returns_vqeresult(
    dummy_ansatz, initial_params, dummy_cost_function, dummy_backend, dummy_optimizer
):
    """
    Test that the execute method:
      - Invokes the optimizer's optimize method with the correct objective function and initial parameters.
      - Uses the optimizer's returned optimal parameters and cost to create a VQEResult.

    The test confirms that:
      - The optimizer is called exactly once.
      - The final result is a VQEResult with optimal_cost equal to 0.2 and optimal_parameters equal to [0.9, 0.1].
    """
    vqe = VQE(dummy_ansatz, initial_params, dummy_cost_function)
    result = vqe.execute(backend=dummy_backend, optimizer=dummy_optimizer, nshots=600)

    # Verify that optimizer.optimize was called with the objective function and initial parameters.
    dummy_optimizer.optimize.assert_called_once()
    _, passed_initial_params = dummy_optimizer.optimize.call_args[0]
    assert passed_initial_params == initial_params

    # Check that the resulting object is a VQEResult with the expected values.
    assert isinstance(result, VQEResult)
    assert result.optimal_cost == 0.2
    assert result.optimal_parameters == [0.9, 0.1]
