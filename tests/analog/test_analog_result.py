import numpy as np
import pytest

from qilisdk.analog.analog_result import AnalogResult
from qilisdk.analog.quantum_objects import QuantumObject


def test_default_values():
    """When no arguments are provided, properties should default correctly."""
    result = AnalogResult()
    # Both expected arrays default to empty arrays.
    np.testing.assert_array_equal(result.final_expected_values, np.array([]))
    np.testing.assert_array_equal(result.expected_values, np.array([]))
    # final_state and intermediate_states default to None.
    assert result.final_state is None
    assert result.intermediate_states is None


@pytest.mark.parametrize(
    ("final_arr_input", "expected_arr_input", "final_arr_expected", "expected_arr_expected"),
    [
        (None, None, np.array([]), np.array([])),
        (np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2]), np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2])),
        (np.array([]), np.array([]), np.array([]), np.array([])),
    ],
)
def test_array_properties(final_arr_input, expected_arr_input, final_arr_expected, expected_arr_expected):
    """Test that final_expected_values and expected_values are set correctly."""
    result = AnalogResult(
        final_expected_values=final_arr_input,
        expected_values=expected_arr_input,
    )
    np.testing.assert_array_equal(result.final_expected_values, final_arr_expected)
    np.testing.assert_array_equal(result.expected_values, expected_arr_expected)


def test_custom_values():
    """When provided, the properties should return the same objects."""
    final_expected = np.array([1.0, 2.0, 3.0])
    expected = np.array([0.1, 0.2])
    # Create a simple QuantumObject; for instance, a valid 1x1 matrix.
    final_state = QuantumObject(np.array([[42]]))
    # Create intermediate states as a list of QuantumObjects.
    intermediate_states = [
        QuantumObject(np.array([[1]])),
        QuantumObject(np.array([[2]])),
    ]
    result = AnalogResult(
        final_expected_values=final_expected,
        expected_values=expected,
        final_state=final_state,
        intermediate_states=intermediate_states,
    )
    np.testing.assert_array_equal(result.final_expected_values, final_expected)
    np.testing.assert_array_equal(result.expected_values, expected)
    assert result.final_state == final_state
    assert result.intermediate_states == intermediate_states


def test_repr_format():
    """The __repr__ output should include the property names and formatted values."""
    final_expected = np.array([1, 2])
    expected = np.array([3, 4])
    final_state = QuantumObject(np.array([[1, 0], [0, 1]]))
    intermediate_states = [QuantumObject(np.array([[0, 1], [1, 0]]))]
    result = AnalogResult(
        final_expected_values=final_expected,
        expected_values=expected,
        final_state=final_state,
        intermediate_states=intermediate_states,
    )
    rep = repr(result)
    # Check that the repr string contains the names of the properties.
    assert "final_expected_values=" in rep
    assert "expected_values=" in rep
    assert "final_state=" in rep
    assert "intermediate_states=" in rep
    # Also check that some of the expected numbers (as strings) appear in the representation.
    for num in (1, 2, 3, 4):
        assert str(num) in rep
