import numpy as np
import pytest

from qilisdk.analog import Schedule, X, Z
from qilisdk.analog.hamiltonian import PauliX, PauliZ
from qilisdk.core.interpolator import Interpolation
from qilisdk.core.variables import BinaryVariable, Parameter

# --- Constructor and Property Tests ---


def test_schedule_constructor_default():
    """Schedule constructed without Hamiltonians or schedule uses default values."""
    sched = Schedule(dt=1)
    assert sched.hamiltonians == {}
    # With no Hamiltonians provided, the default schedule is {0: {}}
    assert sched._coefficients == {}
    assert sched.T == 0
    assert sched.dt == 1
    # With no Hamiltonians, nqubits remains 0.
    assert sched.nqubits == 0

    with pytest.raises(ValueError, match=r"dt must be greater than zero."):
        Schedule(dt=0)


def test_schedule_parameters():
    p = [Parameter(f"p({i})", 0, bounds=(-10, 10)) for i in range(4)]

    H0 = p[0] * X(1) + X(0)
    H1 = p[1] * Z(1) + Z(0)

    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={})

    assert p[0] in list(schedule._parameters.values())
    assert p[1] in list(schedule._parameters.values())
    assert p[2] not in list(schedule._parameters.values())
    assert schedule.nparameters == 2

    schedule = Schedule(
        hamiltonians={"H0": H0, "H1": H1}, coefficients={"H0": {(0, 10): 1 + p[2]}, "H1": {(0, 10): p[3]}}
    )

    assert p[2] in list(schedule._parameters.values())
    assert p[3] in list(schedule._parameters.values())
    assert schedule.nparameters == 4

    schedule = Schedule(hamiltonians={"H0": H0, "H1": H1})
    schedule.update_hamiltonian("H0", new_coefficients={0: 1 + p[2]})
    schedule.update_hamiltonian("H1", new_coefficients={0: p[3]})

    assert p[2] in list(schedule._parameters.values())
    assert p[3] in list(schedule._parameters.values())
    assert schedule.nparameters == 4

    assert all(schedule.get_parameter_values()[i] == 0 for i in range(schedule.nparameters))
    assert all(list(schedule.get_parameter_bounds().values())[i] == (-10, 10) for i in range(schedule.nparameters))
    assert all(schedule.get_parameter_names()[i] == f"p({i})" for i in range(schedule.nparameters))

    vals = [0, 1, 2, 3]
    schedule.set_parameter_values(vals)
    assert all(schedule.get_parameter_values()[i] == vals[i] for i in range(schedule.nparameters))
    key_vals = {p[i].label: i + 1 for i in range(schedule.nparameters)}
    schedule.set_parameters(key_vals)
    assert all(schedule.get_parameter_values()[i] == list(key_vals.values())[i] for i in range(schedule.nparameters))

    schedule.set_parameter_values([0] * schedule.nparameters)
    schedule.set_parameter_bounds({f"p({i})": (-i, i) for i in range(schedule.nparameters)})
    assert all(list(schedule.get_parameter_bounds().values())[i] == (-i, i) for i in range(schedule.nparameters))

    with pytest.raises(
        ValueError,
        match=r"The provided parameter label test is not defined in the list of parameters in this object.",
    ):
        schedule.set_parameter_bounds({"test": (-10, 10)})

    with pytest.raises(
        ValueError,
        match=r"Parameter test is not defined in this Schedule.",
    ):
        schedule.set_parameters({"test": 0})

    with pytest.raises(ValueError, match=r"Provided 8 but this object has 4 parameters."):
        schedule.set_parameter_values([0] * 8)

    with pytest.raises(ValueError, match=r"Provided 2 but this object has 4 parameters."):
        schedule.set_parameter_values([0] * 2)


def test_schedule_constructor_with_hamiltonians_and_schedule():
    """When Hamiltonians and a partial schedule are provided, missing coefficients default to 0."""
    # H1 acts on qubit 0 (nqubits = 1); H2 acts on qubit 1 (nqubits = 2).
    H1 = PauliZ(0).to_hamiltonian()
    H2 = PauliX(1).to_hamiltonian()
    hams = {"H1": H1, "H2": H2}
    # Provide a schedule that sets only H1 at time step 0.
    sch = {"H1": {0: 0.5}}
    sched = Schedule(dt=1, hamiltonians=hams, coefficients=sch)
    # At t=0, H1 coefficient is set; H2 should be filled in with 0.
    assert np.isclose(sched.coefficients["H1"][0], 0.5)
    assert sched.coefficients["H2"][0] == 1
    # nqubits should be the maximum among Hamiltonians (here, 2 because H2 acts on qubit 1).
    assert sched.nqubits == 2


def test_schedule_constructor_invalid_schedule_reference():
    """Providing a schedule that references a non-declared Hamiltonian raises ValueError."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {"H1": {0: 0.5}, "H_unknown": {0: 1.0}}
    with pytest.raises(ValueError, match=r"Missing keys in hamiltonians: {'H_unknown'}"):
        Schedule(dt=1, hamiltonians=hams, coefficients=sch)


@pytest.mark.parametrize(("T", "dt"), [(10, 1), (20, 2)])
def test_len_schedule(T, dt):
    """The length of a Schedule is T/dt (as an integer)."""
    sched = Schedule(dt=dt, total_time=T)
    assert len(sched) == 2


# --- Schedule Modification Tests ---


def test_add_schedule_step_valid():
    """Adding a new schedule step inserts the coefficient and warns on overwrite."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(total_time=10, dt=1, hamiltonians=hams)
    # Add time step 2 with a new coefficient.
    sched.update_hamiltonian("H1", new_coefficients={2: 1.5})
    assert np.isclose(sched.coefficients["H1"][2], 1.5)


def test_add_schedule_step_invalid_reference():
    """Attempting to add a schedule step referencing an undefined Hamiltonian raises ValueError."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(total_time=10, dt=1, hamiltonians=hams)
    with pytest.raises(
        ValueError, match=r"Can't update unknown hamiltonian H_unknown. Did you mean `add_hamiltonian`?"
    ):
        sched.update_hamiltonian("H_unknown", new_coefficients={3: 1.0})


def test_update_hamiltonian_coefficient_valid():
    """Updating the Hamiltonian coefficient at a valid time step works correctly."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(dt=1, hamiltonians=hams)
    sched.update_hamiltonian("H1", new_coefficients={5: 3.0})
    assert np.isclose(sched.coefficients["H1"][5], 3.0)


def test_add_hamiltonian_new():
    """Adding a new Hamiltonian updates the schedule using a coefficient function."""
    sched = Schedule(dt=1)
    # Define a coefficient function: coefficient = factor * t.

    def coeff_func(t, factor=2):
        return t * factor

    H1 = PauliZ(0).to_hamiltonian()
    sched.add_hamiltonian("H1", H1, coefficients={(0, 4): coeff_func})
    # For T=4, dt=1, time steps are 0,1,2,3,4; expect coefficient = 2*t.
    for t in range(int(sched.T / sched.dt)):
        assert sched.coefficients["H1"][t] == 2 * t
    # nqubits should update based on the new Hamiltonian.
    assert sched.nqubits >= 1


def test_add_hamiltonian_existing():
    """Adding a Hamiltonian with an existing label warns and does not override the original."""
    H1 = PauliZ(0).to_hamiltonian()
    sched = Schedule(total_time=10, dt=1, hamiltonians={"H1": H1})
    # The original Hamiltonian (H1) remains unchanged.
    assert sched.hamiltonians["H1"] == H1


# --- Schedule Lookup and Iteration Tests ---


def test_getitem_with_exact_time_step():
    """__getitem__ returns the correct Hamiltonian when the time step is defined, including fallback for missing keys."""
    H1 = PauliZ(0).to_hamiltonian()
    H2 = PauliX(1).to_hamiltonian()
    hams = {"H1": H1, "H2": H2}
    # At time step 0: H1 coefficient 0.5, H2 coefficient 0.0; at time step 2: update H2 to 1.0.
    sch = {"H1": {(0, 10): 0.5}, "H2": {0: 0.0, 2: 1.0}}
    sched = Schedule(dt=1, hamiltonians=hams, coefficients=sch)
    # At t=0, expected Hamiltonian = 0.5 * H1.
    result0 = sched[0]
    expected0 = H1 * 0.5
    assert result0 == expected0
    # At t=2, H1 falls back to 0.5 from t=0 and H2 is updated to 1.0.
    result2 = sched[2]
    expected2 = H1 * 0.5 + H2 * 1.0
    assert result2 == expected2


def test_getitem_without_direct_time_step():
    """If a time step is not directly defined, __getitem__ falls back to the most recent earlier time step."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {"H1": {0: 0.5}}
    sched = Schedule(dt=1, hamiltonians=hams, coefficients=sch)
    # Requesting time step 3 (undefined) should fall back to time step 0.
    result = sched[3]
    expected = H1 * 0.5
    assert result == expected


def test_get_coefficient():
    """get_coefficient returns the correct coefficient with fallback to earlier time steps."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {"H1": {0: 0.5, 4: 1.0}}
    sched = Schedule(hamiltonians=hams, coefficients=sch, interpolation=Interpolation.STEP, dt=0.01)
    assert np.isclose(sched.coefficients["H1"][0], 0.5)
    assert np.isclose(sched.coefficients["H1"][2], 0.5)  # falls back to t=0
    assert np.isclose(sched.coefficients["H1"][4], 1.0)
    assert np.isclose(sched.coefficients["H1"][8], 1.0)  # falls back to t=4
    # For an undefined Hamiltonian key, get_coefficient returns 0.

    with pytest.raises(KeyError):
        sched.coefficients["H_unknown"][2]


def test_iteration():
    """Iterating over the Schedule yields Hamiltonians for time steps 0 through T/dt inclusive."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    # For T=4, dt=1, __len__() returns 4 but iteration yields time steps 0,1,2,3,4 (5 items).
    sched = Schedule(dt=0.01, hamiltonians=hams, coefficients={"H1": {(0, 4): 0.5}})
    results = list(iter(sched))
    assert len(results) == 100
    # The first item from iteration should equal sched[0].
    assert results[0] == sched[0]


def test_schedule_property_sorting():
    """The schedule property returns a dictionary sorted by time step keys."""
    H1 = PauliZ(0).to_hamiltonian()
    hams = {"H1": H1}
    unsorted_schedule = {"H1": {1: 0.3, 0: 0.1, 2: 0.2}}
    sched = Schedule(hamiltonians=hams, coefficients=unsorted_schedule)
    sorted_keys = list(sched.coefficients.keys())
    assert sorted_keys == sorted(sorted_keys)


def test_schedule_term_and_basevariable_errors():
    # Term with non-Parameter variable
    dummy = BinaryVariable("dummy")
    H1 = PauliZ(0).to_hamiltonian()

    # BaseVariable that is not a Parameter
    # sch2 = {"H1": {0: dummy}}
    # with pytest.raises(
    #     ValueError,
    #     match=r"Tlist can only contain parameters and no variables, but the term dummy contains objects other than parameters.",
    # ):
    #     Schedule(T=10, dt=1, hamiltonians={"H1": H1}, coefficients=sch2)

    term = (dummy) * 2
    sch = {"H1": {0: term}}
    with pytest.raises(
        ValueError,
        match=r"Tlist can only contain parameters and no variables, but the term dummy \* \(2\) contains objects other than parameters.",
    ):
        Schedule(total_time=10, dt=1, hamiltonians={"H1": H1}, coefficients=sch)


def test_add_schedule_step_term_basevariable_errors():
    dummy = BinaryVariable("dummy")
    term = dummy * 2
    H1 = PauliZ(0).to_hamiltonian()
    sched = Schedule(dt=1, hamiltonians={"H1": H1})
    with pytest.raises(
        ValueError,
        match=r"Tlist can only contain parameters and no variables, but the term dummy \* \(2\) contains objects other than parameters.",
    ):
        sched.update_hamiltonian("H1", new_coefficients={0: term})
    with pytest.raises(
        ValueError,
        match=r"Can't update unknown hamiltonian H2. Did you mean `add_hamiltonian`?",
    ):
        sched.update_hamiltonian("H2", new_coefficients={2: dummy})


def test_update_hamiltonian_coefficient_term_basevariable_errors():
    dummy = BinaryVariable("dummy")
    term = dummy * 2
    H1 = PauliZ(0).to_hamiltonian()
    sched = Schedule(total_time=10, dt=1, hamiltonians={"H1": H1})
    with pytest.raises(
        ValueError,
        match=r"Tlist can only contain parameters and no variables, but the term dummy \* \(2\) contains objects other than parameters.",
    ):
        sched.update_hamiltonian("H1", new_coefficients={1: term})
    with pytest.raises(ValueError):  # noqa: PT011
        sched.update_hamiltonian("H1", new_coefficients={2: dummy})


def test_draw_method_runs(monkeypatch):
    H1 = PauliZ(0).to_hamiltonian()
    sched = Schedule(total_time=4, dt=1, hamiltonians={"H1": H1}, coefficients={"H1": {0: 0.5}})

    # Monkeypatch renderer to avoid actual plotting
    class DummyRenderer:
        def __init__(self, *a, **kw):
            pass

        def plot(self, *a, **kw):
            return None

        def save(self, *a, **kw):
            return None

        def show(self):
            pass

    monkeypatch.setattr("qilisdk.utils.visualization.schedule_renderers.MatplotlibScheduleRenderer", DummyRenderer)
    sched.draw()


def test_add_hamiltonian_term_basevariable_errors():
    # Schedule function returns Term with non-Parameter variable
    dummy = BinaryVariable("dummy")

    def coeff_func_term(t):
        return dummy * 2

    H1 = PauliZ(0).to_hamiltonian()
    sched = Schedule(hamiltonians={"H1": H1})
    with pytest.raises(
        ValueError,
        match=r"function contains variables that are not time. Only Parameters are allowed.",
    ):
        sched.add_hamiltonian("H2", H1, coefficients={(0, 10): coeff_func_term})

    # Schedule function returns BaseVariable that is not a Parameter
    def coeff_func_basevar(t):
        return dummy

    with pytest.raises(
        ValueError,
        match=r"function contains variables that are not time. Only Parameters are allowed.",
    ):
        sched.add_hamiltonian("H3", H1, coefficients={(0, 10): coeff_func_basevar})


def test_add_schedule_through_function():
    # Define total time and timestep
    T = 10.0
    dt = 0.1
    nqubits = 1

    # Define Hamiltonians
    Hx = sum(X(i) for i in range(nqubits))
    Hz = sum(Z(i) for i in range(nqubits)) - sum(Z(i) * Z(i + 1) for i in range(nqubits - 1))

    # Build a time-dependent schedule
    schedule = Schedule(dt=dt)

    # Add hx with a time-dependent coefficient function
    schedule.add_hamiltonian(label="hx", hamiltonian=Hx, coefficients={(0, T): lambda t: 1 - t / T})

    # Add hz similarly
    schedule.add_hamiltonian(label="hz", hamiltonian=Hz, coefficients={(0, T): lambda t: t / T})

    assert schedule.coefficients["hx"][0] == 1
    assert schedule.coefficients["hz"][0] == 0
    assert schedule.coefficients["hx"][T] == 0
    assert schedule.coefficients["hz"][T] == 1


def test_linear_schedule_interpolation():
    dt = 1
    H1 = PauliZ(0).to_hamiltonian()
    sch = {"H1": {0: 0, 5: 1, 10: 2}}
    sched = Schedule(dt=dt, hamiltonians={"H1": H1}, coefficients=sch)

    # At t=0, should be 0.0
    assert np.isclose(sched.coefficients["H1"][0], 0.0)

    # At t=5, should be 1.0
    assert np.isclose(sched.coefficients["H1"][5], 1.0)
    # At t=10, should be 2.0
    assert np.isclose(sched.coefficients["H1"][10], 2.0)

    # At t=2, should interpolate between 0 and 1

    assert np.isclose(sched.coefficients["H1"][2], 0.4)
    # At t=7, should interpolate between 1 and 2

    assert np.isclose(sched.coefficients["H1"][7], 1.4)


def test_linear_schedule_edge_cases():
    T = 10
    dt = 1
    H1 = PauliZ(0).to_hamiltonian()
    # Only one time step defined
    sch = {"H1": {0: 3.0}}
    sched = Schedule(dt=dt, hamiltonians={"H1": H1}, coefficients=sch)
    # All times should return 3.0
    for t in range(0, T + 1, dt):
        assert np.isclose(sched.coefficients["H1"][t], 3.0)


def test_linear_schedule_expression():
    dt = 1
    p = Parameter("p", 2.0)
    H1 = PauliZ(0).to_hamiltonian()
    sch = {"H1": {0: p, 10: 4.0}}
    sched = Schedule(dt=dt, hamiltonians={"H1": H1}, coefficients=sch)
    # At t=0, should be p
    assert sched.coefficients["H1"].get_coefficient_expression(0) == p
    # At t=10, should be 4.0
    assert np.isclose(sched.coefficients["H1"].get_coefficient_expression(10), 4.0)
    # At t=5, should interpolate between p and 4.0
    expr = sched.coefficients["H1"].get_coefficient(5)
    # Should be a linear combination of p and 4.0
    alpha = (5 - 0) / (10 - 0)
    expected = (1 - alpha) * p + alpha * 4.0
    assert expr == expected.evaluate({p: 2})
