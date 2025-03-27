import pytest

from qilisdk.analog.hamiltonian import X, Z
from qilisdk.analog.schedule import Schedule

# --- Constructor and Property Tests ---


def test_schedule_constructor_default():
    """Schedule constructed without Hamiltonians or schedule uses default values."""
    sched = Schedule(T=10, dt=1)
    assert sched.hamiltonians == {}
    # With no Hamiltonians provided, the default schedule is {0: {}}
    assert sched.schedule == {0: {}}
    assert sched.T == 10
    assert sched.dt == 1
    # With no Hamiltonians, nqubits remains 0.
    assert sched.nqubits == 0


def test_schedule_constructor_with_hamiltonians_and_schedule():
    """When Hamiltonians and a partial schedule are provided, missing coefficients default to 0."""
    # H1 acts on qubit 0 (nqubits = 1); H2 acts on qubit 1 (nqubits = 2).
    H1 = Z(0).to_hamiltonian()
    H2 = X(1).to_hamiltonian()
    hams = {"H1": H1, "H2": H2}
    # Provide a schedule that sets only H1 at time step 0.
    sch = {0: {"H1": 0.5}}
    sched = Schedule(T=10, dt=1, hamiltonians=hams, schedule=sch)
    # At t=0, H1 coefficient is set; H2 should be filled in with 0.
    assert sched.schedule[0]["H1"] == 0.5
    assert sched.schedule[0]["H2"] == 0
    # nqubits should be the maximum among Hamiltonians (here, 2 because H2 acts on qubit 1).
    assert sched.nqubits == 2


def test_schedule_constructor_invalid_schedule_reference():
    """Providing a schedule that references a non-declared Hamiltonian raises ValueError."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {0: {"H1": 0.5, "H_unknown": 1.0}}
    with pytest.raises(ValueError):  # noqa: PT011
        Schedule(T=10, dt=1, hamiltonians=hams, schedule=sch)


@pytest.mark.parametrize(("T", "dt"), [(10, 1), (20, 2)])
def test_len_schedule(T, dt):
    """The length of a Schedule is T/dt (as an integer)."""
    sched = Schedule(T=T, dt=dt)
    assert len(sched) == int(T / dt)


# --- Schedule Modification Tests ---


def test_add_schedule_step_valid():
    """Adding a new schedule step inserts the coefficient and warns on overwrite."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(T=10, dt=1, hamiltonians=hams)
    # Add time step 2 with a new coefficient.
    sched.add_schedule_step(2, {"H1": 1.5})
    assert sched.schedule[2]["H1"] == 1.5
    # Overwriting an existing time step should warn.
    with pytest.warns(RuntimeWarning):
        sched.add_schedule_step(2, {"H1": 2.0})
    assert sched.schedule[2]["H1"] == 2.0


def test_add_schedule_step_invalid_reference():
    """Attempting to add a schedule step referencing an undefined Hamiltonian raises ValueError."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(T=10, dt=1, hamiltonians=hams)
    with pytest.raises(ValueError):  # noqa: PT011
        sched.add_schedule_step(3, {"H_unknown": 1.0})


def test_update_hamiltonian_coefficient_valid():
    """Updating the Hamiltonian coefficient at a valid time step works correctly."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(T=10, dt=1, hamiltonians=hams)
    sched.update_hamiltonian_coefficient_at_time_step(5, "H1", 3.0)
    assert sched.schedule[5]["H1"] == 3.0


def test_update_hamiltonian_coefficient_invalid_time():
    """Updating a coefficient at a time step after the end of the annealing schedule raises ValueError."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sched = Schedule(T=10, dt=1, hamiltonians=hams)
    with pytest.raises(ValueError):  # noqa: PT011
        sched.update_hamiltonian_coefficient_at_time_step(11, "H1", 2.0)


def test_add_hamiltonian_new():
    """Adding a new Hamiltonian updates the schedule using a coefficient function."""
    sched = Schedule(T=4, dt=1)
    # Define a coefficient function: coefficient = factor * t.

    def coeff_func(t, factor=1):
        return t * factor

    H1 = Z(0).to_hamiltonian()
    sched.add_hamiltonian("H1", H1, schedule=coeff_func, factor=2)
    # For T=4, dt=1, time steps are 0,1,2,3,4; expect coefficient = 2*t.
    for t in range(int(sched.T / sched.dt) + 1):
        assert sched.schedule.get(t, {}).get("H1", 0) == 2 * t
    # nqubits should update based on the new Hamiltonian.
    assert sched.nqubits >= 1


def test_add_hamiltonian_existing():
    """Adding a Hamiltonian with an existing label warns and does not override the original."""
    H1 = Z(0).to_hamiltonian()
    sched = Schedule(T=10, dt=1, hamiltonians={"H1": H1})
    with pytest.warns(RuntimeWarning):
        sched.add_hamiltonian("H1", X(0).to_hamiltonian())
    # The original Hamiltonian (H1) remains unchanged.
    assert sched.hamiltonians["H1"] == H1


# --- Schedule Lookup and Iteration Tests ---


def test_getitem_with_exact_time_step():
    """__getitem__ returns the correct Hamiltonian when the time step is defined, including fallback for missing keys."""
    H1 = Z(0).to_hamiltonian()
    H2 = X(1).to_hamiltonian()
    hams = {"H1": H1, "H2": H2}
    # At time step 0: H1 coefficient 0.5, H2 coefficient 0.0; at time step 2: update H2 to 1.0.
    sch = {0: {"H1": 0.5, "H2": 0.0}, 2: {"H2": 1.0}}
    sched = Schedule(T=10, dt=1, hamiltonians=hams, schedule=sch)
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
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {0: {"H1": 0.5}}
    sched = Schedule(T=10, dt=1, hamiltonians=hams, schedule=sch)
    # Requesting time step 3 (undefined) should fall back to time step 0.
    result = sched[3]
    expected = H1 * 0.5
    assert result == expected


def test_get_coefficient():
    """get_coefficient returns the correct coefficient with fallback to earlier time steps."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    sch = {0: {"H1": 0.5}, 4: {"H1": 1.0}}
    sched = Schedule(T=10, dt=1, hamiltonians=hams, schedule=sch)
    assert sched.get_coefficient(0, "H1") == 0.5
    assert sched.get_coefficient(2, "H1") == 0.5  # falls back to t=0
    assert sched.get_coefficient(4, "H1") == 1.0
    assert sched.get_coefficient(8, "H1") == 1.0  # falls back to t=4
    # For an undefined Hamiltonian key, get_coefficient returns 0.
    assert sched.get_coefficient(2, "H_unknown") == 0


def test_iteration():
    """Iterating over the Schedule yields Hamiltonians for time steps 0 through T/dt inclusive."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    # For T=4, dt=1, __len__() returns 4 but iteration yields time steps 0,1,2,3,4 (5 items).
    sched = Schedule(T=4, dt=1, hamiltonians=hams, schedule={0: {"H1": 0.5}})
    results = list(iter(sched))
    assert len(results) == 5
    # The first item from iteration should equal sched[0].
    assert results[0] == sched[0]


def test_schedule_property_sorting():
    """The schedule property returns a dictionary sorted by time step keys."""
    H1 = Z(0).to_hamiltonian()
    hams = {"H1": H1}
    unsorted_schedule = {3: {"H1": 0.3}, 0: {"H1": 0.1}, 2: {"H1": 0.2}}
    sched = Schedule(T=10, dt=1, hamiltonians=hams, schedule=unsorted_schedule)
    sorted_keys = list(sched.schedule.keys())
    assert sorted_keys == sorted(sorted_keys)
