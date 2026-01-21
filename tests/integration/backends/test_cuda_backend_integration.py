import numpy as np
import pytest

from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import ket
from qilisdk.functionals.time_evolution import TimeEvolution

pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
)

from qilisdk.analog.hamiltonian import Hamiltonian, PauliX, PauliZ
from qilisdk.backends.cuda_backend import CudaBackend
from qilisdk.digital.ansatz import TrotterizedTimeEvolution
from qilisdk.digital.gates import H
from qilisdk.functionals.sampling import Sampling


def test_trotterized_time_evolution_results():
    """TrotterizedTimeEvolution should honor schedule dt and trotter_steps."""

    h0 = Hamiltonian({(PauliX(0),): -1})
    h1 = Hamiltonian({(PauliZ(0),): 1})
    schedule = Schedule(
        hamiltonians={"h0": h0, "h1": h1},
        coefficients={"h0": {(0, 1): lambda t: 1 - t}, "h1": {(0, 1): lambda t: t}},
        dt=0.01,
        total_time=10,
    )
    cuda_ansatz = TrotterizedTimeEvolution(schedule)
    cuda_ansatz.insert([H(0)], 0)

    te = TimeEvolution(
        schedule,
        observables=[h1],
        initial_state=(ket(0) + ket(1)).unit(),
    )
    nshots = 10_000
    cuda = CudaBackend()
    te_res = cuda.execute(te)
    sam_res = cuda.execute(Sampling(cuda_ansatz, nshots=nshots))
    probs = np.abs((te_res.final_state.dense()) ** 2).T[0]
    te_probs = {("{" + ":0" + str(schedule.nqubits) + "b}").format(i): float(p) for i, p in enumerate(probs)}
    sam_probs = {key: sam_res.samples[key] / nshots if key in sam_res.samples else 0.0 for key in te_probs}
    assert all(np.isclose(list(te_probs.values()), list(sam_probs.values()), atol=1e-3))
