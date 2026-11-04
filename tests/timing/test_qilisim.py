# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qilisdk.analog.hamiltonian import X, Z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.backend_config import ExecutionConfig
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core import Model
from qilisdk.core.qtensor import InitialState
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital import RX, Circuit
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.functionals import AnalogEvolution, DigitalPropagation
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.noise import Depolarizing, NoiseModel
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
from qilisdk.readout import Readout
from tests.timing.utils import best_time, linear_ramp_schedule

# These take their thread count from here rather than from OMP_NUM_THREADS, so pin it to one too.
execution_config = ExecutionConfig(seed=42, num_threads=1)
backend = QiliSim(execution_config=execution_config)


def test_digital_sampling():
    functional = DigitalPropagation(HardwareEfficientAnsatz(20, layers=10))
    readout = Readout().with_sampling(nshots=10_000)
    assert best_time(lambda: backend.execute(functional, readout)) < 4.5


def test_noisy_digital_sampling():
    noise_model = NoiseModel()
    noise_model.add(Depolarizing(probability=0.01))
    noisy_backend = QiliSim(noise_model=noise_model, execution_config=execution_config)
    functional = DigitalPropagation(HardwareEfficientAnsatz(8, layers=5))
    readout = Readout().with_sampling(nshots=1000)
    assert best_time(lambda: noisy_backend.execute(functional, readout)) < 1.6


def test_variational_program():
    model = Model.random_max_cut(6)
    program = VariationalProgram(
        DigitalPropagation(HardwareEfficientAnsatz(6, layers=2)),
        SciPyOptimizer(method="COBYLA", options={"maxiter": 50}),
        ModelCostFunction(model),
    )
    readout = Readout().with_sampling(nshots=1000)
    assert best_time(lambda: backend.execute(program, readout)) < 0.5


def test_analog_evolution():
    schedule = linear_ramp_schedule(12, dt=0.01)
    functional = AnalogEvolution(schedule, InitialState.UNIFORM)
    readout = Readout().with_sampling(nshots=1000)
    assert best_time(lambda: backend.execute(functional, readout)) < 2.5


def test_quantum_reservoir():
    nqubits = 6
    input_encoding = Circuit(nqubits)
    input_encoding.add(RX(0, theta=ReservoirInput("u", 0.0)))
    hamiltonian = sum(Z(i) * Z(i + 1) for i in range(nqubits - 1)) + 0.5 * sum(X(i) for i in range(nqubits))
    layer = ReservoirLayer(
        evolution_dynamics=Schedule(hamiltonians={"h": hamiltonian}, total_time=1.0, dt=0.1),
        input_encoding=input_encoding,
        qubits_to_reset=[0],
    )
    functional = QuantumReservoir(
        initial_state=InitialState.ZERO,
        reservoir_layer=layer,
        input_per_layer=[{"u": 0.01 * k} for k in range(100)],
    )
    readout = Readout().with_expectation(observables=[Z(i) for i in range(nqubits)])
    assert best_time(lambda: backend.execute(functional, readout)) < 2.6
