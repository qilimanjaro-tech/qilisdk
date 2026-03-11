# Copyright 2025 Qilimanjaro Quantum Tech
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
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from qilisim_module import QiliSimCpp  # ty:ignore[unresolved-import]

from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.digital_evolution import DigitalEvolution
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.settings import get_settings

from .backend import Backend
from .backend_config import AnalogMethod, DigitalMethod, ExecutionConfig, SolverConfigDict

if TYPE_CHECKING:
    from qilisdk.core import QTensor
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution import TimeEvolution
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
    from qilisdk.noise.noise_model import NoiseModel


class QiliSim(Backend):
    """
    Backend that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom C++ simulator.

    Setup Example:
    .. code-block:: python

    from qilisdk.backends import AnalogMethod, DigitalMethod, ExecutionConfig, MonteCarloConfig,  QiliSim

    backend = QiliSim(
        analog_simulation_method=AnalogMethod.arnoldi(
            dim=16,
            num_substeps=2
        ),
        digital_simulation_method=DigitalMethod.statevector(max_cache_size=2_000),
        execution_config=ExecutionConfig(num_threads=4, seed=42, monte_carlo=MonteCarloConfig(trajectories=200),),
    )
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        analog_simulation_method: AnalogMethod | None = None,
        digital_simulation_method: DigitalMethod | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """
        Instantiate a new :class:`QiliSim` backend. This is a CPU-based simulator
        implemented in C++, using pybind11 for bindings.

        Args:
            noise_model: Optional noise model applied during execution.
            analog_simulation_method: Analog simulation configuration. Available options: :meth:`AnalogMethod.integrator`,
                :meth:`AnalogMethod.arnoldi`, or :meth:`AnalogMethod.direct`. Defaults to
                :meth:`AnalogMethod.integrator`.
            digital_simulation_method: Digital simulation configuration. Available options: :meth:`DigitalMethod.statevector`. Defaults to :meth:`DigitalMethod.statevector`.
            execution_config: Execution-level configuration for threading, random seed and monte-carlo executions.
                Defaults to the default configuration in :class:`ExecutionConfig`.

        Raises:
            ValueError: If a configuration argument has an invalid type.
        """

        # Initialize the backend and the class vars
        super().__init__()
        self.qili_sim = QiliSimCpp()
        self._noise_model = noise_model

        analog_simulation_method: AnalogMethod = analog_simulation_method or AnalogMethod.integrator()
        digital_simulation_method: DigitalMethod = digital_simulation_method or DigitalMethod.statevector()
        execution_config: ExecutionConfig = execution_config or ExecutionConfig()

        if not isinstance(analog_simulation_method, AnalogMethod):
            raise ValueError(
                f"Analog simulation method provided ({analog_simulation_method.__class__}) is not a valid analog simulation method."
            )

        if not isinstance(digital_simulation_method, DigitalMethod):
            raise ValueError(
                f"Digital simulation method provided ({digital_simulation_method.__class__}) is not a valid digital simulation method."
            )
        if not isinstance(execution_config, ExecutionConfig):
            raise ValueError(
                f"Execution config provided ({execution_config.__class__}) is not a valid execution configuration."
            )

        self._solver_config = analog_simulation_method.get_config()
        self._solver_config.update(execution_config.get_config())
        self._solver_config.update(digital_simulation_method.get_config())
        self._solver_config.update({"atol": get_settings().atol})

    @property
    def solver_params(self) -> SolverConfigDict:
        """Backward-compatible alias for the backend configuration dictionary."""
        return self._solver_config

    def get_config(self) -> SolverConfigDict:
        """Return the full flattened solver configuration used by the C++ backend."""
        return dict(self._solver_config)

    def _execute_digital_evolution(
        self, functional: DigitalEvolution, initial_state: QTensor | None = None
    ) -> FunctionalResult:
        """
        Execute a digital-circuit sampling functional and return measurement results.

        Args:
            functional: Sampling functional to execute.
            initial_state: Optional initial state to start the simulation from.

        Returns:
            SamplingResult: Measurement samples and computed probabilities.

        """
        result = self.qili_sim.execute_digital_evolution(
            functional, self._noise_model, initial_state, self.solver_params
        )
        logger.success("Sampling finished")
        return result

    def _execute_analog_evolution(
        self, functional: AnalogEvolution, initial_state: QTensor | None = None
    ) -> FunctionalResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        result = self.qili_sim.execute_analog_evolution(functional, self._noise_model, self.solver_params)
        logger.success("Sampling finished")
        return result

    def _execute_sampling(self, functional: Sampling, initial_state: QTensor | None = None) -> FunctionalResult:
        """
        Execute a quantum circuit and return the measurement results.

        Args:
            functional (Sampling): The Sampling function to execute.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling with {} shots", functional.readout[0].nshots)
        result = self.qili_sim.execute_digital_evolution(
            functional, self._noise_model, initial_state, self.solver_params
        )
        logger.success("Sampling finished")
        return result

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        """
        Compute analog time evolution for the provided schedule and initial state.

        Args:
            functional: TimeEvolution functional to execute.

        Returns:
            TimeEvolutionResult: Final state and requested observables.
        """

        # Get the time steps
        logger.info("Executing TimeEvolution (T={}, dt={})", functional.schedule.T, functional.schedule.dt)

        # Execute the time evolution
        result = self.qili_sim.execute_analog_evolution(functional, self._noise_model, self.solver_params)

        logger.success("TimeEvolution finished")
        return result
