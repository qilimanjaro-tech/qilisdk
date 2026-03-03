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

import os
import secrets
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict as PydanticConfigDict
from qilisim_module import QiliSimCpp  # ty:ignore[unresolved-import]

from qilisdk.backends.backend import Backend
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core import QTensor
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution import TimeEvolution
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
    from qilisdk.noise.noise_model import NoiseModel


# ----------------------------
# Base config interface
# ----------------------------

ConfigValue = bool | int | float | str
SolverConfigDict = dict[str, ConfigValue]


class QiliSimConfig(BaseModel, ABC):
    """Abstract base class for all QiliSim configuration sections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Disallow positional arguments to keep configuration explicit.

        Raises:
            TypeError: If positional arguments are provided.
        """
        if args:
            cls = self.__class__
            field_names = ", ".join(cls.model_fields.keys())

            raise TypeError(
                f"{cls.__name__} does not accept positional arguments. "
                f"Use keyword arguments instead. "
                f"Valid fields: [{field_names}].\n"
                f"Received positional arguments: {args!r}."
            )
        super().__init__(**kwargs)

    @abstractmethod
    def get_config(self) -> SolverConfigDict:
        """Serialize the configuration to the flat dictionary consumed by the C++ backend."""


# ----------------------------
# Analog (time evolution) config
# ----------------------------


class MonteCarloConfig(QiliSimConfig):
    """Configuration for Monte Carlo trajectory sampling in open-system simulations.

    Args:
        trajectories: Number of Monte Carlo trajectories to simulate when Monte Carlo mode is enabled.
    """

    trajectories: int = Field(
        default=100,
        gt=0,
        description="Number of Monte Carlo trajectories to simulate when Monte Carlo mode is enabled.",
    )

    def get_config(self) -> SolverConfigDict:
        """Return Monte Carlo settings in backend-compatible key names."""
        return {"num_monte_carlo_trajectories": self.trajectories}


class AnalogMethod(QiliSimConfig):
    """Configuration for analog time-evolution method selection and its hyperparameters.

    Preferred constructors:
        - :meth:`integrator` for integrate-based evolution.
        - :meth:`arnoldi` for Krylov/Arnoldi-based evolution.
        - :meth:`direct` for direct evolution.

    Args:
        evolution_method: Analog time-evolution method to use: ``"direct"``, ``"arnoldi"``, or ``"integrate"``.
        arnoldi_dim: Dimension of the Arnoldi Krylov subspace used when ``evolution_method="arnoldi"``.
        num_arnoldi_substeps: Number of integration substeps per schedule step for the Arnoldi method.
        num_integrate_substeps: Number of integration substeps per schedule step for the Integrate method.
        monte_carlo: Monte Carlo configuration. If ``None``, Monte Carlo is disabled and deterministic evolution is
            used.
    """

    evolution_method: Literal["direct", "arnoldi", "integrate"] = Field(
        default="integrate",
        description="Analog time-evolution method to use: 'direct', 'arnoldi', or 'integrate'.",
    )
    arnoldi_dim: int = Field(
        default=10,
        gt=0,
        description="Dimension of the Arnoldi Krylov subspace used when `evolution_method='arnoldi'`.",
    )
    num_arnoldi_substeps: int = Field(
        default=1,
        gt=0,
        description="Number of integration substeps per schedule step when using the Arnoldi method.",
    )
    num_integrate_substeps: int = Field(
        default=2,
        gt=0,
        description="Number of integration substeps per schedule step when using the Integrate method.",
    )

    # None means Monte-Carlo disabled
    monte_carlo: Annotated[
        MonteCarloConfig | None,
        Field(
            default=None,
            description=(
                "Monte Carlo configuration. If `None`, Monte Carlo is disabled and deterministic evolution is used."
            ),
        ),
    ]

    def get_config(self) -> SolverConfigDict:
        """Return a complete analog solver configuration for the C++ backend."""
        d: SolverConfigDict = {
            "evolution_method": self.evolution_method,
            "arnoldi_dim": self.arnoldi_dim,
            "num_arnoldi_substeps": self.num_arnoldi_substeps,
            "num_integrate_substeps": self.num_integrate_substeps,
            "monte_carlo": self.monte_carlo is not None,
        }
        if self.monte_carlo is not None:
            d.update(self.monte_carlo.get_config())
        else:
            d.update({"num_monte_carlo_trajectories": 100})
        return d

    @classmethod
    def integrator(cls, *, num_substeps: int = 2, monte_carlo: MonteCarloConfig | None = None) -> AnalogMethod:
        """Build an ``integrate`` analog method configuration.

        Args:
            num_substeps: Number of integration substeps per schedule step when using the Integrate method.
            monte_carlo: Monte Carlo configuration. If ``None``, Monte Carlo is disabled.

        Returns:
            AnalogMethod: Configured integrate-method analog configuration.
        """
        return cls(
            evolution_method="integrate",
            num_integrate_substeps=num_substeps,
            monte_carlo=monte_carlo,
        )

    @classmethod
    def arnoldi(
        cls,
        *,
        num_substeps: int = 1,
        dim: int = 10,
        monte_carlo: MonteCarloConfig | None = None,
    ) -> AnalogMethod:
        """Build an ``arnoldi`` analog method configuration.

        Args:
            num_substeps: Number of integration substeps per schedule step when using the Arnoldi method.
            dim: Dimension of the Arnoldi Krylov subspace.
            monte_carlo: Monte Carlo configuration. If ``None``, Monte Carlo is disabled.

        Returns:
            AnalogMethod: Configured arnoldi-method analog configuration.
        """
        return cls(
            evolution_method="arnoldi",
            arnoldi_dim=dim,
            num_arnoldi_substeps=num_substeps,
            monte_carlo=monte_carlo,
        )

    @classmethod
    def direct(cls, *, monte_carlo: MonteCarloConfig | None = None) -> AnalogMethod:
        """Build a ``direct`` analog method configuration.

        Args:
            monte_carlo: Monte Carlo configuration. If ``None``, Monte Carlo is disabled.

        Returns:
            AnalogMethod: Configured direct-method analog configuration.
        """
        return cls(evolution_method="direct", monte_carlo=monte_carlo)


# ----------------------------
# Execution config
# ----------------------------


class ExecutionConfig(QiliSimConfig):
    """Configuration for execution-level controls (threading and randomness).

    Args:
        num_threads: Number of CPU threads used for simulation. If set to ``0``, all available cores are selected.
        seed: Random seed used by the simulator. If ``None``, a random seed is generated.
    """

    model_config = PydanticConfigDict(validate_default=True)
    num_threads: int = Field(
        default=0,
        ge=0,
        description=("Number of CPU threads used for simulation. If set to 0, all available cores are selected."),
    )  # 0 means "use all cores"
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed used by the simulator. If `None`, a random seed is generated.",
    )

    def get_config(self) -> SolverConfigDict:
        """Return execution settings with resolved defaults."""
        return {
            "num_threads": self.num_threads,
            "seed": int(self.seed) if self.seed is not None else 0,  # defensive
        }

    @field_validator("num_threads", mode="after")
    @classmethod
    def _validate_num_threads(cls, num_threads: int) -> int:
        if num_threads <= 0:
            return os.cpu_count() or 1
        return num_threads

    @field_validator("seed", mode="after")
    @classmethod
    def _validate_seed(cls, seed: int | None) -> int:
        if seed is None:
            return secrets.randbelow(2**15)
        return seed


# ----------------------------
# Digital config
# ----------------------------


class DigitalMethod(QiliSimConfig):
    """Configuration for digital-circuit simulation options.

    Preferred constructors:
        - :meth:`state_vector` for standard state-vector simulation settings.

    Args:
        max_cache_size: Maximum number of cached gate representations used by the digital simulator.
    """

    max_cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of cached gate representations used by the digital simulator.",
    )

    def get_config(self) -> SolverConfigDict:
        """Return digital simulation settings in backend-compatible key names."""
        return {
            "max_cache_size": self.max_cache_size,
        }

    @classmethod
    def state_vector(cls, *, max_cache_size: int = 1000) -> DigitalMethod:
        """Build the standard state-vector simulation configuration.

        Args:
            max_cache_size: Maximum number of cached gate representations used by the digital simulator.

        Returns:
            DigitalMethod: Configured state-vector digital configuration.
        """
        return cls(max_cache_size=max_cache_size)


class QiliSim(Backend):
    """
    Backend that runs both digital-circuit sampling and analog
    time-evolution experiments using a custom C++ simulator.
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
            digital_simulation_method: Digital simulation configuration. Available options: :meth:`DigitalMethod.state_vector`. Defaults to
                :meth:`DigitalMethod.state_vector`.
            execution_config: Execution-level configuration for threading and random seed.
                Defaults to the default configuration in :class:`ExecutionConfig`.

        Raises:
            ValueError: If a configuration argument has an invalid type.
        """

        # Initialize the backend and the class vars
        super().__init__()
        self.qili_sim = QiliSimCpp()
        self._noise_model = noise_model

        analog_simulation_method: AnalogMethod = analog_simulation_method or AnalogMethod.integrator()
        digital_simulation_method: DigitalMethod = digital_simulation_method or DigitalMethod.state_vector()
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

    def _execute_sampling(self, functional: Sampling, initial_state: QTensor | None = None) -> SamplingResult:
        """
        Execute a digital-circuit sampling functional and return measurement results.

        Args:
            functional: Sampling functional to execute.
            initial_state: Optional initial state to start the simulation from.

        Returns:
            SamplingResult: Measurement samples and computed probabilities.

        """
        logger.info("Executing Sampling with {} shots", functional.nshots)
        result = self.qili_sim.execute_sampling(functional, self._noise_model, initial_state, self._solver_config)
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
        result = self.qili_sim.execute_time_evolution(functional, self._noise_model, self._solver_config)

        logger.success("TimeEvolution finished")
        return result
