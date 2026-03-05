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
from __future__ import annotations

import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict as PydanticConfigDict

# ----------------------------
# Base config interface
# ----------------------------

ConfigValue = bool | int | float | str
SolverConfigDict = dict[str, ConfigValue]


class BaseSimulatorConfig(BaseModel, ABC):
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


class MonteCarloConfig(BaseSimulatorConfig):
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


class AnalogMethod(BaseSimulatorConfig):
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

    def get_config(self) -> SolverConfigDict:
        """Return a complete analog solver configuration for the C++ backend."""
        d: SolverConfigDict = {
            "evolution_method": self.evolution_method,
            "arnoldi_dim": self.arnoldi_dim,
            "num_arnoldi_substeps": self.num_arnoldi_substeps,
            "num_integrate_substeps": self.num_integrate_substeps,
        }

        return d

    @classmethod
    def integrator(cls, *, num_substeps: int = 2, monte_carlo: MonteCarloConfig | None = None) -> AnalogMethod:
        """Build an ``integrate`` analog method configuration.

        Args:
            num_substeps: Number of integration substeps per schedule step when using the Integrate method.

        Returns:
            AnalogMethod: Configured integrate-method analog configuration.
        """
        return cls(evolution_method="integrate", num_integrate_substeps=num_substeps)

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

        Returns:
            AnalogMethod: Configured arnoldi-method analog configuration.
        """
        return cls(
            evolution_method="arnoldi",
            arnoldi_dim=dim,
            num_arnoldi_substeps=num_substeps,
        )

    @classmethod
    def direct(cls) -> AnalogMethod:
        """Build a ``direct`` analog method configuration.

        Returns:
            AnalogMethod: Configured direct-method analog configuration.
        """
        return cls(evolution_method="direct")


# ----------------------------
# Execution config
# ----------------------------


class ExecutionConfig(BaseSimulatorConfig):
    """Configuration for execution-level controls (threading and randomness).

    Args:
        num_threads: Number of CPU threads used for simulation. If set to ``0``, all available cores are selected.
        seed: Random seed used by the simulator. If ``None``, a random seed is generated.
        monte_carlo: Monte Carlo configuration. If ``None``, Monte Carlo is disabled and deterministic evolution is
            used.

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
    # None means Monte-Carlo disabled
    monte_carlo: MonteCarloConfig | None = Field(
        default=None,
        description=(
            "Monte Carlo configuration. If `None`, Monte Carlo is disabled and deterministic evolution is used."
        ),
    )

    def get_config(self) -> SolverConfigDict:
        """Return execution settings with resolved defaults."""
        d = {
            "num_threads": self.num_threads,
            "seed": int(self.seed) if self.seed is not None else 0,  # defensive
            "monte_carlo": self.monte_carlo is not None,
        }
        if self.monte_carlo is not None:
            d.update(self.monte_carlo.get_config())
        else:
            d.update({"num_monte_carlo_trajectories": 100})
        return d

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


class DigitalMethod(BaseSimulatorConfig):
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
