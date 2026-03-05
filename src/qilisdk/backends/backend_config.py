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

    evolution_method: Literal["direct", "arnoldi", "integrate", "integrate_matrix_free"] = Field(
        default="integrate",
        description="Analog time-evolution method to use: 'direct', 'arnoldi', 'integrate', or 'integrate_matrix_free'.",
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
    def integrator(
        cls, *, num_substeps: int = 2, matrix_free: bool = False, monte_carlo: MonteCarloConfig | None = None
    ) -> AnalogMethod:
        """Build an ``integrate`` analog method configuration.

        Args:
            num_substeps: Number of integration substeps per schedule step when using the Integrate method.
            matrix_free: Whether to use the matrix-free implementation for the Integrate method.

        Returns:
            AnalogMethod: Configured integrate-method analog configuration.
        """
        evolution_method = "integrate_matrix_free" if matrix_free else "integrate"
        return cls(evolution_method=evolution_method, num_integrate_substeps=num_substeps)

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
        - :meth:`statevector` for standard state-vector simulation settings.

    Args:
        max_cache_size: Maximum number of cached gate representations used by the digital simulator.
    """

    max_cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of cached gate representations used by the digital simulator.",
    )
    normalize_after_each_gate: bool = Field(
        default=True,
        description="Whether to normalize the statevector after each gate application to mitigate numerical errors at the cost of increased runtime.",
    )
    combine_single_qubit_gates: bool = Field(
        default=True,
        description="Whether to combine consecutive single-qubit gates into a single operation to reduce overhead at the cost of increased memory usage.",
    )
    matrix_free: bool = Field(
        default=True,
        description="Whether to use the matrix-free implementation for statevector simulation.",
    )

    def get_config(self) -> SolverConfigDict:
        """Return digital simulation settings in backend-compatible key names."""
        return {
            "max_cache_size": self.max_cache_size,
            "sampling_method": "statevector_matrix_free" if self.matrix_free else "statevector",
            "normalize_after_each_gate": self.normalize_after_each_gate,
            "combine_single_qubit_gates": self.combine_single_qubit_gates,
        }

    @classmethod
    def statevector(
        cls,
        *,
        max_cache_size: int = 1000,
        normalize_after_each_gate: bool = False,
        matrix_free: bool = True,
        combine_single_qubit_gates: bool = True,
    ) -> DigitalMethod:
        """Build the standard statevector simulation configuration.

        Args:
            max_cache_size: Maximum number of cached gate representations used by the digital simulator.
            normalize_after_each_gate: Whether to normalize the statevector after each gate application to mitigate numerical errors at the cost of increased runtime.
            matrix_free: Whether to use the matrix-free implementation for statevector simulation.
            combine_single_qubit_gates: Whether to combine consecutive single-qubit gates into a single operation.

        Returns:
            DigitalMethod: Configured statevector digital configuration.
        """
        return cls(
            max_cache_size=max_cache_size,
            normalize_after_each_gate=normalize_after_each_gate,
            combine_single_qubit_gates=combine_single_qubit_gates,
            matrix_free=matrix_free,
        )
