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

import secrets
from abc import ABC, abstractmethod
from typing import Any, Literal

import psutil
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
        trajectories (int): Number of Monte Carlo trajectories to simulate
            when Monte Carlo mode is enabled. Defaults to ``100``.
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
        evolution_method (str): Analog time-evolution method to use:
            ``"direct"``, ``"arnoldi"``, ``"integrate_rk4"``, ``"integrate_rk45_matrix_free"``, ``"integrate_rk4_matrix_free"``, or ``"variational_exponential"``. Defaults to ``"integrate_rk4_matrix_free"``.
        arnoldi_dim (int): Dimension of the Arnoldi Krylov subspace used
            when ``evolution_method="arnoldi"``. Defaults to ``10``.
        num_arnoldi_substeps (int): Number of integration substeps per
            schedule step for the Arnoldi method. Defaults to ``1``.
    """

    evolution_method: Literal[
        "direct",
        "arnoldi",
        "integrate_rk4",
        "integrate_rk45_matrix_free",
        "integrate_rk4_matrix_free",
        "variational_exponential",
    ] = Field(
        default="integrate_rk4_matrix_free",
        description="Analog time-evolution method to use: 'direct', 'arnoldi', 'integrate_rk4', 'integrate_rk45_matrix_free', 'integrate_rk4_matrix_free', or 'variational_exponential'.",
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
    adaptive_tol: float = Field(
        default=1e-2,
        gt=0,
        description="Tolerance for the adaptive integrator method when `evolution_method='integrate_rk45_matrix_free'`.",
    )
    variational_shots: int = Field(
        default=100,
        gt=0,
        description="Number of shots to use when estimating expectation values for the variational optimization when `evolution_method='variational_exponential'`.",
    )
    variational_warmups: int = Field(
        default=10,
        ge=0,
        description="Number of warmup iterations to perform before collecting samples for the variational optimization when `evolution_method='variational_exponential'`.",
    )
    variational_order: int = Field(
        default=2,
        gt=0,
        description="Order of the polynomial expansion used in the variational ansatz when `evolution_method='variational_exponential'`.",
    )

    def get_config(self) -> SolverConfigDict:
        """Return a complete analog solver configuration for the C++ backend."""
        d: SolverConfigDict = {
            "evolution_method": self.evolution_method,
            "arnoldi_dim": self.arnoldi_dim,
            "num_arnoldi_substeps": self.num_arnoldi_substeps,
            "adaptive_tol": self.adaptive_tol,
            "variational_shots": self.variational_shots,
            "variational_warmups": self.variational_warmups,
            "variational_order": self.variational_order,
        }

        return d

    @classmethod
    def integrator(cls, *, matrix_free: bool = True) -> AnalogMethod:
        """Build an ``integrate`` analog method configuration.

        Args:
            matrix_free (bool): Whether to use the matrix-free
                implementation for the Integrate method. Defaults to
                ``False``.

        Returns:
            AnalogMethod: Configured integrate-method analog configuration.
        """
        evolution_method = "integrate_rk4_matrix_free" if matrix_free else "integrate_rk4"
        return cls(evolution_method=evolution_method)

    @classmethod
    def variational_annealing(cls, *, order: int = 2, shots: int = 100, warmups: int = 10) -> AnalogMethod:
        """
        Anneal a variational ansatz rather than the full state.

        Based on this paper: https://arxiv.org/pdf/2403.05147

        Args:
            order (int): Order of the polynomial expansion used in the variational ansatz.
            shots (int): Number of samples to use when estimating expectation values for the variational optimization.
            warmups (int): Number of warmup iterations to perform before collecting samples for the variational optimization.

        Returns:
            AnalogMethod: Configured variational-method analog configuration.
        """
        return cls(
            evolution_method="variational_exponential",
            variational_order=order,
            variational_shots=shots,
            variational_warmups=warmups,
        )

    @classmethod
    def adaptive_integrator(cls, *, tol: float = 1e-2) -> AnalogMethod:
        """Build an ``adaptive_integrate`` analog method configuration.

        This uses a Dormand-Prince Runge-Kutta 4/5 method with adaptive step size control.
        It automatically adjusts the integration timestep to maintain a local error estimate
        below the specified tolerance, which can improve efficiency for problems with varying timescales.

        Args:
            tol (float): Tolerance for the adaptive algorithm. Defaults to ``1e-2``. This relates to the allowed fidelity error between the RK4 and RK5 estimates.

        Returns:
            AnalogMethod: Configured integrate-method analog configuration.
        """
        return cls(evolution_method="integrate_rk45_matrix_free", adaptive_tol=tol)

    @classmethod
    def arnoldi(
        cls,
        *,
        num_substeps: int = 1,
        dim: int = 10,
    ) -> AnalogMethod:
        """Build an ``arnoldi`` analog method configuration.

        Args:
            num_substeps (int): Number of integration substeps per schedule
                step when using the Arnoldi method. Defaults to ``1``.
            dim (int): Dimension of the Arnoldi Krylov subspace. Defaults
                to ``10``.

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
        num_threads (int): Number of CPU threads used for simulation. If
            set to ``0``, all available cores are selected. Defaults to
            ``0``.
        seed (int | None): Random seed used by the simulator. If ``None``,
            a random seed is generated. Defaults to ``None``.
        monte_carlo (MonteCarloConfig | None): Monte Carlo configuration.
            If ``None``, Monte Carlo is disabled and deterministic
            evolution is used. Defaults to ``None``.
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
    measurement_collapse: bool = Field(
        default=False,
        description=(
            "Whether to apply state collapse immediately after measurements. If `False`, measurements are recorded but the state is not collapsed."
        ),
    )

    def get_config(self) -> SolverConfigDict:
        """Return execution settings with resolved defaults."""
        d: dict[str, float | str] = {
            "num_threads": self.num_threads,
            "seed": int(self.seed) if self.seed is not None else 0,  # defensive
            "monte_carlo": self.monte_carlo is not None,
            "measurement_collapse": self.measurement_collapse,
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
            return psutil.cpu_count(logical=False) or 1
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
        max_cache_size (int): Maximum number of cached gate
            representations used by the digital simulator. Defaults to
            ``1000``.
        normalize_after_each_gate (bool): Whether to normalize the
            statevector after each gate application to mitigate numerical
            errors at the cost of increased runtime. Defaults to ``True``.
        combine_single_qubit_gates (bool): Whether to combine consecutive
            single-qubit gates into a single operation to reduce overhead
            at the cost of increased memory usage. Defaults to ``True``.
        fuse_gates (bool): Whether to fuse runs of adjacent gates acting on a
            small set of qubits into a single dense multi-qubit operation. This
            reduces the number of passes over the statevector (the memory
            bandwidth bottleneck for large qubit counts) at the cost of extra
            arithmetic per fused block. Supersedes ``combine_single_qubit_gates``
            when enabled, and only applies to the matrix-free statevector path
            (disabled automatically when a noise model is present). Defaults to
            ``True``.
        max_fused_qubits (int): Maximum number of qubits a single fused block may
            span. Larger values fuse more aggressively but build larger dense
            matrices (``2**max_fused_qubits`` square). ``0`` (the default) selects
            the depth automatically from the qubit count: shallow fusion while the
            statevector fits in cache, deeper fusion once it is DRAM-resident.
        matrix_free (bool): Whether to use the matrix-free implementation
            for statevector simulation. Defaults to ``True``.
    """

    max_cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of cached gate representations used by the digital simulator.",
    )
    normalize_after_each_gate: bool = Field(
        default=False,
        description="Whether to normalize the statevector after each gate application to mitigate numerical errors at the cost of increased runtime.",
    )
    combine_single_qubit_gates: bool = Field(
        default=True,
        description="Whether to combine consecutive single-qubit gates into a single operation to reduce overhead at the cost of increased memory usage.",
    )
    fuse_gates: bool = Field(
        default=True,
        description="Whether to fuse runs of adjacent gates on a small set of qubits into a single dense multi-qubit operation to reduce passes over the statevector. Only applies to the matrix-free statevector path.",
    )
    max_fused_qubits: int = Field(
        default=0,
        ge=0,
        description="Maximum number of qubits a single fused block may span. 0 (the default) selects the depth automatically from the qubit count.",
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
            "fuse_gates": self.fuse_gates,
            "max_fused_qubits": self.max_fused_qubits,
        }

    @classmethod
    def statevector(
        cls,
        *,
        max_cache_size: int = 1000,
        normalize_after_each_gate: bool = False,
        matrix_free: bool = True,
        combine_single_qubit_gates: bool = True,
        fuse_gates: bool = True,
        max_fused_qubits: int = 0,
    ) -> DigitalMethod:
        """Build the standard statevector simulation configuration.

        Args:
            max_cache_size (int): Maximum number of cached gate representations used by the digital simulator.
                Defaults to ``1000``.
            normalize_after_each_gate (bool): Whether to normalize the statevector after each gate application to
                mitigate numerical errors at the cost of increased runtime. Defaults to ``False``.
            matrix_free (bool): Whether to use the matrix-free implementation for statevector simulation.
                Defaults to ``True``.
            combine_single_qubit_gates (bool): Whether to combine consecutive single-qubit gates into a single operation.
                Defaults to ``True``.
            fuse_gates (bool): Whether to fuse runs of adjacent gates on a small set of qubits into a single dense
                multi-qubit operation (matrix-free statevector path only). Defaults to ``True``.
            max_fused_qubits (int): Maximum number of qubits a single fused block may span. ``0`` (the default)
                selects the depth automatically from the qubit count.

        Returns:
            DigitalMethod: Configured statevector digital configuration.
        """
        return cls(
            max_cache_size=max_cache_size,
            normalize_after_each_gate=normalize_after_each_gate,
            combine_single_qubit_gates=combine_single_qubit_gates,
            fuse_gates=fuse_gates,
            max_fused_qubits=max_fused_qubits,
            matrix_free=matrix_free,
        )
