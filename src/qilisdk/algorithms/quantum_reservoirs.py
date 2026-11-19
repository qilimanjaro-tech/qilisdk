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

from copy import copy
from typing import TYPE_CHECKING, Iterator

import numpy as np
from loguru import logger

from qilisdk.algorithms.algorithm import Algorithm
from qilisdk.analog import Schedule
from qilisdk.core import Domain, InitialState, Parameter, QTensor, reset_qubits, tensor_prod
from qilisdk.core.parameterizable import Parameterizable
from qilisdk.digital import Circuit, M
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.functional import PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import Readout
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.backends.backend import Backend
    from qilisdk.core.types import RealNumber
    from qilisdk.readout.readout_result import ReadoutCompositeResults


@yaml.register_class
class ReservoirInput(Parameter):
    """Input-only parameter used to inject layer-wise data into a reservoir program.

    This parameter behaves like a standard :class:`~qilisdk.core.variables.Parameter` but is
    always created as non-trainable, so optimizers do not update it.
    """

    def __init__(
        self,
        label: str,
        value: RealNumber,
    ) -> None:
        """Create a non-trainable reservoir input parameter.

        Args:
            label (str): Parameter label used to reference input values.
            value (RealNumber): Default numerical value.
        """
        super().__init__(
            label=label,
            value=value,
            domain=Domain.REAL,
            bounds=(None, None),
            trainable=False,
        )


class ReservoirLayer(Parameterizable):
    """Single reservoir layer template.

    A reservoir layer can contain up to three ordered stages:
    ``input_encoding -> evolution_dynamics -> output_encoding``.
    The evolution dynamics can be either an analog :class:`~qilisdk.analog.Schedule`
    or a digital :class:`~qilisdk.digital.Circuit`. It also stores the observables measured after each layer and optional qubits to reset.
    """

    def __init__(
        self,
        evolution_dynamics: Schedule | Circuit,
        input_encoding: Circuit | None = None,
        output_encoding: Circuit | None = None,
        qubits_to_reset: list[int] | None = None,
    ) -> None:
        """Build a reservoir layer description.

        Args:
            evolution_dynamics (Schedule | Circuit): Main block driving the
                reservoir dynamics, either an analog schedule or a digital
                circuit (which may contain multi-qubit gates).
            input_encoding (Circuit | None): Optional single-qubit
                pre-processing circuit. Defaults to None.
            output_encoding (Circuit | None): Optional single-qubit
                post-processing circuit. Defaults to None.
            qubits_to_reset (list[int] | None): Optional qubit indices
                to reset between consecutive layers. Defaults to None.

        Raises:
            ValueError: If the evolution dynamics is a circuit containing
                measurements, or if the input or output encoding circuit
                contains measurements, multi-qubit gates, or acts on more
                qubits than the evolution dynamics.
        """

        super().__init__()
        self._input_encoding: Circuit | None = input_encoding
        self._evolution_dynamics: Schedule | Circuit = evolution_dynamics
        self._output_encoding: Circuit | None = output_encoding
        self._qubits_to_reset: list[int] | None = qubits_to_reset
        self._nqubits = self._evolution_dynamics.nqubits

        if isinstance(evolution_dynamics, Circuit) and any(isinstance(g, M) for g in evolution_dynamics.gates):
            raise ValueError("Evolution dynamics circuit can't contain measurements")

        if input_encoding:
            self._validate_input_encoding(input_encoding)

        if output_encoding:
            self._validate_output_encoding(output_encoding)

        self._qtensor_observables: list[QTensor] = []

    def _validate_input_encoding(self, pre_processing: Circuit) -> None:
        """Validate and normalize the optional pre-processing circuit.

        Args:
            pre_processing (Circuit): The candidate input encoding circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are
                used, or the circuit acts on more qubits than the reservoir
                dynamics.
        """
        if any(isinstance(g, M) for g in pre_processing.gates):
            raise ValueError("Pre-Processing Circuit can't contain measurements")
        if pre_processing.nqubits > self._evolution_dynamics.nqubits:
            raise ValueError("Pre-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in pre_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the pre-processing circuit.")
        self._input_encoding = Circuit(self._nqubits)
        self._input_encoding.add(pre_processing.gates)
        self._input_encoding.set_prefix("input_encoding_", where=lambda p: not isinstance(p, ReservoirInput))

    def _validate_output_encoding(self, post_processing: Circuit) -> None:
        """Validate and normalize the optional post-processing circuit.

        Args:
            post_processing (Circuit): The candidate output encoding circuit.

        Raises:
            ValueError: If measurements are present, multi-qubit gates are
                used, or the circuit acts on more qubits than the reservoir
                dynamics.
        """
        if any(isinstance(g, M) for g in post_processing.gates):
            raise ValueError("Post-Processing Circuit can't contain measurements")
        if post_processing.nqubits > self._evolution_dynamics.nqubits:
            raise ValueError("Post-Processing Circuit acts on more qubits than defined by the reservoir dynamics.")
        if any(g.nqubits > 1 for g in post_processing.gates):
            raise ValueError("Only single qubit gates are allowed in the post-processing circuit.")
        self._output_encoding = Circuit(self._nqubits)
        self._output_encoding.add(post_processing.gates)
        self._output_encoding.set_prefix("output_encoding_", where=lambda p: not isinstance(p, ReservoirInput))

    def _process_qtensor(self, observable: QTensor) -> QTensor:
        """Pad observable tensors with identities to match the reservoir width.

        Args:
            observable (QTensor): Observable operator to expand.

        Returns:
            QTensor: Observable expanded (if needed) to match the number of
                reservoir qubits.

        Raises:
            ValueError: If the observable acts on more qubits than the
                reservoir.
        """
        if observable.nqubits < self._nqubits:
            padding = [QTensor(np.identity(2)) for _ in range(self._nqubits - observable.nqubits)]
            full_list = [observable]
            full_list.extend(padding)
            _observable = tensor_prod(full_list)
            return _observable
        if observable.nqubits > self._nqubits:
            raise ValueError("Observable acts on more qubits than the system contains")
        return observable

    @property
    def input_parameter_names(self) -> list[str]:
        """Input-driven (non-trainable) parameter names for this layer."""
        return self.get_parameter_names(
            where=lambda param: isinstance(param, ReservoirInput) and not param.is_trainable
        )

    @property
    def nqubits(self) -> int:
        """Number of qubits acted on by this layer."""
        return self._nqubits

    @property
    def input_encoding(self) -> Circuit | None:
        """Optional pre-processing circuit, or ``None`` if not set."""
        return self._input_encoding

    @property
    def output_encoding(self) -> Circuit | None:
        """Optional post-processing circuit, or ``None`` if not set."""
        return self._output_encoding

    @property
    def observables_as_qtensor(self) -> list[QTensor]:
        """Measured observables converted to qubit-sized ``QTensor`` operators."""
        return self._qtensor_observables

    @property
    def qubits_to_reset(self) -> list[int] | None:
        """Qubit indices reset between layers, or ``None`` if not set."""
        return self._qubits_to_reset

    @property
    def evolution_dynamics(self) -> Schedule | Circuit:
        """Main dynamics block (analog schedule or digital circuit) for each layer."""
        return self._evolution_dynamics

    @property
    def reservoir_dynamics(self) -> Schedule | Circuit:
        """Backward-compatible alias for :attr:`evolution_dynamics`."""
        return self._evolution_dynamics

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        """Yield the encoding circuits and dynamics as parameterizable children.

        Yields:
            Iterator[Parameterizable]: Input encoding (if set), evolution
                dynamics, and output encoding (if set).
        """
        if self._input_encoding:
            yield self._input_encoding
        yield self._evolution_dynamics
        if self._output_encoding:
            yield self._output_encoding

    def set_parameters(self, parameters: dict[str, RealNumber]) -> None:
        known_labels = set(self.get_parameter_names())
        unknown_labels = [p for p in parameters if p not in known_labels]
        if unknown_labels:
            raise ValueError(
                f"Cannot set unknown parameter(s) {sorted(unknown_labels)} on this reservoir layer. "
                f"Valid parameter names are {sorted(known_labels)}."
            )
        if self._input_encoding:
            param_labels = self._input_encoding.get_parameter_names()
            self._input_encoding.set_parameters({p: v for p, v in parameters.items() if p in param_labels})
        param_labels = self._evolution_dynamics.get_parameter_names()
        self._evolution_dynamics.set_parameters({p: v for p, v in parameters.items() if p in param_labels})
        if self._output_encoding:
            param_labels = self._output_encoding.get_parameter_names()
            self._output_encoding.set_parameters({p: v for p, v in parameters.items() if p in param_labels})

    def set_parameter_bounds(self, ranges: dict[str, tuple[float, float]]) -> None:
        if self._input_encoding:
            param_labels = self._input_encoding.get_parameter_names()
            self._input_encoding.set_parameter_bounds({p: v for p, v in ranges.items() if p in param_labels})
        param_labels = self._evolution_dynamics.get_parameter_names()
        self._evolution_dynamics.set_parameter_bounds({p: v for p, v in ranges.items() if p in param_labels})
        if self._output_encoding:
            param_labels = self._output_encoding.get_parameter_names()
            self._output_encoding.set_parameter_bounds({p: v for p, v in ranges.items() if p in param_labels})

    def __len__(self) -> int:
        """
        Get the total number of steps in the reservoir pass (maximum of 3)

        Returns:
            int: The number of steps in the reservoir pass.
        """
        return int(self._input_encoding is not None) + int(self._output_encoding is not None) + 1

    def __iter__(self) -> Iterator[Circuit | Schedule]:
        """
        Return an iterator over the steps in the reservoir pass.

        Yields:
            Iterator[Circuit | Schedule]: The steps in the pass
        """
        if self._input_encoding:
            yield self._input_encoding
        yield self._evolution_dynamics
        if self._output_encoding:
            yield self._output_encoding

    def __copy__(self) -> ReservoirLayer:
        return ReservoirLayer(
            evolution_dynamics=copy(self.evolution_dynamics),
            input_encoding=copy(self.input_encoding),
            output_encoding=copy(self.output_encoding),
            qubits_to_reset=copy(self.qubits_to_reset),
        )


class QuantumReservoir(PrimitiveFunctional, Algorithm):
    """Reservoir functional executed over a sequence of input layers.

    Each element in ``input_per_layer`` is applied to the underlying
    :class:`ReservoirLayer` before executing one pass of dynamics and measurement.
    """

    def __init__(
        self,
        initial_state: QTensor | InitialState,
        reservoir_layer: ReservoirLayer | None = None,
        input_per_layer: list[dict[str, float]] | None = None,
    ) -> None:
        """Construct a quantum reservoir functional.

        Args:
            initial_state (QTensor | InitialState): Initial quantum state before the first
                layer.
            reservoir_layer (ReservoirLayer | None): Reservoir layer
                definition repeated at each step. Defaults to None.
            input_per_layer (list[dict[str, float]] | None): Input parameter
                assignments for each layer. Defaults to None.

        Raises:
            ValueError: If ``reservoir_layer`` or ``input_per_layer`` is
                ``None``, if ``input_per_layer`` is empty, or if the initial
                state qubit count does not match the reservoir layer.
        """
        super().__init__()
        if reservoir_layer is None:
            raise ValueError("`reservoir_layer` must be provided.")
        if input_per_layer is None:
            raise ValueError("`input_per_layer` must be provided.")
        if len(input_per_layer) == 0:
            raise ValueError("`input_per_layer` must contain at least one layer.")

        self._initial_state = initial_state
        self._reservoir_layer = reservoir_layer
        self._input_per_layer = input_per_layer
        if isinstance(self._initial_state, QTensor) and self._reservoir_layer.nqubits != self._initial_state.nqubits:
            raise ValueError(
                f"invalid initial state: the initial state acts on {self._initial_state.nqubits} qubits while the reservoir is defined with {self._reservoir_layer.nqubits} qubits."
            )

        logger.debug(
            "[QuantumReservoir] Created QuantumReservoir with {} qubits and {} layers",
            self._reservoir_layer.nqubits,
            len(input_per_layer),
        )

    @property
    def nqubits(self) -> int:
        """Number of qubits in the reservoir."""
        return self._reservoir_layer.nqubits

    @property
    def initial_state(self) -> QTensor | InitialState:
        """Initial quantum state used before the first layer."""
        return self._initial_state

    @property
    def reservoir_layer(self) -> ReservoirLayer:
        """Reservoir pass definition applied at each layer."""
        return self._reservoir_layer

    @property
    def input_per_layer(self) -> list[dict[str, float]]:
        """Layer-ordered input parameter assignments."""
        return self._input_per_layer

    @property
    def input_parameter_names(self) -> list[str]:
        """Input-driven (non-trainable) parameter names expected per layer."""
        return self._reservoir_layer.input_parameter_names

    def _iter_parameter_children(self) -> Iterator[Parameterizable]:
        """Yield the reservoir layer as the sole parameterizable child.

        Yields:
            Iterator[Parameterizable]: The underlying ``ReservoirLayer``.
        """
        yield self._reservoir_layer

    def set_inputs_per_layer(self, input_per_layer: list[dict[str, float]]) -> None:
        """Sets the reservoir's input per layer.

        Args:
            input_per_layer (list[dict[str, float]]): the new list of inputs per-layers.
        """
        self._input_per_layer = input_per_layer

    def add_inputs_per_layer(self, input_per_layer: list[dict[str, float]] | dict[str, float]) -> None:
        """Add a list of per-layer inputs to the current list of inputs.

        Args:
            input_per_layer (list[dict[str, float]] | dict[str,float]): the new list of inputs to be added.
        """
        if isinstance(input_per_layer, list):
            self._input_per_layer.extend(input_per_layer)
        else:
            self._input_per_layer.append(input_per_layer)

    def __copy__(self) -> QuantumReservoir:
        return QuantumReservoir(
            initial_state=copy(self.initial_state),
            reservoir_layer=copy(self.reservoir_layer),
            input_per_layer=copy(self.input_per_layer),
        )

    def _execute(self, backend: Backend, readout: Readout) -> FunctionalResult:
        """Run every reservoir layer on ``backend``, reading out the state after each one.

        Circuit steps are applied as unitaries, while schedule steps are executed on ``backend`` as an
        :class:`~qilisdk.functionals.analog_evolution.AnalogEvolution`.

        Args:
            backend (Backend): The backend used to execute the analog steps of each layer.
            readout (Readout): The readout applied to the state after each layer.

        Returns:
            FunctionalResult: The readout after the last layer, with the readouts of the previous layers as
                intermediate results.

        Raises:
            ValueError: If throughout the execution the state becomes invalid due to numerical instabilities.
        """
        logger.info("[QuantumReservoir] Executing Quantum Reservoir")
        if backend._noise_model:  # ruff: ignore[private-member-access]
            logger.warning(
                "[QuantumReservoir] Noise Models are not supported with Quantum Reservoirs, so they will be ignored."
            )
        readout_list = readout.to_list()
        state = self._initial_state
        if isinstance(state, InitialState):
            state = state.as_qtensor(self._reservoir_layer.nqubits).to_density_matrix()
        else:
            state = state.to_density_matrix()
        inter_results: list[ReadoutCompositeResults] = []
        cache: dict[Circuit, tuple[tuple[float, ...], QTensor]] = {}
        for layer_index, input_dict in enumerate(self._input_per_layer):
            logger.debug("[QuantumReservoir] Reservoir layer {}/{}", layer_index + 1, len(self._input_per_layer))
            self._reservoir_layer.set_parameters(input_dict)
            for step in self._reservoir_layer:
                if isinstance(step, Circuit):
                    param_signature = tuple(step.get_parameter_values())
                    cached = cache.get(step)
                    if cached is None or cached[0] != param_signature:
                        logger.trace("[QuantumReservoir] Reservoir circuit cache miss, computing unitary")
                        U = step.to_qtensor()
                        cache[step] = (param_signature, U)
                    else:
                        logger.trace("[QuantumReservoir] Reservoir circuit cache hit")
                        U = cached[1]
                    state = U @ state @ U.adjoint()
                elif isinstance(step, Schedule):
                    res = backend.execute(AnalogEvolution(step, state), Readout().with_state_tomography())
                    state = res.get_state()

            try:
                state = state.to_density_matrix()
            except ValueError as exc:
                raise ValueError(
                    "Reservoir Runtime Error: state repair failed before expectation value computation. "
                    f"{exc} "
                    "Try improving simulation precision (e.g., smaller dt, more integrator substeps, or higher precision)."
                ) from exc

            inter_results.append(
                backend._construct_results_list(  # ruff: ignore[private-member-access]
                    state,
                    readout_list,
                    seed=backend._rng,  # ruff: ignore[private-member-access]
                )
            )

            if self._reservoir_layer.qubits_to_reset:
                logger.trace("[QuantumReservoir] Resetting qubits {}", self._reservoir_layer.qubits_to_reset)
                state = reset_qubits(state, self._reservoir_layer.qubits_to_reset)

        logger.info("[QuantumReservoir] Quantum Reservoir finished")
        return FunctionalResult(readout_results=inter_results[-1], intermediate_results=inter_results[:-1])
