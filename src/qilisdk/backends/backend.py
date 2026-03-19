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

from abc import ABC
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast, overload

from loguru import logger

from qilisdk.analog import Schedule
from qilisdk.core import reset_qubits
from qilisdk.digital import Circuit
from qilisdk.functionals import QuantumReservoir
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.noise import NoiseModel
from qilisdk.readout import (
    ExpectationReadout,
    ExpectationReadoutResult,
    ReadoutMethod,
    ReadoutResult,
    SamplingReadout,
    SamplingReadoutResult,
    StateTomographyReadout,
    StateTomographyReadoutResult,
)
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core import QTensor
    from qilisdk.core.result import Result
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional

TResult = TypeVar("TResult", bound=FunctionalResult)


class Backend(ABC):

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self._handlers: dict[type[Functional], Callable[[Functional, list[ReadoutMethod]], FunctionalResult]] = {
            DigitalPropagation: lambda f, readout: self._execute_digital_propagation(
                cast("DigitalPropagation", f), readout
            ),
            AnalogEvolution: lambda f, readout: self._execute_analog_evolution(cast("AnalogEvolution", f), readout),
            QuantumReservoir: lambda f, readout: self._execute_quantum_reservoir(cast("QuantumReservoir", f), readout),
            VariationalProgram: lambda f, readout: self._execute_variational_program(
                cast("VariationalProgram", f), readout
            ),
        }
        self._noise_model = noise_model

    @overload
    def execute(
        self, functional: VariationalProgram, readout: ReadoutMethod | list[ReadoutMethod]
    ) -> VariationalProgramResult: ...

    @overload
    def execute(
        self, functional: PrimitiveFunctional, readout: ReadoutMethod | list[ReadoutMethod]
    ) -> FunctionalResult: ...

    def execute(self, functional: Functional, readout: ReadoutMethod | list[ReadoutMethod]) -> Result:
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        _readout = list(readout)
        if any(not isinstance(ro, ReadoutMethod) for ro in _readout):
            raise ValueError(
                f"One of the readout methods provided are not a valid readout method.\nProvided: {_readout}"
            )
        if len({ro.__class__ for ro in _readout}) != len(_readout):
            raise ValueError(f"Each readout method can only passed once.\nProvided: {_readout}")
        return handler(functional, _readout)

    def _execute_digital_propagation(
        self, functional: DigitalPropagation, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no DigitalPropagation implementation")

    def _execute_analog_evolution(self, functional: AnalogEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no AnalogEvolution implementation")

    def _execute_quantum_reservoir(
        self, functional: QuantumReservoir, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        if self._noise_model:
            logger.warning("Noise Models are not supported with Quantum Reservoirs, so they will be ignored.")
        state = functional.initial_state.to_density_matrix()
        inter_results: list[list[ReadoutResult]] = []
        cache: dict[Circuit, tuple[tuple[float, ...], QTensor]] = {}
        for input_dict in functional.input_per_layer:
            functional.reservoir_layer.set_parameters(input_dict)
            for step in functional.reservoir_layer:
                if isinstance(step, Circuit):
                    param_signature = tuple(step.get_parameter_values())
                    cached = cache.get(step)
                    if cached is None or cached[0] != param_signature:
                        U = step.to_qtensor()
                        cache[step] = (param_signature, U)
                    else:
                        U = cached[1]
                    state = U @ state @ U.adjoint()
                elif isinstance(step, Schedule):
                    res: FunctionalResult = self._execute_analog_evolution(
                        AnalogEvolution(step, state), readout=[StateTomographyReadout()]
                    )
                    state: QTensor = res.final_state

            try:
                state = state.to_density_matrix()
            except ValueError as exc:
                raise ValueError(
                    "Reservoir Runtime Error: state repair failed before expectation value computation. "
                    f"{exc} "
                    "Try improving simulation precision (e.g., smaller dt, more integrator substeps, or higher precision)."
                ) from exc

            inter_results.append(Backend._construct_results_list(state, readout))

            if functional.reservoir_layer.qubits_to_reset:
                state = reset_qubits(state, functional.reservoir_layer.qubits_to_reset)

        return FunctionalResult(readout_results=inter_results[-1], intermediate_results=inter_results[:-1])

    def _execute_variational_program(
        self, functional: VariationalProgram[PrimitiveFunctional[TResult]], readout: list[ReadoutMethod]
    ) -> VariationalProgramResult:
        """Optimize a :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

        Args:
            functional (VariationalProgram): Variational program to optimize.

        Returns:
            VariationalProgramResult[TResult]: Optimizer output and final functional execution.

        Raises:
            ValueError: If the wrapped functional has no trainable parameters.
        """

        def evaluate_sample(parameters: list[float]) -> float:
            param_names = functional.functional.get_parameter_names(where=lambda param: param.is_trainable)
            param_bounds = functional.functional.get_parameter_bounds()
            new_param_dict = {}
            for i, param in enumerate(parameters):
                name = param_names[i]
                lower_bound, upper_bound = param_bounds[name]
                if lower_bound != upper_bound:
                    new_param_dict[name] = param
            err = functional.check_parameter_constraints(new_param_dict)
            if err > 0:
                return err
            functional.functional.set_parameters(new_param_dict)
            results = self.execute(functional.functional, readout)
            final_results = functional.cost_function.compute_cost(results)
            if isinstance(final_results, float):
                return final_results
            if isinstance(final_results, complex) and abs(final_results.imag) < get_settings().atol:
                return final_results.real
            raise ValueError(f"Unsupported result type {type(final_results)}.")

        if len(functional.functional.get_parameters(where=lambda param: param.is_trainable)) == 0:
            raise ValueError("Functional provided does not contain trainable parameters.")

        optimizer_result = functional.optimizer.optimize(
            cost_function=evaluate_sample,
            init_parameters=list(functional.functional.get_parameters(where=lambda param: param.is_trainable).values()),
            bounds=list(functional.functional.get_parameter_bounds(where=lambda param: param.is_trainable).values()),
            store_intermediate_results=functional.store_intermediate_results,
        )

        param_names = functional.functional.get_parameter_names(where=lambda param: param.is_trainable)
        optimal_parameter_dict = {param_names[i]: param for i, param in enumerate(optimizer_result.optimal_parameters)}
        err = functional.check_parameter_constraints(optimal_parameter_dict)
        if err > 0:
            raise ValueError(
                "Optimizer Failed at finding an optimal solution. Check the parameter constraints or try with a different optimization method."
            )
        functional.functional.set_parameters(optimal_parameter_dict)
        optimal_results: FunctionalResult = self.execute(functional.functional, readout)

        return VariationalProgramResult(optimizer_result=optimizer_result, result=optimal_results)

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}()"

    @classmethod
    def _construct_sampling_results(
        cls, final_state: QTensor, readout: SamplingReadout, seed: int | None = None, **kwarg: Any
    ) -> SamplingReadoutResult:
        return SamplingReadoutResult(readout=copy(readout), state=final_state)

    @classmethod
    def _construct_expectation_results(
        cls, final_state: QTensor, readout: ExpectationReadout, **kwarg: Any
    ) -> ExpectationReadoutResult:
        # Pre-scale on the original so the copy inherits cached qtensor_observables
        readout.scale_observables(nqubits=final_state.nqubits)
        return ExpectationReadoutResult(readout=copy(readout), state=final_state)

    @classmethod
    def _construct_state_tomography_results(
        cls, final_state: QTensor, readout: StateTomographyReadout, **kwarg: Any
    ) -> StateTomographyReadoutResult:
        return StateTomographyReadoutResult(readout=copy(readout), final_state=final_state)

    @classmethod
    def _construct_results_list(
        cls, final_state: QTensor, readout: list[ReadoutMethod], seed: int | None = None, **kwarg: Any
    ) -> list[ReadoutResult]:
        results: list[ReadoutResult] = []
        for ro in readout:
            if isinstance(ro, StateTomographyReadout):
                if ro.state_tomography_method != "exact":
                    raise ValueError("State Tomography methods that are not exact are not supported yet.")
                results.append(cls._construct_state_tomography_results(final_state=final_state, readout=ro, **kwarg))
            elif isinstance(ro, ExpectationReadout):
                results.append(cls._construct_expectation_results(final_state=final_state, readout=ro, **kwarg))
            elif isinstance(ro, SamplingReadout):
                results.append(cls._construct_sampling_results(final_state=final_state, readout=ro, seed=seed, **kwarg))
            else:
                raise ValueError(f"Unsupported Readout Method provided: {ro}")
        return results
