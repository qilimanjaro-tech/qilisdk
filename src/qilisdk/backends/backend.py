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

import numpy as np

from qilisdk.analog import Schedule
from qilisdk.core import QTensor, expect_val, reset_qubits
from qilisdk.core.qtensor import probabilities_from_state, samples_from_probabilities
from qilisdk.digital import Circuit
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.digital_evolution import DigitalEvolution
from qilisdk.functionals.functional_result import (
    ExpectationReadoutResults,
    FunctionalResult,
    ReadoutResult,
    SamplingReadoutResults,
    StateTomographyReadoutResults,
)
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir
from qilisdk.functionals.quantum_reservoirs_result import QuantumReservoirResult
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.functionals.sampling import Sampling
    from qilisdk.noise import NoiseModel

TResult = TypeVar("TResult", bound=FunctionalResult)


class Backend(ABC):

    def __init__(self, noise_model: NoiseModel | None = None) -> None:
        self._noise_model = noise_model
        self._handlers: dict[
            type[Functional], Callable[[Functional, ReadoutMethod | list[ReadoutMethod]], FunctionalResult]
        ] = {
            DigitalEvolution: lambda f, readout: self._execute_digital_evolution(cast("DigitalEvolution", f), readout),
            AnalogEvolution: lambda f, readout: self._execute_analog_evolution(cast("AnalogEvolution", f), readout),
            QuantumReservoir: lambda f, readout: self._execute_quantum_reservoir(cast("QuantumReservoir", f), readout),
            VariationalProgram: lambda f, readout: self._execute_variational_program(
                cast("VariationalProgram", f), readout
            ),
        }

    @overload
    def execute(
        self, functional: VariationalProgram, readout: ReadoutMethod | list[ReadoutMethod]
    ) -> VariationalProgramResult[FunctionalResult]: ...

    @overload
    def execute(self, functional: PrimitiveFunctional, readout: ReadoutMethod | list[ReadoutMethod]) -> TResult: ...

    def execute(self, functional: Functional, readout: ReadoutMethod | list[ReadoutMethod]) -> FunctionalResult:
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc
        if not isinstance(readout, list):
            _readout = [readout]
        else:
            if any(not isinstance(ro, ReadoutMethod) for ro in readout):
                raise ValueError(f"Readout method provided is not valid. Provided: {[type(ro) for ro in readout]}")
            _readout = readout
        if len({ro.__class__ for ro in readout}) != len(_readout):
            raise ValueError(
                f"Each type of readout is allowed to be specified once.\nprovided a list with the following types {readout}"
            )
        return handler(functional, _readout)

    def _execute_sampling(self, functional: Sampling, readout: ReadoutMethod | list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no Sampling implementation")

    def _execute_time_evolution(self, functional: TimeEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no TimeEvolution implementation")

    def _execute_digital_evolution(
        self, functional: DigitalEvolution, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no Sampling implementation")

    def _execute_analog_evolution(self, functional: AnalogEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no TimeEvolution implementation")

    def _execute_quantum_reservoir(
        self, functional: QuantumReservoir, readout: list[ReadoutMethod]
    ) -> QuantumReservoirResult:
        state = functional.initial_state.to_density_matrix()
        expected_values: list[list[Number]] = []
        intermediate_states: list[QTensor] = []
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
                    res = self._execute_time_evolution(
                        TimeEvolution(step, [], state, functional.nshots), [StateTomographyReadout()]
                    )

                    if len(res.final_states) != 1:
                        raise ValueError("Reservoir Runtime Error: Time Evolution Failed.")
                    state = res.final_states[0]

            try:
                state = state.to_density_matrix()
            except ValueError as exc:
                raise ValueError(
                    "Reservoir Runtime Error: state repair failed before expectation value computation. "
                    f"{exc} "
                    "Try improving simulation precision (e.g., smaller dt, more integrator substeps, or higher precision)."
                ) from exc

            if functional.store_intermediate_states:
                intermediate_states.append(state)

            expected_values.append(
                [expect_val(operator=obs, state=state) for obs in functional.reservoir_layer.observables_as_qtensor]
            )

            if functional.reservoir_layer.qubits_to_reset:
                state = reset_qubits(state, functional.reservoir_layer.qubits_to_reset)

        return QuantumReservoirResult(
            expected_values=np.array(expected_values),
            final_expected_values=np.array(expected_values[-1]),
            final_state=state.to_density_matrix() if functional.store_final_state else None,
            intermediate_states=intermediate_states if functional.store_intermediate_states else None,
        )

    def _execute_variational_program(
        self, functional: VariationalProgram[PrimitiveFunctional[TResult]], readout: list[ReadoutMethod]
    ) -> VariationalProgramResult[TResult]:
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
        optimal_results: TResult = self.execute(functional.functional, readout)

        return VariationalProgramResult(optimizer_result=optimizer_result, result=optimal_results)

    @classmethod
    def _construct_sampling_results(
        cls, final_state: QTensor, readout: SamplingReadout, seed: int | None = None, **kwarg: Any
    ) -> SamplingReadoutResults:
        probabilities = probabilities_from_state(final_state)
        return SamplingReadoutResults(
            readout=copy(readout),
            samples=samples_from_probabilities(probabilities, nshots=readout.nshots, seed=seed),
            probabilities=probabilities,
        )

    @classmethod
    def _construct_expectation_results(
        cls, final_state: QTensor, readout: ExpectationReadout, **kwarg: Any
    ) -> ExpectationReadoutResults:
        return ExpectationReadoutResults(
            readout=copy(readout),
            expected_values=[
                (
                    expect_val(o, final_state)
                    if isinstance(o, QTensor)
                    else expect_val(o.to_qtensor(final_state.nqubits), final_state)
                )
                for o in readout.observables
            ],
        )

    @classmethod
    def _construct_state_tomography_results(
        cls, final_state: QTensor, readout: StateTomographyReadout, **kwarg: Any
    ) -> StateTomographyReadoutResults:
        return StateTomographyReadoutResults(readout=copy(readout), final_state=final_state)

    @classmethod
    def _construct_results_list(
        cls, final_state: QTensor, readout_methods: list[ReadoutMethod], seed: int | None = None, **kwarg: Any
    ) -> list[ReadoutResult]:
        results: list[ReadoutResult] = []
        for readout in readout_methods:
            if isinstance(readout, StateTomographyReadout):
                if readout.state_tomography_method != "exact":
                    raise ValueError("State Tomography methods that are not exact are not supported yet.")
                results.append(
                    cls._construct_state_tomography_results(final_state=final_state, readout=readout, **kwarg)
                )
            elif isinstance(readout, ExpectationReadout):
                results.append(cls._construct_expectation_results(final_state=final_state, readout=readout, **kwarg))
            elif isinstance(readout, SamplingReadout):
                results.append(
                    cls._construct_sampling_results(final_state=final_state, readout=readout, seed=seed, **kwarg)
                )
            else:
                raise ValueError(f"Unsupported Readout Method provided: {readout}")
        return results
