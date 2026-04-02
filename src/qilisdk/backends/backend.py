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
from typing import TYPE_CHECKING, Any, Callable, cast, overload

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
from qilisdk.readout import (
    E,
    ExpectationReadout,
    ExpectationReadoutResult,
    Readout,
    ReadoutMethod,
    S,
    SamplingReadout,
    SamplingReadoutResult,
    StateTomographyReadout,
    StateTomographyReadoutResult,
    T,
)
from qilisdk.readout.readout_result import ReadoutCompositeResults
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core import QTensor
    from qilisdk.core.result import Result
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.noise import NoiseModel


class Backend(ABC):
    """Abstract base class for all quantum simulation backends.

    Subclasses must override one or more of the ``_execute_*`` methods to
    provide concrete simulation logic. The public :meth:`execute` method
    dispatches to the appropriate handler based on the functional type.
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
    ) -> None:
        """Initialise the backend with an optional noise model.

        Args:
            noise_model (NoiseModel | None): Optional noise model applied
                during execution. Defaults to ``None``.
        """
        self._handlers: dict[type[Functional], Callable[[Functional, list[ReadoutMethod]], Result]] = {
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
    def execute(self, functional: VariationalProgram, readout: Readout[S, E, T]) -> VariationalProgramResult: ...

    @overload
    def execute(self, functional: PrimitiveFunctional, readout: Readout[S, E, T]) -> FunctionalResult[S, E, T]: ...

    def execute(self, functional: Functional, readout: Readout[Any, Any, Any]) -> Result:
        """Execute a quantum functional with the specified readout methods.

        This is the main entry point for running any supported functional on
        the backend.  The method validates the readout specification, then
        dispatches to the appropriate ``_execute_*`` handler registered for
        the functional type.

        Args:
            functional: The quantum functional to execute.
            readout: A :class:`~qilisdk.readout.Readout` built via
                the builder pattern.

        Returns:
            The execution result whose concrete type depends on the
            functional and readout specification.

        Raises:
            NotImplementedError: If the backend does not support the given
                functional type.
            ValueError: If the readout specification is empty.
        """
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        readout_list = readout.to_list()
        if not readout_list:
            raise ValueError("At least one readout method must be provided in the Readout.")
        return handler(functional, readout_list)

    def _execute_digital_propagation(
        self, functional: DigitalPropagation, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no DigitalPropagation implementation")

    def _execute_analog_evolution(self, functional: AnalogEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no AnalogEvolution implementation")

    def _execute_quantum_reservoir(
        self, functional: QuantumReservoir, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        """Execute a quantum reservoir computing functional.

        Returns:
            FunctionalResult: the final quantum reservoir execution results.

        Raises:
            ValueError:if throughout the execution the state becomes invalid due to numerical instabilities.
        """
        if self._noise_model:
            logger.warning("Noise Models are not supported with Quantum Reservoirs, so they will be ignored.")
        state = functional.initial_state.to_density_matrix()
        inter_results: list[ReadoutCompositeResults] = []
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
                    res = self._execute_analog_evolution(
                        AnalogEvolution(step, state), readout=[StateTomographyReadout()]
                    )
                    state: QTensor = res.state

            try:
                state: QTensor = state.to_density_matrix()
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
        self, functional: VariationalProgram, readout: list[ReadoutMethod]
    ) -> VariationalProgramResult:
        # Wrap the flat readout list back into a spec for the recursive execute() call
        spec = _readout_list_to_spec(readout)

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
            results = self.execute(functional.functional, spec)
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
        optimal_results = self.execute(functional.functional, spec)

        return VariationalProgramResult(optimizer_result=optimizer_result, result=optimal_results)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the backend."""
        return f"{type(self).__qualname__}()"

    @classmethod
    def _construct_results_list(
        cls, final_state: QTensor, readout: list[ReadoutMethod], seed: int | None = None
    ) -> ReadoutCompositeResults:
        sampling_result: SamplingReadoutResult | None = None
        expectation_result: ExpectationReadoutResult | None = None
        state_tomography_result: StateTomographyReadoutResult | None = None

        for ro in readout:
            if isinstance(ro, StateTomographyReadout):
                if ro.method != "exact":
                    raise ValueError("State Tomography methods that are not exact are not supported yet.")
                state_tomography_result: StateTomographyReadoutResult = StateTomographyReadoutResult.from_state(
                    state=final_state
                )
            elif isinstance(ro, ExpectationReadout):
                ro.expand_observables(nqubits=final_state.nqubits)
                expectation_result: ExpectationReadoutResult = ExpectationReadoutResult.from_state(
                    expectation_readout=ro, state=final_state
                )
            elif isinstance(ro, SamplingReadout):
                sampling_result: SamplingReadoutResult = SamplingReadoutResult.from_state(
                    sampling_readout=ro, state=final_state
                )
            else:
                raise ValueError(f"Unsupported Readout Method provided: {ro}")

        return ReadoutCompositeResults(
            sampling=sampling_result,  # ty:ignore[invalid-argument-type]
            expectation_values=expectation_result,  # ty:ignore[invalid-argument-type]
            state_tomography=state_tomography_result,  # ty:ignore[invalid-argument-type]
        )


def _readout_list_to_spec(readout: list[ReadoutMethod]) -> Readout:
    """Convert a flat readout list back to a :class:`Readout`.

    Used internally when the variational program handler needs to reconstruct a spec from the list received from the dispatcher.

    Returns:
        Readout: the constructed Readout.
    """
    spec: Readout = Readout()
    for ro in readout:
        if isinstance(ro, SamplingReadout):
            spec = spec.with_sampling(nshots=ro.nshots)
        elif isinstance(ro, ExpectationReadout):
            spec = spec.with_expectation(observables=ro.observables, nshots=ro.nshots)
        elif isinstance(ro, StateTomographyReadout):
            spec = spec.with_state_tomography(method=ro.method)
    return spec
