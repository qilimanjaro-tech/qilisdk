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
from qilisdk.readout.readout_result import ReadoutCompositeResults
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core import QTensor
    from qilisdk.core.result import Result
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.noise import NoiseModel

TResult = TypeVar("TResult", bound=FunctionalResult)


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
        """Execute a quantum functional with the specified readout methods.

        This is the main entry point for running any supported functional on
        the backend. The method validates the readout list, then dispatches
        to the appropriate ``_execute_*`` handler registered for the
        functional type.

        Args:
            functional (Functional): The quantum functional to execute.
            readout (ReadoutMethod | list[ReadoutMethod]): One or more readout
                specifications describing how to extract results from the
                final quantum state.

        Returns:
            Result: The execution result, whose concrete type depends on the
            functional (e.g. ``FunctionalResult`` or
            ``VariationalProgramResult``).

        Raises:
            NotImplementedError: If the backend does not support the given
                functional type.
            ValueError: If any element of ``readout`` is not a valid
                :class:`~qilisdk.readout.ReadoutMethod`, or if duplicate
                readout types are provided.
        """
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
        """Execute a digital-circuit propagation functional.

        Subclasses that support digital circuits must override this method.

        Args:
            functional (DigitalPropagation): The digital propagation
                functional to execute.
            readout (list[ReadoutMethod]): Readout specifications for
                result extraction.

        Returns:
            FunctionalResult: The execution result.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(f"{type(self).__qualname__} has no DigitalPropagation implementation")

    def _execute_analog_evolution(self, functional: AnalogEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        """Execute an analog time-evolution functional.

        Subclasses that support analog evolution must override this method.

        Args:
            functional (AnalogEvolution): The analog evolution functional
                to execute.
            readout (list[ReadoutMethod]): Readout specifications for
                result extraction.

        Returns:
            FunctionalResult: The execution result.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(f"{type(self).__qualname__} has no AnalogEvolution implementation")

    def _execute_quantum_reservoir(
        self, functional: QuantumReservoir, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        """Execute a quantum reservoir computing functional.

        Iterates over each input layer of the reservoir, propagating the
        quantum state through digital circuits and analog schedules.
        Readout results are collected after each layer and the final layer's
        results are returned.

        Args:
            functional (QuantumReservoir): The quantum reservoir functional
                to execute.
            readout (list[ReadoutMethod]): Readout specifications for
                result extraction.

        Returns:
            FunctionalResult: The execution result, including intermediate
                results from earlier layers.

        Raises:
            ValueError: If the quantum state cannot be repaired to a valid
                density matrix between layers.

        Notes:
            Noise models are not supported with quantum reservoirs and will
            be silently ignored if one is set on the backend.
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
                    res: FunctionalResult = self._execute_analog_evolution(
                        AnalogEvolution(step, state), readout=[StateTomographyReadout()]
                    )
                    state: QTensor = res.state

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
        self, functional: VariationalProgram, readout: list[ReadoutMethod]
    ) -> VariationalProgramResult:
        """Optimize a :class:`~qilisdk.functionals.variational_program.VariationalProgram`.

        Runs the classical optimizer loop: at each iteration the inner
        functional is executed with the given ``readout`` methods and the
        cost function is evaluated.

        Args:
            functional (VariationalProgram): Variational program to optimize.
            readout (list[ReadoutMethod]): Readout specifications forwarded to each inner functional execution.

        Returns:
            VariationalProgramResult: Optimizer output together with the optimal functional execution result.

        Raises:
            ValueError: If the wrapped functional has no trainable parameters, or if the optimizer fails to find a valid
                solution within the parameter constraints.
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
        """Return a developer-friendly string representation of the backend."""
        return f"{type(self).__qualname__}()"

    @classmethod
    def _construct_sampling_results(
        cls, final_state: QTensor, readout: SamplingReadout, seed: int | None = None, **kwarg: Any
    ) -> SamplingReadoutResult:
        """Construct a sampling readout result from a final quantum state.

        Args:
            final_state (QTensor): The final quantum state to sample from.
            readout (SamplingReadout): The sampling readout configuration.
            seed (int | None): Optional random seed for reproducible sampling. Defaults to ``None``.
            **kwarg (Any): Additional keyword arguments forwarded to then result constructor.

        Returns:
            SamplingReadoutResult: The constructed sampling result.
        """
        return SamplingReadoutResult.from_state(sampling_readout=copy(readout), state=final_state)

    @classmethod
    def _construct_expectation_results(
        cls, final_state: QTensor, readout: ExpectationReadout, **kwarg: Any
    ) -> ExpectationReadoutResult:
        """Construct an expectation-value readout result from a final quantum state.

        Observables are pre-scaled to match ``final_state.nqubits`` so
        that the copied readout inherits the cached ``QTensor``
        representations.

        Args:
            final_state (QTensor): The final quantum state.
            readout (ExpectationReadout): The expectation readout
                configuration.
            **kwarg (Any): Additional keyword arguments forwarded to the
                result constructor.

        Returns:
            ExpectationReadoutResult: The constructed expectation result.
        """
        # Pre-scale on the original so the copy inherits cached qtensor_observables
        readout.expand_observables(nqubits=final_state.nqubits)
        return ExpectationReadoutResult.from_state(expectation_readout=copy(readout), state=final_state)

    @classmethod
    def _construct_state_tomography_results(
        cls, final_state: QTensor, readout: StateTomographyReadout, **kwarg: Any
    ) -> StateTomographyReadoutResult:
        """Construct a state-tomography readout result from a final quantum state.

        Args:
            final_state (QTensor): The final quantum state.
            readout (StateTomographyReadout): The state-tomography readout
                configuration.
            **kwarg (Any): Additional keyword arguments forwarded to the
                result constructor.

        Returns:
            StateTomographyReadoutResult: The constructed state-tomography
                result.
        """
        return StateTomographyReadoutResult.from_state(state=final_state)

    @classmethod
    def _construct_results_list(
        cls, final_state: QTensor, readout: list[ReadoutMethod], seed: int | None = None, **kwarg: Any
    ) -> ReadoutCompositeResults:
        """Build a list of readout results by dispatching each readout method.

        Iterates over the provided readout methods and delegates to the
        appropriate ``_construct_*_results`` helper for each one.

        Args:
            final_state (QTensor): The final quantum state used by all
                readout methods.
            readout (list[ReadoutMethod]): Readout specifications to
                process.
            seed (int | None): Optional random seed forwarded to sampling
                readouts. Defaults to ``None``.
            **kwarg (Any): Additional keyword arguments forwarded to each
                result constructor.

        Returns:
            list[ReadoutResult]: One result per readout method, in the
                same order as ``readout``.

        Raises:
            ValueError: If a readout method is unsupported or if a
                ``StateTomographyReadout`` uses a method other than
                ``"exact"``.
        """
        results: dict[str, ReadoutResult] = {}
        for ro in readout:
            if isinstance(ro, StateTomographyReadout):
                if ro.state_tomography_method != "exact":
                    raise ValueError("State Tomography methods that are not exact are not supported yet.")
                results["state_tomography"] = cls._construct_state_tomography_results(
                    final_state=final_state, readout=ro, **kwarg
                )
            elif isinstance(ro, ExpectationReadout):
                results["expectation_values"] = cls._construct_expectation_results(
                    final_state=final_state, readout=ro, **kwarg
                )
            elif isinstance(ro, SamplingReadout):
                results["sampling"] = cls._construct_sampling_results(
                    final_state=final_state, readout=ro, seed=seed, **kwarg
                )
            else:
                raise ValueError(f"Unsupported Readout Method provided: {ro}")
        return ReadoutCompositeResults.from_dict(results)
