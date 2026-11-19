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
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, cast, overload

import numpy as np
from loguru import logger

from qilisdk.algorithms.algorithm import Algorithm
from qilisdk.algorithms.quantum_reservoirs import QuantumReservoir
from qilisdk.algorithms.variational_program import VariationalProgram
from qilisdk.core.result import Result
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.digital_propagation import DigitalPropagation
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

if TYPE_CHECKING:
    from qilisdk.algorithms.variational_program_result import VariationalProgramResult
    from qilisdk.core import QTensor
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.functionals.functional_result import FunctionalResult
    from qilisdk.noise import NoiseModel


class Backend(ABC):
    """Abstract base class for all quantum simulation backends.

    Subclasses must override one or more of the ``_execute_*`` methods to
    provide concrete simulation logic for the primitive functionals. The
    public :meth:`execute` method dispatches to the appropriate handler based
    on the functional type, while an :class:`~qilisdk.algorithms.Algorithm`
    without a registered handler runs its own ``_execute`` on this backend.
    A subclass can run an algorithm natively instead by defining
    ``_execute_quantum_reservoir`` or ``_execute_variational_program``.
    """

    # Whether the backend can simulate non-Markovian noise
    _supports_non_markovian_noise: bool = False

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise the backend with an optional noise model.

        Args:
            noise_model (NoiseModel | None): Optional noise model applied
                during execution. Defaults to ``None``.
            seed (int | None): Seed for the random number generator used to draw
                samples from a state. Repeated executions on one backend give new
                samples, reproducible for a given seed. If ``None``, the samples
                are not reproducible. Defaults to ``None``.
        """
        self._handlers: dict[type[Functional], Callable[[Functional, list[ReadoutMethod]], Result]] = {
            DigitalPropagation: lambda f, readout: self._execute_digital_propagation(
                cast("DigitalPropagation", f), readout
            ),
            AnalogEvolution: lambda f, readout: self._execute_analog_evolution(cast("AnalogEvolution", f), readout),
        }
        # Algorithms run their own _execute unless the backend defines a native hook for them
        for algorithm, hook in (
            (QuantumReservoir, "_execute_quantum_reservoir"),
            (VariationalProgram, "_execute_variational_program"),
        ):
            if hasattr(self, hook):
                self._handlers[algorithm] = getattr(self, hook)
        self._noise_model = noise_model
        self._rng = np.random.default_rng(seed)

    @property
    def noise_model(self) -> NoiseModel | None:
        """Noise model applied during execution, or ``None`` for noiseless simulation."""
        return self._noise_model

    @overload
    def execute(
        self, functional: VariationalProgram, readout: Readout[S, E, T]
    ) -> VariationalProgramResult[FunctionalResult[S, E, T]]: ...

    @overload
    def execute(self, functional: QuantumReservoir, readout: Readout[S, E, T]) -> FunctionalResult[S, E, T]: ...

    @overload
    def execute(self, functional: PrimitiveFunctional, readout: Readout[S, E, T]) -> FunctionalResult[S, E, T]: ...

    @overload
    def execute(self, functional: Algorithm, readout: Readout[Any, Any, Any]) -> Result: ...

    def execute(self, functional: Functional, readout: Readout[Any, Any, Any]) -> Result:
        """Execute a quantum functional with the specified readout methods.

        This is the main entry point for running any supported functional on
        the backend.  The method validates the readout specification, then
        dispatches to the appropriate ``_execute_*`` handler registered for
        the functional type, or, for an :class:`~qilisdk.algorithms.Algorithm`
        with no registered handler, to the algorithm's own ``_execute``.

        Args:
            functional: The quantum functional to execute.
            readout: A :class:`~qilisdk.readout.Readout` built via
                the builder pattern.

        Returns:
            The execution result whose concrete type depends on the
            functional and readout specification. Its ``execution_time``
            attribute holds the wall-clock time, in seconds, taken by the
            backend to run the simulation.

        Raises:
            NotImplementedError: If the backend does not support the given
                functional type, or the noise model contains non-Markovian
                noise the backend cannot simulate.
            ValueError: If the readout specification is empty.
        """
        logger.debug("[Backend] Dispatching {} to {}", type(functional).__qualname__, type(self).__qualname__)
        handler = next((h for cls, h in self._handlers.items() if isinstance(functional, cls)), None)
        if handler is None and not isinstance(functional, Algorithm):
            raise NotImplementedError(f"{type(self).__qualname__} does not support {type(functional).__qualname__}")
        if (
            self._noise_model is not None
            and self._noise_model.non_markovian_noise
            and not self._supports_non_markovian_noise
        ):
            raise NotImplementedError(f"{type(self).__qualname__} does not support non-Markovian noise")

        readout_list = readout.to_list()
        if not readout_list:
            raise ValueError("At least one readout method must be provided in the Readout.")
        logger.opt(lazy=True).debug("[Backend] Readout methods: {}", lambda: [type(ro).__name__ for ro in readout_list])

        start = perf_counter()
        if handler is not None:
            result = handler(functional, readout_list)
        else:
            result = cast("Algorithm", functional)._execute(self, readout)  # ruff: ignore[private-member-access]
        if isinstance(result, Result):
            result.execution_time = perf_counter() - start
        return result

    def _execute_digital_propagation(
        self, functional: DigitalPropagation, readout: list[ReadoutMethod]
    ) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no DigitalPropagation implementation")

    def _execute_analog_evolution(self, functional: AnalogEvolution, readout: list[ReadoutMethod]) -> FunctionalResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no AnalogEvolution implementation")

    def readout_state(self, state: QTensor, readout: list[ReadoutMethod]) -> ReadoutCompositeResults:
        """Apply readout methods to a state, drawing any samples from this backend's random number generator.

        Args:
            state (QTensor): The state to read out.
            readout (list[ReadoutMethod]): The readout methods to apply.

        Returns:
            ReadoutCompositeResults: The results of each readout method.
        """
        return self._construct_results_list(state, readout, seed=self._rng)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the backend."""
        return f"{type(self).__qualname__}()"

    @classmethod
    def _construct_results_list(
        cls,
        final_state: QTensor,
        readout: list[ReadoutMethod],
        seed: int | np.random.Generator | None = None,
        qubits_to_measure: list[int] | None = None,
    ) -> ReadoutCompositeResults:
        sampling_result: SamplingReadoutResult | None = None
        expectation_result: ExpectationReadoutResult | None = None
        state_tomography_result: StateTomographyReadoutResult | None = None

        for ro in readout:
            if isinstance(ro, StateTomographyReadout):
                if ro.method != "exact":
                    raise ValueError("State Tomography methods that are not exact are not supported yet.")
                state_tomography_result: StateTomographyReadoutResult = StateTomographyReadoutResult.from_state(
                    state=final_state.partial_trace(set(qubits_to_measure))
                    if qubits_to_measure is not None
                    else final_state
                )
            elif isinstance(ro, ExpectationReadout):
                ro.expanded_observables(nqubits=final_state.nqubits)
                expectation_result: ExpectationReadoutResult = ExpectationReadoutResult.from_state(
                    expectation_readout=ro, state=final_state, seed=seed
                )
            elif isinstance(ro, SamplingReadout):
                sampling_result: SamplingReadoutResult = SamplingReadoutResult.from_state(
                    sampling_readout=ro,
                    state=final_state,
                    qubits_to_measure=qubits_to_measure,
                    expand_samples=ro.expand_samples,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported Readout Method provided: {ro}")

        return ReadoutCompositeResults(
            sampling=sampling_result,  # ty:ignore[invalid-argument-type]
            expectation_values=expectation_result,  # ty:ignore[invalid-argument-type]
            state_tomography=state_tomography_result,  # ty:ignore[invalid-argument-type]
        )
