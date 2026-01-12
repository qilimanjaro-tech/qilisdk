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

from abc import ABC, abstractmethod
from typing import overload

from qilisdk.digital.gates import Gate
from qilisdk.yaml import yaml


@yaml.register_class
class NoisePass(ABC):
    """
    Generic Noise Class.
    """

    @abstractmethod
    def __init__(self) -> None:
        # Keep NoisePass abstract without forcing a noise_type implementation.
        super().__init__()


@yaml.register_class
class ClassicalNoise(NoisePass): ...


@yaml.register_class
class DigitalNoise(NoisePass): ...


@yaml.register_class
class AnalogNoise(NoisePass): ...


@yaml.register_class
class NoiseInformation:
    def __init__(self, noise_pass: NoisePass) -> None:
        self._noise_pass = noise_pass

    @property
    def noise_pass(self) -> NoisePass:
        return self._noise_pass


@yaml.register_class
class DigitalNoiseInformation(NoiseInformation):
    def __init__(
        self,
        noise_pass: DigitalNoise,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        self._affected_qubits = affected_qubits or []
        self._affected_gates = affected_gates or []
        super().__init__(noise_pass)

    @property
    def affected_qubits(self) -> list[int]:
        return self._affected_qubits

    @property
    def affected_gates(self) -> list[type[Gate]]:
        return self._affected_gates


@yaml.register_class
class AnalogNoiseInformation(NoiseInformation):
    def __init__(
        self,
        noise_pass: AnalogNoise,
        affected_qubits: list[int] | None = None,
    ) -> None:
        self._affected_qubits = affected_qubits or []
        super().__init__(noise_pass)

    @property
    def affected_qubits(self) -> list[int]:
        return self._affected_qubits


@yaml.register_class
class ClassicalNoiseInformation(NoiseInformation):
    def __init__(
        self,
        noise_pass: ClassicalNoise,
        affected_parameters: list[str] | None = None,
    ) -> None:
        self._affected_parameters = affected_parameters or []
        super().__init__(noise_pass)

    @property
    def affected_parameters(self) -> list[str]:
        return self._affected_parameters


@yaml.register_class
class NoiseModel:
    """
    Composite Noise Model consisting of multiple noise passes.
    """

    def __init__(self) -> None:
        self._noise_passes_information: list[NoiseInformation] = []

    def get_noise_passes(self, qubit: int | None = None) -> list[NoisePass]:
        """
        Returns the list of noise passes in the composite noise model.

        Returns:
            list[NoiseBase]: The list of noise passes.
        """
        if qubit is None:
            return [noise.noise_pass for noise in self._noise_passes_information]
        noise_passes = []
        for noise in self._noise_passes_information:
            if isinstance(noise, (DigitalNoiseInformation, AnalogNoiseInformation)) and (
                len(noise.affected_qubits) == 0 or qubit in noise.affected_qubits
            ):
                noise_passes.append(noise.noise_pass)
        return noise_passes

    def get_noise_information(self, qubit: int | None = None) -> list[NoiseInformation]:
        """
        Returns the list of noise passes in the composite noise model.

        Returns:
            list[NoiseBase]: The list of noise passes.
        """
        if qubit is None:
            return self._noise_passes_information
        noise_passes: list[NoiseInformation] = []
        for noise in self._noise_passes_information:
            if isinstance(noise, (DigitalNoiseInformation, AnalogNoiseInformation)) and (
                len(noise.affected_qubits) == 0 or qubit in noise.affected_qubits
            ):
                noise_passes.append(noise)
        return noise_passes

    @overload
    def add(
        self,
        noise: DigitalNoise,
        *,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None: ...

    @overload
    def add(self, noise: AnalogNoise, *, affected_qubits: list[int] | None = None) -> None: ...

    @overload
    def add(
        self,
        noise: ClassicalNoise,
        *,
        affected_parameters: list[str] | None = None,
    ) -> None: ...

    def add(
        self,
        noise: NoisePass,
        *,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
        affected_parameters: list[str] | None = None,
    ) -> None:
        """
        Adds a new noise pass to the composite noise model.

        Args:
            noise (NoiseBase): The noise pass to add.
        """
        if isinstance(noise, DigitalNoise):
            self._add_digital_noise(noise, affected_qubits=affected_qubits, affected_gates=affected_gates)
        elif isinstance(noise, AnalogNoise):
            self._add_analog_noise(noise, affected_qubits=affected_qubits)
        elif isinstance(noise, ClassicalNoise):
            self._add_classical_noise(noise, affected_parameters=affected_parameters)

    def _add_digital_noise(
        self,
        noise: DigitalNoise,
        affected_qubits: list[int] | None = None,
        affected_gates: list[type[Gate]] | None = None,
    ) -> None:
        self._noise_passes_information.append(
            DigitalNoiseInformation(noise, affected_qubits=affected_qubits, affected_gates=affected_gates)
        )

    def _add_analog_noise(self, noise: AnalogNoise, affected_qubits: list[int] | None = None) -> None:
        self._noise_passes_information.append(AnalogNoiseInformation(noise, affected_qubits=affected_qubits))

    def _add_classical_noise(self, noise: ClassicalNoise, affected_parameters: list[str] | None = None) -> None:
        self._noise_passes_information.append(ClassicalNoiseInformation(noise, affected_parameters=affected_parameters))
