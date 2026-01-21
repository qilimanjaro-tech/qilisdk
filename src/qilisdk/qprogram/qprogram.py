# Copyright 2023 Qilimanjaro Quantum Tech
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

from copy import deepcopy
from typing import Self

from qilisdk.yaml import yaml

from .blocks import Block
from .operations import (
    Measure,
    Play,
    ResetPhase,
    SetFrequency,
    SetPhase,
    Sync,
    Wait,
)
from .structured_program import StructuredProgram
from .variables import QProgramDomain, requires_domain
from .waveforms import IQWaveform, Waveform


@yaml.register_class
class QProgram(StructuredProgram):
    """QProgram is a hardware-agnostic pulse-level programming interface for describing quantum programs.

    This class provides an interface for building quantum programs,
    including defining operations, managing variables, and handling blocks.
    It contains methods for creating, manipulating and controlling
    the execution flow of quantum operations within a program.

    Examples:

        The following example illustrates how to define a Rabi sequence using QProgram.

        .. code-block:: python3

            from qililab import QProgram, Domain, IQWaveform, Square

            qp = QProgram()

            # Pulse used for changing the state of qubit
            control_wf = IQDRAG(amplitude=1.0, duration=40, num_sigmas=4.0, drag_correction=-2.5)

            # Pulse used for exciting the resonator for readout
            readout_wf = IQPair(I=Square(amplitude=1.0, duration=400), Q=Square(amplitude=0.0, duration=400))

            # Weights used during integration
            weights = IQPair(I=Square(amplitude=1.0, duration=2000), Q=Square(amplitude=1.0, duration=2000))

            # Declare a variable
            gain = qp.variable(label="gain", domain=QProgramDomain.Voltage)

            # Loop the variable's value over the range [0.0, 1.0]
            with qp.for_loop(variable=gain, start=0.0, stop=1.0, step=0.01):
                # Change the gain output of the drive_bus
                qp.set_gain(bus="drive_bus", gain=gain)

                # Play the control pulse
                qp.play(bus="drive_bus", waveform=control_wf)

                # Sync the buses
                qp.sync()

                # Measure
                qp.measure(bus="readout_bus", waveform=readout_wf, weights=weights)

    """

    def __str__(self) -> str:
        ignored_attributes = {"_uuid", "variable", "elements", "waveform", "weights"}

        def format_attr_value(value: object) -> str:
            value_text = str(value)
            return "None" if "UUID" in value_text else value_text

        def format_attributes(element: object, indent: str) -> list[str]:
            return [
                f"{indent}{attr_name}: {format_attr_value(attr_value)}\n"
                for attr_name, attr_value in vars(element).items()
                if attr_name not in ignored_attributes
            ]

        def format_envelope(envelope: object, indent: str) -> list[str]:
            return [f"{indent}{line}\n" for line in str(envelope).splitlines()]

        def format_waveform(waveform: Waveform | IQWaveform, indent: str) -> list[str]:
            if isinstance(waveform, Waveform):
                return [f"{indent}Waveform {type(waveform).__name__}:\n", *format_envelope(waveform.envelope(), indent + "\t")]

            waveform_i = waveform.get_I()
            waveform_q = waveform.get_Q()
            return (
                [f"{indent}Waveform I {type(waveform_i).__name__}:\n", *format_envelope(waveform_i.envelope(), indent + "\t"), f"{indent}Waveform Q {type(waveform_q).__name__}:\n", *format_envelope(waveform_q.envelope(), indent + "\t")]
            )

        def format_weights(weights: IQWaveform, indent: str) -> list[str]:
            weights_i = weights.get_I()
            weights_q = weights.get_Q()
            return (
                [f"{indent}Weights I {type(weights_i).__name__}:\n", *format_envelope(weights_i.envelope(), indent + "\t"), f"{indent}Weights Q {type(weights_q).__name__}:\n", *format_envelope(weights_q.envelope(), indent + "\t")]
            )

        def traverse(block: Block, indent: str = "") -> list[str]:
            string_elements = []
            for element in block.elements:
                string_elements.append(f"{indent}{type(element).__name__}:\n")
                string_elements.extend(format_attributes(element, indent + "\t"))

                if isinstance(element, Block):
                    string_elements.extend(traverse(element, indent + "\t"))

                if hasattr(element, "waveform"):
                    string_elements.extend(format_waveform(element.waveform, indent + "\t"))

                if hasattr(element, "weights"):
                    string_elements.extend(format_weights(element.weights, indent + "\t"))

            return string_elements

        return "".join(traverse(self._body))

    def with_bus_mapping(self, bus_mapping: dict[str, str]) -> Self:
        """Returns a copy of the QProgram with bus mappings applied.

        Args:
            bus_mapping (dict[str, str]): A dictionary mapping old bus names to new bus names.

        Returns:
            QProgram: A new instance of QProgram with updated bus names.
        """

        def traverse(block: Block) -> None:
            for index, element in enumerate(block.elements):
                if isinstance(element, Block):
                    traverse(element)
                elif hasattr(element, "bus"):
                    bus = getattr(element, "bus")
                    if isinstance(bus, str) and bus in bus_mapping:
                        setattr(block.elements[index], "bus", bus_mapping[bus])
                elif hasattr(element, "buses"):
                    buses = getattr(element, "buses")
                    if isinstance(buses, list):
                        setattr(
                            block.elements[index],
                            "buses",
                            [bus_mapping.get(bus, bus) for bus in buses],
                        )

        # Copy qprogram so the original remain unaffected
        copied_qprogram = deepcopy(self)

        # Recursively traverse qprogram applying the bus mapping
        traverse(copied_qprogram.body)

        # Apply the mapping to buses property
        for bus in copied_qprogram.buses:
            if bus in bus_mapping:
                copied_qprogram.buses.remove(bus)
                copied_qprogram.buses.add(bus_mapping[bus])

        return copied_qprogram

    def play(self, bus: str, waveform: Waveform | IQWaveform) -> None:
        """Play a single waveform or an I/Q pair of waveforms on the bus.

        Args:
            bus (str): Unique identifier of the bus.
            waveform (Waveform | IQWaveform): A single waveform or an I/Q pair of waveforms
        """
        operation = Play(bus=bus, waveform=waveform)
        self._active_block.append(operation)
        self._buses.add(bus)

    def measure(self, bus: str, waveform: IQWaveform, weights: IQWaveform) -> None:
        """Play a pulse and acquire results.

        Args:
            bus (str): Unique identifier of the bus.
            waveform (IQWaveform): Waveform played during measurement.
            weights (IQWaveform): Weights used during demodulation/integration.
        """
        operation = Measure(bus=bus, waveform=waveform, weights=weights)
        self._active_block.append(operation)
        self._buses.add(bus)

    @requires_domain("duration", QProgramDomain.Time)
    def wait(self, bus: str, duration: int) -> None:
        """Adds a delay on the bus with a specified time.

        Args:
            bus (str): Unique identifier of the bus.
            duration (int): Duration of the delay.
        """
        operation = Wait(bus=bus, duration=duration)
        self._active_block.append(operation)
        self._buses.add(bus)

    def sync(self, buses: list[str] | None = None) -> None:
        """Synchronize operations between buses, so the operations following will start at the same time.

        If no buses are given, then the synchronization will involve all buses present in the QProgram.

        Args:
            buses (list[str], optional): List of unique identifiers of the buses. Defaults to None.
        """
        operation = Sync(buses=buses)
        self._active_block.append(operation)
        if buses:
            self._buses.update(buses)

    def reset_phase(self, bus: str) -> None:
        """Reset the absolute phase of the NCO associated with the bus.

        Args:
            bus (str): Unique identifier of the bus.
        """
        operation = ResetPhase(bus=bus)
        self._active_block.append(operation)
        self._buses.add(bus)

    @requires_domain("phase", QProgramDomain.Phase)
    def set_phase(self, bus: str, phase: float) -> None:
        """Set the absolute phase of the NCO associated with the bus.

        Args:
            bus (str): Unique identifier of the bus.
            phase (float): The new absolute phase of the NCO.
        """
        operation = SetPhase(bus=bus, phase=phase)
        self._active_block.append(operation)
        self._buses.add(bus)

    @requires_domain("frequency", QProgramDomain.Frequency)
    def set_frequency(self, bus: str, frequency: float) -> None:
        """Set the frequency of the NCO associated with bus.

        Args:
            bus (str): Unique identifier of the bus.
            frequency (float): The new frequency of the NCO.
        """
        operation = SetFrequency(bus=bus, frequency=frequency)
        self._active_block.append(operation)
        self._buses.add(bus)
