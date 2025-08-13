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

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

WHITE: Final[str] = "#FFFFFF"
BLACK: Final[str] = "#000000"

# Neutral ramp (light → dark)
NEUTRAL_050: Final[str] = "#F0F0F0"
NEUTRAL_100: Final[str] = "#DCDCDC"
NEUTRAL_200: Final[str] = "#CACACA"
NEUTRAL_600: Final[str] = "#7A7A7A"
NEUTRAL_800: Final[str] = "#2F2F2F"
NEUTRAL_900: Final[str] = "#1F1F1F"

# Brand colours
VIOLET: Final[str] = "#5E56A1"
MAGENTA: Final[str] = "#AC115F"


class Theme(BaseModel):
    """Colour Theme."""

    model_config = ConfigDict(frozen=True)

    background: str = Field(description="Canvas background.")
    on_background: str = Field(description="Default text/line color on background.")
    surface: str = Field(description="Raised surface/panel fill.")
    on_surface: str = Field(description="Text/line color on surface.")
    surface_muted: str = Field(description="Muted lines on background (wires/grid).")
    border: str = Field(description="Neutral stroke/border color.")
    primary: str = Field(description="Primary/brand fill.")
    on_primary: str = Field(description="Text/icons over primary.")
    accent: str = Field(description="Accent/highlight color.")
    on_accent: str = Field(description="Text/icons over accent.")


light = Theme(
    background=WHITE,
    on_background=BLACK,
    surface=NEUTRAL_100,
    on_surface=BLACK,
    surface_muted=NEUTRAL_050,
    border=NEUTRAL_200,
    primary=VIOLET,
    on_primary=WHITE,
    accent=MAGENTA,
    on_accent=WHITE,
)

dark = Theme(
    background=BLACK,
    on_background=WHITE,
    surface=NEUTRAL_800,
    on_surface=BLACK,
    surface_muted=NEUTRAL_900,
    border=NEUTRAL_600,
    primary=MAGENTA,
    on_primary=WHITE,
    accent=VIOLET,
    on_accent=WHITE,
)
