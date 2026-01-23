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

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qilisdk.analog import Schedule
from qilisdk.analog import X, Z
from qilisdk.utils.visualization.schedule_renderers import MatplotlibScheduleRenderer
from qilisdk.utils.visualization.style import ScheduleStyle
from matplotlib import font_manager as fm

def test_schedule_style_init():

    style = ScheduleStyle()
    assert style.dpi == 150
    assert style.theme.background is not None
    assert style.fontsize == 10
    assert isinstance(style.font, fm.FontProperties)

def test_schedule_renderer_init():

    H0 = X(1) + X(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={})
    style = ScheduleStyle()
    renderer = MatplotlibScheduleRenderer(schedule=schedule, style=style)
    assert renderer.schedule == schedule
    assert renderer.style == style
    assert renderer.ax is not None

def test_schedule_renderer_with_axes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # Prevent actual rendering during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)  # Prevent file saving during tests

    H0 = X(1) + X(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0}, coefficients={})
    style = ScheduleStyle()
    style.grid = True
    style.grid_style = {}
    ax = plt.gca()
    renderer = MatplotlibScheduleRenderer(schedule=schedule, style=style, ax=ax)
    assert renderer.schedule == schedule
    assert renderer.style == style
    assert renderer.ax is not None
    renderer.plot(ax=ax)

def test_schedule_draw(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # Prevent actual rendering during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)  # Prevent file saving during tests
    monkeypatch.setattr(MatplotlibScheduleRenderer, "save", lambda self, filepath: None)  # Prevent file saving during tests

    # Create a simple schedule for testing
    H0 = X(1) + X(0)
    H1 = Z(1) + Z(0)
    schedule = Schedule(total_time=10, hamiltonians={"H0": H0, "H1": H1}, coefficients={})
    schedule.draw()
    schedule.draw(filepath="test_schedule.png")
