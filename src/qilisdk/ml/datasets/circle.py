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

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qilisdk.ml.datasets.dataset import Dataset, DatasetSample

if TYPE_CHECKING:
    from qilisdk.ml.datasets.dataset import FloatArray
    from qilisdk.utils.visualization.dataset_renderers import Transform
    from qilisdk.utils.visualization.style import DatasetStyle

# Distance from the centre of the unit square to a corner, the largest radius a circle centred there can have.
_HALF_DIAGONAL = math.sqrt(0.5)


class Circle(Dataset):
    r"""Circle classification, a two-dimensional binary classification task.

    Points :math:`(x, y)` are drawn uniformly from the unit square
    :math:`[0, 1]^2` and labelled by whether they fall inside a circle of radius
    :math:`r` centred at :math:`(0.5, 0.5)`:

    .. math::

        t(x, y) = \begin{cases}
            1 & \text{if } \lVert (x, y) - (0.5, 0.5) \rVert < r, \\
            0 & \text{otherwise}.
        \end{cases}

    The default radius :math:`r = 1 / \sqrt{2\pi}` makes the circle cover half
    of the square, so the two classes are balanced. A ``margin`` can be left
    empty around the boundary to make the classes linearly separable in the
    radial direction. ``inputs`` are the points, shaped ``(npoints, 2)``, and
    ``targets`` are the labels, shaped ``(npoints, 1)``.
    """

    _DEFAULT_DRAW_STYLE = "2d"
    _DRAW_COMPONENT_LABELS = ("x", "y")
    _DRAW_COLOR_BY_TARGET = True
    _DRAW_STYLE_DEFAULTS: ClassVar[dict[str, dict[str, Any]]] = {"2d": {"figsize": (6, 6)}}

    @classmethod
    def draw(
        cls,
        sample: DatasetSample | FloatArray,
        style: str | None = None,
        *,
        config: DatasetStyle | None = None,
        transform: Transform | None = None,
        filepath: str | None = None,
    ) -> None:
        """
        Render a generated :class:`DatasetSample`, or a raw array of points, with matplotlib.

        Overrides :meth:`Dataset.draw`, except that the ``"3d"`` style is not
        supported, since the points are two-dimensional.

        Args:
            sample (DatasetSample | FloatArray): A sample produced by
                :meth:`generate`, whose points are coloured by class, or an array
                of points to plot directly.
            style (str | None): Plot mode, either ``"1d"`` or ``"2d"``. Defaults to ``"2d"``.
            config (DatasetStyle | None): Visual style configuration. Defaults to
                :class:`DatasetStyle`, merged over the dataset's per-mode defaults.
            transform (Transform | None): Callable mapping the points to the
                coordinates to plot, overriding the default view.
            filepath (str | None): If given, the figure is saved to this path
                (format inferred from the extension) instead of being shown.

        Raises:
            ValueError: If ``style`` is ``"3d"``.
        """
        if style == "3d":
            raise ValueError("Circle does not support the '3d' style, use '1d' or '2d'.")
        super().draw(sample, style, config=config, transform=transform, filepath=filepath)

    def __init__(
        self,
        *,
        radius: float = 1.0 / math.sqrt(2.0 * math.pi),
        margin: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Configure a circle classification generator.

        Args:
            radius (float): Radius of the circle centred at ``(0.5, 0.5)``.
                Defaults to ``1 / sqrt(2 pi)``, which balances the two classes.
            margin (float): Width of the empty band straddling the circle
                boundary, inside which no points are sampled. Defaults to ``0.0``.
            seed (int | None): Seed for the random number generator. Defaults to ``None``.

        Raises:
            ValueError: If ``radius`` is not positive, ``margin`` is negative, or
                the band leaves no room for one of the two classes.
        """
        super().__init__(seed=seed)
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}.")
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}.")
        if margin / 2.0 >= radius or radius + margin / 2.0 >= _HALF_DIAGONAL:
            raise ValueError(
                f"a margin of {margin} around a circle of radius {radius} leaves no room for one of the two classes."
            )
        self._radius = radius
        self._margin = margin

    @property
    def radius(self) -> float:
        """Return the radius of the circle.

        Returns:
            float: The configured radius.
        """
        return self._radius

    @property
    def margin(self) -> float:
        """Return the width of the empty band around the circle boundary.

        Returns:
            float: The configured margin.
        """
        return self._margin

    def generate(self, npoints: int) -> DatasetSample:
        """
        Sample labelled points from the unit square.

        Points falling inside the ``margin`` band around the circle are rejected
        and redrawn, so exactly ``npoints`` points are returned.

        Args:
            npoints (int): Number of points to produce.

        Returns:
            DatasetSample: The points as ``inputs``, shaped ``(npoints, 2)``, and
                their labels as ``targets``, shaped ``(npoints, 1)``, with ``1.0``
                inside the circle and ``0.0`` outside.

        Raises:
            ValueError: If ``npoints`` is not positive.
        """
        if npoints < 1:
            raise ValueError(f"npoints must be a positive integer, got {npoints}.")

        rng = self._rng()
        half_margin = self._margin / 2.0
        points = np.empty((0, 2), dtype=np.float64)
        while len(points) < npoints:
            batch = rng.uniform(0.0, 1.0, size=(npoints, 2))
            distance = np.linalg.norm(batch - 0.5, axis=1)
            points = np.concatenate([points, batch[np.abs(distance - self._radius) >= half_margin]])

        points = points[:npoints]
        inside = np.linalg.norm(points - 0.5, axis=1) < self._radius
        return DatasetSample(inputs=points, targets=inside.astype(np.float64).reshape(-1, 1))
