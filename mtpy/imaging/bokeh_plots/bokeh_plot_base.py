# -*- coding: utf-8 -*-
"""Base class for Bokeh-specific MT plot classes.

Provides a ``param.Parameterized`` foundation that is entirely independent
from the matplotlib ``PlotSettings``/``PlotBase`` hierarchy.  All Bokeh plot
classes should inherit from ``BokehPlotBase`` rather than ``PlotBase``.
"""

from __future__ import annotations


try:
    import param
except ImportError:  # pragma: no cover
    raise ImportError(
        "param is required for Bokeh plot classes. " "Install with `pip install param`."
    )


class BokehPlotBase(param.Parameterized):
    """Common parameters shared by all Bokeh MT plot classes.

    Parameters
    ----------
    show_plot : bool
        Whether to call ``bokeh.io.show`` automatically after ``plot()``,
        by default False.
    plot_title : str
        Title text for the figure, by default "".
    marker : str
        Marker shape key (``"o"``, ``"s"``, ``"v"``, ``"d"``, ``"^"``,
        ``"*"``, ``"+"``, ``"x"``), by default ``"o"``.
    marker_size : int
        Marker diameter in screen pixels, by default 10.
    marker_color : str
        Marker fill color as a CSS hex string, by default ``"#0000ff"``.
    font_size : int
        Base font size in points, by default 10.
    text_color : str
        Text / label color as a CSS hex string, by default ``"#000000"``.
    text_size : int
        Station-label font size in points, by default 10.
    text_y_pad : float
        Vertical offset (in data units) for station labels, by default 8.0.
    x_limits : tuple[float, float] | None
        Manual x-axis limits ``(min, max)``, or ``None`` for auto-scale.
    y_limits : tuple[float, float] | None
        Manual y-axis limits ``(min, max)``, or ``None`` for auto-scale.
    """

    show_plot = param.Boolean(default=False, doc="Auto-show after plot()")
    plot_title = param.String(default="", doc="Figure title")

    marker = param.ObjectSelector(
        default="o",
        objects=["o", "s", "v", "d", "^", "*", "+", "x"],
        doc="Marker shape key",
    )
    marker_size = param.Integer(default=10, bounds=(1, 50), doc="Marker size (px)")
    marker_color = param.Color(default="#0000ff", doc="Marker fill color")

    font_size = param.Integer(default=10, bounds=(4, 30), doc="Base font size (pt)")
    text_color = param.Color(default="#000000", doc="Label text color")
    text_size = param.Integer(default=10, bounds=(4, 30), doc="Label font size (pt)")
    text_y_pad = param.Number(default=8.0, doc="Label vertical offset (data units)")

    x_limits = param.Parameter(default=None, doc="X-axis limits (min, max) or None")
    y_limits = param.Parameter(default=None, doc="Y-axis limits (min, max) or None")
