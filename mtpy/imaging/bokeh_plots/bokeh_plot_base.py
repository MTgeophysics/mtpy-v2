# -*- coding: utf-8 -*-
"""Base class for Bokeh-specific MT plot classes.

Provides a ``param.Parameterized`` foundation that is entirely independent
from the matplotlib ``PlotSettings``/``PlotBase`` hierarchy.  All Bokeh plot
classes should inherit from ``BokehPlotBase`` rather than ``PlotBase``.
"""

from __future__ import annotations

import numpy as np


try:
    import param
except ImportError:  # pragma: no cover
    raise ImportError(
        "param is required for Bokeh plot classes. " "Install with `pip install param`."
    )

try:
    from loguru import logger as _loguru_logger
except ImportError:  # pragma: no cover
    import logging

    _loguru_logger = logging.getLogger(__name__)

_ELLIPSE_COLORBY_OPTIONS = [
    "phimin",
    "phiminang",
    "phimax",
    "phimaxang",
    "phidet",
    "skew",
    "skew_seg",
    "normalized_skew",
    "normalized_skew_seg",
    "ellipticity",
    "strike",
    "azimuth",
]

_MARKER_OPTIONS = ["o", "s", "v", "d", "^", "*", "+", "x"]


class BokehPlotBase(param.Parameterized):
    """Common parameters shared by all Bokeh MT plot classes.

    Provides a ``param.Parameterized`` base independent of the matplotlib
    ``PlotSettings``/``PlotBase`` hierarchy.  Covers display options used by
    the station map as well as MT response / tipper / phase-tensor plots.

    Parameters
    ----------
    show_plot : bool
        Whether to call ``bokeh.io.show`` automatically after ``plot()``,
        by default False.
    plot_title : str
        Title text for the figure, by default "".
    marker : str
        Marker shape for station / scatter glyphs, by default ``"o"``.
    marker_size : int
        Marker diameter in screen pixels, by default 10.
    marker_color : str
        Marker fill color as a CSS hex string, by default ``"#0000ff"``.
    font_size : int
        Base font size in points, by default 10.
    text_color : str
        Text / label color, by default ``"#000000"``.
    text_size : int
        Station-label font size in points, by default 10.
    text_y_pad : float
        Vertical offset (data units) for station labels, by default 8.0.
    x_limits : tuple | None
        Manual x-axis limits ``(min, max)``, or ``None`` for auto-scale.
    y_limits : tuple | None
        Manual y-axis limits ``(min, max)``, or ``None`` for auto-scale.
    lw : float
        Default line width for data curves, by default 1.0.
    xy_color : str
        Color for Zxy component, by default ``"#4059bf"``.
    yx_color : str
        Color for Zyx component, by default ``"#bf4040"``.
    xx_color : str
        Color for Zxx component, by default ``"#5599cc"``.
    yy_color : str
        Color for Zyy component, by default ``"#cc5555"``.
    det_color : str
        Color for det(Z) component, by default ``"#40bf40"``.
    xy_marker : str
        Marker for Zxy component, by default ``"s"``.
    yx_marker : str
        Marker for Zyx component, by default ``"o"``.
    xx_marker : str
        Marker for Zxx component, by default ``"d"``.
    yy_marker : str
        Marker for Zyy component, by default ``"^"``.
    det_marker : str
        Marker for det(Z) component, by default ``"v"``.
    res_limits : tuple | None
        Resistivity axis limits ``(min, max)``, or ``None`` for auto.
    plot_z : bool
        Whether to plot impedance data, by default True.
    plot_tipper : str
        Tipper plotting mode: ``"n"`` (none), ``"y"`` (both),
        ``"yri"`` (real + imag), by default ``"n"``.
    plot_pt : bool
        Whether to plot phase-tensor ellipses, by default False.
    arrow_lw : float
        Line width for induction-vector arrows, by default 1.25.
    arrow_color_real : str
        Color for real induction vector, by default ``"#000000"`` (black).
    arrow_color_imag : str
        Color for imaginary induction vector, by default ``"#00ffff"`` (cyan).
    arrow_direction : int
        Arrow direction convention: 0 = away from conductor,
        1 = Parkinson (toward conductor), by default 1.
    ellipse_size : float
        Size of phase-tensor ellipses in data units, by default 2.0.
    ellipse_range : tuple
        Color range for ellipse coloring ``(min, max, step)``,
        by default ``(0, 90, 10)``.
    ellipse_colorby : str
        Phase-tensor property used to color ellipses, by default ``"phimin"``.
    """

    # ── display ──────────────────────────────────────────────────────────────
    show_plot = param.Boolean(default=False, doc="Auto-show after plot()")
    plot_title = param.String(default="", doc="Figure title")

    # ── station-map markers ───────────────────────────────────────────────────
    marker = param.ObjectSelector(
        default="o",
        objects=_MARKER_OPTIONS,
        doc="Marker shape key",
    )
    marker_size = param.Integer(default=10, bounds=(1, 50), doc="Marker size (px)")
    marker_color = param.Color(default="#0000ff", doc="Marker fill color")

    # ── text / labels ─────────────────────────────────────────────────────────
    font_size = param.Integer(default=10, bounds=(4, 30), doc="Base font size (pt)")
    text_color = param.Color(default="#000000", doc="Label text color")
    text_size = param.Integer(default=10, bounds=(4, 30), doc="Label font size (pt)")
    text_y_pad = param.Number(default=8.0, doc="Label vertical offset (data units)")

    # ── axis limits ───────────────────────────────────────────────────────────
    x_limits = param.Parameter(default=None, doc="X-axis limits (min, max) or None")
    y_limits = param.Parameter(default=None, doc="Y-axis limits (min, max) or None")
    res_limits = param.Parameter(
        default=None, doc="Resistivity axis limits (min, max) or None"
    )

    # ── MT response line/scatter style ────────────────────────────────────────
    lw = param.Number(default=1.0, bounds=(0, 10), doc="Line width")
    xy_color = param.Color(default="#4059bf", doc="Zxy component color")
    yx_color = param.Color(default="#bf4040", doc="Zyx component color")
    xx_color = param.Color(default="#5599cc", doc="Zxx component color")
    yy_color = param.Color(default="#cc5555", doc="Zyy component color")
    det_color = param.Color(default="#40bf40", doc="det(Z) component color")
    xy_marker = param.ObjectSelector(
        default="s", objects=_MARKER_OPTIONS, doc="Zxy marker"
    )
    yx_marker = param.ObjectSelector(
        default="o", objects=_MARKER_OPTIONS, doc="Zyx marker"
    )
    xx_marker = param.ObjectSelector(
        default="d", objects=_MARKER_OPTIONS, doc="Zxx marker"
    )
    yy_marker = param.ObjectSelector(
        default="^", objects=_MARKER_OPTIONS, doc="Zyy marker"
    )
    det_marker = param.ObjectSelector(
        default="v", objects=_MARKER_OPTIONS, doc="det(Z) marker"
    )

    # ── data visibility flags ─────────────────────────────────────────────────
    plot_z = param.Boolean(default=True, doc="Plot impedance components")
    plot_tipper = param.String(default="n", doc='Tipper mode: "n", "y", or "yri"')
    plot_pt = param.Boolean(default=False, doc="Plot phase-tensor ellipses")

    # ── induction-vector arrows ───────────────────────────────────────────────
    arrow_lw = param.Number(default=1.25, bounds=(0, 10), doc="Arrow line width")
    arrow_color_real = param.Color(default="#000000", doc="Real induction vector color")
    arrow_color_imag = param.Color(
        default="#00ffff", doc="Imaginary induction vector color"
    )
    arrow_direction = param.Integer(
        default=1,
        bounds=(0, 1),
        doc="0=away from conductor, 1=Parkinson convention",
    )

    # ── phase-tensor ellipses ─────────────────────────────────────────────────
    ellipse_size = param.Number(
        default=2.0, bounds=(0.01, 100), doc="Ellipse size in data units"
    )
    ellipse_range = param.Parameter(
        default=(0, 90, 10), doc="Ellipse color range (min, max, step)"
    )
    ellipse_colorby = param.ObjectSelector(
        default="phimin",
        objects=_ELLIPSE_COLORBY_OPTIONS,
        doc="PT property used to color ellipses",
    )

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = _loguru_logger

    # ── colorbar label dictionary ─────────────────────────────────────────────
    @property
    def cb_label_dict(self) -> dict:
        return {
            "phiminang": r"$\Phi_{min}$ (deg)",
            "phimin": r"$\Phi_{min}$ (deg)",
            "phimaxang": r"$\Phi_{max}$ (deg)",
            "phimax": r"$\Phi_{max}$ (deg)",
            "phidet": r"Det{$\Phi$} (deg)",
            "skew": r"Skew (deg)",
            "normalized_skew": r"Normalized Skew (deg)",
            "ellipticity": r"Ellipticity",
            "skew_seg": r"Skew (deg)",
            "normalized_skew_seg": r"Normalized Skew (deg)",
            "geometric_mean": r"$\sqrt{\Phi_{min} \cdot \Phi_{max}}$",
            "strike": r"Azimuth (deg)",
            "azimuth": r"Azimuth (deg)",
        }

    # ── utility methods ───────────────────────────────────────────────────────
    def set_period_limits(self, period: np.ndarray) -> tuple:
        """Return period axis limits as floor/ceil powers of 10."""
        return (
            10 ** float(np.floor(np.log10(period.min()))),
            10 ** float(np.ceil(np.log10(period.max()))),
        )

    def set_resistivity_limits(
        self, resistivity: np.ndarray, mode: str = "od", scale: str = "log"
    ) -> list:
        """Return [min, max] resistivity limits as powers of 10."""
        if mode == "od":
            res_list = [resistivity[:, 0, 1], resistivity[:, 1, 0]]
        elif mode == "d":
            res_list = [resistivity[:, 0, 0], resistivity[:, 1, 1]]
        elif mode in ("det", "det_only"):
            res_list = [resistivity]
        elif mode == "all":
            res_list = [
                resistivity[:, 0, 0],
                resistivity[:, 0, 1],
                resistivity[:, 1, 0],
                resistivity[:, 1, 1],
            ]
        else:
            res_list = [resistivity[:, 0, 1], resistivity[:, 1, 0]]

        try:

            def _nonzero_min(arr):
                nz = np.nonzero(arr)
                return float(np.nanmin(arr[nz]))

            def _nonzero_max(arr):
                nz = np.nonzero(arr)
                return float(np.nanmax(arr[nz]))

            rmin = min(_nonzero_min(r) for r in res_list)
            rmax = max(_nonzero_max(r) for r in res_list)
            limits = [
                10 ** float(np.floor(np.log10(rmin))),
                10 ** float(np.ceil(np.log10(rmax))),
            ]
        except (ValueError, TypeError):
            limits = [0.1, 10000]

        if scale == "log" and limits[0] == 0:
            limits[0] = 0.1
        return limits

    def set_phase_limits(self, phase: np.ndarray, mode: str = "od") -> tuple | list:
        """Return (min, max) phase limits in degrees."""
        if mode == "od":
            try:
                nz_xy = np.nonzero(phase[:, 0, 1])
                nz_yx = np.nonzero(phase[:, 1, 0])
                ph_min = min(
                    float(np.nanmin(phase[nz_xy, 0, 1])),
                    float(np.nanmin(phase[nz_yx, 1, 0] + 180)),
                )
                if ph_min > 0:
                    ph_min = 0
                else:
                    ph_min = round(ph_min / 5) * 5
                ph_max = max(
                    float(np.nanmax(phase[nz_xy, 0, 1])),
                    float(np.nanmax(phase[nz_yx, 1, 0] + 180)),
                )
                if ph_max < 91:
                    ph_max = 89.9
                else:
                    ph_max = round(ph_max / 5) * 5
                return (ph_min, ph_max)
            except (ValueError, TypeError):
                return [0, 90]
        elif mode == "d":
            return (-180, 180)
        elif mode in ("det", "det_only"):
            try:
                phase_det = np.linalg.det(phase)
                nz = np.nonzero(phase_det)
                limits = [float(np.amin(phase_det[nz])), float(np.amax(phase_det[nz]))]
                limits[0] = max(limits[0], -180)
                limits[1] = min(limits[1], 180)
                return limits
            except (ValueError, TypeError):
                return [-180, 180]
        return [0, 90]

    def get_pt_color_array(self, pt_object) -> np.ndarray:
        """Extract the phase-tensor property array used to color ellipses."""
        colorby = self.ellipse_colorby
        if colorby in ("phiminang", "phimin"):
            return pt_object.phimin
        if colorby in ("phimaxang", "phimax"):
            return pt_object.phimax
        if colorby == "phidet":
            return np.sqrt(abs(pt_object.det)) * (180 / np.pi)
        if colorby in ("skew", "skew_seg"):
            return pt_object.beta
        if colorby == "ellipticity":
            return pt_object.ellipticity
        if colorby in ("strike", "azimuth"):
            arr = pt_object.azimuth % 180
            arr[np.where(arr > 90)] -= 180
            return arr
        raise NameError(f"{colorby} is not a supported ellipse_colorby option")
