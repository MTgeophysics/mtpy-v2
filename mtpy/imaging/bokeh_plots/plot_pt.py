"""Bokeh implementation of phase tensor element plotting."""

from __future__ import annotations

import importlib

import numpy as np
import param
from bokeh.io import show as bokeh_show
from bokeh.layouts import Column
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    LinearAxis,
    LinearColorMapper,
    Range1d,
    Whisker,
)
from bokeh.palettes import (
    Cividis256,
    Inferno256,
    Magma256,
    Plasma256,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure

from mtpy.imaging.bokeh_plots.bokeh_plot_base import (
    _ELLIPSE_COLORBY_OPTIONS,
    BokehPlotBase,
)


_PALETTE_OPTIONS = ["turbo", "viridis", "magma", "inferno", "plasma", "cividis"]


class PlotPhaseTensor(BokehPlotBase):
    """Plot phase tensor elements in a Bokeh multi-panel layout."""

    ellipse_colorby = param.ObjectSelector(
        default="phimin",
        objects=_ELLIPSE_COLORBY_OPTIONS,
        doc="PT property used to color ellipses",
    )
    ellipse_cmap = param.ObjectSelector(
        default="turbo",
        objects=_PALETTE_OPTIONS,
        doc="Color palette used for PT ellipse fill",
    )

    _MARKER_MAP = {
        "o": "circle",
        "s": "square",
        "v": "triangle",
        "d": "diamond",
        "^": "inverted_triangle",
        "*": "asterisk",
        "+": "cross",
        "x": "x",
    }

    def __init__(self, pt_object, station=None, **kwargs):
        kwargs["ellipse_size"] = kwargs.get("ellipse_size", 2.0)

        param_names = set(type(self).param)
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in param_names}

        self._rotation_angle = 0
        super().__init__(**param_kwargs)

        self.marker_size = kwargs.get("marker_size", 5)

        self.pt = pt_object
        self.station = station
        self.skew_cutoff = 3
        self.ellip_cutoff = 0.1

        self.pt_limits = None
        self.skew_limits = None
        self.strike_limits = None

        self.skew_color = "#008000"
        self.skew_marker = "o"
        self.strike_pt_color = "#000000"
        self.strike_pt_marker = "o"

        self.det_color = "#610968"

        self.cb_position = (0.045, 0.78, 0.015, 0.12)

        self.subplot_left = 0.1
        self.subplot_right = 0.92
        self.subplot_bottom = 0.1
        self.subplot_top = 0.95
        self.subplot_wspace = 0.1
        self.subplot_hspace = 0.15

        self.ellipse_spacing = 10

        self.layout = None
        self.fig = None
        self.figures = {}

        for key, value in other_kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot(show=True)

    @property
    def rotation_angle(self):
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, theta_r):
        self._rotation_angle = theta_r
        if theta_r != 0 and self.pt is not None:
            self.pt.rotate(theta_r, inplace=True)

    def _rotate_pt(self, rotation_angle):
        self.pt.rotate(rotation_angle)

    def _require_bokeh(self):
        if (
            figure is None
            or Column is None
            or ColumnDataSource is None
            or LinearColorMapper is None
            or ColorBar is None
            or BasicTicker is None
            or FixedTicker is None
            or Range1d is None
            or Whisker is None
        ):
            raise ImportError(
                "Bokeh is required for PlotPhaseTensor. Install with `pip install bokeh`."
            )

    def _palette_from_name(self, name):
        if name is None:
            return Turbo256

        lname = str(name).lower()
        if "magma" in lname:
            return Magma256
        if "inferno" in lname:
            return Inferno256
        if "plasma" in lname:
            return Plasma256
        if "viridis" in lname:
            return Viridis256
        if "cividis" in lname:
            return Cividis256
        return Turbo256

    def _marker_name(self, marker):
        return self._MARKER_MAP.get(marker, "circle")

    @staticmethod
    def _to_hex(color):
        if isinstance(color, str):
            short_map = {
                "k": "#000000",
                "w": "#ffffff",
                "r": "#ff0000",
                "g": "#008000",
                "b": "#0000ff",
                "c": "#00ffff",
                "m": "#ff00ff",
                "y": "#ffff00",
            }
            return short_map.get(color.lower(), color)
        if isinstance(color, tuple) and len(color) == 3:
            r, g, b = [int(np.clip(c, 0, 1) * 255) for c in color]
            return f"#{r:02x}{g:02x}{b:02x}"
        return color

    def _add_whiskers(self, fig_obj, source, color, y_range_name=None):
        whisker = Whisker(
            base="period",
            upper="high",
            lower="low",
            source=source,
            line_color=color,
            line_width=max(self.lw, 1),
        )
        whisker.upper_head.size = 4
        whisker.lower_head.size = 4
        whisker.upper_head.line_color = color
        whisker.lower_head.line_color = color
        if y_range_name is not None:
            whisker.y_range_name = y_range_name
        fig_obj.add_layout(whisker)

    def _safe_period(self):
        frequency = np.asarray(self.pt.frequency, dtype=float)
        valid = np.isfinite(frequency) & (frequency > 0)
        period = 1.0 / frequency[valid]
        return period, valid

    def _compute_pt_limits(self, phimin, phimax):
        if self.pt_limits is None:
            pt_min = np.nan_to_num(phimin)
            pt_max = np.nan_to_num(phimax)
            limits = [
                min([pt_max.min(), pt_min.min()]) - 3,
                max([pt_max.max(), pt_min.max()]) + 3,
            ]
            if limits[0] < -10:
                limits[0] = -9.9
            if limits[1] > 100:
                limits[1] = 99.99
            self.pt_limits = limits

    def _make_log_tick_overrides(self, x_spacing=None):
        dmin = int(np.floor(np.log10(self.x_limits[0])))
        dmax = int(np.ceil(np.log10(self.x_limits[1])))
        periods = np.power(10.0, np.arange(dmin, dmax + 1, dtype=float))
        spacing = float(self.ellipse_spacing) if x_spacing is None else float(x_spacing)
        ticks = np.log10(periods) * spacing
        labels = {float(tick): f"{period:g}" for tick, period in zip(ticks, periods)}
        return ticks, labels

    def _get_colorby_limits(self, colorby=None):
        """Return finite min/max limits for the requested PT colorby array."""
        current_colorby = self.ellipse_colorby if colorby is None else colorby
        original_colorby = self.ellipse_colorby
        try:
            self.ellipse_colorby = current_colorby
            arr = np.asarray(self.get_pt_color_array(self.pt), dtype=float)
        finally:
            self.ellipse_colorby = original_colorby

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float(self.ellipse_range[0]), float(self.ellipse_range[1])

        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        return lo, hi

    def plot(self, rotation_angle=None, show=True):
        """Create and return a Bokeh layout with phase tensor elements."""

        self._require_bokeh()

        if self.x_limits is None:
            self.x_limits = self.set_period_limits(1.0 / self.pt.frequency)

        if rotation_angle is not None:
            self._rotate_pt(rotation_angle)

        period, valid = self._safe_period()
        if period.size == 0:
            raise ValueError("No valid phase tensor periods to plot")

        phimin = np.asarray(self.pt.phimin, dtype=float)[valid]
        phimax = np.asarray(self.pt.phimax, dtype=float)[valid]
        phimin_err = np.asarray(self.pt.phimin_error, dtype=float)[valid]
        phimax_err = np.asarray(self.pt.phimax_error, dtype=float)[valid]

        azimuth = np.asarray(self.pt.azimuth, dtype=float)[valid]
        azimuth_err = np.asarray(self.pt.azimuth_error, dtype=float)[valid]
        skew = np.asarray(self.pt.beta, dtype=float)[valid]
        skew_err = np.asarray(self.pt.beta_error, dtype=float)[valid]
        ellipticity = np.asarray(self.pt.ellipticity, dtype=float)[valid]
        ellipticity_err = np.asarray(self.pt.ellipticity_error, dtype=float)[valid]

        color_array = np.asarray(self.get_pt_color_array(self.pt), dtype=float)[valid]

        xy_color = self._to_hex(self.xy_color)
        yx_color = self._to_hex(self.yx_color)
        skew_color = self._to_hex(self.skew_color)
        det_color = self._to_hex(self.det_color)
        strike_color = self._to_hex(self.strike_pt_color)

        self._compute_pt_limits(phimin, phimax)

        x_range_log = Range1d(
            start=float(self.x_limits[0]), end=float(self.x_limits[1])
        )

        self.figures = {}

        fig_pt = figure(
            title=(
                f"Phase Tensor Elements for: {self.station}"
                if self.station is not None
                else "Phase Tensor Elements"
            ),
            width=900,
            height=250,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
        )

        fig_width = fig_pt.width if fig_pt.width else 900
        fig_height = fig_pt.height if fig_pt.height else 250
        x_log_range = np.log10(self.x_limits[1]) - np.log10(self.x_limits[0])
        pt_ylim = 1.5 * self.ellipse_size
        adjusted_spacing = (2 * pt_ylim * fig_width) / (x_log_range * fig_height)

        x = np.log10(period) * adjusted_spacing
        valid_pt = (
            np.isfinite(x)
            & np.isfinite(phimin)
            & np.isfinite(phimax)
            & (phimax > 0)
            & np.isfinite(azimuth)
            & np.isfinite(color_array)
        )

        phimax_station = np.nanmax(phimax[valid_pt]) if np.any(valid_pt) else np.nan
        if np.isfinite(phimax_station) and phimax_station > 0:
            scaling = self.ellipse_size / phimax_station
        else:
            scaling = 0.0

        n_valid = int(np.count_nonzero(valid_pt))
        width = phimax[valid_pt] * scaling
        height = phimin[valid_pt] * scaling
        angle = np.deg2rad(90.0 - azimuth[valid_pt])

        src_pt = ColumnDataSource(
            data={
                "x": x[valid_pt],
                "y": np.zeros(n_valid),
                "width": width,
                "height": height,
                "angle": angle,
                "color_value": color_array[valid_pt],
                "phimin": phimin[valid_pt],
                "phimax": phimax[valid_pt],
                "azimuth": azimuth[valid_pt],
                "period": period[valid_pt],
            }
        )

        mapper = LinearColorMapper(
            palette=self._palette_from_name(self.ellipse_cmap),
            low=float(self.ellipse_range[0]),
            high=float(self.ellipse_range[1]),
        )
        pt_renderer = fig_pt.ellipse(
            x="x",
            y="y",
            width="width",
            height="height",
            angle="angle",
            source=src_pt,
            fill_color={"field": "color_value", "transform": mapper},
            line_color="#222222",
            line_width=0.6,
            fill_alpha=0.9,
        )

        fig_pt.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title=self.cb_label_dict[self.ellipse_colorby],
            ),
            "right",
        )

        pt_ticks, pt_tick_labels = self._make_log_tick_overrides(
            x_spacing=adjusted_spacing
        )
        fig_pt.xaxis.ticker = FixedTicker(ticks=[float(v) for v in pt_ticks])
        fig_pt.xaxis.major_label_overrides = pt_tick_labels
        x_pad = 0.5 * self.ellipse_size
        fig_pt.x_range = Range1d(
            start=float(np.log10(self.x_limits[0]) * adjusted_spacing - x_pad),
            end=float(np.log10(self.x_limits[1]) * adjusted_spacing + x_pad),
        )
        fig_pt.y_range = Range1d(start=-pt_ylim, end=pt_ylim)
        fig_pt.yaxis.visible = False
        fig_pt.xaxis.axis_label = "Period (s)"
        fig_pt.grid.grid_line_alpha = 0.25
        fig_pt.add_tools(
            HoverTool(
                renderers=[pt_renderer],
                tooltips=[
                    ("Period (s)", "@period{0.000}"),
                    (f"{self.ellipse_colorby}", "@color_value{0.0}"),
                    ("phimin (deg)", "@phimin{0.0}"),
                    ("phimax (deg)", "@phimax{0.0}"),
                    ("azimuth (deg)", "@azimuth{0.0}"),
                ],
            )
        )

        fig_phase = figure(
            title="",
            x_axis_type="log",
            width=900,
            height=230,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_range=x_range_log,
            y_axis_label="Phase (deg)",
        )

        src_min = ColumnDataSource(
            data={
                "period": period,
                "value": phimin,
                "low": phimin - phimin_err,
                "high": phimin + phimin_err,
            }
        )
        src_max = ColumnDataSource(
            data={
                "period": period,
                "value": phimax,
                "low": phimax - phimax_err,
                "high": phimax + phimax_err,
            }
        )
        fig_phase.line("period", "value", source=src_min, color=xy_color)
        fig_phase.scatter(
            "period",
            "value",
            source=src_min,
            marker=self._marker_name(self.xy_marker),
            size=max(int(self.marker_size * 2), 5),
            color=xy_color,
            legend_label="phimin",
        )
        fig_phase.line("period", "value", source=src_max, color=yx_color)
        fig_phase.scatter(
            "period",
            "value",
            source=src_max,
            marker=self._marker_name(self.yx_marker),
            size=max(int(self.marker_size * 2), 5),
            color=yx_color,
            legend_label="phimax",
        )
        self._add_whiskers(fig_phase, src_min, xy_color)
        self._add_whiskers(fig_phase, src_max, yx_color)
        fig_phase.y_range = Range1d(
            start=float(self.pt_limits[0]), end=float(self.pt_limits[1])
        )
        fig_phase.legend.location = "bottom_left"
        fig_phase.grid.grid_line_alpha = 0.25

        fig_skew = figure(
            title="",
            x_axis_type="log",
            width=900,
            height=230,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_range=x_range_log,
            y_axis_label="Skew (deg)",
        )

        if self.skew_limits is None:
            self.skew_limits = (-10, 10)

        src_skew = ColumnDataSource(
            data={
                "period": period,
                "value": skew,
                "low": skew - skew_err,
                "high": skew + skew_err,
            }
        )
        fig_skew.line("period", "value", source=src_skew, color=skew_color)
        fig_skew.scatter(
            "period",
            "value",
            source=src_skew,
            marker=self._marker_name(self.skew_marker),
            size=max(int(self.marker_size * 2), 5),
            color=skew_color,
        )
        self._add_whiskers(fig_skew, src_skew, skew_color)
        fig_skew.line(
            [float(self.x_limits[0]), float(self.x_limits[1])],
            [float(self.skew_cutoff), float(self.skew_cutoff)],
            line_dash="dashed",
            color=skew_color,
        )
        fig_skew.line(
            [float(self.x_limits[0]), float(self.x_limits[1])],
            [-float(self.skew_cutoff), -float(self.skew_cutoff)],
            line_dash="dashed",
            color=skew_color,
        )
        fig_skew.y_range = Range1d(
            start=float(self.skew_limits[0]), end=float(self.skew_limits[1])
        )

        fig_skew.extra_y_ranges = {"ellip": Range1d(start=0.0, end=1.0)}
        fig_skew.add_layout(
            LinearAxis(y_range_name="ellip", axis_label="Ellipticity"),
            "right",
        )

        src_ellip = ColumnDataSource(
            data={
                "period": period,
                "value": ellipticity,
                "low": ellipticity - ellipticity_err,
                "high": ellipticity + ellipticity_err,
            }
        )
        fig_skew.line(
            "period",
            "value",
            source=src_ellip,
            color=det_color,
            y_range_name="ellip",
        )
        fig_skew.scatter(
            "period",
            "value",
            source=src_ellip,
            marker=self._marker_name(self.det_marker),
            size=max(int(self.marker_size * 2), 5),
            color=det_color,
            y_range_name="ellip",
        )
        self._add_whiskers(fig_skew, src_ellip, det_color, y_range_name="ellip")
        fig_skew.line(
            [float(self.x_limits[0]), float(self.x_limits[1])],
            [float(self.ellip_cutoff), float(self.ellip_cutoff)],
            line_dash="dashed",
            color=det_color,
            y_range_name="ellip",
        )
        fig_skew.grid.grid_line_alpha = 0.25

        fig_strike = figure(
            title="",
            x_axis_type="log",
            width=900,
            height=230,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_range=x_range_log,
            x_axis_label="Period (s)",
            y_axis_label="Strike (deg)",
        )

        src_strike = ColumnDataSource(
            data={
                "period": period,
                "value": azimuth,
                "low": azimuth - azimuth_err,
                "high": azimuth + azimuth_err,
            }
        )
        fig_strike.line("period", "value", source=src_strike, color=strike_color)
        fig_strike.scatter(
            "period",
            "value",
            source=src_strike,
            marker=self._marker_name(self.strike_pt_marker),
            size=max(int(self.marker_size * 2), 5),
            color=strike_color,
        )
        self._add_whiskers(fig_strike, src_strike, strike_color)

        if self.strike_limits is None:
            self.strike_limits = (0, 359.99)

        fig_strike.y_range = Range1d(
            start=float(self.strike_limits[0]),
            end=float(self.strike_limits[1]),
        )
        fig_strike.grid.grid_line_alpha = 0.25

        self.figures = {
            "pt": fig_pt,
            "phase": fig_phase,
            "skew": fig_skew,
            "strike": fig_strike,
        }

        self.layout = Column(fig_pt, fig_phase, fig_skew, fig_strike)
        self.fig = self.layout

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def make_panel(self, sizing_mode="stretch_width"):
        """Create a standalone interactive Panel app for phase tensor plots."""
        try:
            pn = importlib.import_module("panel")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Panel is required for make_panel(). Install with `pip install panel`."
            ) from exc

        if self.layout is None:
            self.plot(show=False)

        bokeh_pane = pn.pane.Bokeh(self.layout, sizing_mode=sizing_mode)

        colorby_widget = pn.widgets.Select(
            name="Phase Tensor Color By",
            options=list(_ELLIPSE_COLORBY_OPTIONS),
            value=self.ellipse_colorby,
            width=260,
        )

        initial_min, initial_max = self._get_colorby_limits(self.ellipse_colorby)

        color_min_widget = pn.widgets.NumberInput(
            name="Ellipse Min",
            value=float(initial_min),
            step=0.1,
            width=120,
        )
        color_max_widget = pn.widgets.NumberInput(
            name="Ellipse Max",
            value=float(initial_max),
            step=0.1,
            width=120,
        )

        palette_value = str(self.ellipse_cmap).lower()
        if palette_value not in _PALETTE_OPTIONS:
            palette_value = "turbo"

        palette_widget = pn.widgets.Select(
            name="Ellipse Palette",
            options=_PALETTE_OPTIONS,
            value=palette_value,
            width=220,
        )

        marker_size_widget = pn.widgets.IntSlider(
            name="Marker Size",
            start=2,
            end=20,
            value=int(self.marker_size),
            width=220,
        )

        limit_status = pn.pane.Markdown("", styles={"color": "#666"})

        def _on_colorby_change(event):
            lo, hi = self._get_colorby_limits(event.new)
            color_min_widget.value = float(lo)
            color_max_widget.value = float(hi)
            _refresh()

        def _refresh(_event=None):
            self.ellipse_colorby = colorby_widget.value
            self.ellipse_cmap = palette_widget.value
            self.marker_size = int(marker_size_widget.value)
            lo = float(color_min_widget.value)
            hi = float(color_max_widget.value)
            if hi <= lo:
                hi = lo + 1.0
                color_max_widget.value = hi
                limit_status.object = (
                    f"⚠️ Ellipse Max must be greater than Ellipse Min. "
                    f"Adjusted max to {hi:.3g}."
                )
                limit_status.styles = {"color": "#7a5200"}
            else:
                limit_status.object = ""
                limit_status.styles = {"color": "#666"}
            self.ellipse_range = (lo, hi, self.ellipse_range[2])
            self.plot(show=False)
            bokeh_pane.object = self.layout

        colorby_widget.param.watch(_on_colorby_change, "value")
        palette_widget.param.watch(_refresh, "value")
        marker_size_widget.param.watch(_refresh, "value")
        color_min_widget.param.watch(_refresh, "value")
        color_max_widget.param.watch(_refresh, "value")

        title = self.plot_title if self.plot_title else (self.station or "Phase Tensor")
        controls = pn.Row(
            pn.Column(
                pn.pane.Markdown("**Color By**"),
                colorby_widget,
                pn.Row(color_min_widget, color_max_widget),
            ),
            pn.Column(pn.pane.Markdown("**Palette**"), palette_widget),
            pn.Column(pn.pane.Markdown("**Marker Size**"), marker_size_widget),
        )

        return pn.Column(
            pn.pane.Markdown(f"## {title}"),
            controls,
            limit_status,
            bokeh_pane,
            sizing_mode=sizing_mode,
        )

    def panel(self, sizing_mode="stretch_width"):
        """Alias used by MTDataApp plot dispatch."""
        return self.make_panel(sizing_mode=sizing_mode)
