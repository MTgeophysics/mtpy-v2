"""Bokeh implementation of phase tensor element plotting."""

from __future__ import annotations

import numpy as np

from mtpy.imaging.mtplot_tools import PlotBase


try:
    from bokeh.io import show as bokeh_show
    from bokeh.layouts import Column
    from bokeh.models import (
        BasicTicker,
        ColorBar,
        ColumnDataSource,
        FixedTicker,
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
except ImportError:  # pragma: no cover - optional dependency
    bokeh_show = None
    Column = None
    BasicTicker = None
    ColorBar = None
    ColumnDataSource = None
    FixedTicker = None
    LinearAxis = None
    LinearColorMapper = None
    Range1d = None
    Whisker = None
    Cividis256 = None
    Inferno256 = None
    Magma256 = None
    Plasma256 = None
    Turbo256 = None
    Viridis256 = None
    figure = None


class PlotPhaseTensor(PlotBase):
    """Plot phase tensor elements in a Bokeh multi-panel layout."""

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
        kwargs["ellipse_size"] = kwargs.get("ellipse_size", 2)

        self._rotation_angle = 0
        super().__init__(**kwargs)

        self.pt = pt_object
        self.station = station
        self.skew_cutoff = 3
        self.ellip_cutoff = 0.1

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

        for key, value in kwargs.items():
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

    def _make_log_tick_overrides(self):
        dmin = int(np.floor(np.log10(self.x_limits[0])))
        dmax = int(np.ceil(np.log10(self.x_limits[1])))
        periods = np.power(10.0, np.arange(dmin, dmax + 1, dtype=float))
        ticks = np.log10(periods) * float(self.ellipse_spacing)
        labels = {float(tick): f"{period:g}" for tick, period in zip(ticks, periods)}
        return ticks, labels

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

        xmax = float(np.nanmax(np.abs(phimax))) if np.isfinite(phimax).any() else 1.0
        if xmax <= 0:
            xmax = 1.0

        src_pt = ColumnDataSource(
            data={
                "x": np.log10(period) * float(self.ellipse_spacing),
                "y": np.zeros_like(period),
                "width": np.clip(np.abs(phimax) / xmax, 0, None)
                * float(self.ellipse_size),
                "height": np.clip(np.abs(phimin) / xmax, 0, None)
                * float(self.ellipse_size),
                "angle": np.deg2rad(azimuth),
                "color": color_array,
            }
        )

        mapper = LinearColorMapper(
            palette=self._palette_from_name(self.ellipse_cmap),
            low=float(self.ellipse_range[0]),
            high=float(self.ellipse_range[1]),
        )
        fig_pt.ellipse(
            x="x",
            y="y",
            width="width",
            height="height",
            angle="angle",
            source=src_pt,
            fill_color={"field": "color", "transform": mapper},
            line_color="black",
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

        pt_ticks, pt_tick_labels = self._make_log_tick_overrides()
        fig_pt.xaxis.ticker = FixedTicker(ticks=[float(v) for v in pt_ticks])
        fig_pt.xaxis.major_label_overrides = pt_tick_labels
        fig_pt.x_range = Range1d(
            start=float(np.log10(self.x_limits[0]) * self.ellipse_spacing),
            end=float(np.log10(self.x_limits[1]) * self.ellipse_spacing),
        )
        fig_pt.y_range = Range1d(
            start=-1.5 * self.ellipse_size, end=1.5 * self.ellipse_size
        )
        fig_pt.yaxis.visible = False
        fig_pt.xaxis.axis_label = "Period (s)"
        fig_pt.grid.grid_line_alpha = 0.25

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
