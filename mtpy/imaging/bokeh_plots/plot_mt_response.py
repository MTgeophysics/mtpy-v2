"""Bokeh implementation of MT response plotting.

This module provides a first-pass Bokeh translation of the matplotlib
PlotMTResponse class for use in Panel dashboards.
"""

from __future__ import annotations

import numpy as np

from .bokeh_plot_base import BokehPlotBase


try:
    from bokeh.io import show
    from bokeh.layouts import Column, gridplot, Row
    from bokeh.models import (
        Arrow,
        BasicTicker,
        ColorBar,
        ColumnDataSource,
        CustomJSTickFormatter,
        FixedTicker,
        HoverTool,
        LinearColorMapper,
        NormalHead,
        Range1d,
        Whisker,
    )
    from bokeh.palettes import Turbo256
    from bokeh.plotting import figure
    from bokeh.transform import linear_cmap
except ImportError:  # pragma: no cover - optional dependency
    show = None
    Column = None
    Row = None
    gridplot = None
    Arrow = None
    BasicTicker = None
    ColorBar = None
    ColumnDataSource = None
    CustomJSTickFormatter = None
    FixedTicker = None
    HoverTool = None
    LinearColorMapper = None
    NormalHead = None
    Range1d = None
    Turbo256 = None
    Whisker = None
    figure = None
    linear_cmap = None


class PlotMTResponse(BokehPlotBase):
    """Plot MT apparent resistivity and phase using Bokeh.

    The class mirrors core inputs and plotting modes from
    mtpy.imaging.plot_mt_response.PlotMTResponse but returns a Bokeh layout
    object for integration in Panel applications.
    """

    _MARKER_MAP = {
        "o": "circle",
        "s": "square",
        "v": "triangle",
        "d": "diamond",
        "^": "inverted_triangle",
    }

    def __init__(
        self,
        z_object=None,
        t_object=None,
        pt_obj=None,
        station="MT Response",
        **kwargs,
    ):
        self.Z = z_object
        self.Tipper = t_object
        self.pt = pt_obj
        self.station = station
        self._basename = f"{self.station}_mt_response_bokeh"
        self.plot_num = 1
        self.rotation_angle = 0

        self.layout = None
        self.figures = {}
        self.renderers = {}
        self._log_x_figure_keys = set()
        self._linear_x_figure_keys = set()
        self._pt_x_spacing = 1.0

        # param.Parameterized raises TypeError for unknown kwargs; split them.
        param_names = set(type(self).param)
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in param_names}

        super().__init__(**param_kwargs)

        if self.Z is None:
            self.plot_z = False

        if self.Tipper is not None:
            self.plot_tipper = "yri"
        if self.pt is not None and self.plot_z:
            self.plot_pt = True

        self.plot_model_error = False

        for key, value in other_kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    @property
    def plot_model_error(self):
        """Plot model error instead of data error."""
        return self._plot_model_error

    @plot_model_error.setter
    def plot_model_error(self, value):
        if value:
            self._error_str = "model_error"
        else:
            self._error_str = "error"
        self._plot_model_error = value

    @property
    def period(self):
        """Return period array from available transfer functions."""
        if self.Z is not None and not (self.Z.period == np.array([1])).all():
            return self.Z.period
        if self.Tipper is not None and not (self.Tipper.period == np.array([1])).all():
            return self.Tipper.period
        if self.pt is not None and not (self.pt.period == np.array([1])).all():
            return self.pt.period
        raise ValueError("No transfer function data to plot. Check data.")

    @property
    def rotation_angle(self):
        """Rotation angle in degrees."""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, theta_r):
        """Apply station rotation if transfer functions are available."""
        if theta_r == 0:
            self._rotation_angle = theta_r
            return

        if self.Z is not None:
            self.Z.rotate(theta_r, inplace=True)
        if self.Tipper is not None:
            self.Tipper.rotate(theta_r, inplace=True)
        if self.Z is not None:
            self.pt = self.Z.phase_tensor
            self.pt.rotation_angle = self.Z.rotation_angle

        self._rotation_angle += theta_r

    def _require_bokeh(self):
        if (
            figure is None
            or ColumnDataSource is None
            or Whisker is None
            or LinearColorMapper is None
            or Arrow is None
            or NormalHead is None
        ):
            raise ImportError(
                "Bokeh is required for PlotMTResponse bokeh plots. "
                "Install with `pip install bokeh`"
            )

    def _has_z(self):
        if self.plot_z:
            if self.Z is None or self.Z.z is None or (self.Z.z == 0 + 0j).all():
                self.logger.info(f"No Z data for station {self.station}")
                return False
        return self.plot_z

    def _has_tipper(self):
        if self.plot_tipper.find("y") >= 0:
            if (
                self.Tipper is None
                or self.Tipper.tipper is None
                or (self.Tipper.tipper == 0 + 0j).all()
            ):
                self.logger.info(f"No Tipper data for station {self.station}")
                return "n"
        return self.plot_tipper

    def _has_pt(self):
        if self.plot_pt:
            if self.pt is None or self.pt.pt is None:
                self.logger.info(f"No PT data for station {self.station}")
                return False
        return self.plot_pt

    @staticmethod
    def _tuple_to_hex(color):
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

    def _marker_name(self, marker):
        return self._MARKER_MAP.get(marker, "circle")

    def _valid_for_log(self, x, y):
        return np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    def _get_values(self, z_obj, attr, comp):
        values = np.asarray(getattr(z_obj, f"{attr}_{comp}"), dtype=float)
        return values

    def _component_source(self, period, z_obj, comp, kind="res", yx_shift=False):
        y_attr = "res" if kind == "res" else "phase"
        e_attr = f"{y_attr}_{self._error_str}"

        y = self._get_values(z_obj, y_attr, comp)
        err = self._get_values(z_obj, e_attr, comp)
        if yx_shift:
            y = y + 180

        low = y - err
        high = y + err
        x = np.asarray(period, dtype=float)

        if kind == "res":
            valid = self._valid_for_log(x, y)
            valid = valid & np.isfinite(err) & (high > 0)
            low = np.where(low <= 0, np.nan, low)
        else:
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)

        data = {
            "period": x[valid],
            "value": y[valid],
            "low": low[valid],
            "high": high[valid],
        }
        return ColumnDataSource(data=data)

    def _add_component(
        self,
        fig,
        source,
        comp_label,
        color,
        marker,
        comp_key,
        show_error=True,
    ):
        glyph_color = self._tuple_to_hex(color)

        line_renderer = fig.line(
            x="period",
            y="value",
            source=source,
            color=glyph_color,
            line_width=max(self.lw, 1),
        )
        scatter_method = getattr(fig, self._marker_name(marker), fig.circle)
        scatter_renderer = scatter_method(
            x="period",
            y="value",
            source=source,
            size=max(int(self.marker_size * 2), 4),
            color=glyph_color,
            line_color=glyph_color,
            legend_label=comp_label,
        )

        self.renderers.setdefault(comp_key, []).extend(
            [line_renderer, scatter_renderer]
        )

        if show_error:
            whisker = Whisker(
                base="period",
                upper="high",
                lower="low",
                source=source,
                line_color=glyph_color,
                line_width=max(self.lw, 1),
            )
            whisker.upper_head.size = 4
            whisker.upper_head.line_color = glyph_color
            whisker.lower_head.size = 4
            whisker.lower_head.line_color = glyph_color
            fig.add_layout(whisker)
            self.renderers.setdefault(comp_key, []).append(whisker)

    def _make_resistivity_figure(self, x_range=None):
        kw = {}
        if x_range is not None:
            kw["x_range"] = x_range
        return figure(
            title=None,
            x_axis_type="log",
            y_axis_type="log",
            height=320,
            width=800,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            **kw,
        )

    def _make_phase_figure(self, x_range):
        return figure(
            title=None,
            x_axis_type="log",
            x_range=x_range,
            height=250,
            width=800,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

    def _make_tipper_figure(self, width=800):
        return figure(
            title="Tipper",
            x_axis_type="linear",
            height=220,
            width=width,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

    def _make_pt_figure(self, width=800):
        return figure(
            title="Phase Tensor",
            x_axis_type="linear",
            height=240,
            width=width,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

    def _apply_log_period_ticks(self, fig, x_spacing=1.0):
        """Replace a linear-x axis with 10^n period labels.

        Parameters
        ----------
        x_spacing : float
            Multiplier applied to log10(period) positions. Use the same value
            that was used to compute ellipse x-coordinates so tick positions
            align with plotted data.
        """
        pmin_log = np.log10(float(self.x_limits[0]))
        pmax_log = np.log10(float(self.x_limits[1]))
        ticks = [
            t * x_spacing
            for t in range(int(np.floor(pmin_log)), int(np.ceil(pmax_log)) + 1)
        ]
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
        fig.xaxis.formatter = CustomJSTickFormatter(
            code="""
            var superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
            var exp = String(Math.round(tick));
            var result = '10';
            for (var i = 0; i < exp.length; i++) {
                if (exp[i] === '-') result += '⁻';
                else result += superscripts[parseInt(exp[i])];
            }
            return result;
            """
        )
        fig.xaxis.axis_label = "Period (s)"

    def _format_res_axis(self, fig):
        fig.yaxis.axis_label = "App. Res. (Ohm m)"
        fig.xaxis.visible = False
        fig.grid.grid_line_alpha = 0.25

    def _format_phase_axis(self, fig):
        fig.yaxis.axis_label = "Phase (deg)"
        fig.xaxis.axis_label = "Period (s)"
        fig.grid.grid_line_alpha = 0.25

    def _set_axis_limits(self, fig, y_limits):
        if self.x_limits is not None:
            fig.x_range.start = float(self.x_limits[0])
            fig.x_range.end = float(self.x_limits[1])
        if y_limits is not None:
            fig.y_range.start = float(y_limits[0])
            fig.y_range.end = float(y_limits[1])

    def _add_hover(self, fig):
        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Period (s)", "@period{0.000}"),
                    ("Value", "@value{0.000}"),
                    ("Low", "@low{0.000}"),
                    ("High", "@high{0.000}"),
                ]
            )
        )

    def _tipper_vectors(self):
        period = np.asarray(1.0 / self.Tipper.frequency, dtype=float)
        txr = np.asarray(
            self.Tipper.mag_real
            * np.cos(
                np.deg2rad(-self.Tipper.angle_real) + self.arrow_direction * np.pi
            ),
            dtype=float,
        )
        tyr = np.asarray(
            self.Tipper.mag_real
            * np.sin(
                np.deg2rad(-self.Tipper.angle_real) + self.arrow_direction * np.pi
            ),
            dtype=float,
        )
        txi = np.asarray(
            self.Tipper.mag_imag
            * np.cos(
                np.deg2rad(-self.Tipper.angle_imag) + self.arrow_direction * np.pi
            ),
            dtype=float,
        )
        tyi = np.asarray(
            self.Tipper.mag_imag
            * np.sin(
                np.deg2rad(-self.Tipper.angle_imag) + self.arrow_direction * np.pi
            ),
            dtype=float,
        )

        valid = (
            np.isfinite(period)
            & (period > 0)
            & np.isfinite(txr)
            & np.isfinite(tyr)
            & np.isfinite(txi)
            & np.isfinite(tyi)
        )
        period = period[valid]
        txr = txr[valid]
        tyr = tyr[valid]
        txi = txi[valid]
        tyi = tyi[valid]

        log_period = np.log10(period)

        x_end_real = log_period + txr * log_period
        x_end_imag = log_period + txi * log_period

        return {
            "x0": log_period,
            "y0": np.zeros_like(log_period),
            "xr": x_end_real,
            "yr": tyr,
            "xi": x_end_imag,
            "yi": tyi,
            "tyr": tyr,
            "tyi": tyi,
            "period": period,
        }

    def _plot_tipper(self, tip_fig):
        vectors = self._tipper_vectors()
        if vectors["x0"].size == 0:
            self.logger.info("No valid tipper vectors to plot.")
            return
        period_labels = [f"{pp:.3g}" for pp in vectors["period"]]

        source = ColumnDataSource(
            data={
                "x0": vectors["x0"],
                "y0": vectors["y0"],
                "xr": vectors["xr"],
                "yr": vectors["yr"],
                "xi": vectors["xi"],
                "yi": vectors["yi"],
                "period": period_labels,
            }
        )

        real_color = self._tuple_to_hex(self.arrow_color_real)
        imag_color = self._tuple_to_hex(self.arrow_color_imag)

        if "r" in self.plot_tipper:
            real_arrow = Arrow(
                end=NormalHead(
                    size=8,
                    fill_color=real_color,
                    line_color=real_color,
                ),
                source=source,
                x_start="x0",
                y_start="y0",
                x_end="xr",
                y_end="yr",
                line_color=real_color,
                line_width=max(self.arrow_lw * 2, 1),
            )
            tip_fig.add_layout(real_arrow)
            real_legend = tip_fig.line(
                x=[np.nan],
                y=[np.nan],
                color=real_color,
                line_width=max(self.arrow_lw * 2, 1),
                legend_label="tip real",
            )
            self.renderers.setdefault("tip_real", []).extend([real_arrow, real_legend])

        if "i" in self.plot_tipper:
            imag_arrow = Arrow(
                end=NormalHead(
                    size=8,
                    fill_color=imag_color,
                    line_color=imag_color,
                ),
                source=source,
                x_start="x0",
                y_start="y0",
                x_end="xi",
                y_end="yi",
                line_color=imag_color,
                line_width=max(self.arrow_lw * 2, 1),
                line_dash="dashed",
            )
            tip_fig.add_layout(imag_arrow)
            imag_legend = tip_fig.line(
                x=[np.nan],
                y=[np.nan],
                color=imag_color,
                line_dash="dashed",
                line_width=max(self.arrow_lw * 2, 1),
                legend_label="tip imag",
            )
            self.renderers.setdefault("tip_imag", []).extend([imag_arrow, imag_legend])

        tip_fig.line(
            x=[np.log10(self.x_limits[0]), np.log10(self.x_limits[1])],
            y=[0, 0],
            color="#444444",
            line_width=1,
            line_alpha=0.5,
        )

        tmax = min(max(np.nanmax(vectors["tyr"]), np.nanmax(vectors["tyi"])), 0.9)
        tmin = max(min(np.nanmin(vectors["tyr"]), np.nanmin(vectors["tyi"])), -0.9)
        tip_limits = (tmin - 0.1, tmax + 0.1)

        tip_fig.x_range.start = np.log10(self.x_limits[0])
        tip_fig.x_range.end = np.log10(self.x_limits[1])
        tip_fig.y_range.start = tip_limits[0]
        tip_fig.y_range.end = tip_limits[1]
        tip_fig.yaxis.axis_label = "Tipper"
        tip_fig.grid.grid_line_alpha = 0.25
        self._apply_log_period_ticks(tip_fig)

        tip_fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Period (s)", "@period"),
                    ("Real x", "@xr{0.000}"),
                    ("Real y", "@yr{0.000}"),
                    ("Imag x", "@xi{0.000}"),
                    ("Imag y", "@yi{0.000}"),
                ]
            )
        )

    def _plot_phase_tensor(self, pt_fig):
        period = np.asarray(1.0 / self.pt.frequency, dtype=float)

        # Compute adjusted x-spacing so ellipses have equal visual aspect ratio.
        # With tight y-limits at ±pt_ylim = ±1.5*ellipse_size, equal aspect means:
        #   x_data_range / fig_width = y_data_range / fig_height
        #   (x_log_range * spacing) / fig_width = (3 * ellipse_size) / fig_height
        #   spacing = (3 * ellipse_size * fig_width) / (x_log_range * fig_height)
        fig_width = pt_fig.width if pt_fig.width else 800
        fig_height = pt_fig.height if pt_fig.height else 240
        x_log_range = np.log10(self.x_limits[1]) - np.log10(self.x_limits[0])
        pt_ylim = 1.5 * self.ellipse_size
        adjusted_spacing = (2 * pt_ylim * fig_width) / (x_log_range * fig_height)
        self._pt_x_spacing = adjusted_spacing

        x = np.log10(period) * adjusted_spacing
        phimin = np.asarray(self.pt.phimin, dtype=float)
        phimax = np.asarray(self.pt.phimax, dtype=float)
        azimuth = np.asarray(self.pt.azimuth, dtype=float)
        color_array = np.asarray(self.get_pt_color_array(self.pt), dtype=float)

        valid = (
            np.isfinite(x) & np.isfinite(phimin) & np.isfinite(phimax) & (phimax > 0)
        )
        valid &= np.isfinite(azimuth) & np.isfinite(color_array)

        phimax_station = np.nanmax(phimax[valid]) if np.any(valid) else np.nan
        if np.isfinite(phimax_station) and phimax_station > 0:
            scaling = self.ellipse_size / phimax_station
        else:
            scaling = 0.0

        height = phimin[valid] * scaling
        width = phimax[valid] * scaling
        angle = np.deg2rad(90.0 - azimuth[valid])

        n_valid = int(np.count_nonzero(valid))
        source = ColumnDataSource(
            data={
                "x": x[valid],
                "y": np.zeros(n_valid),
                "width": width,
                "height": height,
                "angle": angle,
                "color_value": color_array[valid],
                "phimin": phimin[valid],
                "phimax": phimax[valid],
                "azimuth": azimuth[valid],
                "period": period[valid],
            }
        )

        cmin, cmax = self.ellipse_range[0], self.ellipse_range[1]
        mapper = LinearColorMapper(palette=Turbo256, low=cmin, high=cmax)

        pt_renderer = pt_fig.ellipse(
            x="x",
            y="y",
            width="width",
            height="height",
            angle="angle",
            source=source,
            fill_color=linear_cmap("color_value", Turbo256, cmin, cmax),
            fill_alpha=0.9,
            line_color="#222222",
            line_width=0.6,
        )
        self.renderers.setdefault("pt", []).append(pt_renderer)

        cb = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(),
            label_standoff=8,
            title=self.cb_label_dict[self.ellipse_colorby],
        )
        pt_fig.add_layout(cb, "right")

        pt_fig.add_tools(
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

        # Expand x by half an ellipse width on each side so edge ellipses
        # are not clipped.
        x_pad = 0.5 * self.ellipse_size
        pt_fig.x_range.start = np.log10(self.x_limits[0]) * adjusted_spacing - x_pad
        pt_fig.x_range.end = np.log10(self.x_limits[1]) * adjusted_spacing + x_pad
        pt_fig.y_range = Range1d(-pt_ylim, pt_ylim)
        pt_fig.yaxis.visible = False
        pt_fig.grid.grid_line_alpha = 0.25
        self._apply_log_period_ticks(pt_fig, x_spacing=adjusted_spacing)

    def _plot_od_components(self, res_fig, phase_fig):
        xy_source_res = self._component_source(self.period, self.Z, "xy", kind="res")
        yx_source_res = self._component_source(self.period, self.Z, "yx", kind="res")

        xy_source_phase = self._component_source(
            self.period, self.Z, "xy", kind="phase"
        )
        yx_source_phase = self._component_source(
            self.period, self.Z, "yx", kind="phase", yx_shift=True
        )

        self._add_component(
            res_fig,
            xy_source_res,
            "Zxy",
            self.xy_color,
            self.xy_marker,
            "xy",
        )
        self._add_component(
            res_fig,
            yx_source_res,
            "Zyx",
            self.yx_color,
            self.yx_marker,
            "yx",
        )

        self._add_component(
            phase_fig,
            xy_source_phase,
            "Zxy",
            self.xy_color,
            self.xy_marker,
            "xy",
        )
        self._add_component(
            phase_fig,
            yx_source_phase,
            "Zyx",
            self.yx_color,
            self.yx_marker,
            "yx",
        )

    def _plot_diag_components(self, res_fig, phase_fig):
        xx_source_res = self._component_source(self.period, self.Z, "xx", kind="res")
        yy_source_res = self._component_source(self.period, self.Z, "yy", kind="res")

        xx_source_phase = self._component_source(
            self.period, self.Z, "xx", kind="phase"
        )
        yy_source_phase = self._component_source(
            self.period, self.Z, "yy", kind="phase"
        )

        self._add_component(
            res_fig, xx_source_res, "Zxx", self.xy_color, self.xy_marker, "xx"
        )
        self._add_component(
            res_fig, yy_source_res, "Zyy", self.yx_color, self.yx_marker, "yy"
        )
        self._add_component(
            phase_fig, xx_source_phase, "Zxx", self.xy_color, self.xy_marker, "xx"
        )
        self._add_component(
            phase_fig, yy_source_phase, "Zyy", self.yx_color, self.yx_marker, "yy"
        )

    def _plot_determinant(self, res_fig, phase_fig):
        source_res = ColumnDataSource(
            data={
                "period": np.asarray(self.period, dtype=float),
                "value": np.asarray(self.Z.res_det, dtype=float),
                "low": np.asarray(self.Z.res_det - self.Z.res_error_det, dtype=float),
                "high": np.asarray(self.Z.res_det + self.Z.res_error_det, dtype=float),
            }
        )
        source_phase = ColumnDataSource(
            data={
                "period": np.asarray(self.period, dtype=float),
                "value": np.asarray(self.Z.phase_det, dtype=float),
                "low": np.asarray(
                    self.Z.phase_det - self.Z.phase_error_det, dtype=float
                ),
                "high": np.asarray(
                    self.Z.phase_det + self.Z.phase_error_det, dtype=float
                ),
            }
        )

        self._add_component(
            res_fig,
            source_res,
            "det(Z)",
            self.det_color,
            self.det_marker,
            "det",
        )
        self._add_component(
            phase_fig,
            source_phase,
            "det(Z)",
            self.det_color,
            self.det_marker,
            "det",
        )

    def _set_legends(self, *figs):
        for fig in figs:
            if len(fig.legend) == 0:
                continue
            fig.legend.click_policy = "hide"
            fig.legend.location = "bottom_left"
            fig.legend.label_text_font_size = f"{self.font_size + 2}px"

    def _set_component_visibility(self, selected_components):
        selected = set(selected_components)
        for key, items in self.renderers.items():
            visible = key in selected
            for item in items:
                item.visible = visible

    def _set_period_window(self, log_period_limits):
        pmin_log, pmax_log = log_period_limits

        for key in self._log_x_figure_keys:
            fig = self.figures[key]
            fig.x_range.start = 10**pmin_log
            fig.x_range.end = 10**pmax_log

        for key in self._linear_x_figure_keys:
            fig = self.figures[key]
            if key == "pt":
                # PT uses adjusted x-spacing for equal aspect ratio
                spacing = self._pt_x_spacing
                fig.x_range.start = pmin_log * spacing
                fig.x_range.end = pmax_log * spacing
            else:
                fig.x_range.start = pmin_log
                fig.x_range.end = pmax_log

    def plot(self):
        """Create and optionally show a Bokeh MT response layout."""
        self._require_bokeh()

        self.plot_z = self._has_z()
        self.plot_tipper = self._has_tipper()
        self.plot_pt = self._has_pt()

        if not self.plot_z:
            raise ValueError("Bokeh PlotMTResponse currently requires impedance data.")

        if self.x_limits is None:
            self.x_limits = self.set_period_limits(self.period)

        self.renderers = {}
        self._log_x_figure_keys = set()
        self._linear_x_figure_keys = set()

        if self.res_limits is None:
            if self.plot_num == 1:
                self.res_limits = self.set_resistivity_limits(self.Z.resistivity)
            elif self.plot_num in [2, 3]:
                self.res_limits = self.set_resistivity_limits(
                    self.Z.resistivity, mode="all"
                )

        self.figures = {}

        res_fig = self._make_resistivity_figure()
        phase_fig = self._make_phase_figure(res_fig.x_range)

        self._plot_od_components(res_fig, phase_fig)

        if self.plot_num == 3:
            self._plot_determinant(res_fig, phase_fig)

        self._format_res_axis(res_fig)
        self._format_phase_axis(phase_fig)

        phase_limits = self.set_phase_limits(self.Z.phase, mode="od")
        self._set_axis_limits(res_fig, self.res_limits)
        self._set_axis_limits(phase_fig, phase_limits)

        self._add_hover(res_fig)
        self._add_hover(phase_fig)
        self._set_legends(res_fig, phase_fig)

        self.figures["res"] = res_fig
        self.figures["phase"] = phase_fig
        self._log_x_figure_keys.update(["res", "phase"])

        tip_fig = None
        pt_fig = None
        base_column_width = 800
        n_columns = 2 if self.plot_num == 2 else 1
        panel_width = base_column_width * n_columns
        # For plot_num == 2, tipper and PT figures need to span both columns
        aux_width = panel_width

        if self.plot_tipper.find("y") >= 0:
            tip_fig = self._make_tipper_figure(width=aux_width)
            self._plot_tipper(tip_fig)
            self._set_legends(tip_fig)
            self.figures["tip"] = tip_fig
            self._linear_x_figure_keys.add("tip")

        if self.plot_pt:
            pt_fig = self._make_pt_figure(width=aux_width)
            self._plot_phase_tensor(pt_fig)
            # PT uses an adjusted x-spacing for equal visual aspect ratio so
            # it cannot share the tipper's raw log10(period) x-range.
            self.figures["pt"] = pt_fig
            self._linear_x_figure_keys.add("pt")

        if self.plot_num == 2:
            res_fig_diag = self._make_resistivity_figure(x_range=res_fig.x_range)
            phase_fig_diag = self._make_phase_figure(res_fig.x_range)
            self._plot_diag_components(res_fig_diag, phase_fig_diag)
            self._format_res_axis(res_fig_diag)
            self._format_phase_axis(phase_fig_diag)

            # Remove y-axis labels from diagonal components
            res_fig_diag.yaxis.axis_label = ""
            phase_fig_diag.yaxis.axis_label = ""

            phase_limits_diag = self.set_phase_limits(self.Z.phase, mode="d")
            self._set_axis_limits(res_fig_diag, self.res_limits)
            self._set_axis_limits(phase_fig_diag, phase_limits_diag)

            self._add_hover(res_fig_diag)
            self._add_hover(phase_fig_diag)
            self._set_legends(res_fig_diag, phase_fig_diag)

            self.figures["res_diag"] = res_fig_diag
            self.figures["phase_diag"] = phase_fig_diag
            self._log_x_figure_keys.update(["res_diag", "phase_diag"])

            # Create 2-column grid for OD and diagonal components
            layout_rows = []
            layout_rows.append(
                Row(res_fig, res_fig_diag, width=panel_width, sizing_mode="fixed")
            )
            layout_rows.append(
                Row(phase_fig, phase_fig_diag, width=panel_width, sizing_mode="fixed")
            )

            # Add full-width tipper and PT rows
            if tip_fig is not None:
                layout_rows.append(tip_fig)
            if pt_fig is not None:
                layout_rows.append(pt_fig)

            self.layout = Column(*layout_rows, width=panel_width, sizing_mode="fixed")
        else:
            # Single column layout for plot_num == 1 or 3
            layout_rows = [res_fig, phase_fig]
            if tip_fig is not None:
                layout_rows.append(tip_fig)
            if pt_fig is not None:
                layout_rows.append(pt_fig)
            self.layout = Column(*layout_rows, width=panel_width, sizing_mode="fixed")

        if self.show_plot:
            show(self.layout)

        return self.layout

    def panel(self, sizing_mode="stretch_width", interactive=True):
        """Return a Panel object wrapping the Bokeh layout.

        Parameters
        ----------
        sizing_mode : str
            Panel sizing mode
        interactive : bool
            If True, include first-pass controls for component visibility,
            period window, and data/model errors.
        """
        try:
            import panel as pn
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Panel is required to create a panel object. Install with `pip install panel`."
            ) from error

        if self.layout is None:
            self.plot()

        title = self.plot_title if self.plot_title else self.station
        bokeh_pane = pn.pane.Bokeh(self.layout, sizing_mode=sizing_mode)

        if not interactive:
            return pn.Column(
                pn.pane.Markdown(f"## {title}"),
                bokeh_pane,
                sizing_mode=sizing_mode,
            )

        options = {
            "xy": "Zxy",
            "yx": "Zyx",
            "xx": "Zxx",
            "yy": "Zyy",
            "det": "det(Z)",
            "tip_real": "Tipper Real",
            "tip_imag": "Tipper Imag",
            "pt": "Phase Tensor",
        }
        available = [key for key in options if key in self.renderers]

        component_widget = pn.widgets.CheckButtonGroup(
            name="Visible Components",
            options={options[key]: key for key in available},
            value=available,
            button_type="light",
        )

        error_widget = pn.widgets.RadioButtonGroup(
            name="Error Type",
            options=["data", "model"],
            value="model" if self.plot_model_error else "data",
            button_type="success",
        )

        pmin_log = float(np.floor(np.log10(self.x_limits[0])))
        pmax_log = float(np.ceil(np.log10(self.x_limits[1])))
        period_widget = pn.widgets.RangeSlider(
            name="log10 Period (s)",
            start=pmin_log,
            end=pmax_log,
            value=(pmin_log, pmax_log),
            step=0.1,
        )

        _preset_map = {
            "Off-diagonal": [
                k for k in ["xy", "yx", "tip_real", "tip_imag", "pt"] if k in available
            ],
            "Full tensor": [
                k
                for k in ["xy", "yx", "xx", "yy", "tip_real", "tip_imag", "pt"]
                if k in available
            ],
            "All": available,
        }
        preset_widget = pn.widgets.RadioButtonGroup(
            name="Preset",
            options=list(_preset_map.keys()),
            value="Off-diagonal",
            button_type="warning",
        )

        def _refresh_from_error_mode(event):
            self.plot_model_error = event.new == "model"
            self.plot()
            bokeh_pane.object = self.layout
            self._set_component_visibility(component_widget.value)
            self._set_period_window(period_widget.value)

        def _update_visibility(event):
            self._set_component_visibility(event.new)

        def _update_period(event):
            self._set_period_window(event.new)

        def _apply_preset(event):
            component_widget.value = _preset_map[event.new]

        error_widget.param.watch(_refresh_from_error_mode, "value")
        component_widget.param.watch(_update_visibility, "value")
        period_widget.param.watch(_update_period, "value")
        preset_widget.param.watch(_apply_preset, "value")

        # initialise with Off-diagonal preset active
        component_widget.value = _preset_map["Off-diagonal"]
        self._set_component_visibility(component_widget.value)
        self._set_period_window(period_widget.value)

        controls = pn.Row(
            pn.Column(pn.pane.Markdown("**Preset**"), preset_widget),
            pn.Column(pn.pane.Markdown("**Components**"), component_widget),
            pn.Column(pn.pane.Markdown("**Error Type**"), error_widget),
            pn.Column(pn.pane.Markdown("**Period Window**"), period_widget),
        )

        return pn.Column(
            pn.pane.Markdown(f"## {title}"),
            controls,
            bokeh_pane,
            sizing_mode=sizing_mode,
        )
