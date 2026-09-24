"""Bokeh implementation of a transfer-function-only MT response editor.

This module is a trimmed-down copy of
:class:`~mtpy.imaging.bokeh_plots.plot_mt_response.PlotMTResponse` intended
for embedding as a lightweight "TF Editor" widget in Panel dashboards. Unlike
``PlotMTResponse``, ``EditMTResponse`` only supports the per-component "edit"
layout (mask points, add model error, interpolate, static shift, rotate, and
flip phase for a single station) and never plots the phase tensor.
"""

from __future__ import annotations

import numpy as np
from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FixedTicker,
    HoverTool,
    Whisker,
)
from bokeh.plotting import figure

from .bokeh_plot_base import BokehPlotBase


class EditMTResponse(BokehPlotBase):
    """Edit-mode-only MT response editor using Bokeh.

    Plots apparent resistivity, phase, and tipper (real/imag) for each of
    Zxx, Zxy, Zyx, Zyy in a 4-column x 3-row grid, with tools to mask points,
    add model error to selected points, interpolate, remove static shift,
    rotate, and flip phase. The phase tensor is not plotted.
    """

    _MARKER_MAP = {
        "o": "circle",
        "s": "square",
        "v": "triangle",
        "d": "diamond",
        "^": "inverted_triangle",
    }

    _INTERP_METHODS = [
        "slinear",
        "linear",
        "nearest",
        "zero",
        "quadratic",
        "cubic",
        "previous",
        "next",
        "pchip",
        "akima",
        "spline",
        "barycentric",
        "polynomial",
        "krogh",
    ]

    _MODEL_ERROR_COMP_INDEX = {
        "zxx": (0, 0),
        "zxy": (0, 1),
        "zyx": (1, 0),
        "zyy": (1, 1),
        "tzx": (0, 0),
        "tzy": (0, 1),
    }

    def __init__(
        self,
        z_object=None,
        t_object=None,
        z_response=None,
        t_response=None,
        station="MT Response",
        **kwargs,
    ):
        self.Z = z_object
        self.Tipper = t_object
        # Optional forward-model response (e.g. from a "model" survey),
        # always plotted without error bars alongside the data.
        self.Z_response = z_response
        self.Tipper_response = t_response
        self.station = station
        self._basename = f"{self.station}_mt_response_bokeh"
        self.rotation_angle = 0

        self.layout = None
        self.figures = {}
        self.renderers = {}
        self._log_x_figure_keys = set()
        self.masked_tf_indices: dict[str, set[int]] = {}
        # Only the per-component edit layout is supported.
        self.edit_mode = True

        # Pre-edit snapshots used to draw the original data in gray once the
        # user applies interpolation, static shift, rotation, or phase flips.
        self._original_Z = None
        self._original_Tipper = None
        self._data_manipulated = False

        # param.Parameterized raises TypeError for unknown kwargs; split them.
        param_names = set(type(self).param)
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in param_names}

        super().__init__(**param_kwargs)
        self.marker_size = 5

        if self.Z is None:
            self.plot_z = False

        if self.Tipper is not None:
            self.plot_tipper = "yri"

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

        self._snapshot_original_data()
        if self.Z is not None:
            self.Z.rotate(theta_r, inplace=True)
        if self.Tipper is not None:
            self.Tipper.rotate(theta_r, inplace=True)

        self._rotation_angle += theta_r

    def _snapshot_original_data(self):
        """Capture pre-edit copies of Z/Tipper once, before the first manipulation."""
        if self._original_Z is None and self.Z is not None:
            self._original_Z = self.Z.copy()
        if self._original_Tipper is None and self.Tipper is not None:
            self._original_Tipper = self.Tipper.copy()
        self._data_manipulated = True

    def interpolate(
        self,
        new_period,
        method="slinear",
        extrapolate=False,
    ):
        """Interpolate Z and Tipper onto a new period array, in place.

        Parameters
        ----------
        new_period : array_like
            Periods (in seconds) to interpolate onto.
        method : str, optional
            Interpolation method, one of `_INTERP_METHODS`, by default "slinear".
        extrapolate : bool, optional
            Allow values outside the original period range, by default False.
        """
        self._snapshot_original_data()
        new_period = np.asarray(new_period, dtype=float)

        if self.Z is not None:
            self.Z = self.Z.interpolate(
                new_period, inplace=False, method=method, extrapolate=extrapolate
            )
        if self.Tipper is not None:
            self.Tipper = self.Tipper.interpolate(
                new_period, inplace=False, method=method, extrapolate=extrapolate
            )

        # masked indices refer to the old period axis and no longer apply
        self.masked_tf_indices = {}
        self.x_limits = self.set_period_limits(self.period)
        self.res_limits = None

    def static_shift(self, ss_x=1.0, ss_y=1.0):
        """Remove static shift from Z by the given correction factors, in place.

        Parameters
        ----------
        ss_x : float, optional
            Correction factor for x components (Z[:, 0, :]), by default 1.0
        ss_y : float, optional
            Correction factor for y components (Z[:, 1, :]), by default 1.0
        """
        if self.Z is None:
            return

        self._snapshot_original_data()
        self.Z = self.Z.remove_ss(
            reduce_res_factor_x=ss_x, reduce_res_factor_y=ss_y, inplace=False
        )

    def flip_phase(
        self,
        zxx=False,
        zxy=False,
        zyx=False,
        zyy=False,
        tzx=False,
        tzy=False,
    ):
        """Flip the sign of the transfer function for the given components, in place.

        Parameters
        ----------
        zxx, zxy, zyx, zyy : bool, optional
            Flip the corresponding impedance component, by default False.
        tzx, tzy : bool, optional
            Flip the corresponding tipper component, by default False.
        """
        if zxx or zxy or zyx or zyy or tzx or tzy:
            self._snapshot_original_data()

        if self.Z is not None and (zxx or zxy or zyx or zyy):
            z = self.Z.z.copy()
            if zxx:
                z[:, 0, 0] *= -1
            if zxy:
                z[:, 0, 1] *= -1
            if zyx:
                z[:, 1, 0] *= -1
            if zyy:
                z[:, 1, 1] *= -1
            self.Z.z = z

        if self.Tipper is not None and (tzx or tzy):
            tipper = self.Tipper.tipper.copy()
            if tzx:
                tipper[:, 0, 0] *= -1
            if tzy:
                tipper[:, 0, 1] *= -1
            self.Tipper.tipper = tipper

    def add_model_error(self, comp, z_value=5.0, t_value=0.05, periods=None):
        """Adjust the model error for the given components, in place.

        Parameters
        ----------
        comp : str or list of str
            Components to modify, any of "zxx", "zxy", "zyx", "zyy", "tzx", "tzy".
        z_value : float, optional
            Multiplier applied to impedance model error, by default 5.0.
        t_value : float, optional
            Value added to tipper model error, by default 0.05.
        periods : tuple of float, optional
            (min_period, max_period) to restrict the edit to, by default None
            (applies to all periods).
        """
        if isinstance(comp, str):
            comp = [comp]

        if periods is not None:
            if len(periods) != 2:
                raise ValueError("Must enter a minimum and maximum period value")
            p_min = np.where(self.period >= min(periods))[0][0]
            p_max = np.where(self.period <= max(periods))[0][-1]
        else:
            p_min = 0
            p_max = len(self.period) - 1

        if self.Z is not None:
            z_model_error = self.Z.z_model_error
            if z_model_error is None:
                base = self.Z.z_error
                z_model_error = (
                    base.copy()
                    if base is not None
                    else np.zeros(self.Z.z.shape, dtype=float)
                )
            else:
                z_model_error = z_model_error.copy()
            for cc in [c for c in comp if c.startswith("z")]:
                if cc not in self._MODEL_ERROR_COMP_INDEX:
                    continue
                ii, jj = self._MODEL_ERROR_COMP_INDEX[cc]
                z_model_error[p_min : p_max + 1, ii, jj] *= z_value
            self.Z.z_model_error = z_model_error

        if self.Tipper is not None:
            t_model_error = self.Tipper.tipper_model_error
            if t_model_error is None:
                base = self.Tipper.tipper_error
                t_model_error = (
                    base.copy()
                    if base is not None
                    else np.zeros(self.Tipper.tipper.shape, dtype=float)
                )
            else:
                t_model_error = t_model_error.copy()
            for cc in [c for c in comp if c.startswith("t")]:
                if cc not in self._MODEL_ERROR_COMP_INDEX:
                    continue
                ii, jj = self._MODEL_ERROR_COMP_INDEX[cc]
                t_model_error[p_min : p_max + 1, ii, jj] += t_value
            self.Tipper.tipper_model_error = t_model_error

    def add_model_error_to_indices(self, selected, z_value=5.0, t_value=0.05):
        """Adjust model error only at specific per-component tf_indices, in place.

        Parameters
        ----------
        selected : dict[str, set[int]]
            Mapping of component key ("xx", "xy", "yx", "yy", "tzx", "tzy") to
            the set of tf_index values (as produced by `_selected_tf_indices`)
            to apply the model error to.
        z_value : float, optional
            Multiplier applied to impedance model error, by default 5.0.
        t_value : float, optional
            Value added to tipper model error, by default 0.05.
        """
        for comp, indices in selected.items():
            if not indices:
                continue
            idx = np.asarray(sorted(indices), dtype=int)

            if comp in ("xx", "xy", "yx", "yy") and self.Z is not None:
                ii, jj = self._MODEL_ERROR_COMP_INDEX[f"z{comp}"]
                z_model_error = self.Z.z_model_error
                if z_model_error is None:
                    base = self.Z.z_error
                    z_model_error = (
                        base.copy()
                        if base is not None
                        else np.zeros(self.Z.z.shape, dtype=float)
                    )
                else:
                    z_model_error = z_model_error.copy()
                z_model_error[idx, ii, jj] *= z_value
                self.Z.z_model_error = z_model_error

            elif comp in ("tzx", "tzy") and self.Tipper is not None:
                ii, jj = self._MODEL_ERROR_COMP_INDEX[comp]
                t_model_error = self.Tipper.tipper_model_error
                if t_model_error is None:
                    base = self.Tipper.tipper_error
                    t_model_error = (
                        base.copy()
                        if base is not None
                        else np.zeros(self.Tipper.tipper.shape, dtype=float)
                    )
                else:
                    t_model_error = t_model_error.copy()
                t_model_error[idx, ii, jj] += t_value
                self.Tipper.tipper_model_error = t_model_error

    def _require_bokeh(self):
        if figure is None or ColumnDataSource is None or Whisker is None:
            raise ImportError(
                "Bokeh is required for EditMTResponse plots. "
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

    def _component_source(
        self, period, z_obj, comp, kind="res", yx_shift=False, masked=False
    ):
        """Build a ColumnDataSource for one component.

        The underlying arrays always come straight from `z_obj`, i.e. the
        original data is never mutated by masking. When `masked` is False
        the source contains the currently active (non-masked) points; when
        True it contains only the points a user has masked out, so callers
        can render them separately (e.g. in gray).
        """
        y_attr = "res" if kind == "res" else "phase"
        e_attr = f"{y_attr}_{self._error_str}"

        y = self._get_values(z_obj, y_attr, comp)
        err = self._get_values(z_obj, e_attr, comp)

        # When model error is unavailable (all NaN), fall back to zero errors
        # so data points still render without error bars rather than vanishing.
        if not np.any(np.isfinite(err)):
            err = np.zeros_like(y)

        if yx_shift:
            y = y + 180

        low = y - err
        high = y + err
        x = np.asarray(period, dtype=float)

        if kind == "res":
            valid = self._valid_for_log(x, y)
            valid = valid & np.isfinite(err) & (high > 0)
            # Clamp the lower error bar to a small but positive value so the
            # whisker is always visible on the log-scale resistivity axis.
            # Using 1e-3 * y keeps it three decades below the data point —
            # well below any realistic resistivity but never NaN or ≤0.
            low_floor = np.where(np.isfinite(y) & (y > 0), y * 1e-3, np.nan)
            low = np.where((low <= 0) | ~np.isfinite(low), low_floor, low)
        else:
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)

        tf_index = np.arange(x.size, dtype=int)
        masked_set = self.masked_tf_indices.get(comp, set())
        is_masked = (
            np.isin(tf_index, list(masked_set)) if masked_set else np.zeros_like(valid)
        )
        valid &= is_masked if masked else ~is_masked

        data = {
            "period": x[valid],
            "value": y[valid],
            "low": low[valid],
            "high": high[valid],
            "tf_index": tf_index[valid],
            "component": [comp] * int(np.count_nonzero(valid)),
        }
        return ColumnDataSource(data=data)

    def _original_component_source(self, comp, kind="res", yx_shift=False):
        """Build a ColumnDataSource for one component from the pre-edit Z snapshot."""
        if self._original_Z is None:
            return None
        y_attr = "res" if kind == "res" else "phase"
        y = self._get_values(self._original_Z, y_attr, comp)
        if yx_shift:
            y = y + 180
        x = np.asarray(self._original_Z.period, dtype=float)

        if kind == "res":
            valid = self._valid_for_log(x, y)
        else:
            valid = np.isfinite(x) & np.isfinite(y)

        return ColumnDataSource(data={"period": x[valid], "value": y[valid]})

    def _add_gray_component(self, fig, source, marker="o"):
        """Draw a lightweight gray underlay of pre-edit data beneath current renderers."""
        if source is None or len(source.data.get("period", [])) == 0:
            return
        gray = "#aaaaaa"
        fig.line(
            x="period",
            y="value",
            source=source,
            color=gray,
            line_width=1,
            line_alpha=0.5,
            line_dash="dotted",
        )
        fig.scatter(
            x="period",
            y="value",
            source=source,
            marker=self._marker_name(marker),
            size=max(int(self.marker_size) - 1, 3),
            color=gray,
            line_color=gray,
            fill_alpha=0.4,
            line_alpha=0.4,
        )

    def _response_component_source(self, comp, kind="res", yx_shift=False):
        """Build a ColumnDataSource for one component from the model response Z."""
        if self.Z_response is None:
            return None
        y_attr = "res" if kind == "res" else "phase"
        y = self._get_values(self.Z_response, y_attr, comp)
        if yx_shift:
            y = y + 180
        x = np.asarray(self.Z_response.period, dtype=float)

        if kind == "res":
            valid = self._valid_for_log(x, y)
        else:
            valid = np.isfinite(x) & np.isfinite(y)

        return ColumnDataSource(data={"period": x[valid], "value": y[valid]})

    def _response_tipper_component_source(self, comp_index, part):
        """Build a ColumnDataSource for one tipper component from the model response."""
        if self.Tipper_response is None:
            return None
        period = np.asarray(1.0 / self.Tipper_response.frequency, dtype=float)
        tf_values = self.Tipper_response.tipper[:, 0, comp_index]
        value = tf_values.real if part == "real" else tf_values.imag
        value = np.asarray(value, dtype=float)
        valid = np.isfinite(period) & (period > 0) & np.isfinite(value)
        return ColumnDataSource(data={"period": period[valid], "value": value[valid]})

    def _add_response_component(self, fig, source, color, comp_key):
        """Draw the model response as a solid line with 'x' markers, no error bars."""
        if source is None or len(source.data.get("period", [])) == 0:
            return
        glyph_color = self._tuple_to_hex(color)
        line_renderer = fig.line(
            x="period",
            y="value",
            source=source,
            color=glyph_color,
            line_width=max(self.lw, 1.5),
            line_dash="solid",
        )
        scatter_renderer = fig.scatter(
            x="period",
            y="value",
            source=source,
            marker="x",
            size=max(int(self.marker_size) + 2, 6),
            color=glyph_color,
            line_color=glyph_color,
        )
        self.renderers.setdefault(comp_key, []).extend(
            [line_renderer, scatter_renderer]
        )

    def _add_component(
        self,
        fig,
        source,
        comp_label,
        color,
        marker,
        comp_key,
        show_error=True,
        masked_source=None,
    ):
        glyph_color = self._tuple_to_hex(color)

        if masked_source is not None and len(masked_source.data["period"]) > 0:
            gray_color = "#999999"
            masked_renderer = fig.scatter(
                x="period",
                y="value",
                source=masked_source,
                marker=self._marker_name(marker),
                size=max(int(self.marker_size), 4),
                color=gray_color,
                line_color=gray_color,
                fill_alpha=0.5,
                line_alpha=0.5,
            )
            self.renderers.setdefault(comp_key, []).append(masked_renderer)

        line_renderer = fig.line(
            x="period",
            y="value",
            source=source,
            color=glyph_color,
            line_width=max(self.lw, 1),
            line_dash="dashed",
        )
        scatter_renderer = fig.scatter(
            x="period",
            y="value",
            source=source,
            marker=self._marker_name(marker),
            size=max(int(self.marker_size), 4),
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
                line_width=max(self.lw, 2),
            )
            whisker.upper_head.size = self.marker_size
            whisker.upper_head.line_color = glyph_color
            whisker.lower_head.size = self.marker_size
            whisker.lower_head.line_color = glyph_color
            fig.add_layout(whisker)
            self.renderers.setdefault(comp_key, []).append(whisker)

    def _make_resistivity_figure(self, x_range=None, width=800, height=320):
        kw = {}
        if x_range is not None:
            kw["x_range"] = x_range
        return figure(
            title=None,
            x_axis_type="log",
            y_axis_type="log",
            height=height,
            width=width,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select,lasso_select",
            active_scroll="wheel_zoom",
            **kw,
        )

    def _make_phase_figure(self, x_range, width=800, height=250):
        return figure(
            title=None,
            x_axis_type="log",
            x_range=x_range,
            height=height,
            width=width,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select,lasso_select",
            active_scroll="wheel_zoom",
        )

    def _apply_log_period_ticks(self, fig, x_spacing=1.0):
        """Replace a linear-x axis with 10^n period labels."""
        pmin_log = np.log10(float(self.x_limits[0]))
        pmax_log = np.log10(float(self.x_limits[1]))
        ticks = [
            t * x_spacing
            for t in range(int(np.floor(pmin_log)), int(np.ceil(pmax_log)) + 1)
        ]
        fig.xaxis.ticker = FixedTicker(ticks=ticks)
        fig.xaxis.formatter = CustomJSTickFormatter(
            args={"x_spacing": x_spacing},
            code="""
            var superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
            var exp = String(Math.round(tick / x_spacing));
            var result = '10';
            for (var i = 0; i < exp.length; i++) {
                if (exp[i] === '-') result += '⁻';
                else result += superscripts[parseInt(exp[i])];
            }
            return result;
            """,
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

    def _tipper_component_source(self, comp_index, part, masked=False, tipper_obj=None):
        """Build a period-indexed source for one tipper component (tzx/tzy).

        `comp_index` is 0 for tzx, 1 for tzy. `part` is "real" or "imag".
        The component key used for masking ("tzx"/"tzy") covers both the
        real and imaginary parts so masking one masks both together.
        """
        comp = "tzx" if comp_index == 0 else "tzy"
        tipper_obj = self.Tipper if tipper_obj is None else tipper_obj
        period = np.asarray(1.0 / tipper_obj.frequency, dtype=float)
        tf_values = tipper_obj.tipper[:, 0, comp_index]
        value = tf_values.real if part == "real" else tf_values.imag
        value = np.asarray(value, dtype=float)

        err_arr = getattr(tipper_obj, f"tipper_{self._error_str}", None)
        if err_arr is not None:
            err = np.asarray(err_arr[:, 0, comp_index], dtype=float)
        else:
            err = np.zeros_like(value)
        if not np.any(np.isfinite(err)):
            err = np.zeros_like(value)

        valid = (
            np.isfinite(period) & (period > 0) & np.isfinite(value) & np.isfinite(err)
        )
        low = value - err
        high = value + err

        tf_index = np.arange(period.size, dtype=int)
        masked_set = self.masked_tf_indices.get(comp, set())
        is_masked = (
            np.isin(tf_index, list(masked_set)) if masked_set else np.zeros_like(valid)
        )
        valid &= is_masked if masked else ~is_masked

        data = {
            "period": period[valid],
            "value": value[valid],
            "low": low[valid],
            "high": high[valid],
            "tf_index": tf_index[valid],
            "component": [comp] * int(np.count_nonzero(valid)),
        }
        return ColumnDataSource(data=data)

    def _plot_edit_layout(self):
        """Build the 4-column x 3-row edit-mode layout.

        Columns are Zxx, Zxy, Zyx, Zyy; rows are apparent resistivity,
        phase, and tipper (real tzx, imag tzx, real tzy, imag tzy).
        """
        # Smaller than the default figure heights so all 3 rows fit on one
        # screen without scrolling.
        fig_w = 300
        res_h = 210
        phase_h = 170
        tip_h = 150
        comps = [
            ("xx", "Zxx", self.xx_color, self.xx_marker),
            ("xy", "Zxy", self.xy_color, self.xy_marker),
            ("yx", "Zyx", self.yx_color, self.yx_marker),
            ("yy", "Zyy", self.yy_color, self.yy_marker),
        ]

        res_figs = {}
        phase_figs = {}
        shared_x_range = None

        for comp, label, color, marker in comps:
            res_fig = self._make_resistivity_figure(
                x_range=shared_x_range, width=fig_w, height=res_h
            )
            if shared_x_range is None:
                shared_x_range = res_fig.x_range
            if self._data_manipulated:
                self._add_gray_component(
                    res_fig, self._original_component_source(comp, kind="res"), marker
                )
            source_res = self._component_source(self.period, self.Z, comp, kind="res")
            masked_res = self._component_source(
                self.period, self.Z, comp, kind="res", masked=True
            )
            self._add_component(
                res_fig,
                source_res,
                label,
                color,
                marker,
                comp,
                masked_source=masked_res,
            )
            if self.Z_response is not None:
                self._add_response_component(
                    res_fig,
                    self._response_component_source(comp, kind="res"),
                    color,
                    comp,
                )
            self._format_res_axis(res_fig)
            res_figs[comp] = res_fig

            phase_fig = self._make_phase_figure(
                shared_x_range, width=fig_w, height=phase_h
            )
            yx_shift = comp == "yx"
            if self._data_manipulated:
                self._add_gray_component(
                    phase_fig,
                    self._original_component_source(
                        comp, kind="phase", yx_shift=yx_shift
                    ),
                    marker,
                )
            source_phase = self._component_source(
                self.period, self.Z, comp, kind="phase", yx_shift=yx_shift
            )
            masked_phase = self._component_source(
                self.period, self.Z, comp, kind="phase", yx_shift=yx_shift, masked=True
            )
            self._add_component(
                phase_fig,
                source_phase,
                label,
                color,
                marker,
                comp,
                masked_source=masked_phase,
            )
            if self.Z_response is not None:
                self._add_response_component(
                    phase_fig,
                    self._response_component_source(
                        comp, kind="phase", yx_shift=yx_shift
                    ),
                    color,
                    comp,
                )
            self._format_phase_axis(phase_fig)
            # Row 2 of 3 in the edit grid; the tipper row below already
            # carries the "Period (s)" x-axis label.
            phase_fig.xaxis.axis_label = ""
            phase_fig.xaxis.visible = False
            phase_figs[comp] = phase_fig

        res_limits = self.res_limits
        phase_limits_od = self.set_phase_limits(self.Z.phase, mode="od")
        phase_limits_diag = self.set_phase_limits(self.Z.phase, mode="d")
        for comp, _label, _color, _marker in comps:
            self._set_axis_limits(res_figs[comp], res_limits)
            phase_limits = (
                phase_limits_od if comp in ("xy", "yx") else phase_limits_diag
            )
            self._set_axis_limits(phase_figs[comp], phase_limits)
            self._add_hover(res_figs[comp])
            self._add_hover(phase_figs[comp])
            self._set_legends(res_figs[comp], phase_figs[comp])
            if comp != "xx":
                res_figs[comp].yaxis.axis_label = ""
                phase_figs[comp].yaxis.axis_label = ""

        tip_defs = [
            ("tip_real_zx", 0, "real", self.arrow_color_real, "Re(Tzx)"),
            ("tip_imag_zx", 0, "imag", self.arrow_color_imag, "Im(Tzx)"),
            ("tip_real_zy", 1, "real", self.arrow_color_real, "Re(Tzy)"),
            ("tip_imag_zy", 1, "imag", self.arrow_color_imag, "Im(Tzy)"),
        ]
        tip_figs = {}
        for key, comp_index, part, color, label in tip_defs:
            tip_fig = self._make_phase_figure(shared_x_range, width=fig_w, height=tip_h)
            if self._data_manipulated and self._original_Tipper is not None:
                gray_source = self._tipper_component_source(
                    comp_index, part, tipper_obj=self._original_Tipper
                )
                self._add_gray_component(tip_fig, gray_source, "o")
            source = self._tipper_component_source(comp_index, part)
            masked_source = self._tipper_component_source(comp_index, part, masked=True)
            self._add_component(
                tip_fig,
                source,
                label,
                color,
                "o",
                key,
                show_error=True,
                masked_source=masked_source,
            )
            if self.Tipper_response is not None:
                self._add_response_component(
                    tip_fig,
                    self._response_tipper_component_source(comp_index, part),
                    color,
                    key,
                )
            tip_fig.yaxis.axis_label = label if key == "tip_real_zx" else ""
            tip_fig.xaxis.axis_label = "Period (s)"
            tip_fig.grid.grid_line_alpha = 0.25
            self._add_hover(tip_fig)
            self._set_legends(tip_fig)
            tip_figs[key] = tip_fig

        self.figures.update({f"res_{c}": res_figs[c] for c, *_ in comps})
        self.figures.update({f"phase_{c}": phase_figs[c] for c, *_ in comps})
        self.figures.update(tip_figs)
        self._log_x_figure_keys.update(
            [f"res_{c}" for c, *_ in comps]
            + [f"phase_{c}" for c, *_ in comps]
            + list(tip_figs.keys())
        )

        row1 = [res_figs[c] for c, *_ in comps]
        row2 = [phase_figs[c] for c, *_ in comps]
        row3 = [
            tip_figs["tip_real_zx"],
            tip_figs["tip_imag_zx"],
            tip_figs["tip_real_zy"],
            tip_figs["tip_imag_zy"],
        ]
        # gridplot merges all 12 figures' toolbars into a single shared one so
        # a tool (e.g. lasso/box select) only needs to be activated once.
        # sizing_mode lets the grid (and each stretch_width figure inside it)
        # expand to fill the available screen width.
        self.layout = gridplot(
            [row1, row2, row3],
            toolbar_location="above",
            merge_tools=True,
            sizing_mode="stretch_width",
        )
        return self.layout

    def _set_legends(self, *figs):
        for fig in figs:
            if len(fig.legend) == 0:
                continue
            fig.legend.click_policy = "hide"
            fig.legend.location = "top_left"
            fig.legend.label_text_font_size = f"{self.font_size + 2}px"

    def _set_component_visibility(self, selected_components):
        """Show/hide only the Zxx/Zxy/Zyx/Zyy renderers; tipper stays visible.

        Only "xx"/"xy"/"yx"/"yy" are ever passed through the Components
        widget, so limiting the toggle to those keys avoids incidentally
        hiding the tipper (tip_real_zx/tip_imag_zx/tip_real_zy/tip_imag_zy)
        renderers, which should always remain visible.
        """
        selected = set(selected_components)
        for key in ("xx", "xy", "yx", "yy"):
            for item in self.renderers.get(key, []):
                item.visible = key in selected

    def _set_period_window(self, log_period_limits):
        pmin_log, pmax_log = log_period_limits
        for key in self._log_x_figure_keys:
            fig = self.figures[key]
            fig.x_range.start = 10**pmin_log
            fig.x_range.end = 10**pmax_log

    def plot(self):
        """Create and optionally show the Bokeh edit-mode layout."""
        self._require_bokeh()

        self.plot_z = self._has_z()
        self.plot_tipper = self._has_tipper()

        if not self.plot_z:
            raise ValueError("Bokeh EditMTResponse currently requires impedance data.")

        if self.x_limits is None:
            self.x_limits = self.set_period_limits(self.period)

        self.renderers = {}
        self._log_x_figure_keys = set()

        if self.res_limits is None:
            self.res_limits = self.set_resistivity_limits(
                self.Z.resistivity, mode="all"
            )

        self.figures = {}
        self._plot_edit_layout()
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
            If True, include controls for masking, model error, interpolation,
            static shift, rotation, phase flips, and period window.
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

        selection_status = pn.pane.Markdown(
            "_Select impedance points with tap, box, or lasso, then mask them._",
            styles={"color": "#555"},
        )
        mask_selected_button = pn.widgets.Button(
            name="Mask Selected",
            button_type="warning",
            width=130,
        )
        add_model_error_selected_button = pn.widgets.Button(
            name="Add Model Error",
            button_type="warning",
            width=150,
        )
        reset_button = pn.widgets.Button(
            name="Reset",
            button_type="default",
            width=130,
        )

        def _selected_tf_indices() -> dict[str, set[int]]:
            selected: dict[str, set[int]] = {}
            seen_sources: set[int] = set()
            for fig in self.figures.values():
                for renderer in fig.renderers:
                    source = getattr(renderer, "data_source", None)
                    if source is None or id(source) in seen_sources:
                        continue
                    seen_sources.add(id(source))
                    component = source.data.get("component", [])
                    tf_index = source.data.get("tf_index", [])
                    if len(component) == 0 or len(tf_index) == 0:
                        continue
                    for index in source.selected.indices:
                        if index < len(tf_index):
                            selected.setdefault(component[index], set()).add(
                                int(tf_index[index])
                            )
            return selected

        def _refresh_masked_plot() -> None:
            self.plot()
            bokeh_pane.object = self.layout
            bokeh_pane.param.trigger("object")

        def _mask_selected(_event) -> None:
            selected = _selected_tf_indices()
            if not selected:
                selection_status.object = (
                    "⚠️ Select one or more impedance points first."
                )
                selection_status.styles = {"color": "#7a5200"}
                return

            count = 0
            for component, indices in selected.items():
                self.masked_tf_indices.setdefault(component, set()).update(indices)
                count += len(indices)
            _refresh_masked_plot()
            selection_status.object = f"Masked {count} selected point(s)."
            selection_status.styles = {"color": "#7a5200"}

        def _add_model_error_selected(_event) -> None:
            selected = _selected_tf_indices()
            if not selected:
                selection_status.object = (
                    "⚠️ Select one or more impedance points first."
                )
                selection_status.styles = {"color": "#7a5200"}
                return

            self.add_model_error_to_indices(
                selected,
                z_value=float(model_err_z_widget.value),
                t_value=float(model_err_t_widget.value),
            )
            self.plot_model_error = True
            error_widget.value = "model"
            count = sum(len(indices) for indices in selected.values())
            _refresh_masked_plot()
            selection_status.object = f"Added model error to {count} selected point(s)."
            selection_status.styles = {"color": "#7a5200"}

        def _reset_to_original(_event) -> None:
            """Restore Z/Tipper to their pre-edit state and clear all edits."""
            if not self._data_manipulated and not self.masked_tf_indices:
                selection_status.object = "Nothing to reset."
                selection_status.styles = {"color": "#555"}
                return

            if self._original_Z is not None:
                self.Z = self._original_Z.copy()
            if self._original_Tipper is not None:
                self.Tipper = self._original_Tipper.copy()

            self.masked_tf_indices = {}
            self._original_Z = None
            self._original_Tipper = None
            self._data_manipulated = False
            self._rotation_angle = 0
            self.plot_model_error = False
            error_widget.value = "data"
            self.x_limits = self.set_period_limits(self.period)
            self.res_limits = None

            _refresh_plot_and_widgets()
            rotate_status.object = (
                f"Current rotation angle: {self.rotation_angle:.4g} deg"
            )
            selection_status.object = "Reset to original data."
            selection_status.styles = {"color": "#1a6600"}

        mask_selected_button.on_click(_mask_selected)
        add_model_error_selected_button.on_click(_add_model_error_selected)
        reset_button.on_click(_reset_to_original)

        def _refresh_plot_and_widgets():
            """Replot after an in-place edit and resync period-dependent widgets."""
            period_widget.start = float(np.floor(np.log10(self.x_limits[0])))
            period_widget.end = float(np.ceil(np.log10(self.x_limits[1])))
            period_widget.value = (period_widget.start, period_widget.end)
            _apply_edit_mode()

        # ── interpolation controls ─────────────────────────────────────────
        period_min, period_max = float(self.period.min()), float(self.period.max())
        interp_min_widget = pn.widgets.FloatInput(
            name="Min period (s)", value=period_min, width=120
        )
        interp_max_widget = pn.widgets.FloatInput(
            name="Max period (s)", value=period_max, width=120
        )
        interp_num_widget = pn.widgets.IntInput(
            name="Num periods", value=int(self.period.size), start=2, width=100
        )
        interp_type_widget = pn.widgets.Select(
            name="Interp. type",
            options=self._INTERP_METHODS,
            value="slinear",
            width=120,
        )
        interp_extrapolate_widget = pn.widgets.Checkbox(name="Extrapolate", value=False)
        interp_apply_button = pn.widgets.Button(
            name="Apply Interpolation", button_type="primary", width=150
        )
        interp_status = pn.pane.Markdown("", styles={"color": "#555"})

        def _apply_interpolation(_event):
            try:
                pmin = float(interp_min_widget.value)
                pmax = float(interp_max_widget.value)
                num = int(interp_num_widget.value)
                if pmin <= 0 or pmax <= 0 or pmin >= pmax:
                    raise ValueError(
                        "Min period must be positive and less than max period."
                    )
                new_period = np.logspace(np.log10(pmin), np.log10(pmax), num=num)
                self.interpolate(
                    new_period,
                    method=interp_type_widget.value,
                    extrapolate=interp_extrapolate_widget.value,
                )
            except Exception as error:
                interp_status.object = f"⚠️ Interpolation failed: {error}"
                interp_status.styles = {"color": "#7a0000"}
                return

            _refresh_plot_and_widgets()
            interp_status.object = (
                f"Interpolated onto {num} periods "
                f"({pmin:.4g}s to {pmax:.4g}s, method={interp_type_widget.value})."
            )
            interp_status.styles = {"color": "#1a6600"}

        interp_apply_button.on_click(_apply_interpolation)

        interp_row = pn.Row(
            interp_min_widget,
            interp_max_widget,
            interp_num_widget,
            interp_type_widget,
            interp_extrapolate_widget,
            interp_apply_button,
            interp_status,
            align="center",
        )
        interp_card = pn.Card(
            interp_row,
            title="Interpolate",
            collapsed=True,
        )

        # ── static shift controls ───────────────────────────────────────────
        ss_x_widget = pn.widgets.FloatInput(name="Shift X", value=1.0, width=100)
        ss_y_widget = pn.widgets.FloatInput(name="Shift Y", value=1.0, width=100)
        ss_apply_button = pn.widgets.Button(
            name="Apply Static Shift", button_type="primary", width=150
        )
        ss_status = pn.pane.Markdown("", styles={"color": "#555"})

        def _apply_static_shift(_event):
            try:
                ss_x = float(ss_x_widget.value)
                ss_y = float(ss_y_widget.value)
                self.static_shift(ss_x=ss_x, ss_y=ss_y)
            except Exception as error:
                ss_status.object = f"⚠️ Static shift failed: {error}"
                ss_status.styles = {"color": "#7a0000"}
                return

            _refresh_plot_and_widgets()
            ss_status.object = f"Applied static shift x={ss_x:.4g}, y={ss_y:.4g}."
            ss_status.styles = {"color": "#1a6600"}

        ss_apply_button.on_click(_apply_static_shift)

        ss_row = pn.Row(
            ss_x_widget,
            ss_y_widget,
            ss_apply_button,
            ss_status,
            align="center",
        )
        ss_card = pn.Card(
            ss_row,
            title="Static Shift",
            collapsed=True,
        )

        # ── rotation controls ───────────────────────────────────────────────
        rotate_angle_widget = pn.widgets.FloatInput(
            name="Rotate by (deg)", value=0.0, width=120
        )
        rotate_apply_button = pn.widgets.Button(
            name="Apply Rotation", button_type="primary", width=150
        )
        rotate_status = pn.pane.Markdown(
            f"Current rotation angle: {self.rotation_angle:.4g} deg",
            styles={"color": "#555"},
        )

        def _apply_rotation(_event):
            try:
                theta_r = float(rotate_angle_widget.value)
                self.rotation_angle = theta_r
            except Exception as error:
                rotate_status.object = f"⚠️ Rotation failed: {error}"
                rotate_status.styles = {"color": "#7a0000"}
                return

            _refresh_plot_and_widgets()
            rotate_status.object = (
                f"Rotated by {theta_r:.4g} deg. "
                f"Current rotation angle: {self.rotation_angle:.4g} deg"
            )
            rotate_status.styles = {"color": "#1a6600"}

        rotate_apply_button.on_click(_apply_rotation)

        rotate_row = pn.Row(
            rotate_angle_widget,
            rotate_apply_button,
            rotate_status,
            align="center",
        )
        rotate_card = pn.Card(
            rotate_row,
            title="Rotate",
            collapsed=True,
        )

        # ── flip phase controls ─────────────────────────────────────────────
        flip_zxx_widget = pn.widgets.Checkbox(name="Zxx", value=False)
        flip_zxy_widget = pn.widgets.Checkbox(name="Zxy", value=False)
        flip_zyx_widget = pn.widgets.Checkbox(name="Zyx", value=False)
        flip_zyy_widget = pn.widgets.Checkbox(name="Zyy", value=False)
        flip_tzx_widget = pn.widgets.Checkbox(name="Tzx", value=False)
        flip_tzy_widget = pn.widgets.Checkbox(name="Tzy", value=False)
        flip_apply_button = pn.widgets.Button(
            name="Apply Flip", button_type="primary", width=120
        )
        flip_status = pn.pane.Markdown("", styles={"color": "#555"})

        def _apply_flip_phase(_event):
            selected = {
                "zxx": flip_zxx_widget.value,
                "zxy": flip_zxy_widget.value,
                "zyx": flip_zyx_widget.value,
                "zyy": flip_zyy_widget.value,
                "tzx": flip_tzx_widget.value,
                "tzy": flip_tzy_widget.value,
            }
            if not any(selected.values()):
                flip_status.object = "⚠️ Select one or more components to flip."
                flip_status.styles = {"color": "#7a5200"}
                return

            try:
                self.flip_phase(**selected)
            except Exception as error:
                flip_status.object = f"⚠️ Flip phase failed: {error}"
                flip_status.styles = {"color": "#7a0000"}
                return

            _refresh_plot_and_widgets()
            flipped = ", ".join(key for key, value in selected.items() if value)
            flip_status.object = f"Flipped phase for: {flipped}."
            flip_status.styles = {"color": "#1a6600"}
            for widget in (
                flip_zxx_widget,
                flip_zxy_widget,
                flip_zyx_widget,
                flip_zyy_widget,
                flip_tzx_widget,
                flip_tzy_widget,
            ):
                widget.value = False

        flip_apply_button.on_click(_apply_flip_phase)

        flip_row = pn.Row(
            flip_zxx_widget,
            flip_zxy_widget,
            flip_zyx_widget,
            flip_zyy_widget,
            flip_tzx_widget,
            flip_tzy_widget,
            flip_apply_button,
            flip_status,
            align="center",
        )
        flip_card = pn.Card(
            flip_row,
            title="Flip Phase",
            collapsed=True,
        )

        # ── add model error controls ────────────────────────────────────────
        model_err_zxx_widget = pn.widgets.Checkbox(name="Zxx", value=False)
        model_err_zxy_widget = pn.widgets.Checkbox(name="Zxy", value=False)
        model_err_zyx_widget = pn.widgets.Checkbox(name="Zyx", value=False)
        model_err_zyy_widget = pn.widgets.Checkbox(name="Zyy", value=False)
        model_err_tzx_widget = pn.widgets.Checkbox(name="Tzx", value=False)
        model_err_tzy_widget = pn.widgets.Checkbox(name="Tzy", value=False)
        model_err_z_widget = pn.widgets.FloatInput(
            name="Z multiplier", value=5.0, width=100
        )
        model_err_t_widget = pn.widgets.FloatInput(
            name="T add (abs)", value=0.05, width=100
        )
        model_err_pmin_widget = pn.widgets.FloatInput(
            name="Min period (s)", value=period_min, width=120
        )
        model_err_pmax_widget = pn.widgets.FloatInput(
            name="Max period (s)", value=period_max, width=120
        )
        model_err_apply_button = pn.widgets.Button(
            name="Apply Model Error", button_type="primary", width=160
        )
        model_err_status = pn.pane.Markdown("", styles={"color": "#555"})

        def _apply_add_model_error(_event):
            selected = {
                "zxx": model_err_zxx_widget.value,
                "zxy": model_err_zxy_widget.value,
                "zyx": model_err_zyx_widget.value,
                "zyy": model_err_zyy_widget.value,
                "tzx": model_err_tzx_widget.value,
                "tzy": model_err_tzy_widget.value,
            }
            comps = [key for key, value in selected.items() if value]
            if not comps:
                model_err_status.object = "⚠️ Select one or more components."
                model_err_status.styles = {"color": "#7a5200"}
                return

            try:
                pmin = float(model_err_pmin_widget.value)
                pmax = float(model_err_pmax_widget.value)
                self.add_model_error(
                    comps,
                    z_value=float(model_err_z_widget.value),
                    t_value=float(model_err_t_widget.value),
                    periods=(pmin, pmax),
                )
            except Exception as error:
                model_err_status.object = f"⚠️ Add model error failed: {error}"
                model_err_status.styles = {"color": "#7a0000"}
                return

            self.plot_model_error = True
            error_widget.value = "model"
            _refresh_plot_and_widgets()
            model_err_status.object = (
                f"Applied model error to: {', '.join(comps)} "
                f"({pmin:.4g}s to {pmax:.4g}s)."
            )
            model_err_status.styles = {"color": "#1a6600"}

        model_err_apply_button.on_click(_apply_add_model_error)

        model_err_row = pn.Row(
            model_err_zxx_widget,
            model_err_zxy_widget,
            model_err_zyx_widget,
            model_err_zyy_widget,
            model_err_tzx_widget,
            model_err_tzy_widget,
            model_err_z_widget,
            model_err_t_widget,
            model_err_pmin_widget,
            model_err_pmax_widget,
            model_err_apply_button,
            model_err_status,
            align="center",
        )
        model_err_card = pn.Card(
            model_err_row,
            title="Add Model Error",
            collapsed=True,
        )

        options = {
            "xy": "Zxy",
            "yx": "Zyx",
            "xx": "Zxx",
            "yy": "Zyy",
        }

        def _available():
            return [key for key in options if key in self.renderers]

        component_widget = pn.widgets.CheckButtonGroup(
            name="Visible Components",
            options={options[key]: key for key in _available()},
            value=_available(),
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

        # ── per-component color / marker / size styling ───────────────────────
        _style_defs = [
            ("xy", "Zxy", "xy_color", "xy_marker"),
            ("yx", "Zyx", "yx_color", "yx_marker"),
            ("xx", "Zxx", "xx_color", "xx_marker"),
            ("yy", "Zyy", "yy_color", "yy_marker"),
        ]
        _marker_options = ["o", "s", "v", "d", "^"]
        _style_widgets = {}

        marker_size_widget = pn.widgets.IntSlider(
            name="Marker size",
            start=2,
            end=20,
            value=self.marker_size,
            width=160,
        )

        lw_widget = pn.widgets.FloatSlider(
            name="Line width",
            start=0.5,
            end=5.0,
            step=0.25,
            value=float(self.lw),
            width=160,
        )

        # Mutable Row whose objects are replaced after each replot
        style_row = pn.Row()

        def _build_style_row(avail):
            """Rebuild style_row contents to match currently available components."""
            _style_widgets.clear()
            new_cols = []
            for key, label, color_attr, marker_attr in _style_defs:
                if key not in avail:
                    continue
                cw = pn.widgets.ColorPicker(
                    name=f"{label} color",
                    value=getattr(self, color_attr),
                    width=60,
                )
                mw = pn.widgets.Select(
                    name=f"{label} marker",
                    options=_marker_options,
                    value=getattr(self, marker_attr),
                    width=80,
                )
                _style_widgets[key] = (cw, mw, color_attr, marker_attr)
                cw.param.watch(_refresh_after_style_change, "value")
                mw.param.watch(_refresh_after_style_change, "value")
                new_cols.append(
                    pn.Column(
                        pn.pane.Markdown(f"**{label}**", width=90),
                        cw,
                        mw,
                        width=100,
                    )
                )
            new_cols.append(
                pn.Column(pn.pane.Markdown("**Marker size**"), marker_size_widget)
            )
            new_cols.append(pn.Column(pn.pane.Markdown("**Line width**"), lw_widget))
            style_row.objects = new_cols

        def _apply_edit_mode(_event=None):
            """Render the (only) edit-mode layout and refresh dependent widgets."""
            self.res_limits = None
            self.plot()
            bokeh_pane.object = self.layout
            avail = _available()
            component_widget.options = {options[k]: k for k in avail}
            component_widget.value = avail
            self._set_component_visibility(avail)
            _build_style_row(avail)
            self._set_period_window(period_widget.value)
            bokeh_pane.param.trigger("object")

        def _refresh_from_error_mode(event):
            self.plot_model_error = event.new == "model"
            _apply_edit_mode()

        def _update_visibility(event):
            self._set_component_visibility(event.new)
            bokeh_pane.param.trigger("object")

        def _update_period(event):
            self._set_period_window(event.new)

        def _refresh_after_style_change(event):
            for _key, (_cw, _mw, _ca, _ma) in _style_widgets.items():
                setattr(self, _ca, _cw.value)
                setattr(self, _ma, _mw.value)
            self.marker_size = marker_size_widget.value
            self.lw = lw_widget.value
            _apply_edit_mode()

        error_widget.param.watch(_refresh_from_error_mode, "value")
        component_widget.param.watch(_update_visibility, "value")
        period_widget.param.watch(_update_period, "value")
        marker_size_widget.param.watch(_refresh_after_style_change, "value")
        lw_widget.param.watch(_refresh_after_style_change, "value")
        # Initialise: only edit mode is supported
        _apply_edit_mode()

        style_card = pn.Card(
            style_row,
            title="Component Styling",
            collapsed=True,
        )

        controls = pn.Row(
            pn.Column(pn.pane.Markdown("**Components**"), component_widget),
            pn.Column(pn.pane.Markdown("**Error Type**"), error_widget),
            pn.Column(pn.pane.Markdown("**Period Window**"), period_widget),
        )
        edit_controls = pn.Row(
            mask_selected_button,
            add_model_error_selected_button,
            reset_button,
            selection_status,
            align="center",
        )

        cards = pn.Row(
            style_card,
            interp_card,
            ss_card,
            rotate_card,
            flip_card,
            model_err_card,
        )

        return pn.Column(
            pn.pane.Markdown(f"## {title}"),
            controls,
            cards,
            edit_controls,
            bokeh_pane,
            sizing_mode=sizing_mode,
        )
