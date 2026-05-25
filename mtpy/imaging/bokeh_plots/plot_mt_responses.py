"""Bokeh implementation of multiple-station MT response plotting.

This module provides a Bokeh translation of
`mtpy.imaging.plot_mt_responses.PlotMultipleResponses`.
"""

from __future__ import annotations

import numpy as np

from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase
from mtpy.imaging.bokeh_plots.plot_mt_response import PlotMTResponse


try:
    from bokeh.io import show
    from bokeh.layouts import Column, Row
    from bokeh.models import (
        Arrow,
        BasicTicker,
        ColorBar,
        ColumnDataSource,
        Div,
        HoverTool,
        LinearColorMapper,
        NormalHead,
        Range1d,
    )
    from bokeh.palettes import Turbo256
    from bokeh.transform import linear_cmap
except ImportError:  # pragma: no cover - optional dependency
    show = None
    Column = None
    Row = None
    Arrow = None
    BasicTicker = None
    ColorBar = None
    ColumnDataSource = None
    Div = None
    HoverTool = None
    LinearColorMapper = None
    NormalHead = None
    Range1d = None
    Turbo256 = None
    linear_cmap = None


class PlotMultipleResponses(BokehPlotBase):
    """Plot multiple MT responses using Bokeh layouts.

    Parameters
    ----------
    mt_data : object
        Container holding MT objects. Supports either:
        - objects exposing ``values()`` with MT objects as values
        - MTData-like objects exposing ``_iter_station_paths()`` and
          ``get_station()``

    Notes
    -----
    ``plot_style`` options:

    - ``"1"`` or ``"single"``: one plot per station (returned as dict)
    - ``"all"``: full response per station arranged side-by-side
    - ``"compare"``: component rows arranged by station for visual comparison
    """

    def __init__(self, mt_data, **kwargs):
        self.mt_data = mt_data

        # Multi-station layout options (plain attrs, not param-managed).
        self.plot_style = kwargs.pop("plot_style", "1")
        self.plot_num = kwargs.pop("plot_num", 1)
        self.include_survey = kwargs.pop("include_survey", True)
        self.compare_mode = kwargs.pop("compare_mode", "overlay")
        self.compare_legend_mode = kwargs.pop("compare_legend_mode", "station")
        self.compare_compact_legend = kwargs.pop("compare_compact_legend", True)
        self.compare_legend_max_items = kwargs.pop("compare_legend_max_items", 16)
        self.compare_legend_font_size = kwargs.pop("compare_legend_font_size", 8)

        self.layout = None
        self.layouts = {}
        self.plotters = {}

        # Per-station style overrides populated by panel() in compare mode.
        # Maps station_label -> {"color": hex_str, "marker": marker_str}.
        self.custom_station_styles: dict = {}

        # Rotation angle backing; applied to all MT objects when set.
        self._rotation_angle = 0

        # model_error flag — set before super() so the property setter works.
        self._plot_model_error = False
        self._error_str = "error"

        # param.Parameterized raises TypeError for unknown kwargs; filter them.
        param_names = set(type(self).param)
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in param_names}

        super().__init__(**param_kwargs)

        # Apply any remaining kwargs as plain attributes.
        for key, value in other_kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    @property
    def rotation_angle(self):
        """Rotation angle applied to all MT objects in the collection."""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        """Apply rotation to all MT objects."""
        if hasattr(self.mt_data, "rotate") and hasattr(self.mt_data, "get_station"):
            self.mt_data.rotate(value, inplace=True)
        else:
            for tf in self._iter_mt_objects():
                tf.rotation_angle = value
        self._rotation_angle = value

    @property
    def plot_model_error(self):
        """Plot model error instead of data error."""
        return self._plot_model_error

    @plot_model_error.setter
    def plot_model_error(self, value):
        """Toggle model/data error mode."""
        if value:
            self._error_str = "model_error"
        else:
            self._error_str = "error"
        self._plot_model_error = bool(value) if value is not None else False

    def _require_bokeh(self):
        if Column is None or Row is None or Div is None:
            raise ImportError(
                "Bokeh is required for PlotMultipleResponses. Install with `pip install bokeh`."
            )

    def _iter_mt_objects(self):
        """Yield MT objects from supported containers."""
        if hasattr(self.mt_data, "values"):
            yield from self.mt_data.values()
            return

        if hasattr(self.mt_data, "_iter_station_paths") and hasattr(
            self.mt_data, "get_station"
        ):
            if hasattr(self.mt_data, "compute"):
                self.mt_data.compute()
            for station_path in self.mt_data._iter_station_paths():
                yield self.mt_data.get_station(station_path, as_mt=True)
            return

        raise TypeError("mt_data must provide values() or MTData-style station access")

    def _get_mt_objects(self):
        return list(self._iter_mt_objects())

    def _station_label(self, mt_obj):
        station = getattr(mt_obj, "station", "Unknown")
        if self.include_survey:
            survey_id = getattr(getattr(mt_obj, "survey_metadata", None), "id", None)
            if survey_id:
                return f"{survey_id}.{station}"
        return station

    def _global_x_limits(self, mt_objects):
        mins = []
        maxs = []
        for mt_obj in mt_objects:
            period = np.asarray(mt_obj.period, dtype=float)
            finite = period[np.isfinite(period) & (period > 0)]
            if finite.size > 0:
                mins.append(finite.min())
                maxs.append(finite.max())

        if not mins:
            return None

        return [
            10 ** np.floor(np.log10(min(mins))),
            10 ** np.ceil(np.log10(max(maxs))),
        ]

    def _make_station_plotter(self, mt_obj, x_limits=None):
        kwargs = {
            "show_plot": False,
            "plot_num": self.plot_num,
            "plot_tipper": self.plot_tipper,
            "plot_pt": self.plot_pt,
            "plot_model_error": self.plot_model_error,
        }

        if x_limits is not None:
            kwargs["x_limits"] = x_limits

        # Forward common visual options when present on this instance.
        forward_attrs = [
            "res_limits",
            "phase_limits",
            "tipper_limits",
            "font_size",
            "marker_size",
            "lw",
            "xy_color",
            "yx_color",
            "det_color",
            "xy_marker",
            "yx_marker",
            "det_marker",
            "arrow_color_real",
            "arrow_color_imag",
            "arrow_direction",
            "ellipse_size",
            "ellipse_colorby",
            "ellipse_range",
            "ellipse_cmap",
        ]
        for attr in forward_attrs:
            if hasattr(self, attr):
                kwargs[attr] = getattr(self, attr)

        return PlotMTResponse(
            z_object=mt_obj.Z,
            t_object=mt_obj.Tipper,
            pt_obj=mt_obj.pt,
            station=self._station_label(mt_obj),
            **kwargs,
        )

    def _plot_single(self):
        """Return one layout per station."""
        mt_objects = self._get_mt_objects()
        x_limits = self._global_x_limits(mt_objects)

        layout_dict = {}
        plotter_dict = {}
        for mt_obj in mt_objects:
            plotter = self._make_station_plotter(mt_obj, x_limits=x_limits)
            layout_dict[self._station_label(mt_obj)] = plotter.plot()
            plotter_dict[self._station_label(mt_obj)] = plotter

        self.layouts = layout_dict
        self.plotters = plotter_dict
        return layout_dict

    def _plot_all(self):
        """Arrange full station responses side-by-side."""
        mt_objects = self._get_mt_objects()
        x_limits = self._global_x_limits(mt_objects)

        station_columns = []
        self.plotters = {}
        for mt_obj in mt_objects:
            station = self._station_label(mt_obj)
            plotter = self._make_station_plotter(mt_obj, x_limits=x_limits)
            station_layout = plotter.plot()
            self.plotters[station] = plotter

            station_columns.append(
                Column(
                    Div(text=f"<b>{station}</b>"),
                    station_layout,
                    sizing_mode="fixed",
                )
            )

        self.layout = Row(*station_columns, sizing_mode="fixed")
        return self.layout

    def _set_compare_legends(self, figures):
        """Apply compare legend formatting with optional compaction."""
        for fig in figures:
            if fig is None or len(fig.legend) == 0:
                continue
            legend = fig.legend[0]
            legend.click_policy = "hide"
            legend.location = "bottom_left"
            legend.label_text_font_size = f"{self.compare_legend_font_size}px"
            if (
                self.compare_compact_legend
                and len(legend.items) > self.compare_legend_max_items
            ):
                legend.items = legend.items[: self.compare_legend_max_items]

    def _compare_legend_label(self, station, component):
        """Return compare legend label based on configured mode."""
        mode = str(self.compare_legend_mode).lower()
        if mode in ["station", "stations"]:
            return station
        return f"{station}_{component}"

    def _plot_compare_overlay(self):
        """Overlay multiple stations on shared compare axes."""
        mt_objects = self._get_mt_objects()
        x_limits = self._global_x_limits(mt_objects)

        ns = len(mt_objects)
        if ns == 0:
            raise ValueError("No MT objects available to compare.")

        # Match matplotlib compare color progression (dark to light).
        # Convert RGB tuples to hex so they satisfy param.Color validation on
        # BokehPlotBase (which requires CSS colour strings).
        _to_hex = PlotMTResponse._tuple_to_hex
        cxy = [_to_hex((0, float(cc) / ns, 1 - float(cc) / ns)) for cc in range(ns)]
        cyx = [_to_hex((1, float(cc) / ns, 0)) for cc in range(ns)]
        cdet = [_to_hex((0, 1 - float(cc) / ns, 0)) for cc in range(ns)]
        ctipr = [
            _to_hex((0.75 * cc / ns, 0.75 * cc / ns, 0.75 * cc / ns))
            for cc in range(ns)
        ]
        ctipi = [
            _to_hex((float(cc) / ns, 1 - float(cc) / ns, 0.25)) for cc in range(ns)
        ]

        mxy = ["s", "d", "+", "x", "^", "*", "v", "o"]
        myx = ["o", "v", "^", "*", "+", "x", "s", "d"]

        base = self._make_station_plotter(mt_objects[0], x_limits=x_limits)
        base._require_bokeh()
        base.renderers = {}

        n_columns = 2 if self.plot_num in [1, 2] else 1
        panel_width = 800 * n_columns

        if self.plot_num in [1, 2]:
            res_fig_xy = base._make_resistivity_figure()
            res_fig_yx = base._make_resistivity_figure(x_range=res_fig_xy.x_range)
            phase_fig_xy = base._make_phase_figure(res_fig_xy.x_range)
            phase_fig_yx = base._make_phase_figure(res_fig_xy.x_range)
        else:
            res_fig_xy = base._make_resistivity_figure()
            res_fig_yx = None
            phase_fig_xy = base._make_phase_figure(res_fig_xy.x_range)
            phase_fig_yx = None

        tip_fig = None
        tip_min = np.inf
        tip_max = -np.inf
        if self.plot_tipper.find("y") >= 0:
            tip_fig = base._make_tipper_figure(width=panel_width)

        pt_fig = None
        pt_mapper = None
        pt_spacing = 1.0
        if self.plot_pt:
            pt_fig = base._make_pt_figure(width=panel_width)
            cmin, cmax = self.ellipse_range[0], self.ellipse_range[1]
            pt_mapper = LinearColorMapper(palette=Turbo256, low=cmin, high=cmax)
            pt_fig.add_layout(
                ColorBar(
                    color_mapper=pt_mapper,
                    ticker=BasicTicker(),
                    label_standoff=8,
                    title=base.cb_label_dict[self.ellipse_colorby],
                ),
                "right",
            )

            fig_width = pt_fig.width if pt_fig.width else panel_width
            fig_height = pt_fig.height if pt_fig.height else 240
            x_log_range = np.log10(x_limits[1]) - np.log10(x_limits[0])
            pt_ylim = 1.5 * self.ellipse_size
            pt_spacing = (2 * pt_ylim * fig_width) / (x_log_range * fig_height)

        res_limits_list = []
        phase_limits_list = []

        self.plotters = {}
        for ii, mt_obj in enumerate(mt_objects):
            station = self._station_label(mt_obj)
            self.plotters[station] = base

            base.xy_color = cxy[ii]
            base.yx_color = cyx[ii]
            base.det_color = cdet[ii]
            base.xy_marker = mxy[ii % len(mxy)]
            base.yx_marker = myx[ii % len(myx)]

            # Apply per-station overrides from panel() styling widgets.
            _custom = self.custom_station_styles.get(station, {})
            if _custom:
                _color = _custom.get("color")
                _marker = _custom.get("marker")
                if _color:
                    base.xy_color = _color
                    base.yx_color = _color
                    base.det_color = _color
                if _marker:
                    base.xy_marker = _marker
                    base.yx_marker = _marker

            period = np.asarray(mt_obj.period, dtype=float)

            if self.plot_num == 1:
                src_xy_res = base._component_source(period, mt_obj.Z, "xy", kind="res")
                src_yx_res = base._component_source(period, mt_obj.Z, "yx", kind="res")
                src_xy_phase = base._component_source(
                    period, mt_obj.Z, "xy", kind="phase"
                )
                src_yx_phase = base._component_source(
                    period, mt_obj.Z, "yx", kind="phase", yx_shift=True
                )

                base._add_component(
                    res_fig_xy,
                    src_xy_res,
                    self._compare_legend_label(station, "Zxy"),
                    base.xy_color,
                    base.xy_marker,
                    f"xy_{ii}",
                )
                base._add_component(
                    res_fig_yx,
                    src_yx_res,
                    self._compare_legend_label(station, "Zyx"),
                    base.yx_color,
                    base.yx_marker,
                    f"yx_{ii}",
                )
                base._add_component(
                    phase_fig_xy,
                    src_xy_phase,
                    self._compare_legend_label(station, "Zxy"),
                    base.xy_color,
                    base.xy_marker,
                    f"xy_p_{ii}",
                )
                base._add_component(
                    phase_fig_yx,
                    src_yx_phase,
                    self._compare_legend_label(station, "Zyx"),
                    base.yx_color,
                    base.yx_marker,
                    f"yx_p_{ii}",
                )

                res_limits_list.append(
                    base.set_resistivity_limits(mt_obj.Z.resistivity, mode="od")
                )
                phase_limits_list.append(
                    base.set_phase_limits(mt_obj.Z.phase, mode="od")
                )
            elif self.plot_num == 2:
                src_xy_res = base._component_source(period, mt_obj.Z, "xy", kind="res")
                src_yx_res = base._component_source(period, mt_obj.Z, "yx", kind="res")
                src_xy_phase = base._component_source(
                    period, mt_obj.Z, "xy", kind="phase"
                )
                src_yx_phase = base._component_source(
                    period, mt_obj.Z, "yx", kind="phase", yx_shift=True
                )

                src_xx_res = base._component_source(period, mt_obj.Z, "xx", kind="res")
                src_yy_res = base._component_source(period, mt_obj.Z, "yy", kind="res")
                src_xx_phase = base._component_source(
                    period, mt_obj.Z, "xx", kind="phase"
                )
                src_yy_phase = base._component_source(
                    period, mt_obj.Z, "yy", kind="phase"
                )

                base._add_component(
                    res_fig_xy,
                    src_xy_res,
                    self._compare_legend_label(station, "Zxy"),
                    base.xy_color,
                    base.xy_marker,
                    f"xy_{ii}",
                )
                base._add_component(
                    res_fig_xy,
                    src_yx_res,
                    self._compare_legend_label(station, "Zyx"),
                    base.yx_color,
                    base.yx_marker,
                    f"yx_{ii}",
                )
                base._add_component(
                    phase_fig_xy,
                    src_xy_phase,
                    self._compare_legend_label(station, "Zxy"),
                    base.xy_color,
                    base.xy_marker,
                    f"xy_p_{ii}",
                )
                base._add_component(
                    phase_fig_xy,
                    src_yx_phase,
                    self._compare_legend_label(station, "Zyx"),
                    base.yx_color,
                    base.yx_marker,
                    f"yx_p_{ii}",
                )

                base._add_component(
                    res_fig_yx,
                    src_xx_res,
                    self._compare_legend_label(station, "Zxx"),
                    base.xy_color,
                    base.xy_marker,
                    f"xx_{ii}",
                )
                base._add_component(
                    res_fig_yx,
                    src_yy_res,
                    self._compare_legend_label(station, "Zyy"),
                    base.yx_color,
                    base.yx_marker,
                    f"yy_{ii}",
                )
                base._add_component(
                    phase_fig_yx,
                    src_xx_phase,
                    self._compare_legend_label(station, "Zxx"),
                    base.xy_color,
                    base.xy_marker,
                    f"xx_p_{ii}",
                )
                base._add_component(
                    phase_fig_yx,
                    src_yy_phase,
                    self._compare_legend_label(station, "Zyy"),
                    base.yx_color,
                    base.yx_marker,
                    f"yy_p_{ii}",
                )

                res_od_limits = base.set_resistivity_limits(
                    mt_obj.Z.resistivity, mode="od"
                )
                res_d_limits = base.set_resistivity_limits(
                    mt_obj.Z.resistivity, mode="d"
                )
                phase_od_limits = base.set_phase_limits(mt_obj.Z.phase, mode="od")
                phase_d_limits = base.set_phase_limits(mt_obj.Z.phase, mode="d")

                if res_od_limits is not None and res_d_limits is not None:
                    res_limits_list.append(
                        [
                            min(res_od_limits[0], res_d_limits[0]),
                            max(res_od_limits[1], res_d_limits[1]),
                        ]
                    )
                elif res_od_limits is not None:
                    res_limits_list.append(res_od_limits)
                elif res_d_limits is not None:
                    res_limits_list.append(res_d_limits)

                if phase_od_limits is not None and phase_d_limits is not None:
                    phase_limits_list.append(
                        [
                            min(phase_od_limits[0], phase_d_limits[0]),
                            max(phase_od_limits[1], phase_d_limits[1]),
                        ]
                    )
                elif phase_od_limits is not None:
                    phase_limits_list.append(phase_od_limits)
                elif phase_d_limits is not None:
                    phase_limits_list.append(phase_d_limits)
            else:
                src_det_res = base._component_source(
                    period, mt_obj.Z, "det", kind="res"
                )
                src_det_phase = base._component_source(
                    period, mt_obj.Z, "det", kind="phase"
                )

                base._add_component(
                    res_fig_xy,
                    src_det_res,
                    self._compare_legend_label(station, "det(Z)"),
                    base.det_color,
                    base.xy_marker,
                    f"det_{ii}",
                )
                base._add_component(
                    phase_fig_xy,
                    src_det_phase,
                    self._compare_legend_label(station, "det(Z)"),
                    base.det_color,
                    base.xy_marker,
                    f"det_p_{ii}",
                )

                res_limits_list.append(
                    base.set_resistivity_limits(mt_obj.Z.resistivity, mode="det")
                )
                phase_limits_list.append(
                    base.set_phase_limits(mt_obj.Z.phase, mode="det")
                )

            if tip_fig is not None and mt_obj.Tipper is not None:
                base.Tipper = mt_obj.Tipper
                base.arrow_color_real = ctipr[ii]
                base.arrow_color_imag = ctipi[ii]

                vectors = base._tipper_vectors()
                if vectors["x0"].size > 0:
                    tip_min = min(tip_min, np.nanmin([vectors["yr"], vectors["yi"]]))
                    tip_max = max(tip_max, np.nanmax([vectors["yr"], vectors["yi"]]))

                    source = ColumnDataSource(
                        data={
                            "x0": vectors["x0"],
                            "y0": vectors["y0"],
                            "xr": vectors["xr"],
                            "yr": vectors["yr"],
                            "xi": vectors["xi"],
                            "yi": vectors["yi"],
                            "station": [station] * len(vectors["x0"]),
                        }
                    )

                    real_color = base._tuple_to_hex(base.arrow_color_real)
                    imag_color = base._tuple_to_hex(base.arrow_color_imag)

                    if "r" in self.plot_tipper:
                        tip_fig.add_layout(
                            Arrow(
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
                        )
                        tip_fig.line(
                            x=[np.nan],
                            y=[np.nan],
                            color=real_color,
                            line_width=max(self.arrow_lw * 2, 1),
                            legend_label=self._compare_legend_label(
                                station, "tip real"
                            ),
                        )

                    if "i" in self.plot_tipper:
                        tip_fig.add_layout(
                            Arrow(
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
                        )
                        tip_fig.line(
                            x=[np.nan],
                            y=[np.nan],
                            color=imag_color,
                            line_dash="dashed",
                            line_width=max(self.arrow_lw * 2, 1),
                            legend_label=self._compare_legend_label(
                                station, "tip imag"
                            ),
                        )

            if pt_fig is not None and mt_obj.pt is not None:
                period_pt = np.asarray(1.0 / mt_obj.pt.frequency, dtype=float)
                phimin = np.asarray(mt_obj.pt.phimin, dtype=float)
                phimax = np.asarray(mt_obj.pt.phimax, dtype=float)
                azimuth = np.asarray(mt_obj.pt.azimuth, dtype=float)
                color_array = np.asarray(
                    base.get_pt_color_array(mt_obj.pt), dtype=float
                )

                x = np.log10(period_pt) * pt_spacing
                valid = (
                    np.isfinite(x)
                    & np.isfinite(phimin)
                    & np.isfinite(phimax)
                    & (phimax > 0)
                    & np.isfinite(azimuth)
                    & np.isfinite(color_array)
                )

                phimax_station = np.nanmax(phimax[valid]) if np.any(valid) else np.nan
                scaling = (
                    self.ellipse_size / phimax_station
                    if np.isfinite(phimax_station) and phimax_station > 0
                    else 0.0
                )

                y_shift = ii * self.ellipse_size
                src_pt = ColumnDataSource(
                    data={
                        "x": x[valid],
                        "y": np.full(np.count_nonzero(valid), y_shift),
                        "width": phimax[valid] * scaling,
                        "height": phimin[valid] * scaling,
                        "angle": np.deg2rad(90.0 - azimuth[valid]),
                        "color_value": color_array[valid],
                        "station": [station] * np.count_nonzero(valid),
                    }
                )

                edge_color = base._tuple_to_hex(cxy[ii])
                pt_renderer = pt_fig.ellipse(
                    x="x",
                    y="y",
                    width="width",
                    height="height",
                    angle="angle",
                    source=src_pt,
                    fill_color=linear_cmap(
                        "color_value",
                        Turbo256,
                        self.ellipse_range[0],
                        self.ellipse_range[1],
                    ),
                    fill_alpha=0.9,
                    line_color=edge_color,
                    line_width=0.8,
                )
                pt_fig.add_tools(
                    HoverTool(
                        renderers=[pt_renderer],
                        tooltips=[
                            ("Station", "@station"),
                            ("Color", "@color_value{0.0}"),
                        ],
                    )
                )

        res_limits_list = [
            lim
            for lim in res_limits_list
            if lim is not None
            and len(lim) == 2
            and lim[0] is not None
            and lim[1] is not None
        ]
        phase_limits_list = [
            lim
            for lim in phase_limits_list
            if lim is not None
            and len(lim) == 2
            and lim[0] is not None
            and lim[1] is not None
        ]

        if res_limits_list:
            res_limits = self.res_limits
            if res_limits is None:
                res_limits = [
                    min(lim[0] for lim in res_limits_list),
                    max(lim[1] for lim in res_limits_list),
                ]
            base._set_axis_limits(res_fig_xy, res_limits)
            if res_fig_yx is not None:
                base._set_axis_limits(res_fig_yx, res_limits)

        if phase_limits_list:
            phase_limits = self.phase_limits
            if phase_limits is None:
                phase_limits = [
                    min(lim[0] for lim in phase_limits_list),
                    max(lim[1] for lim in phase_limits_list),
                ]
            base._set_axis_limits(phase_fig_xy, phase_limits)
            if phase_fig_yx is not None:
                base._set_axis_limits(phase_fig_yx, phase_limits)

        base._format_res_axis(res_fig_xy)
        base._format_phase_axis(phase_fig_xy)
        if res_fig_yx is not None:
            base._format_res_axis(res_fig_yx)
            base._format_phase_axis(phase_fig_yx)
            if self.plot_num == 2:
                res_fig_yx.yaxis.axis_label = ""
                phase_fig_yx.yaxis.axis_label = ""

        # Match matplotlib compare where top row has no x tick labels.
        if res_fig_yx is not None:
            res_fig_xy.xaxis.visible = False
            res_fig_yx.xaxis.visible = False

        base._add_hover(res_fig_xy)
        base._add_hover(phase_fig_xy)
        if res_fig_yx is not None:
            base._add_hover(res_fig_yx)
            base._add_hover(phase_fig_yx)

        legends = [res_fig_xy, phase_fig_xy]
        if res_fig_yx is not None:
            legends.extend([res_fig_yx, phase_fig_yx])

        if tip_fig is not None:
            tip_fig.line(
                x=[np.log10(x_limits[0]), np.log10(x_limits[1])],
                y=[0, 0],
                color="#444444",
                line_width=1,
                line_alpha=0.5,
            )
            tip_fig.x_range.start = np.log10(x_limits[0])
            tip_fig.x_range.end = np.log10(x_limits[1])
            if np.isfinite(tip_min) and np.isfinite(tip_max):
                tip_fig.y_range.start = max(tip_min - 0.1, -1.0)
                tip_fig.y_range.end = min(tip_max + 0.1, 1.0)
            tip_fig.yaxis.axis_label = "Tipper"
            tip_fig.grid.grid_line_alpha = 0.25
            base._apply_log_period_ticks(tip_fig)
            legends.append(tip_fig)

        if pt_fig is not None:
            pt_ylim = 1.5 * self.ellipse_size
            x_pad = 0.5 * self.ellipse_size
            pt_fig.x_range.start = np.log10(x_limits[0]) * pt_spacing - x_pad
            pt_fig.x_range.end = np.log10(x_limits[1]) * pt_spacing + x_pad
            y_top = (ns - 1) * self.ellipse_size + pt_ylim
            pt_fig.y_range = Range1d(-pt_ylim, y_top)
            pt_fig.yaxis.visible = False
            pt_fig.grid.grid_line_alpha = 0.25
            base._apply_log_period_ticks(pt_fig, x_spacing=pt_spacing)
            legends.append(pt_fig)

        self._set_compare_legends(legends)

        if self.plot_num in [1, 2]:
            rows = [
                Row(res_fig_xy, res_fig_yx, width=panel_width, sizing_mode="fixed"),
                Row(phase_fig_xy, phase_fig_yx, width=panel_width, sizing_mode="fixed"),
            ]
        else:
            rows = [res_fig_xy, phase_fig_xy]

        if tip_fig is not None:
            rows.append(tip_fig)
        if pt_fig is not None:
            rows.append(pt_fig)

        self.layout = Column(*rows, width=panel_width, sizing_mode="fixed")
        return self.layout

    def _plot_compare_tiled(self):
        """Tiled compare mode: station columns grouped by component rows."""
        mt_objects = self._get_mt_objects()
        x_limits = self._global_x_limits(mt_objects)

        self.plotters = {}
        for mt_obj in mt_objects:
            station = self._station_label(mt_obj)
            plotter = self._make_station_plotter(mt_obj, x_limits=x_limits)
            plotter.plot()
            self.plotters[station] = plotter

        stations = list(self.plotters.keys())
        title_row = Row(
            *[Div(text=f"<b>{station}</b>") for station in stations],
            sizing_mode="fixed",
        )
        rows = [title_row]

        if self.plot_num == 1:
            component_order = ["res", "phase", "tip", "pt"]
        elif self.plot_num == 2:
            component_order = ["res", "res_diag", "phase", "phase_diag", "tip", "pt"]
        else:
            component_order = ["res", "phase", "tip", "pt"]

        for key in component_order:
            figures = []
            for station in stations:
                fig = self.plotters[station].figures.get(key)
                if fig is not None:
                    figures.append(fig)
            if figures:
                rows.append(Row(*figures, sizing_mode="fixed"))

        self.layout = Column(*rows, sizing_mode="fixed")
        return self.layout

    def _plot_compare(self):
        """Dispatch compare layout mode."""
        if str(self.compare_mode).lower() == "tiled":
            return self._plot_compare_tiled()
        return self._plot_compare_overlay()

    def plot(self):
        """Create Bokeh multi-station response layout."""
        self._require_bokeh()

        if self.plot_style == "all":
            self.layout = self._plot_all()
            if self.show_plot:
                show(self.layout)
            return self.layout

        if self.plot_style == "compare":
            self.layout = self._plot_compare()
            if self.show_plot:
                show(self.layout)
            return self.layout

        if self.plot_style in [1, "1", "single"]:
            return self._plot_single()

        raise ValueError(
            f"Unsupported plot_style '{self.plot_style}'. Use '1', 'all', or 'compare'."
        )

    def make_panel(self, sizing_mode="stretch_width", interactive=False):
        """Return a Panel object wrapping the multi-station Bokeh layout."""
        try:
            import panel as pn
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Panel is required to create a panel object. Install with `pip install panel`."
            ) from error

        result = self.plot()

        if isinstance(result, dict):
            # single style
            children = []
            for station, layout in result.items():
                if interactive:
                    station_plotter = self.plotters[station]
                    children.append(
                        station_plotter.make_panel(
                            sizing_mode=sizing_mode, interactive=True
                        )
                    )
                else:
                    children.append(
                        pn.Column(
                            pn.pane.Markdown(f"## {station}"), pn.pane.Bokeh(layout)
                        )
                    )
            return pn.Column(*children, sizing_mode=sizing_mode)

        return pn.pane.Bokeh(result, sizing_mode=sizing_mode)

    def panel(self, sizing_mode="stretch_width"):
        """Return an interactive Panel app for multi-station MT responses.

        Provides station selection checkboxes, a plot-style toggle
        (``"single"`` / ``"compare"``), tensor-preset selector, a Generate
        button, and – for compare mode – per-station colour/marker styling.

        Parameters
        ----------
        sizing_mode : str
            Panel ``sizing_mode`` forwarded to the outer container.

        Returns
        -------
        panel.Column
            A ``panel.Column`` containing all controls and the live plot area.
        """
        try:
            import panel as pn
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "Panel is required to create a panel object.  "
                "Install with `pip install panel`."
            ) from error

        mt_objects = self._get_mt_objects()
        if not mt_objects:
            return pn.pane.Markdown("_No MT data loaded._")

        station_labels = [self._station_label(mt_obj) for mt_obj in mt_objects]
        label_to_mt = dict(zip(station_labels, mt_objects))
        x_limits = self._global_x_limits(mt_objects)

        # ── Widgets ───────────────────────────────────────────────────────────
        style_widget = pn.widgets.RadioButtonGroup(
            name="Plot Style",
            options=["single", "compare"],
            value="single",
            button_type="primary",
            width=200,
        )

        station_widget = pn.widgets.CheckBoxGroup(
            name="Stations",
            options=station_labels,
            value=[],
            inline=False,
            width=240,
        )

        select_all_btn = pn.widgets.Button(
            name="Select All", button_type="light", width=90
        )
        clear_btn = pn.widgets.Button(name="Clear", button_type="light", width=70)

        _preset_labels = ["Off-diagonal", "Full tensor", "All"]
        plot_num_widget = pn.widgets.RadioButtonGroup(
            name="Tensor",
            options=_preset_labels,
            value=_preset_labels[max(0, self.plot_num - 1)],
            button_type="warning",
            width=300,
        )

        generate_btn = pn.widgets.Button(name="Plot", button_type="success", width=120)
        status = pn.pane.Markdown(
            "_Select stations and click **Generate**._",
            styles={"color": "#555"},
        )

        plot_display = pn.Column(sizing_mode=sizing_mode)
        style_card_area = pn.Column()

        # ── Per-station compare styling ────────────────────────────────────────
        _DEFAULT_COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        _MARKER_OPTIONS = ["s", "o", "d", "v", "^", "x", "+"]
        _station_style_widgets: dict = {}

        def _build_compare_style_card(selected_labels):
            _station_style_widgets.clear()
            cols = []
            for ii, label in enumerate(selected_labels):
                default_color = _DEFAULT_COLORS[ii % len(_DEFAULT_COLORS)]
                cw = pn.widgets.ColorPicker(name=label, value=default_color, width=60)
                mw = pn.widgets.Select(
                    name="marker",
                    options=_MARKER_OPTIONS,
                    value=_MARKER_OPTIONS[ii % len(_MARKER_OPTIONS)],
                    width=80,
                )
                _station_style_widgets[label] = (cw, mw)
                cols.append(
                    pn.Column(
                        pn.pane.Markdown(f"**{label}**", margin=(0, 5)),
                        pn.Row(cw, mw, sizing_mode="fixed"),
                        width=180,
                    )
                )

            card_content = (
                pn.FlexBox(*cols)
                if cols
                else pn.pane.Markdown("_No stations selected._")
            )
            style_card_area.objects = [
                pn.Card(card_content, title="Station Styling", collapsed=True)
            ]

        # ── Render callback ────────────────────────────────────────────────────
        def _render(event=None):
            selected = list(station_widget.value)
            if not selected:
                status.object = "⚠️ No stations selected."
                status.styles = {"color": "#7a5200"}
                plot_display.objects = []
                return

            pn_map = {"Off-diagonal": 1, "Full tensor": 2, "All": 3}
            new_plot_num = pn_map[plot_num_widget.value]
            style = style_widget.value
            plot_all = new_plot_num == 3

            status.object = f"⏳ Generating **{style}** for {len(selected)} station(s)…"
            status.styles = {"color": "#555"}

            try:
                if style == "single":
                    style_card_area.objects = []
                    panels = []
                    for label in selected:
                        mt_obj = label_to_mt[label]
                        plotter = self._make_station_plotter(mt_obj, x_limits=x_limits)
                        plotter.plot_num = new_plot_num
                        if plot_all:
                            plotter.plot_tipper = "y"
                            plotter.plot_pt = True
                        plotter.show_plot = False
                        # Use the full station label as the panel title.
                        plotter.station = label
                        panels.append(pn.layout.Divider())
                        panels.append(plotter.panel(interactive=True))

                    plot_display.objects = (
                        panels if panels else [pn.pane.Markdown("_Nothing rendered._")]
                    )
                    status.object = f"✅ {len(selected)} station(s) rendered."
                    status.styles = {"color": "#1a6600"}

                else:  # compare
                    _build_compare_style_card(selected)

                    class _ObjList:
                        def __init__(self, objs):
                            self._objs = objs

                        def values(self):
                            return iter(self._objs)

                    subset_objs = [label_to_mt[lbl] for lbl in selected]
                    cmp = self.__class__(
                        mt_data=_ObjList(subset_objs),
                        show_plot=False,
                    )

                    # Forward visual attrs from self.
                    _forward = [
                        "plot_tipper",
                        "plot_pt",
                        "plot_model_error",
                        "lw",
                        "marker_size",
                        "arrow_lw",
                        "arrow_direction",
                        "ellipse_size",
                        "ellipse_colorby",
                        "ellipse_range",
                        "res_limits",
                        "phase_limits",
                        "tipper_limits",
                        "font_size",
                        "include_survey",
                        "compare_legend_mode",
                        "compare_compact_legend",
                        "compare_legend_max_items",
                        "compare_legend_font_size",
                    ]
                    for attr in _forward:
                        if hasattr(self, attr):
                            try:
                                setattr(cmp, attr, getattr(self, attr))
                            except Exception:
                                pass

                    cmp.plot_num = new_plot_num
                    if plot_all:
                        cmp.plot_tipper = "y"
                        cmp.plot_pt = True

                    # Collect styling from per-station widgets.
                    cmp.custom_station_styles = {
                        label: {"color": cw.value, "marker": mw.value}
                        for label, (cw, mw) in _station_style_widgets.items()
                    }

                    bokeh_layout = cmp._plot_compare_overlay()
                    plot_display.objects = [
                        pn.pane.Bokeh(bokeh_layout, sizing_mode="fixed")
                    ]
                    status.object = f"✅ Compare: {len(selected)} station(s) rendered."
                    status.styles = {"color": "#1a6600"}

            except Exception as exc:
                import traceback

                status.object = (
                    f"❌ `{type(exc).__name__}: {exc}\n\n" f"{traceback.format_exc()}`"
                )
                status.styles = {"color": "#b00020"}

        # ── Wire callbacks ─────────────────────────────────────────────────────
        select_all_btn.on_click(
            lambda e: setattr(station_widget, "value", station_labels)
        )
        clear_btn.on_click(lambda e: setattr(station_widget, "value", []))
        generate_btn.on_click(_render)

        controls = pn.Column(
            pn.pane.Markdown("### MT Responses — Multi-Station"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("**Plot Style**"),
                    style_widget,
                    pn.pane.Markdown("**Tensor Components**"),
                    plot_num_widget,
                    pn.Row(generate_btn),
                    status,
                    width=380,
                ),
                pn.Column(
                    pn.pane.Markdown("**Stations**"),
                    pn.Row(select_all_btn, clear_btn),
                    station_widget,
                ),
                align="start",
            ),
            sizing_mode=sizing_mode,
        )

        return pn.Column(
            controls,
            pn.layout.Divider(),
            style_card_area,
            plot_display,
            sizing_mode=sizing_mode,
        )


__all__ = ["PlotMultipleResponses"]
