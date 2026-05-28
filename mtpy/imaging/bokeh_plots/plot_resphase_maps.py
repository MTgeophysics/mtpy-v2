"""Bokeh implementation of apparent resistivity and phase map plotting."""

from __future__ import annotations

import importlib

import numpy as np
from bokeh.io import show as bokeh_show
from bokeh.layouts import Column, gridplot
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    Div,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    Range1d,
)
from bokeh.plotting import figure

from mtpy.core import Z
from mtpy.imaging.bokeh_plots.bokeh_plot_base_maps import BokehPlotBaseMaps


class PlotResPhaseMaps(BokehPlotBaseMaps):
    """Plot apparent resistivity and phase maps for a target period using Bokeh."""

    def __init__(self, mt_data, **kwargs):
        super().__init__(**kwargs)

        self.mt_data = mt_data
        self._interpolated_mt_data_cache = None
        self._interpolated_mt_data_cache_period = None

        self.map_units = "deg"
        self.scale = 1
        self.res_cmap = "rainbow_r"
        self.phase_cmap = "rainbow"
        self.plot_period = 1

        self.plot_xx = False
        self.plot_xy = True
        self.plot_yx = True
        self.plot_yy = False
        self.plot_det = False

        self.plot_resistivity = True
        self.plot_phase = True

        self.plot_stations = True

        self.marker_color = "#000000"
        self.marker_size = 8

        self.cmap_limits = {
            "res_xx": (-1, 2),
            "res_xy": (0, 3),
            "res_yx": (0, 3),
            "res_yy": (-1, 2),
            "res_det": (0, 3),
            "phase_xx": (-180, 180),
            "phase_xy": (0, 100),
            "phase_yx": (0, 100),
            "phase_yy": (-180, 180),
            "phase_det": (0, 100),
        }

        self.label_dict = {
            "res_xx": r"$\rho_{xx}  \mathrm{[\Omega m]}$",
            "res_xy": r"$\rho_{xy}  \mathrm{[\Omega m]}$",
            "res_yx": r"$\rho_{yx}  \mathrm{[\Omega m]}$",
            "res_yy": r"$\rho_{yy}  \mathrm{[\Omega m]}$",
            "res_det": r"$\rho_{det}  \mathrm{[\Omega m]}$",
            "phase_xx": r"$\phi_{xx}$",
            "phase_xy": r"$\phi_{xy}$",
            "phase_yx": r"$\phi_{yx}$",
            "phase_yy": r"$\phi_{yy}$",
            "phase_det": r"$\phi_{det}$",
        }

        self.fig = None
        self.layout = None
        self.figures = {}
        self.renderers = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot(show=True)

    @property
    def map_units(self):
        return self._map_units

    @map_units.setter
    def map_units(self, value):
        self._map_units = value
        if value in ["km"]:
            self.scale = 1.0 / 1000
            self.cell_size = 0.2
        elif value in ["m"]:
            self.scale = 1.0
            self.cell_size = 200
        else:
            self.scale = 1.0

    def _require_bokeh(self):
        if (
            figure is None
            or Column is None
            or gridplot is None
            or LinearColorMapper is None
            or ColorBar is None
            or BasicTicker is None
            or ColumnDataSource is None
            or Range1d is None
            or FixedTicker is None
            or Div is None
        ):
            raise ImportError(
                "Bokeh is required for PlotResPhaseMaps. Install with `pip install bokeh`."
            )

    def _get_n_rows(self):
        n = 0
        if self.plot_resistivity:
            n += 1
        if self.plot_phase:
            n += 1
        return n

    def _get_n_columns(self):
        n = 0
        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if getattr(self, f"plot_{cc}"):
                n += 1
        return n

    def _get_n_subplots(self):
        nr = self._get_n_rows()
        nc = self._get_n_columns()

        subplot_dict = {
            "res_xx": None,
            "res_xy": None,
            "res_yx": None,
            "res_yy": None,
            "res_det": None,
            "phase_xx": None,
            "phase_xy": None,
            "phase_yx": None,
            "phase_yy": None,
            "phase_det": None,
        }

        plot_num = 0
        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if self.plot_resistivity and getattr(self, f"plot_{cc}"):
                plot_num += 1
                subplot_dict[f"res_{cc}"] = (nr, nc, plot_num)

        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if self.plot_phase and getattr(self, f"plot_{cc}"):
                plot_num += 1
                subplot_dict[f"phase_{cc}"] = (nr, nc, plot_num)

        return subplot_dict

    def _get_data_array(self):
        mt_objects = self._get_mt_objects()

        plot_array = np.zeros(
            len(mt_objects),
            dtype=[
                ("station", "U20"),
                ("latitude", float),
                ("longitude", float),
                ("elevation", float),
                ("res_xx", float),
                ("res_xy", float),
                ("res_yx", float),
                ("res_yy", float),
                ("res_det", float),
                ("phase_xx", float),
                ("phase_xy", float),
                ("phase_yx", float),
                ("phase_yy", float),
                ("phase_det", float),
            ],
        )

        for ii, tf in enumerate(mt_objects):
            try:
                z = self._get_interpolated_z(tf)
            except ValueError:
                self.logger.warning(
                    f"Could not interpolate period {self.plot_period} for station {tf.station}"
                )
                continue

            z_object = Z(
                z,
                frequency=[1.0 / self.plot_period],
                units=tf.impedance_units,
            )

            plot_array["station"][ii] = tf.station
            plot_array["latitude"][ii] = tf.latitude
            plot_array["longitude"][ii] = tf.longitude
            if tf.elevation is not None:
                plot_array["elevation"][ii] = tf.elevation * self.scale

            plot_array["res_xx"][ii] = z_object.res_xx[0]
            plot_array["res_xy"][ii] = z_object.res_xy[0]
            plot_array["res_yx"][ii] = z_object.res_yx[0]
            plot_array["res_yy"][ii] = z_object.res_yy[0]
            plot_array["res_det"][ii] = z_object.res_det[0]

            plot_array["phase_xx"][ii] = z_object.phase_xx[0]
            plot_array["phase_xy"][ii] = z_object.phase_xy[0]
            if z_object.phase_yx[0] != 0:
                plot_array["phase_yx"][ii] = z_object.phase_yx[0] + 180
            plot_array["phase_yy"][ii] = z_object.phase_yy[0]
            plot_array["phase_det"][ii] = z_object.phase_det[0]

        return plot_array

    def _palette_from_name(self, name):
        if name is None:
            return self.palette_options["turbo"]

        lname = str(name).lower()

        return self.palette_options.get(lname, self.palette_options["turbo"])

    def _get_cmap(self, component):
        if "res" in component:
            return self._palette_from_name(self.res_cmap)
        return self._palette_from_name(self.phase_cmap)

    def _image_from_map_output(self, output):
        if hasattr(output[0], "triangles"):
            triangulation, image, _inside_indices = output
            x_values = np.unique(np.asarray(triangulation.x, dtype=float))
            y_values = np.unique(np.asarray(triangulation.y, dtype=float))
            image = np.asarray(image, dtype=float).reshape(len(y_values), len(x_values))
            grid_x, grid_y = np.meshgrid(x_values, y_values)
            return grid_x, grid_y, image

        plot_x, plot_y, image = output
        return (
            np.asarray(plot_x, dtype=float),
            np.asarray(plot_y, dtype=float),
            np.asarray(image, dtype=float),
        )

    def _color_limits(self, component, values):
        vmin, vmax = self.cmap_limits[component]
        if np.isfinite(vmin) and np.isfinite(vmax):
            return float(vmin), float(vmax)

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _axis_labels(self):
        if self.map_units == "deg":
            return "Longitude (deg)", "Latitude (deg)"
        if self.map_units == "km":
            return "Easting (km)", "Northing (km)"
        return "Easting (m)", "Northing (m)"

    def _transform_component_values(self, comp, values):
        values = np.asarray(values, dtype=float)
        if comp.startswith("res_"):
            out = np.full(values.shape, np.nan, dtype=float)
            good = values > 0
            out[good] = np.log10(values[good])
            return out
        return values

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

    def _plot_component(self, fig_obj, plot_array, comp):
        values = self._transform_component_values(comp, plot_array[comp])

        map_array = np.zeros(
            plot_array.shape[0],
            dtype=[("longitude", float), ("latitude", float), (comp, float)],
        )
        map_array["longitude"] = plot_array["longitude"]
        map_array["latitude"] = plot_array["latitude"]
        map_array[comp] = values

        map_output = self.interpolate_to_map(map_array, comp)
        plot_x, plot_y, image = self._image_from_map_output(map_output)

        mapper = LinearColorMapper(
            palette=self._get_cmap(comp),
            low=self._color_limits(comp, values)[0],
            high=self._color_limits(comp, values)[1],
        )

        x0 = float(np.nanmin(plot_x))
        x1 = float(np.nanmax(plot_x))
        y0 = float(np.nanmin(plot_y))
        y1 = float(np.nanmax(plot_y))

        image_renderer = fig_obj.image(
            image=[image],
            x=x0,
            y=y0,
            dw=x1 - x0,
            dh=y1 - y0,
            color_mapper=mapper,
        )

        ticker_model = BasicTicker()
        if comp.startswith("res_"):
            ticks = np.arange(
                int(np.round(self.cmap_limits[comp][0])),
                int(np.round(self.cmap_limits[comp][1])) + 1,
            )
            ticker_model = FixedTicker(ticks=[float(v) for v in ticks])

        fig_obj.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=ticker_model,
                label_standoff=8,
                title=self.label_dict[comp],
            ),
            "right",
        )

        renderers = [image_renderer]

        if self.plot_stations:
            marker_color = self._to_hex(self.marker_color)
            station_source = ColumnDataSource(
                data={
                    "longitude": plot_array["longitude"],
                    "latitude": plot_array["latitude"],
                    "station": plot_array["station"],
                }
            )
            station_renderer = fig_obj.scatter(
                x="longitude",
                y="latitude",
                source=station_source,
                marker="circle",
                size=self.marker_size,
                color=marker_color,
                line_color=marker_color,
            )
            fig_obj.add_tools(
                HoverTool(
                    renderers=[station_renderer],
                    tooltips=[
                        ("Station", "@station"),
                        ("Longitude", "@longitude{0.000}"),
                        ("Latitude", "@latitude{0.000}"),
                    ],
                )
            )
            renderers.append(station_renderer)

        self.renderers[comp] = renderers

    def plot(self, show=True):
        """Plot apparent resistivity and phase maps for the active period."""

        self._require_bokeh()

        data_array = self._get_data_array()
        if data_array.size == 0:
            self.layout = None
            self.fig = None
            return None

        subplot_numbers = self._get_n_subplots()

        valid_lon = data_array["longitude"][np.isfinite(data_array["longitude"])]
        valid_lat = data_array["latitude"][np.isfinite(data_array["latitude"])]
        if valid_lon.size == 0 or valid_lat.size == 0:
            self.layout = None
            self.fig = None
            return None

        x_range = Range1d(
            start=float(np.nanmin(valid_lon)), end=float(np.nanmax(valid_lon))
        )
        y_range = Range1d(
            start=float(np.nanmin(valid_lat)), end=float(np.nanmax(valid_lat))
        )

        x_label, y_label = self._axis_labels()

        self.figures = {}
        self.renderers = {}

        rows = self._get_n_rows()
        cols = self._get_n_columns()

        for comp, subplot in subplot_numbers.items():
            if subplot is None:
                continue
                continue

            nr, nc, nindex = subplot
            row_idx = int((nindex - 1) // nc)
            col_idx = int((nindex - 1) % nc)

            fig_obj = figure(
                title=self.label_dict[comp],
                width=440,
                height=360,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom",
                x_range=x_range,
                y_range=y_range,
                x_axis_label=(
                    x_label if (nr == 1 or (nr == 2 and row_idx == rows - 1)) else ""
                ),
                y_axis_label=y_label if col_idx == 0 else "",
                match_aspect=True,
            )

            fig_obj.grid.grid_line_alpha = 0.25
            self._plot_component(fig_obj, data_array, comp)
            self.figures[comp] = fig_obj

        if len(self.figures) == 0:
            self.layout = None
            self.fig = None
            return None

        grid_rows = []
        row_items = []
        current_row = 0
        current_items = [None] * cols
        for comp, subplot in subplot_numbers.items():
            if subplot is None or comp not in self.figures:
                continue
            _nr, nc, nindex = subplot
            row_idx = int((nindex - 1) // nc)
            col_idx = int((nindex - 1) % nc)
            if row_idx != current_row:
                grid_rows.append(current_items)
                current_items = [None] * cols
                current_row = row_idx
            current_items[col_idx] = self.figures[comp]
        grid_rows.append(current_items)

        title = Div(text=f"<b>Plot Period: {self.plot_period:.5g} s</b>")
        self.layout = Column(title, gridplot(grid_rows))
        self.fig = self.layout

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def panel(self):
        """Return an interactive, embeddable Panel app for map controls."""

        try:
            pn = importlib.import_module("panel")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "panel is required for PlotResPhaseMaps.panel(). "
                "Install with `pip install panel`."
            ) from exc

        palette_options = [
            "turbo",
            "viridis",
            "magma",
            "inferno",
            "plasma",
            "cividis",
            "rainbow",
            "rainbow_r",
        ]

        active_components = [
            comp
            for comp in ["xx", "xy", "yx", "yy", "det"]
            if getattr(self, f"plot_{comp}")
        ]
        active_rows = []
        if self.plot_resistivity:
            active_rows.append("Resistivity")
        if self.plot_phase:
            active_rows.append("Phase")

        w_period = pn.widgets.NumberInput(
            name="Plot Period (s)",
            value=float(self.plot_period),
            step=0.1,
            width=160,
        )
        w_components = pn.widgets.CheckButtonGroup(
            name="Components",
            options=["xx", "xy", "yx", "yy", "det"],
            value=active_components,
        )
        w_rows = pn.widgets.CheckBoxGroup(
            name="Rows",
            options=["Resistivity", "Phase"],
            value=active_rows,
            inline=True,
        )
        w_res_palette = pn.widgets.Select(
            name="Resistivity Palette",
            value=(self.res_cmap if self.res_cmap in palette_options else "turbo"),
            options=palette_options,
            width=170,
        )
        w_phase_palette = pn.widgets.Select(
            name="Phase Palette",
            value=(self.phase_cmap if self.phase_cmap in palette_options else "turbo"),
            options=palette_options,
            width=170,
        )
        w_plot_stations = pn.widgets.Checkbox(
            name="Plot Stations", value=bool(self.plot_stations)
        )
        w_map_units = pn.widgets.Select(
            name="Map Units",
            value=self.map_units,
            options=["deg", "m", "km"],
            width=120,
        )
        refresh_btn = pn.widgets.Button(
            name="Refresh",
            button_type="success",
            width=120,
        )

        status = pn.pane.Markdown(
            "_Adjust controls and click **Refresh** to update the map._",
            styles={"color": "#555"},
        )
        plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

        def _refresh(_event=None):
            self.plot_period = float(w_period.value)
            self.plot_resistivity = "Resistivity" in list(w_rows.value)
            self.plot_phase = "Phase" in list(w_rows.value)
            selected_components = set(w_components.value)
            for comp in ["xx", "xy", "yx", "yy", "det"]:
                setattr(self, f"plot_{comp}", comp in selected_components)

            self.res_cmap = str(w_res_palette.value)
            self.phase_cmap = str(w_phase_palette.value)
            self.plot_stations = bool(w_plot_stations.value)
            self.map_units = str(w_map_units.value)

            layout = self.plot(show=False)
            plot_pane.object = layout

            if layout is None:
                status.object = "⚠️ No mappable data for the current control selection."
                status.styles = {"color": "#7a5200"}
            else:
                status.object = "✅ Map rendered."
                status.styles = {"color": "#1a6600"}

        refresh_btn.on_click(_refresh)
        _refresh()

        controls = pn.Row(
            pn.Column(
                w_period,
                w_map_units,
                w_rows,
                w_components,
                width=320,
                margin=(0, 20, 0, 0),
            ),
            pn.Column(
                w_res_palette,
                w_phase_palette,
                w_plot_stations,
                refresh_btn,
                width=260,
            ),
            sizing_mode="fixed",
        )

        return pn.Column(controls, status, plot_pane, sizing_mode="stretch_width")

    def servable(self, title: str | None = None):
        """Return a standalone servable Panel view of this plot app."""

        app = self.panel()
        if title is None:
            return app.servable()
        return app.servable(title=title)
