"""Bokeh implementation of depth-of-investigation map plotting."""

from __future__ import annotations

import numpy as np

from mtpy.core import Z
from mtpy.imaging.bokeh_plots.bokeh_plot_base_maps import BokehPlotBaseMaps

try:
    from bokeh.io import show
    from bokeh.layouts import Column, Row
    from bokeh.models import (
        BasicTicker,
        ColorBar,
        ColumnDataSource,
        Div,
        HoverTool,
        LinearColorMapper,
        Range1d,
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
    show = None
    Column = None
    Row = None
    BasicTicker = None
    ColorBar = None
    ColumnDataSource = None
    Div = None
    HoverTool = None
    LinearColorMapper = None
    Range1d = None
    Cividis256 = None
    Inferno256 = None
    Magma256 = None
    Plasma256 = None
    Turbo256 = None
    Viridis256 = None
    figure = None


class PlotPenetrationDepthMap(BokehPlotBaseMaps):
    """Plot the depth of penetration based on the Niblett-Bostick approximation."""

    def __init__(self, mt_data, **kwargs):
        self.mt_data = mt_data
        self._depth_array_cache = None
        self._interpolated_mt_data_cache = None
        self._interpolated_mt_data_cache_period = None

        self.fig = None
        self.layout = None
        self.figures = {}
        self.renderers = {}

        super().__init__(**kwargs)

        self.depth_units = "km"
        self.plot_period = 1
        self._old_plot_period = None
        self.plot_det = True
        self.plot_te = True
        self.plot_tm = True
        self.plot_stations = True
        self.depth_cmap = "magma"
        self.marker_color = "#000000"
        self.marker_size = 10
        self.subplot_title_dict = {
            "det": "Determinant",
            "xy": "TE Mode",
            "yx": "TM Mode",
        }
        self.depth_range = [None, None]
        self.depth_tolerance = 1

        self.subplot_wspace = 0.2
        self.subplot_hspace = 0.1
        self.font_size = 8

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    @property
    def depth_units(self):
        """Depth units."""

        return self._depth_units

    @depth_units.setter
    def depth_units(self, value):
        """Depth units."""

        self._depth_units = value
        if value in ["km"]:
            self.depth_scale = 1.0 / 1000
        elif value in ["m"]:
            self.depth_scale = 1
        else:
            raise ValueError("depth_units must be either 'km' or 'm'")

    def _require_bokeh(self):
        if (
            Column is None
            or Row is None
            or Div is None
            or figure is None
            or ColumnDataSource is None
            or LinearColorMapper is None
            or ColorBar is None
            or BasicTicker is None
            or Range1d is None
        ):
            raise ImportError(
                "Bokeh is required for PlotPenetrationDepthMap. Install with `pip install bokeh`."
            )

    def _get_nb_estimation(self, z_object):
        """Get the depth of investigation estimation."""

        return z_object.estimate_depth_of_investigation()

    def _filter_depth_array(self, depth_array, comp):
        """Filter out some bad data points."""

        depth_array = depth_array[np.nonzero(depth_array)]
        d_median = np.median(depth_array[comp])
        d_min = d_median - (depth_array[comp].std() * self.depth_tolerance)
        d_max = d_median + (depth_array[comp].std() * self.depth_tolerance)
        good_index = np.where(
            (depth_array[comp] >= d_min) & (depth_array[comp] <= d_max)
        )

        return depth_array[good_index]

    def _get_depth_array(self):
        """Get a depth array with xyz values."""

        mt_objects = self._get_mt_objects()

        depth_array = np.zeros(
            len(mt_objects),
            dtype=[
                ("station", "U20"),
                ("latitude", float),
                ("longitude", float),
                ("elevation", float),
                ("det", float),
                ("xy", float),
                ("yx", float),
            ],
        )

        for ii, tf in enumerate(mt_objects):
            z = self._get_interpolated_z(tf)
            z_object = Z(
                z=z,
                frequency=[1.0 / self.plot_period],
                units=tf.impedance_units,
            )
            if (np.nan_to_num(z_object.z) == 0).all():
                continue
            d = self._get_nb_estimation(z_object)

            depth_array["station"][ii] = tf.station
            depth_array["latitude"][ii] = tf.latitude
            depth_array["longitude"][ii] = tf.longitude
            elev = 0
            if tf.elevation is not None:
                depth_array["elevation"][ii] = tf.elevation * self.depth_scale
                elev = tf.elevation * self.depth_scale
            depth_array["det"][ii] = (d["depth_det"][0] - elev) * self.depth_scale
            depth_array["xy"][ii] = (d["depth_xy"][0] - elev) * self.depth_scale
            depth_array["yx"][ii] = (d["depth_yx"][0] - elev) * self.depth_scale

        return depth_array

    def _get_n_subplots(self):
        """Get the number of subplots."""

        n = 0
        if self.plot_det:
            n += 1
        if self.plot_te:
            n += 1
        if self.plot_tm:
            n += 1

        return n

    def _get_plot_component_dict(self):
        """Get all the components to plot."""

        components = {}
        if self.plot_det:
            components["det"] = True
        if self.plot_te:
            components["xy"] = True
        if self.plot_tm:
            components["yx"] = True

        if len(components) == 0:
            raise ValueError(
                "must set at least one of the following to True  'plot_det', 'plot_te', 'plot_tm'"
            )

        return components

    def _get_palette(self):
        palette_map = {
            "magma": Magma256,
            "inferno": Inferno256,
            "plasma": Plasma256,
            "viridis": Viridis256,
            "cividis": Cividis256,
            "turbo": Turbo256,
        }
        palette = palette_map.get(str(self.depth_cmap).lower(), Magma256)
        return palette if palette is not None else Magma256

    def _image_from_map_output(self, output):
        if hasattr(output[0], "triangles"):
            triangulation, image, _inside_indices = output
            x_values = np.unique(np.asarray(triangulation.x, dtype=float))
            y_values = np.unique(np.asarray(triangulation.y, dtype=float))
            image = np.asarray(image, dtype=float).reshape(len(y_values), len(x_values))
            return x_values, y_values, image

        plot_x, plot_y, image = output
        return (
            np.asarray(plot_x, dtype=float),
            np.asarray(plot_y, dtype=float),
            np.asarray(image, dtype=float),
        )

    def _component_range(self, image):
        if self.depth_range[0] is not None and self.depth_range[1] is not None:
            return float(self.depth_range[0]), float(self.depth_range[1])

        finite = np.asarray(image, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _make_figure(self, x_range, y_range, title):
        return figure(
            title=title,
            width=520,
            height=480,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_axis_label="Longitude (deg)",
            y_axis_label="Latitude (deg)",
            x_range=x_range,
            y_range=y_range,
            match_aspect=True,
        )

    def _plot_component(self, fig, plot_depth_array, comp):
        map_output = self.interpolate_to_map(plot_depth_array, comp)
        plot_x, plot_y, image = self._image_from_map_output(map_output)

        palette = self._get_palette()
        low, high = self._component_range(image)
        mapper = LinearColorMapper(palette=palette, low=low, high=high)

        if plot_x.ndim == 1 and plot_y.ndim == 1:
            x0 = float(np.min(plot_x))
            x1 = float(np.max(plot_x))
            y0 = float(np.min(plot_y))
            y1 = float(np.max(plot_y))
            x_coords = plot_x
            y_coords = plot_y
        else:
            x0 = float(np.nanmin(plot_x))
            x1 = float(np.nanmax(plot_x))
            y0 = float(np.nanmin(plot_y))
            y1 = float(np.nanmax(plot_y))
            x_coords = np.unique(plot_x)
            y_coords = np.unique(plot_y)

        image_renderer = fig.image(
            image=[image],
            x=x0,
            y=y0,
            dw=x1 - x0,
            dh=y1 - y0,
            color_mapper=mapper,
        )

        fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title=f"Penetration Depth ({self.depth_units})",
            ),
            "right",
        )

        renderers = [image_renderer]
        if self.plot_stations:
            station_source = ColumnDataSource(
                data={
                    "longitude": plot_depth_array["longitude"],
                    "latitude": plot_depth_array["latitude"],
                    "station": plot_depth_array["station"],
                }
            )
            station_renderer = fig.scatter(
                x="longitude",
                y="latitude",
                source=station_source,
                marker="circle",
                size=self.marker_size,
                color=self.marker_color,
                line_color=self.marker_color,
            )
            fig.add_tools(
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
        return fig

    def plot(self):
        """Plot the depth of investigation as a Bokeh map layout."""

        self._require_bokeh()

        if self._old_plot_period != self.plot_period:
            self._depth_array_cache = self._get_depth_array()
            self._old_plot_period = self.plot_period

        depth_array = self._depth_array_cache
        if depth_array is None or depth_array.size == 0:
            self.logger.warning(f"No stations have data for period {self.plot_period} ")
            return None

        if np.count_nonzero(depth_array["det"]) == 0:
            self.logger.warning(f"No stations have data for period {self.plot_period} ")
            return None

        plot_components = self._get_plot_component_dict()
        self.figures = {}
        self.renderers = {}

        x_min = float(np.nanmin(depth_array["longitude"]))
        x_max = float(np.nanmax(depth_array["longitude"]))
        y_min = float(np.nanmin(depth_array["latitude"]))
        y_max = float(np.nanmax(depth_array["latitude"]))
        x_range = Range1d(start=x_min, end=x_max)
        y_range = Range1d(start=y_min, end=y_max)

        for comp in ["det", "xy", "yx"]:
            if comp not in plot_components:
                continue
            plot_depth_array = self._filter_depth_array(depth_array, comp)
            if plot_depth_array.size == 0:
                self.logger.warning(
                    f"No stations have data for period {self.plot_period} "
                )
                continue

            fig = self._make_figure(x_range, y_range, self.subplot_title_dict[comp])
            self._plot_component(fig, plot_depth_array, comp)
            self.figures[comp] = fig

        if len(self.figures) == 0:
            self.layout = None
            return None

        title = Div(
            text=f"<b>Depth of investigation for period {self.plot_period:5g} (s)</b>"
        )
        if len(self.figures) == 1:
            body = next(iter(self.figures.values()))
        else:
            body = Row(*self.figures.values())
        self.layout = Column(title, body)

        if self.show_plot and show is not None:
            show(self.layout)

        return self.layout
