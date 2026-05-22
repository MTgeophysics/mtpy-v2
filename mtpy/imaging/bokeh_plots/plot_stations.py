# -*- coding: utf-8 -*-
"""Bokeh implementation of station location plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mtpy.utils.exceptions as mtex
from mtpy.imaging.mtplot_tools import PlotBase


try:
    from bokeh.io import show as bokeh_show
    from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Range1d
    from bokeh.plotting import figure
except ImportError:  # pragma: no cover - optional dependency
    bokeh_show = None
    ColumnDataSource = None
    HoverTool = None
    LabelSet = None
    Range1d = None
    figure = None


class PlotStations(PlotBase):
    """Plot station locations in map view using Bokeh."""

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

    def __init__(self, geo_df, **kwargs):
        self.plot_title = None
        self.station_id = None
        self.ref_point = (0, 0)

        self.map_epsg = 4326
        self.plot_names = True
        self.plot_cx = True

        self.image_file = None
        self.image_extent = None
        self.pad = None

        self.bokeh_tile_provider = "CartoDB Positron"

        super().__init__(**kwargs)

        self._basename = "stations_map"
        self.gdf = geo_df

        self.fig = None
        self.layout = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.image_file is not None and self.image_extent is None:
            raise mtex.MTpyError_input_arguments(
                "Need to input extents of the image as(x0, y0, x1, y1)"
            )

        if self.show_plot:
            self.plot(show=True)

    def _require_bokeh(self):
        if (
            figure is None
            or ColumnDataSource is None
            or HoverTool is None
            or LabelSet is None
            or Range1d is None
        ):
            raise ImportError(
                "Bokeh is required for PlotStations. Install with `pip install bokeh`."
            )

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

    def _get_plot_gdf(self):
        plot_gdf = self.gdf
        if not self.plot_cx:
            return plot_gdf

        if getattr(plot_gdf, "crs", None) is None:
            if self.map_epsg is not None and hasattr(plot_gdf, "set_crs"):
                plot_gdf = plot_gdf.set_crs(
                    epsg=int(self.map_epsg), allow_override=True
                )
            else:
                raise ValueError(
                    "GeoDataFrame CRS is required for basemap plotting; set gdf.crs or map_epsg."
                )

        if hasattr(plot_gdf, "to_crs"):
            plot_gdf = plot_gdf.to_crs(epsg=3857)

        return plot_gdf

    def _get_pad(self, plot_gdf):
        return max(
            [
                np.abs(plot_gdf.geometry.x.min() - plot_gdf.geometry.x.max()) * 0.05,
                np.abs(plot_gdf.geometry.y.min() - plot_gdf.geometry.y.max()) * 0.05,
            ]
        )

    def _get_xlimits(self, x_values):
        return (x_values.min() - self.pad, x_values.max() + self.pad)

    def _get_ylimits(self, y_values):
        return (y_values.min() - self.pad, y_values.max() + self.pad)

    def _get_limits(self, plot_gdf):
        if self.image_extent:
            x_limits = (self.image_extent[0], self.image_extent[2])
            y_limits = (self.image_extent[1], self.image_extent[3])
        elif self.x_limits is not None:
            x_limits = self.x_limits
            if self.y_limits is None:
                y_limits = self._get_ylimits(plot_gdf.geometry.y)
            else:
                y_limits = self.y_limits
        elif self.y_limits is not None:
            y_limits = self.y_limits
            if self.x_limits is None:
                x_limits = self._get_xlimits(plot_gdf.geometry.x)
            else:
                x_limits = self.x_limits
        else:
            x_limits = self._get_xlimits(plot_gdf.geometry.x)
            y_limits = self._get_ylimits(plot_gdf.geometry.y)

        return x_limits, y_limits

    def _make_figure(self, x_limits, y_limits):
        kwargs = {
            "width": 900,
            "height": 700,
            "tools": "pan,wheel_zoom,box_zoom,reset,save",
            "active_scroll": "wheel_zoom",
            "x_range": Range1d(start=float(x_limits[0]), end=float(x_limits[1])),
            "y_range": Range1d(start=float(y_limits[0]), end=float(y_limits[1])),
            "match_aspect": True,
        }
        if self.plot_cx:
            kwargs["x_axis_type"] = "mercator"
            kwargs["y_axis_type"] = "mercator"

        return figure(**kwargs)

    def plot(self, show=True):
        """Plot stations using Bokeh, optionally with a built-in tile basemap."""

        self._require_bokeh()

        plot_gdf = self._get_plot_gdf()

        if self.pad is None:
            self.pad = self._get_pad(plot_gdf)

        x_limits, y_limits = self._get_limits(plot_gdf)

        fig_obj = self._make_figure(x_limits, y_limits)

        if self.plot_title is not None:
            fig_obj.title.text = self.plot_title

        if self.plot_cx:
            fig_obj.add_tile(self.bokeh_tile_provider)

        if self.image_file is not None:
            image_uri = Path(self.image_file).resolve().as_uri()
            fig_obj.image_url(
                url=[image_uri],
                x=float(self.image_extent[0]),
                y=float(self.image_extent[1]),
                w=float(self.image_extent[2] - self.image_extent[0]),
                h=float(self.image_extent[3] - self.image_extent[1]),
                anchor="bottom_left",
            )

        x = np.asarray(plot_gdf.geometry.x, dtype=float)
        y = np.asarray(plot_gdf.geometry.y, dtype=float)
        station = np.asarray(plot_gdf.station, dtype=str)

        source = ColumnDataSource(data={"x": x, "y": y, "station": station})
        marker_color = self._to_hex(self.marker_color)

        station_renderer = fig_obj.scatter(
            x="x",
            y="y",
            source=source,
            marker=self._marker_name(self.marker),
            size=self.marker_size,
            color=marker_color,
            line_color=marker_color,
        )

        fig_obj.add_tools(
            HoverTool(
                renderers=[station_renderer],
                tooltips=[
                    ("Station", "@station"),
                    ("x", "@x{0.000}"),
                    ("y", "@y{0.000}"),
                ],
            )
        )

        if self.plot_names:
            labels = LabelSet(
                x="x",
                y="y",
                text="station",
                source=source,
                text_align="center",
                x_offset=0,
                y_offset=int(self.text_y_pad),
                text_color=self._to_hex(self.text_color),
                text_font_size=f"{int(self.text_size)}pt",
            )
            fig_obj.add_layout(labels)

        if self.plot_cx:
            fig_obj.xaxis.axis_label = "longitude (deg)"
            fig_obj.yaxis.axis_label = "latitude (deg)"
        else:
            fig_obj.xaxis.axis_label = "relative east (m)"
            fig_obj.yaxis.axis_label = "relative north (m)"

        fig_obj.grid.grid_line_alpha = 0.35

        self.fig = fig_obj
        self.layout = fig_obj

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(fig_obj)

        return fig_obj
