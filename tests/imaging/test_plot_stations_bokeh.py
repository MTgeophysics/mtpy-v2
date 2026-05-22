# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotStations behavior."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

import mtpy.utils.exceptions as mtex


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_stations_class():
    """Import Bokeh PlotStations or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotStations

    return PlotStations


@pytest.fixture
def station_gdf():
    """Build a simple geographic GeoDataFrame for station plotting."""

    return gpd.GeoDataFrame(
        {"station": ["STA01", "STA02", "STA03"]},
        geometry=[
            Point(-120.20, 40.00),
            Point(-120.10, 40.05),
            Point(-120.00, 40.10),
        ],
        crs="EPSG:4326",
    )


class TestPlotStationsBokeh:
    def test_requires_image_extent_when_image_file_set(
        self, bokeh_plot_stations_class, station_gdf
    ):
        with pytest.raises(mtex.MTpyError_input_arguments):
            bokeh_plot_stations_class(
                station_gdf,
                show_plot=False,
                image_file="dummy.png",
            )

    def test_plot_with_basemap_uses_mercator_coords(
        self, bokeh_plot_stations_class, station_gdf
    ):
        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=True)

        fig = plotter.plot(show=False)

        assert fig is not None
        assert plotter.fig is fig
        assert fig.xaxis[0].axis_label == "longitude (deg)"
        assert fig.yaxis[0].axis_label == "latitude (deg)"

        # Converted to Web Mercator for tile plotting, values are large magnitude.
        assert abs(fig.x_range.start) > 1_000_000
        assert abs(fig.x_range.end) > 1_000_000

    def test_plot_without_basemap_keeps_native_coordinates(
        self, bokeh_plot_stations_class, station_gdf
    ):
        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)

        fig = plotter.plot(show=False)

        assert fig is not None
        assert fig.xaxis[0].axis_label == "relative east (m)"
        assert fig.yaxis[0].axis_label == "relative north (m)"

        assert fig.x_range.start < -119
        assert fig.x_range.end < -119

    def test_plot_names_toggle(self, bokeh_plot_stations_class, station_gdf):
        plotter = bokeh_plot_stations_class(
            station_gdf,
            show_plot=False,
            plot_cx=False,
            plot_names=False,
        )

        fig = plotter.plot(show=False)

        label_sets = [r for r in fig.center if r.__class__.__name__ == "LabelSet"]
        assert len(label_sets) == 0

    def test_plot_with_image_extent(self, bokeh_plot_stations_class, station_gdf):
        plotter = bokeh_plot_stations_class(
            station_gdf,
            show_plot=False,
            plot_cx=False,
            image_file=str(Path(__file__).resolve()),
            image_extent=(-121.0, 39.0, -119.0, 41.0),
        )

        fig = plotter.plot(show=False)

        assert fig is not None
        assert fig.x_range.start == pytest.approx(-121.0)
        assert fig.x_range.end == pytest.approx(-119.0)

    def test_top_level_alias_import(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging import PlotStationsBokeh

        assert PlotStationsBokeh is not None
