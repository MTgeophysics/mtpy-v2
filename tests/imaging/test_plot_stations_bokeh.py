# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotStations behavior and BokehPlotBase."""

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


class TestBokehPlotBase:
    """Tests for the BokehPlotBase param.Parameterized base class."""

    def test_is_param_parameterized(self):
        import param

        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        assert issubclass(BokehPlotBase, param.Parameterized)

    def test_exported_from_package(self):
        from mtpy.imaging.bokeh_plots import BokehPlotBase

        assert BokehPlotBase is not None

    def test_show_plot_default_false(self):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase()
        assert obj.show_plot is False

    def test_marker_default(self):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase()
        assert obj.marker == "o"

    def test_marker_color_default_hex(self):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase()
        assert obj.marker_color == "#0000ff"

    def test_marker_size_bounds(self):
        pass

        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase(marker_size=20)
        assert obj.marker_size == 20

    def test_x_limits_default_none(self):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase()
        assert obj.x_limits is None

    def test_kwargs_set_params(self):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        obj = BokehPlotBase(plot_title="My Plot", marker_size=15)
        assert obj.plot_title == "My Plot"
        assert obj.marker_size == 15

    def test_plot_stations_is_subclass(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotStations
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        assert issubclass(PlotStations, BokehPlotBase)

    def test_plot_stations_not_inheriting_plot_base(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotStations
        from mtpy.imaging.mtplot_tools import PlotBase

        assert not issubclass(PlotStations, PlotBase)


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


class TestPlotStationsPanel:
    """Tests for PlotStations.panel() interactive app."""

    @pytest.fixture(autouse=True)
    def require_panel(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")

    def test_panel_returns_column(self, bokeh_plot_stations_class, station_gdf):
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        assert isinstance(result, pn.Column)

    def test_panel_has_row_of_controls(self, bokeh_plot_stations_class, station_gdf):
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        # First child should be a Row containing the two control columns
        assert len(result) >= 2
        controls = result[0]
        assert isinstance(controls, pn.Row)

    def test_panel_contains_title_widget(self, bokeh_plot_stations_class, station_gdf):
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        def _find_widgets(obj, widget_type):
            found = []
            if isinstance(obj, widget_type):
                found.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    found.extend(_find_widgets(child, widget_type))
            return found

        text_inputs = _find_widgets(result, pn.widgets.TextInput)
        assert len(text_inputs) >= 1
        assert any(w.name == "Title" for w in text_inputs)

    def test_panel_contains_marker_color_picker(
        self, bokeh_plot_stations_class, station_gdf
    ):
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        def _find_widgets(obj, widget_type):
            found = []
            if isinstance(obj, widget_type):
                found.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    found.extend(_find_widgets(child, widget_type))
            return found

        color_pickers = _find_widgets(result, pn.widgets.ColorPicker)
        assert len(color_pickers) >= 1

    def test_panel_contains_refresh_button(
        self, bokeh_plot_stations_class, station_gdf
    ):
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        def _find_widgets(obj, widget_type):
            found = []
            if isinstance(obj, widget_type):
                found.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    found.extend(_find_widgets(child, widget_type))
            return found

        buttons = _find_widgets(result, pn.widgets.Button)
        assert any("Refresh" in (b.name or "") for b in buttons)

    def test_panel_renders_initial_plot(self, bokeh_plot_stations_class, station_gdf):
        """panel() should render a plot immediately (pane.object is set)."""
        import panel as pn

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        result = plotter.panel()

        # Last object should be the plot pane with a figure already set
        plot_pane = result[-1]
        assert isinstance(plot_pane, pn.pane.Bokeh)
        assert plot_pane.object is not None

    def test_panel_plot_updates_on_title_change(
        self, bokeh_plot_stations_class, station_gdf
    ):
        """After panel() is returned, changing title param should be reflected
        on next explicit refresh.  Here we test by directly calling plot()."""

        plotter = bokeh_plot_stations_class(station_gdf, show_plot=False, plot_cx=False)
        plotter.panel()  # renders initial plot

        plotter.plot_title = "Updated Title"
        fig = plotter.plot(show=False)
        assert fig.title.text == "Updated Title"
