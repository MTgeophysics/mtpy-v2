# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPenetrationDepthMap behavior."""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_penetration_depth_map_class():
    """Import Bokeh PlotPenetrationDepthMap or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPenetrationDepthMap

    return PlotPenetrationDepthMap


@pytest.fixture
def depth_array():
    arr = np.zeros(
        3,
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
    arr["station"] = ["S1", "S2", "S3"]
    arr["latitude"] = [40.0, 40.1, 40.2]
    arr["longitude"] = [-120.2, -120.1, -120.0]
    arr["det"] = [1.0, 1.5, 2.0]
    arr["xy"] = [0.9, 1.4, 1.9]
    arr["yx"] = [1.1, 1.6, 2.1]
    return arr


class TestPlotPenetrationDepthMapBokeh:
    def test_depth_units_maps_to_scale(
        self, bokeh_plot_penetration_depth_map_class, depth_array
    ):
        plotter = bokeh_plot_penetration_depth_map_class({}, show_plot=False)

        assert plotter.depth_units == "km"
        assert plotter.depth_scale == 1.0 / 1000

        plotter.depth_units = "m"
        assert plotter.depth_scale == 1

    def test_depth_units_rejects_invalid_values(
        self, bokeh_plot_penetration_depth_map_class
    ):
        plotter = bokeh_plot_penetration_depth_map_class({}, show_plot=False)

        with pytest.raises(ValueError, match="depth_units must be either 'km' or 'm'"):
            plotter.depth_units = "cm"

    def test_plot_reuses_cached_depth_array_for_same_period(
        self, bokeh_plot_penetration_depth_map_class, depth_array, monkeypatch
    ):
        plotter = bokeh_plot_penetration_depth_map_class({}, show_plot=False)
        plotter.plot_det = True
        plotter.plot_te = False
        plotter.plot_tm = False

        call_count = {"n": 0}

        def _fake_get_depth_array():
            call_count["n"] += 1
            return depth_array

        def _fake_filter_depth_array(arr, _comp):
            return arr

        def _fake_interpolate_to_map(_arr, _comp):
            x = np.array([[-120.2, -120.0], [-120.2, -120.0]])
            y = np.array([[40.0, 40.0], [40.2, 40.2]])
            image = np.array([[1.0, 1.1], [1.2, 1.3]])
            return x, y, image

        monkeypatch.setattr(plotter, "_get_depth_array", _fake_get_depth_array)
        monkeypatch.setattr(plotter, "_filter_depth_array", _fake_filter_depth_array)
        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)

        plotter.plot_period = 10.0
        layout_1 = plotter.plot()
        layout_2 = plotter.plot()

        assert layout_1 is not None
        assert layout_2 is not None
        assert call_count["n"] == 1

    def test_plot_builds_layout_with_component_figures(
        self, bokeh_plot_penetration_depth_map_class, depth_array, monkeypatch
    ):
        plotter = bokeh_plot_penetration_depth_map_class({}, show_plot=False)

        def _fake_get_depth_array():
            return depth_array

        def _fake_filter_depth_array(arr, _comp):
            return arr

        def _fake_interpolate_to_map(_arr, _comp):
            x = np.array([[-120.2, -120.0], [-120.2, -120.0]])
            y = np.array([[40.0, 40.0], [40.2, 40.2]])
            image = np.array([[1.0, 1.1], [1.2, 1.3]])
            return x, y, image

        monkeypatch.setattr(plotter, "_get_depth_array", _fake_get_depth_array)
        monkeypatch.setattr(plotter, "_filter_depth_array", _fake_filter_depth_array)
        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)

        layout = plotter.plot()

        assert layout is not None
        assert len(plotter.figures) == 3
        assert set(plotter.figures.keys()) == {"det", "xy", "yx"}
        assert all(comp in plotter.renderers for comp in ["det", "xy", "yx"])

        det_fig = plotter.figures["det"]
        assert det_fig.xaxis.axis_label == "Longitude (deg)"
        assert det_fig.yaxis.axis_label == "Latitude (deg)"
        assert det_fig.title.text == "Determinant"

        image_renderers = [
            r
            for r in det_fig.renderers
            if getattr(getattr(r, "glyph", None), "__class__", None)
            and r.glyph.__class__.__name__ == "Image"
        ]
        assert len(image_renderers) == 1
