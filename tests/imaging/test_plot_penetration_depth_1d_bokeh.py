# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPenetrationDepth1D behavior."""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_penetration_depth_class():
    """Import the Bokeh penetration-depth plotter or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

    return PlotPenetrationDepth1D


class _FakeZ:
    def __init__(self, depth_array):
        self._depth_array = depth_array

    def estimate_depth_of_investigation(self):
        return self._depth_array


class _FakeTF:
    def __init__(self, station, depth_array):
        self.station = station
        self.Z = _FakeZ(depth_array)


@pytest.fixture
def depth_array_unsorted():
    """Create a small unsorted depth array for regression checks."""

    return {
        "period": np.array([10.0, 1.0, 100.0]),
        "depth_min": np.array([800.0, 80.0, 8000.0]),
        "depth_max": np.array([1200.0, 120.0, 12000.0]),
        "depth_xy": np.array([900.0, 90.0, 9000.0]),
        "depth_yx": np.array([1000.0, 100.0, 10000.0]),
        "depth_det": np.array([1100.0, 110.0, 11000.0]),
    }


@pytest.fixture
def fake_tf(depth_array_unsorted):
    return _FakeTF("TEST01", depth_array_unsorted)


class TestPlotPenetrationDepth1DBokeh:
    def test_depth_units_maps_to_scale(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)

        assert plotter.depth_units == "km"
        assert plotter.depth_scale == 1.0 / 1000

        plotter.depth_units = "m"
        assert plotter.depth_scale == 1

    def test_depth_units_rejects_invalid_values(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)

        with pytest.raises(ValueError, match="depth_units must be either 'km' or 'm'"):
            plotter.depth_units = "cm"

    def test_plot_builds_bokeh_figure_with_expected_elements(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        layout = plotter.plot()

        assert layout is plotter.fig
        assert plotter.fig.title.text == "Depth of investigation for TEST01"
        assert plotter.fig.xaxis.axis_label == "Depth (km)"
        assert plotter.fig.yaxis.axis_label == "Period (s)"
        assert plotter.fig.y_range.start > plotter.fig.y_range.end
        assert len(plotter.fig.legend) == 1
        assert len(plotter.fig.legend[0].items) == 3

        patch_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Patch"
        ]
        line_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Line"
        ]

        assert len(patch_renderers) == 1
        assert len(line_renderers) == 3
        assert np.isclose(plotter.fig.x_range.start, 0.01)
        assert np.isclose(plotter.fig.x_range.end, 100.0)

    def test_plot_sorts_periods_before_rendering(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        plotter.plot()

        line_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Line"
        ]
        periods = line_renderers[0].data_source.data["period"]

        assert list(periods) == sorted(periods)
