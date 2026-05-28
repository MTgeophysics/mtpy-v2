# -*- coding: utf-8 -*-
"""Tests for BokehPlotBaseMaps behavior."""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_base_maps_class():
    """Import BokehPlotBaseMaps or skip if param is unavailable."""

    pytest.importorskip("param")
    from mtpy.imaging.bokeh_plots import BokehPlotBaseMaps

    return BokehPlotBaseMaps


def test_class_hierarchy_and_default_parameters(bokeh_plot_base_maps_class):
    from mtpy.imaging.bokeh_plots import BokehPlotBase

    assert issubclass(bokeh_plot_base_maps_class, BokehPlotBase)

    obj = bokeh_plot_base_maps_class()
    assert obj.cell_size == pytest.approx(0.002)
    assert obj.n_padding_cells == 10
    assert obj.interpolation_method == "delaunay"
    assert obj.interpolation_power == pytest.approx(5)
    assert obj.nearest_neighbors == 7


def test_palette_options_include_reverse_variants(bokeh_plot_base_maps_class):
    obj = bokeh_plot_base_maps_class()

    assert "magma" in obj.palette_options
    assert "magma_r" in obj.palette_options
    assert "rainbow" in obj.palette_options
    assert "rainbow_r" in obj.palette_options
    assert obj.palette_options["magma_r"] == list(
        reversed(obj.palette_options["magma"])
    )
    assert obj.palette_options["rainbow_r"] == list(
        reversed(obj.palette_options["rainbow"])
    )


def test_interpolate_to_map_uses_param_settings(
    bokeh_plot_base_maps_class, monkeypatch
):
    from mtpy.imaging.bokeh_plots import bokeh_plot_base_maps

    captured = {}

    def _fake_interpolate_to_map(
        plot_array,
        component,
        cell_size,
        n_padding_cells,
        interpolation_method,
        interpolation_power,
        nearest_neighbors,
    ):
        captured["plot_array"] = plot_array
        captured["component"] = component
        captured["cell_size"] = cell_size
        captured["n_padding_cells"] = n_padding_cells
        captured["interpolation_method"] = interpolation_method
        captured["interpolation_power"] = interpolation_power
        captured["nearest_neighbors"] = nearest_neighbors
        return "xgrid", "ygrid", "image"

    monkeypatch.setattr(
        bokeh_plot_base_maps, "interpolate_to_map", _fake_interpolate_to_map
    )

    obj = bokeh_plot_base_maps_class(
        cell_size=0.01,
        n_padding_cells=4,
        interpolation_method="linear",
        interpolation_power=2,
        nearest_neighbors=3,
    )

    plot_array = np.array(
        [(1.0, 2.0, 10.0)],
        dtype=[("longitude", float), ("latitude", float), ("res_xy", float)],
    )

    result = obj.interpolate_to_map(plot_array, "res_xy")

    assert result == ("xgrid", "ygrid", "image")
    assert captured["plot_array"] is plot_array
    assert captured["component"] == "res_xy"
    assert captured["cell_size"] == pytest.approx(0.01)
    assert captured["n_padding_cells"] == 4
    assert captured["interpolation_method"] == "linear"
    assert captured["interpolation_power"] == pytest.approx(2)
    assert captured["nearest_neighbors"] == 1


class _DummyZ:
    def __init__(self):
        self.z = np.array(
            [
                [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
                [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]],
            ],
            dtype=complex,
        )
        self.z_error = np.ones((2, 2, 2), dtype=float)
        self.z_model_error = np.ones((2, 2, 2), dtype=float) * 2.0
        self.frequency = np.array([1.0, 0.1], dtype=float)

    def _has_tf_error(self):
        return True

    def _has_tf_model_error(self):
        return True


class _DummyTF:
    def __init__(self):
        self.period = np.array([1.0, 10.0], dtype=float)
        self.Z = _DummyZ()
        self.Tipper = None

    def has_tipper(self):
        return False


def test_interpolated_z_uses_exact_period_sample(bokeh_plot_base_maps_class):
    obj = bokeh_plot_base_maps_class()
    obj.plot_period = 10.0

    tf = _DummyTF()

    z = obj._get_interpolated_z(tf)
    z_err = obj._get_interpolated_z_error(tf)
    z_model_err = obj._get_interpolated_z_model_error(tf)
    t = obj._get_interpolated_t(tf)
    t_err = obj._get_interpolated_t_err(tf)
    t_model_err = obj._get_interpolated_t_model_err(tf)

    np.testing.assert_allclose(z, tf.Z.z[1:2])
    np.testing.assert_allclose(z_err, tf.Z.z_error[1:2])
    np.testing.assert_allclose(z_model_err, tf.Z.z_model_error[1:2])
    np.testing.assert_allclose(t, np.zeros((1, 1, 2), dtype=complex))
    np.testing.assert_allclose(t_err, np.zeros((1, 1, 2), dtype=float))
    np.testing.assert_allclose(t_model_err, np.zeros((1, 1, 2), dtype=float))


def test_bokeh_map_plot_classes_inherit_bokeh_plot_base_maps(
    bokeh_plot_base_maps_class,
):
    from mtpy.imaging.bokeh_plots import (
        PlotPenetrationDepthMap,
        PlotPhaseTensorMaps,
        PlotResPhaseMaps,
    )

    assert issubclass(PlotPenetrationDepthMap, bokeh_plot_base_maps_class)
    assert issubclass(PlotResPhaseMaps, bokeh_plot_base_maps_class)
    assert issubclass(PlotPhaseTensorMaps, bokeh_plot_base_maps_class)
