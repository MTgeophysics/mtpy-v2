# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPhaseTensorMaps container compatibility."""

import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_phase_tensor_maps_class():
    """Import Bokeh PlotPhaseTensorMaps or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPhaseTensorMaps

    return PlotPhaseTensorMaps


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for the test session."""

    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    """Build an MTData with two stations."""

    mt_1 = mt_object_cache.copy()
    mt_1.station = "TEST01"

    mt_2 = mt_object_cache.copy()
    mt_2.station = "TEST02"
    if mt_2.longitude is not None:
        mt_2.longitude = float(mt_2.longitude) + 0.02

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotPhaseTensorMapsBokehMTData:
    def test_iter_mt_objects_from_tree(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "Z") for mt in out)

    def test_rotation_angle_uses_tree_rotate(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)

        plotter.rotation_angle = 15.0

        assert called["value"] == 15.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 15.0

    def test_plot_uses_tree_objects(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        plotter.plot_pt = False
        plotter.plot_tipper = "n"
        plotter.pt_type = "wedges"
        plotter.plot_station = False

        call_count = {"n": 0}

        def _fake_get_patch_wedges(_tf):
            index = call_count["n"]
            call_count["n"] += 1
            return float(index + 1), float(index + 2)

        monkeypatch.setattr(plotter, "_get_patch_wedges", _fake_get_patch_wedges)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert call_count["n"] == 2
        assert plotter.plot_xarr.shape[0] == 2
        assert plotter.plot_yarr.shape[0] == 2

    def test_get_mt_objects_preinterpolates_once_per_period(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

        call_count = {"n": 0}

        def _fake_interpolate(new_periods, inplace=True, bounds_error=True, **kwargs):
            call_count["n"] += 1
            assert inplace is False
            assert bounds_error is False
            assert new_periods.shape == (1,)
            return mt_data_tree

        monkeypatch.setattr(mt_data_tree, "interpolate", _fake_interpolate)

        plotter.plot_period = 1.0
        out_1 = plotter._get_mt_objects()
        out_2 = plotter._get_mt_objects()

        assert len(out_1) == 2
        assert len(out_2) == 2
        assert call_count["n"] == 1

        plotter.plot_period = 10.0
        out_3 = plotter._get_mt_objects()

        assert len(out_3) == 2
        assert call_count["n"] == 2

    def test_get_interpolated_z_uses_direct_period_sample(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        tf = plotter._get_mt_objects()[0]

        plotter.plot_period = float(tf.period[0])

        def _raise_interp(*args, **kwargs):
            raise AssertionError("interp1d path should not be used for exact periods")

        monkeypatch.setattr(plotter, "get_interp1d_functions_z", _raise_interp)

        z = plotter._get_interpolated_z(tf)

        assert z.shape == (1, 2, 2)
