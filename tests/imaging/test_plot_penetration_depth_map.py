# -*- coding: utf-8 -*-
"""Tests for PlotPenetrationDepthMap container compatibility."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData
from mtpy.imaging.plot_penetration_depth_map import PlotPenetrationDepthMap

pytestmark = pytest.mark.plotting


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


class TestPlotPenetrationDepthMapMTData:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotPenetrationDepthMap(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "Z") for mt in out)
        assert all(hasattr(mt, "period") for mt in out)

    def test_get_depth_array_from_tree(self, mt_data_tree):
        plotter = PlotPenetrationDepthMap(mt_data_tree, show_plot=False)
        plotter.plot_period = 1.0

        depth_array = plotter._get_depth_array()

        assert depth_array.dtype.names == (
            "station",
            "latitude",
            "longitude",
            "elevation",
            "det",
            "xy",
            "yx",
        )
        assert depth_array.shape[0] == 2

    def test_plot_reuses_cached_depth_array_for_same_period(
        self, mt_data_tree, monkeypatch
    ):
        plotter = PlotPenetrationDepthMap(mt_data_tree, show_plot=False)
        plotter.interpolation_method = "nearest"
        plotter.plot_stations = False
        plotter.plot_det = True
        plotter.plot_te = False
        plotter.plot_tm = False

        depth_array = np.zeros(
            1,
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
        depth_array["station"][0] = "TEST01"
        depth_array["latitude"][0] = 0.0
        depth_array["longitude"][0] = 0.0
        depth_array["det"][0] = 1.0

        call_count = {"n": 0}

        def _fake_get_depth_array():
            call_count["n"] += 1
            return depth_array

        monkeypatch.setattr(plotter, "_get_depth_array", _fake_get_depth_array)
        monkeypatch.setattr(plotter, "_filter_depth_array", lambda arr, _comp: arr)
        monkeypatch.setattr(Figure, "suptitle", lambda *args, **kwargs: None)

        def _fake_interpolate_to_map(_arr, _comp):
            x = np.array([[0.0, 1.0], [0.0, 1.0]])
            y = np.array([[0.0, 0.0], [1.0, 1.0]])
            image = np.array([[1.0]])
            return x, y, image

        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)

        plotter.plot_period = 10.0
        plotter.plot()
        plotter.plot()

        assert call_count["n"] == 1
        if plotter.fig is not None:
            plt.close(plotter.fig)

    def test_get_mt_objects_preinterpolates_once_per_period(
        self, mt_data_tree, monkeypatch
    ):
        plotter = PlotPenetrationDepthMap(mt_data_tree, show_plot=False)

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

    def test_plot_suptitle_uses_fontdict(self, mt_data_tree, monkeypatch):
        plotter = PlotPenetrationDepthMap(mt_data_tree, show_plot=False)
        plotter.interpolation_method = "nearest"
        plotter.plot_stations = False
        plotter.plot_det = True
        plotter.plot_te = False
        plotter.plot_tm = False

        depth_array = np.zeros(
            1,
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
        depth_array["station"][0] = "TEST01"
        depth_array["latitude"][0] = 0.0
        depth_array["longitude"][0] = 0.0
        depth_array["det"][0] = 1.0

        monkeypatch.setattr(plotter, "_get_depth_array", lambda: depth_array)
        monkeypatch.setattr(plotter, "_filter_depth_array", lambda arr, _comp: arr)

        def _fake_interpolate_to_map(_arr, _comp):
            x = np.array([[0.0, 1.0], [0.0, 1.0]])
            y = np.array([[0.0, 0.0], [1.0, 1.0]])
            image = np.array([[1.0]])
            return x, y, image

        captured = {"kwargs": None}

        def _fake_suptitle(_self, *_args, **kwargs):
            captured["kwargs"] = kwargs
            return None

        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)
        monkeypatch.setattr(Figure, "suptitle", _fake_suptitle)

        plotter.plot_period = 10.0
        plotter.plot()

        assert captured["kwargs"] is not None
        assert "fontdict" in captured["kwargs"]
        assert "fontproperties" not in captured["kwargs"]
        if plotter.fig is not None:
            plt.close(plotter.fig)
