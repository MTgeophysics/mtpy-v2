# -*- coding: utf-8 -*-
"""Tests for PlotResPhaseMaps MTDataTree compatibility."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTDataTree
from mtpy.imaging.plot_resphase_maps import PlotResPhaseMaps


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for test session."""
    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    """Build MTDataTree with two stations."""
    mt_1 = mt_object_cache.copy()
    mt_1.station = "TEST01"

    mt_2 = mt_object_cache.copy()
    mt_2.station = "TEST02"
    if mt_2.longitude is not None:
        mt_2.longitude = float(mt_2.longitude) + 0.02
    if mt_2.latitude is not None:
        mt_2.latitude = float(mt_2.latitude) + 0.01

    tree = MTDataTree()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotResPhaseMapsMTDataTree:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotResPhaseMaps(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "Z") for mt in out)

    def test_get_data_array_from_tree(self, mt_data_tree):
        plotter = PlotResPhaseMaps(mt_data_tree, show_plot=False)

        arr = plotter._get_data_array()

        assert arr.shape[0] == 2
        assert arr.dtype.names is not None
        assert "res_xy" in arr.dtype.names
        assert "phase_xy" in arr.dtype.names

    def test_plot_runs_with_tree_backed_data(self, mt_data_tree, monkeypatch):
        plotter = PlotResPhaseMaps(mt_data_tree, show_plot=False)
        plotter.interpolation_method = "nearest"

        arr = np.zeros(
            2,
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
        arr["station"] = ["TEST01", "TEST02"]
        arr["longitude"] = [127.2, 127.3]
        arr["latitude"] = [-30.9, -30.8]
        arr["res_xy"] = [10.0, 12.0]
        arr["res_yx"] = [11.0, 13.0]
        arr["phase_xy"] = [30.0, 35.0]
        arr["phase_yx"] = [40.0, 45.0]

        monkeypatch.setattr(plotter, "_get_data_array", lambda: arr)

        def _fake_interpolate_to_map(_plot_array, _comp):
            x = np.array([[127.2, 127.3], [127.2, 127.3]])
            y = np.array([[-30.9, -30.9], [-30.8, -30.8]])
            image = np.array([[1.0]])
            return x, y, image

        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)
        monkeypatch.setattr(plotter, "_get_colorbar", lambda *_args, **_kwargs: None)

        plotter.plot()

        assert plotter.fig is not None
        plt.close(plotter.fig)
