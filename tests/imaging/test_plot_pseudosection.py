# -*- coding: utf-8 -*-
"""Tests for PlotResPhasePseudoSection MTDataTree compatibility."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTDataTree
from mtpy.imaging.plot_pseudosection import PlotResPhasePseudoSection


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


class TestPlotResPhasePseudoSectionMTDataTree:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotResPhasePseudoSection(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "Z") for mt in out)

    def test_rotation_angle_uses_tree_rotate(self, mt_data_tree, monkeypatch):
        plotter = PlotResPhasePseudoSection(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)

        plotter.rotation_angle = 10.0

        assert called["value"] == 10.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 10.0

    def test_get_data_df_from_tree(self, mt_data_tree, monkeypatch):
        plotter = PlotResPhasePseudoSection(mt_data_tree, show_plot=False)
        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)

        df = plotter._get_data_df()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert set(["station", "offset", "period", "res_xy", "phase_xy"]).issubset(
            set(df.columns)
        )

    def test_plot_runs_with_tree_backed_data(self, mt_data_tree, monkeypatch):
        plotter = PlotResPhasePseudoSection(mt_data_tree, show_plot=False)

        plotter.data_df = pd.DataFrame(
            {
                "station": ["TEST01", "TEST01", "TEST02", "TEST02"],
                "offset": [0.0, 0.0, 1.0, 1.0],
                "period": [0.0, 1.0, 0.0, 1.0],
                "res_xx": [1.0, 1.2, 1.1, 1.3],
                "res_xy": [1.0, 1.2, 1.1, 1.3],
                "res_yx": [1.0, 1.2, 1.1, 1.3],
                "res_yy": [1.0, 1.2, 1.1, 1.3],
                "res_det": [1.0, 1.2, 1.1, 1.3],
                "phase_xx": [20.0, 25.0, 22.0, 28.0],
                "phase_xy": [30.0, 35.0, 31.0, 36.0],
                "phase_yx": [40.0, 45.0, 42.0, 46.0],
                "phase_yy": [15.0, 18.0, 16.0, 19.0],
                "phase_det": [25.0, 30.0, 27.0, 31.0],
            }
        )

        def _fake_griddata_interpolate(_x, _y, _z, _x_nodes, y_nodes, _method):
            grid_x = np.array([[0.0, 1.0], [0.0, 1.0]])
            grid_y = np.array(
                [[y_nodes.min(), y_nodes.min()], [y_nodes.max(), y_nodes.max()]]
            )
            image = np.array([[1.0]])
            return grid_x, grid_y, image

        monkeypatch.setattr(
            "mtpy.imaging.plot_pseudosection.griddata_interpolate",
            _fake_griddata_interpolate,
        )
        monkeypatch.setattr(plotter, "_get_colorbar", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(plt, "show", lambda: None)

        plotter.plot()

        assert plotter.fig is not None
        plt.close(plotter.fig)
