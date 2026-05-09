# -*- coding: utf-8 -*-
"""Tests for PlotPhaseTensorPseudoSection MTData compatibility."""

import matplotlib.pyplot as plt
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData
from mtpy.imaging.plot_phase_tensor_pseudosection import PlotPhaseTensorPseudoSection

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for test session."""
    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    """Build MTData with two stations."""
    mt_1 = mt_object_cache.copy()
    mt_1.station = "TEST01"

    mt_2 = mt_object_cache.copy()
    mt_2.station = "TEST02"
    if mt_2.longitude is not None:
        mt_2.longitude = float(mt_2.longitude) + 0.02

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotPhaseTensorPseudoSectionMTData:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotPhaseTensorPseudoSection(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "pt") for mt in out)

    def test_rotation_angle_uses_tree_rotate(self, mt_data_tree, monkeypatch):
        plotter = PlotPhaseTensorPseudoSection(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)

        plotter.rotation_angle = 12.0

        assert called["value"] == 12.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 12.0

    def test_plot_uses_tree_objects(self, mt_data_tree, monkeypatch):
        plotter = PlotPhaseTensorPseudoSection(mt_data_tree, show_plot=False)
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        call_count = {"n": 0}

        def _fake_get_patch(tf):
            index = call_count["n"]
            call_count["n"] += 1
            return float(index + 1), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_colorbar", lambda: None)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot()

        assert call_count["n"] == 2
        assert plotter.station_list.shape[0] == 2

        if plotter.fig is not None:
            plt.close(plotter.fig)

    def test_plot_honors_aspect_kwarg(self, mt_data_tree, monkeypatch):
        plotter = PlotPhaseTensorPseudoSection(
            mt_data_tree,
            show_plot=False,
            aspect="auto",
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        def _fake_get_patch(tf):
            return float(len(tf.station)), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_colorbar", lambda: None)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot()

        assert plotter.ax.get_aspect() == "auto"

        if plotter.fig is not None:
            plt.close(plotter.fig)
