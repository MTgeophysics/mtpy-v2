# -*- coding: utf-8 -*-
"""Tests for PlotStrike MTDataTree compatibility."""

import pandas as pd
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTDataTree
from mtpy.imaging.plot_strike import PlotStrike


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


class TestPlotStrikeMTDataTree:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotStrike(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "period") for mt in out)

    def test_rotation_angle_uses_tree_rotate(self, mt_data_tree, monkeypatch):
        plotter = PlotStrike(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)
        monkeypatch.setattr(plotter, "make_strike_df", lambda: None)

        plotter.rotation_angle = 15.0

        assert called["value"] == 15.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 15.0

    def test_make_strike_df_from_tree(self, mt_data_tree):
        plotter = PlotStrike(mt_data_tree, show_plot=False)

        plotter.make_strike_df()

        assert isinstance(plotter.strike_df, pd.DataFrame)
        assert not plotter.strike_df.empty
        assert set(["estimate", "period", "plot_strike", "measured_strike"]).issubset(
            set(plotter.strike_df.columns)
        )
