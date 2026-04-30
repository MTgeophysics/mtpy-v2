# -*- coding: utf-8 -*-
"""Tests for PlotResidualPTMaps MTData compatibility."""

import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData
from mtpy.imaging.plot_residual_pt_maps import PlotResidualPTMaps


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for test session."""
    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def paired_trees(mt_object_cache):
    """Build two MTData containers with matching stations."""
    mt1_a = mt_object_cache.copy()
    mt1_a.station = "TEST01"
    mt1_b = mt_object_cache.copy()
    mt1_b.station = "TEST02"
    if mt1_b.longitude is not None:
        mt1_b.longitude = float(mt1_b.longitude) + 0.02
    if mt1_b.latitude is not None:
        mt1_b.latitude = float(mt1_b.latitude) + 0.01

    mt2_a = mt1_a.copy()
    mt2_b = mt1_b.copy()

    tree_01 = MTData()
    tree_01.add_stations([mt1_a, mt1_b])

    tree_02 = MTData()
    tree_02.add_stations([mt2_a, mt2_b])

    return tree_01, tree_02


class TestPlotResidualPTMapsMTData:
    def test_container_items_from_tree(self, paired_trees):
        tree_01, tree_02 = paired_trees
        plotter = PlotResidualPTMaps(tree_01, tree_02, show_plot=False)

        items_01 = plotter._container_items(tree_01)
        items_02 = plotter._container_items(tree_02)

        assert len(items_01) == 2
        assert len(items_02) == 2
        assert all(hasattr(mt, "station") for _, mt in items_01)

    def test_match_lists_with_tree_inputs(self, paired_trees):
        tree_01, tree_02 = paired_trees
        plotter = PlotResidualPTMaps(tree_01, tree_02, show_plot=False)

        matches = plotter._match_lists(tree_01, tree_02)

        assert len(matches) == 2
        assert all(len(pair) == 2 for pair in matches)

    def test_rotation_angle_uses_tree_rotate(self, paired_trees, monkeypatch):
        tree_01, tree_02 = paired_trees
        plotter = PlotResidualPTMaps(tree_01, tree_02, show_plot=False)

        calls = []

        def _fake_rotate_01(value, inplace=True):
            calls.append(("one", value, inplace))

        def _fake_rotate_02(value, inplace=True):
            calls.append(("two", value, inplace))

        monkeypatch.setattr(tree_01, "rotate", _fake_rotate_01)
        monkeypatch.setattr(tree_02, "rotate", _fake_rotate_02)

        plotter.rotation_angle = 20.0

        assert plotter.rotation_angle == 20.0
        assert ("one", 20.0, True) in calls
        assert ("two", 20.0, True) in calls

    def test_compute_residual_pt_with_tree_inputs(self, paired_trees):
        tree_01, tree_02 = paired_trees
        plotter = PlotResidualPTMaps(
            tree_01,
            tree_02,
            frequencies=tree_01.get_periods() ** -1,
            show_plot=False,
        )

        plotter._compute_residual_pt()

        assert plotter.rpt_array is not None
        assert plotter.rpt_array.shape[0] == 2
        assert plotter.residual_pt_list is not None
        assert len(plotter.residual_pt_list) == 2
