# -*- coding: utf-8 -*-
"""Tests for PlotMultipleResponses container compatibility."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData
from mtpy.imaging.plot_mt_responses import PlotMultipleResponses

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for the session."""
    m1 = MT(TF_EDI_CGG)
    m1.read()
    return m1


@pytest.fixture
def mt_object(mt_object_cache):
    """Provide MT object fixture."""
    return mt_object_cache


@pytest.fixture
def mt_data_tree(mt_object):
    """Build a single-station MTData."""
    tree = MTData()
    tree.add_station(mt_object)
    return tree


class TestPlotMultipleResponsesMTData:
    def test_iter_mt_objects_from_tree(self, mt_data_tree):
        plotter = PlotMultipleResponses(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 1
        assert hasattr(out[0], "Z")
        assert hasattr(out[0], "Tipper")
        assert hasattr(out[0], "pt")
        assert hasattr(out[0], "period")

    def test_rotation_angle_uses_tree_rotate(self, mt_data_tree):
        plotter = PlotMultipleResponses(mt_data_tree, show_plot=False)

        with patch.object(mt_data_tree, "rotate", autospec=True) as mock_rotate:
            plotter.rotation_angle = 22.5

        mock_rotate.assert_called_once_with(22.5, inplace=True)
        assert plotter.rotation_angle == 22.5

    def test_plot_single_uses_tree_iterator(self, mt_data_tree):
        plotter = PlotMultipleResponses(mt_data_tree, show_plot=False)

        out = plotter._plot_single()

        assert isinstance(out, dict)
        assert len(out) == 1

        for value in out.values():
            assert value is not None
            fig = getattr(value, "fig", None)
            if fig is not None:
                plt.close(fig)

    def test_get_mt_objects_returns_station_count(self, mt_object):
        tree = MTData()
        mt_2 = mt_object.copy()
        mt_2.station = f"{mt_object.station}_2"
        tree.add_stations([mt_object, mt_2])
        plotter = PlotMultipleResponses(tree, show_plot=False)

        out = plotter._get_mt_objects()

        assert len(out) == 2
