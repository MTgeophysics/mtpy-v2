# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotMultipleResponses and its panel() method."""

import pytest
from mt_metadata import TF_EDI_CGG, TF_EDI_NO_ERROR

from mtpy import MT


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def multi_responses_class():
    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotMultipleResponses

    return PlotMultipleResponses


@pytest.fixture(scope="session")
def two_mt_objects():
    """Two distinct MT objects for multi-station tests."""
    mt1 = MT(TF_EDI_CGG)
    mt1.read()
    mt2 = MT(TF_EDI_NO_ERROR)
    mt2.read()
    return [mt1, mt2]


class _ObjList:
    """Minimal mt_data shim that exposes values()."""

    def __init__(self, objs):
        self._objs = objs

    def values(self):
        return iter(self._objs)


# ── Basic construction ─────────────────────────────────────────────────────────


class TestPlotMultipleResponsesConstruction:
    def test_constructs_with_obj_list(self, multi_responses_class, two_mt_objects):
        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        assert obj is not None

    def test_custom_station_styles_initialised_empty(
        self, multi_responses_class, two_mt_objects
    ):
        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        assert obj.custom_station_styles == {}

    def test_has_panel_method(self, multi_responses_class, two_mt_objects):
        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        assert callable(getattr(obj, "panel", None))


# ── plot() ─────────────────────────────────────────────────────────────────────


class TestPlotMultipleResponsesPlot:
    def test_compare_overlay_returns_column(
        self, multi_responses_class, two_mt_objects
    ):
        obj = multi_responses_class(
            mt_data=_ObjList(two_mt_objects),
            show_plot=False,
            plot_style="compare",
        )
        layout = obj.plot()
        from bokeh.layouts import Column

        assert isinstance(layout, Column)

    def test_compare_overlay_custom_colors_no_error(
        self, multi_responses_class, two_mt_objects
    ):
        obj = multi_responses_class(
            mt_data=_ObjList(two_mt_objects),
            show_plot=False,
            plot_style="compare",
        )
        labels = [obj._station_label(mt_obj) for mt_obj in two_mt_objects]
        obj.custom_station_styles = {
            labels[0]: {"color": "#ff0000", "marker": "o"},
            labels[1]: {"color": "#00ff00", "marker": "s"},
        }
        layout = obj._plot_compare_overlay()
        assert layout is not None


# ── panel() ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    pytest.importorskip("panel", reason="panel not installed") is None,
    reason="panel not installed",
)
class TestPlotMultipleResponsesPanel:
    def test_panel_returns_panel_column(self, multi_responses_class, two_mt_objects):
        import panel as pn

        pytest.importorskip("panel")
        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        result = obj.panel()
        assert isinstance(result, pn.Column)

    def test_panel_contains_plot_display(self, multi_responses_class, two_mt_objects):
        pytest.importorskip("panel")

        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        result = obj.panel()
        # The column must contain at least the controls and the plot area.
        assert len(result.objects) >= 2

    def test_panel_empty_mt_data_returns_markdown(self, multi_responses_class):
        pytest.importorskip("panel")
        import panel as pn

        obj = multi_responses_class(mt_data=_ObjList([]), show_plot=False)
        result = obj.panel()
        assert isinstance(result, pn.pane.Markdown)
