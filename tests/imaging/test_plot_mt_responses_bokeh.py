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

    def test_station_selection_renders_and_clear_removes_plot(
        self, multi_responses_class, two_mt_objects
    ):
        import panel as pn

        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        app = obj.panel()
        station = next(
            widget
            for widget in app.select(pn.widgets.MultiSelect)
            if widget.name == "Stations"
        )
        assert station.height == 200
        assert station.get_root().size == 8
        assert station.options == [obj._station_label(mt) for mt in two_mt_objects]
        assert station.value == []
        assert not any(
            button.name == "Plot" for button in app.select(pn.widgets.Button)
        )

        station.value = [station.options[0]]
        assert any(
            "1 station(s) rendered" in str(pane.object)
            for pane in app.select(pn.pane.Markdown)
        )
        assert app.select(pn.pane.Bokeh)
        next(
            button
            for button in app.select(pn.widgets.Button)
            if button.name == "Select All"
        ).clicks += 1
        assert station.value == station.options
        assert any(
            "2 station(s) rendered" in str(pane.object)
            for pane in app.select(pn.pane.Markdown)
        )
        next(
            button for button in app.select(pn.widgets.Button) if button.name == "Clear"
        ).clicks += 1
        assert station.value == []
        assert not app.select(pn.pane.Bokeh)
        assert any(
            "No stations selected" in str(pane.object)
            for pane in app.select(pn.pane.Markdown)
        )

    def test_controls_and_compare_styles_update_without_losing_edits(
        self, multi_responses_class, two_mt_objects, monkeypatch
    ):
        import panel as pn
        from bokeh.layouts import column
        from bokeh.plotting import figure

        calls = []

        def fake_overlay(plotter):
            calls.append(
                (
                    plotter.plot_num,
                    plotter.plot_tipper,
                    plotter.plot_pt,
                    plotter.ellipse_colorby,
                    plotter.ellipse_range,
                    dict(plotter.custom_station_styles),
                )
            )
            return column(figure())

        monkeypatch.setattr(
            multi_responses_class, "_plot_compare_overlay", fake_overlay
        )
        obj = multi_responses_class(mt_data=_ObjList(two_mt_objects), show_plot=False)
        app = obj.panel()
        widgets = {widget.name: widget for widget in app.select(pn.widgets.Widget)}
        widgets["Plot Style"].value = "compare"
        assert calls == []
        station = widgets["Stations"]
        station.value = [station.options[0]]
        assert len(calls) == 1
        card = next(
            card for card in app.select(pn.Card) if card.title == "Station Styling"
        )
        card.collapsed = False
        color = next(widget for widget in card.select(pn.widgets.ColorPicker))
        marker = next(widget for widget in card.select(pn.widgets.Select))
        color.value = "#123456"
        assert len(calls) == 2
        assert calls[-1][-1][station.options[0]]["color"] == "#123456"
        assert not card.collapsed

        marker.value = "o"
        assert len(calls) == 3
        assert calls[-1][-1][station.options[0]]["marker"] == "o"
        widgets["Tensor"].value = "All"
        assert len(calls) == 4
        assert calls[-1][0:3] == (2, "yri", True)
        widgets["PT Color by"].value = "ellipticity"
        assert len(calls) == 5
        assert calls[-1][3:5] == ("ellipticity", (0.0, 1.0, 10))
        assert calls[-1][-1][station.options[0]]["color"] == "#123456"
        assert not card.collapsed

        station.value = station.options
        assert len(calls) == 6
        assert calls[-1][-1][station.options[0]]["color"] == "#123456"
        assert calls[-1][-1][station.options[0]]["marker"] == "o"

        for name, value in (
            ("Tipper", ["Real"]),
            ("Phase Tensor", False),
            ("PT min", 0.1),
            ("PT max", 0.9),
        ):
            previous = len(calls)
            widgets[name].value = value
            assert len(calls) == previous + 1
        assert calls[-1][1:3] == ("yr", False)
        assert calls[-1][4] == (0.1, 0.9, 10)

    def test_long_station_list_remains_bounded(self, multi_responses_class):
        from types import SimpleNamespace

        import panel as pn

        stations = [
            SimpleNamespace(station=f"st{index:03}", period=[1.0, 10.0])
            for index in range(100)
        ]
        plotter = multi_responses_class(mt_data=_ObjList(stations), show_plot=False)
        app = plotter.panel()
        widget = app.select(pn.widgets.MultiSelect)[0]
        assert len(widget.options) == 100
        assert widget.height == 200
        assert widget.get_root().size == 8
