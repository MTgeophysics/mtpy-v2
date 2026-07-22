# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotResPhaseMaps behavior."""

from __future__ import annotations

import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_resphase_maps_class():
    """Import Bokeh PlotResPhaseMaps or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotResPhaseMaps

    return PlotResPhaseMaps


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
    if mt_2.latitude is not None:
        mt_2.latitude = float(mt_2.latitude) + 0.01

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotResPhaseMapsBokehMTData:
    def test_iter_mt_objects_from_tree(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "Z") for mt in out)

    def test_get_data_array_from_tree(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

        arr = plotter._get_data_array()

        assert arr.shape[0] == 2
        assert arr.dtype.names is not None
        assert "res_xy" in arr.dtype.names
        assert "phase_xy" in arr.dtype.names

    def test_get_mt_objects_preinterpolates_once_per_period(
        self, bokeh_plot_resphase_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

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

    def test_plot_runs_with_tree_backed_data(
        self, bokeh_plot_resphase_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

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
            image = np.array([[1.0, 1.2], [1.1, 1.3]])
            return x, y, image

        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert set(plotter.figures.keys()) == {
            "res_xy",
            "res_yx",
            "phase_xy",
            "phase_yx",
        }
        assert all(
            k in plotter.renderers for k in ["res_xy", "res_yx", "phase_xy", "phase_yx"]
        )

    def test_plot_honors_component_toggles(
        self, bokeh_plot_resphase_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)
        plotter.plot_phase = False
        plotter.plot_yx = False
        plotter.plot_xx = False
        plotter.plot_yy = False
        plotter.plot_det = False
        plotter.plot_xy = True

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

        monkeypatch.setattr(plotter, "_get_data_array", lambda: arr)

        def _fake_interpolate_to_map(_plot_array, _comp):
            x = np.array([[127.2, 127.3], [127.2, 127.3]])
            y = np.array([[-30.9, -30.9], [-30.8, -30.8]])
            image = np.array([[1.0, 1.2], [1.1, 1.3]])
            return x, y, image

        monkeypatch.setattr(plotter, "interpolate_to_map", _fake_interpolate_to_map)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert set(plotter.figures.keys()) == {"res_xy"}

    def test_top_level_alias_import(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging import PlotResPhaseMapsBokeh

        assert PlotResPhaseMapsBokeh is not None

    def test_rainbow_reverse_palette_is_reversed(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

        rainbow = plotter._palette_from_name("rainbow")
        rainbow_r = plotter._palette_from_name("rainbow_r")

        assert rainbow_r == list(reversed(rainbow))
        assert rainbow_r is not rainbow


class TestPlotResPhaseMapsPanel:
    @pytest.fixture(autouse=True)
    def require_panel(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")

    @staticmethod
    def _find_widgets(obj, widget_type):
        found = []
        if isinstance(obj, widget_type):
            found.append(obj)
        if hasattr(obj, "objects"):
            for child in obj.objects:
                found.extend(
                    TestPlotResPhaseMapsPanel._find_widgets(child, widget_type)
                )
        return found

    def test_panel_returns_column(self, bokeh_plot_resphase_maps_class, mt_data_tree):
        import panel as pn

        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        assert isinstance(panel_obj, pn.Column)
        assert len(panel_obj) >= 3

    def test_panel_contains_expected_controls(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        import panel as pn

        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        select_widgets = self._find_widgets(panel_obj, pn.widgets.Select)
        checkbox_widgets = self._find_widgets(panel_obj, pn.widgets.Checkbox)
        button_widgets = self._find_widgets(panel_obj, pn.widgets.Button)

        select_names = {w.name for w in select_widgets}
        checkbox_names = {w.name for w in checkbox_widgets}

        assert "Map Units" in select_names
        assert "Resistivity Palette" in select_names
        assert "Phase Palette" in select_names
        assert "Resistivity Limits" in {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.RangeSlider)
        }
        assert "Phase Limits" in {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.RangeSlider)
        }
        assert "Plot Stations" in checkbox_names
        assert any(("Refresh" in (b.name or "")) for b in button_widgets)

    def test_panel_refresh_updates_plotter_parameters(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        import panel as pn

        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        period_widget = [
            w
            for w in panel_obj.select()
            if getattr(w, "name", "") == "Plot Period (s)" and hasattr(w, "value")
        ][0]
        map_units_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Select)
            if w.name == "Map Units"
        ][0]
        res_palette_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Select)
            if w.name == "Resistivity Palette"
        ][0]
        phase_palette_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Select)
            if w.name == "Phase Palette"
        ][0]
        res_limits_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.RangeSlider)
            if w.name == "Resistivity Limits"
        ][0]
        phase_limits_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.RangeSlider)
            if w.name == "Phase Limits"
        ][0]
        station_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Checkbox)
            if w.name == "Plot Stations"
        ][0]
        component_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.CheckButtonGroup)
            if w.name == "Components"
        ][0]
        rows_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.CheckBoxGroup)
            if w.name == "Rows"
        ][0]
        refresh_btn = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Button)
            if "Refresh" in (w.name or "")
        ][0]

        period_widget.value = 10.0
        map_units_widget.value = "km"
        res_palette_widget.value = "magma"
        phase_palette_widget.value = "viridis"
        res_limits_widget.value = (-1.0, 2.0)
        phase_limits_widget.value = (10.0, 80.0)
        station_widget.value = False
        component_widget.value = ["xy"]
        rows_widget.value = ["Resistivity"]

        refresh_btn.clicks = refresh_btn.clicks + 1

        assert plotter.plot_period == pytest.approx(10.0)
        assert plotter.map_units == "km"
        assert str(plotter.res_cmap).lower() == "magma"
        assert str(plotter.phase_cmap).lower() == "viridis"
        assert plotter.cmap_limits["res_xy"] == (-1.0, 2.0)
        assert plotter.cmap_limits["phase_xy"] == (10.0, 80.0)
        assert plotter.cmap_limits["res_yx"] == (-1.0, 2.0)
        assert plotter.cmap_limits["phase_det"] == (10.0, 80.0)
        assert plotter.plot_stations is False
        assert plotter.plot_xy is True
        assert plotter.plot_yx is False
        assert plotter.plot_resistivity is True
        assert plotter.plot_phase is False

    def test_servable_returns_viewable(
        self, bokeh_plot_resphase_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_resphase_maps_class(mt_data_tree, show_plot=False)

        viewable = plotter.servable(title="ResPhase Maps")

        assert viewable is not None
