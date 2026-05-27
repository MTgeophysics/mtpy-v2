# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPenetrationDepth1D and associated Panel apps."""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_penetration_depth_class():
    """Import the Bokeh penetration-depth plotter or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

    return PlotPenetrationDepth1D


class _FakeZ:
    def __init__(self, depth_array):
        self._depth_array = depth_array

    def estimate_depth_of_investigation(self):
        return self._depth_array


class _FakeTF:
    def __init__(self, station, depth_array):
        self.station = station
        self.Z = _FakeZ(depth_array)


@pytest.fixture
def depth_array_unsorted():
    """Create a small unsorted depth array for regression checks."""

    return {
        "period": np.array([10.0, 1.0, 100.0]),
        "depth_min": np.array([800.0, 80.0, 8000.0]),
        "depth_max": np.array([1200.0, 120.0, 12000.0]),
        "depth_xy": np.array([900.0, 90.0, 9000.0]),
        "depth_yx": np.array([1000.0, 100.0, 10000.0]),
        "depth_det": np.array([1100.0, 110.0, 11000.0]),
    }


@pytest.fixture
def fake_tf(depth_array_unsorted):
    return _FakeTF("TEST01", depth_array_unsorted)


class TestPlotPenetrationDepth1DBokeh:
    def test_depth_units_maps_to_scale(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)

        assert plotter.depth_units == "km"
        assert plotter.depth_scale == 1.0 / 1000

        plotter.depth_units = "m"
        assert plotter.depth_scale == 1

    def test_depth_units_rejects_invalid_values(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)

        with pytest.raises(ValueError, match="depth_units must be either 'km' or 'm'"):
            plotter.depth_units = "cm"

    def test_plot_builds_bokeh_figure_with_expected_elements(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        layout = plotter.plot()

        assert layout is plotter.fig
        assert plotter.fig.title.text == "Depth of investigation for TEST01"
        assert plotter.fig.xaxis.axis_label == "Depth (km)"
        assert plotter.fig.yaxis.axis_label == "Period (s)"
        assert plotter.fig.y_range.start > plotter.fig.y_range.end
        assert len(plotter.fig.legend) == 1
        assert len(plotter.fig.legend[0].items) == 3

        patch_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Patch"
        ]
        line_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Line"
        ]

        assert len(patch_renderers) == 1
        assert len(line_renderers) == 3
        assert np.isclose(plotter.fig.x_range.start, 0.01)
        assert np.isclose(plotter.fig.x_range.end, 100.0)

    def test_plot_sorts_periods_before_rendering(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        plotter.plot()

        line_renderers = [
            renderer
            for renderer in plotter.fig.renderers
            if getattr(getattr(renderer, "glyph", None), "__class__", None)
            and renderer.glyph.__class__.__name__ == "Line"
        ]
        periods = line_renderers[0].data_source.data["period"]

        assert list(periods) == sorted(periods)

    # ── New tests: BokehPlotBase and make_panel ───────────────────────────

    def test_inherits_from_bokeh_plot_base(self, bokeh_plot_penetration_depth_class):
        """PlotPenetrationDepth1D should inherit from BokehPlotBase."""
        from mtpy.imaging.bokeh_plots.base import BokehPlotBase

        assert issubclass(bokeh_plot_penetration_depth_class, BokehPlotBase)

    def test_plot_stores_renderers_by_component(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """plot() should populate _renderers with xy, yx, det entries."""
        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        plotter.plot()

        assert set(plotter._renderers.keys()) == {"xy", "yx", "det"}
        for comp, renderers in plotter._renderers.items():
            assert (
                len(renderers) == 2
            ), f"Expected 2 renderers for {comp}, got {len(renderers)}"

    def test_make_panel_returns_panel_column(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """make_panel() should return a Panel Column containing controls and plot."""
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        result = plotter.make_panel()

        assert isinstance(result, pn.Column)

    def test_make_panel_triggers_plot_when_layout_is_none(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """make_panel() should call plot() automatically if not done yet."""
        pytest.importorskip("panel")

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        assert plotter.layout is None

        plotter.make_panel()

        assert plotter.layout is not None

    def test_make_panel_contains_mode_controls(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """make_panel() should include CheckButtonGroup for xy/yx/det modes."""
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        panel_layout = plotter.make_panel()

        # Recursively collect all Panel objects
        def collect_widgets(obj, cls):
            results = []
            if isinstance(obj, cls):
                results.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    results.extend(collect_widgets(child, cls))
            return results

        check_groups = collect_widgets(panel_layout, pn.widgets.CheckButtonGroup)
        assert (
            len(check_groups) >= 1
        ), "Expected at least one CheckButtonGroup for modes"

        # Verify the mode widget covers all three components
        mode_options = {}
        for cg in check_groups:
            mode_options.update(
                cg.options
                if isinstance(cg.options, dict)
                else {v: v for v in cg.options}
            )
        assert "xy" in mode_options.values() or "xy" in mode_options
        assert "yx" in mode_options.values() or "yx" in mode_options
        assert "det" in mode_options.values() or "det" in mode_options

    def test_make_panel_contains_depth_units_control(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """make_panel() should include a RadioButtonGroup for depth units."""
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        panel_layout = plotter.make_panel()

        def collect_widgets(obj, cls):
            results = []
            if isinstance(obj, cls):
                results.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    results.extend(collect_widgets(child, cls))
            return results

        radio_groups = collect_widgets(panel_layout, pn.widgets.RadioButtonGroup)
        assert len(radio_groups) >= 1

        all_options = []
        for rg in radio_groups:
            all_options.extend(rg.options)
        assert "km" in all_options
        assert "m" in all_options

    def test_make_panel_mode_toggle_changes_renderer_visibility(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """Toggling a mode widget value should hide the corresponding renderers."""
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        panel_layout = plotter.make_panel()

        # Find the mode CheckButtonGroup
        def collect_widgets(obj, cls):
            results = []
            if isinstance(obj, cls):
                results.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    results.extend(collect_widgets(child, cls))
            return results

        check_groups = collect_widgets(panel_layout, pn.widgets.CheckButtonGroup)
        mode_widget = check_groups[0]

        # All renderers should be visible by default
        for comp in ["xy", "yx", "det"]:
            for renderer in plotter._renderers[comp]:
                assert renderer.visible

        # Deselect "xy" — its renderers should become invisible
        mode_widget.value = ["yx", "det"]
        for renderer in plotter._renderers["xy"]:
            assert not renderer.visible
        for comp in ["yx", "det"]:
            for renderer in plotter._renderers[comp]:
                assert renderer.visible

    def test_make_panel_depth_units_switch_rebuilds_figure(
        self, bokeh_plot_penetration_depth_class, fake_tf
    ):
        """Changing depth units should rebuild the figure and update the pane."""
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_penetration_depth_class(fake_tf, show_plot=False)
        panel_layout = plotter.make_panel()

        def collect_widgets(obj, cls):
            results = []
            if isinstance(obj, cls):
                results.append(obj)
            if hasattr(obj, "objects"):
                for child in obj.objects:
                    results.extend(collect_widgets(child, cls))
            return results

        radio_groups = collect_widgets(panel_layout, pn.widgets.RadioButtonGroup)
        depth_widget = radio_groups[0]

        original_fig = plotter.fig

        # Switch to metres
        depth_widget.value = "m"

        assert plotter.depth_units == "m"
        assert plotter.depth_scale == 1
        # A new figure should have been created
        assert plotter.fig is not original_fig


class TestBokehPlotBase:
    """Tests for the BokehPlotBase mixin."""

    def test_make_panel_default_wraps_layout(self, fake_tf):
        """BokehPlotBase.make_panel() should wrap the layout in a Panel pane."""
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")
        import panel as pn

        from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

        plotter = PlotPenetrationDepth1D(fake_tf, show_plot=False)
        plotter.plot()

        # Call the base-class make_panel directly to exercise the default behaviour
        from mtpy.imaging.bokeh_plots.base import BokehPlotBase

        result = BokehPlotBase.make_panel(plotter)
        assert isinstance(result, pn.pane.Bokeh)

    def test_make_panel_calls_plot_when_layout_none(self, fake_tf):
        """BokehPlotBase.make_panel() calls plot() automatically."""
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D
        from mtpy.imaging.bokeh_plots.base import BokehPlotBase

        plotter = PlotPenetrationDepth1D(fake_tf, show_plot=False)
        assert plotter.layout is None

        BokehPlotBase.make_panel(plotter)

    def test_panel_alias_returns_same_result_as_make_panel(self, fake_tf):
        """panel() should be an alias for make_panel() on BokehPlotBase."""
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

        plotter = PlotPenetrationDepth1D(fake_tf, show_plot=False)
        plotter.plot()

        result_make = plotter.make_panel()
        result_panel = plotter.panel()

        assert type(result_make) is type(result_panel)

    def test_panel_method_exists_on_subclass(self, fake_tf):
        """PlotPenetrationDepth1D should expose .panel() via BokehPlotBase."""
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

        plotter = PlotPenetrationDepth1D(fake_tf, show_plot=False)
        assert hasattr(plotter, "panel") and callable(plotter.panel)


class TestPenetrationDepth1DApp:
    """Tests for the standalone PenetrationDepth1DApp."""

    @pytest.fixture(scope="class")
    def app_class(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.panel_penetration_depth_1d_app import (
            PenetrationDepth1DApp,
        )

        return PenetrationDepth1DApp

    def test_app_instantiates(self, app_class):
        app = app_class()
        assert app is not None

    def test_view_returns_panel_column(self, app_class):
        import panel as pn

        app = app_class()
        assert isinstance(app.view, pn.Column)

    def test_mt_object_is_none_before_load(self, app_class):
        app = app_class()
        assert app.mt_object is None

    def test_load_with_no_selection_sets_warning_status(self, app_class):
        app = app_class()
        app._file_selector.value = []
        app._on_load_clicked(None)
        assert "No file selected" in app._status.object

    def test_load_with_unsupported_suffix_sets_error_status(self, app_class):
        import os
        import tempfile

        app = app_class()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            app._file_selector.value = [tmp_path]
            app._on_load_clicked(None)
            assert "Unsupported file type" in app._status.object
        finally:
            os.unlink(tmp_path)

    def test_file_pattern_widget_updates_file_selector(self, app_class):
        app = app_class()
        app._file_pattern_widget.value = "*.xml"
        assert app._file_selector.file_pattern == "*.xml"


class TestMTDataAppPenetrationDepthIntegration:
    """Tests for penetration depth integration in MTDataApp."""

    @pytest.fixture(scope="class")
    def app_class(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTDataApp

        return MTDataApp

    def test_app_has_pen_depth_container(self, app_class):
        import panel as pn

        app = app_class()
        assert hasattr(app, "_pen_depth_container")
        assert isinstance(app._pen_depth_container, pn.Column)

    def test_view_contains_penetration_depth_section(self, app_class):
        import panel as pn

        app = app_class()
        view = app.view

        def find_markdown_text(obj, text):
            if isinstance(obj, pn.pane.Markdown) and text in (obj.object or ""):
                return True
            if hasattr(obj, "objects"):
                return any(find_markdown_text(child, text) for child in obj.objects)
            return False

        assert find_markdown_text(view, "Penetration Depth")

    def test_update_penetration_depth_multiple_stations_shows_message(
        self, app_class, depth_array_unsorted
    ):
        """Selecting >1 station should show an instructional message, not a plot."""
        app = app_class()
        # Provide a dummy mt_data so get_station is not called
        app._mt_data = object()  # non-None sentinel

        two_paths = {"/Surveys/s1/Stations/ST01", "/Surveys/s1/Stations/ST02"}
        app._update_penetration_depth(two_paths)

        import panel as pn

        has_msg = any(
            isinstance(obj, pn.pane.Markdown) and "exactly" in (obj.object or "")
            for obj in app._pen_depth_container.objects
        )
        assert has_msg

    def test_table_deselection_resets_pen_depth_container(self, app_class):
        """Deselecting all rows should reset the penetration depth container."""
        import panel as pn

        app = app_class()
        app._mt_data = object()  # make non-None so callback runs

        # Simulate deselection
        class FakeEvent:
            new = []

        app._on_table_selection_changed(FakeEvent())

        has_placeholder = any(
            isinstance(obj, pn.pane.Markdown)
            and "penetration depth" in (obj.object or "").lower()
            for obj in app._pen_depth_container.objects
        )
        assert has_placeholder

    def test_table_selection_calls_update_penetration_depth(self, app_class):
        """Selecting stations via the table should trigger _update_penetration_depth."""
        app = app_class()
        called_with = []

        def _spy(station_paths):
            called_with.append(set(station_paths))

        app._update_penetration_depth = _spy
        app._mt_data = object()  # non-None so callback proceeds past guard

        import pandas as pd

        app._station_table.value = pd.DataFrame(
            [
                {
                    "survey": "s1",
                    "station": "ST01",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "elevation": 0.0,
                    "n_periods": 3,
                    "has_impedandance": True,
                    "has_tipper": False,
                }
            ],
        )

        class FakeEvent:
            new = [0]

        app._on_table_selection_changed(FakeEvent())

        assert len(called_with) == 1
        assert any("ST01" in p for p in next(iter(called_with)))

    def test_plot_tab_dispatches_via_panel_method(self, app_class, fake_tf):
        """The Plots-tab generate button uses .panel() when available."""
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

        # Verify the dispatcher protocol: plot object must expose .panel()
        plotter = PlotPenetrationDepth1D(fake_tf, show_plot=False)
        plotter.plot()

        assert hasattr(plotter, "panel") and callable(plotter.panel)
        result = plotter.panel()

        import panel as pn

        assert isinstance(result, pn.viewable.Viewable)
