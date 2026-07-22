# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPhaseTensor functionality."""

import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_pt_class():
    """Import Bokeh PlotPhaseTensor or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPhaseTensor

    return PlotPhaseTensor


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for test session."""

    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def pt_object(mt_object_cache):
    """Get a phase tensor object from a copied MT object."""

    mt = mt_object_cache.copy()
    return mt.Z.phase_tensor


class TestPlotPhaseTensorBokeh:
    def test_inherits_bokeh_plot_base(self, bokeh_plot_pt_class):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        assert issubclass(bokeh_plot_pt_class, BokehPlotBase)

    def test_does_not_inherit_plot_base(self, bokeh_plot_pt_class):
        from mtpy.imaging.mtplot_tools import PlotBase

        assert not issubclass(bokeh_plot_pt_class, PlotBase)

    def test_rotation_angle_rotates_pt(
        self, bokeh_plot_pt_class, pt_object, monkeypatch
    ):
        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(plotter.pt, "rotate", _fake_rotate)

        plotter.rotation_angle = 22.5

        assert called["value"] == 22.5
        assert called["inplace"] is True
        assert plotter.rotation_angle == 22.5

    def test_plot_rotation_angle_kwarg_uses_rotate_pt(
        self, bokeh_plot_pt_class, pt_object, monkeypatch
    ):
        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)

        called = {"value": None}

        def _fake_rotate_pt(value):
            called["value"] = value

        monkeypatch.setattr(plotter, "_rotate_pt", _fake_rotate_pt)

        layout = plotter.plot(rotation_angle=10.0, show=False)

        assert layout is not None
        assert called["value"] == 10.0

    def test_plot_builds_expected_figures(self, bokeh_plot_pt_class, pt_object):
        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert set(plotter.figures.keys()) == {"pt", "phase", "skew", "strike"}
        assert plotter.figures["pt"].xaxis[0].axis_label == "Period (s)"
        assert plotter.figures["strike"].yaxis[0].axis_label == "Strike (deg)"

    def test_constructor_honors_kwargs(self, bokeh_plot_pt_class, pt_object):
        plotter = bokeh_plot_pt_class(
            pt_object,
            station="TEST01",
            show_plot=False,
            skew_cutoff=5,
            ellip_cutoff=0.2,
            ellipse_spacing=7,
        )

        assert plotter.skew_cutoff == 5
        assert plotter.ellip_cutoff == 0.2
        assert plotter.ellipse_spacing == 7

    def test_top_level_alias_import(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging import PlotPhaseTensorBokeh

        assert PlotPhaseTensorBokeh is not None


class TestPlotPhaseTensorPanel:
    @pytest.fixture(autouse=True)
    def require_panel(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")

    def test_panel_returns_column(self, bokeh_plot_pt_class, pt_object):
        import panel as pn

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        assert isinstance(panel_obj, pn.Column)

    def test_panel_exposes_colorby_and_palette_controls(
        self, bokeh_plot_pt_class, pt_object
    ):
        import panel as pn

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        colorby_widgets = [
            w
            for w in panel_obj.select(pn.widgets.Select)
            if w.name == "Phase Tensor Color By"
        ]
        palette_widgets = [
            w
            for w in panel_obj.select(pn.widgets.Select)
            if w.name == "Ellipse Palette"
        ]

        assert len(colorby_widgets) == 1
        assert len(palette_widgets) == 1

        colorby_widget = colorby_widgets[0]
        palette_widget = palette_widgets[0]

        colorby_widget.value = "strike"
        palette_widget.value = "magma"

        assert plotter.ellipse_colorby == "strike"
        assert str(plotter.ellipse_cmap).lower() == "magma"

    def test_panel_exposes_marker_size_control(self, bokeh_plot_pt_class, pt_object):
        import panel as pn

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        marker_widgets = [
            w for w in panel_obj.select(pn.widgets.IntSlider) if w.name == "Marker Size"
        ]
        assert len(marker_widgets) == 1

        marker_widget = marker_widgets[0]
        marker_widget.value = 11

        assert int(plotter.marker_size) == 11

    def test_panel_colorby_change_updates_min_max_from_array(
        self, bokeh_plot_pt_class, pt_object
    ):
        import panel as pn

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        colorby_widget = [
            w
            for w in panel_obj.select(pn.widgets.Select)
            if w.name == "Phase Tensor Color By"
        ][0]
        min_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Min"
        ][0]
        max_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Max"
        ][0]

        colorby_widget.value = "ellipticity"

        expected = np.asarray(plotter.get_pt_color_array(plotter.pt), dtype=float)
        expected = expected[np.isfinite(expected)]
        assert expected.size > 0

        assert float(min_widget.value) == pytest.approx(float(np.nanmin(expected)))
        assert float(max_widget.value) == pytest.approx(float(np.nanmax(expected)))

    def test_panel_manual_min_max_updates_ellipse_range(
        self, bokeh_plot_pt_class, pt_object
    ):
        pass

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        min_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Min"
        ][0]
        max_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Max"
        ][0]

        min_widget.value = -5.0
        max_widget.value = 25.0

        assert float(plotter.ellipse_range[0]) == pytest.approx(-5.0)
        assert float(plotter.ellipse_range[1]) == pytest.approx(25.0)

    def test_panel_min_max_safeguard_shows_warning(
        self, bokeh_plot_pt_class, pt_object
    ):
        import panel as pn

        plotter = bokeh_plot_pt_class(pt_object, station="TEST01", show_plot=False)
        panel_obj = plotter.panel()

        min_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Min"
        ][0]
        max_widget = [
            w for w in panel_obj.select() if getattr(w, "name", "") == "Ellipse Max"
        ][0]

        min_widget.value = 10.0
        max_widget.value = 5.0

        # Max is auto-corrected and a warning message is shown to the user.
        assert float(max_widget.value) == pytest.approx(11.0)
        warning_panes = [
            p
            for p in panel_obj.select(pn.pane.Markdown)
            if "Ellipse Max must be greater than Ellipse Min"
            in str(getattr(p, "object", ""))
        ]
        assert len(warning_panes) == 1
