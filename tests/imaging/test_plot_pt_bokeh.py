# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPhaseTensor functionality."""

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
