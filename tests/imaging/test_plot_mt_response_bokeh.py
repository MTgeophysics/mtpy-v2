# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotMTResponse behavior."""

import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_mt_response_class():
    """Import Bokeh PlotMTResponse or skip if Bokeh is unavailable."""
    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotMTResponse

    return PlotMTResponse


@pytest.fixture(scope="session")
def mt_object_bokeh():
    """Load a representative MT object once for bokeh plotting tests."""
    mt_obj = MT(TF_EDI_CGG)
    mt_obj.read()
    return mt_obj


def test_plot_tipper_arrows_with_default_colors_does_not_error(
    bokeh_plot_mt_response_class, mt_object_bokeh
):
    """Regression test for Bokeh rejecting matplotlib shorthand color values."""
    plotter = bokeh_plot_mt_response_class(
        z_object=mt_object_bokeh.Z.copy(),
        t_object=mt_object_bokeh.Tipper.copy(),
        pt_obj=mt_object_bokeh.pt.copy(),
        station=mt_object_bokeh.station,
        show_plot=False,
        plot_num=2,
    )

    layout = plotter.plot()

    assert layout is not None
    assert "tip_real" in plotter.renderers
    assert "tip_imag" in plotter.renderers


def test_tuple_to_hex_handles_matplotlib_short_color(bokeh_plot_mt_response_class):
    """Bokeh needs CSS-like color values, so 'k' must map to black hex."""
    assert bokeh_plot_mt_response_class._tuple_to_hex("k") == "#000000"
