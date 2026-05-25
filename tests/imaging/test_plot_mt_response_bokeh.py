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


class TestPlotMTResponseBase:
    """Tests verifying PlotMTResponse inherits from BokehPlotBase."""

    def test_inherits_bokeh_plot_base(self, bokeh_plot_mt_response_class):
        from mtpy.imaging.bokeh_plots.bokeh_plot_base import BokehPlotBase

        assert issubclass(bokeh_plot_mt_response_class, BokehPlotBase)

    def test_does_not_inherit_plot_base(self, bokeh_plot_mt_response_class):
        from mtpy.imaging.mtplot_tools import PlotBase

        assert not issubclass(bokeh_plot_mt_response_class, PlotBase)

    def test_default_params_accessible(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        assert plotter.lw == 1.0
        assert plotter.plot_z is True
        assert plotter.plot_tipper == "n" or plotter.plot_tipper == "yri"
        assert plotter.ellipse_size == 2.0
        assert plotter.arrow_direction == 1

    def test_kwargs_override_params(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
            lw=2.5,
            xy_color="#ff0000",
        )
        assert plotter.lw == 2.5
        assert plotter.xy_color == "#ff0000"

    def test_has_logger(self, bokeh_plot_mt_response_class, mt_object_bokeh):
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        assert hasattr(plotter, "logger")

    def test_set_period_limits(self, bokeh_plot_mt_response_class, mt_object_bokeh):
        pass

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        period = mt_object_bokeh.Z.period
        limits = plotter.set_period_limits(period)
        assert len(limits) == 2
        assert limits[0] < limits[1]
        assert limits[0] > 0


class TestPlotMTResponsePanel:
    """Tests for PlotMTResponse.panel() interactive app."""

    @pytest.fixture(autouse=True)
    def require_panel(self):
        pytest.importorskip("panel")
        pytest.importorskip("bokeh")

    def test_panel_method_exists(self, bokeh_plot_mt_response_class):
        assert hasattr(bokeh_plot_mt_response_class, "panel")
        assert callable(bokeh_plot_mt_response_class.panel)

    def test_make_panel_renamed_to_panel(self, bokeh_plot_mt_response_class):
        """make_panel() was renamed to panel() for MTDataApp compatibility."""
        assert not hasattr(bokeh_plot_mt_response_class, "make_panel")

    def test_panel_returns_column(self, bokeh_plot_mt_response_class, mt_object_bokeh):
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            t_object=mt_object_bokeh.Tipper.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        result = plotter.panel()
        assert isinstance(result, pn.Column)

    def test_panel_contains_controls(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        result = plotter.panel()
        assert len(result) >= 2
        # Controls row should be present
        controls = result[1]
        assert isinstance(controls, pn.Row)

    def test_panel_non_interactive_mode(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        result = plotter.panel(interactive=False)
        assert isinstance(result, pn.Column)

    def test_panel_uses_station_as_title_when_plot_title_empty(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        import panel as pn

        station = mt_object_bokeh.station
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=station,
            show_plot=False,
        )
        result = plotter.panel()
        # First child should be a Markdown with station name
        first = result[0]
        assert isinstance(first, pn.pane.Markdown)
        assert station in first.object

    def test_panel_uses_plot_title_when_set(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        pass

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
            plot_title="Custom Title",
        )
        result = plotter.panel()
        first = result[0]
        assert "Custom Title" in first.object
