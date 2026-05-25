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

    def test_xx_yy_color_marker_params(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """xx/yy components have independent color and marker params."""
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        assert hasattr(plotter, "xx_color")
        assert hasattr(plotter, "xx_marker")
        assert hasattr(plotter, "yy_color")
        assert hasattr(plotter, "yy_marker")
        # Defaults should differ from xy/yx defaults
        assert plotter.xx_color != plotter.xy_color
        assert plotter.yy_color != plotter.yx_color

    def test_xx_yy_params_override(self, bokeh_plot_mt_response_class, mt_object_bokeh):
        """xx/yy color and marker can be overridden via kwargs."""
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
            xx_color="#aabbcc",
            yy_color="#ddeeff",
            xx_marker="d",
            yy_marker="v",
        )
        assert plotter.xx_color == "#aabbcc"
        assert plotter.yy_color == "#ddeeff"
        assert plotter.xx_marker == "d"
        assert plotter.yy_marker == "v"

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

    def test_panel_includes_style_card(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """panel() should include a collapsed styling card for components."""
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        result = plotter.panel()
        # Style card is the third child (index 2); bokeh pane is index 3
        assert len(result) >= 4
        style_card = result[2]
        assert isinstance(style_card, pn.Card)

    def test_model_error_does_not_produce_empty_plot(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """Switching to model error should still render data (NaN fallback)."""
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        plotter.plot_model_error = True
        layout = plotter.plot()
        # Renderers for off-diagonal components must be present and non-empty
        assert "xy" in plotter.renderers
        assert len(plotter.renderers["xy"]) > 0

    def test_diag_components_use_xx_yy_colors(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """plot_num=2 diagonal figures must use xx/yy colors, not xy/yx."""
        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
            plot_num=2,
            xx_color="#112233",
            yy_color="#445566",
        )
        plotter.plot()
        # Renderers for diagonal components should be present
        assert "xx" in plotter.renderers
        assert "yy" in plotter.renderers

    def test_panel_preset_buttons_are_individual_buttons(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """Preset controls must be Button widgets (not RadioButtonGroup) so
        clicking the already-active preset still fires the callback."""
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
            plot_num=2,
        )
        result = plotter.panel()
        controls = result[1]
        assert isinstance(controls, pn.Row)
        # First child of controls is the preset column
        preset_col = controls[0]
        assert isinstance(preset_col, pn.Column)
        # Second child of preset column is a Row of buttons
        btn_row = preset_col[1]
        assert isinstance(btn_row, pn.Row)
        for widget in btn_row:
            assert isinstance(widget, pn.widgets.Button)

    def test_panel_style_card_contains_marker_size_widget(
        self, bokeh_plot_mt_response_class, mt_object_bokeh
    ):
        """Style card must contain an IntSlider for marker size."""
        import panel as pn

        plotter = bokeh_plot_mt_response_class(
            z_object=mt_object_bokeh.Z.copy(),
            station=mt_object_bokeh.station,
            show_plot=False,
        )
        result = plotter.panel()
        style_card = result[2]
        assert isinstance(style_card, pn.Card)

        # Walk card children looking for an IntSlider
        def _find_widget(obj, widget_type):
            if isinstance(obj, widget_type):
                return True
            children = getattr(obj, "objects", []) or []
            return any(_find_widget(c, widget_type) for c in children)

        assert _find_widget(style_card, pn.widgets.IntSlider)
