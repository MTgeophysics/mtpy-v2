# -*- coding: utf-8 -*-
"""Tests for the Bokeh PlotStrike class."""

import numpy as np
import pandas as pd
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData


pytestmark = pytest.mark.plotting


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bokeh_plot_strike_class():
    """Import PlotStrike (Bokeh) or skip the whole module if Bokeh is absent."""
    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots.plot_strike import PlotStrike

    return PlotStrike


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create a single MT object for the whole test session."""
    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    """MTData container with two stations."""
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


# ---------------------------------------------------------------------------
# Data-processing tests (mirrored from test_plot_strike.py)
# ---------------------------------------------------------------------------


class TestPlotStrikeBokehData:
    """Tests for the data-building methods shared with the matplotlib version."""

    def test_iter_mt_objects_returns_two_stations(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "period") for mt in out)

    def test_rotation_angle_delegates_to_tree_rotate(
        self, bokeh_plot_strike_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)
        monkeypatch.setattr(plotter, "make_strike_df", lambda: None)

        plotter.rotation_angle = 15.0

        assert called["value"] == 15.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 15.0

    def test_make_strike_df_produces_expected_columns(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        plotter.make_strike_df()

        assert isinstance(plotter.strike_df, pd.DataFrame)
        assert not plotter.strike_df.empty
        required = {"estimate", "period", "plot_strike", "measured_strike"}
        assert required.issubset(set(plotter.strike_df.columns))

    def test_make_strike_df_contains_all_estimate_types(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()

        estimates = set(plotter.strike_df["estimate"].unique())
        # Impedance data must produce both invariant and pt estimates
        assert "invariant" in estimates
        assert "pt" in estimates

    def test_get_estimate_filters_by_type(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()

        inv_df = plotter.get_estimate("invariant")

        assert (inv_df["estimate"] == "invariant").all()
        assert len(inv_df) > 0

    def test_get_estimate_filters_by_period_range(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()

        p_min = plotter.strike_df["period"].min()
        p_max = plotter.strike_df["period"].max()
        mid = (p_min + p_max) / 2

        filtered = plotter.get_estimate("invariant", (p_min, mid))

        assert (filtered["period"] >= p_min).all()
        assert (filtered["period"] < mid).all()

    def test_get_stats_returns_three_floats(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()

        median, mode, mean = plotter.get_stats("invariant")

        assert isinstance(median, float)
        assert isinstance(mode, float)
        assert isinstance(mean, float)
        assert 0 <= median <= 360
        assert 0 <= mode <= 360
        assert 0 <= mean <= 360

    def test_get_mean_in_range(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()
        inv_df = plotter.get_estimate("invariant")

        val = plotter.get_mean(inv_df)

        assert 0 <= val <= 360

    def test_get_median_in_range(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()
        inv_df = plotter.get_estimate("invariant")

        val = plotter.get_median(inv_df)

        assert 0 <= val <= 360

    def test_get_mode_in_range(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        plotter.make_strike_df()
        inv_df = plotter.get_estimate("invariant")

        val = plotter.get_mode(inv_df)

        assert 0 <= val <= 360

    def test_get_plot_array_no_fold(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False, fold=False)
        plotter.make_strike_df()

        arr = plotter.get_plot_array("invariant")

        assert arr.ndim == 1
        assert np.all((arr >= 0) & (arr <= 360))

    def test_get_plot_array_with_fold(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False, fold=True)
        plotter.make_strike_df()

        arr = plotter.get_plot_array("invariant")

        assert np.all((arr >= 0) & (arr <= 180))

    def test_get_plot_array_with_orthogonal(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(
            mt_data_tree, show_plot=False, plot_orthogonal=False
        )
        plotter.make_strike_df()
        arr_normal = plotter.get_plot_array("invariant")

        plotter.plot_orthogonal = True
        arr_ortho = plotter.get_plot_array("invariant")

        # Orthogonal includes extra directions → longer array
        assert len(arr_ortho) > len(arr_normal)


# ---------------------------------------------------------------------------
# Bokeh-specific plot tests
# ---------------------------------------------------------------------------


class TestPlotStrikeBokehPlot:
    """Tests for the Bokeh plotting outputs."""

    def test_plot_type2_returns_layout(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False, plot_type=2)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert plotter.layout is layout
        assert plotter.fig is layout

    def test_plot_type2_figures_keys(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=2,
            plot_invariant=True,
            plot_pt=True,
            plot_tipper=True,
        )
        plotter.plot(show=False)

        # All three enabled estimates should appear
        assert "invariant" in plotter.figures
        assert "pt" in plotter.figures
        assert "tipper" in plotter.figures

    def test_plot_type2_only_invariant(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=2,
            plot_invariant=True,
            plot_pt=False,
            plot_tipper=False,
        )
        plotter.plot(show=False)

        assert set(plotter.figures.keys()) == {"invariant"}

    def test_plot_type2_vertical_orientation(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=2,
            plot_orientation="v",
        )
        layout = plotter.plot(show=False)

        assert layout is not None

    def test_plot_type1_returns_layout(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False, plot_type=1)

        layout = plotter.plot(show=False)

        assert layout is not None

    def test_plot_type1_figures_have_per_period_keys(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=1,
            plot_invariant=True,
            plot_pt=False,
            plot_tipper=False,
        )
        plotter.plot(show=False)

        # Keys are "invariant_<decade_int>"
        keys = list(plotter.figures.keys())
        assert all(k.startswith("invariant_") for k in keys)
        assert len(keys) >= 1

    def test_plot_type1_titles_use_superscript_decades(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=1,
            plot_invariant=True,
            plot_pt=False,
            plot_tipper=False,
        )
        plotter.plot(show=False)

        first_key = sorted(plotter.figures.keys())[0]
        title_text = plotter.figures[first_key].title.text
        assert "10" in title_text
        assert any(ch in title_text for ch in ["\u207b", "\u00b9", "\u00b2", "\u00b3"])

    def test_figures_are_bokeh_figure_instances(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        from bokeh.plotting import figure as bokeh_figure

        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=2,
        )
        plotter.plot(show=False)

        for fig in plotter.figures.values():
            assert isinstance(fig, bokeh_figure)

    def test_rose_figure_has_no_visible_axes(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False, plot_type=2)
        plotter.plot(show=False)

        fig = plotter.figures["invariant"]
        assert not fig.xaxis.visible
        assert not fig.yaxis.visible

    def test_fold_true_changes_histogram_range(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        p_fold = bokeh_plot_strike_class(mt_data_tree, show_plot=False, fold=True)
        p_nofold = bokeh_plot_strike_class(mt_data_tree, show_plot=False, fold=False)

        assert p_fold._get_histogram_range() == (0, 180)
        assert p_nofold._get_histogram_range() == (0, 360)

    def test_ring_limits_respected(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            plot_type=2,
            ring_limits=(0, 50),
        )

        max_count = plotter._get_max_count("invariant")

        assert max_count == 50

    def test_constructor_kwargs_are_applied(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        plotter = bokeh_plot_strike_class(
            mt_data_tree,
            show_plot=False,
            fold=True,
            bin_width=10,
            plot_orthogonal=True,
            color=False,
        )

        assert plotter.fold is True
        assert plotter.bin_width == 10
        assert plotter.plot_orthogonal is True
        assert plotter.color is False

    def test_require_bokeh_raises_when_unavailable(
        self, bokeh_plot_strike_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        import mtpy.imaging.bokeh_plots.plot_strike as ps_module

        monkeypatch.setattr(ps_module, "figure", None)
        monkeypatch.setattr(ps_module, "ColumnDataSource", None)
        monkeypatch.setattr(ps_module, "Range1d", None)

        with pytest.raises(ImportError, match="Bokeh is required"):
            plotter._require_bokeh()

    def test_rgb_to_hex_conversion(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        assert plotter._rgb_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
        assert plotter._rgb_to_hex((0.0, 1.0, 0.0)) == "#00ff00"
        assert plotter._rgb_to_hex((0.0, 0.0, 1.0)) == "#0000ff"
        assert plotter._rgb_to_hex((0.0, 0.0, 0.0)) == "#000000"


class TestPlotStrikeBokehPanel:
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
                found.extend(TestPlotStrikeBokehPanel._find_widgets(child, widget_type))
        return found

    def test_panel_returns_column(self, bokeh_plot_strike_class, mt_data_tree):
        import panel as pn

        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        assert isinstance(panel_obj, pn.Column)
        assert len(panel_obj) >= 3

    def test_panel_contains_expected_controls(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        import panel as pn

        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        select_names = {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.Select)
        }
        text_input_names = {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.TextInput)
        }
        checkbox_names = {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.Checkbox)
        }
        color_picker_names = {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.ColorPicker)
        }
        check_button_names = {
            w.name for w in self._find_widgets(panel_obj, pn.widgets.CheckButtonGroup)
        }

        assert "Plot Mode" in select_names
        assert "Rotation Angle" in text_input_names
        assert "Estimates" in check_button_names
        assert "Use Dynamic Colors" in checkbox_names
        assert "Invariant Color" in color_picker_names
        assert "Phase Tensor Color" in color_picker_names
        assert "Tipper Color" in color_picker_names

    def test_panel_refresh_updates_parameters(
        self, bokeh_plot_strike_class, mt_data_tree
    ):
        import panel as pn

        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)
        panel_obj = plotter.panel()

        mode_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Select)
            if w.name == "Plot Mode"
        ][0]
        rotation_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.TextInput)
            if w.name == "Rotation Angle"
        ][0]
        estimates_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.CheckButtonGroup)
            if w.name == "Estimates"
        ][0]
        dynamic_color_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Checkbox)
            if w.name == "Use Dynamic Colors"
        ][0]
        inv_color_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.ColorPicker)
            if w.name == "Invariant Color"
        ][0]
        pt_color_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.ColorPicker)
            if w.name == "Phase Tensor Color"
        ][0]
        tip_color_widget = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.ColorPicker)
            if w.name == "Tipper Color"
        ][0]
        refresh_btn = [
            w
            for w in self._find_widgets(panel_obj, pn.widgets.Button)
            if "Refresh" in (w.name or "")
        ][0]

        mode_widget.value = "By Period"
        rotation_widget.value = "15.0"
        estimates_widget.value = ["Invariant", "Tipper"]
        dynamic_color_widget.value = False
        inv_color_widget.value = "#ff0000"
        pt_color_widget.value = "#0000ff"
        tip_color_widget.value = "#00ff00"

        refresh_btn.clicks = refresh_btn.clicks + 1

        assert plotter.rotation_angle == pytest.approx(15.0)
        assert plotter.plot_type == 1
        assert plotter.plot_invariant is True
        assert plotter.plot_pt is False
        assert plotter.plot_tipper is True
        assert plotter.color is False
        assert plotter.color_inv == pytest.approx((1.0, 0.0, 0.0))
        assert plotter.color_pt == pytest.approx((0.0, 0.0, 1.0))
        assert plotter.color_tip == pytest.approx((0.0, 1.0, 0.0))

    def test_servable_returns_viewable(self, bokeh_plot_strike_class, mt_data_tree):
        plotter = bokeh_plot_strike_class(mt_data_tree, show_plot=False)

        viewable = plotter.servable(title="Strike")

        assert viewable is not None


# ---------------------------------------------------------------------------
# Import alias test
# ---------------------------------------------------------------------------


class TestPlotStrikeBokehImport:
    def test_top_level_alias_import(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging import PlotStrikeBokeh

        assert PlotStrikeBokeh is not None

    def test_bokeh_subpackage_import(self):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots import PlotStrike

        assert PlotStrike is not None
