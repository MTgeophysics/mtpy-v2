# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPhaseTensorPseudoSection MTData compatibility."""

from collections import deque

import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_phase_tensor_pseudosection_class():
    """Import Bokeh PlotPhaseTensorPseudoSection or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPhaseTensorPseudoSection

    return PlotPhaseTensorPseudoSection


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

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotPhaseTensorPseudoSectionBokehMTData:
    @staticmethod
    def _find_widget_by_name(root, widget_name):
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if getattr(node, "name", None) == widget_name:
                return node
            for child in getattr(node, "objects", []) or []:
                queue.append(child)
        return None

    def test_iter_mt_objects_from_tree(
        self, bokeh_plot_phase_tensor_pseudosection_class, mt_data_tree
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "pt") for mt in out)

    def test_rotation_angle_uses_tree_rotate(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)

        plotter.rotation_angle = 12.0

        assert called["value"] == 12.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 12.0

    def test_plot_uses_tree_objects(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        call_count = {"n": 0}

        def _fake_get_patch(tf):
            index = call_count["n"]
            call_count["n"] += 1
            return float(index + 1), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_colorbar", lambda: None)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert call_count["n"] == 2
        assert plotter.station_list.shape[0] == 2

    def test_plot_honors_aspect_kwarg(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree,
            show_plot=False,
            aspect="auto",
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        def _fake_get_patch(tf):
            return float(len(tf.station)), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_colorbar", lambda: None)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot(show=False)

        assert plotter.fig.match_aspect is False

    def test_plot_defaults_to_equal_aspect(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree,
            show_plot=False,
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        def _fake_get_patch(tf):
            return float(len(tf.station)), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot(show=False)

        assert plotter.aspect == "equal"
        assert plotter.fig.match_aspect is True

    def test_constructor_default_scaling_values(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree,
            show_plot=False,
        )

        assert plotter.ellipse_size == 20
        assert plotter.x_stretch == 1000
        assert plotter.y_stretch == 10

    def test_plot_pads_explicit_x_limits_by_half_ellipse_size(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree,
            show_plot=False,
            x_limits=(0.0, 1.0),
            ellipse_size=20,
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        def _fake_get_patch(tf):
            return float(len(tf.station)), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot(show=False)

        assert plotter.fig.x_range.start == pytest.approx(-10.0)
        assert plotter.fig.x_range.end == pytest.approx(11.0)

    def test_panel_returns_embeddable_panel_layout(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )

        monkeypatch.setattr(plotter, "plot", lambda show=False: None)

        panel_view = plotter.panel()

        assert isinstance(panel_view, pn.Column)
        assert len(panel_view.objects) == 3

    def test_panel_full_refresh_runs_without_missing_ellipse_alpha(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        pytest.importorskip("panel")
        import panel as pn

        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )
        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)

        panel_view = plotter.panel()

        assert isinstance(panel_view, pn.Column)
        assert hasattr(plotter, "ellipse_alpha")

    def test_panel_ellipse_size_widget_limits(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        pytest.importorskip("panel")

        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree, show_plot=False
        )
        monkeypatch.setattr(plotter, "plot", lambda show=False: None)

        panel_view = plotter.panel()
        ellipse_size_widget = self._find_widget_by_name(panel_view, "Ellipse size")

        assert ellipse_size_widget is not None
        assert ellipse_size_widget.start == 1
        assert ellipse_size_widget.end == 1000

    def test_plot_uses_plain_text_bokeh_labels(
        self,
        bokeh_plot_phase_tensor_pseudosection_class,
        mt_data_tree,
        monkeypatch,
    ):
        from bokeh.models import ColorBar

        plotter = bokeh_plot_phase_tensor_pseudosection_class(
            mt_data_tree,
            show_plot=False,
        )
        plotter.plot_tipper = "n"
        plotter.plot_pt = False

        def _fake_get_patch(tf):
            return float(len(tf.station)), tf.station[:5]

        monkeypatch.setattr(plotter, "_get_profile_line", lambda *args, **kwargs: None)
        monkeypatch.setattr(plotter, "_get_patch", _fake_get_patch)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        plotter.plot(show=False)

        y_labels = list(plotter.fig.yaxis[0].major_label_overrides.values())
        assert y_labels
        assert all("$" not in label for label in y_labels)

        colorbars = [obj for obj in plotter.fig.right if isinstance(obj, ColorBar)]
        assert colorbars
        assert "$" not in colorbars[0].title
