# -*- coding: utf-8 -*-
"""Tests for Bokeh PlotPhaseTensorMaps container compatibility."""

import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_phase_tensor_maps_class():
    """Import Bokeh PlotPhaseTensorMaps or skip if Bokeh is unavailable."""

    pytest.importorskip("bokeh")
    from mtpy.imaging.bokeh_plots import PlotPhaseTensorMaps

    return PlotPhaseTensorMaps


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for the test session."""

    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    """Build an MTData with two stations."""

    mt_1 = mt_object_cache.copy()
    mt_1.station = "TEST01"

    mt_2 = mt_object_cache.copy()
    mt_2.station = "TEST02"
    if mt_2.longitude is not None:
        mt_2.longitude = float(mt_2.longitude) + 0.02

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


class TestPlotPhaseTensorMapsBokehMTData:
    def test_iter_mt_objects_from_tree(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

        out = list(plotter._iter_mt_objects())

        assert len(out) == 2
        assert all(hasattr(mt, "station") for mt in out)
        assert all(hasattr(mt, "Z") for mt in out)

    def test_rotation_angle_uses_tree_rotate(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

        called = {"value": None, "inplace": None}

        def _fake_rotate(value, inplace=True):
            called["value"] = value
            called["inplace"] = inplace

        monkeypatch.setattr(mt_data_tree, "rotate", _fake_rotate)

        plotter.rotation_angle = 15.0

        assert called["value"] == 15.0
        assert called["inplace"] is True
        assert plotter.rotation_angle == 15.0

    def test_plot_uses_tree_objects(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        plotter.plot_pt = False
        plotter.plot_tipper = "n"
        plotter.pt_type = "wedges"
        plotter.plot_station = False

        call_count = {"n": 0}

        def _fake_get_patch_wedges(_tf):
            index = call_count["n"]
            call_count["n"] += 1
            return float(index + 1), float(index + 2)

        monkeypatch.setattr(plotter, "_get_patch_wedges", _fake_get_patch_wedges)
        monkeypatch.setattr(plotter, "_add_tipper_legend", lambda: None)

        layout = plotter.plot(show=False)

        assert layout is not None
        assert call_count["n"] == 2
        assert plotter.plot_xarr.shape[0] == 2
        assert plotter.plot_yarr.shape[0] == 2

    def test_get_mt_objects_preinterpolates_once_per_period(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)

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

    def test_get_interpolated_z_uses_direct_period_sample(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        tf = plotter._get_mt_objects()[0]

        plotter.plot_period = float(tf.period[0])

        def _raise_interp(*args, **kwargs):
            raise AssertionError("interp1d path should not be used for exact periods")

        monkeypatch.setattr(plotter, "get_interp1d_functions_z", _raise_interp)

        z = plotter._get_interpolated_z(tf)

        assert z.shape == (1, 2, 2)

    @pytest.mark.parametrize("plot_period", [1.0, 2.0])
    @pytest.mark.parametrize("has_measurement_error", [False, True])
    def test_plot_ellipses_with_preinterpolated_model_errors(
        self, bokeh_plot_phase_tensor_maps_class, plot_period, has_measurement_error
    ):
        from bokeh.models import Ellipse

        impedance = np.array(
            [
                [
                    [0.158 - 1.347j, 3.621 + 3.651j],
                    [-11.553 - 2.123j, -1.427 + 1.871j],
                ],
                [
                    [0.158 - 1.347j, 3.621 + 3.651j],
                    [-11.553 - 2.123j, -1.427 + 1.871j],
                ],
            ]
        )
        mt = MT(station="TEST01", survey="data", period=np.array([1.0, 3.0]))
        mt.latitude = 38.834
        mt.longitude = -118.236
        mt.impedance = impedance
        mt.impedance_model_error = np.full((2, 2, 2), 0.4)
        if has_measurement_error:
            mt.impedance_error = np.full((2, 2, 2), 0.2)
        tree = MTData()
        tree.add_station(mt)
        plotter = bokeh_plot_phase_tensor_maps_class(
            tree, show_plot=False, plot_period=plot_period
        )

        tf = plotter._get_mt_objects()[0]
        assert tf.period.tolist() == [plot_period]
        assert tf.Z._has_tf_model_error()
        assert tf.Z._has_tf_error() == has_measurement_error

        plotter.plot(show=False)

        ellipses = [
            renderer
            for renderer in plotter.fig.renderers
            if isinstance(getattr(renderer, "glyph", None), Ellipse)
        ]
        assert len(ellipses) == 1
        assert plotter.plot_xarr[0] == pytest.approx(mt.longitude)
        assert plotter.plot_yarr[0] == pytest.approx(mt.latitude)

    @pytest.mark.parametrize("plot_period", [1.0, 2.0])
    @pytest.mark.parametrize("has_measurement_error", [False, True])
    @pytest.mark.parametrize("plot_pt", [False, True])
    def test_plot_tipper_with_preinterpolated_model_errors(
        self,
        bokeh_plot_phase_tensor_maps_class,
        plot_period,
        has_measurement_error,
        plot_pt,
    ):
        from bokeh.models import Arrow, Ellipse

        mt = MT(station="TEST01", survey="data", period=np.array([1.0, 3.0]))
        mt.latitude = 38.834
        mt.longitude = -118.236
        mt.impedance = np.tile(
            np.array(
                [
                    [0.158 - 1.347j, 3.621 + 3.651j],
                    [-11.553 - 2.123j, -1.427 + 1.871j],
                ]
            ),
            (2, 1, 1),
        )
        mt.tipper = np.tile(np.array([[0.2 - 0.1j, 0.36 - 0.09j]]), (2, 1, 1))
        mt.tipper_model_error = np.full((2, 1, 2), 0.02)
        if has_measurement_error:
            mt.tipper_error = np.full((2, 1, 2), 0.01)
        tree = MTData()
        tree.add_station(mt)
        plotter = bokeh_plot_phase_tensor_maps_class(
            tree,
            show_plot=False,
            plot_period=plot_period,
            plot_pt=plot_pt,
            plot_tipper="yri",
        )

        tf = plotter._get_mt_objects()[0]
        assert tf.period.tolist() == [plot_period]
        assert tf.Tipper._has_tf_model_error()
        assert tf.Tipper._has_tf_error() == has_measurement_error
        _, tipper = plotter._get_pt(tf)
        assert tipper is not None
        np.testing.assert_allclose(tipper.frequency, [1.0 / plot_period])

        plotter.plot(show=False)

        arrows = list(plotter.fig.select({"type": Arrow}))
        ellipses = [
            renderer
            for renderer in plotter.fig.renderers
            if isinstance(getattr(renderer, "glyph", None), Ellipse)
        ]
        assert len(arrows) == 2
        assert len(ellipses) == int(plot_pt)
        assert plotter.plot_xarr[0] == pytest.approx(mt.longitude)
        assert plotter.plot_yarr[0] == pytest.approx(mt.latitude)

    def test_map_dataframe_is_reused_and_invalidated_for_changed_data(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree, monkeypatch
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        calls = []
        prepare = mt_data_tree.to_phase_tensor_map_dataframe

        def wrapped(periods):
            calls.append(periods.copy())
            return prepare(periods)

        monkeypatch.setattr(mt_data_tree, "to_phase_tensor_map_dataframe", wrapped)
        monkeypatch.setattr(
            plotter,
            "_get_mt_objects",
            lambda: pytest.fail("MTData map should use the prepared dataframe"),
        )
        plotter.plot(show=False)
        assert len(calls) == 1
        plotter.plot_period = float(plotter._map_periods[1])
        plotter.plot(show=False)
        assert len(calls) == 1

        station = mt_data_tree.get_station(
            list(mt_data_tree._iter_station_paths())[0], as_mt=True
        )
        station.impedance = station.impedance.values * (1 + 1j)
        mt_data_tree.add_station(station, overwrite=True)
        plotter.plot(show=False)
        assert len(calls) == 2

    def test_panel_period_selector_uses_prepared_periods(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree
    ):
        import panel as pn

        plotter = bokeh_plot_phase_tensor_maps_class(mt_data_tree, show_plot=False)
        app = plotter.panel()
        selectors = [
            widget
            for widget in app.select(pn.widgets.Select)
            if widget.name == "Plot period (s)"
        ]
        assert len(selectors) == 1
        np.testing.assert_allclose(
            list(selectors[0].options.values()), plotter._map_periods
        )
        assert selectors[0].value == plotter.plot_period
        refresh = next(
            button
            for button in app.select(pn.widgets.Button)
            if "Refresh" in button.name
        )
        selectors[0].value = float(plotter._map_periods[0])
        refresh.clicks += 1
        assert any(
            "No phase tensor ellipses" in str(pane.object)
            for pane in app.select(pn.pane.Markdown)
        )
        previous_max = plotter._map_periods[-1]
        extra = MT(
            station="extra",
            survey="survey_a",
            period=np.array([previous_max * 10, previous_max * 100]),
        )
        extra.impedance = np.tile(
            np.array([[0.2 - 1j, 3 + 4j], [-10 - 2j, -1 + 2j]]), (2, 1, 1)
        )
        mt_data_tree.add_station(extra)
        refresh.clicks += 1
        assert plotter._map_periods[-1] > previous_max
        np.testing.assert_allclose(
            list(selectors[0].options.values()), plotter._map_periods
        )

    def test_map_supports_programmatic_off_grid_period(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(
            mt_data_tree, show_plot=False, plot_period=1.2345
        )
        plotter.plot(show=False)
        assert not np.any(plotter._map_periods == 1.2345)
        assert len(plotter._map_dataframe[plotter._map_dataframe.period == 1.2345]) == 2
        assert np.count_nonzero(plotter.plot_xarr) == 2

    def test_map_supports_positive_periods_below_previous_limit(
        self, bokeh_plot_phase_tensor_maps_class
    ):
        mt = MT(station="short", survey="data", period=np.array([1e-12, 1e-11]))
        mt.latitude = 38.8
        mt.longitude = -118.2
        mt.impedance = np.tile(
            np.array([[0.2 - 1j, 3 + 4j], [-10 - 2j, -1 + 2j]]), (2, 1, 1)
        )
        tree = MTData()
        tree.add_station(mt)
        plotter = bokeh_plot_phase_tensor_maps_class(
            tree, show_plot=False, plot_period=1e-12
        )
        plotter.panel()
        assert plotter.plot_period == pytest.approx(1e-12)
        assert np.count_nonzero(plotter.plot_xarr) == 1

    def test_map_dataframe_supports_projected_coordinates(
        self, bokeh_plot_phase_tensor_maps_class, mt_data_tree
    ):
        plotter = bokeh_plot_phase_tensor_maps_class(
            mt_data_tree, show_plot=False, map_scale="km", map_epsg=32611
        )
        plotter.plot(show=False)
        assert np.isfinite(plotter.plot_xarr).all()
        assert np.count_nonzero(plotter.plot_xarr) == 2

    @pytest.mark.integration
    def test_modem_example_renders_ellipses_and_arrows(
        self, bokeh_plot_phase_tensor_maps_class
    ):
        from pathlib import Path

        from bokeh.models import Arrow, Ellipse

        mtpy_data = pytest.importorskip("mtpy_data")
        filename = (
            Path(mtpy_data.__file__).parent
            / "data"
            / "grid"
            / "gv_modem_data_z05_t03.dat"
        )
        if not filename.is_file():
            pytest.skip("ModEM example file is not installed")
        tree = MTData()
        tree.from_modem(filename)
        plotter = bokeh_plot_phase_tensor_maps_class(
            tree, show_plot=False, plot_tipper="yri"
        )
        plotter.panel()
        assert plotter.plot_period == 1.0
        assert len(plotter._map_periods) == 25
        assert len(plotter._map_dataframe) == 59 * 25
        ellipses = [
            renderer
            for renderer in plotter.fig.renderers
            if isinstance(getattr(renderer, "glyph", None), Ellipse)
        ]
        assert len(ellipses) == 59
        assert len(list(plotter.fig.select({"type": Arrow}))) == 118
