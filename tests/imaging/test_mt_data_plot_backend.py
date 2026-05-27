# -*- coding: utf-8 -*-
"""Tests for the ``backend`` parameter on MTData plot methods."""

import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core import MTData


pytestmark = pytest.mark.plotting


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mt_object_cache():
    mt = MT(TF_EDI_CGG)
    mt.read()
    return mt


@pytest.fixture
def mt_data_tree(mt_object_cache):
    mt_1 = mt_object_cache.copy()
    mt_1.station = "BACKEND01"

    mt_2 = mt_object_cache.copy()
    mt_2.station = "BACKEND02"
    if mt_2.longitude is not None:
        mt_2.longitude = float(mt_2.longitude) + 0.02
    if mt_2.latitude is not None:
        mt_2.latitude = float(mt_2.latitude) + 0.01

    tree = MTData()
    tree.add_stations([mt_1, mt_2])
    return tree


@pytest.fixture
def single_station_key(mt_data_tree):
    return next(iter(mt_data_tree._iter_station_paths()))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _bokeh_class(module_path, class_name):
    """Import a Bokeh plotting class, skipping if Bokeh is absent."""
    pytest.importorskip("bokeh")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# plot_strike
# ---------------------------------------------------------------------------


class TestPlotStrikeBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_strike import PlotStrike as BokehPlotStrike

        result = mt_data_tree.plot_strike(backend="bokeh", show_plot=False)
        assert isinstance(result, BokehPlotStrike)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_strike import PlotStrike as MplPlotStrike

        result = mt_data_tree.plot_strike(backend="matplotlib", show_plot=False)
        assert isinstance(result, MplPlotStrike)

    def test_default_backend_is_bokeh(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_strike import PlotStrike as BokehPlotStrike

        result = mt_data_tree.plot_strike(show_plot=False)
        assert isinstance(result, BokehPlotStrike)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_strike(backend="plotly")


# ---------------------------------------------------------------------------
# plot_stations
# ---------------------------------------------------------------------------


class TestPlotStationsBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_stations import (
            PlotStations as BokehPlotStations,
        )

        result = mt_data_tree.plot_stations(backend="bokeh", show_plot=False)
        assert isinstance(result, BokehPlotStations)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_stations import PlotStations as MplPlotStations

        result = mt_data_tree.plot_stations(backend="matplotlib", show_plot=False)
        assert isinstance(result, MplPlotStations)

    def test_default_backend_is_bokeh(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_stations import (
            PlotStations as BokehPlotStations,
        )

        result = mt_data_tree.plot_stations(show_plot=False)
        assert isinstance(result, BokehPlotStations)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_stations(backend="plotly")


# ---------------------------------------------------------------------------
# plot_mt_response (multi-station)
# ---------------------------------------------------------------------------


class TestPlotMTResponseBackend:
    def test_multi_station_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_mt_responses import (
            PlotMultipleResponses as BokehMultiple,
        )

        result = mt_data_tree.plot_mt_response(
            station_key=list(mt_data_tree._iter_station_paths()),
            backend="bokeh",
            show_plot=False,
        )
        assert isinstance(result, BokehMultiple)

    def test_multi_station_mpl_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_mt_responses import PlotMultipleResponses as MplMultiple

        result = mt_data_tree.plot_mt_response(
            station_key=list(mt_data_tree._iter_station_paths()),
            backend="matplotlib",
            show_plot=False,
        )
        assert isinstance(result, MplMultiple)

    def test_single_station_bokeh_returns_bokeh_instance(
        self, mt_data_tree, single_station_key
    ):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_mt_response import (
            PlotMTResponse as BokehSingle,
        )

        result = mt_data_tree.plot_mt_response(
            station_key=single_station_key,
            backend="bokeh",
            show_plot=False,
        )
        assert isinstance(result, BokehSingle)

    def test_single_station_mpl_returns_mpl_instance(
        self, mt_data_tree, single_station_key
    ):
        from mtpy.imaging.plot_mt_response import PlotMTResponse as MplSingle

        result = mt_data_tree.plot_mt_response(
            station_key=single_station_key,
            backend="matplotlib",
            show_plot=False,
        )
        assert isinstance(result, MplSingle)

    def test_default_backend_is_bokeh(self, mt_data_tree, single_station_key):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_mt_response import (
            PlotMTResponse as BokehSingle,
        )

        result = mt_data_tree.plot_mt_response(
            station_key=single_station_key, show_plot=False
        )
        assert isinstance(result, BokehSingle)

    def test_invalid_backend_raises_value_error(self, mt_data_tree, single_station_key):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_mt_response(
                station_key=single_station_key, backend="seaborn"
            )


# ---------------------------------------------------------------------------
# plot_phase_tensor
# ---------------------------------------------------------------------------


class TestPlotPhaseTensorBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree, single_station_key):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_pt import PlotPhaseTensor as BokehPT

        result = mt_data_tree.plot_phase_tensor(
            station_key=single_station_key, backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehPT)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree, single_station_key):
        from mtpy.imaging.plot_pt import PlotPhaseTensor as MplPT

        result = mt_data_tree.plot_phase_tensor(
            station_key=single_station_key, backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplPT)

    def test_default_backend_is_bokeh(self, mt_data_tree, single_station_key):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_pt import PlotPhaseTensor as BokehPT

        result = mt_data_tree.plot_phase_tensor(
            station_key=single_station_key, show_plot=False
        )
        assert isinstance(result, BokehPT)

    def test_invalid_backend_raises_value_error(self, mt_data_tree, single_station_key):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_phase_tensor(station_key=single_station_key, backend="d3")


# ---------------------------------------------------------------------------
# plot_phase_tensor_map
# ---------------------------------------------------------------------------


class TestPlotPhaseTensorMapBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_phase_tensor_maps import (
            PlotPhaseTensorMaps as BokehPTMaps,
        )

        result = mt_data_tree.plot_phase_tensor_map(backend="bokeh", show_plot=False)
        assert isinstance(result, BokehPTMaps)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_phase_tensor_maps import PlotPhaseTensorMaps as MplPTMaps

        result = mt_data_tree.plot_phase_tensor_map(
            backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplPTMaps)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_phase_tensor_map(backend="vega")


# ---------------------------------------------------------------------------
# plot_tipper_map
# ---------------------------------------------------------------------------


class TestPlotTipperMapBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_phase_tensor_maps import (
            PlotPhaseTensorMaps as BokehPTMaps,
        )

        result = mt_data_tree.plot_tipper_map(backend="bokeh", show_plot=False)
        assert isinstance(result, BokehPTMaps)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_phase_tensor_maps import PlotPhaseTensorMaps as MplPTMaps

        result = mt_data_tree.plot_tipper_map(backend="matplotlib", show_plot=False)
        assert isinstance(result, MplPTMaps)

    def test_tipper_defaults_set_regardless_of_backend(self, mt_data_tree):
        pytest.importorskip("bokeh")
        result = mt_data_tree.plot_tipper_map(backend="bokeh", show_plot=False)
        assert result.plot_pt is False
        assert result.plot_tipper == "yri"

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_tipper_map(backend="highcharts")


# ---------------------------------------------------------------------------
# plot_phase_tensor_pseudosection
# ---------------------------------------------------------------------------


class TestPlotPhaseTensorPseudosectionBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_phase_tensor_pseudosection import (
            PlotPhaseTensorPseudoSection as BokehPTPS,
        )

        result = mt_data_tree.plot_phase_tensor_pseudosection(
            backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehPTPS)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_phase_tensor_pseudosection import (
            PlotPhaseTensorPseudoSection as MplPTPS,
        )

        result = mt_data_tree.plot_phase_tensor_pseudosection(
            backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplPTPS)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_phase_tensor_pseudosection(backend="plotly")


# ---------------------------------------------------------------------------
# plot_penetration_depth_1d
# ---------------------------------------------------------------------------


class TestPlotPenetrationDepth1DBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree, single_station_key):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_penetration_depth_1d import (
            PlotPenetrationDepth1D as BokehPD1D,
        )

        result = mt_data_tree.plot_penetration_depth_1d(
            station_key=single_station_key, backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehPD1D)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree, single_station_key):
        from mtpy.imaging.plot_penetration_depth_1d import (
            PlotPenetrationDepth1D as MplPD1D,
        )

        result = mt_data_tree.plot_penetration_depth_1d(
            station_key=single_station_key, backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplPD1D)

    def test_default_backend_is_bokeh(self, mt_data_tree, single_station_key):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_penetration_depth_1d import (
            PlotPenetrationDepth1D as BokehPD1D,
        )

        result = mt_data_tree.plot_penetration_depth_1d(
            station_key=single_station_key, show_plot=False
        )
        assert isinstance(result, BokehPD1D)

    def test_invalid_backend_raises_value_error(self, mt_data_tree, single_station_key):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_penetration_depth_1d(
                station_key=single_station_key, backend="canvas"
            )


# ---------------------------------------------------------------------------
# plot_penetration_depth_map
# ---------------------------------------------------------------------------


class TestPlotPenetrationDepthMapBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_penetration_depth_map import (
            PlotPenetrationDepthMap as BokehPDMap,
        )

        result = mt_data_tree.plot_penetration_depth_map(
            backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehPDMap)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_penetration_depth_map import (
            PlotPenetrationDepthMap as MplPDMap,
        )

        result = mt_data_tree.plot_penetration_depth_map(
            backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplPDMap)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_penetration_depth_map(backend="svg")


# ---------------------------------------------------------------------------
# plot_resistivity_phase_maps
# ---------------------------------------------------------------------------


class TestPlotResistivityPhaseMapsBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_resphase_maps import (
            PlotResPhaseMaps as BokehRPMaps,
        )

        result = mt_data_tree.plot_resistivity_phase_maps(
            backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehRPMaps)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_resphase_maps import PlotResPhaseMaps as MplRPMaps

        result = mt_data_tree.plot_resistivity_phase_maps(
            backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplRPMaps)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_resistivity_phase_maps(backend="webgl")


# ---------------------------------------------------------------------------
# plot_resistivity_phase_pseudosections
# ---------------------------------------------------------------------------


class TestPlotResistivityPhasePseudosectionsBackend:
    def test_bokeh_returns_bokeh_instance(self, mt_data_tree):
        pytest.importorskip("bokeh")
        from mtpy.imaging.bokeh_plots.plot_pseudosection import (
            PlotResPhasePseudoSection as BokehRPPS,
        )

        result = mt_data_tree.plot_resistivity_phase_pseudosections(
            backend="bokeh", show_plot=False
        )
        assert isinstance(result, BokehRPPS)

    def test_matplotlib_returns_mpl_instance(self, mt_data_tree):
        from mtpy.imaging.plot_pseudosection import PlotResPhasePseudoSection as MplRPPS

        result = mt_data_tree.plot_resistivity_phase_pseudosections(
            backend="matplotlib", show_plot=False
        )
        assert isinstance(result, MplRPPS)

    def test_invalid_backend_raises_value_error(self, mt_data_tree):
        with pytest.raises(ValueError, match="Unknown backend"):
            mt_data_tree.plot_resistivity_phase_pseudosections(backend="canvas")
