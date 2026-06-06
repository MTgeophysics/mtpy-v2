"""Tests for the Simpeg1D Panel modeling app."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


panel = pytest.importorskip("panel")

from mtpy.imaging.bokeh_plots.panel_simpeg1d_app import Simpeg1DPanelApp


@pytest.fixture
def sample_station_df() -> pd.DataFrame:
    freq = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
    res = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    phase = np.array([45.0, 40.0, 35.0, 30.0], dtype=float)
    return pd.DataFrame(
        {
            "frequency": freq,
            "res_xy": res,
            "res_xy_error": res * 0.1,
            "phase_xy": phase,
            "phase_xy_error": np.ones_like(phase) * 2.5,
            "res_yx": res * 1.1,
            "res_yx_error": res * 0.12,
            "phase_yx": phase * 0.95,
            "phase_yx_error": np.ones_like(phase) * 2.8,
        }
    )


def _mock_mt_data(sample_station_df: pd.DataFrame):
    mt_obj = MagicMock()
    mt_obj.to_dataframe.return_value = SimpleNamespace(
        dataframe=sample_station_df.copy()
    )

    mt_data = MagicMock()
    mt_data._iter_station_paths.return_value = iter(["/surveys/s1/stations/MT01"])
    mt_data.get_station.return_value = mt_obj
    return mt_data


@pytest.fixture
def patch_fake_simpeg(monkeypatch):
    class FakeSimpeg1D:
        def __init__(
            self, mt_dataframe, mode, resistivity_error=10.0, phase_error=2.5, **kwargs
        ):
            self.mt_dataframe = mt_dataframe
            self.mode = mode
            self.resistivity_error = resistivity_error
            self.phase_error = phase_error
            self.output_dict = {}
            self.n_layers = 50
            self.dz = 5.0
            self.z_factor = 1.2
            self.rho_initial = 100.0
            self.rho_reference = 100.0

            if mode == "tm":
                res_col, res_err_col = "res_yx", "res_yx_error"
                phase_col, phase_err_col = "phase_yx", "phase_yx_error"
            else:
                res_col, res_err_col = "res_xy", "res_xy_error"
                phase_col, phase_err_col = "phase_xy", "phase_xy_error"

            self._sub_df = pd.DataFrame(
                {
                    "frequency": mt_dataframe["frequency"].to_numpy(dtype=float),
                    "res": mt_dataframe[res_col].to_numpy(dtype=float),
                    "res_error": mt_dataframe[res_err_col].to_numpy(dtype=float),
                    "phase": mt_dataframe[phase_col].to_numpy(dtype=float),
                    "phase_error": mt_dataframe[phase_err_col].to_numpy(dtype=float),
                }
            )

        @property
        def periods(self):
            return 1.0 / self._sub_df["frequency"]

        @staticmethod
        def _phase_for_plotting(values):
            arr = np.asarray(values, dtype=float)
            return np.where(arr < 0.0, arr + 180.0, arr)

        @property
        def _plot_z(self):
            return np.arange(self.n_layers + 1, dtype=float) * 0.1

        def run_fixed_layer_inversion(self, **kwargs):
            n = len(self._sub_df)
            dpred = np.column_stack(
                [
                    self._sub_df["res"].to_numpy(dtype=float),
                    self._sub_df["phase"].to_numpy(dtype=float),
                ]
            ).reshape((n * 2,))
            self.output_dict = {
                1: {
                    "dpred": dpred,
                    "m": np.log(np.ones(self.n_layers) / 100.0),
                    "phi_d": 1.2,
                    "phi_m": 0.9,
                    "beta": 2.5,
                }
            }

    monkeypatch.setattr(
        "mtpy.imaging.bokeh_plots.panel_simpeg1d_app.Simpeg1D",
        FakeSimpeg1D,
    )


@pytest.mark.plotting
def test_set_mt_data_populates_station_options(sample_station_df, patch_fake_simpeg):
    app = Simpeg1DPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))

    assert app._station_widget.options == ["/surveys/s1/stations/MT01"]
    assert app._station_widget.value == "/surveys/s1/stations/MT01"


@pytest.mark.plotting
def test_load_station_data_creates_working_dataframe(
    sample_station_df, patch_fake_simpeg
):
    app = Simpeg1DPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._mode_widget.value = "xy"
    app._on_load_station_clicked()

    assert not app._data_df.empty
    assert {"frequency", "res", "res_error", "phase", "phase_error", "use"}.issubset(
        set(app._data_df.columns)
    )


@pytest.mark.plotting
def test_delete_selected_points_removes_rows(sample_station_df, patch_fake_simpeg):
    app = Simpeg1DPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._mode_widget.value = "xy"
    app._on_load_station_clicked()

    n0 = len(app._data_df)
    app._source.selected.indices = [0, 2]
    app._on_delete_selected_clicked()

    assert len(app._data_df) == n0 - 2


@pytest.mark.plotting
def test_apply_errors_updates_selected_rows(sample_station_df, patch_fake_simpeg):
    app = Simpeg1DPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._mode_widget.value = "xy"
    app._on_load_station_clicked()

    app._source.selected.indices = [0]
    before_res_err = float(app._data_df.loc[0, "res_error"])
    before_phase_err = float(app._data_df.loc[0, "phase_error"])

    app._selected_res_error_widget.value = 25.0
    app._selected_phase_error_widget.value = 5.0
    app._on_apply_errors_clicked()

    assert float(app._data_df.loc[0, "res_error"]) >= before_res_err
    assert float(app._data_df.loc[0, "phase_error"]) >= before_phase_err


@pytest.mark.plotting
def test_run_clicked_updates_output_with_mocked_inversion(
    sample_station_df, patch_fake_simpeg
):
    app = Simpeg1DPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._mode_widget.value = "xy"
    app._on_load_station_clicked()

    app._on_run_clicked()

    assert "Inversion complete" in app._output.object
    assert app._simpeg is not None
