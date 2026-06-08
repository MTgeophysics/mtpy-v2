"""Tests for the transfer-function editor Panel application."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


panel = pytest.importorskip("panel")

from mtpy.imaging.bokeh_plots.panel_transfer_function_editor_app import (
    TransferFunctionEditorPanelApp,
)


@pytest.fixture
def sample_station_df() -> pd.DataFrame:
    freq = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
    res_xy = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    res_yx = res_xy * 1.1
    res_xx = res_xy * 0.8
    res_yy = res_xy * 1.2
    phase_xy = np.array([45.0, 40.0, 35.0, 30.0], dtype=float)
    phase_yx = phase_xy * 0.95
    phase_xx = phase_xy * 0.9
    phase_yy = phase_xy * 1.05

    t_zx = np.array([0.20 + 0.05j, 0.18 + 0.04j, 0.12 + 0.03j, 0.09 + 0.02j])
    t_zy = np.array([0.15 - 0.04j, 0.10 - 0.03j, 0.08 - 0.02j, 0.05 - 0.01j])

    return pd.DataFrame(
        {
            "frequency": freq,
            "res_xx": res_xx,
            "res_xx_error": res_xx * 0.1,
            "res_xy": res_xy,
            "res_xy_error": res_xy * 0.1,
            "res_yx": res_yx,
            "res_yx_error": res_yx * 0.1,
            "res_yy": res_yy,
            "res_yy_error": res_yy * 0.1,
            "res_det": np.sqrt(res_xy * res_yx),
            "res_det_error": np.sqrt(res_xy * res_yx) * 0.1,
            "phase_xx": phase_xx,
            "phase_xx_error": np.ones_like(phase_xx) * 2.0,
            "phase_xy": phase_xy,
            "phase_xy_error": np.ones_like(phase_xy) * 2.5,
            "phase_yx": phase_yx,
            "phase_yx_error": np.ones_like(phase_yx) * 2.8,
            "phase_yy": phase_yy,
            "phase_yy_error": np.ones_like(phase_yy) * 2.2,
            "phase_det": 0.5 * (phase_xy + phase_yx),
            "phase_det_error": np.ones_like(phase_xy) * 2.6,
            "t_zx": t_zx,
            "t_zx_error": np.ones_like(freq) * 0.02,
            "t_zy": t_zy,
            "t_zy_error": np.ones_like(freq) * 0.02,
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


@pytest.mark.plotting
def test_load_station_populates_default_1d_plot_data(sample_station_df):
    app = TransferFunctionEditorPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))

    app._on_load_station_clicked()

    assert len(app._active_res_editors) == 1
    assert len(app._active_phase_editors) == 1
    assert len(app._active_tipper_editors) == 1
    assert len(app._active_res_editors[0]._source.data["period"]) > 0
    assert len(app._active_phase_editors[0]._source.data["period"]) > 0
    assert len(app._active_tipper_editors[0]._source.data["period"]) > 0


@pytest.mark.plotting
def test_dimension_2d_xyyx_shows_two_res_and_phase_plots(sample_station_df):
    app = TransferFunctionEditorPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._on_load_station_clicked()

    app._edit_dimension_widget.value = "2D"
    app._edit_mode_widget.value = "xy/yx"

    assert len(app._active_res_editors) == 2
    assert len(app._active_phase_editors) == 2
    assert len(app._active_tipper_editors) == 2
    assert all(len(ed._source.data["period"]) > 0 for ed in app._active_res_editors)


@pytest.mark.plotting
def test_dimension_3d_shows_tensor_and_tipper_real_imag(sample_station_df):
    app = TransferFunctionEditorPanelApp()
    app.set_mt_data(_mock_mt_data(sample_station_df))
    app._on_load_station_clicked()

    app._edit_dimension_widget.value = "3D"

    assert len(app._active_res_editors) == 4
    assert len(app._active_phase_editors) == 4
    assert len(app._active_tipper_editors) == 4
    assert all(len(ed._source.data["period"]) > 0 for ed in app._active_tipper_editors)
