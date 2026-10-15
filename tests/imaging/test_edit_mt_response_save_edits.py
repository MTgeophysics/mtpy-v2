"""Tests that Save Edits persists removed points and model error changes."""

from __future__ import annotations

import numpy as np
import pytest


pn = pytest.importorskip("panel")

from mtpy.core.mt import MT
from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse
from mtpy.imaging.bokeh_plots.panel_mt_response_editor_tab import MTResponseEditorTab


N = 9


@pytest.fixture
def setup():
    mt = MT(station="MT01", survey="a", frequency=np.logspace(2, -2, N))
    z = np.zeros((N, 2, 2), dtype=complex)
    z[:, 0, 1] = np.linspace(1, 9, N) * (1 + 1j)
    z[:, 1, 0] = -z[:, 0, 1]
    mt.impedance = z
    mt.impedance_error = np.full((N, 2, 2), 0.1)
    tipper = np.zeros((N, 1, 2), dtype=complex)
    tipper[:, 0, 0] = 0.1 + 0.1j
    mt.tipper = tipper
    mt.tipper_error = np.full((N, 1, 2), 0.01)

    mt_data = MTData()
    path = mt_data.add_station(mt)
    tab = MTResponseEditorTab()
    tab.set_mt_data(mt_data)
    station_mt = mt_data.get_station(path, as_mt=True)
    editor = EditMTResponse(
        z_object=station_mt.Z,
        t_object=station_mt.Tipper,
        station="MT01",
        save_edits_callback=lambda z, t: tab._save_edits(path, station_mt, z, t),
        show_plot=False,
    )
    return mt_data, path, editor


def test_save_edits_persists_removed_points_and_model_error(setup):
    mt_data, path, editor = setup
    editor.add_model_error("zxy", z_value=5.0)
    editor.masked_tf_indices = {"xy": {4}, "tzx": {2}}
    editor.save_edits()

    saved = mt_data.get_station(path, as_mt=True)
    z = saved.Z.z[:, 0, 1]
    keep = [i for i in range(N) if i != 4]
    assert np.isnan(z[4])
    assert np.isfinite(z[keep]).all()
    np.testing.assert_allclose(saved.Z.z_model_error[keep, 0, 1], 0.5)
    # other components are untouched by the mask
    assert np.isfinite(saved.Z.z[:, 1, 0]).all()
    assert np.isnan(saved.Tipper.tipper[2, 0, 0])
    assert np.isfinite(saved.Tipper.tipper[:, 0, 1]).all()

    # editor now reflects the saved baseline
    assert editor.masked_tf_indices == {}
    assert np.isnan(editor.Z.z[4, 0, 1])


def test_save_edits_persists_interpolated_replacement_and_model_error(setup):
    mt_data, path, editor = setup
    editor.add_model_error("zxy", z_value=5.0)
    editor.masked_tf_indices = {"xy": {4}}
    editor.interpolate()
    editor.save_edits()

    saved = mt_data.get_station(path, as_mt=True)
    assert saved.Z.z.shape[0] == N
    assert np.isfinite(saved.Z.z[:, 0, 1]).all()
    np.testing.assert_allclose(saved.Z.z[4, 0, 1], 4.48 + 4.48j, atol=0.01)
    np.testing.assert_allclose(saved.Z.z_model_error[:, 0, 1], 0.5)


def test_save_edits_persists_new_period_grid(setup):
    mt_data, path, editor = setup
    editor.masked_tf_indices = {"xy": {4}}
    editor.interpolate(np.logspace(-2, 2, 5), extrapolate=True)
    editor.save_edits()

    saved = mt_data.get_station(path, as_mt=True)
    assert saved.Z.z.shape[0] == 5
    assert saved.Tipper.tipper.shape[0] == 5
    np.testing.assert_allclose(saved.period, np.logspace(-2, 2, 5))
