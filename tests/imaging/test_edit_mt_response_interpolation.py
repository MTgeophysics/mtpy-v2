"""Tests that EditMTResponse edits operate on an editable copy of the data."""

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse


@pytest.fixture
def z_obj():
    frequency = np.logspace(2, -2, 9)
    z = np.zeros((9, 2, 2), dtype=complex)
    base = np.linspace(1.0, 9.0, 9)
    z[:, 0, 1] = base + 1j * base
    z[:, 1, 0] = -(base + 1j * base)
    # the point to be masked is a spike, so interpolation must change it
    z[4, 0, 1] = 100 + 100j
    err = np.full((9, 2, 2), 0.1)
    return Z(z=z, z_error=err, frequency=frequency)


def test_constructor_copies_inputs(z_obj):
    editor = EditMTResponse(z_object=z_obj, show_plot=False)
    editor.flip_phase(zxy=True)
    assert z_obj.z[0, 0, 1] == 1 + 1j
    assert editor.Z.z[0, 0, 1] == -(1 + 1j)


def test_interpolate_replaces_masked_points_on_same_periods(z_obj):
    editor = EditMTResponse(z_object=z_obj, show_plot=False)
    period_before = np.asarray(editor.period).copy()
    spike = editor.Z.z[4, 0, 1]

    editor.masked_tf_indices = {"xy": {4}}
    editor.interpolate()

    np.testing.assert_allclose(editor.period, period_before)
    assert editor.Z.z[4, 0, 1] != spike
    assert np.isfinite(editor.Z.z[4, 0, 1])
    # unmasked points are unchanged and the original is preserved for display
    assert editor.Z.z[3, 0, 1] == z_obj.z[3, 0, 1]
    assert editor._original_Z.z[4, 0, 1] == spike
    assert editor.masked_tf_indices == {}


def test_interpolate_uses_edited_data(z_obj):
    editor = EditMTResponse(z_object=z_obj, show_plot=False)
    editor.flip_phase(zxy=True)
    editor.masked_tf_indices = {"xy": {4}}
    editor.interpolate()
    # flip is retained and the masked point is estimated from flipped neighbours
    assert editor.Z.z[3, 0, 1] == -z_obj.z[3, 0, 1]
    assert editor.Z.z[4, 0, 1].real < 0
