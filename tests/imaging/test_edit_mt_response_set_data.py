"""Tests for swapping data into a persistent EditMTResponse."""

from __future__ import annotations

import numpy as np
import pytest


pn = pytest.importorskip("panel")

from mtpy.core.mt import MT
from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse


def _make_mt(station: str, n_periods: int = 9, scale: float = 1.0) -> MT:
    frequency = np.logspace(2, -2, n_periods)
    mt = MT(station=station, survey="s", frequency=frequency)
    z = np.zeros((n_periods, 2, 2), dtype=complex)
    z[:, 0, 1] = np.linspace(1, 9, n_periods) * (1 + 1j) * scale
    z[:, 1, 0] = -z[:, 0, 1]
    mt.impedance = z
    mt.impedance_error = np.full((n_periods, 2, 2), 0.1)
    tipper = np.zeros((n_periods, 1, 2), dtype=complex)
    tipper[:, 0, 0] = 0.1 + 0.1j
    tipper[:, 0, 1] = 0.2
    mt.tipper = tipper
    mt.tipper_error = np.full((n_periods, 1, 2), 0.01)
    return mt


def _editor(mt: MT, **kwargs) -> EditMTResponse:
    return EditMTResponse(
        z_object=mt.Z, t_object=mt.Tipper, station=mt.station, show_plot=False, **kwargs
    )


def test_set_data_swaps_data_and_clears_edit_state():
    mt_a, mt_b = _make_mt("A"), _make_mt("B", n_periods=5, scale=2.0)
    editor = _editor(mt_a)
    editor.flip_phase(zxy=True)
    editor.masked_tf_indices = {"xy": {1}}
    editor.rotation_angle = 10
    assert editor._data_manipulated

    editor.set_data(z_object=mt_b.Z, t_object=mt_b.Tipper, station="B")

    assert editor.station == "B"
    assert editor.period.size == 5
    np.testing.assert_allclose(editor.Z.z, mt_b.Z.z)
    assert editor.masked_tf_indices == {}
    assert editor._original_Z is None
    assert editor._data_manipulated is False
    assert editor.rotation_angle == 0
    assert editor.x_limits is None and editor.res_limits is None
    assert editor.layout is None


def test_set_data_copies_inputs_and_sets_callback_and_error_mode():
    mt_a, mt_b = _make_mt("A"), _make_mt("B")
    editor = _editor(mt_a)
    callback = lambda z, t: None
    editor.set_data(
        z_object=mt_b.Z,
        t_object=mt_b.Tipper,
        responses=[("r", mt_a.Z, mt_a.Tipper)],
        station="B",
        save_edits_callback=callback,
        plot_model_error=True,
    )
    assert editor.Z is not mt_b.Z
    assert editor.save_edits_callback is callback
    assert editor.responses[0][0] == "r"
    assert editor._error_str == "model_error"
    # reset_to_original restores the new initial error mode
    editor.flip_phase(zxy=True)
    editor.plot_model_error = False
    editor.reset_to_original()
    assert editor.plot_model_error is True


def test_set_data_edits_do_not_mutate_new_source():
    mt_a, mt_b = _make_mt("A"), _make_mt("B")
    original = mt_b.Z.z.copy()
    editor = _editor(mt_a)
    editor.set_data(z_object=mt_b.Z, t_object=mt_b.Tipper, station="B")
    editor.flip_phase(zxy=True)
    np.testing.assert_allclose(mt_b.Z.z, original)


def test_set_data_with_panel_reuses_layout_and_resyncs_widgets():
    mt_a, mt_b = _make_mt("A"), _make_mt("B", n_periods=5, scale=2.0)
    editor = _editor(mt_a)
    panel_obj = editor.panel()
    pane_before = [
        o for o in panel_obj.select(pn.pane.Bokeh) if o.object is editor.layout
    ]
    assert pane_before
    editor.flip_phase(zxy=True)

    editor.set_data(
        z_object=mt_b.Z,
        t_object=mt_b.Tipper,
        station="B",
        save_edits_callback=lambda z, t: None,
    )

    assert editor._panel_sync is not None
    assert editor.layout is not None
    assert pane_before[0].object is editor.layout
    assert panel_obj[0].object == "## B"
    assert editor._data_manipulated is False
    assert editor.x_limits is not None
    assert editor.period.size == 5
    int_widgets = {w.name: w for w in panel_obj.select(pn.widgets.IntInput)}
    assert int_widgets["Num periods"].value == 5
    save_buttons = [
        w for w in panel_obj.select(pn.widgets.Button) if w.name == "Save Edits"
    ]
    assert save_buttons[0].visible is True


def test_set_data_with_panel_replots_each_call():
    mt_a, mt_b = _make_mt("A"), _make_mt("B")
    editor = _editor(mt_a)
    editor.panel()
    layouts = [editor.layout]
    for mt in (mt_b, mt_a):
        editor.set_data(z_object=mt.Z, t_object=mt.Tipper, station=mt.station)
        layouts.append(editor.layout)
    assert len({id(layout) for layout in layouts}) == 3


def _n_figures(layout):
    from bokeh.plotting import figure

    return len([r for r in layout.select({"type": figure})])


def _no_tipper(mt):
    return mt.Z, None


def test_plot_without_tipper_keeps_all_twelve_subplots():
    mt = _make_mt("A")
    editor = EditMTResponse(z_object=mt.Z, t_object=None, station="A", show_plot=False)
    layout = editor.plot()
    assert _n_figures(layout) == 12
    assert editor.plot_tipper == "n"
    assert "xy" in editor.renderers


def test_plot_without_z_keeps_all_twelve_subplots():
    mt = _make_mt("A")
    editor = EditMTResponse(
        z_object=None, t_object=mt.Tipper, station="A", show_plot=False
    )
    layout = editor.plot()
    assert _n_figures(layout) == 12
    assert editor.plot_z is False


def test_plot_with_no_data_is_blank():
    editor = EditMTResponse(z_object=None, t_object=None, station="A", show_plot=False)
    layout = editor.plot()
    assert _n_figures(layout) == 12
    assert not any(editor.renderers.values())


@pytest.mark.parametrize("which", ["no_tipper", "no_z", "none"])
def test_panel_with_missing_data(which):
    mt = _make_mt("A")
    z = None if which in ("no_z", "none") else mt.Z
    t = None if which in ("no_tipper", "none") else mt.Tipper
    editor = EditMTResponse(z_object=z, t_object=t, station="A", show_plot=False)
    assert editor.panel() is not None


def test_set_data_switches_between_full_partial_and_blank():
    mt = _make_mt("A")
    editor = _editor(mt)
    editor.panel()
    for z, t in [(None, None), (mt.Z, None), (None, mt.Tipper), (mt.Z, mt.Tipper)]:
        editor.set_data(z_object=z, t_object=t, station="B")
        layout = editor.plot()
        assert _n_figures(layout) == 12
        assert editor.plot_z is (z is not None)
        assert (editor.plot_tipper != "n") is (t is not None)


def test_single_legend_per_column_with_rms_and_tipper_legends():
    from bokeh.models import Legend

    mt = _make_mt("A")
    resp = _make_mt("A", scale=1.1)
    editor = EditMTResponse(
        z_object=mt.Z,
        t_object=mt.Tipper,
        responses=[("Resp", resp.Z, resp.Tipper)],
        station="A",
        show_plot=False,
    )
    editor.plot()
    f = editor.figures

    def legends(key):
        return [m for m in f[key].center + f[key].above if isinstance(m, Legend)]

    for comp in ("xx", "xy", "yx", "yy"):
        assert len(legends(f"res_{comp}")) == 1
        assert legends(f"phase_{comp}") == []
    labels = [i.label["value"] for i in legends("res_xy")[0].items]
    assert labels[0] == "Zxy"
    assert len(labels) == 2
    assert labels[1].startswith("Resp (RMS ") and "/" not in labels[1]

    assert len(legends("tip_real_zx")) == 1 and len(legends("tip_real_zy")) == 1
    assert legends("tip_imag_zx") == [] and legends("tip_imag_zy") == []
    t_labels = [i.label["value"] for i in legends("tip_real_zx")[0].items]
    assert len(t_labels) == 2 and t_labels[0] == "Tzx"
    assert t_labels[1].startswith("Resp (RMS ")
    assert legends("res_xy")[0].orientation == "vertical"


def test_normalized_rms_is_data_minus_response_over_error():
    x = np.logspace(-2, 2, 9)
    y = np.linspace(1, 9, 9) + 0j
    err = np.full(9, 0.5)
    assert EditMTResponse._normalized_rms(y, y, err, x, x) == pytest.approx(0.0)
    # offset of 2 over error 0.5 -> 4
    assert EditMTResponse._normalized_rms(y, y - 2, err, x, x) == pytest.approx(4.0)
    # only shared periods count; masked points are skipped
    assert EditMTResponse._normalized_rms(y, y[2:] - 2, err, x, x[2:]) == (
        pytest.approx(4.0)
    )
    assert EditMTResponse._normalized_rms(y, y - 2, err, x, x, masked={0}) == (
        pytest.approx(4.0)
    )
    assert EditMTResponse._normalized_rms(y, y, err, x, x * 1e6) is None


def test_legend_label_strips_station_suffix():
    mt = _make_mt("A")
    editor = _editor(mt)
    assert editor._legend_label("model/A") == "model"
    assert editor._legend_label("model/B") == "model/B"
    assert editor._legend_label("/A") == "/A"
