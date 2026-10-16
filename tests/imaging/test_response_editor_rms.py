"""The editor tab's RMS must match EditMTResponse's and follow edits."""

from __future__ import annotations

import re

import numpy as np
import pytest


pn = pytest.importorskip("panel")

from mtpy.core.mt import MT
from mtpy.imaging.bokeh_plots.panel_mt_response_editor_tab import MTResponseEditorTab


def _make_mt(station: str, scale: float = 1.0, n_periods: int = 9) -> MT:
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
    mt.tipper = tipper * scale
    mt.tipper_error = np.full((n_periods, 1, 2), 0.01)
    return mt


def _legend_rms(editor) -> dict[str, float]:
    """Parse the ``RMS x.xx`` values from the plot legends, keyed by name."""
    found = {}
    for key, fig in editor.figures.items():
        for legend in fig.legend:
            for item in legend.items:
                match = re.search(r"RMS (\d+\.\d+)", item.label["value"])
                if match:
                    found.setdefault(key, set()).add(float(match.group(1)))
    return found


def _pane_values(tab) -> dict[str, float]:
    return {
        name: float(val)
        for name, val in re.findall(r"(\w+)=(\d+\.\d+)", tab._rms_pane.object)
    }


@pytest.fixture
def tab():
    data, response = _make_mt("MT01"), _make_mt("MT01", scale=1.5)
    tab = MTResponseEditorTab()
    tab._show_station(
        data,
        [("b/MT01", response.Z, response.Tipper)],
        "/surveys/a/stations/MT01",
        False,
    )
    return tab


def test_tab_rms_matches_editor(tab):
    editor = tab._plot_obj
    z, z_resp = editor.Z, editor.responses[0][1]
    expected = np.sqrt(
        np.mean(np.abs((z.z[:, 0, 1] - z_resp.z[:, 0, 1]) / z.z_error[:, 0, 1]) ** 2)
    )
    assert editor.rms_summary()[0][1]["Zxy"] == pytest.approx(expected)
    summary = editor.rms_summary()
    assert summary and summary[0][0] == "b/MT01"
    assert {"Zxy", "Zyx", "Tx", "Ty"} <= set(summary[0][1])

    pane = _pane_values(tab)
    assert pane.keys() == summary[0][1].keys()
    for name, value in summary[0][1].items():
        assert pane[name] == pytest.approx(value, abs=0.006)

    # The legend shows the same numbers the editor reports.
    legend_values = {v for vals in _legend_rms(editor).values() for v in vals}
    summary_values = {round(v, 2) for v in summary[0][1].values()}
    assert legend_values and legend_values <= summary_values


def test_tab_rms_updates_when_data_is_edited(tab):
    editor = tab._plot_obj
    before = dict(_pane_values(tab))

    editor.static_shift(ss_x=2.0, ss_y=2.0)
    editor.plot()
    after_shift = _pane_values(tab)
    assert after_shift != before
    for name, value in dict(editor.rms_summary()[0][1]).items():
        assert after_shift[name] == pytest.approx(value, abs=0.006)

    editor.masked_tf_indices = {"xy": {0, 1, 2}}
    editor.plot()
    after_mask = _pane_values(tab)
    assert after_mask["Zxy"] != after_shift["Zxy"]
    assert after_mask["Zxy"] == pytest.approx(
        editor.rms_summary()[0][1]["Zxy"], abs=0.006
    )

    assert editor.reset_to_original()
    editor.masked_tf_indices = {}
    editor.plot()
    assert _pane_values(tab) == before
