"""Tests for the Panel MT response application."""

import pytest

panel = pytest.importorskip("panel")

from mtpy import MT
from mtpy.imaging.bokeh_plots.panel_mt_response_app import MTResponseApp

pytest.importorskip("scipy")  # Required for interpolation


@pytest.mark.plotting
def test_rejects_unsupported_selected_files():
    """Unsupported files should be rejected before MT.read is called."""
    app = MTResponseApp()

    app._file_selector.value = ["example.py"]
    app._on_controls_changed(None)

    assert "Unsupported file type" in app._status.object
    assert "example.py" in app._status.object
    assert len(app._plot_container) == 0


@pytest.mark.plotting
def test_file_type_filter_clears_stale_selection():
    """Changing the file filter should clear old selections and plots."""
    app = MTResponseApp()

    app._file_selector.value = ["example.edi"]
    app._plot_container.append(panel.pane.Markdown("plot"))
    app._status.object = "busy"
    app._file_pattern_widget.value = "*.xml"

    assert app._file_selector.file_pattern == "*.xml"
    assert app._file_selector.value == []
    assert len(app._plot_container) == 0
    assert app._status.object == ""


@pytest.mark.plotting
def test_interpolation_periods_stored():
    """Selected periods should be stored when manually added."""
    pytest.importorskip("mt_metadata")  # Required for loading MT data
    from mt_metadata import TF_EDI_CGG

    app = MTResponseApp()
    mt_obj = MT(TF_EDI_CGG)
    mt_obj.read()

    # Simulate adding periods manually
    app._current_mt_object = mt_obj
    app._all_periods = mt_obj.Z.period.copy()
    app._periods_removed.add(1.5)
    app._periods_removed.add(3.0)

    assert len(app._periods_removed) == 2
    assert 1.5 in app._periods_removed
    assert 3.0 in app._periods_removed


@pytest.mark.plotting
def test_interpolation_requires_loaded_data():
    """Interpolation should fail gracefully if no data is loaded."""
    app = MTResponseApp()
    app._on_interpolate_clicked(None)

    assert "No MT data loaded" in app._status.object


@pytest.mark.plotting
def test_interpolation_requires_selected_periods():
    """Interpolation should fail if no periods are selected."""
    pytest.importorskip("mt_metadata")
    from mt_metadata import TF_EDI_CGG

    app = MTResponseApp()
    mt_obj = MT(TF_EDI_CGG)
    mt_obj.read()

    app._current_mt_object = mt_obj
    app._current_plotter = None
    app._periods_removed.clear()

    app._on_interpolate_clicked(None)

    assert "No periods selected" in app._status.object
