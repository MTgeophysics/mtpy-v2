"""Tests for the Panel MT response application."""

import pytest


panel = pytest.importorskip("panel")

from mtpy.imaging.bokeh_plots.panel_mt_response_app import MTResponseApp


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
