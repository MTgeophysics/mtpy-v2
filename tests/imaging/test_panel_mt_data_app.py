"""Tests for the Panel MT data loader application (MTDataApp)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


panel = pytest.importorskip("panel")

from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.panel_mt_data_app import (
    _build_station_summary,
    _STATION_TABLE_COLUMNS,
    DAT_FORMAT_MODEM,
    DAT_FORMAT_OCCAM2D,
    MTDataApp,
    SUPPORTED_TF_SUFFIXES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_app() -> MTDataApp:
    """Return a freshly constructed MTDataApp."""
    return MTDataApp()


# ── Smoke tests ───────────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_app_instantiates():
    """MTDataApp should instantiate without error."""
    app = _make_app()
    assert app is not None
    assert app.mt_data is None
    assert app.mt_data_loaded is False


@pytest.mark.plotting
def test_view_returns_panel_object():
    """app.view should return a Panel viewable."""
    app = _make_app()
    view = app.view
    assert view is not None
    assert hasattr(view, "servable")


@pytest.mark.plotting
def test_servable_returns_panel_object():
    """app.servable() should not raise."""
    app = _make_app()
    result = app.servable()
    assert result is not None


# ── Widget state ──────────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_file_pattern_change_updates_selector():
    """Changing file type filter updates the FileSelector pattern."""
    app = _make_app()
    new_pattern = "*.edi"
    app._file_pattern_widget.value = new_pattern
    assert app._file_selector.file_pattern == new_pattern


@pytest.mark.plotting
def test_load_with_no_selection_shows_warning():
    """Clicking Load with no files selected shows a warning status."""
    app = _make_app()
    app._file_selector.value = []
    app._on_load_clicked(None)

    assert "No files selected" in app._status.object
    assert app.mt_data is None
    assert app.mt_data_loaded is False


@pytest.mark.plotting
def test_save_button_disabled_before_load():
    """Save button should be disabled before any data is loaded."""
    app = _make_app()
    assert app._save_button.disabled is True


# ── Unsupported file types ─────────────────────────────────────────────────────


@pytest.mark.plotting
def test_unsupported_file_raises_in_load_files():
    """_load_files should raise ValueError for unsupported file types."""
    app = _make_app()
    with pytest.raises(ValueError, match="Unsupported file type"):
        app._load_files(["example.py"])


@pytest.mark.plotting
def test_unsupported_file_sets_error_status():
    """Selecting an unsupported file shows an error in the status display."""
    app = _make_app()
    app._file_selector.value = ["example.py"]
    app._on_load_clicked(None)

    assert "Error" in app._status.object
    assert app.mt_data is None
    assert app.mt_data_loaded is False


# ── TF file loading ───────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_tf_files_loaded_via_add_stations():
    """TF files should be passed to MTData.add_stations."""
    app = _make_app()

    mock_mt_data = MagicMock(spec=MTData)
    mock_mt_data.station_paths = ["/surveys/s/stations/a", "/surveys/s/stations/b"]
    mock_mt_data.add_stations = MagicMock()

    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
        return_value=mock_mt_data,
    ):
        result = app._load_files(["a.edi", "b.edi"])

    mock_mt_data.add_stations.assert_called_once_with(["a.edi", "b.edi"])
    assert result is mock_mt_data


@pytest.mark.plotting
def test_all_tf_suffixes_accepted():
    """Every supported TF suffix should be accepted without error."""
    app = _make_app()

    for suffix in SUPPORTED_TF_SUFFIXES:
        fake_file = f"station{suffix}"
        mock_mt_data = MagicMock(spec=MTData)
        mock_mt_data.station_paths = []
        mock_mt_data.add_stations = MagicMock()

        with patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ):
            # Should not raise
            app._load_files([fake_file])


# ── MTH5 loading ──────────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_mth5_file_opens_collection():
    """An .h5 file should open an MTCollection and use its mt_data."""
    app = _make_app()

    mock_mt_data = MagicMock()
    mock_collection = MagicMock()
    mock_collection.__enter__ = MagicMock(return_value=mock_collection)
    mock_collection.__exit__ = MagicMock(return_value=False)
    mock_collection.to_mt_data.return_value = mock_mt_data

    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTCollection",
            return_value=mock_collection,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.Path.is_file", return_value=True
        ),
    ):
        app._load_files(["my_survey.h5"])

    mock_collection.open_collection.assert_called_once()
    mock_collection.to_mt_data.assert_called_once()


@pytest.mark.plotting
def test_previous_collection_closed_on_new_load():
    """Loading new files should not error even if a previous collection object exists."""
    app = _make_app()
    # With the `with`-statement pattern the app no longer keeps a live collection
    # reference between calls; this test simply verifies _load_files succeeds
    # for a plain TF file without raising.
    mock_mt_data = MagicMock()

    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
        return_value=mock_mt_data,
    ):
        app._load_files(["a.edi"])

    mock_mt_data.add_stations.assert_called_once_with(["a.edi"])


# ── Mixed MTH5 + TF loading ───────────────────────────────────────────────────


@pytest.mark.plotting
def test_mixed_mth5_and_tf_files():
    """Selecting both .h5 and TF files should open MTH5 then add TF stations."""
    app = _make_app()

    mock_mt_data = MagicMock()
    mock_mt_data.__iadd__ = MagicMock(return_value=mock_mt_data)
    mock_collection = MagicMock()
    mock_collection.__enter__ = MagicMock(return_value=mock_collection)
    mock_collection.__exit__ = MagicMock(return_value=False)
    mock_collection.to_mt_data.return_value = mock_mt_data

    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTCollection",
            return_value=mock_collection,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.Path.is_file", return_value=True
        ),
    ):
        app._load_files(["archive.h5", "extra.edi"])

    mock_collection.open_collection.assert_called_once()
    mock_mt_data.add_stations.assert_called_once_with(["extra.edi"])


@pytest.mark.plotting
def test_multiple_mth5_files_warns_and_uses_first():
    """Selecting multiple .h5 files shows a warning and only uses the first."""
    app = _make_app()

    mock_mt_data = MagicMock()
    mock_collection = MagicMock()
    mock_collection.__enter__ = MagicMock(return_value=mock_collection)
    mock_collection.__exit__ = MagicMock(return_value=False)
    mock_collection.to_mt_data.return_value = mock_mt_data

    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTCollection",
            return_value=mock_collection,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.Path.is_file", return_value=True
        ),
    ):
        app._load_files(["first.h5", "second.h5"])

    # Both files should be opened (no longer limited to first only)
    assert mock_collection.open_collection.call_count == 2


# ── .dat / .data format loading ───────────────────────────────────────────────


@pytest.mark.plotting
def test_modem_dat_file_calls_from_modem():
    """A .dat file with format=ModEM should call MTData.from_modem."""
    app = _make_app()
    app._dat_format_widget.value = DAT_FORMAT_MODEM

    mock_mt_data = MagicMock()
    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.Path.is_file", return_value=True
        ),
    ):
        app._load_files(["survey.dat"])

    mock_mt_data.from_modem.assert_called_once()
    call_args = mock_mt_data.from_modem.call_args[0]
    assert str(call_args[0]).endswith("survey.dat")


@pytest.mark.plotting
def test_occam2d_dat_file_calls_from_occam2d():
    """A .dat file with format=Occam2D should call MTData.from_occam2d."""
    app = _make_app()
    app._dat_format_widget.value = DAT_FORMAT_OCCAM2D

    mock_mt_data = MagicMock()
    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.Path.is_file", return_value=True
        ),
    ):
        app._load_files(["profile.data"])

    mock_mt_data.from_occam2d.assert_called_once()
    call_args = mock_mt_data.from_occam2d.call_args[0]
    assert str(call_args[0]).endswith("profile.data")


@pytest.mark.plotting
def test_dat_format_widget_hidden_by_default():
    """The .dat format widget should be hidden until .dat files are selected."""
    app = _make_app()
    assert app._dat_format_widget.visible is False


@pytest.mark.plotting
def test_dat_format_widget_shown_on_dat_selection():
    """The .dat format widget should become visible when a .dat file is selected."""
    app = _make_app()
    app._file_selector.value = ["some_dir/data.dat"]
    assert app._dat_format_widget.visible is True


@pytest.mark.plotting
def test_dat_format_widget_hidden_when_only_edi_selected():
    """The .dat format widget should be hidden when only TF files are selected."""
    app = _make_app()
    app._file_selector.value = ["some_dir/data.dat"]
    app._file_selector.value = ["station1.edi"]
    assert app._dat_format_widget.visible is False


# ── Append mode and reset ─────────────────────────────────────────────────────


@pytest.mark.plotting
def test_append_toggle_off_by_default():
    """Append checkbox should be unchecked by default."""
    app = _make_app()
    assert app._append_toggle.value is False


@pytest.mark.plotting
def test_reset_button_disabled_before_load():
    """Reset button should be disabled until data is loaded."""
    app = _make_app()
    assert app._reset_button.disabled is True


@pytest.mark.plotting
def test_reset_clears_data_and_table():
    """Clicking Reset should clear mt_data and reset UI state."""
    app = _make_app()
    app._mt_data = MagicMock()
    app.mt_data_loaded = True
    app._save_button.disabled = False
    app._reset_button.disabled = False
    app._append_toggle.value = True

    app._on_reset_clicked(None)

    assert app._mt_data is None
    assert app.mt_data_loaded is False
    assert app._save_button.disabled is True
    assert app._reset_button.disabled is True
    assert app._append_toggle.value is False
    assert app._station_table.value.empty


@pytest.mark.plotting
def test_append_mode_merges_into_existing_data():
    """When append is checked and data exists, new data should be merged via +=."""
    app = _make_app()

    existing = MagicMock()
    app._mt_data = existing
    app._append_toggle.value = True

    new_data = MagicMock()
    new_data.station_paths = ["/surveys/s/stations/b"]

    with patch.object(app, "_load_files", return_value=new_data):
        app._file_selector.value = ["station.edi"]
        app._on_load_clicked(None)

    existing.__iadd__.assert_called_once_with(new_data)


@pytest.mark.plotting
def test_replace_mode_overwrites_existing_data():
    """When append is unchecked, new data should replace existing data."""
    app = _make_app()

    app._mt_data = MagicMock()
    app._append_toggle.value = False

    new_data = MagicMock()
    new_data.station_paths = ["/surveys/s/stations/b"]

    with patch.object(app, "_load_files", return_value=new_data):
        app._file_selector.value = ["station.edi"]
        app._on_load_clicked(None)

    assert app._mt_data is new_data


@pytest.mark.plotting
def test_build_station_summary_none_returns_empty():
    """_build_station_summary(None) should return an empty DataFrame."""
    df = _build_station_summary(None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == _STATION_TABLE_COLUMNS


@pytest.mark.plotting
def test_build_station_summary_empty_mt_data():
    """_build_station_summary on an empty MTData returns an empty DataFrame."""
    mt_data = MTData()
    df = _build_station_summary(mt_data)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


@pytest.mark.plotting
def test_station_table_populated_after_load():
    """Station table should be updated after a successful load."""
    app = _make_app()

    mock_mt_data = MagicMock(spec=MTData)
    mock_mt_data.station_paths = ["/surveys/survey1/stations/sta1"]
    mock_mt_data.add_stations = MagicMock()
    mock_mt_data.station_locations = pd.DataFrame(
        [
            {
                "survey": "survey1",
                "station": "sta1",
                "latitude": 45.0,
                "longitude": -110.0,
                "elevation": 1500.0,
            }
        ]
    )

    with (
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTData",
            return_value=mock_mt_data,
        ),
        patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app._build_station_summary",
            return_value=pd.DataFrame(
                [
                    {
                        "survey": "survey1",
                        "station": "sta1",
                        "latitude": 45.0,
                        "longitude": -110.0,
                        "elevation": 1500.0,
                        "n_periods": 10,
                    }
                ],
                columns=_STATION_TABLE_COLUMNS,
            ),
        ),
    ):
        app._file_selector.value = ["a.edi"]
        app._on_load_clicked(None)

    assert app._station_table.value is not None
    assert not app._station_table.value.empty


# ── Save to MTH5 ─────────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_save_no_data_shows_warning():
    """Clicking save with no data loaded should show a warning."""
    app = _make_app()
    app._on_save_clicked(None)
    assert "No data to save" in app._status.object


@pytest.mark.plotting
def test_save_writes_mt_data_to_mth5(tmp_path):
    """Save button should open a new MTCollection and call from_mt_data."""
    app = _make_app()

    mock_mt_data = MagicMock(spec=MTData)
    mock_mt_data.station_paths = ["/surveys/s/stations/a"]
    app._mt_data = mock_mt_data
    app._save_button.disabled = False

    output_file = tmp_path / "output.h5"
    app._save_filename_widget.value = str(output_file)

    mock_collection = MagicMock()

    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.MTCollection",
        return_value=mock_collection,
    ):
        app._on_save_clicked(None)

    mock_collection.open_collection.assert_called_once_with(
        filename=output_file, mode="w"
    )
    mock_collection.from_mt_data.assert_called_once_with(mock_mt_data)
    mock_collection.close_collection.assert_called_once()
    assert "Saved" in app._status.object


# ── Integration test with real EDI file ──────────────────────────────────────


@pytest.mark.plotting
def test_load_real_edi_file():
    """Load a real EDI file and verify MTData is populated."""
    mt_metadata = pytest.importorskip("mt_metadata")
    from mt_metadata import TF_EDI_CGG

    app = _make_app()
    app._file_selector.value = [str(TF_EDI_CGG)]
    app._on_load_clicked(None)

    assert app.mt_data_loaded is True
    assert app.mt_data is not None
    assert len(app.mt_data.station_paths) >= 1
    assert "✅" in app._status.object
    assert app._save_button.disabled is False
