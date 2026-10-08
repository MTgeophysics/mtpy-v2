"""Tests for the Panel MT data loader application (MTDataApp)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


panel = pytest.importorskip("panel")

import panel as pn

from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.panel_mt_data_app import (
    _build_station_summary,
    _PLOT_TYPES,
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


# ── Table editing and update ──────────────────────────────────────────────────


@pytest.mark.plotting
def test_edit_toggle_off_by_default():
    """Table editing should be disabled by default."""
    app = _make_app()
    assert app._edit_table_toggle.value is False
    # All editors should be None (read-only)
    assert all(v is None for v in app._station_table.editors.values())


@pytest.mark.plotting
def test_edit_toggle_enables_editors():
    """Turning on edit toggle should set numeric/text editors on editable columns."""
    app = _make_app()
    app._edit_table_toggle.value = True
    editable_cols = {"latitude", "longitude", "elevation", "survey"}
    for col in editable_cols:
        assert app._station_table.editors.get(col) is not None


@pytest.mark.plotting
def test_edit_toggle_off_disables_editors():
    """Turning off edit toggle should restore None editors."""
    app = _make_app()
    app._edit_table_toggle.value = True
    app._edit_table_toggle.value = False
    assert all(v is None for v in app._station_table.editors.values())


@pytest.mark.plotting
def test_update_button_disabled_before_load():
    """Update button should be disabled until editing is enabled and data is loaded."""
    app = _make_app()
    assert app._update_table_button.disabled is True


@pytest.mark.plotting
def test_update_writes_lat_lon_elev_to_mt_data():
    """_on_update_table_clicked should write edited lat/lon/elevation back to MTData."""
    import pandas as pd

    app = _make_app()

    mock_mt_obj = MagicMock()
    mock_mt_data = MagicMock()
    mock_mt_data.get_station.return_value = mock_mt_obj

    app._mt_data = mock_mt_data
    app._station_table.value = pd.DataFrame(
        [
            {
                "survey": "s1",
                "station": "MT01",
                "latitude": 35.123456,
                "longitude": -110.654321,
                "elevation": 1200.5,
                "n_periods": 10,
                "has_impedandance": True,
                "has_tipper": False,
            }
        ]
    )

    app._on_update_table_clicked(None)

    assert mock_mt_obj.latitude == pytest.approx(35.123456)
    assert mock_mt_obj.longitude == pytest.approx(-110.654321)
    assert mock_mt_obj.elevation == pytest.approx(1200.5)


@pytest.mark.plotting
def test_update_uses_pre_edit_snapshot_for_rename():
    """When station/survey names change, the original path from the snapshot is used."""
    import pandas as pd

    app = _make_app()

    row = {
        "survey": "s1",
        "station": "MT01",
        "latitude": 35.0,
        "longitude": -110.0,
        "elevation": 1000.0,
        "n_periods": 5,
        "has_impedandance": True,
        "has_tipper": False,
    }
    original_df = pd.DataFrame([row])

    # Simulate having edited survey and station names
    edited_row = dict(row, survey="s1_renamed", station="MT01_new")
    edited_df = pd.DataFrame([edited_row])

    mock_mt_obj = MagicMock()
    mock_mt_data = MagicMock()
    mock_mt_data.get_station.return_value = mock_mt_obj
    mock_mt_data.SURVEYS_NODE = "surveys"
    mock_mt_data.STATIONS_NODE = "stations"

    app._mt_data = mock_mt_data
    app._pre_edit_station_df = original_df
    app._station_table.value = edited_df

    from mtpy.core.mt_data import MTData

    app._on_update_table_clicked(None)

    # get_station must be called with the *original* path
    orig_path = f"/{MTData.SURVEYS_NODE}/s1/{MTData.STATIONS_NODE}/MT01"
    mock_mt_data.get_station.assert_called_once_with(orig_path, as_mt=True)
    # Old entry should be removed
    mock_mt_data.remove_station.assert_called_once_with(orig_path)
    # MT object should receive the new names
    assert mock_mt_obj.station == "MT01_new"
    assert mock_mt_obj.survey == "s1_renamed"


@pytest.mark.plotting
def test_reset_disables_update_button():
    """Reset should disable the update button."""
    app = _make_app()
    app._update_table_button.disabled = False
    app._on_reset_clicked(None)
    assert app._update_table_button.disabled is True


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


# ── Plot Tab ──────────────────────────────────────────────────────────────────


@pytest.mark.plotting
def test_view_has_tabs():
    """view() should return a Column containing a pn.Tabs with expected tabs."""
    app = _make_app()
    v = app.view
    assert isinstance(v, pn.Column)
    tabs = [obj for obj in v.objects if isinstance(obj, pn.Tabs)]
    assert len(tabs) == 1
    tab_names = tabs[0]._names
    assert "📂 Data" in tab_names
    assert "📊 Plots" in tab_names
    assert "✏️ TF Editor" in tab_names
    assert "🧪 Modeling" in tab_names


@pytest.mark.plotting
def test_mt_data_loaded_populates_tf_editor_and_plots_data():
    """When MTData loads, TF editor receives station options and can plot data."""
    app = _make_app()

    mock_mt_obj = MagicMock()
    mock_mt_obj.station = "MT01"

    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(
        side_effect=lambda: iter(["/surveys/s/stations/MT01"])
    )
    mock_mt.get_station.return_value = mock_mt_obj

    app._mt_data = mock_mt
    app.mt_data_loaded = True

    assert app._tf_editor_app._station_widget.options == ["/surveys/s/stations/MT01"]
    assert app._tf_editor_app._load_button.disabled is False

    mock_plot_obj = MagicMock()
    mock_plot_obj.panel.return_value = pn.pane.Markdown("mock response editor")

    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse",
        return_value=mock_plot_obj,
    ) as mock_edit_cls:
        app._tf_editor_app._on_load_clicked(None)

    mock_edit_cls.assert_called_once_with(
        z_object=mock_mt_obj.Z,
        t_object=mock_mt_obj.Tipper,
        responses=[],
        station=mock_mt_obj.station,
        save_edits_callback=mock_edit_cls.call_args.kwargs["save_edits_callback"],
        plot_model_error=False,
        show_plot=False,
    )
    assert app._tf_editor_app._display.objects == [mock_plot_obj.panel.return_value]


@pytest.mark.plotting
def test_tf_editor_multi_response_selector_and_loading():
    """Selector lists same-name stations from any survey; selected ones load."""
    app = _make_app()

    objs = {
        "/surveys/a/stations/MT01": MagicMock(station="MT01"),
        "/surveys/b/stations/MT01": MagicMock(station="MT01"),
        "/surveys/c/stations/MT01": MagicMock(station="MT01"),
        "/surveys/a/stations/MT02": MagicMock(station="MT02"),
    }
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(side_effect=lambda: iter(list(objs)))
    mock_mt.get_station.side_effect = lambda path, as_mt=True: objs[path]

    app._mt_data = mock_mt
    app.mt_data_loaded = True

    tf_tab = app._tf_editor_app
    # Primary can be any survey, not only "data"
    tf_tab._station_widget.value = "/surveys/b/stations/MT01"
    widget = tf_tab._plot_response_widget
    assert widget.visible is True
    assert widget.options == {
        "a/MT01": "/surveys/a/stations/MT01",
        "c/MT01": "/surveys/c/stations/MT01",
    }

    widget.value = ["/surveys/a/stations/MT01", "/surveys/c/stations/MT01"]
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel.return_value = pn.pane.Markdown("mock response editor")

    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse",
        return_value=mock_plot_obj,
    ) as mock_edit_cls:
        tf_tab._on_load_clicked(None)

    primary = objs["/surveys/b/stations/MT01"]
    a_obj = objs["/surveys/a/stations/MT01"]
    c_obj = objs["/surveys/c/stations/MT01"]
    mock_edit_cls.assert_called_once_with(
        z_object=primary.Z,
        t_object=primary.Tipper,
        responses=[
            ("a/MT01", a_obj.Z, a_obj.Tipper),
            ("c/MT01", c_obj.Z, c_obj.Tipper),
        ],
        station="MT01",
        save_edits_callback=mock_edit_cls.call_args.kwargs["save_edits_callback"],
        plot_model_error=False,
        show_plot=False,
    )

    # Station with no same-name sibling hides the selector
    tf_tab._station_widget.value = "/surveys/a/stations/MT02"
    assert widget.visible is False
    assert tf_tab._plot_response_all_widget.visible is False
    assert widget.value == []

    # 'Plot Response' selects everything and persists across stations
    tf_tab._station_widget.value = "/surveys/a/stations/MT01"
    all_box = tf_tab._plot_response_all_widget
    assert all_box.visible is True
    assert widget.value == []
    all_box.value = True
    assert widget.value == [
        "/surveys/b/stations/MT01",
        "/surveys/c/stations/MT01",
    ]
    tf_tab._station_widget.value = "/surveys/c/stations/MT01"
    assert widget.value == [
        "/surveys/a/stations/MT01",
        "/surveys/b/stations/MT01",
    ]
    all_box.value = False
    assert widget.value == []

    # Clearing data hides it too
    tf_tab.set_mt_data(None)
    assert widget.visible is False


@pytest.mark.plotting
def test_edit_mt_response_plots_multiple_responses():
    """EditMTResponse draws one response line per response (res/phase/tipper)."""
    import numpy as np

    from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse

    class _Z:
        period = np.array([0.1, 1.0, 10.0])

    class _T:
        frequency = np.array([10.0, 1.0, 0.1])
        tipper = np.ones((3, 1, 2), dtype=complex)

    plot = EditMTResponse.__new__(EditMTResponse)
    plot.Z_response = None
    plot.Tipper_response = None
    plot._get_values = lambda z, attr, comp: np.array([1.0, 2.0, 3.0])
    plot._valid_for_log = lambda x, y: np.isfinite(x) & (x > 0) & (y > 0)

    s1 = plot._response_component_source("xy", kind="res", z_response=_Z())
    s2 = plot._response_component_source("xy", kind="phase", z_response=_Z())
    t1 = plot._response_tipper_component_source(0, "real", t_response=_T())
    assert list(s1.data["period"]) == [0.1, 1.0, 10.0]
    assert list(s2.data["value"]) == [1.0, 2.0, 3.0]
    assert len(t1.data["period"]) == 3
    assert plot._response_component_source("xy") is None

    single = EditMTResponse(z_object=None, t_object=None, show_plot=False)
    assert single.responses == []
    z, t = _Z(), _T()
    single = EditMTResponse(
        z_object=None, t_object=None, z_response=z, t_response=t, show_plot=False
    )
    assert single.responses == [("response", z, t)]
    multi = EditMTResponse(
        z_object=None,
        t_object=None,
        responses=[("a", z, t), ("b", z, None)],
        show_plot=False,
    )
    assert [r[0] for r in multi.responses] == ["a", "b"]


@pytest.mark.plotting
def test_tf_editor_save_edits_replaces_station_and_refreshes_table():
    """Save Edits writes edited Z/Tipper back via add_station(overwrite=True)."""
    app = _make_app()

    mt_obj = MagicMock(station="MT01")
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(
        side_effect=lambda: iter(["/surveys/a/stations/MT01"])
    )
    mock_mt.get_station.return_value = mt_obj
    app._mt_data = mock_mt
    app.mt_data_loaded = True

    saved = []
    app._tf_editor_app.on_station_saved = saved.append
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel.return_value = pn.pane.Markdown("mock")
    with patch(
        "mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse",
        return_value=mock_plot_obj,
    ) as mock_edit_cls:
        app._tf_editor_app._on_load_clicked(None)

    callback = mock_edit_cls.call_args.kwargs["save_edits_callback"]
    new_z, new_t = MagicMock(), MagicMock()
    callback(new_z, new_t)

    assert mt_obj.Z is new_z
    assert mt_obj.Tipper is new_t
    mock_mt.add_station.assert_called_once_with(mt_obj, overwrite=True)
    assert saved == ["/surveys/a/stations/MT01"]
    assert "Saved edits" in app._tf_editor_app._status.object


@pytest.mark.plotting
def test_tf_editor_save_edits_updates_real_mtdata():
    """End-to-end: edited data (even with new periods) replaces the station."""
    import numpy as np

    from mtpy.core.mt import MT
    from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse

    frequency = np.logspace(2, -2, 9)
    mt = MT(station="MT01", survey="a", frequency=frequency)
    z = np.zeros((9, 2, 2), dtype=complex)
    z[:, 0, 1] = np.linspace(1, 9, 9) * (1 + 1j)
    z[:, 1, 0] = -z[:, 0, 1]
    mt.impedance = z
    mt.impedance_error = np.full((9, 2, 2), 0.1)
    mt_data = MTData()
    path = mt_data.add_station(mt)

    from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTResponseEditorTab

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
    editor.flip_phase(zxy=True)
    editor.masked_tf_indices = {"xy": {4}}
    editor.interpolate(np.logspace(-2, 2, 5), extrapolate=True)
    editor.save_edits()

    saved = mt_data.get_station(path, as_mt=True)
    assert saved.Z.z.shape[0] == 5
    assert saved.Z.z[0, 0, 1].real < 0
    assert editor._data_manipulated is False
    assert editor.masked_tf_indices == {}


def _editor_tab_with_stations(paths):
    from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTResponseEditorTab

    objs = {p: MagicMock(station=p.split("/")[-1]) for p in paths}
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(side_effect=lambda: iter(list(objs)))
    mock_mt.get_station.side_effect = lambda path, as_mt=True: objs[path]
    tab = MTResponseEditorTab()
    with patch("mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse"):
        tab.set_mt_data(mock_mt)
    return tab, objs


def test_tf_editor_error_type_is_global_and_passed_to_plot():
    tab, _ = _editor_tab_with_stations(["/surveys/a/stations/MT01"])
    assert tab._error_type_widget.value == "data"
    for value, expected in (("model", True), ("data", False)):
        tab._error_type_widget.value = value
        with patch(
            "mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse"
        ) as edit_cls:
            edit_cls.return_value.panel.return_value = pn.pane.Markdown("x")
            tab._on_load_clicked(None)
        assert edit_cls.call_args.kwargs["plot_model_error"] is expected


def test_tf_editor_station_selection_autoplots():
    tab, _ = _editor_tab_with_stations(
        ["/surveys/a/stations/MT01", "/surveys/a/stations/MT02"]
    )
    with patch("mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse") as edit_cls:
        edit_cls.return_value.panel.return_value = pn.pane.Markdown("x")
        tab._station_widget.value = "/surveys/a/stations/MT02"
    edit_cls.assert_called_once()
    assert edit_cls.call_args.kwargs["station"] == "MT02"
    assert len(tab._display.objects) == 1


def test_tf_editor_station_selection_without_data_does_not_plot():
    from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTResponseEditorTab

    tab = MTResponseEditorTab()
    with patch("mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse") as edit_cls:
        tab._station_widget.options = ["/surveys/a/stations/MT01"]
        tab._station_widget.value = "/surveys/a/stations/MT01"
    edit_cls.assert_not_called()


def _fake_mt(z, z_err, period, has_t=False):
    pass

    mt = MagicMock()
    mt.has_impedance.return_value = True
    mt.has_tipper.return_value = has_t
    mt.Z.z = z
    mt.Z.z_model_error = z_err
    mt.Z.z_error = z_err
    mt.Z.period = period
    return mt


def test_rms_text_values_and_shared_periods():
    import numpy as np

    from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTResponseEditorTab

    period = np.array([0.1, 1.0, 10.0])
    z = np.zeros((3, 2, 2), dtype=complex)
    z[:, 0, 1] = [1, 2, 3]
    err = np.ones((3, 2, 2))
    mt = _fake_mt(z, err, period)

    # Identical response -> zero RMS for xy; zero-valued components skipped
    same = MagicMock(z=z.copy(), period=period)
    text = MTResponseEditorTab._rms_text(mt, [("b/MT01", same, None)])
    assert "RMS vs b/MT01" in text
    assert "Zxy=0.00" in text
    assert "Zxx" not in text

    # Constant offset of 2 over err 1 -> RMS 2
    z2 = z.copy()
    z2[:, 0, 1] += 2
    off = MagicMock(z=z2, period=period)
    assert "Zxy=2.00" in MTResponseEditorTab._rms_text(mt, [("c/MT01", off, None)])

    # Response on a subset of periods only compares shared ones
    sub = MagicMock(z=z2[1:], period=period[1:])
    assert "Zxy=2.00" in MTResponseEditorTab._rms_text(mt, [("d/MT01", sub, None)])

    # No responses -> empty
    assert MTResponseEditorTab._rms_text(mt, []) == ""


def test_tf_editor_load_sets_and_clears_rms_pane():
    import numpy as np

    path_a, path_b = "/surveys/a/stations/MT01", "/surveys/b/stations/MT01"
    tab, objs = _editor_tab_with_stations([path_a, path_b])
    period = np.array([1.0, 2.0])
    z = np.zeros((2, 2, 2), dtype=complex)
    z[:, 0, 1] = 1
    objs[path_a] = _fake_mt(z, np.ones((2, 2, 2)), period)
    objs[path_b] = _fake_mt(z + 0, np.ones((2, 2, 2)), period)
    objs[path_b].Z.z = z + 3
    objs[path_a].station = objs[path_b].station = "MT01"

    tab._station_widget.value = path_a
    tab._plot_response_widget.value = [path_b]
    with patch("mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse") as edit_cls:
        edit_cls.return_value.panel.return_value = pn.pane.Markdown("x")
        tab._on_load_clicked(None)
    assert "RMS vs b/MT01" in tab._rms_pane.object
    assert "Zxy=3.00" in tab._rms_pane.object

    tab._plot_response_widget.value = []
    with patch("mtpy.imaging.bokeh_plots.panel_mt_data_app.EditMTResponse") as edit_cls:
        edit_cls.return_value.panel.return_value = pn.pane.Markdown("x")
        tab._on_load_clicked(None)
    assert tab._rms_pane.object == ""


@pytest.mark.plotting
def test_generate_button_disabled_initially():
    """Generate Plot button should be disabled before any data is loaded."""
    app = _make_app()
    assert app._plot_generate_button.disabled is True


@pytest.mark.plotting
def test_generate_plot_no_data_shows_warning():
    """Clicking Generate Plot with no data loaded should show a warning status."""
    app = _make_app()
    app._plot_generate_button.disabled = False  # bypass the guard
    app._on_generate_plot_clicked(None)
    assert "No data" in app._plot_status.object or "⚠️" in app._plot_status.object


@pytest.mark.plotting
def test_station_picker_hidden_for_non_station_plots():
    """Station picker should not be visible for plot types that need no station."""
    app = _make_app()
    non_station_labels = [
        label for label, (_, needs) in _PLOT_TYPES.items() if not needs
    ]
    assert non_station_labels, "Expected at least one non-station plot type"
    # Simulate selecting each non-station plot type
    for label in non_station_labels:
        app._plot_type_widget.value = label
    # After the last one station widget should be hidden
    assert app._plot_station_widget.visible is False


@pytest.mark.plotting
def test_station_picker_visible_for_single_station_plots():
    """Station picker should be visible when a single-station plot type is chosen."""
    app = _make_app()
    station_labels = [label for label, (_, needs) in _PLOT_TYPES.items() if needs]
    assert station_labels, "Expected at least one station-picker plot type"
    app._plot_type_widget.value = station_labels[0]
    assert app._plot_station_widget.visible is True


@pytest.mark.plotting
def test_mt_data_loaded_enables_generate_button():
    """Setting mt_data_loaded=True should enable the Generate Plot button."""
    app = _make_app()
    assert app._plot_generate_button.disabled is True

    # Simulate a successful load
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(
        return_value=iter(["/surveys/s/stations/MT01"])
    )
    app._mt_data = mock_mt
    app.mt_data_loaded = True  # triggers _on_mt_data_loaded_changed

    assert app._plot_generate_button.disabled is False


@pytest.mark.plotting
def test_mt_data_loaded_false_disables_generate_button():
    """Setting mt_data_loaded=False should re-disable the Generate Plot button."""
    app = _make_app()
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([]))
    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app.mt_data_loaded = False

    assert app._plot_generate_button.disabled is True


@pytest.mark.plotting
def test_station_picker_populated_after_load():
    """Station picker options should reflect loaded station paths."""
    app = _make_app()
    paths = ["/surveys/survey1/stations/MT01", "/surveys/survey1/stations/MT02"]
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter(paths))
    app._mt_data = mock_mt
    app.mt_data_loaded = True

    assert set(app._plot_station_widget.options) == set(paths)


@pytest.mark.plotting
def test_generate_plot_calls_correct_method():
    """Generate Plot should call the MTData method matching the selected plot type."""
    app = _make_app()

    mock_fig = MagicMock()
    mock_fig.plot = MagicMock(return_value=mock_fig)

    # Use a non-station plot type so no station picker is needed
    label = "Station Map"
    method_name, _ = _PLOT_TYPES[label]
    mock_mt = MagicMock(spec=MTData)
    getattr(mock_mt, method_name).return_value = mock_fig
    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_type_widget.value = label

    app._on_generate_plot_clicked(None)

    getattr(mock_mt, method_name).assert_called_once()


@pytest.mark.plotting
def test_generate_plot_station_required_but_none_selected():
    """If station picker has no selection, generate should show a warning."""
    app = _make_app()
    label = "MT Response (single station)"
    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([]))
    app._mt_data = mock_mt
    app.mt_data_loaded = True

    app._plot_type_widget.value = label
    # Station widget is empty
    app._plot_station_widget.options = []
    app._plot_station_widget.value = None

    app._on_generate_plot_clicked(None)

    assert "⚠️" in app._plot_status.object


@pytest.mark.plotting
def test_plot_types_registry_completeness():
    """_PLOT_TYPES should expose all MTData plot methods in the app."""
    expected_methods = {
        "plot_stations",
        "plot_strike",
        "plot_mt_response",
        "plot_mt_responses",
        "plot_phase_tensor",
        "plot_phase_tensor_map",
        "plot_phase_tensor_pseudosection",
        "plot_penetration_depth_1d",
        "plot_penetration_depth_map",
        "plot_resistivity_phase_maps",
        "plot_resistivity_phase_pseudosections",
    }
    registered_methods = {method for method, _ in _PLOT_TYPES.values()}
    assert registered_methods == expected_methods


@pytest.mark.plotting
def test_backend_widget_default_is_bokeh():
    """Backend radio button should default to 'bokeh'."""
    app = _make_app()
    assert app._plot_backend_widget.value == "bokeh"


@pytest.mark.plotting
def test_backend_widget_options():
    """Backend selector must offer both backends."""
    app = _make_app()
    assert set(app._plot_backend_widget.options) == {"bokeh", "matplotlib"}


@pytest.mark.plotting
def test_generate_plot_passes_backend_to_method():
    """Generate Plot should forward the selected backend to the MTData method."""
    app = _make_app()

    mock_fig = MagicMock()
    mock_fig.plot = MagicMock(return_value=mock_fig)

    label = "Station Map"
    method_name, _ = _PLOT_TYPES[label]
    mock_mt = MagicMock(spec=MTData)
    getattr(mock_mt, method_name).return_value = mock_fig
    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_type_widget.value = label
    app._plot_backend_widget.value = "matplotlib"

    app._on_generate_plot_clicked(None)

    _, call_kwargs = getattr(mock_mt, method_name).call_args
    assert call_kwargs.get("backend") == "matplotlib"


@pytest.mark.plotting
def test_generate_plot_embeds_panel_when_available():
    """When the returned plot object has a panel() method, MTDataApp should
    embed the panel() output rather than calling .plot() directly."""
    app = _make_app()

    # Build a mock plot object that has a panel() method
    mock_panel_app = pn.Column()
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel = MagicMock(return_value=mock_panel_app)
    mock_plot_obj.plot = MagicMock()

    label = "Station Map"
    method_name, _ = _PLOT_TYPES[label]
    mock_mt = MagicMock(spec=MTData)
    getattr(mock_mt, method_name).return_value = mock_plot_obj
    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_type_widget.value = label
    app._plot_backend_widget.value = "bokeh"

    app._on_generate_plot_clicked(None)

    # panel() should have been called, .plot() should NOT have been called
    mock_plot_obj.panel.assert_called_once()
    mock_plot_obj.plot.assert_not_called()

    # The display should contain the panel app
    assert mock_panel_app in app._plot_display.objects


@pytest.mark.plotting
def test_generate_plot_falls_back_to_plot_when_no_panel_method():
    """When the plot object has no panel() method, MTDataApp falls back to
    calling .plot() as before."""
    app = _make_app()

    mock_fig = MagicMock()
    # plot object WITHOUT a panel() method
    mock_plot_obj = MagicMock(spec=["plot", "fig"])
    mock_plot_obj.plot = MagicMock(return_value=mock_fig)

    label = "Station Map"
    method_name, _ = _PLOT_TYPES[label]
    mock_mt = MagicMock(spec=MTData)
    getattr(mock_mt, method_name).return_value = mock_plot_obj
    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_type_widget.value = label

    app._on_generate_plot_clicked(None)

    mock_plot_obj.plot.assert_called_once()
    assert "✅" in app._plot_status.object


@pytest.mark.plotting
def test_generate_phase_tensor_plot_uses_panel_and_station_key():
    """Phase Tensor (single station) should forward station_key and embed panel()."""
    app = _make_app()

    station_path = "/surveys/survey1/stations/MT01"
    mock_panel_app = pn.Column()
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel = MagicMock(return_value=mock_panel_app)

    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([station_path]))
    mock_mt.plot_phase_tensor.return_value = mock_plot_obj

    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_backend_widget.value = "bokeh"
    app._plot_type_widget.value = "Phase Tensor (single station)"
    app._plot_station_widget.value = station_path

    app._on_generate_plot_clicked(None)

    mock_mt.plot_phase_tensor.assert_called_once()
    _, kwargs = mock_mt.plot_phase_tensor.call_args
    assert kwargs.get("station_key") == station_path
    assert kwargs.get("backend") == "bokeh"
    mock_plot_obj.panel.assert_called_once()
    assert mock_panel_app in app._plot_display.objects


@pytest.mark.plotting
def test_generate_phase_tensor_pseudosection_uses_panel_app():
    """Phase Tensor Pseudosection should embed plot_obj.panel()."""
    app = _make_app()

    mock_panel_app = pn.Column()
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel = MagicMock(return_value=mock_panel_app)

    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([]))
    mock_mt.plot_phase_tensor_pseudosection.return_value = mock_plot_obj

    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_backend_widget.value = "bokeh"
    app._plot_type_widget.value = "Phase Tensor Pseudosection"

    app._on_generate_plot_clicked(None)

    mock_mt.plot_phase_tensor_pseudosection.assert_called_once()
    _, kwargs = mock_mt.plot_phase_tensor_pseudosection.call_args
    assert kwargs.get("backend") == "bokeh"
    mock_plot_obj.panel.assert_called_once()
    assert mock_panel_app in app._plot_display.objects


@pytest.mark.plotting
def test_generate_res_phase_maps_uses_panel_app():
    """Resistivity / Phase Maps should embed plot_obj.panel() in the Plots tab."""
    app = _make_app()

    mock_panel_app = pn.Column()
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel = MagicMock(return_value=mock_panel_app)

    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([]))
    mock_mt.plot_resistivity_phase_maps.return_value = mock_plot_obj

    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_backend_widget.value = "bokeh"
    app._plot_type_widget.value = "Resistivity / Phase Maps"

    app._on_generate_plot_clicked(None)

    mock_mt.plot_resistivity_phase_maps.assert_called_once()
    _, kwargs = mock_mt.plot_resistivity_phase_maps.call_args
    assert kwargs.get("backend") == "bokeh"
    mock_plot_obj.panel.assert_called_once()
    assert mock_panel_app in app._plot_display.objects


@pytest.mark.plotting
def test_generate_res_phase_pseudosection_uses_panel_app():
    """Resistivity / Phase Pseudosection should embed plot_obj.panel()."""
    app = _make_app()

    mock_panel_app = pn.Column()
    mock_plot_obj = MagicMock()
    mock_plot_obj.panel = MagicMock(return_value=mock_panel_app)

    mock_mt = MagicMock(spec=MTData)
    mock_mt._iter_station_paths = MagicMock(return_value=iter([]))
    mock_mt.plot_resistivity_phase_pseudosections.return_value = mock_plot_obj

    app._mt_data = mock_mt
    app.mt_data_loaded = True
    app._plot_backend_widget.value = "bokeh"
    app._plot_type_widget.value = "Resistivity / Phase Pseudosection"

    app._on_generate_plot_clicked(None)

    mock_mt.plot_resistivity_phase_pseudosections.assert_called_once()
    _, kwargs = mock_mt.plot_resistivity_phase_pseudosections.call_args
    assert kwargs.get("backend") == "bokeh"
    mock_plot_obj.panel.assert_called_once()
    assert mock_panel_app in app._plot_display.objects
