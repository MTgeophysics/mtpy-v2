"""Panel application for loading MT data into an MTData object.

This module provides a standalone Panel application that allows users to select
MT data files (transfer function files and/or an MTH5 file), load them into a
unified :class:`~mtpy.core.mt_data.MTData` object, and optionally save the
result back to an MTH5 file.  Once data is loaded the **Plots** tab exposes
all Bokeh-backed plot methods from :class:`~mtpy.core.mt_data.MTData`.

Supported transfer function file types
---------------------------------------
- EDI (``.edi``)
- XML (``.xml``)
- AVG (``.avg``)
- ZMM / ZSS / ZRR (``.zmm``, ``.zss``, ``.zrr``)
- ModEM data file (``.dat``, ``.data``) — select "ModEM" as the format
- Occam2D data file (``.dat``, ``.data``) — select "Occam2D" as the format

MTH5 files (``.h5``) are supported alongside or instead of TF files.
If an MTH5 file is selected, it is opened first via
:class:`~mtpy.core.mt_collection.MTCollection` and its stations form the base
of the :class:`~mtpy.core.mt_data.MTData`.  Any additional TF files are then
merged in.

Usage
-----
As a Panel app in a Jupyter notebook::

    >>> import panel as pn
    >>> pn.extension("tabulator", notifications=True)
    >>> from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTDataApp
    >>> app = MTDataApp()
    >>> app.template.servable()   # full app: sidebar + tabs
    >>> # app.sidebar and app.view are the two halves, if you want to embed them

As a standalone script (from terminal)::

    panel serve path/to/panel_mt_data_app.py --autoreload

Accessing the loaded data::

    >>> mt_data = app.mt_data  # MTData object, None until files are loaded

Watching for data-ready events in a downstream panel::

    >>> def on_data_loaded(event):
    ...     if event.new:
    ...         print("MTData ready:", app.mt_data)
    >>> app.param.watch(on_data_loaded, "mt_data_loaded")
"""

from __future__ import annotations

import asyncio
import inspect
import math
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure as bokeh_figure

from mtpy.core.mt_collection import MTCollection
from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.panel_mt_response_editor_tab import MTResponseEditorTab
from mtpy.imaging.bokeh_plots.panel_simpeg1d_app import Simpeg1DPanelApp
from mtpy.imaging.bokeh_plots.plot_penetration_depth_1d import PlotPenetrationDepth1D


pn.extension("tabulator", notifications=True)

_MERCATOR_RADIUS: float = 6378137.0


SUPPORTED_TF_SUFFIXES: frozenset[str] = frozenset(
    {".edi", ".xml", ".avg", ".zmm", ".zss", ".zrr"}
)
SUPPORTED_MTH5_SUFFIX: str = ".h5"
SUPPORTED_DAT_SUFFIXES: frozenset[str] = frozenset({".dat", ".data"})

DAT_FORMAT_MODEM: str = "ModEM"
DAT_FORMAT_OCCAM2D: str = "Occam2D"

SURVEY_LABEL_KEEP: str = "Keep Original"
SURVEY_LABEL_DATA: str = "data"
SURVEY_LABEL_MODEL: str = "model"
SURVEY_LABEL_CUSTOM: str = "Custom…"

SUPPORTED_FILE_PATTERNS: dict[str, str] = {
    "All MT Files (*.edi *.xml *.avg *.z* *.h5 *.dat *.data)": "*.*",
    "EDI (*.edi)": "*.edi",
    "XML (*.xml)": "*.xml",
    "AVG (*.avg)": "*.avg",
    "Z files (*.zmm, *.zss, *.zrr)": "*.z*",
    "MTH5 (*.h5)": "*.h5",
    "ModEM / Occam2D data (*.dat, *.data)": "*.dat",
}

_STATION_TABLE_COLUMNS: list[str] = [
    "survey",
    "station",
    "latitude",
    "longitude",
    "elevation",
    "n_periods",
    "has_impedandance",
    "has_tipper",
]

# Maps human-readable label → (MTData method name, needs_station_picker)
_PLOT_TYPES: dict[str, tuple[str, bool]] = {
    "Station Map": ("plot_stations", False),
    "Strike": ("plot_strike", False),
    "MT Response (single station)": ("plot_mt_response", True),
    "MT Responses (multiple stations)": ("plot_mt_responses", False),
    "Phase Tensor (single station)": ("plot_phase_tensor", True),
    "Phase Tensor Map": ("plot_phase_tensor_map", False),
    # "Tipper Map": ("plot_tipper_map", False),
    "Phase Tensor Pseudosection": ("plot_phase_tensor_pseudosection", False),
    "Penetration Depth 1D (single station)": ("plot_penetration_depth_1d", True),
    "Penetration Depth Map": ("plot_penetration_depth_map", False),
    "Resistivity / Phase Maps": ("plot_resistivity_phase_maps", False),
    "Resistivity / Phase Pseudosection": (
        "plot_resistivity_phase_pseudosections",
        False,
    ),
}


def _filesystem_root(path: str | None = None) -> str:
    """Return the root of the filesystem for the given path (drive root on Windows)."""
    p = Path(path).resolve() if path else Path.cwd().resolve()
    return str(p.anchor)  # e.g. "C:\\" on Windows, "/" on Unix


def _default_data_directory() -> str:
    """Return a sensible default browse directory."""
    workspace_root = Path(__file__).resolve().parents[4]
    candidates = [
        workspace_root / "mtpy_data" / "mtpy_data" / "data" / "grid",
        workspace_root / "mtpy_data" / "mtpy_data" / "data",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(Path.cwd())


def _build_station_summary(mt_data: "MTData") -> pd.DataFrame:
    """Build a summary DataFrame from an MTData object.

    Parameters
    ----------
    mt_data : MTData
        Populated MTData container.

    Returns
    -------
    pd.DataFrame
        One row per station with survey, station, latitude, longitude,
        elevation, and n_periods columns.
    """
    if mt_data is None:
        return pd.DataFrame(columns=_STATION_TABLE_COLUMNS)

    if len(mt_data.station_paths) == 0:
        return pd.DataFrame(columns=_STATION_TABLE_COLUMNS)

    rows = []
    for station_path in mt_data.station_paths:
        mt_obj = mt_data.get_station(station_path, as_mt=True)
        rows.append(
            {
                "survey": mt_obj.survey,
                "station": mt_obj.station,
                "latitude": round(float(mt_obj.latitude), 6),
                "longitude": round(float(mt_obj.longitude), 6),
                "elevation": round(float(mt_obj.elevation), 2),
                "n_periods": len(mt_obj.period) if mt_obj.period is not None else 0,
                "has_impedandance": mt_obj.has_impedance(),
                "has_tipper": mt_obj.has_tipper(),
            }
        )

    return pd.DataFrame(rows, columns=_STATION_TABLE_COLUMNS)


def _to_web_mercator(longitude: float, latitude: float) -> tuple[float, float]:
    """Project WGS84 lon/lat (degrees) to Web Mercator metres (clipped at ±85°)."""
    lat = max(min(float(latitude), 85.0), -85.0)
    x = math.radians(float(longitude)) * _MERCATOR_RADIUS
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * _MERCATOR_RADIUS
    return x, y


def _station_path(survey: Any, station: Any) -> str:
    """Return the MTData tree path for a survey/station pair."""
    return f"/{MTData.SURVEYS_NODE}/{survey}/{MTData.STATIONS_NODE}/{station}"


def _relabel_survey(mt_data: "MTData", new_survey: str) -> None:
    """Rename every station's survey label in place to *new_survey*."""
    for station_path in list(mt_data._iter_station_paths()):
        mt_obj = mt_data.get_station(station_path, as_mt=True)
        if mt_obj.survey == new_survey:
            continue
        mt_data.remove_station(station_path)
        mt_obj.survey = new_survey
        mt_data.add_station(mt_obj, overwrite=True)


class MTDataApp(param.Parameterized):
    """Interactive Panel application for loading MT data into an MTData object.

    Users select any combination of transfer function files and/or a single
    MTH5 file.  The selected data is merged into a single
    :class:`~mtpy.core.mt_data.MTData` object that is accessible via
    :attr:`mt_data`.

    Parameters
    ----------
    sizing_mode : str
        Panel sizing mode for the layout.
    data_directory : str
        Initial directory shown in the file selector.

    Attributes
    ----------
    mt_data : MTData or None
        The currently loaded data.  ``None`` until files have been selected
        and loaded successfully.
    mt_data_loaded : bool
        Reactive flag that becomes ``True`` after a successful load and
        ``False`` when the selection is cleared.  Downstream Panel components
        can ``param.watch`` this flag.
    """

    sizing_mode: str = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode for the application layout.",
    )

    data_directory: str = param.String(
        default=_default_data_directory(),
        doc="Initial directory shown in the file browser.",
    )

    selected_station_paths: list = param.List(
        default=[],
        doc="Station paths selected in the table or on the map. Shared by all "
        "tabs; an empty list means 'no selection' (use every station).",
    )

    mt_data_loaded: bool = param.Boolean(
        default=False,
        doc="True when mt_data has been populated successfully.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._mt_data = None  # MTData | None
        self._mt_collection = None  # MTCollection | None
        self._pre_edit_station_df = None  # snapshot taken when editing is enabled
        self._selected_station_paths: list[str] = []
        self._mt_data_subset = None

        # ── File type filter ──────────────────────────────────────────────
        self._file_pattern_widget = pn.widgets.Select(
            name="File Type Filter",
            options=SUPPORTED_FILE_PATTERNS,
            value="*.*",
            width=320,
        )

        # ── File selector ─────────────────────────────────────────────────
        self._file_selector = pn.widgets.FileSelector(
            directory=str(Path(self.data_directory).expanduser().resolve()),
            root_directory=_filesystem_root(self.data_directory),
            name="Select MT Data Files",
            file_pattern=self._file_pattern_widget.value,
            only_files=True,
        )

        # ── Load button ───────────────────────────────────────────────────
        self._append_toggle = pn.widgets.Checkbox(
            name="Append to existing data (uncheck to replace)",
            value=False,
        )
        self._load_button = pn.widgets.Button(
            name="Load Selected Files",
            button_type="primary",
            width=200,
        )
        self._load_button.on_click(self._on_load_clicked_async)

        # ── Reset button ──────────────────────────────────────────────────
        self._reset_button = pn.widgets.Button(
            name="Reset",
            button_type="warning",
            width=100,
            disabled=True,
        )
        self._reset_button.on_click(self._on_reset_clicked)

        # ── .dat/.data format selector (visible only when such files selected)
        self._dat_format_widget = pn.widgets.RadioButtonGroup(
            name="Format for .dat / .data files",
            options=[DAT_FORMAT_MODEM, DAT_FORMAT_OCCAM2D],
            value=DAT_FORMAT_MODEM,
            button_type="default",
            visible=False,
        )

        # ── Survey label override (applies to any selected file type) ─────
        self._survey_preset_widget = pn.widgets.Select(
            name="Survey label",
            options=[
                SURVEY_LABEL_KEEP,
                SURVEY_LABEL_DATA,
                SURVEY_LABEL_MODEL,
                SURVEY_LABEL_CUSTOM,
            ],
            value=SURVEY_LABEL_KEEP,
            width=160,
        )
        self._survey_custom_widget = pn.widgets.TextInput(
            name="Custom survey name",
            placeholder="e.g. line1",
            width=160,
            visible=False,
        )
        self._survey_preset_widget.param.watch(self._on_survey_preset_changed, "value")
        self._file_selector.param.watch(self._on_file_selection_changed, "value")

        # ── Status display ────────────────────────────────────────────────
        self._status = pn.pane.Markdown(
            "_No files loaded yet._",
            styles={"color": "#555"},
        )

        # ── Station summary table ─────────────────────────────────────────
        self._edit_table_toggle = pn.widgets.Checkbox(
            name="Enable table editing",
            value=False,
        )
        self._update_table_button = pn.widgets.Button(
            name="Apply Edits to MTData",
            button_type="primary",
            width=200,
            disabled=True,
        )
        self._station_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=_STATION_TABLE_COLUMNS),
            name="Loaded Stations",
            selectable="checkbox",
            pagination="local",
            page_size=20,
            sizing_mode="stretch_width",
            show_index=False,
            editors={col: None for col in _STATION_TABLE_COLUMNS},
            configuration={"columnDefaults": {"headerFilter": True}},
        )
        self._station_table.param.watch(self._on_table_selection_changed, "selection")
        self._edit_table_toggle.param.watch(self._on_edit_toggle_changed, "value")
        self._update_table_button.on_click(self._on_update_table_clicked)

        # ── Penetration depth preview (single selected station) ──────────
        self._pen_depth_container = pn.Column(
            pn.pane.Markdown(
                "_Select a single station in the table above to view its "
                "1-D penetration depth._",
                styles={"color": "#777"},
            ),
            sizing_mode=self.sizing_mode,
        )

        # ── Save-to-MTH5 controls ─────────────────────────────────────────
        self._save_filename_widget = pn.widgets.TextInput(
            name="Output MTH5 filename",
            value=str(Path(self._file_selector.directory) / "mt_collection.h5"),
            placeholder="e.g. my_survey.h5",
            width=320,
        )
        self._save_button = pn.widgets.Button(
            name="Save to MTH5",
            button_type="success",
            width=160,
            disabled=True,
        )
        self._save_button.on_click(self._on_save_clicked)

        # The save folder follows the folder browsed in the data file selector.
        self._file_selector.param.watch(self._on_save_directory_changed, "directory")

        # ── Wire file-pattern and directory changes ───────────────────────
        self._file_pattern_widget.param.watch(self._on_file_pattern_changed, "value")
        self.param.watch(self._on_data_directory_changed, "data_directory")

        # ── Plot tab widgets ──────────────────────────────────────────────
        self._plot_type_widget = pn.widgets.Select(
            name="Plot Type",
            options=list(_PLOT_TYPES.keys()),
            value="Station Map",
            width=320,
        )
        self._plot_backend_widget = pn.widgets.RadioButtonGroup(
            name="Backend",
            options=["bokeh", "matplotlib"],
            value="bokeh",
            button_type="default",
            width=200,
        )
        self._plot_station_widget = pn.widgets.Select(
            name="Station",
            options=[],
            visible=False,
            width=320,
        )
        self._plot_generate_button = pn.widgets.Button(
            name="Generate Plot",
            button_type="primary",
            width=160,
            disabled=True,
        )
        self._plot_status = pn.pane.Markdown(
            "_Load data first, then select a plot type and click **Generate Plot**._",
            styles={"color": "#555"},
        )
        self._plot_display = pn.Column(sizing_mode="stretch_width")

        self._plot_type_widget.param.watch(self._on_plot_type_changed, "value")
        self._plot_generate_button.on_click(self._on_generate_plot_async)
        self.param.watch(self._on_mt_data_loaded_changed, "mt_data_loaded")

        self._plot_auto_widget = pn.widgets.Checkbox(
            name="Auto-update plot", value=True
        )
        self._plot_selected_only_widget = pn.widgets.Checkbox(
            name="Selected stations only", value=True
        )
        self._suspend_auto_plot = False
        # Re-render whenever any plot input (type, station, backend, selection)
        # changes; the Generate button remains as a manual refresh.
        pn.bind(
            self._on_plot_inputs_changed,
            self._plot_type_widget,
            self._plot_station_widget,
            self._plot_backend_widget,
            self._plot_selected_only_widget,
            self.param.selected_station_paths,
            watch=True,
        )

        # ── Station map linked to the table ───────────────────────────────
        self._syncing_selection = False
        self._station_source = ColumnDataSource(
            data={"x": [], "y": [], "station": [], "survey": [], "path": []}
        )
        self._station_source.selected.on_change("indices", self._on_map_selection)
        self._station_map_figure = self._build_station_map_figure()
        self._station_map_pane = pn.pane.Bokeh(
            self._station_map_figure, sizing_mode="stretch_both", min_height=420
        )
        self._selection_label = pn.pane.Markdown(
            "_No stations selected — plots use all stations._",
            styles={"color": "#777", "font-size": "0.85em"},
        )
        self._clear_selection_button = pn.widgets.Button(
            name="Clear selection", button_type="light", width=140
        )
        self._clear_selection_button.on_click(
            lambda event: self._apply_selection([], origin="clear")
        )

        # ── Modeling tab app ─────────────────────────────────────────────
        self._modeling_app = Simpeg1DPanelApp(sizing_mode=self.sizing_mode)
        self._tf_editor_app = MTResponseEditorTab(sizing_mode=self.sizing_mode)
        self._tf_editor_app.on_station_saved = self._on_editor_station_saved

    def _on_editor_station_saved(self, station_key: str) -> None:
        """Refresh the station table after the TF editor replaces a station."""
        if self._mt_data is not None:
            self._update_station_table(self._mt_data)

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def mt_data(self) -> MTData | None:
        """The currently loaded :class:`~mtpy.core.mt_data.MTData` object."""
        return self._mt_data

    # ── Widget callbacks ──────────────────────────────────────────────────

    def _on_data_directory_changed(self, event: param.parameterized.Event) -> None:
        """Update the file selector when data_directory param changes."""
        resolved = str(Path(event.new).expanduser().resolve())
        if Path(resolved).is_dir():
            self._file_selector.directory = resolved
            self._directory_input.value = resolved

    def _on_directory_go_clicked(self, event: param.parameterized.Event) -> None:
        """Navigate the file selector to the typed directory path."""
        self._navigate_to_directory(self._directory_input.value)

    def _on_directory_input_enter(self, event: param.parameterized.Event) -> None:
        """Navigate when the user presses Enter in the directory text box."""
        self._navigate_to_directory(self._directory_input.value)

    def _navigate_to_directory(self, raw_path: str) -> None:
        """Resolve *raw_path* and update the file selector directory."""
        path = Path(raw_path.strip()).expanduser().resolve()
        if path.is_dir():
            self._file_selector.directory = str(path)
            self._file_selector._update_files()
            self._directory_input.value = str(path)
            self._set_status(f"📂 Browsing `{path}`")
        else:
            self._set_status(
                f"⚠️ Directory not found: `{raw_path.strip()}`", warning=True
            )

    def _on_file_pattern_changed(self, event: param.parameterized.Event) -> None:
        """Update the file selector filter pattern."""
        self._file_selector.file_pattern = event.new

    def _on_file_selection_changed(self, event: param.parameterized.Event) -> None:
        """Show the .dat/.data format picker only when such files are selected."""
        has_dat = any(
            Path(f).suffix.lower() in SUPPORTED_DAT_SUFFIXES for f in (event.new or [])
        )
        self._dat_format_widget.visible = has_dat

    def _on_survey_preset_changed(self, event: param.parameterized.Event) -> None:
        """Reveal the custom survey name field only when 'Custom…' is selected."""
        self._survey_custom_widget.visible = event.new == SURVEY_LABEL_CUSTOM

    def _current_survey_override(self) -> str | None:
        """Resolve the survey label override to apply to all newly loaded stations.

        Returns ``None`` when the user chose to keep each file's own survey
        label (e.g. ModEM default "data", an EDI's own survey attribute).
        """
        preset = self._survey_preset_widget.value
        if preset == SURVEY_LABEL_KEEP:
            return None
        if preset == SURVEY_LABEL_CUSTOM:
            custom = self._survey_custom_widget.value.strip()
            return custom if custom else SURVEY_LABEL_DATA
        return preset

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        """Load selected files into MTData (blocking)."""
        selected = self._begin_load()
        if selected is None:
            return
        try:
            new_data = self._load_files(selected)
        except Exception as exc:
            self._fail_load(exc)
        else:
            self._complete_load(new_data)
        finally:
            self._end_load()

    async def _on_load_clicked_async(self, event: param.parameterized.Event) -> None:
        """Load selected files in a worker thread so the UI stays responsive."""
        selected = self._begin_load()
        if selected is None:
            return
        try:
            new_data = await asyncio.to_thread(self._load_files, selected)
        except Exception as exc:
            self._fail_load(exc)
        else:
            self._complete_load(new_data)
        finally:
            self._end_load()

    def _begin_load(self) -> list[str] | None:
        """Validate the selection and show the loading state."""
        selected: list[str] = list(self._file_selector.value or [])
        if not selected:
            self._set_status("⚠️ No files selected.", warning=True)
            return None
        self._set_status("⏳ Loading files…")
        self._load_button.disabled = True
        self._station_table.loading = True
        self._station_map_pane.loading = True
        return selected

    def _end_load(self) -> None:
        self._load_button.disabled = False
        self._station_table.loading = False
        self._station_map_pane.loading = False

    def _complete_load(self, new_data: MTData) -> None:
        """Install freshly loaded data and refresh every dependent view."""
        try:
            if self._append_toggle.value and self._mt_data is not None:
                self._mt_data += new_data
                mt_data = self._mt_data
            else:
                self._mt_data = new_data
                mt_data = new_data
            self._apply_selection([], origin="load")
            # `mt_data_loaded` may already be True (e.g. on a second append),
            # in which case param.watch won't re-fire; refresh dependents here
            # directly rather than relying solely on the watcher.
            self.mt_data_loaded = True
            self._save_button.disabled = False
            self._reset_button.disabled = False
            if self._edit_table_toggle.value:
                self._update_table_button.disabled = False
            self._update_station_table(mt_data)
            self._refresh_plot_station_picker()
            self._modeling_app.set_mt_data(mt_data)
            self._tf_editor_app.set_mt_data(mt_data)
            n = len(mt_data.station_paths)
            self._set_status(f"✅ Loaded **{n}** station(s).", notify=True)
        except Exception as exc:
            self._fail_load(exc)

    def _fail_load(self, exc: Exception) -> None:
        self._set_status(
            f"❌ Error: `{type(exc).__name__}: {exc}`", error=True, notify=True
        )
        self._mt_data = None
        self.mt_data_loaded = False
        self._save_button.disabled = True
        self._update_station_table(None)

    def _on_reset_clicked(self, event: param.parameterized.Event) -> None:
        """Clear all loaded data and reset the app to its initial state."""
        self._mt_data = None
        self._modeling_app.set_mt_data(None)
        self._tf_editor_app.set_mt_data(None)
        self.mt_data_loaded = False
        self._save_button.disabled = True
        self._reset_button.disabled = True
        self._update_table_button.disabled = True
        self._edit_table_toggle.value = False
        self._append_toggle.value = False
        self._file_selector.value = []
        self._apply_selection([], origin="reset")
        self._update_station_table(None)
        self._set_status("_No files loaded yet._")

    def _on_edit_toggle_changed(self, event: param.parameterized.Event) -> None:
        """Enable or disable in-place editing of the station table."""
        editable_cols = {"latitude", "longitude", "elevation", "survey"}
        if event.new:
            self._station_table.editors = {
                col: (
                    "number"
                    if col in {"latitude", "longitude", "elevation"}
                    else "input"
                )
                for col in _STATION_TABLE_COLUMNS
                if col in editable_cols
            }
            self._update_table_button.disabled = self._mt_data is None
        else:
            self._station_table.editors = {col: None for col in _STATION_TABLE_COLUMNS}
            self._update_table_button.disabled = True

    def _on_update_table_clicked(self, event: param.parameterized.Event) -> None:
        """Write edited table values back to the MTData object."""
        if self._mt_data is None:
            self._set_status("⚠️ No data loaded.", warning=True)
            return

        df = self._station_table.value
        if df is None or df.empty:
            return

        updated = 0
        errors = []
        for _, row in df.iterrows():
            survey = row.get("survey", "")
            station = row.get("station", "")
            path = (
                f"/{MTData.SURVEYS_NODE}/{survey}" f"/{MTData.STATIONS_NODE}/{station}"
            )
            try:
                mt_obj = self._mt_data.get_station(path, as_mt=True)
                mt_obj.latitude = float(row["latitude"])
                mt_obj.longitude = float(row["longitude"])
                mt_obj.elevation = float(row["elevation"])
                updated += 1
            except Exception as exc:
                errors.append(f"{station}: {exc}")

        if errors:
            self._set_status(
                f"⚠️ Updated {updated} station(s) with {len(errors)} error(s): "
                + "; ".join(errors),
                warning=True,
            )
        else:
            self._set_status(f"✅ Updated **{updated}** station(s) in MTData.")

    def _on_table_selection_changed(self, event: param.parameterized.Event) -> None:
        """Propagate table row selection to the shared station selection."""
        if self._syncing_selection or self._mt_data is None:
            return
        df = self._station_table.value
        paths: list[str] = []
        if df is not None and not df.empty:
            for idx in event.new or []:
                if idx < len(df):
                    row = df.iloc[idx]
                    paths.append(_station_path(row["survey"], row["station"]))
        self._apply_selection(paths, origin="table")

    def _on_map_selection(self, attr: str, old: list[int], new: list[int]) -> None:
        """Propagate map (tap / box / lasso) selection to the shared selection."""
        if self._syncing_selection:
            return
        paths = self._station_source.data.get("path", [])
        self._apply_selection([paths[i] for i in new if i < len(paths)], origin="map")

    def _apply_selection(self, paths: list[str], origin: str) -> None:
        """Single entry point that keeps the table, map and plots in sync.

        Parameters
        ----------
        paths : list of str
            Selected station paths (empty clears the selection).
        origin : str
            ``"table"``, ``"map"`` or another label for programmatic changes;
            the originating widget is not re-written to avoid feedback loops.
        """
        paths = list(dict.fromkeys(paths))
        self._selected_station_paths = paths
        self._syncing_selection = True
        try:
            if origin != "table":
                df = self._station_table.value
                rows = []
                if paths and df is not None and not df.empty:
                    keys = [
                        _station_path(s, t) for s, t in zip(df["survey"], df["station"])
                    ]
                    rows = [i for i, key in enumerate(keys) if key in set(paths)]
                self._station_table.selection = rows
            if origin != "map":
                source_paths = list(self._station_source.data.get("path", []))
                self._station_source.selected.indices = [
                    i for i, key in enumerate(source_paths) if key in set(paths)
                ]
            if len(paths) == 1 and paths[0] in (
                self._plot_station_widget.options or []
            ):
                self._suspend_auto_plot = True
                try:
                    self._plot_station_widget.value = paths[0]
                finally:
                    self._suspend_auto_plot = False
        finally:
            self._syncing_selection = False

        if paths:
            self._selection_label.object = (
                f"**{len(paths)}** station(s) selected — available as "
                "`app.selected_station_paths`."
            )
        else:
            self._selection_label.object = (
                "_No stations selected — plots use all stations._"
            )
        if self._mt_data is not None:
            self._update_penetration_depth(set(paths))
        self.selected_station_paths = list(paths)

    def _build_station_map_figure(self) -> Any:
        """Create the Bokeh station map (OSM-style basemap, selectable points)."""
        fig = bokeh_figure(
            x_axis_type="mercator",
            y_axis_type="mercator",
            tools="pan,wheel_zoom,box_zoom,tap,box_select,lasso_select,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
            sizing_mode="stretch_both",
            title="Stations (click, box or lasso to select)",
        )
        try:
            fig.add_tile("CartoDB Positron")
        except Exception:  # basemap is cosmetic; keep the map usable without it
            pass
        renderer = fig.scatter(
            "x",
            "y",
            source=self._station_source,
            size=10,
            marker="triangle",
            color="#1f77b4",
            alpha=0.85,
            line_color="white",
            selection_color="#d62728",
            selection_alpha=1.0,
            nonselection_alpha=0.45,
        )
        fig.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[("Station", "@station"), ("Survey", "@survey")],
            )
        )
        return fig

    def _update_station_map(self, mt_data: MTData | None) -> None:
        """Refill the map data source from the station summary table."""
        df = (
            self._station_table.value
            if mt_data is not None
            else pd.DataFrame(columns=_STATION_TABLE_COLUMNS)
        )
        xs: list[float] = []
        ys: list[float] = []
        if df is not None and not df.empty:
            for lon, lat in zip(df["longitude"], df["latitude"]):
                x, y = _to_web_mercator(lon, lat)
                xs.append(x)
                ys.append(y)
        data = {
            "x": xs,
            "y": ys,
            "station": list(df["station"]) if len(xs) else [],
            "survey": list(df["survey"]) if len(xs) else [],
            "path": (
                [_station_path(s, t) for s, t in zip(df["survey"], df["station"])]
                if len(xs)
                else []
            ),
        }
        self._syncing_selection = True
        try:
            self._station_source.selected.indices = []
            self._station_source.data = data
        finally:
            self._syncing_selection = False
        if xs:
            pad = max((max(xs) - min(xs)), (max(ys) - min(ys)), 2000.0) * 0.1
            self._station_map_figure.x_range.start = min(xs) - pad
            self._station_map_figure.x_range.end = max(xs) + pad
            self._station_map_figure.y_range.start = min(ys) - pad
            self._station_map_figure.y_range.end = max(ys) + pad

    def _on_edit_toggle_changed(self, event: param.parameterized.Event) -> None:
        """Enable or disable table editing.

        When editing is turned on a snapshot of the current table is stored so
        that ``_on_update_table_clicked`` can match edited rows back to their
        original station paths even if the station or survey name was changed.
        """
        editing = bool(event.new)
        if editing:
            # Snapshot the table before any edits so we can look up original paths
            self._pre_edit_station_df = self._station_table.value.copy()
            editors = {
                "survey": "input",
                "station": "input",
                "latitude": {"type": "number", "min": -90, "max": 90, "step": 0.000001},
                "longitude": {
                    "type": "number",
                    "min": -180,
                    "max": 180,
                    "step": 0.000001,
                },
                "elevation": {"type": "number", "step": 0.01},
            }
        else:
            self._pre_edit_station_df = None
            editors = {col: None for col in _STATION_TABLE_COLUMNS}
        self._station_table.editors = editors
        self._update_table_button.disabled = not editing or self._mt_data is None

    def _on_update_table_clicked(self, event: param.parameterized.Event) -> None:
        """Write edited table values back to the in-memory MTData object.

        Rows are matched to their original station path via the pre-edit snapshot
        (stored when editing was enabled), so renaming survey or station works
        correctly.
        """
        if self._mt_data is None:
            self._set_status("⚠️ No data loaded.", warning=True)
            return

        df = self._station_table.value
        if df is None or df.empty:
            return

        # Fall back to current df as the reference if no snapshot exists
        original_df = (
            self._pre_edit_station_df
            if self._pre_edit_station_df is not None
            else df.copy()
        )

        updated = 0
        errors = []
        for idx, row in df.iterrows():
            # Resolve the *original* survey/station from the snapshot by row index
            if idx < len(original_df):
                orig_row = original_df.iloc[idx]
                orig_survey = str(orig_row.get("survey", ""))
                orig_station = str(orig_row.get("station", ""))
            else:
                orig_survey = str(row.get("survey", ""))
                orig_station = str(row.get("station", ""))

            orig_path = (
                f"/{MTData.SURVEYS_NODE}/{orig_survey}"
                f"/{MTData.STATIONS_NODE}/{orig_station}"
            )
            try:
                mt_obj = self._mt_data.get_station(orig_path, as_mt=True)
                mt_obj.latitude = float(row["latitude"])
                mt_obj.longitude = float(row["longitude"])
                mt_obj.elevation = float(row["elevation"])

                new_survey = str(row.get("survey", orig_survey))
                new_station = str(row.get("station", orig_station))
                renamed = new_survey != orig_survey or new_station != orig_station
                if renamed:
                    # Remove old entry then re-add with updated identity
                    self._mt_data.remove_station(orig_path)
                    mt_obj.station = new_station
                    mt_obj.survey = new_survey
                self._mt_data.add_station(mt_obj, overwrite=True)
                updated += 1
            except Exception as exc:
                errors.append(f"{orig_station}: {exc}")

        if errors:
            self._set_status(
                f"⚠️ Updated {updated} stations; {len(errors)} error(s): "
                + "; ".join(errors),
                warning=True,
            )
        else:
            self._set_status(f"✅ Updated {updated} station(s) in MTData.")
        # Refresh snapshot and table from MTData
        self._station_table.value = _build_station_summary(self._mt_data)
        self._update_station_map(self._mt_data)
        self._pre_edit_station_df = self._station_table.value.copy()

    def _on_save_directory_changed(self, event: param.parameterized.Event) -> None:
        """Point the output path at the browsed folder, keeping the file name."""
        current = Path(self._save_filename_widget.value.strip() or "mt_collection.h5")
        name = current.name or "mt_collection.h5"
        self._save_filename_widget.value = str(Path(event.new) / name)

    def _on_save_clicked(self, event: param.parameterized.Event) -> None:
        """Save current MTData to an MTH5 file."""
        if self._mt_data is None:
            self._set_status("⚠️ No data to save.", warning=True)
            return

        output_path = Path(self._save_filename_widget.value.strip())
        if not output_path.suffix:
            output_path = output_path.with_suffix(".h5")

        self._save_button.disabled = True
        self._set_status(f"⏳ Saving to `{output_path}`…")

        try:
            mc = MTCollection()
            mc.open_collection(filename=output_path, mode="w")
            mc.from_mt_data(self._mt_data)
            mc.close_collection()
            self._set_status(
                f"✅ Saved **{len(self._mt_data.station_paths)}** station(s) "
                f"to `{output_path}`.",
                notify=True,
            )
        except Exception as exc:
            self._set_status(
                f"❌ Save failed: `{type(exc).__name__}: {exc}`",
                error=True,
                notify=True,
            )
        finally:
            self._save_button.disabled = False

    # ── Plot callbacks ────────────────────────────────────────────────────

    def _on_mt_data_loaded_changed(self, event: param.parameterized.Event) -> None:
        """Enable/disable the Generate Plot button and refresh station picker."""
        self._plot_generate_button.disabled = not event.new
        if event.new and self._mt_data is not None:
            self._refresh_plot_station_picker()
            self._modeling_app.set_mt_data(self._mt_data)
            self._tf_editor_app.set_mt_data(self._mt_data)
            self._plot_status.object = (
                "_Select a plot type and click **Generate Plot**._"
            )
            self._plot_status.styles = {"color": "#555"}
        else:
            self._modeling_app.set_mt_data(None)
            self._tf_editor_app.set_mt_data(None)
            self._plot_station_widget.options = []
            self._plot_status.object = (
                "_Load data first, then select a plot type and click "
                "**Generate Plot**._"
            )
            self._plot_status.styles = {"color": "#555"}
            self._plot_display.objects = []

    def _refresh_plot_station_picker(self) -> None:
        """Repopulate the station Select widget from the current MTData."""
        if self._mt_data is None:
            self._plot_station_widget.options = []
            return
        paths = list(self._mt_data._iter_station_paths())
        self._suspend_auto_plot = True
        try:
            self._plot_station_widget.options = paths
            if paths:
                self._plot_station_widget.value = paths[0]
        finally:
            self._suspend_auto_plot = False

    def _on_plot_type_changed(self, event: param.parameterized.Event) -> None:
        """Show/hide the station picker based on whether the plot needs one."""
        _, needs_station = _PLOT_TYPES.get(event.new, ("", False))
        self._plot_station_widget.visible = needs_station

    def _on_plot_inputs_changed(self, *args: Any) -> None:
        """Re-render automatically when a plot input changes (pn.bind target)."""
        if (
            not self._plot_auto_widget.value
            or self._suspend_auto_plot
            or self._mt_data is None
            or self._plot_generate_button.disabled
        ):
            return
        pn.state.execute(self._generate_plot_async_task)

    async def _generate_plot_async_task(self) -> None:
        await self._on_generate_plot_async(None)

    async def _on_generate_plot_async(self, event: Any) -> None:
        """Show the spinner, yield to the event loop so it paints, then plot."""
        self._plot_display.loading = True
        await asyncio.sleep(0.05)
        self._on_generate_plot_clicked(event)

    def _plot_source(self, selected_only: bool = True) -> MTData:
        """Return the MTData to plot: the selection subset, or everything."""
        paths = list(self._selected_station_paths)
        if selected_only and self._plot_selected_only_widget.value and paths:
            return self._mt_data.get_subset(paths)
        return self._mt_data

    def _on_generate_plot_clicked(self, event: param.parameterized.Event) -> None:
        """Call the selected MTData plot method and render the result."""
        if self._mt_data is None:
            self._plot_status.object = "⚠️ No data loaded."
            self._plot_status.styles = {"color": "#7a5200"}
            self._plot_display.loading = False
            return

        plot_label = self._plot_type_widget.value
        method_name, needs_station = _PLOT_TYPES.get(plot_label, ("", False))
        if not method_name:
            self._plot_display.loading = False
            return

        self._plot_generate_button.disabled = True
        self._plot_display.loading = True
        self._plot_status.object = f"⏳ Generating **{plot_label}**…"
        self._plot_status.styles = {"color": "#555"}
        self._plot_display.objects = []

        try:
            backend = self._plot_backend_widget.value
            kwargs: dict = {"show_plot": False, "backend": backend}
            if needs_station:
                station_key = self._plot_station_widget.value
                if not station_key:
                    self._plot_status.object = "⚠️ No station selected."
                    self._plot_status.styles = {"color": "#7a5200"}
                    return
                method = getattr(self._mt_data, method_name)
                plot_obj = method(station_key=station_key, **kwargs)
            else:
                method = getattr(self._plot_source(), method_name)
                plot_obj = method(**kwargs)

            # If the plot object has a panel() method, embed the full
            # interactive Panel app (controls + live plot).  Otherwise fall
            # back to the plain .plot() render for classes without panel().
            if hasattr(plot_obj, "panel") and callable(plot_obj.panel):
                panel_app = plot_obj.panel()
                self._plot_display.objects = [panel_app]
                self._plot_status.object = f"✅ **{plot_label}** rendered."
                self._plot_status.styles = {"color": "#1a6600"}
            else:
                # Render: call .plot() if available; pass show=False only if the
                # method accepts it (some Bokeh classes have plot(self) with no args).
                if hasattr(plot_obj, "plot"):
                    plot_params = inspect.signature(plot_obj.plot).parameters
                    if "show" in plot_params:
                        fig = plot_obj.plot(show=False)
                    else:
                        fig = plot_obj.plot()
                elif hasattr(plot_obj, "fig"):
                    fig = plot_obj.fig
                else:
                    fig = plot_obj

                if fig is not None:
                    self._plot_display.objects = [
                        pn.pane.panel(fig, sizing_mode="stretch_width")
                    ]
                    self._plot_status.object = f"✅ **{plot_label}** rendered."
                    self._plot_status.styles = {"color": "#1a6600"}
                else:
                    self._plot_status.object = (
                        f"⚠️ **{plot_label}** returned no figure."
                    )
                    self._plot_status.styles = {"color": "#7a5200"}
        except Exception as exc:
            self._plot_status.object = (
                f"❌ Error: `{type(exc).__name__}: {exc}\n{traceback.format_exc()}`"
            )
            self._plot_status.styles = {"color": "#b00020"}
        finally:
            self._plot_generate_button.disabled = False
            self._plot_display.loading = False

    # ── Loading logic ─────────────────────────────────────────────────────

    def _update_penetration_depth(self, station_paths: set[str]) -> None:
        """Render a penetration depth plot for the selected station(s).

        If exactly one station is selected its 1-D Niblett-Bostick
        depth-of-investigation plot is displayed in
        ``_pen_depth_container``.  If multiple stations are selected a
        message is shown instead — multi-station comparisons are not
        supported here.

        Parameters
        ----------
        station_paths : set of str
            HDF5-style station paths, e.g.
            ``{'/Surveys/survey01/Stations/MT01'}``.
        """
        self._pen_depth_container.clear()

        if len(station_paths) != 1:
            self._pen_depth_container.append(
                pn.pane.Markdown(
                    "_Select exactly **one** station to view its "
                    "penetration depth._",
                    styles={"color": "#777"},
                )
            )
            return

        station_path = next(iter(station_paths))
        try:
            mt_obj = self._mt_data.get_station(station_path, as_mt=True)
            if mt_obj is None or mt_obj.Z is None:
                raise ValueError("No impedance data available for this station.")
            plotter = PlotPenetrationDepth1D(mt_obj, show_plot=False)
            panel_plot = plotter.make_panel(sizing_mode=self.sizing_mode)
            self._pen_depth_container.append(panel_plot)
        except Exception as exc:
            self._pen_depth_container.append(
                pn.pane.Markdown(
                    f"⚠️ Could not render penetration depth: "
                    f"`{type(exc).__name__}: {exc}`",
                    styles={"color": "#b00020"},
                )
            )

    def _load_files(self, selected: list[str]) -> MTData:
        """Partition files and load them into a single MTData.

        Parameters
        ----------
        selected : list of str
            File paths chosen by the user.

        Returns
        -------
        MTData
            Merged dataset from all selected files.

        Raises
        ------
        ValueError
            If any selected file has an unsupported suffix.
        """
        mth5_files: list[Path] = []
        tf_files: list[Path] = []
        dat_files: list[Path] = []
        unsupported: list[str] = []

        for fp in selected:
            path = Path(fp)
            suffix = path.suffix.lower()
            if suffix == SUPPORTED_MTH5_SUFFIX:
                mth5_files.append(path)
            elif suffix in SUPPORTED_TF_SUFFIXES:
                tf_files.append(path)
            elif suffix in SUPPORTED_DAT_SUFFIXES:
                dat_files.append(path)
            else:
                unsupported.append(fp)

        if unsupported:
            names = ", ".join(Path(u).name for u in unsupported)
            all_supported = sorted(SUPPORTED_TF_SUFFIXES | SUPPORTED_DAT_SUFFIXES) + [
                ".h5"
            ]
            raise ValueError(
                f"Unsupported file type(s): {names}. "
                f"Supported: {', '.join(all_supported)}"
            )

        mt_data = MTData()

        if mth5_files:
            for mth5_fn in mth5_files:
                if not mth5_fn.is_file():
                    raise ValueError(f"MTH5 file not found: `{mth5_fn}`")
                with MTCollection() as mc:
                    mc.open_collection(mth5_fn)
                    mt_data += mc.to_mt_data()

        if tf_files:
            mt_data.add_stations([str(p) for p in tf_files])

        if dat_files:
            dat_format = self._dat_format_widget.value
            for dat_fn in dat_files:
                if not dat_fn.is_file():
                    raise ValueError(f"Data file not found: `{dat_fn}`")
                chunk = MTData()
                if dat_format == DAT_FORMAT_MODEM:
                    chunk.from_modem(dat_fn, survey=SURVEY_LABEL_DATA)
                else:
                    chunk.from_occam2d(dat_fn, file_type=SURVEY_LABEL_DATA)
                mt_data += chunk

        survey_override = self._current_survey_override()
        if survey_override is not None:
            _relabel_survey(mt_data, survey_override)

        return mt_data

    def _close_collection(self) -> None:
        """Close any open MTCollection cleanly."""
        if self._mt_collection is not None:
            try:
                self._mt_collection.close_collection()
            except Exception:
                pass
            self._mt_collection = None

    # ── UI helpers ────────────────────────────────────────────────────────

    def _set_status(
        self,
        message: str,
        *,
        warning: bool = False,
        error: bool = False,
        notify: bool = False,
    ) -> None:
        color = "#b00020" if error else ("#7a5200" if warning else "#1a6600")
        self._status.object = message
        self._status.styles = {"color": color}
        if notify:
            self._notify(message, error=error, warning=warning)

    @staticmethod
    def _notify(message: str, *, error: bool = False, warning: bool = False) -> None:
        """Show a toast notification when running inside a Panel session."""
        notifications = pn.state.notifications
        if notifications is None:
            return
        text = message.replace("`", "").replace("*", "")
        if error:
            notifications.error(text, duration=0)
        elif warning:
            notifications.warning(text, duration=6000)
        else:
            notifications.success(text, duration=4000)

    def _update_station_table(self, mt_data) -> None:
        """Rebuild the station summary table and the linked map."""
        df = (
            _build_station_summary(mt_data)
            if mt_data is not None
            else pd.DataFrame(columns=_STATION_TABLE_COLUMNS)
        )
        self._station_table.value = df
        self._update_station_map(mt_data)

    # ── Layout ────────────────────────────────────────────────────────────

    @property
    def sidebar(self) -> pn.viewable.Viewable:
        """File-loading controls, intended for a template sidebar."""
        return pn.Column(
            pn.pane.Markdown("### Select MT Data Files"),
            self._file_pattern_widget,
            self._file_selector,
            pn.Column(
                pn.pane.Markdown(
                    "**Format for .dat / .data files:**",
                    styles={"font-size": "0.9em"},
                    visible=self._dat_format_widget.param.visible,
                ),
                self._dat_format_widget,
            ),
            pn.Row(
                self._survey_preset_widget,
                self._survey_custom_widget,
            ),
            self._append_toggle,
            pn.Row(self._load_button, self._reset_button),
            self._status,
            pn.layout.Divider(),
            pn.pane.Markdown("### Save"),
            self._save_filename_widget,
            self._save_button,
            sizing_mode="stretch_width",
        )

    @property
    def view(self) -> pn.viewable.Viewable:
        """Return the main-area layout (tabs) for this application.

        The file-loading controls live in :attr:`sidebar`; use
        :attr:`template` for the complete application.

        Returns
        -------
        pn.viewable.Viewable
            A Panel layout object suitable for serving or embedding.
        """
        station_section = pn.Column(
            pn.Row(
                pn.Column(
                    self._station_map_pane,
                    pn.Row(self._selection_label, self._clear_selection_button),
                    sizing_mode="stretch_width",
                    min_height=480,
                ),
                pn.Column(
                    pn.pane.Markdown("### Penetration depth"),
                    self._pen_depth_container,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown("### Loaded Stations"),
            pn.pane.Markdown(
                "_Select rows here or on the map; the selection drives every "
                "plot tab._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(
                self._edit_table_toggle,
                pn.Spacer(width=10),
                self._update_table_button,
                align="center",
            ),
            self._station_table,
            sizing_mode=self.sizing_mode,
        )

        data_tab = pn.Column(
            station_section,
            sizing_mode=self.sizing_mode,
        )

        plots_tab = pn.Column(
            pn.pane.Markdown("### Plots"),
            pn.pane.Markdown(
                "_Plots update automatically. Single-station plots follow the "
                "map/table selection (or the **Station** picker)._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(
                self._plot_type_widget,
                pn.Spacer(width=10),
                self._plot_station_widget,
                align="end",
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(
                        "**Backend:**",
                        styles={"font-size": "0.9em", "margin-bottom": "2px"},
                    ),
                    self._plot_backend_widget,
                ),
                pn.Spacer(width=20),
                self._plot_auto_widget,
                self._plot_selected_only_widget,
                self._plot_generate_button,
                pn.Spacer(width=10),
                self._plot_status,
                align="center",
            ),
            pn.layout.Divider(),
            self._plot_display,
            sizing_mode=self.sizing_mode,
        )

        return pn.Column(
            pn.Tabs(
                ("📂 Data", data_tab),
                ("📊 Plots", plots_tab),
                ("✏️ TF Editor", self._tf_editor_app.view),
                ("🧪 Modeling", self._modeling_app.view),
                sizing_mode=self.sizing_mode,
            ),
            sizing_mode=self.sizing_mode,
        )

    @property
    def template(self) -> pn.template.FastListTemplate:
        """Full application: header, file-controls sidebar and tabbed main area."""
        return pn.template.FastListTemplate(
            title="MT Data Loader",
            sidebar=[self.sidebar],
            sidebar_width=420,
            main=[self.view],
        )

    def servable(self) -> pn.viewable.Viewable:
        """Mark the full template as servable and return it.

        Convenience wrapper around :attr:`template` for use with
        ``panel serve``.
        """
        return self.template.servable()


# Support `panel serve panel_mt_data_app.py`; skip when merely imported.
if pn.state.served:
    _app = MTDataApp()
    _app.servable()
