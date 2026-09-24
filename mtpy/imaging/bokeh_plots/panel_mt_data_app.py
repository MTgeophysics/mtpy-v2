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
    >>> pn.extension("tabulator")
    >>> from mtpy.imaging.bokeh_plots.panel_mt_data_app import MTDataApp
    >>> app = MTDataApp()
    >>> app.view.servable()

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

import inspect
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import param

from mtpy.core.mt_collection import MTCollection
from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse
from mtpy.imaging.bokeh_plots.panel_simpeg1d_app import Simpeg1DPanelApp
from mtpy.imaging.bokeh_plots.plot_penetration_depth_1d import PlotPenetrationDepth1D


pn.extension("tabulator")


SUPPORTED_TF_SUFFIXES: frozenset[str] = frozenset(
    {".edi", ".xml", ".avg", ".zmm", ".zss", ".zrr"}
)
SUPPORTED_MTH5_SUFFIX: str = ".h5"
SUPPORTED_DAT_SUFFIXES: frozenset[str] = frozenset({".dat", ".data"})

DAT_FORMAT_MODEM: str = "ModEM"
DAT_FORMAT_OCCAM2D: str = "Occam2D"

DAT_SURVEY_DATA: str = "data"
DAT_SURVEY_MODEL: str = "model"
DAT_SURVEY_CUSTOM: str = "Custom…"

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


def _relabel_survey(mt_data: "MTData", new_survey: str) -> None:
    """Rename every station's survey label in place to *new_survey*."""
    for station_path in list(mt_data._iter_station_paths()):
        mt_obj = mt_data.get_station(station_path, as_mt=True)
        if mt_obj.survey == new_survey:
            continue
        mt_data.remove_station(station_path)
        mt_obj.survey = new_survey
        mt_data.add_station(mt_obj, overwrite=True)


class MTResponseEditorTab(param.Parameterized):
    """Panel tab for editing a single station's transfer function.

    Wraps :class:`~mtpy.imaging.bokeh_plots.edit_mt_response.EditMTResponse`,
    which only exposes an edit-mode layout (mask points, add model error,
    interpolate, static shift, rotation, and phase flips) for one station at
    a time, without plotting the phase tensor.
    """

    sizing_mode: str = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode for this tab's layout.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._mt_data = None

        self._station_widget = pn.widgets.Select(name="Station", options=[], width=320)
        self._plot_response_widget = pn.widgets.Checkbox(
            name="Plot Response",
            value=False,
            visible=False,
        )
        self._station_widget.param.watch(self._on_station_selected, "value")
        self._load_button = pn.widgets.Button(
            name="Load Station",
            button_type="primary",
            width=160,
            disabled=True,
        )
        self._load_button.on_click(self._on_load_clicked)
        self._status = pn.pane.Markdown(
            "_Load data in the **Data** tab, select a station, then click "
            "**Load Station**._",
            styles={"color": "#555"},
        )
        self._display = pn.Column(sizing_mode=self.sizing_mode)

    def _paired_station_path(self, station_key: str, target_survey: str) -> str | None:
        """Return the sibling path with the same station name in *target_survey*.

        Station paths are ``/{SURVEYS_NODE}/{survey}/{STATIONS_NODE}/{station}``;
        returns ``None`` if the sibling path isn't among the known stations.
        """
        parts = station_key.strip("/").split("/")
        if len(parts) != 4:
            return None
        surveys_node, _survey, stations_node, station = parts
        candidate = f"/{surveys_node}/{target_survey}/{stations_node}/{station}"
        options = self._station_widget.options or []
        return candidate if candidate in options else None

    def _on_station_selected(self, event: param.parameterized.Event) -> None:
        """Show 'Plot Response' only when a data/model pair exists for this station."""
        self._update_response_visibility(event.new)

    def _update_response_visibility(self, station_key: str | None) -> None:
        """Show/hide the 'Plot Response' checkbox based on data/model pairing."""
        has_pair = False
        if station_key:
            parts = station_key.strip("/").split("/")
            if len(parts) == 4:
                survey = parts[1]
                other_survey = "data" if survey == "model" else "model"
                has_pair = (
                    self._paired_station_path(station_key, other_survey) is not None
                )
        self._plot_response_widget.visible = has_pair
        if not has_pair:
            self._plot_response_widget.value = False

    def set_mt_data(self, mt_data: MTData | None) -> None:
        """Refresh the station picker for a newly loaded (or cleared) MTData."""
        self._mt_data = mt_data
        self._display.objects = []

        if mt_data is None:
            self._station_widget.options = []
            self._load_button.disabled = True
            self._plot_response_widget.visible = False
            self._plot_response_widget.value = False
            self._status.object = (
                "_Load data in the **Data** tab, select a station, then click "
                "**Load Station**._"
            )
            self._status.styles = {"color": "#555"}
            return

        paths = list(mt_data._iter_station_paths())
        self._station_widget.options = paths
        if paths:
            self._station_widget.value = paths[0]
        self._update_response_visibility(self._station_widget.value)
        self._load_button.disabled = not bool(paths)
        self._status.object = "_Select a station and click **Load Station**._"
        self._status.styles = {"color": "#555"}

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        if self._mt_data is None:
            self._status.object = "⚠️ No data loaded."
            self._status.styles = {"color": "#7a5200"}
            return

        station_key = self._station_widget.value
        if not station_key:
            self._status.object = "⚠️ No station selected."
            self._status.styles = {"color": "#7a5200"}
            return

        self._load_button.disabled = True
        self._status.object = f"⏳ Loading **{station_key}**…"
        self._status.styles = {"color": "#555"}

        try:
            # Prefer the "data" survey as the editable primary object; if the
            # selected station is the "model" one, swap so the response is
            # always overlaid on the data rather than the reverse.
            primary_key = station_key
            response_key = None
            if self._plot_response_widget.visible and self._plot_response_widget.value:
                parts = station_key.strip("/").split("/")
                survey = parts[1] if len(parts) == 4 else None
                if survey == "model":
                    data_key = self._paired_station_path(station_key, "data")
                    if data_key is not None:
                        primary_key = data_key
                        response_key = station_key
                else:
                    response_key = self._paired_station_path(station_key, "model")

            mt_obj = self._mt_data.get_station(primary_key, as_mt=True)
            z_response = t_response = None
            if response_key is not None:
                response_obj = self._mt_data.get_station(response_key, as_mt=True)
                z_response = response_obj.Z
                t_response = response_obj.Tipper

            plot_obj = EditMTResponse(
                z_object=mt_obj.Z,
                t_object=mt_obj.Tipper,
                z_response=z_response,
                t_response=t_response,
                station=mt_obj.station,
                show_plot=False,
            )
            self._display.objects = [plot_obj.panel(sizing_mode=self.sizing_mode)]
            self._status.object = f"✅ Loaded **{station_key}**."
            self._status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._display.objects = []
            self._status.object = f"❌ Error: `{type(exc).__name__}: {exc}`"
            self._status.styles = {"color": "#b00020"}
        finally:
            self._load_button.disabled = False

    @property
    def view(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout for this tab."""
        return pn.Column(
            pn.pane.Markdown("### MT Response Editor"),
            pn.pane.Markdown(
                "_Edit mode only: mask points, add model error, interpolate, "
                "static shift, rotate, and flip phase for a single station._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(
                self._station_widget,
                pn.Spacer(width=10),
                self._plot_response_widget,
                pn.Spacer(width=10),
                self._load_button,
                align="end",
            ),
            self._status,
            pn.layout.Divider(),
            self._display,
            sizing_mode=self.sizing_mode,
        )


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
        self._load_button.on_click(self._on_load_clicked)

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

        # ── .dat/.data survey label (data/model preset or custom name) ─────
        self._dat_survey_preset_widget = pn.widgets.Select(
            name="Survey label",
            options=[DAT_SURVEY_DATA, DAT_SURVEY_MODEL, DAT_SURVEY_CUSTOM],
            value=DAT_SURVEY_DATA,
            width=160,
            visible=False,
        )
        self._dat_survey_custom_widget = pn.widgets.TextInput(
            name="Custom survey name",
            placeholder="e.g. line1",
            width=160,
            visible=False,
        )
        self._dat_survey_preset_widget.param.watch(
            self._on_dat_survey_preset_changed, "value"
        )
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
        self._plot_generate_button.on_click(self._on_generate_plot_clicked)
        self.param.watch(self._on_mt_data_loaded_changed, "mt_data_loaded")

        # ── Modeling tab app ─────────────────────────────────────────────
        self._modeling_app = Simpeg1DPanelApp(sizing_mode=self.sizing_mode)
        self._tf_editor_app = MTResponseEditorTab(sizing_mode=self.sizing_mode)

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
        """Show the .dat format/survey pickers only when .dat/.data files are selected."""
        has_dat = any(
            Path(f).suffix.lower() in SUPPORTED_DAT_SUFFIXES for f in (event.new or [])
        )
        self._dat_format_widget.visible = has_dat
        self._dat_survey_preset_widget.visible = has_dat
        self._dat_survey_custom_widget.visible = (
            has_dat and self._dat_survey_preset_widget.value == DAT_SURVEY_CUSTOM
        )

    def _on_dat_survey_preset_changed(self, event: param.parameterized.Event) -> None:
        """Reveal the custom survey name field only when 'Custom…' is selected."""
        self._dat_survey_custom_widget.visible = (
            self._dat_survey_preset_widget.visible and event.new == DAT_SURVEY_CUSTOM
        )

    def _current_dat_survey_name(self) -> str:
        """Resolve the survey label to apply to stations loaded from .dat/.data files."""
        preset = self._dat_survey_preset_widget.value
        if preset == DAT_SURVEY_CUSTOM:
            custom = self._dat_survey_custom_widget.value.strip()
            return custom if custom else DAT_SURVEY_DATA
        return preset

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        """Load selected files into MTData."""
        selected: list[str] = list(self._file_selector.value or [])

        if not selected:
            self._set_status("⚠️ No files selected.", warning=True)
            return

        self._set_status("⏳ Loading files…")
        self._load_button.disabled = True

        try:
            new_data = self._load_files(selected)
            if self._append_toggle.value and self._mt_data is not None:
                self._mt_data += new_data
                mt_data = self._mt_data
            else:
                self._mt_data = new_data
                mt_data = new_data
            self.mt_data_loaded = True
            self._save_button.disabled = False
            self._reset_button.disabled = False
            if self._edit_table_toggle.value:
                self._update_table_button.disabled = False
            self._update_station_table(mt_data)
            n = len(mt_data.station_paths)
            self._set_status(f"✅ Loaded **{n}** station(s).")
        except Exception as exc:
            self._set_status(f"❌ Error: `{type(exc).__name__}: {exc}`", error=True)
            self._mt_data = None
            self.mt_data_loaded = False
            self._save_button.disabled = True
            self._update_station_table(None)
        finally:
            self._load_button.disabled = False

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
        """Subset the active MTData to the table-selected rows."""
        if self._mt_data is None:
            return

        selected_indices: list[int] = list(event.new or [])
        if not selected_indices:
            # Nothing selected → expose full dataset and clear penetration depth
            self._mt_data_subset = None
            self._pen_depth_container.clear()
            self._pen_depth_container.append(
                pn.pane.Markdown(
                    "_Select a single station in the table above to view its "
                    "1-D penetration depth._",
                    styles={"color": "#777"},
                )
            )
            return

        df = self._station_table.value
        if df is None or df.empty:
            return

        selected_rows = df.iloc[selected_indices]
        selected_paths = set()
        for _, row in selected_rows.iterrows():
            survey = row.get("survey", "")
            station = row.get("station", "")
            path = f"/{MTData.SURVEYS_NODE}/{survey}/{MTData.STATIONS_NODE}/{station}"
            selected_paths.add(path)

        n = len(selected_paths)
        self._set_status(
            f"✅ **{n}** station(s) selected — access via `app.selected_station_paths`."
        )
        self._selected_station_paths = list(selected_paths)

        # ── Penetration depth for the selected station(s) ─────────────────
        self._update_penetration_depth(selected_paths)

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
        self._pre_edit_station_df = self._station_table.value.copy()

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
                f"to `{output_path}`."
            )
        except Exception as exc:
            self._set_status(
                f"❌ Save failed: `{type(exc).__name__}: {exc}`", error=True
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
        self._plot_station_widget.options = paths
        if paths:
            self._plot_station_widget.value = paths[0]

    def _on_plot_type_changed(self, event: param.parameterized.Event) -> None:
        """Show/hide the station picker based on whether the plot needs one."""
        _, needs_station = _PLOT_TYPES.get(event.new, ("", False))
        self._plot_station_widget.visible = needs_station

    def _on_generate_plot_clicked(self, event: param.parameterized.Event) -> None:
        """Call the selected MTData plot method and render the result."""
        if self._mt_data is None:
            self._plot_status.object = "⚠️ No data loaded."
            self._plot_status.styles = {"color": "#7a5200"}
            return

        plot_label = self._plot_type_widget.value
        method_name, needs_station = _PLOT_TYPES.get(plot_label, ("", False))
        if not method_name:
            return

        self._plot_generate_button.disabled = True
        self._plot_status.object = f"⏳ Generating **{plot_label}**…"
        self._plot_status.styles = {"color": "#555"}
        self._plot_display.objects = []

        try:
            method = getattr(self._mt_data, method_name)
            backend = self._plot_backend_widget.value
            kwargs: dict = {"show_plot": False, "backend": backend}
            if needs_station:
                station_key = self._plot_station_widget.value
                if not station_key:
                    self._plot_status.object = "⚠️ No station selected."
                    self._plot_status.styles = {"color": "#7a5200"}
                    return
                plot_obj = method(station_key=station_key, **kwargs)
            else:
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
            survey_name = self._current_dat_survey_name()
            for dat_fn in dat_files:
                if not dat_fn.is_file():
                    raise ValueError(f"Data file not found: `{dat_fn}`")
                chunk = MTData()
                if dat_format == DAT_FORMAT_MODEM:
                    chunk.from_modem(dat_fn, survey=survey_name)
                else:
                    # Occam2D only labels rows "data"/"model" internally; a
                    # custom name is applied afterward via a rename pass.
                    file_type = (
                        "response" if survey_name == DAT_SURVEY_MODEL else "data"
                    )
                    chunk.from_occam2d(dat_fn, file_type=file_type)
                    if survey_name not in (DAT_SURVEY_DATA, DAT_SURVEY_MODEL):
                        _relabel_survey(chunk, survey_name)
                mt_data += chunk

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
    ) -> None:
        color = "#b00020" if error else ("#7a5200" if warning else "#1a6600")
        self._status.object = message
        self._status.styles = {"color": color}

    def _update_station_table(self, mt_data) -> None:
        """Rebuild the station summary table."""
        df = (
            _build_station_summary(mt_data)
            if mt_data is not None
            else pd.DataFrame(columns=_STATION_TABLE_COLUMNS)
        )
        self._station_table.value = df

    # ── Layout ────────────────────────────────────────────────────────────

    @property
    def view(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout for this application.

        Returns
        -------
        pn.viewable.Viewable
            A Panel layout object suitable for serving or embedding.
        """
        file_controls = pn.Column(
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
                pn.Row(
                    self._dat_survey_preset_widget,
                    self._dat_survey_custom_widget,
                ),
            ),
            self._append_toggle,
            pn.Row(
                self._load_button,
                pn.Spacer(width=10),
                self._reset_button,
                align="center",
            ),
            sizing_mode=self.sizing_mode,
        )

        status_row = pn.Row(self._status, sizing_mode="stretch_width")

        station_section = pn.Column(
            pn.pane.Markdown("### Loaded Stations"),
            pn.pane.Markdown(
                "_Select a row to filter the active working set and view its "
                "penetration depth._",
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

        save_section = pn.Column(
            pn.pane.Markdown("### Save Data"),
            pn.Row(
                self._save_filename_widget,
                pn.Spacer(width=10),
                self._save_button,
                align="end",
            ),
            sizing_mode=self.sizing_mode,
        )

        data_tab = pn.Column(
            file_controls,
            pn.layout.Divider(),
            status_row,
            station_section,
            # pn.layout.Divider(),
            # pn.Column(
            #     pn.pane.Markdown("### Penetration Depth"),
            #     self._pen_depth_container,
            #     sizing_mode=self.sizing_mode,
            # ),
            pn.layout.Divider(),
            save_section,
            sizing_mode=self.sizing_mode,
        )

        plots_tab = pn.Column(
            pn.pane.Markdown("### Generate a Plot"),
            pn.pane.Markdown(
                "_Load data in the **Data** tab first. "
                "Single-station plots use the **Station** selector._",
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
            pn.pane.Markdown("# MT Data Loader"),
            pn.Tabs(
                ("📂 Data", data_tab),
                ("📊 Plots", plots_tab),
                ("✏️ TF Editor", self._tf_editor_app.view),
                ("🧪 Modeling", self._modeling_app.view),
                sizing_mode=self.sizing_mode,
            ),
            sizing_mode=self.sizing_mode,
        )

    def servable(self) -> pn.viewable.Viewable:
        """Mark the layout as servable and return it.

        Convenience wrapper around :meth:`view` for use with
        ``panel serve``.
        """
        return self.view.servable()


# Support `panel serve panel_mt_data_app.py`
_app = MTDataApp()
_app.view.servable()
