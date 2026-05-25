"""Panel application for loading MT data into an MTData object.

This module provides a standalone Panel application that allows users to select
MT data files (transfer function files and/or an MTH5 file), load them into a
unified :class:`~mtpy.core.mt_data.MTData` object, and optionally save the
result back to an MTH5 file.

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

from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import param

from mtpy.core.mt_collection import MTCollection
from mtpy.core.mt_data import MTData


pn.extension("tabulator")


SUPPORTED_TF_SUFFIXES: frozenset[str] = frozenset(
    {".edi", ".xml", ".avg", ".zmm", ".zss", ".zrr"}
)
SUPPORTED_MTH5_SUFFIX: str = ".h5"
SUPPORTED_DAT_SUFFIXES: frozenset[str] = frozenset({".dat", ".data"})

DAT_FORMAT_MODEM: str = "ModEM"
DAT_FORMAT_OCCAM2D: str = "Occam2D"

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

        # # ── Directory path input (lets users type/paste any path) ────────
        # self._directory_input = pn.widgets.TextInput(
        #     name="Browse directory",
        #     value=str(Path(self.data_directory).expanduser().resolve()),
        #     placeholder="Paste or type an absolute directory path…",
        #     sizing_mode="stretch_width",
        # )
        # self._directory_go_button = pn.widgets.Button(
        #     name="Go",
        #     button_type="default",
        #     width=60,
        # )
        # self._directory_go_button.on_click(self._on_directory_go_clicked)
        # # Navigate when the user presses Enter in the text box
        # self._directory_input.param.watch(
        #     self._on_directory_input_enter, "enter_pressed"
        # )

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
        """Show the .dat format picker only when .dat/.data files are selected."""
        has_dat = any(
            Path(f).suffix.lower() in SUPPORTED_DAT_SUFFIXES for f in (event.new or [])
        )
        self._dat_format_widget.visible = has_dat

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
                col: "number"
                if col in {"latitude", "longitude", "elevation"}
                else "input"
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
            # Nothing selected → expose full dataset
            self._mt_data_subset = None
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

        # Expose subset info via status (actual subsetting can be done
        # by consumers via mt_data.get_station() with the paths)
        n = len(selected_paths)
        self._set_status(
            f"✅ **{n}** station(s) selected — access via `app.selected_station_paths`."
        )
        self._selected_station_paths = list(selected_paths)

    def _on_edit_toggle_changed(self, event: param.parameterized.Event) -> None:
        """Enable or disable table editing."""
        editing = bool(event.new)
        if editing:
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
            editors = {col: None for col in _STATION_TABLE_COLUMNS}
        self._station_table.editors = editors
        self._update_table_button.disabled = not editing or self._mt_data is None

    def _on_update_table_clicked(self, event: param.parameterized.Event) -> None:
        """Write edited table values back to the in-memory MTData object."""
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
            path = f"/{MTData.SURVEYS_NODE}/{survey}/{MTData.STATIONS_NODE}/{station}"
            try:
                mt_obj = self._mt_data.get_station(path, as_mt=True)
                mt_obj.latitude = float(row["latitude"])
                mt_obj.longitude = float(row["longitude"])
                mt_obj.elevation = float(row["elevation"])
                self._mt_data.add_station(mt_obj, overwrite=True)
                updated += 1
            except Exception as exc:
                errors.append(f"{station}: {exc}")

        if errors:
            self._set_status(
                f"⚠️ Updated {updated} stations; {len(errors)} error(s): "
                + "; ".join(errors),
                warning=True,
            )
        else:
            self._set_status(f"✅ Updated {updated} station(s) in MTData.")
        self._station_table.value = _build_station_summary(self._mt_data)

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

    # ── Loading logic ─────────────────────────────────────────────────────

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
                    chunk.from_modem(dat_fn)
                else:
                    chunk.from_occam2d(dat_fn)
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
            # pn.pane.Markdown(
            #     "_Paste or type an absolute path and click **Go** to jump to any directory. "
            #     "Then browse and select files below._",
            #     styles={"color": "#555", "font-size": "0.85em"},
            # ),
            # pn.Row(
            #     self._directory_input,
            #     self._directory_go_button,
            #     align="end",
            #     sizing_mode="stretch_width",
            # ),
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
                "_Select rows to filter the active working set._",
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

        return pn.Column(
            pn.pane.Markdown("# MT Data Loader"),
            pn.layout.Divider(),
            file_controls,
            pn.layout.Divider(),
            status_row,
            station_section,
            pn.layout.Divider(),
            save_section,
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
