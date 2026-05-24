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

SUPPORTED_FILE_PATTERNS: dict[str, str] = {
    "All MT Files (*.edi *.xml *.avg *.z* *.h5)": "*.*",
    "EDI (*.edi)": "*.edi",
    "XML (*.xml)": "*.xml",
    "AVG (*.avg)": "*.avg",
    "Z files (*.zmm, *.zss, *.zrr)": "*.z*",
    "MTH5 (*.h5)": "*.h5",
}

_STATION_TABLE_COLUMNS: list[str] = [
    "survey",
    "station",
    "latitude",
    "longitude",
    "elevation",
    "n_periods",
]


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

    locs = mt_data.station_locations
    if locs.empty:
        return pd.DataFrame(columns=_STATION_TABLE_COLUMNS)

    rows = []
    for _, row in locs.iterrows():
        survey = row.get("survey", "")
        station = row.get("station", "")
        # Build the tree path to retrieve period count
        station_path = (
            f"/{MTData.SURVEYS_NODE}/{survey}/{MTData.STATIONS_NODE}/{station}"
        )
        try:
            mt_obj = mt_data.get_station(station_path, as_mt=True)
            n_periods = len(mt_obj.period) if mt_obj.period is not None else 0
        except Exception:
            n_periods = 0
        rows.append(
            {
                "survey": survey,
                "station": station,
                "latitude": row.get("latitude", float("nan")),
                "longitude": row.get("longitude", float("nan")),
                "elevation": row.get("elevation", float("nan")),
                "n_periods": n_periods,
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
            name="Select MT Data Files",
            file_pattern=self._file_pattern_widget.value,
            only_files=True,
        )

        # ── Load button ───────────────────────────────────────────────────
        self._load_button = pn.widgets.Button(
            name="Load Selected Files",
            button_type="primary",
            width=200,
        )
        self._load_button.on_click(self._on_load_clicked)

        # ── Status display ────────────────────────────────────────────────
        self._status = pn.pane.Markdown(
            "_No files loaded yet._",
            styles={"color": "#555"},
        )

        # ── Station summary table ─────────────────────────────────────────
        self._station_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=_STATION_TABLE_COLUMNS),
            name="Loaded Stations",
            selectable="checkbox",
            pagination="local",
            page_size=20,
            sizing_mode="stretch_width",
            show_index=False,
            configuration={"columnDefaults": {"headerFilter": True}},
        )
        self._station_table.param.watch(self._on_table_selection_changed, "selection")

        # ── Save-to-MTH5 controls ─────────────────────────────────────────
        self._save_filename_widget = pn.widgets.TextInput(
            name="Output MTH5 filename",
            value="mt_collection.h5",
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

        # ── Wire file-pattern changes ─────────────────────────────────────
        self._file_pattern_widget.param.watch(self._on_file_pattern_changed, "value")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def mt_data(self) -> MTData | None:
        """The currently loaded :class:`~mtpy.core.mt_data.MTData` object."""
        return self._mt_data

    # ── Widget callbacks ──────────────────────────────────────────────────

    def _on_file_pattern_changed(self, event: param.parameterized.Event) -> None:
        """Update the file selector filter pattern."""
        self._file_selector.file_pattern = event.new

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        """Load selected files into MTData."""
        selected: list[str] = list(self._file_selector.value or [])

        if not selected:
            self._set_status("⚠️ No files selected.", warning=True)
            return

        self._set_status("⏳ Loading files…")
        self._load_button.disabled = True

        try:
            mt_data = self._load_files(selected)
            self._mt_data = mt_data
            self.mt_data_loaded = True
            self._save_button.disabled = False
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
        unsupported: list[str] = []

        for fp in selected:
            path = Path(fp)
            suffix = path.suffix.lower()
            if suffix == SUPPORTED_MTH5_SUFFIX:
                mth5_files.append(path)
            elif suffix in SUPPORTED_TF_SUFFIXES:
                tf_files.append(path)
            else:
                unsupported.append(fp)

        if unsupported:
            names = ", ".join(Path(u).name for u in unsupported)
            raise ValueError(
                f"Unsupported file type(s): {names}. "
                f"Supported: {', '.join(sorted(SUPPORTED_TF_SUFFIXES))} and .h5"
            )

        # Close any previously open collection
        self._close_collection()

        mt_data = MTData()

        if mth5_files:
            if len(mth5_files) > 1:
                extra = ", ".join(p.name for p in mth5_files[1:])
                self._set_status(
                    f"⚠️ Multiple MTH5 files selected — only **{mth5_files[0].name}** "
                    f"will be opened. Ignored: {extra}",
                    warning=True,
                )
            mc = MTCollection()
            mc.open_collection(filename=mth5_files[0], mode="r")
            self._mt_collection = mc
            if mc.mt_data is not None:
                mt_data = mc.mt_data

        if tf_files:
            mt_data.add_stations([str(p) for p in tf_files])

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
            self._load_button,
            sizing_mode=self.sizing_mode,
        )

        status_row = pn.Row(self._status, sizing_mode="stretch_width")

        station_section = pn.Column(
            pn.pane.Markdown("### Loaded Stations"),
            pn.pane.Markdown(
                "_Select rows to filter the active working set._",
                styles={"color": "#777", "font-size": "0.85em"},
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
