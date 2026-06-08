"""Standalone Panel application for 1-D penetration depth plotting.

This module provides an interactive Panel application that lets users select
a transfer function file, load it as an :class:`~mtpy.MT` object, and view
the Niblett-Bostick depth-of-investigation estimate with controls for toggling
the TE (Zxy), TM (Zyx), and Determinant modes and for switching between
kilometre and metre depth units.

Usage
-----
As a Panel app in a Jupyter notebook::

    >>> import panel as pn
    >>> pn.extension("bokeh")
    >>> from mtpy.imaging.bokeh_plots.panel_penetration_depth_1d_app import (
    ...     PenetrationDepth1DApp,
    ... )
    >>> app = PenetrationDepth1DApp()
    >>> app.view.servable()

As a standalone script (from terminal)::

    panel serve path/to/panel_penetration_depth_1d_app.py --autoreload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import panel as pn
import param

from mtpy import MT
from mtpy.imaging.bokeh_plots import PlotPenetrationDepth1D

pn.extension("bokeh")


SUPPORTED_FILE_PATTERNS: dict[str, str] = {
    "EDI (*.edi)": "*.edi",
    "XML (*.xml)": "*.xml",
    "AVG (*.avg)": "*.avg",
    "Z files (*.zmm, *.zss, *.zrr)": "*.z*",
    "All MT files": "*.*",
}

SUPPORTED_FILE_SUFFIXES: frozenset[str] = frozenset(
    {".edi", ".xml", ".avg", ".zmm", ".zss", ".zrr"}
)


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


class PenetrationDepth1DApp(param.Parameterized):
    """Interactive Panel application for 1-D penetration depth plotting.

    Users select a single MT transfer function file.  Upon loading, the
    Niblett-Bostick depth-of-investigation estimate is computed and rendered
    as an interactive Bokeh figure with per-mode visibility toggles and a
    depth-unit selector.

    Parameters
    ----------
    sizing_mode : str
        Panel sizing mode for the application layout.
    data_directory : str
        Initial directory shown in the file selector.

    Attributes
    ----------
    mt_object : MT or None
        The currently loaded MT object.  ``None`` until a file is loaded.
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

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._mt_object: MT | None = None

        # ── File type filter ──────────────────────────────────────────────
        self._file_pattern_widget = pn.widgets.Select(
            name="File Type Filter",
            options=SUPPORTED_FILE_PATTERNS,
            value="*.edi",
            width=260,
        )

        # ── File selector ─────────────────────────────────────────────────
        self._file_selector = pn.widgets.FileSelector(
            directory=str(Path(self.data_directory).expanduser().resolve()),
            name="Select MT Transfer Function File",
            file_pattern=self._file_pattern_widget.value,
            only_files=True,
        )

        # ── Load button ───────────────────────────────────────────────────
        self._load_button = pn.widgets.Button(
            name="Plot Penetration Depth",
            button_type="primary",
            width=220,
        )
        self._load_button.on_click(self._on_load_clicked)

        # ── Status / error display ────────────────────────────────────────
        self._status = pn.pane.Markdown(
            "_No file loaded yet — select a transfer function file and click Plot._",
            styles={"color": "#555"},
        )

        # ── Plot container ────────────────────────────────────────────────
        self._plot_container = pn.Column(sizing_mode=self.sizing_mode)

        # ── Wire file-pattern changes ─────────────────────────────────────
        self._file_pattern_widget.param.watch(self._on_file_pattern_changed, "value")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def mt_object(self) -> MT | None:
        """The currently loaded :class:`~mtpy.MT` object."""
        return self._mt_object

    # ── Widget callbacks ──────────────────────────────────────────────────

    def _on_file_pattern_changed(self, event: param.parameterized.Event) -> None:
        self._file_selector.file_pattern = event.new

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        """Load selected file and render penetration depth plot."""
        selected: list[str] = list(self._file_selector.value or [])

        if not selected:
            self._set_status("⚠️ No file selected.", warning=True)
            return

        file_path = Path(selected[0])
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_FILE_SUFFIXES:
            self._set_status(
                f"❌ Unsupported file type `{suffix}`.  Supported: "
                + ", ".join(sorted(SUPPORTED_FILE_SUFFIXES)),
                error=True,
            )
            return

        self._set_status("⏳ Loading…")
        self._load_button.disabled = True
        self._plot_container.clear()

        try:
            mt_obj = MT()
            mt_obj.read(str(file_path))
            self._mt_object = mt_obj

            plotter = PlotPenetrationDepth1D(mt_obj, show_plot=False)
            panel_plot = plotter.make_panel(sizing_mode=self.sizing_mode)

            self._plot_container.append(panel_plot)
            self._set_status(f"✅ Loaded **{mt_obj.station}** from `{file_path.name}`.")
        except Exception as exc:
            self._set_status(f"❌ Error: `{type(exc).__name__}: {exc}`", error=True)
            self._mt_object = None
        finally:
            self._load_button.disabled = False

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

    # ── Layout ────────────────────────────────────────────────────────────

    @property
    def view(self) -> pn.viewable.Viewable:
        """Return the complete Panel application layout."""
        return pn.Column(
            pn.pane.Markdown("# Penetration Depth 1-D Viewer"),
            pn.pane.Markdown(
                "Select a transfer function file (EDI, XML, AVG, ZMM/ZSS/ZRR) "
                "and click **Plot Penetration Depth** to visualise the "
                "Niblett-Bostick depth-of-investigation estimate."
            ),
            pn.layout.Divider(),
            pn.Column(
                pn.pane.Markdown("### Select Transfer Function File"),
                self._file_pattern_widget,
                self._file_selector,
                pn.Row(self._load_button),
            ),
            pn.layout.Divider(),
            self._status,
            self._plot_container,
            sizing_mode=self.sizing_mode,
        )

    def servable(self) -> pn.viewable.Viewable:
        """Mark the layout as servable and return it."""
        return self.view.servable()


# Support `panel serve panel_penetration_depth_1d_app.py`
_app = PenetrationDepth1DApp()
_app.view.servable()
