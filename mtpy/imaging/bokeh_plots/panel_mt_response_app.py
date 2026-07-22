"""Panel application for interactive MT response plotting.

This module provides a standalone Panel application for visualizing MT response
data (impedance, phase, tipper, phase tensor) with interactive controls and
file selection.

Usage
-----
As a Panel app (in a Jupyter cell or script):
    >>> from mtpy.imaging.bokeh_plots.panel_mt_response_app import MTResponseApp
    >>> app = MTResponseApp()
    >>> app.view.servable()

As a standalone script (from terminal):
    >>> panel serve path/to/panel_mt_response_app.py --autoreload

Example
-------
    >>> # In a Jupyter notebook
    >>> import panel as pn
    >>> pn.extension("bokeh")
    >>> app = MTResponseApp()
    >>> app.view.show()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import panel as pn
import param

from mtpy import MT
from mtpy.imaging.bokeh_plots import PlotMTResponse, PlotMultipleResponses

pn.extension()


SUPPORTED_FILE_PATTERNS = {
    "EDI (*.edi)": "*.edi",
    "XML (*.xml)": "*.xml",
    "J (*.j)": "*.j",
    "AVG (*.avg)": "*.avg",
    "Z files (*.zmm, *.zrr, *.zss)": "*.z*",
}
SUPPORTED_FILE_SUFFIXES = {".edi", ".xml", ".j", ".avg", ".zmm", ".zrr", ".zss"}


def _filesystem_root(path: str | None = None) -> str:
    """Return the root of the filesystem for the given path (drive root on Windows)."""
    p = Path(path).resolve() if path else Path.cwd().resolve()
    return str(p.anchor)  # e.g. "C:\\" on Windows, "/" on Unix


def _default_data_directory() -> str:
    """Return a useful default browse directory for MT sample data."""
    workspace_root = Path(__file__).resolve().parents[4]
    candidate_directories = [
        workspace_root / "mtpy_data" / "mtpy_data" / "data" / "grid",
        workspace_root / "mtpy_data" / "mtpy_data" / "data",
        Path.cwd(),
    ]

    for candidate in candidate_directories:
        if candidate.is_dir():
            return str(candidate)

    return str(Path.cwd())


class MTResponseApp(param.Parameterized):
    """Interactive MT response plotter with file selection.

    This class provides a Panel-based application that allows users to select
    MT data files and interactively visualize MT response parameters.

    Parameters
    ----------
    sizing_mode : str, default "stretch_width"
        Panel sizing mode for the application layout.
    data_directory : str, optional
        Initial directory for file browsing. Defaults to current working directory.
    """

    sizing_mode = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode",
    )

    data_directory = param.String(
        default=_default_data_directory(),
        doc="Directory for file browsing",
    )

    def __init__(self, **params):
        super().__init__(**params)

        # File selector widget
        self._file_pattern_widget = pn.widgets.Select(
            name="File Type",
            options=SUPPORTED_FILE_PATTERNS,
            value="*.edi",
        )

        self._file_selector = pn.widgets.FileSelector(
            directory=str(Path(self.data_directory).expanduser().resolve()),
            root_directory=_filesystem_root(self.data_directory),
            name="Select MT Data File",
            file_pattern=self._file_pattern_widget.value,
            only_files=True,
        )

        # Directory path text input (workaround: FileSelector path bar is read-only)
        self._directory_input = pn.widgets.TextInput(
            name="Browse directory",
            value=str(Path(self.data_directory).expanduser().resolve()),
            placeholder="Paste or type an absolute directory path…",
            sizing_mode="stretch_width",
        )
        self._directory_go_button = pn.widgets.Button(
            name="Go",
            button_type="default",
            width=60,
        )
        self._directory_go_button.on_click(self._on_directory_go_clicked)
        self._directory_input.param.watch(
            self._on_directory_input_enter, "enter_pressed"
        )

        # Plot options
        self._plot_style_widget = pn.widgets.RadioButtonGroup(
            name="Multi-file Plot Style",
            options=["compare", "single"],
            value="compare",
            button_type="primary",
        )
        self._tipper_widget = pn.widgets.Select(
            name="Tipper",
            options={
                "Off": "n",
                "Real": "yr",
                "Imag": "yi",
                "Real + Imag": "yri",
            },
            value="yri",
        )
        self._pt_widget = pn.widgets.Checkbox(name="Phase Tensor", value=True)

        # Interactive control widgets for editing bad data points
        self._apply_edits_button = pn.widgets.Button(
            name="Apply Edits",
            button_type="success",
        )
        self._apply_edits_button.on_click(self._on_apply_edits_clicked)

        self._marked_points_display = pn.pane.Markdown(
            "Click on points in the plots to mark them (they will show as black X marks). Then click 'Apply Edits' to remove them."
        )

        # Info and error display
        self._info = pn.pane.Markdown("")

        # Plotter container (will hold the interactive plot)
        self._plot_container = pn.Column()

        # Status indicator
        self._status = pn.pane.Markdown("")

        # Interactive control for period-level removal / interpolation
        self._interpolate_button = pn.widgets.Button(
            name="Remove / Interpolate Periods",
            button_type="warning",
        )
        self._interpolate_button.on_click(self._on_interpolate_clicked)

        # State tracking for point editing
        self._current_mt_object = None
        self._current_plotter = None
        self._original_z_data = None
        self._original_tipper_data = None
        self._marked_points = {}  # {component_key: set of indices to remove}
        self._component_indices = {}  # Maps period index to period value
        self._z_components = ["xy", "yx", "xx", "yy"]  # Z impedance components

        # State tracking for period-level removal
        self._periods_removed: set = set()
        self._all_periods = None  # np.ndarray of all periods when file is loaded

        # Watch for file selection changes
        self._file_selector.param.watch(self._on_controls_changed, "value")
        self._file_pattern_widget.param.watch(self._on_file_pattern_changed, "value")
        self._plot_style_widget.param.watch(self._on_controls_changed, "value")
        self._tipper_widget.param.watch(self._on_controls_changed, "value")
        self._pt_widget.param.watch(self._on_controls_changed, "value")

        # Keep file selector in sync when data_directory param is changed externally
        self.param.watch(self._on_data_directory_changed, "data_directory")

    @staticmethod
    def _unsupported_selected_files(selected_files):
        """Return selected files with suffixes unsupported by MT.read."""
        return [
            file_path
            for file_path in selected_files
            if Path(file_path).suffix.lower() not in SUPPORTED_FILE_SUFFIXES
        ]

    def _on_data_directory_changed(self, event) -> None:
        """Update the file selector when data_directory param changes."""
        resolved = str(Path(event.new).expanduser().resolve())
        if Path(resolved).is_dir():
            self._file_selector.directory = resolved
            self._directory_input.value = resolved

    def _on_directory_go_clicked(self, event) -> None:
        """Navigate the file selector to the typed directory path."""
        self._navigate_to_directory(self._directory_input.value)

    def _on_directory_input_enter(self, event) -> None:
        """Navigate when the user presses Enter in the directory text box."""
        self._navigate_to_directory(self._directory_input.value)

    def _navigate_to_directory(self, raw_path: str) -> None:
        """Resolve *raw_path* and update the file selector directory."""
        path = Path(raw_path.strip()).expanduser().resolve()
        if path.is_dir():
            self._file_selector.directory = str(path)
            self._file_selector._update_files()
            self._directory_input.value = str(path)
            self._status.object = f"📂 Browsing `{path}`"
        else:
            self._status.object = f"⚠️ Directory not found: `{raw_path.strip()}`"

    def _on_file_pattern_changed(self, event):
        """Update the file selector filter and clear stale selections."""
        self._file_selector.file_pattern = event.new
        self._file_selector.value = []
        self._plot_container.clear()
        self._status.object = ""

    @staticmethod
    def _format_station_label(mt_obj):
        """Format station display as survey.station when survey is available."""
        station = getattr(mt_obj, "station", "Unknown")
        survey_id = getattr(getattr(mt_obj, "survey_metadata", None), "id", None)
        if survey_id:
            return f"{survey_id}.{station}"
        return station

    def _load_mt(self, file_path):
        """Load a file path into an MT object."""
        mt_obj = MT()
        mt_obj.read(file_path)
        return mt_obj

    def _on_controls_changed(self, event):
        """Handle file or option changes and refresh plot."""
        selected_files = self._file_selector.value

        # FileSelector returns a list; get the first file if available
        if not selected_files or len(selected_files) == 0:
            self._plot_container.clear()
            self._status.object = ""
            self._current_mt_object = None
            self._current_plotter = None
            self._marked_points.clear()
            self._update_marked_points_display()
            return

        unsupported_files = self._unsupported_selected_files(selected_files)
        if unsupported_files:
            supported_suffixes = ", ".join(sorted(SUPPORTED_FILE_SUFFIXES))
            file_names = ", ".join(
                Path(file_name).name for file_name in unsupported_files
            )
            self._plot_container.clear()
            self._status.object = (
                f"Unsupported file type for: {file_names}. "
                f"Supported formats: {supported_suffixes}."
            )
            self._current_mt_object = None
            self._current_plotter = None
            return

        try:
            self._status.object = "Loading selected MT data files..."
            self._plot_container.clear()
            self._marked_points.clear()

            selected_files = list(selected_files)

            if len(selected_files) == 1:
                selected_file = selected_files[0]

                # Single-file mode uses PlotMTResponse.
                mt_obj = self._load_mt(selected_file)
                self._current_mt_object = mt_obj
                self._original_z_data = (
                    mt_obj.Z.z.copy() if mt_obj.Z is not None else None
                )
                self._original_tipper_data = (
                    mt_obj.Tipper.tipper.copy() if mt_obj.Tipper is not None else None
                )

                # Initialize marked points and period tracking
                self._marked_points = {}
                self._periods_removed = set()
                self._all_periods = (
                    mt_obj.Z.period.copy() if mt_obj.Z is not None else None
                )
                self._component_indices = {
                    f"{i}_{j}": i for i, j in enumerate(mt_obj.Z.period)
                }

                station_label = self._format_station_label(mt_obj)
                plotter = PlotMTResponse(
                    z_object=mt_obj.Z,
                    t_object=mt_obj.Tipper,
                    pt_obj=mt_obj.pt,
                    station=station_label,
                    show_plot=False,
                    plot_num=2,
                    plot_tipper=self._tipper_widget.value,
                    plot_pt=self._pt_widget.value,
                )
                self._current_plotter = plotter
                panel_plot = plotter.make_panel(
                    sizing_mode=self.sizing_mode,
                    interactive=True,
                )

                # Add click handlers to the plot figures
                self._add_click_handlers_to_plotter(plotter)

                status_label = station_label
            else:
                # Multi-file mode uses PlotMultipleResponses.
                mt_objects = {}
                for index, file_path in enumerate(selected_files):
                    mt_objects[f"mt_{index:03d}"] = self._load_mt(file_path)

                multi_plotter = PlotMultipleResponses(
                    mt_objects,
                    show_plot=False,
                    plot_style=self._plot_style_widget.value,
                    plot_tipper=self._tipper_widget.value,
                    plot_pt=self._pt_widget.value,
                    plot_num=1,
                )
                multi_plotter.compare_mode = "overlay"
                multi_plotter.compare_legend_mode = "station"
                panel_plot = multi_plotter.make_panel(
                    sizing_mode=self.sizing_mode,
                    interactive=False,
                )
                status_label = f"{len(selected_files)} files"
                self._current_mt_object = None
                self._current_plotter = None
                self._marked_points.clear()
                self._update_marked_points_display()

            # Display the plot
            self._plot_container.clear()
            self._plot_container.append(panel_plot)

            self._status.object = (
                f"Successfully loaded {status_label} "
                f"(style={self._plot_style_widget.value}, "
                f"tipper={self._tipper_widget.value}, "
                f"pt={self._pt_widget.value})."
            )

        except Exception as e:
            self._status.object = f"Error: {type(e).__name__}: {str(e)}"
            self._plot_container.clear()
            self._current_mt_object = None
            self._current_plotter = None

    def _update_marked_points_display(self):
        """Update the display of marked points."""
        if not self._marked_points:
            self._marked_points_display.object = "Click on points in the plots to mark them (they will show as black X marks). Then click 'Apply Edits' to remove them."
        else:
            total_marked = sum(len(indices) for indices in self._marked_points.values())
            components_str = ", ".join(
                f"{comp} ({len(indices)} point{'s' if len(indices) != 1 else ''})"
                for comp, indices in sorted(self._marked_points.items())
            )
            self._marked_points_display.object = (
                f"**Marked points ({total_marked} total):** {components_str}"
            )

    def _add_click_handlers_to_plotter(self, plotter):
        """Add click handlers to mark points on the plots."""
        try:
            from bokeh.events import Tap
            from bokeh.models import HoverTool

            # Get figures from plotter
            if not hasattr(plotter, "figures"):
                return

            figures = plotter.figures

            # Map each figure to Z components it displays
            # res and phase figures show all 4 components (xy, yx, xx, yy)
            figure_component_map = {
                "res": ["xy", "yx", "xx", "yy"],
                "phase": ["xy", "yx", "xx", "yy"],
                "tip": ["tx", "ty"],  # Tipper real and imaginary
                "pt": [],  # Phase tensor has no component-wise editing
            }

            for fig_name in ["res", "phase"]:
                if fig_name not in figures:
                    continue

                fig = figures[fig_name]
                components = figure_component_map.get(fig_name, [])

                if not hasattr(fig, "renderers"):
                    continue

                # Add hover tool to show data values
                hover = HoverTool(
                    tooltips=[
                        ("Period", "@index"),
                        ("Value", "@value"),
                    ]
                )
                fig.add_tools(hover)

                # Store a reference to the components for this figure
                fig._app_components = components
                fig._app_instance = self

                # Add tap callback
                def make_tap_callback(figure, component_list):
                    def callback(event):
                        if not hasattr(event, "x") or not hasattr(event, "y"):
                            return

                        # Try to find which data point was clicked
                        x_coord, y_coord = event.x, event.y

                        # Get the first circle glyph renderer to identify clicked point
                        for renderer in figure.renderers:
                            if hasattr(renderer, "data_source") and hasattr(
                                renderer, "glyph"
                            ):
                                source = renderer.data_source
                                if hasattr(source, "data") and "index" in source.data:
                                    # Get indices that are close to the click
                                    indices_array = np.array(
                                        source.data.get("index", [])
                                    )

                                    # For multi-component figures, we'll mark all components
                                    # at the same index
                                    if len(indices_array) > 0:
                                        # Find closest index to the click
                                        closest_idx = int(
                                            np.argmin(np.abs(indices_array - x_coord))
                                        )

                                        # Mark this point for all components in this figure
                                        for component in component_list:
                                            if component not in self._marked_points:
                                                self._marked_points[component] = set()
                                            self._marked_points[component].add(
                                                closest_idx
                                            )

                                        self._update_marked_points_display()
                                        return

                    return callback

                def make_tap_callback(figure, component_list):
                    def callback(event):
                        if not hasattr(event, "x") or not hasattr(event, "y"):
                            return

                        # Convert screen coordinates to data coordinates
                        x_screen, y_screen = event.x, event.y

                        # Get figure canvas dimensions and axis ranges
                        x_range = figure.x_range
                        y_range = figure.y_range

                        # Calculate pixels per unit
                        # Note: These are approximate conversions
                        x_data = x_range.start + (x_screen - figure.left) / (
                            figure.width - figure.left - figure.right
                        ) * (x_range.end - x_range.start)
                        y_data = y_range.end - (y_screen - figure.top) / (
                            figure.height - figure.top - figure.bottom
                        ) * (y_range.end - y_range.start)

                        # Find closest data point
                        for renderer in figure.renderers:
                            if hasattr(renderer, "data_source") and hasattr(
                                renderer, "glyph"
                            ):
                                source = renderer.data_source
                                if hasattr(source, "data") and "index" in source.data:
                                    indices_array = np.array(
                                        source.data.get("index", [])
                                    )

                                    if len(indices_array) > 0:
                                        # Find closest index to the click in data coordinates
                                        closest_idx = int(
                                            np.argmin(np.abs(indices_array - x_data))
                                        )

                                        # Mark this point for all components in this figure
                                        for component in component_list:
                                            if component not in self._marked_points:
                                                self._marked_points[component] = set()
                                            self._marked_points[component].add(
                                                closest_idx
                                            )

                                        self._update_marked_points_display()
                                        return

                    return callback

                fig.on_event(Tap, make_tap_callback(fig, components))

        except Exception as e:
            pass

            # Silently continue if handlers can't be added

    def _on_apply_edits_clicked(self, event):
        """Remove marked points from the MT data by setting them to NaN."""
        if self._current_mt_object is None:
            self._status.object = "No MT data loaded. Select a file first."
            return

        if not self._marked_points:
            self._status.object = "No points marked for removal. Click on points in the plots to mark them."
            return

        try:
            self._status.object = "Applying edits..."

            # Map component names to Z array indices
            # Z tensor shape: (n_periods, 2, 2)
            # z[:, 0, 0] = Zxx, z[:, 0, 1] = Zxy
            # z[:, 1, 0] = Zyx, z[:, 1, 1] = Zyy
            component_to_indices = {
                "xx": (0, 0),
                "xy": (0, 1),
                "yx": (1, 0),
                "yy": (1, 1),
            }

            # Remove marked points from Z data
            if self._current_mt_object.Z is not None:
                z_data = self._current_mt_object.Z.z.copy()
                for component, indices in self._marked_points.items():
                    if component in component_to_indices:
                        i, j = component_to_indices[component]
                        for idx in indices:
                            if 0 <= idx < z_data.shape[0]:
                                z_data[idx, i, j] = np.nan
                self._current_mt_object.Z.z = z_data

            # Remove marked points from Tipper data
            if (
                self._current_mt_object.Tipper is not None
                and "tx" in self._marked_points
            ):
                t_data = self._current_mt_object.Tipper.tipper.copy()
                for idx in self._marked_points.get("tx", []):
                    if 0 <= idx < t_data.shape[0]:
                        t_data[idx, 0, 0] = np.nan
                        t_data[idx, 0, 1] = np.nan
                self._current_mt_object.Tipper.tipper = t_data

            # Reload the plotter with updated data
            station_label = self._format_station_label(self._current_mt_object)
            plotter = PlotMTResponse(
                z_object=self._current_mt_object.Z,
                t_object=self._current_mt_object.Tipper,
                pt_obj=self._current_mt_object.pt,
                station=station_label,
                show_plot=False,
                plot_num=2,
                plot_tipper=self._tipper_widget.value,
                plot_pt=self._pt_widget.value,
            )
            self._current_plotter = plotter

            panel_plot = plotter.make_panel(
                sizing_mode=self.sizing_mode,
                interactive=True,
            )

            # Add click handlers to the new plotter
            self._add_click_handlers_to_plotter(plotter)

            # Display the updated plot
            self._plot_container.clear()
            self._plot_container.append(panel_plot)

            # Clear marked points for next editing session
            removed_count = sum(
                len(indices) for indices in self._marked_points.values()
            )
            self._marked_points.clear()
            self._update_marked_points_display()

            self._status.object = f"Edits applied! Removed {removed_count} point{'s' if removed_count != 1 else ''} (marked as NaN)."

        except Exception as e:
            pass

            self._status.object = f"Error applying edits: {type(e).__name__}: {str(e)}"

    def _on_interpolate_clicked(self, event) -> None:
        """Remove selected periods from MT data and refresh the plot.

        Periods stored in ``_periods_removed`` are dropped from the Z / Tipper
        arrays.  The original data is not modified on disk; this only affects
        the in-memory MT object for the current session.
        """
        if self._current_mt_object is None:
            self._status.object = "No MT data loaded. Select a file first."
            return

        if not self._periods_removed:
            self._status.object = "No periods selected for removal/interpolation."
            return

        try:
            self._status.object = "Removing selected periods…"
            mt_obj = self._current_mt_object

            if mt_obj.Z is not None and self._all_periods is not None:
                keep_mask = np.array(
                    [p not in self._periods_removed for p in mt_obj.Z.period]
                )
                z_new = mt_obj.Z.z[keep_mask]
                z_err_new = (
                    mt_obj.Z.z_error[keep_mask]
                    if mt_obj.Z.z_error is not None
                    else None
                )
                period_new = mt_obj.Z.period[keep_mask]
                mt_obj.Z.z = z_new
                if z_err_new is not None:
                    mt_obj.Z.z_error = z_err_new
                mt_obj.Z.period = period_new

            if mt_obj.Tipper is not None and mt_obj.Tipper.tipper is not None:
                keep_mask_t = np.array(
                    [p not in self._periods_removed for p in mt_obj.Tipper.period]
                )
                mt_obj.Tipper.tipper = mt_obj.Tipper.tipper[keep_mask_t]
                if mt_obj.Tipper.tipper_error is not None:
                    mt_obj.Tipper.tipper_error = mt_obj.Tipper.tipper_error[keep_mask_t]
                mt_obj.Tipper.period = mt_obj.Tipper.period[keep_mask_t]

            n_removed = len(self._periods_removed)
            self._periods_removed.clear()

            # Rebuild the plot with updated data
            station_label = self._format_station_label(mt_obj)
            plotter = PlotMTResponse(
                z_object=mt_obj.Z,
                t_object=mt_obj.Tipper,
                pt_obj=mt_obj.pt,
                station=station_label,
                show_plot=False,
                plot_num=2,
                plot_tipper=self._tipper_widget.value,
                plot_pt=self._pt_widget.value,
            )
            self._current_plotter = plotter
            panel_plot = plotter.make_panel(
                sizing_mode=self.sizing_mode,
                interactive=True,
            )
            self._add_click_handlers_to_plotter(plotter)
            self._plot_container.clear()
            self._plot_container.append(panel_plot)

            self._status.object = (
                f"Removed {n_removed} period{'s' if n_removed != 1 else ''}."
            )

        except Exception as exc:
            self._status.object = f"Error removing periods: {type(exc).__name__}: {exc}"

    @property
    def view(self) -> pn.Column:
        """Return the complete Panel application layout."""
        return pn.Column(
            pn.pane.Markdown("# MT Response Viewer"),
            pn.pane.Markdown(
                "Select one file for a single-station response plot, or select "
                "multiple files to use multi-response compare/single layouts."
            ),
            pn.pane.Markdown(
                "Use the file-type filter before browsing. Supported formats are "
                ".edi, .xml, .j, .avg, .zmm, .zrr, and .zss."
            ),
            pn.pane.Markdown("---"),
            pn.pane.Markdown(
                "_Paste or type an absolute path and click **Go** to jump to any "
                "directory. Then browse and select files below._",
                styles={"color": "#555", "font-size": "0.85em"},
            ),
            pn.Row(
                self._directory_input,
                self._directory_go_button,
                align="end",
                sizing_mode="stretch_width",
            ),
            self._file_selector,
            pn.Row(
                self._file_pattern_widget,
                self._plot_style_widget,
                self._tipper_widget,
                self._pt_widget,
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown("### Edit Bad Data Points (Single File Mode Only)"),
            pn.pane.Markdown(
                "**Instructions:** Hover over points to see their values. Click on a point to mark it (it will show as a black X). "
                "Then click 'Apply Edits' to remove the marked points. This works per component (xy, yx, xx, yy)."
            ),
            pn.Row(
                self._marked_points_display,
                self._apply_edits_button,
                self._interpolate_button,
                sizing_mode="stretch_width",
            ),
            self._status,
            pn.pane.Markdown("---"),
            self._plot_container,
            sizing_mode=self.sizing_mode,
        )


# Make the app servable when running `panel serve panel_mt_response_app.py`
_app = MTResponseApp()
app = _app.view.servable()
