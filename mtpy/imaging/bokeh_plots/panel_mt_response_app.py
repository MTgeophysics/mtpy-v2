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

import panel as pn
import param

from mtpy import MT
from mtpy.imaging.bokeh_plots import PlotMTResponse


# Load bokeh extension if available (optional dependency)
try:
    pn.extension("bokeh")
except Exception:
    pass  # Bokeh will be auto-loaded when needed


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
        default=".",
        doc="Directory for file browsing",
    )

    def __init__(self, **params):
        super().__init__(**params)

        # File selector widget
        self._file_selector = pn.widgets.FileSelector(
            directory=self.data_directory,
            name="Select MT Data File",
        )

        # Info and error display
        self._info = pn.pane.Markdown("")

        # Plotter container (will hold the interactive plot)
        self._plot_container = pn.Column()

        # Status indicator
        self._status = pn.pane.Markdown("")

        # Watch for file selection changes
        self._file_selector.param.watch(self._on_file_selected, "value")

    def _on_file_selected(self, event):
        """Handle file selection and load MT data."""
        selected_files = event.new

        # FileSelector returns a list; get the first file if available
        if not selected_files or len(selected_files) == 0:
            self._plot_container.clear()
            self._status.object = ""
            return

        selected_file = (
            selected_files[0] if isinstance(selected_files, list) else selected_files
        )

        try:
            self._status.object = f"📋 Loading {Path(selected_file).name}..."
            self._plot_container.clear()

            # Load MT data
            m = MT()
            m.read(selected_file)

            # Create plotter
            plotter = PlotMTResponse(
                z_object=m.Z,
                t_object=m.Tipper,
                pt_obj=m.pt,
                station=m.station,
                show_plot=False,
                plot_num=2,
            )

            # Generate interactive panel
            panel_plot = plotter.make_panel(
                sizing_mode=self.sizing_mode,
                interactive=True,
            )

            # Display the plot
            self._plot_container.clear()
            self._plot_container.append(panel_plot)

            self._status.object = (
                f"✅ Successfully loaded **{m.station}** from {Path(selected_file).name}"
            )

        except Exception as e:
            self._status.object = f"❌ Error: {type(e).__name__}: {str(e)}"
            self._plot_container.clear()

    @property
    def view(self) -> pn.Column:
        """Return the complete Panel application layout."""
        return pn.Column(
            pn.pane.Markdown("# MT Response Viewer"),
            pn.pane.Markdown(
                "Select an MT data file (EDI, XML, J format) to visualize its "
                "impedance, phase, tipper, and phase tensor response."
            ),
            pn.pane.Markdown("---"),
            self._file_selector,
            self._status,
            pn.pane.Markdown("---"),
            self._plot_container,
            sizing_mode=self.sizing_mode,
        )


# Make the app servable when running `panel serve panel_mt_response_app.py`
_app = MTResponseApp()
app = _app.view.servable()
