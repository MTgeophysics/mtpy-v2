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

import panel as pn
import param

from mtpy import MT
from mtpy.imaging.bokeh_plots import PlotMTResponse, PlotMultipleResponses


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

        # Info and error display
        self._info = pn.pane.Markdown("")

        # Plotter container (will hold the interactive plot)
        self._plot_container = pn.Column()

        # Status indicator
        self._status = pn.pane.Markdown("")

        # Watch for file selection changes
        self._file_selector.param.watch(self._on_controls_changed, "value")
        self._plot_style_widget.param.watch(self._on_controls_changed, "value")
        self._tipper_widget.param.watch(self._on_controls_changed, "value")
        self._pt_widget.param.watch(self._on_controls_changed, "value")

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
            return

        try:
            self._status.object = "Loading selected MT data files..."
            self._plot_container.clear()

            selected_files = list(selected_files)

            if len(selected_files) == 1:
                selected_file = selected_files[0]

                # Single-file mode uses PlotMTResponse.
                mt_obj = self._load_mt(selected_file)
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
                panel_plot = plotter.make_panel(
                    sizing_mode=self.sizing_mode,
                    interactive=True,
                )
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

    @property
    def view(self) -> pn.Column:
        """Return the complete Panel application layout."""
        return pn.Column(
            pn.pane.Markdown("# MT Response Viewer"),
            pn.pane.Markdown(
                "Select one file for a single-station response plot, or select "
                "multiple files to use multi-response compare/single layouts."
            ),
            pn.pane.Markdown("---"),
            self._file_selector,
            pn.Row(
                self._plot_style_widget,
                self._tipper_widget,
                self._pt_widget,
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
