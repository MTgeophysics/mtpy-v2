"""Base class for Bokeh-based MT plotting classes with Panel support."""

from __future__ import annotations

import importlib

from mtpy.imaging.mtplot_tools import PlotBase


class BokehPlotBase(PlotBase):
    """Base class providing Panel integration for Bokeh plotting classes.

    Extends :class:`~mtpy.imaging.mtplot_tools.PlotBase` with a default
    :meth:`make_panel` implementation that wraps the Bokeh layout in a
    ``panel.pane.Bokeh`` pane.  Subclasses should override :meth:`make_panel`
    to add interactive widget controls.
    """

    def make_panel(self, sizing_mode: str = "stretch_width"):
        """Return a Panel pane wrapping the Bokeh layout.

        If :meth:`plot` has not been called yet, it is called automatically.

        Parameters
        ----------
        sizing_mode : str, optional
            Panel sizing mode, by default ``"stretch_width"``.

        Returns
        -------
        panel.pane.Bokeh
            A Panel pane containing the Bokeh figure.

        Raises
        ------
        ImportError
            If ``panel`` is not installed.
        """
        try:
            pn = importlib.import_module("panel")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Panel is required for make_panel(). Install with `pip install panel`."
            ) from exc

        if self.layout is None:
            self.plot()

        return pn.pane.Bokeh(self.layout, sizing_mode=sizing_mode)
