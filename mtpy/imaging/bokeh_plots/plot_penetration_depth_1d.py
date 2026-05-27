"""Bokeh implementation of 1D penetration depth plotting."""

from __future__ import annotations

import importlib

import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.plotting import figure

from mtpy.imaging.bokeh_plots.base import BokehPlotBase


class PlotPenetrationDepth1D(BokehPlotBase):
    """Plot the depth of penetration based on the Niblett-Bostick approximation."""

    _MARKER_MAP = {
        "o": "circle",
        "s": "square",
        "v": "triangle",
        "^": "inverted_triangle",
        "d": "diamond",
        "x": "x",
        "+": "cross",
        "*": "asterisk",
    }

    _LINE_DASH_MAP = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdot",
        "solid": "solid",
        "dashed": "dashed",
        "dotted": "dotted",
        "dashdot": "dashdot",
    }

    def __init__(self, tf, **kwargs):
        self.tf = tf
        self.fig = None
        self.layout = None
        self._renderers: dict[str, list] = {}

        super().__init__(**kwargs)

        self.depth_units = "km"

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    @property
    def depth_units(self):
        """Depth units."""

        return self._depth_units

    @depth_units.setter
    def depth_units(self, value):
        """Depth units."""

        self._depth_units = value
        if value in ["km"]:
            self.depth_scale = 1.0 / 1000
        elif value in ["m"]:
            self.depth_scale = 1
        else:
            raise ValueError("depth_units must be either 'km' or 'm'")

    def _require_bokeh(self):
        if ColumnDataSource is None or HoverTool is None or Range1d is None:
            raise ImportError(
                "Bokeh is required for PlotPenetrationDepth1D. Install with `pip install bokeh`."
            )

    def _get_nb_estimation(self):
        """Get the depth of investigation estimation."""

        return self.tf.Z.estimate_depth_of_investigation()

    @staticmethod
    def _tuple_to_hex(color):
        if isinstance(color, str):
            short_map = {
                "k": "#000000",
                "w": "#ffffff",
                "r": "#ff0000",
                "g": "#008000",
                "b": "#0000ff",
                "c": "#00ffff",
                "m": "#ff00ff",
                "y": "#ffff00",
            }
            return short_map.get(color.lower(), color)
        if isinstance(color, tuple) and len(color) == 3:
            r, g, b = [int(np.clip(c, 0, 1) * 255) for c in color]
            return f"#{r:02x}{g:02x}{b:02x}"
        return color

    def _marker_name(self, marker):
        return self._MARKER_MAP.get(marker, "circle")

    def _line_dash(self, linestyle):
        return self._LINE_DASH_MAP.get(linestyle, "solid")

    def _figure_dimensions(self):
        if self.fig_size is None:
            return 720, 660

        width = max(int(self.fig_size[0] * self.fig_dpi), 300)
        height = max(int(self.fig_size[1] * self.fig_dpi), 300)
        return width, height

    def _sorted_depth_data(self, depth_array):
        period = np.asarray(depth_array["period"], dtype=float)
        order = np.argsort(period)

        data = {
            "period": period[order],
            "depth_min": np.asarray(depth_array["depth_min"], dtype=float)[order],
            "depth_max": np.asarray(depth_array["depth_max"], dtype=float)[order],
        }

        for comp in ["xy", "yx", "det"]:
            data[f"depth_{comp}"] = np.asarray(
                depth_array[f"depth_{comp}"], dtype=float
            )[order]

        return data

    def _make_figure(self, x_limits, y_limits, title=""):
        width, height = self._figure_dimensions()
        return figure(
            title=title,
            x_axis_type="log",
            y_axis_type="log",
            width=width,
            height=height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
            x_range=Range1d(start=float(x_limits[0]), end=float(x_limits[1])),
            y_range=Range1d(start=float(y_limits[1]), end=float(y_limits[0])),
        )

    def _component_renderer(self, fig, source, comp_label, color, marker, ls):
        glyph_color = self._tuple_to_hex(color)

        line_renderer = fig.line(
            x="depth",
            y="period",
            source=source,
            color=glyph_color,
            line_width=max(self.lw, 1),
            line_dash=self._line_dash(ls),
        )

        scatter_method = getattr(fig, self._marker_name(marker), fig.circle)
        scatter_renderer = scatter_method(
            x="depth",
            y="period",
            source=source,
            size=max(int(self.marker_size * 2), 4),
            color=glyph_color,
            line_color=glyph_color,
            legend_label=comp_label,
        )

        return line_renderer, scatter_renderer

    def plot(self):
        """Plot the depth of investigation as a Bokeh 1D plot."""

        self._require_bokeh()

        depth_array = self._sorted_depth_data(self._get_nb_estimation())

        depth_values = (
            np.concatenate([depth_array["depth_min"], depth_array["depth_max"]])
            * self.depth_scale
        )
        finite_depth = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
        if finite_depth.size == 0:
            raise ValueError("No valid depth values found for plotting.")

        period_values = depth_array["period"]
        finite_period = period_values[np.isfinite(period_values) & (period_values > 0)]
        if finite_period.size == 0:
            raise ValueError("No valid period values found for plotting.")

        x_limits = self.set_period_limits(finite_depth)
        y_limits = self.set_period_limits(finite_period)

        self.fig = self._make_figure(
            x_limits,
            y_limits,
            title=f"Depth of investigation for {self.tf.station}",
        )
        self.layout = self.fig

        band_source = ColumnDataSource(
            data={
                "x": np.concatenate(
                    [
                        depth_array["depth_min"] * self.depth_scale,
                        (depth_array["depth_max"] * self.depth_scale)[::-1],
                    ]
                ),
                "y": np.concatenate(
                    [depth_array["period"], depth_array["period"][::-1]]
                ),
            }
        )
        self.fig.patch(
            x="x",
            y="y",
            source=band_source,
            fill_color=self._tuple_to_hex((0.5, 0.5, 0.5)),
            fill_alpha=0.5,
            line_color=None,
        )

        self._renderers = {}
        label_list = ["TE", "TM", "DET"]
        for comp, label in zip(["xy", "yx", "det"], label_list):
            source = ColumnDataSource(
                data={
                    "depth": depth_array[f"depth_{comp}"] * self.depth_scale,
                    "period": depth_array["period"],
                }
            )
            line_renderer, scatter_renderer = self._component_renderer(
                self.fig,
                source,
                label,
                getattr(self, f"{comp}_color"),
                getattr(self, f"{comp}_marker"),
                getattr(self, f"{comp}_ls"),
            )
            self._renderers[comp] = [line_renderer, scatter_renderer]
            self.fig.add_tools(
                HoverTool(
                    renderers=[scatter_renderer, line_renderer],
                    tooltips=[
                        ("Period (s)", "@period{0.000}"),
                        ("Depth", "@depth{0.000}"),
                    ],
                )
            )

        self.fig.xaxis.axis_label = f"Depth ({self.depth_units})"
        self.fig.yaxis.axis_label = "Period (s)"
        self.fig.grid.grid_line_alpha = 0.25

        if len(self.fig.legend) > 0:
            self.fig.legend.click_policy = "hide"
            self.fig.legend.location = "top_right"
            self.fig.legend.label_text_font_size = f"{self.font_size}px"

        if self.show_plot and show is not None:
            show(self.fig)

        return self.layout

    def make_panel(self, sizing_mode: str = "stretch_width"):
        """Return a Panel layout with interactive mode and units controls.

        Renders the penetration depth figure wrapped in a Panel column with:

        * **Modes** — a :class:`~panel.widgets.CheckButtonGroup` to toggle
          the TE (Zxy), TM (Zyx) and Determinant curves on/off.
        * **Depth Units** — a :class:`~panel.widgets.RadioButtonGroup` to
          switch between kilometres and metres.  Switching rebuilds the
          figure in-place and updates the embedded Bokeh pane.

        Parameters
        ----------
        sizing_mode : str, optional
            Panel sizing mode, by default ``"stretch_width"``.

        Returns
        -------
        panel.Column
            A Panel layout containing the controls and the Bokeh figure.

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

        # ── Controls ─────────────────────────────────────────────────────
        mode_widget = pn.widgets.CheckButtonGroup(
            name="Modes",
            options={"TE (Zxy)": "xy", "TM (Zyx)": "yx", "Determinant": "det"},
            value=["xy", "yx", "det"],
            button_type="light",
        )

        depth_units_widget = pn.widgets.RadioButtonGroup(
            name="Depth Units",
            options=["km", "m"],
            value=self.depth_units,
            button_type="primary",
        )

        # ── Bokeh pane (mutable reference) ────────────────────────────────
        bokeh_pane = pn.pane.Bokeh(self.fig, sizing_mode=sizing_mode)

        # ── Callbacks ─────────────────────────────────────────────────────
        def _toggle_modes(event):
            for comp, renderers in self._renderers.items():
                visible = comp in (event.new or [])
                for r in renderers:
                    r.visible = visible

        def _change_depth_units(event):
            self.depth_units = event.new
            self.plot()
            bokeh_pane.object = self.fig

        mode_widget.param.watch(_toggle_modes, "value")
        depth_units_widget.param.watch(_change_depth_units, "value")

        controls = pn.Row(
            pn.Column(
                pn.pane.Markdown("**Modes:**", margin=(0, 0, 2, 0)),
                mode_widget,
            ),
            pn.Column(
                pn.pane.Markdown("**Depth Units:**", margin=(0, 0, 2, 0)),
                depth_units_widget,
            ),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            controls,
            bokeh_pane,
            sizing_mode=sizing_mode,
        )
