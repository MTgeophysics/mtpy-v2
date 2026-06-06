"""Bokeh implementation of resistivity/phase pseudosection plotting."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy import signal

from mtpy.imaging.mtplot_tools import (
    griddata_interpolate,
    PlotBaseProfile,
    triangulate_interpolation,
)


try:
    from bokeh.io import show as bokeh_show
    from bokeh.layouts import Column, gridplot
    from bokeh.models import (
        BasicTicker,
        ColorBar,
        FixedTicker,
        LinearColorMapper,
        Range1d,
    )
    from bokeh.palettes import (
        Cividis256,
        Inferno256,
        Magma256,
        Plasma256,
        Turbo256,
        Viridis256,
    )
    from bokeh.plotting import figure
except ImportError:  # pragma: no cover - optional dependency
    bokeh_show = None
    Column = None
    gridplot = None
    BasicTicker = None
    ColorBar = None
    FixedTicker = None
    LinearColorMapper = None
    Range1d = None
    Cividis256 = None
    Inferno256 = None
    Magma256 = None
    Plasma256 = None
    Turbo256 = None
    Viridis256 = None
    figure = None


class PlotResPhasePseudoSection(PlotBaseProfile):
    """Plot resistivity and phase pseudosections using Bokeh."""

    def __init__(self, mt_data, **kwargs):
        super().__init__(mt_data, **kwargs)

        self.aspect = kwargs.pop("aspect", "auto")

        self.xtickspace = kwargs.pop("xtickspace", 1)
        self.station_id = kwargs.pop("station_id", kwargs.pop("stationid", [0, 4]))
        self.stationid = self.station_id
        self.linedir = kwargs.pop("linedir", "ew")

        self.plot_xx = False
        self.plot_xy = True
        self.plot_yx = True
        self.plot_yy = False
        self.plot_det = False
        self.plot_resistivity = True
        self.plot_phase = True
        self.data_df = None
        self.n_periods = 60
        self.interpolation_method = "nearest"
        self.nearest_neighbors = 7
        self.interpolation_power = 4

        self.median_filter_kernel = None

        self.x_stretch = 1
        self.y_stretch = 1

        self.station_step = 1

        self.cmap_limits = {
            "res_xx": (-1, 2),
            "res_xy": (0, 3),
            "res_yx": (0, 3),
            "res_yy": (-1, 2),
            "res_det": (0, 3),
            "phase_xx": (-180, 180),
            "phase_xy": (0, 100),
            "phase_yx": (0, 100),
            "phase_yy": (-180, 180),
            "phase_det": (0, 100),
        }

        self.label_dict = {
            "res_xx": r"$\rho_{xx}  \mathrm{[\Omega m]}$",
            "res_xy": r"$\rho_{xy}  \mathrm{[\Omega m]}$",
            "res_yx": r"$\rho_{yx}  \mathrm{[\Omega m]}$",
            "res_yy": r"$\rho_{yy}  \mathrm{[\Omega m]}$",
            "res_det": r"$\rho_{det}  \mathrm{[\Omega m]}$",
            "phase_xx": r"$\phi_{xx}$",
            "phase_xy": r"$\phi_{xy}$",
            "phase_yx": r"$\phi_{yx}$",
            "phase_yy": r"$\phi_{yy}$",
            "phase_det": r"$\phi_{det}$",
        }

        self.res_cmap = "turbo"
        self.phase_cmap = "viridis"

        self.fig = None
        self.layout = None
        self.figures = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot(show=True)

    def _require_bokeh(self):
        if (
            figure is None
            or Column is None
            or gridplot is None
            or ColorBar is None
            or BasicTicker is None
            or FixedTicker is None
            or LinearColorMapper is None
            or Range1d is None
        ):
            raise ImportError(
                "Bokeh is required for PlotResPhasePseudoSection. Install with `pip install bokeh`."
            )

    def _palette_from_name(self, name):
        if name is None:
            return Turbo256
        lname = str(name).lower()
        if "magma" in lname:
            return Magma256
        if "inferno" in lname:
            return Inferno256
        if "plasma" in lname:
            return Plasma256
        if "viridis" in lname:
            return Viridis256
        if "cividis" in lname:
            return Cividis256
        return Turbo256

    def _get_period_array(self, df):
        """Get the period array to interpolate on to."""

        p_min = df.period.min() * self.y_stretch
        p_max = df.period.max() * self.y_stretch

        return np.linspace(p_min, p_max, self.n_periods)

    def _get_n_rows(self):
        n = 0
        if self.plot_resistivity:
            n += 1
        if self.plot_phase:
            n += 1
        return n

    def _get_n_columns(self):
        n = 0
        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if getattr(self, f"plot_{cc}"):
                n += 1
        return n

    def _get_n_subplots(self):
        nr = self._get_n_rows()
        nc = self._get_n_columns()

        subplot_dict = {
            "res_xx": None,
            "res_xy": None,
            "res_yx": None,
            "res_yy": None,
            "res_det": None,
            "phase_xx": None,
            "phase_xy": None,
            "phase_yx": None,
            "phase_yy": None,
            "phase_det": None,
        }

        plot_num = 0
        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if self.plot_resistivity and getattr(self, f"plot_{cc}"):
                plot_num += 1
                subplot_dict[f"res_{cc}"] = (nr, nc, plot_num)

        for cc in ["xx", "xy", "yx", "yy", "det"]:
            if self.plot_phase and getattr(self, f"plot_{cc}"):
                plot_num += 1
                subplot_dict[f"phase_{cc}"] = (nr, nc, plot_num)

        return subplot_dict

    def _get_data_df(self):
        """Get resistivity and phase values in profile order."""

        self._get_profile_line()

        entries = []
        mt_objects = self._get_mt_objects()

        for tf in mt_objects:
            offset = self._get_offset(tf)
            rp = tf.Z

            def _safe_log10(value):
                value = float(value)
                if value <= 0:
                    return 0.0
                return np.log10(value)

            for ii, period in enumerate(tf.period):
                if rp.phase_yx[ii] != 0:
                    rp.phase_yx[ii] += 180

                entry = {
                    "station": tf.station,
                    "offset": offset,
                    "period": np.log10(period),
                    "res_xx": _safe_log10(rp.res_xx[ii]),
                    "res_xy": _safe_log10(rp.res_xy[ii]),
                    "res_yx": _safe_log10(rp.res_yx[ii]),
                    "res_yy": _safe_log10(rp.res_yy[ii]),
                    "res_det": _safe_log10(rp.res_det[ii]),
                    "phase_xx": rp.phase_xx[ii],
                    "phase_xy": rp.phase_xy[ii],
                    "phase_yx": rp.phase_yx[ii] + 180,
                    "phase_yy": rp.phase_yy[ii],
                    "phase_det": rp.phase_det[ii],
                }
                entries.append(entry)

        return pd.DataFrame(entries)

    def _get_offset_station(self, df):
        plot_dict = {"station": [], "offset": []}
        for station in df.station.unique():
            plot_dict["station"].append(
                station[self.station_id[0] : self.station_id[1]]
            )
            plot_dict["offset"].append(
                df.loc[df.station == station, "offset"].unique()[0] * self.x_stretch
            )

        plot_dict["station"] = np.array(plot_dict["station"])
        plot_dict["offset"] = np.array(plot_dict["offset"])

        if self.station_step > 1:
            plot_dict["station"][
                np.arange(1, len(plot_dict["station"]), self.station_step)
            ] = ""

        return plot_dict

    def _get_cmap(self, component):
        if "res" in component:
            return self._palette_from_name(self.res_cmap)
        return self._palette_from_name(self.phase_cmap)

    def _add_colorbar(self, fig, mapper, component):
        cb = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(),
            label_standoff=8,
            title=self.label_dict[component],
        )
        fig.add_layout(cb, "right")

    def _image_from_interpolation(self, comp_df, comp, plot_periods):
        if self.interpolation_method in ["nearest", "linear", "cubic"]:
            x, y, image = griddata_interpolate(
                comp_df.offset * self.x_stretch,
                comp_df.period * self.y_stretch,
                comp_df[comp].to_numpy(),
                self.data_df.offset * self.x_stretch,
                plot_periods,
                self.interpolation_method,
            )

            if self.median_filter_kernel is not None:
                image = signal.medfilt2d(image, self.median_filter_kernel)

            return x, y, image

        triangulation, image, _indices = triangulate_interpolation(
            comp_df.offset * self.x_stretch,
            comp_df.period * self.y_stretch,
            comp_df[comp].to_numpy(),
            comp_df.offset * self.x_stretch,
            comp_df.period * self.y_stretch,
            self.data_df.offset * self.x_stretch,
            plot_periods,
            nearest_neighbors=self.nearest_neighbors,
            interp_pow=self.interpolation_power,
        )

        x_values = np.unique(np.asarray(triangulation.x, dtype=float))
        y_values = np.unique(np.asarray(triangulation.y, dtype=float))
        image = np.asarray(image, dtype=float).reshape(len(y_values), len(x_values))
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        return grid_x, grid_y, image

    def plot(self, show=True):
        """Create Bokeh resistivity/phase pseudosection layout."""

        self._require_bokeh()

        if self.data_df is None:
            self.data_df = self._get_data_df()

        plot_periods = self._get_period_array(self.data_df)
        plot_dict = self._get_offset_station(self.data_df)

        subplot_numbers = self._get_n_subplots()
        rows = self._get_n_rows()
        cols = self._get_n_columns()

        y_ticks = np.arange(plot_periods.max(), plot_periods.min(), -1 * self.y_stretch)
        y_overrides = {}
        for tk in y_ticks:
            key = int(tk / self.y_stretch)
            y_overrides[float(tk)] = self.period_label_dict.get(key, "")

        self.figures = {}

        shared_x = None
        shared_y = None

        for comp, subplot in subplot_numbers.items():
            if subplot is None:
                continue

            cmap = self._get_cmap(comp)
            comp_df = self.data_df.iloc[self.data_df.res_xx.to_numpy().nonzero()]
            x, y, image = self._image_from_interpolation(comp_df, comp, plot_periods)

            nr, nc, nindex = subplot
            row_idx = int((nindex - 1) // nc)
            col_idx = int((nindex - 1) % nc)

            fig = figure(
                title="",
                width=420,
                height=280,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom",
                match_aspect=(self.aspect == "equal"),
                x_axis_label=(
                    "Station" if (nr == 1 or (nr == 2 and row_idx == 1)) else ""
                ),
                y_axis_label="Period (s)" if col_idx == 0 else "",
            )

            if shared_x is None:
                shared_x = Range1d(
                    start=float(np.min(plot_dict["offset"])),
                    end=float(np.max(plot_dict["offset"])),
                )
                shared_y = Range1d(
                    start=float(plot_periods.max()),
                    end=float(plot_periods.min()),
                )
            fig.x_range = shared_x
            fig.y_range = shared_y

            mapper = LinearColorMapper(
                palette=cmap,
                low=self.cmap_limits[comp][0],
                high=self.cmap_limits[comp][1],
            )

            fig.image(
                image=[image],
                x=float(np.nanmin(x)),
                y=float(np.nanmin(y)),
                dw=float(np.nanmax(x) - np.nanmin(x)),
                dh=float(np.nanmax(y) - np.nanmin(y)),
                color_mapper=mapper,
            )

            self._add_colorbar(fig, mapper, comp)

            fig.yaxis.ticker = FixedTicker(ticks=[float(v) for v in y_ticks])
            fig.yaxis.major_label_overrides = y_overrides

            fig.xaxis.ticker = FixedTicker(
                ticks=[float(v) for v in plot_dict["offset"]]
            )
            fig.xaxis.major_label_overrides = {
                float(plot_dict["offset"][ii]): str(plot_dict["station"][ii])
                for ii in range(plot_dict["offset"].shape[0])
            }

            fig.grid.grid_line_alpha = 0.25
            self.figures[comp] = fig

        if len(self.figures) == 0:
            self.layout = None
            self.fig = None
            return None

        row_order = []
        for comp, subplot in subplot_numbers.items():
            if subplot is None or comp not in self.figures:
                continue
            _nr, nc, nindex = subplot
            row_idx = int((nindex - 1) // nc)
            col_idx = int((nindex - 1) % nc)
            if row_idx >= len(row_order):
                row_order.extend([None] * (row_idx - len(row_order) + 1))
            if row_order[row_idx] is None:
                row_order[row_idx] = [None] * cols
            row_order[row_idx][col_idx] = self.figures[comp]

        grid_children = []
        for row in row_order:
            if row is None:
                continue
            grid_children.append(row)

        self.layout = Column(gridplot(grid_children))
        self.fig = self.layout

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def panel(self):
        """Return an interactive, embeddable Panel app for pseudosection controls."""

        try:
            pn = importlib.import_module("panel")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "panel is required for PlotResPhasePseudoSection.panel(). "
                "Install with `pip install panel`."
            ) from exc

        active_components = [
            comp
            for comp in ["xx", "xy", "yx", "yy", "det"]
            if getattr(self, f"plot_{comp}")
        ]
        active_rows = []
        if self.plot_resistivity:
            active_rows.append("Resistivity")
        if self.plot_phase:
            active_rows.append("Phase")

        w_components = pn.widgets.CheckButtonGroup(
            name="Components",
            options=["xx", "xy", "yx", "yy", "det"],
            value=active_components,
        )
        w_rows = pn.widgets.CheckBoxGroup(
            name="Rows",
            options=["Resistivity", "Phase"],
            value=active_rows,
            inline=True,
        )

        palette_options = ["turbo", "viridis", "magma", "inferno", "plasma", "cividis"]
        w_res_palette = pn.widgets.Select(
            name="Resistivity Palette",
            value=(self.res_cmap if self.res_cmap in palette_options else "turbo"),
            options=palette_options,
            width=170,
        )
        w_phase_palette = pn.widgets.Select(
            name="Phase Palette",
            value=(
                self.phase_cmap if self.phase_cmap in palette_options else "viridis"
            ),
            options=palette_options,
            width=170,
        )
        w_res_limits = pn.widgets.RangeSlider(
            name="Resistivity Limits",
            start=-2.0,
            end=4.0,
            value=tuple(self.cmap_limits.get("res_xy", (0.0, 3.0))),
            step=0.1,
            width=240,
        )
        w_phase_limits = pn.widgets.RangeSlider(
            name="Phase Limits",
            start=-180.0,
            end=180.0,
            value=tuple(self.cmap_limits.get("phase_xy", (0.0, 100.0))),
            step=1.0,
            width=240,
        )

        w_interp = pn.widgets.Select(
            name="Interpolation",
            options=["nearest", "linear", "cubic", "delaunay"],
            value=(
                self.interpolation_method
                if self.interpolation_method
                in ["nearest", "linear", "cubic", "delaunay"]
                else "nearest"
            ),
            width=140,
        )
        w_n_periods = pn.widgets.IntInput(
            name="N periods",
            value=int(self.n_periods),
            step=1,
            start=10,
            end=400,
            width=120,
        )
        w_station_step = pn.widgets.IntInput(
            name="Station label step",
            value=int(self.station_step),
            step=1,
            start=1,
            end=50,
            width=140,
        )
        w_x_stretch = pn.widgets.FloatInput(
            name="X stretch",
            value=float(self.x_stretch),
            step=0.1,
            width=120,
        )
        w_y_stretch = pn.widgets.FloatInput(
            name="Y stretch",
            value=float(self.y_stretch),
            step=0.1,
            width=120,
        )

        refresh_btn = pn.widgets.Button(
            name="Refresh", button_type="success", width=120
        )
        status = pn.pane.Markdown(
            "_Adjust controls and click **Refresh** to update the pseudosection._",
            styles={"color": "#555"},
        )
        plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

        def _refresh(_event=None):
            self.plot_resistivity = "Resistivity" in list(w_rows.value)
            self.plot_phase = "Phase" in list(w_rows.value)

            selected_components = set(w_components.value)
            for comp in ["xx", "xy", "yx", "yy", "det"]:
                setattr(self, f"plot_{comp}", comp in selected_components)

            self.res_cmap = str(w_res_palette.value)
            self.phase_cmap = str(w_phase_palette.value)
            self.interpolation_method = str(w_interp.value)
            self.n_periods = int(w_n_periods.value)
            self.station_step = int(w_station_step.value)

            old_x_stretch = float(self.x_stretch)
            self.x_stretch = float(w_x_stretch.value)
            self.y_stretch = float(w_y_stretch.value)

            if not np.isclose(old_x_stretch, self.x_stretch):
                # Offsets are computed with x_stretch in _get_data_df/_get_offset.
                self.data_df = None

            res_limits = tuple(float(value) for value in w_res_limits.value)
            phase_limits = tuple(float(value) for value in w_phase_limits.value)
            for comp in ["res_xx", "res_xy", "res_yx", "res_yy", "res_det"]:
                self.cmap_limits[comp] = res_limits
            for comp in ["phase_xx", "phase_xy", "phase_yx", "phase_yy", "phase_det"]:
                self.cmap_limits[comp] = phase_limits

            layout = self.plot(show=False)
            plot_pane.object = layout

            if layout is None:
                status.object = (
                    "⚠️ No pseudosection data for the current control selection."
                )
                status.styles = {"color": "#7a5200"}
            else:
                status.object = "✅ Pseudosection rendered."
                status.styles = {"color": "#1a6600"}

        refresh_btn.on_click(_refresh)
        _refresh()

        controls = pn.Row(
            pn.Column(
                w_rows,
                w_components,
                w_interp,
                pn.Row(w_n_periods, w_station_step),
                pn.Row(w_x_stretch, w_y_stretch),
                width=360,
                margin=(0, 20, 0, 0),
            ),
            pn.Column(
                w_res_palette,
                w_phase_palette,
                w_res_limits,
                w_phase_limits,
                refresh_btn,
                width=280,
            ),
            sizing_mode="fixed",
        )

        return pn.Column(controls, status, plot_pane, sizing_mode="stretch_width")

    def servable(self, title: str | None = None):
        """Return a standalone servable Panel view of this plot app."""

        app = self.panel()
        if title is None:
            return app.servable()
        return app.servable(title=title)
