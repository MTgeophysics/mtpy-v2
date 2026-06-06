"""Bokeh implementation of phase tensor pseudosection plotting."""

from __future__ import annotations

import importlib

import numpy as np

from mtpy.imaging.mtplot_tools import PlotBaseProfile


try:
    from bokeh.io import show as bokeh_show
    from bokeh.layouts import Column
    from bokeh.models import (
        Arrow,
        BasicTicker,
        ColorBar,
        FixedTicker,
        LinearColorMapper,
        NormalHead,
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
    Arrow = None
    BasicTicker = None
    ColorBar = None
    FixedTicker = None
    LinearColorMapper = None
    NormalHead = None
    Range1d = None
    Cividis256 = None
    Inferno256 = None
    Magma256 = None
    Plasma256 = None
    Turbo256 = None
    Viridis256 = None
    figure = None


class PlotPhaseTensorPseudoSection(PlotBaseProfile):
    """Plot phase tensor ellipses in pseudosection format using Bokeh."""

    _PLAIN_COLORBAR_LABELS = {
        "phiminang": "Phi_min (deg)",
        "phimin": "Phi_min (deg)",
        "phimaxang": "Phi_max (deg)",
        "phimax": "Phi_max (deg)",
        "phidet": "Det(Phi) (deg)",
        "skew": "Skew (deg)",
        "normalized_skew": "Normalized Skew (deg)",
        "ellipticity": "Ellipticity",
        "skew_seg": "Skew (deg)",
        "normalized_skew_seg": "Normalized Skew (deg)",
        "geometric_mean": "sqrt(Phi_min * Phi_max)",
        "strike": "Azimuth (deg)",
        "azimuth": "Azimuth (deg)",
    }

    def __init__(self, mt_data, **kwargs):
        super().__init__(mt_data, **kwargs)

        self._rotation_angle = 0
        self.plot_pt = True
        self.aspect = kwargs.pop("aspect", "equal")

        self.x_stretch = 1000
        self.y_stretch = 10
        self.y_scale = "period"

        self.station_id = [0, None]
        self.profile_vector = None
        self.profile_angle = None
        self.profile_line = None
        self.profile_reverse = False

        self.ellipse_size = 20
        self.ellipse_alpha = 0.85
        self.arrow_size = 4000
        self.arrow_head_width = 250
        self.arrow_head_length = 400

        self.fig = None
        self.layout = None
        self.station_list = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot(show=True)

    def _require_bokeh(self):
        if (
            figure is None
            or Column is None
            or Arrow is None
            or NormalHead is None
            or ColorBar is None
            or BasicTicker is None
            or FixedTicker is None
            or Range1d is None
            or LinearColorMapper is None
        ):
            raise ImportError(
                "Bokeh is required for PlotPhaseTensorPseudoSection. Install with `pip install bokeh`."
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

    def _scalar_to_color(self, value, value_min, value_max):
        palette = self._palette_from_name(self.ellipse_cmap)
        if palette is None:
            return "#808080"

        if not np.isfinite(value):
            return "#808080"

        if value_max <= value_min:
            return palette[-1]

        alpha = (float(value) - float(value_min)) / (
            float(value_max) - float(value_min)
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        idx = int(alpha * (len(palette) - 1))
        return palette[idx]

    def _get_patch(self, tf):
        """Get and draw phase-tensor ellipses and optional tipper arrows."""

        plot_x = self._get_offset(tf)

        pt_obj = tf.pt
        has_pt = False
        if pt_obj is not None and pt_obj._has_tf() and self.plot_pt:
            phimax = pt_obj.phimax
            phimin = pt_obj.phimin
            azimuth = pt_obj.azimuth
            has_pt = True

        has_tipper = False
        if tf.Tipper is not None and "y" in self.plot_tipper:
            t_obj = tf.Tipper
            has_tipper = True
            t_mag_re = t_obj.mag_real
            t_mag_im = t_obj.mag_imag
            t_ang_re = t_obj.angle_real
            t_ang_im = t_obj.angle_imag

        color_array = self.get_pt_color_array(pt_obj) if pt_obj is not None else []
        if pt_obj is None:
            return (
                plot_x,
                getattr(tf, "tf_id", tf.station)[
                    self.station_id[0] : self.station_id[1]
                ],
            )

        for index, ff in enumerate(pt_obj.frequency):
            if self.y_scale == "period":
                plot_y = np.log10(1.0 / ff) * self.y_stretch
            else:
                plot_y = np.log10(ff) * self.y_stretch

            if has_pt:
                if (
                    phimax[index] == 0
                    or phimax[index] > 100
                    or phimin[index] == 0
                    or phimin[index] > 100
                ):
                    pass
                else:
                    scaling = self.ellipse_size / np.nanmax(phimax)
                    eheight = float(phimin[index] * scaling)
                    ewidth = float(phimax[index] * scaling)
                    azm = 90 - float(azimuth[index])
                    if self.y_scale == "period":
                        azm = 90 + float(azimuth[index])

                    fill_color = self._scalar_to_color(
                        float(color_array[index]),
                        self.ellipse_range[0],
                        self.ellipse_range[1],
                    )

                    self.fig.ellipse(
                        x=[plot_x],
                        y=[plot_y],
                        width=[ewidth],
                        height=[eheight],
                        angle=[np.deg2rad(azm)],
                        fill_color=fill_color,
                        fill_alpha=self.ellipse_alpha,
                        line_color="#222222",
                        line_width=max(self.lw, 1),
                    )

            if has_tipper:
                if (
                    "r" in self.plot_tipper
                    and t_obj.mag_real[index] <= self.arrow_threshold
                ):
                    if self.y_scale == "period":
                        txr = (
                            t_mag_re[index]
                            * self.arrow_size
                            * np.sin(
                                np.deg2rad(-t_ang_re[index] + 180)
                                + self.arrow_direction * np.pi
                            )
                        )
                        tyr = (
                            t_mag_re[index]
                            * self.arrow_size
                            * np.cos(
                                np.deg2rad(-t_ang_re[index] + 180)
                                + self.arrow_direction * np.pi
                            )
                        )
                    else:
                        txr = (
                            t_mag_re[index]
                            * self.arrow_size
                            * np.sin(
                                np.deg2rad(t_ang_re[index])
                                + self.arrow_direction * np.pi
                            )
                        )
                        tyr = (
                            t_mag_re[index]
                            * self.arrow_size
                            * np.cos(
                                np.deg2rad(t_ang_re[index])
                                + self.arrow_direction * np.pi
                            )
                        )

                    self.fig.add_layout(
                        Arrow(
                            end=NormalHead(
                                size=max(int(self.arrow_head_width / 50), 6),
                                fill_color=self.arrow_color_real,
                                line_color=self.arrow_color_real,
                            ),
                            x_start=float(plot_x),
                            y_start=float(plot_y),
                            x_end=float(plot_x + txr),
                            y_end=float(plot_y + tyr),
                            line_color=self.arrow_color_real,
                            line_width=max(self.arrow_lw * 150, 1),
                        )
                    )

                if "i" in self.plot_tipper and t_mag_im[index] <= self.arrow_threshold:
                    if self.y_scale == "period":
                        txi = (
                            t_mag_im[index]
                            * self.arrow_size
                            * np.sin(
                                np.deg2rad(-t_ang_im[index] + 180)
                                + self.arrow_direction * np.pi
                            )
                        )
                        tyi = (
                            t_mag_im[index]
                            * self.arrow_size
                            * np.cos(
                                np.deg2rad(-t_ang_im[index] + 180)
                                + self.arrow_direction * np.pi
                            )
                        )
                    else:
                        txi = (
                            t_mag_im[index]
                            * self.arrow_size
                            * np.sin(
                                np.deg2rad(t_ang_im[index])
                                + self.arrow_direction * np.pi
                            )
                        )
                        tyi = (
                            t_mag_im[index]
                            * self.arrow_size
                            * np.cos(
                                np.deg2rad(t_ang_im[index])
                                + self.arrow_direction * np.pi
                            )
                        )

                    self.fig.add_layout(
                        Arrow(
                            end=NormalHead(
                                size=max(int(self.arrow_head_width / 50), 6),
                                fill_color=self.arrow_color_imag,
                                line_color=self.arrow_color_imag,
                            ),
                            x_start=float(plot_x),
                            y_start=float(plot_y),
                            x_end=float(plot_x + txi),
                            y_end=float(plot_y + tyi),
                            line_color=self.arrow_color_imag,
                            line_width=max(self.arrow_lw * 150, 1),
                            line_dash="dashed",
                        )
                    )

        station_id = getattr(tf, "tf_id", tf.station)
        return plot_x, station_id[self.station_id[0] : self.station_id[1]]

    def _add_colorbar(self):
        mapper = LinearColorMapper(
            palette=self._palette_from_name(self.ellipse_cmap),
            low=self.ellipse_range[0],
            high=self.ellipse_range[1],
        )
        colorbar_title = self._PLAIN_COLORBAR_LABELS.get(
            self.ellipse_colorby,
            str(self.ellipse_colorby),
        )
        self.fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title=colorbar_title,
            ),
            "right",
        )

    def _add_tipper_legend(self):
        if "y" in self.plot_tipper:
            if "r" in self.plot_tipper:
                self.fig.line(
                    x=[np.nan],
                    y=[np.nan],
                    color=self.arrow_color_real,
                    legend_label="Real",
                    line_width=2,
                )
            if "i" in self.plot_tipper:
                self.fig.line(
                    x=[np.nan],
                    y=[np.nan],
                    color=self.arrow_color_imag,
                    legend_label="Imag",
                    line_width=2,
                    line_dash="dashed",
                )

            if len(self.fig.legend) > 0:
                self.fig.legend.click_policy = "hide"
                self.fig.legend.location = "bottom_right"

    def plot(self, show=True):
        """Plot phase tensor pseudosection using Bokeh."""

        self._require_bokeh()

        self.fig = figure(
            title="",
            width=1000,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=(self.aspect == "equal"),
        )

        self._get_profile_line()
        mt_objects = self._get_mt_objects()

        y_min = 1
        y_max = 1
        station_list = np.zeros(
            len(mt_objects),
            dtype=[("offset", float), ("station", "U10")],
        )

        for ii, tf in enumerate(mt_objects):
            offset, station = self._get_patch(tf)
            station_list[ii]["station"] = station
            station_list[ii]["offset"] = offset

            if np.log10(tf.frequency.min()) < y_min:
                y_min = np.log10(tf.frequency.min()) * self.y_stretch
            if np.log10(tf.frequency.max()) > y_max:
                y_max = np.log10(tf.frequency.max()) * self.y_stretch

        y_min = np.floor(y_min / self.y_stretch) * self.y_stretch
        y_max = np.ceil(y_max / self.y_stretch) * self.y_stretch

        self.station_list = np.sort(station_list, order="offset")

        if self.y_scale == "period":
            y_label = "Period (s)"
            p_min = float(y_min)
            p_max = float(y_max)
            y_min = -1 * p_min
            y_max = -1 * p_max
        else:
            y_label = "Frequency (Hz)"

        self.fig.yaxis.axis_label = y_label

        y_ticks = np.arange(y_min, y_max, self.y_stretch * np.sign(y_max))
        y_overrides = {}
        for tk in y_ticks:
            key = int(tk / self.y_stretch)
            y_overrides[float(tk)] = f"10^{key}"
        self.fig.yaxis.ticker = FixedTicker(ticks=[float(v) for v in y_ticks])
        self.fig.yaxis.major_label_overrides = y_overrides

        x_ticks = [float(v) for v in self.station_list["offset"]]
        x_overrides = {
            float(self.station_list["offset"][ii]): str(
                self.station_list["station"][ii]
            )
            for ii in range(self.station_list.shape[0])
        }
        self.fig.xaxis.ticker = FixedTicker(ticks=x_ticks)
        self.fig.xaxis.major_label_overrides = x_overrides
        self.fig.xaxis.axis_label = "Station"

        x_padding = float(self.ellipse_size) / 2.0
        if self.x_limits is None:
            self.fig.x_range = Range1d(
                start=float(np.floor(self.station_list["offset"].min()) - x_padding),
                end=float(np.ceil(self.station_list["offset"].max()) + x_padding),
            )
        else:
            self.fig.x_range = Range1d(
                start=float(self.x_limits[0]) - x_padding,
                end=float(self.x_limits[1]) + x_padding,
            )

        if self.y_limits is None:
            self.fig.y_range = Range1d(start=float(y_min), end=float(y_max))
        else:
            pmin = np.log10(self.y_limits[0]) * self.y_stretch
            pmax = np.log10(self.y_limits[1]) * self.y_stretch
            self.fig.y_range = Range1d(
                start=float(np.floor(pmin)), end=float(np.ceil(pmax))
            )

        if self.plot_title is not None:
            self.fig.title.text = self.plot_title

        self.fig.grid.grid_line_alpha = 0.25
        self.fig.grid.grid_line_color = "#404040"

        self._add_colorbar()
        self._add_tipper_legend()

        self.layout = Column(self.fig)

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def panel(self):
        """Return an interactive, embeddable Panel app for pseudosection controls."""

        try:
            pn = importlib.import_module("panel")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "panel is required for PlotPhaseTensorPseudoSection.panel(). "
                "Install with `pip install panel`."
            ) from exc

        palette_options = ["turbo", "viridis", "magma", "inferno", "plasma", "cividis"]

        w_plot_pt = pn.widgets.Checkbox(
            name="Plot Phase Tensor", value=bool(self.plot_pt)
        )
        w_plot_tipper = pn.widgets.Select(
            name="Tipper",
            options=["n", "ri", "r", "i"],
            value=(
                self.plot_tipper if self.plot_tipper in {"n", "ri", "r", "i"} else "n"
            ),
            width=140,
        )
        w_y_scale = pn.widgets.Select(
            name="Y scale",
            options=["period", "frequency"],
            value=(
                self.y_scale if self.y_scale in {"period", "frequency"} else "period"
            ),
            width=140,
        )
        colorby_options = [
            "phimin",
            "phiminang",
            "phimax",
            "phimaxang",
            "phidet",
            "skew",
            "skew_seg",
            "ellipticity",
            "strike",
            "azimuth",
        ]

        w_ellipse_colorby = pn.widgets.Select(
            name="Ellipse color by",
            options=colorby_options,
            value=(
                self.ellipse_colorby
                if self.ellipse_colorby in colorby_options
                else "phimin"
            ),
            width=220,
        )
        w_ellipse_palette = pn.widgets.Select(
            name="Ellipse palette",
            options=palette_options,
            value=(
                self.ellipse_cmap if self.ellipse_cmap in palette_options else "turbo"
            ),
            width=170,
        )
        w_ellipse_range = pn.widgets.RangeSlider(
            name="Ellipse limits",
            start=-180.0,
            end=180.0,
            value=(float(self.ellipse_range[0]), float(self.ellipse_range[1])),
            step=1.0,
            width=240,
        )
        w_ellipse_size = pn.widgets.IntInput(
            name="Ellipse size",
            value=int(self.ellipse_size),
            step=1,
            start=1,
            end=1000,
            width=140,
        )
        w_station_step = pn.widgets.IntInput(
            name="Station label step",
            value=int(getattr(self, "station_step", 1)),
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
            step=100.0,
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
            self.plot_pt = bool(w_plot_pt.value)
            self.plot_tipper = str(w_plot_tipper.value)
            self.y_scale = str(w_y_scale.value)
            self.ellipse_colorby = str(w_ellipse_colorby.value)
            self.ellipse_cmap = str(w_ellipse_palette.value)
            self.ellipse_range = tuple(float(v) for v in w_ellipse_range.value)
            self.ellipse_size = int(w_ellipse_size.value)
            self.station_step = int(w_station_step.value)
            self.x_stretch = float(w_x_stretch.value)
            self.y_stretch = float(w_y_stretch.value)

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
                w_plot_pt,
                w_plot_tipper,
                w_y_scale,
                w_ellipse_colorby,
                width=320,
                margin=(0, 20, 0, 0),
            ),
            pn.Column(
                w_ellipse_palette,
                w_ellipse_range,
                w_ellipse_size,
                pn.Row(w_station_step, w_x_stretch, w_y_stretch),
                refresh_btn,
                width=420,
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
