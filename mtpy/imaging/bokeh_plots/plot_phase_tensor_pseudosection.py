"""Bokeh implementation of phase tensor pseudosection plotting."""

from __future__ import annotations

import numpy as np

from mtpy.imaging.mtplot_tools import period_label_dict, PlotBaseProfile


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

    def __init__(self, mt_data, **kwargs):
        super().__init__(mt_data, **kwargs)

        self._rotation_angle = 0
        self.plot_pt = True
        self.aspect = kwargs.pop("aspect", getattr(self, "aspect", "equal"))

        self.x_stretch = 1
        self.y_stretch = 10000
        self.y_scale = "period"

        self.station_id = [0, None]
        self.profile_vector = None
        self.profile_angle = None
        self.profile_line = None
        self.profile_reverse = False

        self.ellipse_size = 2000
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
        self.fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title=self.cb_label_dict[self.ellipse_colorby],
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
            y_overrides[float(tk)] = period_label_dict.get(key, "")
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

        if self.x_limits is None:
            self.fig.x_range = Range1d(
                start=float(
                    np.floor(self.station_list["offset"].min()) - self.ellipse_size / 2
                ),
                end=float(
                    np.ceil(self.station_list["offset"].max()) + self.ellipse_size / 2
                ),
            )
        else:
            self.fig.x_range = Range1d(
                start=float(self.x_limits[0]), end=float(self.x_limits[1])
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
