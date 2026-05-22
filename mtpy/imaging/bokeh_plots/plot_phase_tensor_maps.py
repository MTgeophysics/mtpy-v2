"""Bokeh implementation of phase tensor map plotting."""

from __future__ import annotations

import numpy as np

from mtpy.core import Tipper
from mtpy.core.transfer_function import PhaseTensor
from mtpy.imaging.mtplot_tools import PlotBaseMaps


try:
    from bokeh.io import show as bokeh_show
    from bokeh.layouts import Column
    from bokeh.models import (
        Arrow,
        BasicTicker,
        ColorBar,
        Div,
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
    Div = None
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


class PlotPhaseTensorMaps(PlotBaseMaps):
    """Plots phase tensor ellipses/wedges in map view using Bokeh."""

    def __init__(self, mt_data, **kwargs):
        super().__init__(**kwargs)

        self._rotation_angle = 0
        self.mt_data = mt_data
        self.use_mt_data_preinterpolation = True
        self._interpolated_mt_data_cache = None
        self._interpolated_mt_data_cache_period = None

        self.plot_station = False
        self.plot_period = 1.0
        self.plot_pt = True

        self.pt_type = "wedges"
        self.skew_cmap = "mt_seg_bl2wh2rd"
        self.phase_limits = (0, 90)
        self.skew_limits = (-9, 9)
        self.skew_step = 3
        self.skew_lw = 2.5
        self.ellipse_alpha = 0.85
        self.wedge_width = 7.0

        self.map_scale = "deg"
        self.map_utm_zone = None
        self.map_epsg = None

        self.minorticks_on = True
        self.x_pad = 0.01
        self.y_pad = 0.01

        self.arrow_legend_position = "lower right"
        self.arrow_legend_xborderpad = 0.2
        self.arrow_legend_yborderpad = 0.2
        self.arrow_legend_fontpad = 0.05

        self.reference_point = (0, 0)
        self.station_id = (0, 2)
        self.station_pad = 0.0005

        self.arrow_legend_fontdict = {"size": self.font_size, "weight": "bold"}
        self.station_font_dict = {"size": self.font_size, "weight": "bold"}

        self.fig = None
        self.layout = None
        self.plot_xarr = None
        self.plot_yarr = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot(show=True)

    @property
    def skew_cmap_bounds(self):
        return np.arange(
            self.skew_limits[0],
            self.skew_limits[1] + self.skew_step,
            self.skew_step,
        )

    @property
    def map_scale(self):
        return self._map_scale

    @map_scale.setter
    def map_scale(self, map_scale):
        self._map_scale = map_scale

        if self._map_scale == "deg":
            self.xpad = 0.005
            self.ypad = 0.005
            self.ellipse_size = 0.005
            self.arrow_size = 0.005
            self.arrow_head_length = 0.0025
            self.arrow_head_width = 0.0035
            self.arrow_lw = 0.00075
            self.tickstrfmt = "%.3f"
            self.y_label = "Latitude (deg)"
            self.x_label = "Longitude (deg)"
        elif self._map_scale == "m":
            self.xpad = 1000
            self.ypad = 1000
            self.ellipse_size = 500
            self.arrow_size = 500
            self.arrow_head_length = 250
            self.arrow_head_width = 350
            self.arrow_lw = 50
            self.tickstrfmt = "%.0f"
            self.x_label = "Easting (m)"
            self.y_label = "Northing (m)"
        elif self._map_scale == "km":
            self.xpad = 1
            self.ypad = 1
            self.ellipse_size = 0.500
            self.arrow_size = 0.5
            self.arrow_head_length = 0.25
            self.arrow_head_width = 0.35
            self.arrow_lw = 0.075
            self.tickstrfmt = "%.0f"
            self.x_label = "Easting (km)"
            self.y_label = "Northing (km)"
        else:
            raise ValueError(f"map scale {map_scale} is not supported.")

    @property
    def rotation_angle(self):
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        if hasattr(self.mt_data, "rotate") and hasattr(self.mt_data, "get_station"):
            self.mt_data.rotate(value, inplace=True)
        else:
            for tf in self._iter_mt_objects():
                tf.rotation_angle = value
        self._rotation_angle = value

    def _require_bokeh(self):
        if (
            figure is None
            or Column is None
            or Div is None
            or LinearColorMapper is None
            or ColorBar is None
            or Arrow is None
            or NormalHead is None
            or Range1d is None
        ):
            raise ImportError(
                "Bokeh is required for PlotPhaseTensorMaps. Install with `pip install bokeh`."
            )

    def _iter_mt_objects(self):
        if hasattr(self.mt_data, "values"):
            yield from self.mt_data.values()
            return

        if hasattr(self.mt_data, "_iter_station_paths") and hasattr(
            self.mt_data, "get_station"
        ):
            if hasattr(self.mt_data, "compute"):
                self.mt_data.compute()
            for station_path in self.mt_data._iter_station_paths():
                yield self.mt_data.get_station(station_path, as_mt=True)
            return

        raise TypeError("mt_data must provide values() or MTData-style station access")

    def _get_mt_data_for_plot_period(self):
        if not self.use_mt_data_preinterpolation:
            return self.mt_data

        if not (
            hasattr(self.mt_data, "interpolate")
            and hasattr(self.mt_data, "_iter_station_paths")
            and hasattr(self.mt_data, "get_station")
        ):
            return self.mt_data

        target_period = float(self.plot_period)
        if (
            self._interpolated_mt_data_cache is not None
            and self._interpolated_mt_data_cache_period is not None
            and np.isclose(self._interpolated_mt_data_cache_period, target_period)
        ):
            return self._interpolated_mt_data_cache

        try:
            interpolated = self.mt_data.interpolate(
                np.array([target_period], dtype=float),
                inplace=False,
                bounds_error=False,
            )
            if interpolated is not None:
                self._interpolated_mt_data_cache = interpolated
                self._interpolated_mt_data_cache_period = target_period
                return interpolated
        except Exception as error:
            self.logger.debug(
                "Falling back to per-station interpolation for plot period "
                f"{target_period}: {error}"
            )

        return self.mt_data

    def _get_mt_objects(self):
        data_source = self._get_mt_data_for_plot_period()

        if hasattr(data_source, "values"):
            return list(data_source.values())

        if hasattr(data_source, "_iter_station_paths") and hasattr(
            data_source, "get_station"
        ):
            if hasattr(data_source, "compute"):
                data_source.compute()
            return [
                data_source.get_station(station_path, as_mt=True)
                for station_path in data_source._iter_station_paths()
            ]

        return list(self._iter_mt_objects())

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

    def _scalar_to_color(self, value, value_min, value_max, cmap_name):
        palette = self._palette_from_name(cmap_name)
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

    def _get_pt(self, tf):
        pt_obj = None
        if tf.has_impedance() and self.plot_pt:
            try:
                z = self._get_interpolated_z(tf)
                z_error = self._get_interpolated_z_error(tf)
                pt_obj = PhaseTensor(z=z, z_error=z_error)
            except ValueError as error:
                self.logger.warning(
                    f"Could not estimate phase tensor for {tf.station} at period "
                    f"{self.plot_period} s."
                )
                self.logger.error(error)
                pt_obj = None

        new_t_obj = None
        if tf.has_tipper() and "y" in self.plot_tipper:
            try:
                t = self._get_interpolated_t(tf)
                t_err = self._get_interpolated_t_err(tf)
                if (t != 0).all():
                    new_t_obj = Tipper(t, t_err, [1.0 / self.plot_period])
            except ValueError:
                self.logger.warning(
                    f"Could not estimate tipper for {tf.station} at period "
                    f"{self.plot_period} s."
                )
        return pt_obj, new_t_obj

    def _get_tick_format(self):
        if self.map_scale == "deg":
            self.tickstrfmt = "%.2f"
        elif self.map_scale in ["m", "km"]:
            self.tickstrfmt = "%.0f"

    def _set_axis_labels(self):
        self.fig.xaxis.axis_label = self.x_label
        self.fig.yaxis.axis_label = self.y_label

    def _get_location(self, tf):
        if self.map_scale == "deg":
            plot_x = tf.longitude - self.reference_point[0]
            plot_y = tf.latitude - self.reference_point[1]
        elif self.map_scale in ["m", "km"]:
            tf.project_point_ll2utm(epsg=self.map_epsg, utm_zone=self.map_utm_zone)
            plot_x = tf.east - self.reference_point[0]
            plot_y = tf.north - self.reference_point[1]
            if self.map_scale == "km":
                plot_x /= 1000.0
                plot_y /= 1000.0
        else:
            raise NameError("mapscale not recognized")
        return plot_x, plot_y

    def _get_tipper_patch(self, plot_x, plot_y, t_obj):
        has_tipper = False
        if t_obj is not None and "y" in self.plot_tipper:
            if "r" in self.plot_tipper and t_obj.mag_real[0] <= self.arrow_threshold:
                has_tipper = True
                txr = (
                    t_obj.mag_real[0]
                    * self.arrow_size
                    * np.sin(
                        np.deg2rad(t_obj.angle_real[0]) + self.arrow_direction * np.pi
                    )
                )
                tyr = (
                    t_obj.mag_real[0]
                    * self.arrow_size
                    * np.cos(
                        np.deg2rad(t_obj.angle_real[0]) + self.arrow_direction * np.pi
                    )
                )
                self.fig.add_layout(
                    Arrow(
                        end=NormalHead(
                            size=max(int(self.arrow_head_width * 600), 6),
                            fill_color=self.arrow_color_real,
                            line_color=self.arrow_color_real,
                        ),
                        x_start=float(plot_x),
                        y_start=float(plot_y),
                        x_end=float(plot_x + txr),
                        y_end=float(plot_y + tyr),
                        line_color=self.arrow_color_real,
                        line_width=max(self.arrow_lw * 800, 1),
                    )
                )

            if "i" in self.plot_tipper and t_obj.mag_imag[0] <= self.arrow_threshold:
                has_tipper = True
                txi = (
                    t_obj.mag_imag[0]
                    * self.arrow_size
                    * np.sin(
                        np.deg2rad(t_obj.angle_imag[0]) + self.arrow_direction * np.pi
                    )
                )
                tyi = (
                    t_obj.mag_imag[0]
                    * self.arrow_size
                    * np.cos(
                        np.deg2rad(t_obj.angle_imag[0]) + self.arrow_direction * np.pi
                    )
                )
                self.fig.add_layout(
                    Arrow(
                        end=NormalHead(
                            size=max(int(self.arrow_head_width * 600), 6),
                            fill_color=self.arrow_color_imag,
                            line_color=self.arrow_color_imag,
                        ),
                        x_start=float(plot_x),
                        y_start=float(plot_y),
                        x_end=float(plot_x + txi),
                        y_end=float(plot_y + tyi),
                        line_color=self.arrow_color_imag,
                        line_width=max(self.arrow_lw * 800, 1),
                        line_dash="dashed",
                    )
                )
        return has_tipper

    def _get_patch_ellipse(self, tf):
        pt_obj, t_obj = self._get_pt(tf)

        if pt_obj is None and t_obj is None:
            return (0, 0)

        plot_x, plot_y = self._get_location(tf)
        has_ellipse = False

        if pt_obj is not None and self.plot_pt:
            phimin = float(np.nan_to_num(pt_obj.phimin)[0])
            phimax = float(np.nan_to_num(pt_obj.phimax)[0])
            eangle = float(np.nan_to_num(pt_obj.azimuth)[0])

            color_array = self.get_pt_color_array(pt_obj)
            color_value = float(np.nan_to_num(color_array)[0])
            has_ellipse = True

            if phimax == 0 or phimax > 100 or phimin == 0 or phimin > 100:
                has_ellipse = False
            else:
                scaling = self.ellipse_size / phimax
                eheight = phimin * scaling
                ewidth = phimax * scaling

                fill_color = self._scalar_to_color(
                    color_value,
                    self.ellipse_range[0],
                    self.ellipse_range[1],
                    self.ellipse_cmap,
                )
                self.fig.ellipse(
                    x=[plot_x],
                    y=[plot_y],
                    width=[ewidth],
                    height=[eheight],
                    angle=[np.deg2rad(90 - eangle)],
                    fill_color=fill_color,
                    fill_alpha=self.ellipse_alpha,
                    line_color="#222222",
                    line_width=max(self.lw, 1),
                )

        has_tipper = self._get_tipper_patch(plot_x, plot_y, t_obj)
        if has_ellipse or has_tipper:
            return plot_x, plot_y
        return (0, 0)

    def _get_patch_wedges(self, tf):
        pt_obj, t_obj = self._get_pt(tf)
        plot_x, plot_y = self._get_location(tf)
        has_ellipse = False

        if pt_obj is not None and self.plot_pt:
            phimin = float(np.nan_to_num(pt_obj.phimin)[0])
            phimax = float(np.nan_to_num(pt_obj.phimax)[0])
            eangle = float(np.nan_to_num(pt_obj.azimuth)[0])
            skew = float(np.nan_to_num(pt_obj.skew)[0])

            if phimax > 0:
                has_ellipse = True

                major_color = self._scalar_to_color(
                    phimax,
                    self.phase_limits[0],
                    self.phase_limits[1],
                    self.ellipse_cmap,
                )
                minor_color = self._scalar_to_color(
                    phimin,
                    self.phase_limits[0],
                    self.phase_limits[1],
                    self.ellipse_cmap,
                )
                gm = np.sqrt(abs(phimin) * abs(phimax))
                gm_color = self._scalar_to_color(
                    gm,
                    self.phase_limits[0],
                    self.phase_limits[1],
                    self.ellipse_cmap,
                )
                skew_color = self._scalar_to_color(
                    skew,
                    self.skew_limits[0],
                    self.skew_limits[1],
                    self.skew_cmap,
                )

                ratio = max(phimin / phimax, 1e-6)
                self.fig.ellipse(
                    x=[plot_x],
                    y=[plot_y],
                    width=[2 * self.ellipse_size],
                    height=[2 * self.ellipse_size * ratio],
                    angle=[np.deg2rad(90 - eangle)],
                    fill_color=gm_color,
                    fill_alpha=self.ellipse_alpha,
                    line_color=skew_color,
                    line_width=self.skew_lw,
                )

                for start_deg, end_deg, radius, color in [
                    (
                        90 - eangle - self.wedge_width,
                        90 - eangle + self.wedge_width,
                        self.ellipse_size,
                        major_color,
                    ),
                    (
                        270 - eangle - self.wedge_width,
                        270 - eangle + self.wedge_width,
                        self.ellipse_size,
                        major_color,
                    ),
                    (
                        -1 * eangle - self.wedge_width,
                        -1 * eangle + self.wedge_width,
                        self.ellipse_size * ratio,
                        minor_color,
                    ),
                    (
                        180 - eangle - self.wedge_width,
                        180 - eangle + self.wedge_width,
                        self.ellipse_size * ratio,
                        minor_color,
                    ),
                ]:
                    self.fig.wedge(
                        x=[plot_x],
                        y=[plot_y],
                        radius=radius,
                        start_angle=np.deg2rad(start_deg),
                        end_angle=np.deg2rad(end_deg),
                        fill_color=color,
                        line_color=None,
                        fill_alpha=0.95,
                    )

        has_tipper = self._get_tipper_patch(plot_x, plot_y, t_obj)
        if has_ellipse or has_tipper:
            return plot_x, plot_y
        return (0, 0)

    def _add_colorbar_ellipse(self):
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

    def _add_colorbar_wedges(self):
        phase_mapper = LinearColorMapper(
            palette=self._palette_from_name(self.ellipse_cmap),
            low=self.phase_limits[0],
            high=self.phase_limits[1],
        )
        self.fig.add_layout(
            ColorBar(
                color_mapper=phase_mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title="Phase (deg)",
            ),
            "right",
        )

        skew_mapper = LinearColorMapper(
            palette=self._palette_from_name(self.skew_cmap),
            low=self.skew_limits[0],
            high=self.skew_limits[1],
        )
        self.fig.add_layout(
            ColorBar(
                color_mapper=skew_mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title="Skew (deg)",
            ),
            "left",
        )

    def _add_tipper_legend(self):
        if "y" not in self.plot_tipper:
            return

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

    def plot(
        self,
        fig=None,
        save_path=None,
        show=True,
        raster_file=None,
        raster_kwargs={},
    ):
        """Plot phase tensor map as a Bokeh layout."""

        del fig
        del save_path
        del raster_file
        del raster_kwargs

        self._require_bokeh()
        self._get_tick_format()

        self.fig = figure(
            title="",
            width=900,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
        )

        mt_objects = self._get_mt_objects()
        self.plot_xarr = np.zeros(len(mt_objects))
        self.plot_yarr = np.zeros(len(mt_objects))

        for index, tf in enumerate(mt_objects):
            if self.pt_type == "ellipses":
                plot_x, plot_y = self._get_patch_ellipse(tf)
            elif self.pt_type == "wedges":
                plot_x, plot_y = self._get_patch_wedges(tf)
            else:
                raise ValueError(
                    f"{self.pt_type} not supported. Use ['ellipses' | 'wedges']"
                )

            self.plot_xarr[index] = plot_x
            self.plot_yarr[index] = plot_y

            if self.plot_station and (plot_x != 0 or plot_y != 0):
                self.fig.text(
                    x=[plot_x],
                    y=[plot_y + self.station_pad],
                    text=[tf.station[self.station_id[0] : self.station_id[1]]],
                    text_align="center",
                    text_baseline="bottom",
                    text_font_size=f"{self.font_size}px",
                )

        self._set_axis_labels()

        non_zero_x = self.plot_xarr[np.nonzero(self.plot_xarr)]
        non_zero_y = self.plot_yarr[np.nonzero(self.plot_yarr)]

        if non_zero_x.size == 0 or non_zero_y.size == 0:
            non_zero_x = np.array([0.0])
            non_zero_y = np.array([0.0])

        self.fig.x_range = Range1d(
            start=float(non_zero_x.min() - self.x_pad),
            end=float(non_zero_x.max() + self.x_pad),
        )
        self.fig.y_range = Range1d(
            start=float(non_zero_y.min() - self.y_pad),
            end=float(non_zero_y.max() + self.y_pad),
        )

        titlefreq = f"{self.plot_period:.5g} (s)"
        if not self.plot_title:
            self.fig.title.text = f"Phase Tensor Map for {titlefreq}"
        else:
            self.fig.title.text = f"{self.plot_title}{titlefreq}"

        self.fig.grid.grid_line_alpha = 0.3
        self.fig.grid.grid_line_width = 0.75

        if self.pt_type == "ellipses" and self.plot_pt:
            self._add_colorbar_ellipse()
        elif self.pt_type == "wedges" and self.plot_pt:
            self._add_colorbar_wedges()
        self._add_tipper_legend()

        self.layout = Column(
            Div(text=f"<b>Phase Tensor Map</b>"),
            self.fig,
        )

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout
