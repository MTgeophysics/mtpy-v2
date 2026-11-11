"""Bokeh implementation of phase tensor map plotting."""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import param
from bokeh.io import show as bokeh_show
from bokeh.layouts import Column
from bokeh.models import (
    Arrow,
    BasicTicker,
    ColorBar,
    Div,
    Ellipse,
    LinearColorMapper,
    NormalHead,
    Range1d,
    Wedge,
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

from mtpy.core import Tipper
from mtpy.core.transfer_function import PhaseTensor
from mtpy.imaging.bokeh_plots.bokeh_plot_base import _ELLIPSE_COLORBY_OPTIONS
from mtpy.imaging.bokeh_plots.bokeh_plot_base_maps import BokehPlotBaseMaps


_PALETTE_OPTIONS = ["turbo", "viridis", "magma", "inferno", "plasma", "cividis"]

_TILE_PROVIDERS = [
    "None",
    "CartoDB.Positron",
    "CartoDB.Voyager",
    "CartoDB.DarkMatter",
    "OpenStreetMap.Mapnik",
    "Esri.WorldImagery",
    "Esri.WorldStreetMap",
    "Esri.WorldTopoMap",
    "Esri.NatGeoWorldMap",
    "Esri.WorldShadedRelief",
    "Stadia.StamenTerrain",
    "Stadia.StamenToner",
    "Stadia.StamenWatercolor",
]


class PlotPhaseTensorMaps(BokehPlotBaseMaps):
    """Plots phase tensor ellipses/wedges in map view using Bokeh.

    Inherits shared Bokeh plotting and map interpolation helpers from
    :class:`BokehPlotBaseMaps`.

    Parameters
    ----------
    mt_data : MTData or dict-like
        Container of MT objects.

    Notes
    -----
    MTData inputs are prepared on a decade-aligned grid with four intervals
    per decade. The Panel period selector uses this grid; a programmatically
    requested period outside the grid is computed and cached separately.
    Dict-like inputs use the per-station interpolation path.
    """

    # ── map / coordinate ────────────────────────────────────────────────────
    plot_period = param.Number(
        default=1.0,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Plot period (s)",
    )
    map_scale = param.ObjectSelector(
        default="deg",
        objects=["deg", "m", "km"],
        doc="Coordinate units for the map",
    )
    map_epsg = param.Parameter(default=None, doc="Input CRS EPSG code (or None)")
    map_utm_zone = param.Parameter(default=None, doc="UTM zone string (or None)")
    reference_point = param.Parameter(default=(0, 0), doc="(lon, lat) reference offset")
    x_pad = param.Number(default=0.01, doc="Map x padding (data units)")
    y_pad = param.Number(default=0.01, doc="Map y padding (data units)")

    # ── basemap (tile) ───────────────────────────────────────────────────────
    bokeh_tile_provider = param.ObjectSelector(
        default="None",
        objects=_TILE_PROVIDERS,
        doc="Tile basemap provider key (xyzservices dot-notation)",
    )

    # ── phase-tensor display ─────────────────────────────────────────────────
    plot_pt = param.Boolean(default=True, doc="Plot phase-tensor ellipses/wedges")
    pt_type = param.ObjectSelector(
        default="ellipses",
        objects=["ellipses", "wedges"],
        doc="PT glyph type",
    )
    ellipse_cmap = param.ObjectSelector(
        default="turbo",
        objects=_PALETTE_OPTIONS,
        doc="Fill-color palette for PT ellipses",
    )
    ellipse_alpha = param.Number(
        default=0.85, bounds=(0.0, 1.0), doc="PT ellipse fill opacity"
    )

    # ── ellipse edge ─────────────────────────────────────────────────────────
    edge_colorby = param.ObjectSelector(
        default="skew",
        objects=_ELLIPSE_COLORBY_OPTIONS,
        doc="PT property used to color ellipse edges",
    )
    edge_range = param.Parameter(
        default=(-10.0, 10.0), doc="Edge color range (min, max)"
    )
    edge_lw = param.Number(default=1.5, bounds=(0, 10), doc="Ellipse edge line width")
    edge_cmap = param.String(default="turbo", doc="Ellipse edge color palette")

    # ── wedge-specific ───────────────────────────────────────────────────────
    wedge_width = param.Number(
        default=7.0, bounds=(0.1, 90.0), doc="Half-width of wedge arcs (degrees)"
    )
    skew_limits = param.Parameter(
        default=(-9.0, 9.0), doc="Skew color range (min, max)"
    )
    skew_step = param.Number(default=3.0, doc="Skew colorbar tick step")
    skew_cmap = param.String(default="turbo", doc="Wedge skew-border palette")
    skew_lw = param.Number(default=2.5, bounds=(0, 10), doc="Wedge skew border width")

    # ── arrows (tipper) ──────────────────────────────────────────────────────
    arrow_size = param.Number(default=0.005, bounds=(0, None), doc="Arrow length scale")
    arrow_head_length = param.Number(
        default=0.0025, bounds=(0, None), doc="Arrow head length (data units)"
    )
    arrow_head_width = param.Number(
        default=0.0035, bounds=(0, None), doc="Arrow head width (data units)"
    )
    arrow_threshold = param.Number(
        default=2.0, bounds=(0, None), doc="Max tipper magnitude to plot"
    )

    # ── station labels ────────────────────────────────────────────────────────
    plot_station = param.Boolean(default=False, doc="Annotate station positions")
    station_id = param.Parameter(
        default=(0, None), doc="Station name slice (start, stop)"
    )
    station_pad = param.Parameter(default=0.0005, doc="Vertical label offset")

    # ── pre-interpolation cache ───────────────────────────────────────────────
    def __init__(self, mt_data, **kwargs):
        super().__init__(**kwargs)

        self.mt_data = mt_data
        self._interpolated_mt_data_cache = None
        self._interpolated_mt_data_cache_period = None
        self._map_dataframe = None
        self._map_data_signature = None
        self._map_periods = None
        self._rotation_angle = 0.0
        self.fig = None
        self.layout = None
        self.plot_xarr = None
        self.plot_yarr = None

        # Apply map_scale-dependent size/label defaults after kwargs are set.
        self._update_scale_defaults()

        if self.show_plot:
            self.plot(show=True)

    # ── map-scale helpers ─────────────────────────────────────────────────────

    @param.depends("map_scale", watch=True)
    def _update_scale_defaults(self):
        """Set scale-dependent size and axis-label defaults."""
        scale = self.map_scale
        if scale == "deg":
            self.x_pad = 0.005
            self.y_pad = 0.005
            self.ellipse_size = 0.005
            self.arrow_size = 0.005
            self.arrow_head_length = 0.0025
            self.arrow_head_width = 0.0035
            self.arrow_lw = 0.00075
            self._x_label = "Longitude (deg)"
            self._y_label = "Latitude (deg)"
        elif scale == "m":
            self.x_pad = 1000
            self.y_pad = 1000
            self.ellipse_size = 500
            self.arrow_size = 500
            self.arrow_head_length = 250
            self.arrow_head_width = 350
            self.arrow_lw = 50
            self._x_label = "Easting (m)"
            self._y_label = "Northing (m)"
        elif scale == "km":
            self.x_pad = 1
            self.y_pad = 1
            self.ellipse_size = 0.5
            self.arrow_size = 0.5
            self.arrow_head_length = 0.25
            self.arrow_head_width = 0.35
            self.arrow_lw = 0.075
            self._x_label = "Easting (km)"
            self._y_label = "Northing (km)"

    @property
    def x_label(self):
        return getattr(self, "_x_label", "Longitude (deg)")

    @property
    def y_label(self):
        return getattr(self, "_y_label", "Latitude (deg)")

    # ── rotation property ─────────────────────────────────────────────────────

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
        self.invalidate_map_cache()

    def invalidate_map_cache(self):
        """Discard prepared map values after the source data has changed."""
        self._map_dataframe = None
        self._map_data_signature = None
        self._map_periods = None

    def _map_source_signature(self):
        """Identify changes to station transfer functions, periods and locations."""
        digest = hashlib.blake2b()
        self.mt_data.compute()
        for path in self.mt_data._iter_station_paths():
            ds = self.mt_data.get_station(path)
            digest.update(path.encode())
            for key in ("station", "survey", "latitude", "longitude", "datum_crs"):
                digest.update(repr(ds.attrs.get(key)).encode())
            for name in ("period", "input", "output", "transfer_function"):
                if name in ds:
                    values = np.asarray(ds[name].values)
                    digest.update(repr((name, values.shape, values.dtype.str)).encode())
                    if values.dtype.hasobject or values.dtype.kind in "US":
                        digest.update(repr(values.tolist()).encode())
                    else:
                        digest.update(np.ascontiguousarray(values).tobytes())
        return digest.digest()

    def _get_map_dataframe(self):
        """Prepare and cache the numerical map data for an MTData source."""
        signature = self._map_source_signature()
        if self._map_dataframe is None or signature != self._map_data_signature:
            periods = self.mt_data.get_phase_tensor_map_periods()
            frame = self.mt_data.to_phase_tensor_map_dataframe(periods)
            self._map_periods = periods
            self._map_dataframe = frame
            self._map_data_signature = signature
        return self._map_dataframe

    def _get_dataframe_stations(self):
        """Return lightweight station records for the selected plot period."""
        frame = self._get_map_dataframe()
        matches = frame["period"] == float(self.plot_period)
        if not matches.any():
            extra = self.mt_data.to_phase_tensor_map_dataframe(
                np.array([self.plot_period], dtype=float)
            )
            frame = pd.concat([frame, extra], ignore_index=True)
            self._map_dataframe = frame
            matches = frame["period"] == float(self.plot_period)

        stations = []
        for row in frame.loc[matches].itertuples(index=False):
            pt = None
            if np.isfinite([row.phimin, row.phimax, row.azimuth, row.beta]).all():
                pt = SimpleNamespace(
                    **{
                        name: np.array([getattr(row, name)])
                        for name in (
                            "phimin",
                            "phimax",
                            "azimuth",
                            "beta",
                            "skew",
                            "ellipticity",
                            "det",
                        )
                    }
                )
            tipper = None
            if np.isfinite(
                [row.mag_real, row.mag_imag, row.angle_real, row.angle_imag]
            ).all():
                tipper = SimpleNamespace(
                    **{
                        name: np.array([getattr(row, name)])
                        for name in ("mag_real", "mag_imag", "angle_real", "angle_imag")
                    }
                )
            stations.append(
                SimpleNamespace(
                    station=row.station,
                    latitude=row.latitude,
                    longitude=row.longitude,
                    datum_crs=row.datum_crs,
                    _map_pt=pt,
                    _map_tipper=tipper,
                )
            )
        return stations

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
        if hasattr(tf, "_map_pt"):
            return tf._map_pt if self.plot_pt else None, (
                tf._map_tipper if "y" in self.plot_tipper else None
            )
        pt_obj = None
        if tf.has_impedance() and self.plot_pt:
            try:
                z = self._get_interpolated_z(tf)
                pt_obj = PhaseTensor(z=z)
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
                if (t != 0).all():
                    new_t_obj = Tipper(tipper=t, frequency=[1.0 / self.plot_period])
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

    @staticmethod
    def _lonlat_to_webmercator(lon, lat):
        """Convert lon/lat degrees to Web Mercator (EPSG:3857) metres.

        Parameters
        ----------
        lon, lat : float
            Geographic coordinates in decimal degrees.

        Returns
        -------
        x, y : float
            Web Mercator easting/northing in metres.
        """
        R = 6378137.0  # WGS-84 equatorial radius in metres
        x = np.radians(lon) * R
        y = (
            np.log(np.tan(np.pi / 4.0 + np.radians(np.clip(lat, -85.0, 85.0)) / 2.0))
            * R
        )
        return x, y

    def _get_location(self, tf):
        if getattr(self, "_use_mercator", False):
            # Project lon/lat to Web Mercator (EPSG:3857).
            x, y = self._lonlat_to_webmercator(tf.longitude, tf.latitude)
            if tuple(self.reference_point) != (0, 0):
                rx, ry = self._lonlat_to_webmercator(
                    self.reference_point[0], self.reference_point[1]
                )
                x -= rx
                y -= ry
            return x, y
        if self.map_scale == "deg":
            plot_x = tf.longitude - self.reference_point[0]
            plot_y = tf.latitude - self.reference_point[1]
        elif self.map_scale in ["m", "km"]:
            from pyproj import CRS, Transformer

            if self.map_epsg is not None:
                target_crs = CRS.from_user_input(self.map_epsg)
            else:
                if self.map_utm_zone is None:
                    zone = min(60, max(1, int((tf.longitude + 180) // 6) + 1))
                else:
                    match = re.fullmatch(r"(\d{1,2})[A-Za-z]?", str(self.map_utm_zone))
                    if match is None or not 1 <= int(match.group(1)) <= 60:
                        raise ValueError(f"Invalid UTM zone: {self.map_utm_zone!r}")
                    zone = int(match.group(1))
                target_crs = CRS.from_epsg(
                    (32600 if tf.latitude >= 0 else 32700) + zone
                )
            transformer = Transformer.from_crs(
                getattr(tf, "datum_crs", None) or "EPSG:4326",
                target_crs,
                always_xy=True,
            )
            east, north = transformer.transform(tf.longitude, tf.latitude)
            plot_x = east - self.reference_point[0]
            plot_y = north - self.reference_point[1]
            if self.map_scale == "km":
                plot_x /= 1000.0
                plot_y /= 1000.0
        else:
            raise NameError("mapscale not recognized")
        return plot_x, plot_y

    def _get_tipper_patch(self, plot_x, plot_y, t_obj):
        has_tipper = False
        if t_obj is not None and "y" in self.plot_tipper:
            merc = getattr(self, "_merc_size_scale", 1.0)
            arrow_size = self.arrow_size * merc
            if "r" in self.plot_tipper and t_obj.mag_real[0] <= self.arrow_threshold:
                has_tipper = True
                txr = (
                    t_obj.mag_real[0]
                    * arrow_size
                    * np.sin(
                        np.deg2rad(t_obj.angle_real[0]) + self.arrow_direction * np.pi
                    )
                )
                tyr = (
                    t_obj.mag_real[0]
                    * arrow_size
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
                    * arrow_size
                    * np.sin(
                        np.deg2rad(t_obj.angle_imag[0]) + self.arrow_direction * np.pi
                    )
                )
                tyi = (
                    t_obj.mag_imag[0]
                    * arrow_size
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
                merc = getattr(self, "_merc_size_scale", 1.0)
                scaling = (self.ellipse_size * merc) / phimax
                eheight = phimin * scaling
                ewidth = phimax * scaling

                fill_color = self._scalar_to_color(
                    color_value,
                    self.ellipse_range[0],
                    self.ellipse_range[1],
                    self.ellipse_cmap,
                )

                # Edge colored by edge_colorby (default "skew")
                edge_array = self._get_color_array_by(pt_obj, self.edge_colorby)
                edge_value = float(np.nan_to_num(edge_array)[0])
                edge_color = self._scalar_to_color(
                    edge_value,
                    self.edge_range[0],
                    self.edge_range[1],
                    self.edge_cmap,
                )

                self.fig.ellipse(
                    x=[plot_x],
                    y=[plot_y],
                    width=[ewidth],
                    height=[eheight],
                    angle=[np.deg2rad(90 - eangle)],
                    fill_color=fill_color,
                    fill_alpha=self.ellipse_alpha,
                    line_color=edge_color,
                    line_width=max(self.edge_lw, 0.5),
                )

        has_tipper = self._get_tipper_patch(plot_x, plot_y, t_obj)
        if has_ellipse or has_tipper:
            return plot_x, plot_y
        return (0, 0)

    def _get_patch_wedges(self, tf):
        pt_obj, t_obj = self._get_pt(tf)
        plot_x, plot_y = self._get_location(tf)
        has_ellipse = False

        phase_lo, phase_hi = (self.phase_limits or (0, 90))[:2]

        if pt_obj is not None and self.plot_pt:
            phimin = float(np.nan_to_num(pt_obj.phimin)[0])
            phimax = float(np.nan_to_num(pt_obj.phimax)[0])
            eangle = float(np.nan_to_num(pt_obj.azimuth)[0])
            skew = float(np.nan_to_num(pt_obj.skew)[0])

            if phimax > 0:
                has_ellipse = True

                major_color = self._scalar_to_color(
                    phimax,
                    phase_lo,
                    phase_hi,
                    self.ellipse_cmap,
                )
                minor_color = self._scalar_to_color(
                    phimin,
                    phase_lo,
                    phase_hi,
                    self.ellipse_cmap,
                )
                gm = np.sqrt(abs(phimin) * abs(phimax))
                gm_color = self._scalar_to_color(
                    gm,
                    phase_lo,
                    phase_hi,
                    self.ellipse_cmap,
                )
                skew_color = self._scalar_to_color(
                    skew,
                    self.skew_limits[0],
                    self.skew_limits[1],
                    self.skew_cmap,
                )

                ratio = max(phimin / phimax, 1e-6)
                merc = getattr(self, "_merc_size_scale", 1.0)
                esize = self.ellipse_size * merc
                self.fig.ellipse(
                    x=[plot_x],
                    y=[plot_y],
                    width=[2 * esize],
                    height=[2 * esize * ratio],
                    angle=[np.deg2rad(90 - eangle)],
                    fill_color=gm_color,
                    fill_alpha=self.ellipse_alpha,
                    line_color=skew_color,
                    line_width=self.skew_lw,
                )

                for start_deg, end_deg, radius, color in [
                    (
                        0 - eangle - self.wedge_width,
                        0 - eangle + self.wedge_width,
                        2 * esize,
                        major_color,
                    ),
                    (
                        180 - eangle - self.wedge_width,
                        180 - eangle + self.wedge_width,
                        2 * esize,
                        major_color,
                    ),
                    (
                        90 - eangle - self.wedge_width,
                        90 - eangle + self.wedge_width,
                        2 * esize * ratio,
                        minor_color,
                    ),
                    (
                        270 - eangle - self.wedge_width,
                        270 - eangle + self.wedge_width,
                        2 * esize * ratio,
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
        phase_lo, phase_hi = (self.phase_limits or (0, 90))[:2]
        phase_mapper = LinearColorMapper(
            palette=self._palette_from_name(self.ellipse_cmap),
            low=phase_lo,
            high=phase_hi,
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

    def _get_color_array_by(self, pt_obj, colorby: str) -> np.ndarray:
        """Return the PT scalar array for a given colorby key."""
        if colorby in ("phiminang", "phimin"):
            return pt_obj.phimin
        if colorby in ("phimaxang", "phimax"):
            return pt_obj.phimax
        if colorby == "phidet":
            return np.sqrt(abs(pt_obj.det)) * (180 / np.pi)
        if colorby in ("skew", "skew_seg"):
            return pt_obj.beta
        if colorby == "ellipticity":
            return pt_obj.ellipticity
        if colorby in ("strike", "azimuth"):
            arr = pt_obj.azimuth % 180
            arr[np.where(arr > 90)] -= 180
            return arr
        return pt_obj.beta  # fallback to skew

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

        self._use_mercator = (
            bool(self.bokeh_tile_provider)
            and self.bokeh_tile_provider != "None"
            and self.map_scale == "deg"
        )

        # When projecting to Web Mercator, scale degree-sized glyphs to metres.
        # 1 degree latitude ≈ 111 320 m; use that as the base scale factor.
        # A more accurate factor accounts for the latitude of the dataset.
        self._merc_size_scale = 111320.0 if self._use_mercator else 1.0

        fig_kwargs = dict(
            title="",
            width=900,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
        )
        if self._use_mercator:
            fig_kwargs["x_axis_type"] = "mercator"
            fig_kwargs["y_axis_type"] = "mercator"

        self.fig = figure(**fig_kwargs)

        # Optionally add a tile basemap.
        if self.bokeh_tile_provider and self.bokeh_tile_provider != "None":
            try:
                self.fig.add_tile(self.bokeh_tile_provider)
            except Exception as exc:
                self.logger.warning(f"Could not add tile provider: {exc}")

        if (
            self.use_mt_data_preinterpolation
            and hasattr(self.mt_data, "to_phase_tensor_map_dataframe")
            and hasattr(self.mt_data, "get_phase_tensor_map_periods")
        ):
            mt_objects = self._get_dataframe_stations()
        else:
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
                    y=[plot_y + self.station_pad * self._merc_size_scale],
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
            start=float(non_zero_x.min() - self.x_pad * self._merc_size_scale),
            end=float(non_zero_x.max() + self.x_pad * self._merc_size_scale),
        )
        self.fig.y_range = Range1d(
            start=float(non_zero_y.min() - self.y_pad * self._merc_size_scale),
            end=float(non_zero_y.max() + self.y_pad * self._merc_size_scale),
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
            self._add_colorbar_edge()
        elif self.pt_type == "wedges" and self.plot_pt:
            self._add_colorbar_wedges()
        self._add_tipper_legend()

        self.layout = Column(
            Div(text="<b>Phase Tensor Map</b>"),
            self.fig,
        )

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def _add_colorbar_edge(self):
        """Add a left-side colorbar for ellipse edge color."""
        mapper = LinearColorMapper(
            palette=self._palette_from_name(self.edge_cmap),
            low=self.edge_range[0],
            high=self.edge_range[1],
        )
        self.fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                ticker=BasicTicker(),
                label_standoff=8,
                title=f"Edge: {self.cb_label_dict.get(self.edge_colorby, self.edge_colorby)}",
            ),
            "left",
        )

    # ── Default colorby ranges for panel auto-limits ──────────────────────────

    _COLORBY_DEFAULTS = {
        "phimin": (0, 90),
        "phiminang": (0, 90),
        "phimax": (0, 90),
        "phimaxang": (0, 90),
        "phidet": (0, 90),
        "skew": (-10, 10),
        "skew_seg": (-10, 10),
        "normalized_skew": (-0.5, 0.5),
        "normalized_skew_seg": (-0.5, 0.5),
        "ellipticity": (0, 1),
        "strike": (-90, 90),
        "azimuth": (-90, 90),
    }

    # ── Interactive Panel app ─────────────────────────────────────────────────

    def panel(self):
        """Return an embeddable Panel application with live plot controls.

        Returns
        -------
        pn.Column
            A Panel layout whose widgets update the Bokeh figure on change.
        """
        try:
            import panel as pn
        except ImportError:
            raise ImportError(
                "panel is required for PlotPhaseTensorMaps.panel(). "
                "Install with `pip install panel`."
            )

        # ── Period & map ──────────────────────────────────────────────────────
        if (
            self.use_mt_data_preinterpolation
            and hasattr(self.mt_data, "to_phase_tensor_map_dataframe")
            and hasattr(self.mt_data, "get_phase_tensor_map_periods")
        ):
            self._get_map_dataframe()
            nearest = int(
                np.argmin(np.abs(np.log(self._map_periods / self.plot_period)))
            )
            self.plot_period = float(self._map_periods[nearest])
            w_period = pn.widgets.Select(
                name="Plot period (s)",
                value=self.plot_period,
                options={
                    f"{period:.8g}": float(period) for period in self._map_periods
                },
                width=140,
            )
        else:
            w_period = pn.widgets.NumberInput(
                name="Plot period (s)",
                value=float(self.plot_period),
                step=0.1,
                width=140,
            )
        w_map_scale = pn.widgets.Select(
            name="Map scale",
            value=self.map_scale,
            options=["deg", "m", "km"],
            width=120,
        )
        w_tile = pn.widgets.Select(
            name="Basemap",
            value=self.bokeh_tile_provider,
            options=_TILE_PROVIDERS,
            width=220,
        )
        w_plot_station = pn.widgets.Checkbox(
            name="Station labels", value=self.plot_station
        )
        w_x_pad = pn.widgets.NumberInput(
            name="X pad", value=float(self.x_pad), step=0.001, width=120
        )
        w_y_pad = pn.widgets.NumberInput(
            name="Y pad", value=float(self.y_pad), step=0.001, width=120
        )

        # ── Phase tensor ──────────────────────────────────────────────────────
        w_pt_type = pn.widgets.Select(
            name="Glyph type",
            value=self.pt_type,
            options=["ellipses", "wedges"],
            width=120,
        )
        w_ellipse_size = pn.widgets.NumberInput(
            name="Ellipse size",
            value=float(self.ellipse_size),
            step=0.001,
            width=130,
        )

        def _on_ellipse_size_change(event):
            new_size = float(event.new)
            if float(w_x_pad.value) < new_size:
                w_x_pad.value = new_size
            if float(w_y_pad.value) < new_size:
                w_y_pad.value = new_size

        w_ellipse_alpha = pn.widgets.FloatSlider(
            name="Fill alpha", value=self.ellipse_alpha, start=0.0, end=1.0, step=0.05
        )
        w_colorby = pn.widgets.Select(
            name="Color fill by",
            value=self.ellipse_colorby,
            options=_ELLIPSE_COLORBY_OPTIONS,
            width=160,
        )
        _lo, _hi = self._COLORBY_DEFAULTS.get(
            self.ellipse_colorby, (self.ellipse_range[0], self.ellipse_range[1])
        )
        w_fill_min = pn.widgets.NumberInput(
            name="Fill min", value=float(_lo), step=1.0, width=90
        )
        w_fill_max = pn.widgets.NumberInput(
            name="Fill max", value=float(_hi), step=1.0, width=90
        )
        w_cmap = pn.widgets.Select(
            name="Color palette",
            value=self.ellipse_cmap,
            options=_PALETTE_OPTIONS,
            width=120,
        )

        def _on_colorby_change(event):
            lo, hi = self._COLORBY_DEFAULTS.get(event.new, (0, 90))
            w_fill_min.value = float(lo)
            w_fill_max.value = float(hi)

        # ── Ellipse edge ──────────────────────────────────────────────────────
        w_edge_colorby = pn.widgets.Select(
            name="Color edge by",
            value=self.edge_colorby,
            options=_ELLIPSE_COLORBY_OPTIONS,
            width=160,
        )
        _elo, _ehi = self._COLORBY_DEFAULTS.get(
            self.edge_colorby, tuple(self.edge_range[:2])
        )
        w_edge_min = pn.widgets.NumberInput(
            name="Edge min", value=float(_elo), step=1.0, width=90
        )
        w_edge_max = pn.widgets.NumberInput(
            name="Edge max", value=float(_ehi), step=1.0, width=90
        )
        w_edge_lw = pn.widgets.NumberInput(
            name="Edge line width", value=float(self.edge_lw), step=0.5, width=120
        )
        w_edge_cmap = pn.widgets.Select(
            name="Edge palette",
            value=self.edge_cmap,
            options=_PALETTE_OPTIONS,
            width=120,
        )

        def _on_edge_colorby_change(event):
            lo, hi = self._COLORBY_DEFAULTS.get(event.new, (-10, 10))
            w_edge_min.value = float(lo)
            w_edge_max.value = float(hi)

        # ── Tipper ────────────────────────────────────────────────────────────
        w_tipper = pn.widgets.CheckBoxGroup(
            name="Tipper",
            options=["Real", "Imaginary"],
            value=(
                (["Real"] if "r" in self.plot_tipper else [])
                + (["Imaginary"] if "i" in self.plot_tipper else [])
            ),
            inline=True,
        )
        w_arrow_size = pn.widgets.NumberInput(
            name="Arrow size", value=float(self.arrow_size), step=0.001, width=120
        )
        w_arrow_head_length = pn.widgets.NumberInput(
            name="Arrow head length",
            value=float(self.arrow_head_length),
            step=0.001,
            width=140,
        )
        w_arrow_head_width = pn.widgets.NumberInput(
            name="Arrow head width",
            value=float(self.arrow_head_width),
            step=0.001,
            width=140,
        )
        w_arrow_threshold = pn.widgets.NumberInput(
            name="Arrow threshold",
            value=float(self.arrow_threshold),
            step=0.1,
            width=130,
        )
        w_arrow_color_real = pn.widgets.ColorPicker(
            name="Arrow real color", value=str(self.arrow_color_real)
        )
        w_arrow_color_imag = pn.widgets.ColorPicker(
            name="Arrow imag color", value=str(self.arrow_color_imag)
        )
        w_arrow_dir = pn.widgets.Select(
            name="Arrow direction",
            value=(
                "Parkinson (toward)"
                if self.arrow_direction == 1
                else "Away from conductor"
            ),
            options=["Parkinson (toward)", "Away from conductor"],
            width=180,
        )

        status = pn.pane.Markdown("", styles={"color": "#555"})
        plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

        def _refresh(_event=None):
            # Read widget values back to self.
            self.map_scale = w_map_scale.value
            self.bokeh_tile_provider = w_tile.value
            self.plot_station = w_plot_station.value
            self.pt_type = w_pt_type.value
            self.ellipse_size = float(w_ellipse_size.value)
            self.ellipse_alpha = float(w_ellipse_alpha.value)
            self.ellipse_colorby = w_colorby.value
            self.ellipse_range = (float(w_fill_min.value), float(w_fill_max.value), 10)
            self.ellipse_cmap = w_cmap.value
            self.edge_colorby = w_edge_colorby.value
            self.edge_range = (float(w_edge_min.value), float(w_edge_max.value))
            self.edge_lw = float(w_edge_lw.value)
            self.edge_cmap = w_edge_cmap.value
            self.x_pad = float(w_x_pad.value)
            self.y_pad = float(w_y_pad.value)
            # Tipper mode
            tip_sel = w_tipper.value
            if "Real" in tip_sel and "Imaginary" in tip_sel:
                self.plot_tipper = "yri"
            elif "Real" in tip_sel:
                self.plot_tipper = "yr"
            elif "Imaginary" in tip_sel:
                self.plot_tipper = "yi"
            else:
                self.plot_tipper = "n"
            self.arrow_size = float(w_arrow_size.value)
            self.arrow_head_length = float(w_arrow_head_length.value)
            self.arrow_head_width = float(w_arrow_head_width.value)
            self.arrow_threshold = float(w_arrow_threshold.value)
            self.arrow_color_real = w_arrow_color_real.value
            self.arrow_color_imag = w_arrow_color_imag.value
            self.arrow_direction = 1 if "Parkinson" in w_arrow_dir.value else 0
            status.object = "\u23f3 Rendering…"
            try:
                if isinstance(w_period, pn.widgets.Select):
                    requested = float(w_period.value)
                    self._get_map_dataframe()
                    options = {
                        f"{period:.8g}": float(period) for period in self._map_periods
                    }
                    if w_period.options != options:
                        w_period.options = options
                        nearest = int(
                            np.argmin(np.abs(np.log(self._map_periods / requested)))
                        )
                        w_period.value = float(self._map_periods[nearest])
                self.plot_period = float(w_period.value)
                layout = self.plot(show=False)
                plot_pane.object = layout
                has_ellipses = any(
                    isinstance(getattr(renderer, "glyph", None), (Ellipse, Wedge))
                    for renderer in self.fig.renderers
                )
                has_arrows = bool(list(self.fig.select({"type": Arrow})))
                if has_ellipses or has_arrows:
                    status.object = f"\u2705 Period {self.plot_period:.5g} s rendered."
                    status.styles = {"color": "#1a6600"}
                else:
                    status.object = (
                        f"\u26a0\ufe0f No phase tensor ellipses or tipper arrows "
                        f"available at {self.plot_period:.5g} s."
                    )
                    status.styles = {"color": "#7a5200"}
            except Exception as exc:
                import traceback

                status.object = (
                    f"\u274c `{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}`"
                )
                status.styles = {"color": "#b00020"}

        updating = False

        def _update_plot(event=None):
            nonlocal updating
            if updating:
                return
            updating = True
            try:
                if event is not None:
                    if event.obj is w_ellipse_size:
                        _on_ellipse_size_change(event)
                    elif event.obj is w_colorby:
                        _on_colorby_change(event)
                    elif event.obj is w_edge_colorby:
                        _on_edge_colorby_change(event)
                _refresh()
            except Exception as exc:
                import traceback

                status.object = (
                    f"\u274c `{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}`"
                )
                status.styles = {"color": "#b00020"}
            finally:
                updating = False

        for widget in (
            w_period,
            w_map_scale,
            w_tile,
            w_plot_station,
            w_x_pad,
            w_y_pad,
            w_pt_type,
            w_ellipse_size,
            w_ellipse_alpha,
            w_colorby,
            w_fill_min,
            w_fill_max,
            w_cmap,
            w_edge_colorby,
            w_edge_min,
            w_edge_max,
            w_edge_lw,
            w_edge_cmap,
            w_tipper,
            w_arrow_size,
            w_arrow_head_length,
            w_arrow_head_width,
            w_arrow_threshold,
            w_arrow_color_real,
            w_arrow_color_imag,
            w_arrow_dir,
        ):
            widget.param.watch(_update_plot, "value")
        _update_plot()

        options = pn.Card(
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("**Map**"),
                    w_map_scale,
                    w_tile,
                    w_plot_station,
                    pn.Row(w_x_pad, w_y_pad),
                    width=260,
                    margin=(0, 16, 0, 0),
                ),
                pn.Column(
                    pn.pane.Markdown("**Phase Tensor**"),
                    w_pt_type,
                    w_ellipse_size,
                    w_ellipse_alpha,
                    w_cmap,
                    w_colorby,
                    pn.Row(w_fill_min, w_fill_max),
                    width=240,
                    margin=(0, 16, 0, 0),
                ),
                pn.Column(
                    pn.pane.Markdown("**Ellipse Edge**"),
                    w_edge_colorby,
                    pn.Row(w_edge_min, w_edge_max),
                    w_edge_lw,
                    w_edge_cmap,
                    width=240,
                    margin=(0, 16, 0, 0),
                ),
                pn.Column(
                    pn.pane.Markdown("**Tipper**"),
                    w_tipper,
                    w_arrow_size,
                    pn.Row(w_arrow_head_length, w_arrow_head_width),
                    w_arrow_threshold,
                    w_arrow_color_real,
                    w_arrow_color_imag,
                    w_arrow_dir,
                    width=300,
                ),
                align="start",
            ),
            title="Plot options",
            collapsed=True,
            sizing_mode="stretch_width",
        )
        controls = pn.Column(
            pn.pane.Markdown("### Phase Tensor Map"),
            pn.Row(w_period, align="start"),
            options,
            status,
        )

        return pn.Column(
            controls, pn.layout.Divider(), plot_pane, sizing_mode="stretch_width"
        )
