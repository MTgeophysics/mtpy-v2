#! /usr/bin/env python
"""
Description:
    Create shape files for Phase Tensor Ellipses, Tipper Real/Imag.
    export the phase tensor map and tippers into jpeg/png images

CreationDate:   2017-03-06
Developer:      fei.zhang@ga.gov.au

Revision History:
    LastUpdate:     10/11/2017   FZ fix bugs after the big merge
    LastUpdate:     20/11/2017   change from freq to period filenames, allow to specify a period
    LastUpdate:     30/10/2018   combine ellipses and tippers together, refactorings

    brenainn.moushall@ga.gov.au 27-03-2020 17:33:23 AEDT:
        Fix outfile/directory issue (see commit messages)

update to v2 jpeacock 2024-04-15
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from pyproj import CRS
from shapely.geometry import LinearRing, LineString, Polygon

from mtpy.core import MTDataFrame


# =============================================================================


class ShapefileCreator:
    """
    Create phase tensor and tipper shape files using geopandas and shapely tools.

    Attributes
    ----------
    mt_dataframe : MTDataFrame
        MTDataFrame object containing MT station data
    save_dir : Path
        Directory where shapefiles will be saved
    output_crs : CRS
        Output coordinate reference system
    ellipse_size : float
        Size scaling factor for phase tensor ellipses
    ellipse_resolution : int
        Number of points to use when drawing ellipses
    arrow_size : float
        Size scaling factor for tipper arrows
    utm : bool
        Whether to use UTM coordinates instead of lat/lon

    """

    def __init__(
        self,
        mt_dataframe: MTDataFrame,
        output_crs: str | int | CRS,
        save_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize ShapefileCreator instance.

        Parameters
        ----------
        mt_dataframe : MTDataFrame
            MTDataFrame object containing MT station data with phase tensor
            and tipper information
        output_crs : str | int | CRS
            Output coordinate reference system. Can be an EPSG code (int),
            EPSG string (e.g., 'EPSG:4326'), or pyproj CRS object
        save_dir : str | Path | None, optional
            Directory where shapefiles will be saved. If None, uses current
            working directory, by default None
        **kwargs : dict
            Additional keyword arguments to set as instance attributes
            (e.g., ellipse_size, arrow_size, utm)

        """
        self.logger = logger
        self.mt_dataframe = mt_dataframe
        self.save_dir = save_dir
        self.output_crs = output_crs

        # properties
        self.ellipse_size = 2
        self.ellipse_resolution = 180
        self.arrow_size = 2
        self.utm = False

        self._tipper_columns = [
            "index",
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "datum_epsg",
            "east",
            "north",
            "utm_epsg",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
            "t_mag_real",
            "t_mag_real_error",
            "t_mag_real_model_error",
            "t_mag_imag",
            "t_mag_imag_error",
            "t_mag_imag_model_error",
            "t_angle_real",
            "t_angle_real_error",
            "t_angle_real_model_error",
            "t_angle_imag",
            "t_angle_imag_error",
            "t_angle_imag_model_error",
        ]

        self._pt_columns = [
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "datum_epsg",
            "east",
            "north",
            "utm_epsg",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
            "pt_xx",
            "pt_xx_error",
            "pt_xx_model_error",
            "pt_xy",
            "pt_xy_error",
            "pt_xy_model_error",
            "pt_yx",
            "pt_yx_error",
            "pt_yx_model_error",
            "pt_yy",
            "pt_yy_error",
            "pt_yy_model_error",
            "pt_phimin",
            "pt_phimin_error",
            "pt_phimin_model_error",
            "pt_phimax",
            "pt_phimax_error",
            "pt_phimax_model_error",
            "pt_azimuth",
            "pt_azimuth_error",
            "pt_azimuth_model_error",
            "pt_skew",
            "pt_skew_error",
            "pt_skew_model_error",
            "pt_ellipticity",
            "pt_ellipticity_error",
            "pt_ellipticity_model_error",
            "pt_det",
            "pt_det_error",
            "pt_det_model_error",
            "geometry",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mt_dataframe(self):
        """MTDataFrame object."""

        return self._mt_dataframe

    @mt_dataframe.setter
    def mt_dataframe(self, df: MTDataFrame) -> None:
        """Make sure input is a MTDataFrame or converted to one."""

        if isinstance(df, MTDataFrame):
            self._mt_dataframe = df
        else:
            self._mt_dataframe = MTDataFrame(df)

    @property
    def save_dir(self):
        """Save dir."""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value: str | Path | None) -> None:
        """Save dir."""
        if value is not None:
            self._save_dir = Path(value)
            if not self._save_dir.exists():
                self._save_dir.mkdir()
        else:
            self._save_dir = Path().cwd()

    @property
    def output_crs(self):
        """Output crs."""
        return self._output_crs

    @output_crs.setter
    def output_crs(self, value: str | int | CRS | None) -> None:
        """Output crs."""
        if value is None:
            self._output_crs = None
        else:
            self._output_crs = CRS.from_user_input(value)

    @property
    def x_key(self):
        """X key."""
        if self.utm:
            return "east"
        else:
            return "longitude"

    @property
    def y_key(self):
        """Y key."""
        if self.utm:
            return "north"
        else:
            return "latitude"

    def estimate_ellipse_size(self, quantile: float = 0.015) -> float:
        """
        Estimate ellipse size from station distances.

        Parameters
        ----------
        quantile : float, optional
            Quantile of station distances to use for size estimation,
            by default 0.015

        Returns
        -------
        float
            Estimated ellipse size based on station spacing

        """

        return self.mt_dataframe.get_station_distances(utm=self.utm).quantile(quantile)

    def estimate_arrow_size(self, quantile: float = 0.03) -> float:
        """
        Estimate arrow size from station distances.

        Parameters
        ----------
        quantile : float, optional
            Quantile of station distances to use for size estimation,
            by default 0.03

        Returns
        -------
        float
            Estimated arrow size based on station spacing

        """

        return self.mt_dataframe.get_station_distances(utm=self.utm).quantile(quantile)

    def _export_shapefiles(
        self, gpdf: gpd.GeoDataFrame, element_type: str, period: float
    ) -> Path:
        """
        Save a GeoDataFrame as an ESRI shapefile.

        Parameters
        ----------
        gpdf : gpd.GeoDataFrame
            GeoDataFrame containing shapefile data
        element_type : str
            Name of the element type (e.g., 'Phase_Tensor', 'Tipper_Real')
        period : float
            Period of the data in seconds

        Returns
        -------
        Path
            Path to the saved shapefile

        """

        if self.output_crs is not None:
            gpdf.to_crs(crs=self.output_crs, inplace=True)

            filename = (
                f"{element_type}_EPSG_{self.output_crs.to_epsg()}_Period_{period}s.shp"
            )
        else:
            filename = f"{element_type}_Period_{period}s.shp"
        out_path = self.save_dir.joinpath(filename)

        gpdf.to_file(out_path, driver="ESRI Shapefile")
        self.logger.info(f"Saved shapefile to {out_path}")

        return out_path

    def _get_period_geodf(
        self, period: float, comp: str, tol: float | None = None
    ) -> tuple[CRS, gpd.GeoDataFrame] | None:
        """
        Get a GeoDataFrame for a specific period and component.

        Parameters
        ----------
        period : float
            Period value in seconds
        comp : str
            Component type: 'pt'/'phase_tensor' for phase tensor,
            't'/'tip'/'tipper' for tipper
        tol : float | None, optional
            Tolerance for period matching, by default None

        Returns
        -------
        tuple[CRS, gpd.GeoDataFrame] | None
            Tuple of (CRS object, GeoDataFrame) or None if no data found

        """

        # get period
        period_df = self.mt_dataframe.get_period(period, tol=tol)
        if comp in ["pt", "phase_tensor"]:
            pt_df = period_df.phase_tensor
        elif comp in ["t", "tip", "tipper"]:
            pt_df = period_df.tipper

        self.logger.debug("phase tensor values =: %s", len(pt_df))

        if len(pt_df) < 1:
            self.logger.warning(
                f"No phase tensors for the period {period} for any MT station"
            )
            return None

        if self.utm:
            points = gpd.points_from_xy(pt_df.east, pt_df.north)
            crs = CRS.from_epsg(pt_df.utm_epsg.iloc[0])
        else:
            points = gpd.points_from_xy(pt_df.longitude, pt_df.latitude)
            crs = CRS.from_epsg(pt_df.datum_epsg.iloc[0])

        geodf = gpd.GeoDataFrame(pt_df, crs=crs, geometry=points)
        geodf = geodf.fillna(0)

        return crs, geodf

    def _create_phase_tensor_shp(self, period: float, tol: float | None = None) -> Path:
        """
        Create phase tensor ellipses shapefile for a given MT period.

        Parameters
        ----------
        period : float
            Period value in seconds
        tol : float | None, optional
            Tolerance for period matching, by default None

        Returns
        -------
        Path
            Path to the created shapefile

        Notes
        -----
        Creates ellipses using the phase tensor parameters (phimin, phimax,
        azimuth) with configurable resolution and size scaling.

        """

        crs, geopdf = self._get_period_geodf(period, "pt", tol=tol)

        # points to trace out the polygon-ellipse
        theta = np.linspace(0, 2 * np.pi, self.ellipse_resolution)
        azimuth = -np.deg2rad(geopdf["pt_azimuth"])
        scaling = self.ellipse_size / geopdf["pt_phimax"]
        width = geopdf["pt_phimax"] * scaling
        height = geopdf["pt_phimin"] * scaling
        x0 = geopdf[self.x_key]
        y0 = geopdf[self.y_key]

        # Find invalid ellipses
        bad_min = np.where(
            np.logical_or(geopdf["pt_phimin"] == 0, geopdf["pt_phimin"] > 100)
        )[0]
        bad_max = np.where(
            np.logical_or(geopdf["pt_phimax"] == 0, geopdf["pt_phimax"] > 100)
        )[0]
        dot = 0.0000001 * self.ellipse_size
        height[bad_min] = dot
        height[bad_max] = dot
        width[bad_min] = dot
        width[bad_max] = dot
        width[np.where(np.isinf(width))] = dot
        height[np.where(np.isinf(height))] = dot

        # apply formula to generate ellipses
        ellipse_list = []
        for i in range(0, len(azimuth)):
            x = (
                x0[i]
                + height[i] * np.cos(theta) * np.cos(azimuth[i])
                - width[i] * np.sin(theta) * np.sin(azimuth[i])
            )
            y = (
                y0[i]
                + height[i] * np.cos(theta) * np.sin(azimuth[i])
                + width[i] * np.sin(theta) * np.cos(azimuth[i])
            )

            polyg = Polygon(LinearRing([xy for xy in zip(x, y)]))

            # print polyg  # an ellispe

            ellipse_list.append(polyg)

        geopdf = gpd.GeoDataFrame(
            geopdf[self._pt_columns], crs=crs, geometry=ellipse_list
        )

        shp_fn = self._export_shapefiles(geopdf, "Phase_Tensor", period)

        return shp_fn

    def _create_tipper_real_shp(
        self, period: float, tol: float | None = None
    ) -> Path | None:
        """
        Create real tipper lines shapefile for a given period.

        Parameters
        ----------
        period : float
            Period value in seconds
        tol : float | None, optional
            Tolerance for period matching, by default None

        Returns
        -------
        Path | None
            Path to the created shapefile, or None if no data available

        Notes
        -----
        The shapefile consists of lines without arrows. GIS software
        such as ArcGIS can be used to display and add arrows at line ends.
        Line length is determined by the arrow_size attribute and tipper
        magnitude.

        """

        crs, tdf = self._get_period_geodf(period, "tip", tol=tol)

        if len(tdf) < 1:
            self.logger.warning(
                f"No phase tensor for the period {period} for any MT station"
            )
            return None

        tdf["tip_re"] = tdf.apply(
            lambda x: LineString(
                [
                    (float(x[self.x_key]), float(x[self.y_key])),
                    (
                        float(x[self.x_key])
                        + self.arrow_size
                        * x["t_mag_real"]
                        * np.cos(-np.deg2rad(x["t_angle_real"])),
                        float(x[self.y_key])
                        + self.arrow_size
                        * x["t_mag_real"]
                        * np.sin(-np.deg2rad(x["t_angle_real"])),
                    ),
                ]
            ),
            axis=1,
        )

        del tdf["geometry"]
        geopdf = tdf[self._tipper_columns + ["tip_re"]].set_geometry("tip_re")
        geopdf = geopdf.set_crs(crs)
        # geopdf = gpd.GeoDataFrame(tdf, crs=crs, geometry=tdf["tip_re"])

        shp_fn = self._export_shapefiles(geopdf, "Tipper_Real", period)

        return shp_fn

    def _create_tipper_imag_shp(
        self, period: float, tol: float | None = None
    ) -> Path | None:
        """
        Create imaginary tipper lines shapefile for a given period.

        Parameters
        ----------
        period : float
            Period value in seconds
        tol : float | None, optional
            Tolerance for period matching, by default None

        Returns
        -------
        Path | None
            Path to the created shapefile, or None if no data available

        Notes
        -----
        The shapefile consists of lines without arrows. GIS software
        such as ArcGIS can be used to display and add arrows at line ends.
        Line length is determined by the arrow_size attribute and tipper
        magnitude.

        """

        crs, tdf = self._get_period_geodf(period, "tip", tol=tol)

        if len(tdf) < 1:
            self.logger.warning(
                f"No phase tensor for the period {period} for any MT station"
            )
            return None

        tdf["tip_im"] = tdf.apply(
            lambda x: LineString(
                [
                    (float(x[self.x_key]), float(x[self.y_key])),
                    (
                        float(x[self.x_key])
                        + self.arrow_size
                        * x["t_mag_imag"]
                        * np.cos(-np.deg2rad(x["t_angle_imag"])),
                        float(x[self.y_key])
                        + self.arrow_size
                        * x["t_mag_imag"]
                        * np.sin(-np.deg2rad(x["t_angle_imag"])),
                    ),
                ]
            ),
            axis=1,
        )
        del tdf["geometry"]
        geopdf = tdf[self._tipper_columns + ["tip_im"]].set_geometry("tip_im")
        geopdf = geopdf.set_crs(crs)
        # geopdf = gpd.GeoDataFrame(tdf, crs=crs, geometry=tdf["tip_im"])

        shp_fn = self._export_shapefiles(geopdf, "Tipper_Imag", period)

        return shp_fn

    def make_shp_files(
        self,
        pt: bool = True,
        tipper: bool = True,
        periods: list[float] | np.ndarray | None = None,
        period_tol: float | None = None,
    ) -> dict[str, list[Path]]:
        """
        Create shapefiles for phase tensors and/or tippers at specified periods.

        Parameters
        ----------
        pt : bool, optional
            Whether to create phase tensor shapefiles, by default True
        tipper : bool, optional
            Whether to create tipper shapefiles (real and imaginary),
            by default True
        periods : list[float] | np.ndarray | None, optional
            List of periods in seconds to create shapefiles for. If None,
            uses all periods in the MTDataFrame, by default None
        period_tol : float | None, optional
            Tolerance for period matching, by default None

        Returns
        -------
        dict[str, list[Path]]
            Dictionary with keys 'pt', 'tipper_real', 'tipper_imag' containing
            lists of paths to created shapefiles

        Notes
        -----
        If you want all stations on the same period map, you need to
        interpolate before converting to an MTDataFrame.

        Examples
        --------
        >>> md.interpolate(new_periods)
        >>> mt_df = MTDataFrame(md)
        >>> creator = ShapefileCreator(mt_df, output_crs='EPSG:4326')
        >>> shp_files = creator.make_shp_files()

        """

        if periods is None:
            periods = self.mt_dataframe.period

        shp_files = {"pt": [], "tipper_real": [], "tipper_imag": []}
        for period in periods:
            if pt:
                shp_files["pt"].append(self._create_phase_tensor_shp(period))
            if tipper:
                shp_files["tipper_real"].append(
                    self._create_tipper_real_shp(period, tol=period_tol)
                )
                shp_files["tipper_imag"].append(
                    self._create_tipper_imag_shp(period, tol=period_tol)
                )

        return shp_files
