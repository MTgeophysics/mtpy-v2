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
from pathlib import Path

import numpy as np
import geopandas as gpd
from pyproj import CRS

from loguru import logger
from shapely.geometry import Polygon, LineString, LinearRing

from mtpy.core import MTDataFrame

# =============================================================================


class ShapefileCreator:
    """Create phase tensor and tipper shape files using geopandas and shapely tools."""

    def __init__(
        self,
        mt_dataframe,
        output_crs,
        save_dir=None,
    ):
        """Init function.
        :param mt_dataframe: DESCRIPTION.
        :type mt_dataframe: TYPE
        :param output_crs: DESCRIPTION.
        :type output_crs: TYPE
        :param save_dir: DESCRIPTION, defaults to None.
        :type save_dir: TYPE, optional
        :param: DESCRIPTION.
        :type: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
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

    @property
    def mt_dataframe(self):
        """MTDataFrame object."""

        return self._mt_dataframe

    @mt_dataframe.setter
    def mt_dataframe(self, df):
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
    def save_dir(self, value):
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
    def output_crs(self, value):
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

    def estimate_ellipse_size(self, quantile=0.015):
        """Estimate ellipse size from station distances."""

        return self.mt_dataframe.get_station_distances(utm=self.utm).quantile(
            quantile
        )

    def estimate_arrow_size(self, quantile=0.03):
        """Arrow size from station distances."""

        return self.mt_dataframe.get_station_distances(utm=self.utm).quantile(
            quantile
        )

    def _export_shapefiles(self, gpdf, element_type, period):
        """Convenience function for saving shapefiles.
        :param gpdf: Dataframe containg shapefile data.
        :type gpdf: geopandas.GeoDataFrame
        :param element_type: Name of the element type, e.g. 'Phase_Tensor'.
        :type element_type: str
        :param epsg_code: EPSG code for CRS of the shapefile.
        :type epsg_code: int
        :param period: The period of the data.
        :type period: float
        :param export_fig: Whether or not to export the shapefile as an image.
        :type export_fig: bool
        :return: Path to the shapefile.
        :rtype: str
        """

        if self.output_crs is not None:
            gpdf.to_crs(crs=self.output_crs, inplace=True)

        filename = f"{element_type}_EPSG_{self.output_crs.to_epsg()}_Period_{period}s.shp"
        out_path = self.save_dir.joinpath(filename)

        gpdf.to_file(out_path, driver="ESRI Shapefile")
        self.logger.info(f"Saved shapefile to {out_path}")

        return out_path

    def _get_period_geodf(self, period, comp, tol=None):
        """Get a period geodf for given component.
        :param tol:
            Defaults to None.
        :param df: DESCRIPTION.
        :type df: TYPE
        :param period: DESCRIPTION.
        :type period: TYPE
        :param comp: DESCRIPTION.
        :type comp: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
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

        return crs, gpd.GeoDataFrame(pt_df, crs=crs, geometry=points)

    def _create_phase_tensor_shp(self, period, tol=None):
        """Create phase tensor ellipses shape file correspond to a MT period.
        :return: (geopdf_obj, path_to_shapefile).
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

    def _create_tipper_real_shp(self, period, tol=None):
        """Create real tipper lines shapefile from a csv file
        The shapefile consists of lines without arrow.
        User can use GIS software such as ArcGIS to display and add an arrow at each line's end
        line_length is how long will be the line, auto-calculatable
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

    def _create_tipper_imag_shp(self, period, tol=None):
        """Create imagery tipper lines shapefile from a csv file
        The shapefile consists of lines without arrow.
        User can use GIS software such as ArcGIS to display and add an arrow at each line's end
        line_length is how long will be the line, auto-calculatable
        :return :(geopdf_obj, path_to_shapefile):
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
        self, pt=True, tipper=True, periods=None, period_tol=None
    ):
        """If you want all stations on the same period map need to interpolate
        before converting to an MTDataFrame

        md.interpolate(new_periods)
        :param period_tol:
            Defaults to None.
        :param periods:
            Defaults to None.
        :param pt: DESCRIPTION, defaults to True.
        :type pt: TYPE, optional
        :param tipper: DESCRIPTION, defaults to True.
        :type tipper: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
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
