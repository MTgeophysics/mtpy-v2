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
import pandas as pd
import geopandas as gpd
from pyproj import CRS

from loguru import logger
from shapely.geometry import Point, Polygon, LineString, LinearRing

from mtpy.core import MTDataFrame

# =============================================================================


class ShapefilesCreator:
    """

    Create shape files from an MT geoDataFrame

    :param outdir: path2output dir, where the shp file will be written.
    :param output_crs: CRS of output files
    """

    def __init__(
        self,
        mt_dataframe,
        output_crs,
        save_dir=None,
    ):
        """

        :param mt_dataframe: DESCRIPTION
        :type mt_dataframe: TYPE
        :param output_crs: DESCRIPTION
        :type output_crs: TYPE
        :param save_dir: DESCRIPTION, defaults to None
        :type save_dir: TYPE, optional
        :param : DESCRIPTION
        :type : TYPE
        :return: DESCRIPTION
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
        self.in_utm = False

    @property
    def mt_dataframe(self):
        """MTDataFrame object"""

        return self._mt_dataframe

    @mt_dataframe.setter
    def mt_dataframe(self, df):
        """make sure input is a MTDataFrame or converted to one"""

        if isinstance(df, MTDataFrame):
            self._mt_dataframe = df
        else:
            self._mt_dataframe = MTDataFrame(df)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if value is not None:
            self._save_dir = Path(value)
            if not self._save_dir.exists():
                self._save_dir.mkdir()
        else:
            self._save_dir = Path().cwd()

    @property
    def output_crs(self):
        return self._output_crs

    @output_crs.setter
    def output_crs(self, value):
        if value is None:
            self._output_crs = None
        else:
            self._output_crs = CRS.from_user_input(value)

    def _export_shapefiles(self, gpdf, element_type, period):
        """
        Convenience function for saving shapefiles.

        Parameters
        ----------
        gpdf : geopandas.GeoDataFrame
            Dataframe containg shapefile data.
        element_type : str
            Name of the element type, e.g. 'Phase_Tensor'.
        epsg_code : int
            EPSG code for CRS of the shapefile.
        period : float
            The period of the data.
        export_fig : bool
            Whether or not to export the shapefile as an image.

        Returns
        -------
        str
            Path to the shapefile.
        """

        if self.output_crs is not None:
            gpdf.to_crs(crs=self.output_crs, inplace=True)

        filename = f"{element_type}_EPSG_{self.output_crs}_Period_{period}.shp"
        out_path = self.save_dir.joinpath(filename)

        gpdf.to_file(out_path, driver="ESRI Shapefile")
        self.logger.info(f"Saved shapefile to {out_path}")

        return out_path

    def _create_phase_tensor_shp(self, period, tol=None):
        """
        create phase tensor ellipses shape file correspond to a MT period
        :return: (geopdf_obj, path_to_shapefile)
        """

        # get period
        period_df = self.mt_dataframe.get_period(period, tol=tol)
        pt_df = period_df.phase_tensor

        self.logger.debug("phase tensor values =: %s", len(pt_df))

        if len(pt_df) < 1:
            self.logger.warning(
                f"No phase tensors for the period {period} for any MT station"
            )
            return None

        if self.in_utm:
            points = gpd.points_from_xy(pt_df.east, pt_df.north)
            crs = CRS().from_epsg(pt_df.utm_epsg)
        else:
            points = gpd.points_from_xy(pt_df.longitude, pt_df.latitude)
            crs = CRS().from_epsg(pt_df.datum_epsg)

        geopdf = gpd.GeoDataFrame(pt_df.dataframe, crs=crs, geometry=points)

        # points to trace out the polygon-ellipse
        theta = np.linspace(0, 2 * np.pi, self.ellipse_resolution)
        azimuth = -np.deg2rad(geopdf["pt_azimuth"])
        scaling = self.ellipse_size / geopdf["pt_phimax"]
        width = geopdf["pt_phimax"] * scaling
        height = geopdf["pt_phimin"] * scaling
        if self.in_utm:
            x0 = geopdf["east"]
            y0 = geopdf["north"]
        else:
            x0 = geopdf["longitude"]
            y0 = geopdf["latitude"]

        # Find invalid ellipses
        bad_min = np.where(
            np.logical_or(geopdf["phi_min"] == 0, geopdf["phi_min"] > 100)
        )[0]
        bad_max = np.where(
            np.logical_or(geopdf["phi_max"] == 0, geopdf["phi_max"] > 100)
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
            geopdf, crs=self.orig_crs, geometry=ellipse_list
        )

        shp_fn = self._export_shapefiles(geopdf, "Phase_Tensor", period)

        return shp_fn

    def create_tipper_real_shp(
        self, period, line_length=None, target_epsg_code=4283, export_fig=False
    ):
        """
        create real tipper lines shapefile from a csv file
        The shapefile consists of lines without arrow.
        User can use GIS software such as ArcGIS to display and add an arrow at each line's end
        line_length is how long will be the line, auto-calculatable
        """

        if line_length is None:  # auto-calculate the tipper arrow length
            line_length = self.stations_distances.get("Q1PERCENT")
            self.logger.info(
                "Automatically Selected Max Tipper Length  = %s", line_length
            )

        pt = self.get_phase_tensor_tippers(period)
        self.logger.debug("phase tensor values =: %s", pt)

        if len(pt) < 1:
            self.logger.warn(
                "No phase tensor for the period %s for any MT station", period
            )
            return None

        pdf = pd.DataFrame(pt)

        tip_mag_re_maxval = pdf["tip_mag_re"].max()

        if tip_mag_re_maxval > 0.00000001:
            line_length_normalized = line_length / tip_mag_re_maxval
        else:
            line_length_normalized = line_length

        self.logger.debug(pdf["period"])

        pdf["tip_re"] = pdf.apply(
            lambda x: LineString(
                [
                    (float(x.lon), float(x.lat)),
                    (
                        float(x.lon)
                        + line_length_normalized
                        * x.tip_mag_re
                        * np.cos(-np.deg2rad(x.tip_ang_re)),
                        float(x.lat)
                        + line_length_normalized
                        * x.tip_mag_re
                        * np.sin(-np.deg2rad(x.tip_ang_re)),
                    ),
                ]
            ),
            axis=1,
        )

        geopdf = gpd.GeoDataFrame(pdf, crs=self.orig_crs, geometry="tip_re")

        if target_epsg_code is None:
            self.logger.info("Geopandas Datframe CRS: %s", geopdf.crs)
            # {'init': 'epsg:4283', 'no_defs': True}
            # raise Exception("Must provide a target_epsg_code")
            target_epsg_code = geopdf.crs["init"][5:]
        else:
            geopdf.to_crs(epsg=target_epsg_code, inplace=True)
            # world = world.to_crs({'init': 'epsg:3395'})
            # world.to_crs(epsg=3395) would also work

        path2shp = self._export_shapefiles(
            geopdf, "Tipper_Real", target_epsg_code, period, export_fig
        )

        return (geopdf, path2shp)

    def create_tipper_imag_shp(
        self, period, line_length=None, target_epsg_code=4283, export_fig=False
    ):
        """
        create imagery tipper lines shapefile from a csv file
        The shapefile consists of lines without arrow.
        User can use GIS software such as ArcGIS to display and add an arrow at each line's end
        line_length is how long will be the line, auto-calculatable
        :return:(geopdf_obj, path_to_shapefile)
        """

        if line_length is None:  # auto-calculate the tipper arrow length
            line_length = self.stations_distances.get("Q1PERCENT")
            self.logger.info(
                "Automatically Selected Max-Tipper Length =: %s", line_length
            )

        pt = self.get_phase_tensor_tippers(period)
        self.logger.debug("phase tensor values =: %s", pt)

        if len(pt) < 1:
            self.logger.warn(
                "No phase tensor for the period %s for any MT station", period
            )
            return None

        pdf = pd.DataFrame(pt)

        tip_mag_im_maxval = pdf["tip_mag_im"].max()

        if tip_mag_im_maxval > 0.00000001:
            line_length_normalized = line_length / tip_mag_im_maxval
        else:
            line_length_normalized = line_length

        self.logger.debug(pdf["period"])

        pdf["tip_im"] = pdf.apply(
            lambda x: LineString(
                [
                    (float(x.lon), float(x.lat)),
                    (
                        float(x.lon)
                        + line_length_normalized
                        * x.tip_mag_im
                        * np.cos(-np.deg2rad(x.tip_ang_im)),
                        float(x.lat)
                        + line_length_normalized
                        * x.tip_mag_im
                        * np.sin(-np.deg2rad(x.tip_ang_im)),
                    ),
                ]
            ),
            axis=1,
        )

        geopdf = gpd.GeoDataFrame(pdf, crs=self.orig_crs, geometry="tip_im")

        if target_epsg_code is None:
            self.logger.info(
                "Keep the Default/Original Geopandas Dataframe CRS: %s",
                geopdf.crs,
            )
            # {'init': 'epsg:4283', 'no_defs': True}
            # raise Exception("Must provide a target_epsg_code")
            target_epsg_code = geopdf.crs["init"][5:]
        else:
            geopdf.to_crs(epsg=target_epsg_code, inplace=True)
            # world = world.to_crs({'init': 'epsg:3395'})
            # world.to_crs(epsg=3395) would also work

        path2shp = self._export_shapefiles(
            geopdf, "Tipper_Imag", target_epsg_code, period, export_fig
        )

        return (geopdf, path2shp)
