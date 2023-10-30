# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:15:37 2014

@author: jrpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
from mtpy.core.mt_location import MTLocation

try:
    from osgeo import ogr, gdal, osr
except ImportError:
    raise ImportError(
        "Did not find GDAL, be sure it is installed correctly and "
        "all the paths are correct."
    )

ogr.UseExceptions()
# =============================================================================


# ==============================================================================
# create a raster from an array
# ==============================================================================


def array2raster(
    raster_fn,
    utm_lower_left_mt_location,
    cell_size,
    res_array,
    utm_epsg,
    rotation_angle=0.0,
):
    """
    converts an array into a raster file that can be read into a GIS program.

    utm_lower_left_mt_location should be a MTLocation object with a UTM
    projection and represents the lower left hand corner of the grid.
    :param raster_fn: DESCRIPTION
    :type raster_fn: TYPE
    :param utm_lower_left_mt_location: DESCRIPTION
    :type utm_lower_left_mt_location: TYPE
    :param cell_size: DESCRIPTION
    :type cell_size: TYPE
    :param res_array: DESCRIPTION
    :type res_array: TYPE
    :param utm_epsg: DESCRIPTION
    :type utm_epsg: TYPE
    :param rotation_angle: DESCRIPTION, defaults to 0.0
    :type rotation_angle: TYPE, optional
    :raises TypeError: DESCRIPTION
    :raises ValueError: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    """

    # convert rotation angle to radians
    r_theta = np.deg2rad(rotation_angle)

    res_array = np.flipud(res_array[::-1])

    ncols = res_array.shape[1]
    nrows = res_array.shape[0]

    if not isinstance(utm_lower_left_mt_location, MTLocation):
        raise TypeError(
            "utm_lower_left_mt_location must be type mtpy.MTLocation, not "
            f"{type(utm_lower_left_mt_location)}."
        )
    ll_origin = utm_lower_left_mt_location.copy()
    if ll_origin.utm_crs is None:
        raise ValueError("Must set UTM CRS")

    # set drive to make a geo tiff
    driver = gdal.GetDriverByName("GTiff")

    # make a raster with the shape of the array to be written
    if isinstance(raster_fn, Path):
        raster_fn = raster_fn.as_posix()
    out_raster = driver.Create(raster_fn, ncols, nrows, 1, gdal.GDT_Float32)

    out_raster.SetGeoTransform(
        (
            ll_origin.east,
            np.cos(r_theta) * cell_size,
            -np.sin(r_theta) * cell_size,
            ll_origin.north,
            np.sin(r_theta) * cell_size,
            np.cos(r_theta) * cell_size,
        )
    )

    # create a band for the raster data to be put in
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(res_array)

    # geo reference the raster
    utm_cs = osr.SpatialReference(wkt=ll_origin.utm_crs.to_wkt())

    out_raster.SetProjection(utm_cs.ExportToWkt())

    # be sure to flush the data
    outband.FlushCache()
