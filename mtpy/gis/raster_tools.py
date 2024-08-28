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
import rasterio
from rasterio.transform import Affine

from mtpy.core.mt_location import MTLocation

# =============================================================================


def array2raster(
    raster_fn,
    array,
    lower_left,
    cell_size_north,
    cell_size_east,
    crs,
    rotation_angle=0,
):
    """Use rasterio to write raster file.
    :param rotation_angle:
        Defaults to 0.
    :param array:
    :param raster_fn: DESCRIPTION.
    :type raster_fn: TYPE
    :param lower_left: DESCRIPTION.
    :type lower_left: TYPE
    :param cell_size_north: DESCRIPTION.
    :type cell_size_north: TYPE
    :param cell_size_east: DESCRIPTION.
    :type cell_size_east: TYPE
    :param crs: DESCRIPTION.
    :type crs: TYPE
    :return: DESCRIPTION.
    :rtype: TYPE
    """

    if not isinstance(lower_left, MTLocation):
        raise TypeError(
            f"lower_left must be a MTLocation object not {type(lower_left)}."
        )

    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy array not {type(array)}.")

    transform = (
        Affine.translation(
            lower_left.east,
            lower_left.north,
        )
        * Affine.scale(cell_size_east, cell_size_north)
        * Affine.rotation(rotation_angle)
    )

    with rasterio.open(
        raster_fn,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(array, 1)


# def dem_to_ply(geotiff_file, save_path=None):
