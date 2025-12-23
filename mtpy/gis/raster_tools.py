# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:15:37 2014

@author: jrpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import Affine

from mtpy.core.mt_location import MTLocation


# =============================================================================


def array2raster(
    raster_fn: str,
    array: np.ndarray,
    lower_left: MTLocation,
    cell_size_north: float,
    cell_size_east: float,
    crs: str | int,
    rotation_angle: float = 0,
) -> None:
    """
    Write a numpy array to a GeoTIFF raster file using rasterio.

    Parameters
    ----------
    raster_fn : str
        Output raster filename (GeoTIFF format)
    array : np.ndarray
        2D numpy array containing the raster data to write
    lower_left : MTLocation
        MTLocation object specifying the lower-left corner coordinates
        of the raster in the target coordinate system
    cell_size_north : float
        Cell size in the north (vertical) direction
    cell_size_east : float
        Cell size in the east (horizontal) direction
    crs : str | int
        Coordinate reference system (CRS) specification.
        Can be an EPSG code (e.g., 'EPSG:4326') or other CRS string
    rotation_angle : float, optional
        Rotation angle in degrees for the raster grid, by default 0

    Raises
    ------
    TypeError
        If lower_left is not an MTLocation object
    TypeError
        If array is not a numpy ndarray

    Notes
    -----
    The function uses an affine transformation to properly georeference
    the output raster. The transformation combines translation, scaling,
    and rotation operations.

    Examples
    --------
    >>> import numpy as np
    >>> from mtpy.core.mt_location import MTLocation
    >>> data = np.random.rand(100, 100)
    >>> ll_corner = MTLocation(latitude=40.0, longitude=-120.0, datum='WGS84')
    >>> array2raster(
    ...     'output.tif',
    ...     data,
    ...     ll_corner,
    ...     cell_size_north=0.01,
    ...     cell_size_east=0.01,
    ...     crs='EPSG:4326'
    ... )

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
