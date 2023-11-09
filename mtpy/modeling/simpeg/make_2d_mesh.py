# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:43:06 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np

from discretize import TensorMesh

# from dask.distributed import Client, LocalCluster
from geoana.em.fdem import skin_depth
import discretize.utils as dis_utils
import warnings

warnings.filterwarnings("ignore")
# =============================================================================


def generate_2d_mesh_structured(
    rx_locs,
    frequencies,
    sigma_background,
    z_factor_max=5,
    z_factor_min=5,
    pfz_down=1.2,
    pfz_up=1.5,
    npadz_up=5,
    x_factor_max=2,
    spacing_factor=4,
    pfx=1.5,
    n_max=1000,
):
    """
    creat a 2D structured mesh, the typical way to model the data with uniform
    horizontal cells in the station area and geometrically increasing down and
    padding cells.

    :param rx_locs: DESCRIPTION
    :type rx_locs: TYPE
    :param frequencies: DESCRIPTION
    :type frequencies: TYPE
    :param sigma_background: DESCRIPTION
    :type sigma_background: TYPE
    :param z_factor_max: DESCRIPTION, defaults to 5
    :type z_factor_max: TYPE, optional
    :param z_factor_min: DESCRIPTION, defaults to 5
    :type z_factor_min: TYPE, optional
    :param pfz_down: DESCRIPTION, defaults to 1.2
    :type pfz_down: TYPE, optional
    :param pfz_up: DESCRIPTION, defaults to 1.5
    :type pfz_up: TYPE, optional
    :param npadz_up: DESCRIPTION, defaults to 5
    :type npadz_up: TYPE, optional
    :param x_factor_max: DESCRIPTION, defaults to 2
    :type x_factor_max: TYPE, optional
    :param spacing_factor: DESCRIPTION, defaults to 4
    :type spacing_factor: TYPE, optional
    :param pfx: DESCRIPTION, defaults to 1.5
    :type pfx: TYPE, optional
    :param n_max: DESCRIPTION, defaults to 1000
    :type n_max: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    # Setting the cells in depth dimension
    f_min = frequencies.min()
    f_max = frequencies.max()
    dz_min = np.round(skin_depth(f_max, sigma_background) / z_factor_max)
    lz = skin_depth(sigma_background, f_min) * z_factor_max
    # Setting the domain length in z-direction
    for nz_down in range(n_max):
        hz_down = dz_min * pfz_down ** np.arange(nz_down)[::-1]
        if hz_down.sum() > lz:
            break
    hz_up = [(dz_min, npadz_up, pfz_up)]
    hz_up = dis_utils.unpack_widths(hz_up)
    hz = np.r_[hz_down, hz_up]
    # Setting the cells in lateral dimension
    d_station = np.diff(rx_locs[:, 0]).min()
    dx_min = np.round(d_station / spacing_factor)
    lx = rx_locs[:, 0].max() - rx_locs[:, 0].min()
    ncx = int(lx / dx_min)
    lx_pad = skin_depth(sigma_background, f_min) * x_factor_max
    for npadx in range(n_max):
        hx_pad = dis_utils.meshTensor([(dx_min, npadx, -pfx)])
        if hx_pad.sum() > lx_pad:
            break
    hx = [(dx_min, npadx, -pfx), (dx_min, ncx), (dx_min, npadx, pfx)]

    mesh = discretize.TensorMesh([hx, hz])
    mesh.origin = np.r_[
        -mesh.hx[:npadx].sum() + rx_locs[:, 0].min(), -hz_down.sum()
    ]
    return mesh


# =============================================================================
