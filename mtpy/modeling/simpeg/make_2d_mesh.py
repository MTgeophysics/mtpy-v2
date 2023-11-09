# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:43:06 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np

import discretize

from SimPEG import (
    maps,
    utils,
    optimization,
    objective_function,
    inversion,
    inverse_problem,
    directives,
    data_misfit,
    regularization,
    data,
)
from discretize import TensorMesh
from pymatsolver import Pardiso
from scipy.spatial import cKDTree
from scipy.stats import norm

# from dask.distributed import Client, LocalCluster
import dill
from geoana.em.fdem import skin_depth
import discretize.utils as dis_utils
import warnings

warnings.filterwarnings("ignore")


def generate_2d_mesh_for_mt(
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
    print(mesh)
    return mesh


# =============================================================================
