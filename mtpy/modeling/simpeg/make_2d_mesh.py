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
from discretize import TreeMesh
from discretize.utils import mkvc
import matplotlib.pyplot as plt
from discretize.utils import active_from_xyz

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

    mesh = TensorMesh([hx, hz])
    mesh.origin = np.r_[
        -mesh.hx[:npadx].sum() + rx_locs[:, 0].min(), -hz_down.sum()
    ]
    return mesh


class QuadTreeMesh:
    """
    build a quad tree mesh based on station locations and frequencies to invert

    dimensions are x for the lateral dimension and z for the vertical.

    station locations should be offsets from a single point [offset, elevation].
    should be shape [n, 2]

    topography should be [x, z] -: [n, 2] and in station location coordinate
    system.

    """

    def __init__(self, station_locations, frequencies, **kwargs):
        self.station_locations = station_locations
        self.frequencies = frequencies
        self.topography = None
        self.topography_padding = [[0, 2], [0, 2]]
        self.station_padding = [2, 2]

        self.sigma_background = 0.01
        # station spacing (only used for this example)
        self.station_spacing = None
        # factor for cell spacing
        self.factor_spacing = 4
        # factor to pad in the x-direction
        self.factor_x_pad = 2
        # factor to pad in the subsurface
        self.factor_z_pad_down = 2
        # padding factor within the core model volume
        self.factor_z_core = 1
        self.factor_z_pad_up = 1
        self.topography_level = -1
        self.station_level = -1

        self.update_from_stations = True
        self.update_from_topography = True

        self.origin = "CC"
        self.mesh = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def x_pad(self):
        return (
            skin_depth(self.frequencies.min(), self.sigma_background)
            * self.factor_x_pad
        )

    @property
    def x_total(self):
        """
        get the total distance in the horizontal direction.
        """
        return (
            self.station_locations.max()
            - self.station_locations.min()
            + self.x_pad * 2
        )

    @property
    def z_pad_down(self):
        return (
            skin_depth(self.frequencies.min(), self.sigma_background)
            * self.factor_z_pad_down
        )

    @property
    def z_core(self):
        return (
            skin_depth(self.frequencies.min(), self.sigma_background)
            * self.factor_z_core
        )

    @property
    def z_pad_up(self):
        return (
            skin_depth(self.frequencies.min(), self.sigma_background)
            * self.factor_z_pad_up
        )

    @property
    def z_total(self):
        return self.z_pad_up + self.z_core + self.z_pad_down

    @property
    def dx(self):
        return np.diff(
            self.station_locations[:, 0] / self.factor_spacing
        ).mean()

    @property
    def dz(self):
        return np.round(
            skin_depth(self.frequencies.max(), self.sigma_background) / 4,
            decimals=-1,
        )

    @property
    def nx(self):
        return 2 ** int(np.ceil(np.log(self.x_total / self.dx) / np.log(2.0)))

    @property
    def nz(self):
        return 2 ** int(np.ceil(np.log(self.z_total / self.dz) / np.log(2.0)))

    @property
    def z_max(self):
        if self.topography:
            return self.topography.max()
        else:
            return self.station_locations[:, 1].max()

    def make_mesh(self, **kwargs):
        """
        create mesh
        """

        mesh = TreeMesh(
            [[(self.dx, self.nx)], [(self.dz, self.nz)]], x0=self.origin
        )

        # Refine surface topography
        if self.topography is not None and self.update_from_topography:
            pts = np.c_[self.topography[:, 0], self.topography[:, 1]]
            mesh.refine_surface(
                pts,
                level=self.topography_level,
                padding_cells_by_level=self.topography_padding,
                finalize=False,
            )

        # Refine mesh near points
        if self.update_from_stations:
            pts = np.c_[
                self.station_locations[:, 0], self.station_locations[:, 1]
            ]
            mesh.refine_points(
                pts,
                level=self.station_level,
                padding_cells_by_level=self.station_padding,
                finalize=False,
            )

        mesh.finalize()

        self.mesh = mesh
        return self.plot_mesh()

    def plot_mesh(self, **kwargs):
        """
        plot the mesh
        """

        if self.mesh is not None:
            ax = self.mesh.plot_grid(**kwargs)
            ax.scatter(
                self.station_locations[:, 0],
                self.station_locations[:, 1],
                marker=kwargs.get("marker", "v"),
                s=kwargs.get("s", 35),
                c=kwargs.get("c", (0, 0, 0)),
                zorder=1000,
            )
            ax.set_xlim(
                self.station_locations[:, 0].min() - (2 * self.dx),
                self.station_locations[:, 0].max() + (2 * self.dx),
            )
            ax.set_ylim(kwargs.get("ylim", (-10000, 1000)))
            return ax

    @property
    def active_cell_index(self):
        """
        return active cell mask

        TODO: include topographic surface

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.topography is None:
            return self.mesh.cell_centers[:, 1] < 0

        else:
            raise NotImplementedError("Have not included topography yet.")

    @property
    def number_of_active_cells(self):
        """
        number of active cells
        """
        return int(self.active_cell_index.sum())


# =============================================================================
