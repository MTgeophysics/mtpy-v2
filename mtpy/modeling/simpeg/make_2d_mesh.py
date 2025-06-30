# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:43:06 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from discretize import TensorMesh
from discretize import TreeMesh

# from dask.distributed import Client, LocalCluster
from geoana.em.fdem import skin_depth
import discretize.utils as dis_utils
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# =============================================================================


class StructuredMesh:
    def __init__(
        self,
        station_locations: pd.DataFrame,
        frequencies: NDArray | list[float],
        **kwargs
    ) -> None:
        self.station_locations = station_locations
        self.frequencies = frequencies
        self.topography = None

        self.sigma_background = 0.01

        self.z_factor_min = 5
        self.z_factor_max = 15

        self.z_geometric_factor_up = 1.5
        self.z_geometric_factor_down = 1.2

        self.n_pad_z_up = 5
        self.x_factor_max = 2
        self.x_spacing_factor = 4
        self.x_padding_geometric_factor = 1.5
        self.n_max = 1000

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def frequency_max(self) -> float:
        return self.frequencies.max()

    @property
    def frequency_min(self) -> float:
        return self.frequencies.min()

    @property
    def z1_layer_thickness(self) -> float:
        return np.round(
            skin_depth(self.frequency_max, self.sigma_background)
            / self.z_factor_max
        )

    @property
    def z_bottom(self) -> float:
        return (
            skin_depth(self.sigma_background, self.frequency_min)
            * self.z_factor_max
        )

    @property
    def z_mesh_down(self) -> NDArray[np.float64]:
        for nz_down in range(self.n_max):
            z_mesh_down = (
                self.z1_layer_thickness
                * self.z_geometric_factor_down ** np.arange(nz_down)[::-1]
            )
            if z_mesh_down.sum() > self.z_bottom:
                break
        return z_mesh_down

    @property
    def z_mesh_up(self) -> NDArray[np.float64]:
        z_mesh_up = [
            (
                self.z1_layer_thickness,
                self.n_pad_z_up,
                self.z_geometric_factor_up,
            )
        ]
        return dis_utils.unpack_widths(z_mesh_up)

    def _make_z_mesh(self) -> NDArray:
        """
        create vertical mesh
        """

        return np.r_[self.z_mesh_down, self.z_mesh_up]

    @property
    def dx(self) -> float:
        d_station = np.diff(self.station_locations[:, 0]).min()
        return np.round(d_station / self.x_spacing_factor)

    @property
    def station_total_length(self) -> float:
        return (
            self.station_locations[:, 0].max()
            - self.station_locations[:, 0].min()
        )

    @property
    def n_station_x_cells(self) -> int:
        return int(self.station_total_length / self.dx)

    @property
    def x_padding_cells(self) -> float:
        return (
            skin_depth(self.sigma_background, self.frequency_min)
            * self.x_factor_max
        )

    @property
    def n_x_padding(self) -> int:
        for npadx in range(self.n_max):
            x_pad = dis_utils.unpack_widths(
                [
                    (
                        self.dx,
                        npadx,
                        -self.x_padding_geometric_factor,
                    )
                ]
            )
            if x_pad.sum() > self.x_padding_cells:
                break
        return npadx

    def _make_x_mesh(self) -> list[tuple[float, int, float]]:
        """
        make horizontal mesh

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return [
            (
                self.dx,
                self.n_x_padding,
                -self.x_padding_geometric_factor,
            ),
            (self.dx, self.n_station_x_cells),
            (
                self.dx,
                self.n_x_padding,
                self.x_padding_geometric_factor,
            ),
        ]

    def make_mesh(self):
        """
        create structured mesh

        :return: DESCRIPTION
        :rtype: TYPE

        """

        z_mesh = self._make_z_mesh()
        x_mesh = self._make_x_mesh()

        mesh = TensorMesh([x_mesh, z_mesh])
        mesh.origin = np.r_[
            -mesh.h[0][: self.n_x_padding].sum()
            + self.station_locations[:, 0].min(),
            -self.z_mesh_down.sum(),
        ]

        self.mesh = mesh
        return self.plot_mesh()

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

    def plot_mesh(self, **kwargs):
        """
        plot the mesh
        """

        if self.mesh is not None:
            fig = plt.figure(kwargs.get("fig_num", 1))
            ax = fig.add_subplot(1, 1, 1)
            self.mesh.plot_image(
                self.active_cell_index,
                ax=ax,
                grid=True,
                grid_opts={"color": (0.75, 0.75, 0.75), "linewidth": 0.1},
                **kwargs
            )
            ax.scatter(
                self.station_locations[:, 0],
                self.station_locations[:, 1],
                marker=kwargs.get("marker", "v"),
                s=kwargs.get("s", 35),
                c=kwargs.get("c", (0, 0, 0)),
                zorder=1000,
            )
            ax.set_xlim(
                kwargs.get(
                    "xlim",
                    (
                        self.station_locations[:, 0].min() - (2 * self.dx),
                        self.station_locations[:, 0].max() + (2 * self.dx),
                    ),
                )
            )
            ax.set_ylim(kwargs.get("ylim", (-10000, 1000)))
            return ax


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
