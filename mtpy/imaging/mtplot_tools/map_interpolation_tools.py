# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:06:58 2022

@author: jpeacock
"""

from __future__ import annotations

import matplotlib.tri as tri

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree, Delaunay


# =============================================================================
def in_hull(p: np.ndarray, hull: np.ndarray | Delaunay) -> np.ndarray:
    """
    Test if points are within a convex hull.

    Uses Delaunay triangulation for efficient point-in-hull testing. Falls
    back to linear programming for collinear point configurations.

    Parameters
    ----------
    p : np.ndarray
        Points to test, shape (n_points, n_dimensions)
    hull : np.ndarray | Delaunay
        Either array of hull vertices or pre-computed Delaunay object

    Returns
    -------
    np.ndarray
        Boolean array indicating which points are inside the hull

    """

    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0
    except:
        from scipy.optimize import linprog

        # Delaunay triangulation will fail if there are collinear points;
        # in those instances use linear programming (much slower) to define
        # a convex hull.
        def in_hull_lp(points, x):
            """In hull lp.
            :param points:
            :param x:
            """
            n_points = len(points)
            c = np.zeros(n_points)
            A = np.r_[points.T, np.ones((1, n_points))]
            b = np.r_[x, np.ones(1)]
            lp = linprog(c, A_eq=A, b_eq=b)
            return not lp.success

        result = []
        for cp in p:
            result.append(in_hull_lp(hull, cp))

        return np.array(result)


def get_plot_xy(
    plot_array: np.ndarray, cell_size: float, n_padding_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform x and y coordinates for interpolation grid.

    Creates regular grid coordinates with padding around the data extent.

    Parameters
    ----------
    plot_array : np.ndarray
        Structured array with 'longitude' and 'latitude' fields
    cell_size : float
        Size of grid cells in decimal degrees
    n_padding_cells : int
        Number of padding cells to add around data extent

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        plot_x : 1D array of x (longitude) coordinates
        plot_y : 1D array of y (latitude) coordinates

    """

    # create uniform x, y to plot on.
    ds = cell_size * n_padding_cells
    n_points = int(
        abs(plot_array["longitude"].max() - plot_array["longitude"].min() + 2 * ds)
        / cell_size
    )
    plot_x = np.linspace(
        plot_array["longitude"].min() - ds,
        plot_array["longitude"].max() + ds,
        n_points,
    )

    n_points = int(
        abs(plot_array["latitude"].max() - plot_array["latitude"].min() + 2 * ds)
        / cell_size
    )
    plot_y = np.linspace(
        plot_array["latitude"].min() - ds,
        plot_array["latitude"].max() + ds,
        n_points,
    )

    return plot_x, plot_y


def griddata_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
    interpolation_method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered data to regular grid using scipy.interpolate.griddata.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of data points
    y : np.ndarray
        Y coordinates of data points
    values : np.ndarray
        Values at data points
    new_x : np.ndarray
        Target x coordinates for interpolation
    new_y : np.ndarray
        Target y coordinates for interpolation
    interpolation_method : str, optional
        Interpolation method: 'nearest', 'linear', or 'cubic',
        by default 'cubic'

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        grid_x : 2D array of x coordinates
        grid_y : 2D array of y coordinates
        image : 2D array of interpolated values

    """
    points = np.array([x, y]).T
    grid_x, grid_y = np.meshgrid(new_x, new_y)

    return (
        grid_x,
        grid_y,
        interpolate.griddata(
            points,
            values,
            (grid_x, grid_y),
            method=interpolation_method,
        ),
    )


def interpolate_to_map_griddata(
    plot_array: np.ndarray,
    component: str,
    cell_size: float = 0.002,
    n_padding_cells: int = 10,
    interpolation_method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate MT data to regular map grid using griddata method.

    Parameters
    ----------
    plot_array : np.ndarray
        Structured array with 'longitude', 'latitude', and component fields
    component : str
        Name of component to interpolate (e.g., 'res_xy', 'phase_xy')
    cell_size : float, optional
        Size of grid cells in decimal degrees, by default 0.002
    n_padding_cells : int, optional
        Number of padding cells around data extent, by default 10
    interpolation_method : str, optional
        Interpolation method: 'nearest', 'linear', or 'cubic',
        by default 'cubic'

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        grid_x : 2D array of x (longitude) coordinates
        grid_y : 2D array of y (latitude) coordinates
        image : 2D array of interpolated values (log10 for resistivity)

    """

    plot_x, plot_y = get_plot_xy(plot_array, cell_size, n_padding_cells)

    grid_x, grid_y, image = griddata_interpolate(
        plot_array["longitude"],
        plot_array["latitude"],
        plot_array[component],
        plot_x,
        plot_y,
    )

    if "res" in component:
        image = np.log10(image)

    return grid_x, grid_y, image


def triangulate_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    padded_x: np.ndarray,
    padded_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
    nearest_neighbors: int = 7,
    interp_pow: float = 4,
) -> tuple[tri.Triangulation, np.ndarray, np.ndarray]:
    """
    Interpolate using Delaunay triangulation and inverse distance weighting.

    Creates triangulation on target grid and uses IDW interpolation with
    k-nearest neighbors from source data points. Masks triangles outside
    the convex hull.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of source data points
    y : np.ndarray
        Y coordinates of source data points
    values : np.ndarray
        Values at source data points
    padded_x : np.ndarray
        Padded x coordinates defining convex hull boundary
    padded_y : np.ndarray
        Padded y coordinates defining convex hull boundary
    new_x : np.ndarray
        Target x coordinates for interpolation
    new_y : np.ndarray
        Target y coordinates for interpolation
    nearest_neighbors : int, optional
        Number of nearest neighbors for IDW interpolation, by default 7
    interp_pow : float, optional
        Power parameter for IDW (higher = more weight to closer points),
        by default 4

    Returns
    -------
    tuple[tri.Triangulation, np.ndarray, np.ndarray]
        triangulation : Matplotlib triangulation object with mask
        image : Flattened array of interpolated values
        inside_indices : Boolean array indicating triangles inside hull

    """

    grid_x, grid_y = np.meshgrid(new_x, new_y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    triangulation = tri.Triangulation(grid_x, grid_y)

    mean_x = grid_x[triangulation.triangles].mean(axis=1)
    mean_y = grid_y[triangulation.triangles].mean(axis=1)

    points_mean = np.array([mean_x, mean_y]).T
    padded_points = np.array([padded_x, padded_y]).T

    inside_indices = in_hull(points_mean, padded_points)
    inside_indices = np.bool_(inside_indices)
    triangulation.set_mask(~inside_indices)

    tree = cKDTree(np.array([x, y]).T)

    xy = np.array([grid_x, grid_y]).T
    d, l = tree.query(xy, k=nearest_neighbors)

    image = None

    if nearest_neighbors == 1:
        # extract nearest neighbour values
        image = values[l]
    else:
        image = np.zeros((xy.shape[0]))

        # field values are directly assigned for coincident locations
        coincident_indices = d[:, 0] == 0
        image[coincident_indices] = values[l[coincident_indices, 0]]

        # perform idw interpolation for non-coincident locations
        idw_indices = d[:, 0] != 0
        w = np.zeros(d.shape)
        w[idw_indices, :] = 1.0 / np.power(d[idw_indices, :], interp_pow)

        image[idw_indices] = np.sum(
            w[idw_indices, :] * values[l[idw_indices, :]], axis=1
        ) / np.sum(w[idw_indices, :], axis=1)

    return triangulation, image, inside_indices


def interpolate_to_map_triangulate(
    plot_array: np.ndarray,
    component: str,
    cell_size: float = 0.002,
    n_padding_cells: int = 10,
    nearest_neighbors: int = 7,
    interp_pow: float = 4,
) -> tuple[tri.Triangulation, np.ndarray, np.ndarray]:
    """
    Interpolate MT data to map using Delaunay triangulation method.

    Uses triangulation with inverse distance weighting (IDW) for interpolation.
    Automatically applies log10 transform to resistivity components.

    Parameters
    ----------
    plot_array : np.ndarray
        Structured array with required fields:
        - 'latitude': latitude in decimal degrees
        - 'longitude': longitude in decimal degrees
        - component field (e.g., 'res_xy', 'phase_xy')
    component : str
        Name of component to interpolate (must be field in plot_array)
    cell_size : float, optional
        Size of grid cells in decimal degrees, by default 0.002
    n_padding_cells : int, optional
        Number of padding cells around data extent, by default 10
    nearest_neighbors : int, optional
        Number of nearest neighbors for IDW interpolation, by default 7
    interp_pow : float, optional
        Power parameter for IDW interpolation, by default 4

    Returns
    -------
    tuple[tri.Triangulation, np.ndarray, np.ndarray]
        triangulation : Matplotlib triangulation object with mask
        image : Flattened array of interpolated values (log10 for resistivity)
        inside_indices : Boolean array indicating triangles inside hull

    """

    # add padding to the locations
    ds = cell_size * n_padding_cells

    padded_x = plot_array["longitude"].copy()
    padded_y = plot_array["latitude"].copy()

    padded_x[np.argmin(padded_x)] -= ds
    padded_x[np.argmax(padded_x)] += ds
    padded_y[np.argmin(padded_y)] -= ds
    padded_y[np.argmax(padded_y)] += ds

    plot_x, plot_y = get_plot_xy(plot_array, cell_size, n_padding_cells)

    triangulation, image, inside_indices = triangulate_interpolation(
        plot_array["longitude"],
        plot_array["latitude"],
        plot_array[component],
        padded_x,
        padded_y,
        plot_x,
        plot_y,
        nearest_neighbors=nearest_neighbors,
        interp_pow=interp_pow,
    )

    if "res" in component:
        image = np.log10(image)

    return triangulation, image, inside_indices


def interpolate_to_map(
    plot_array: np.ndarray,
    component: str,
    cell_size: float = 0.002,
    n_padding_cells: int = 10,
    interpolation_method: str = "delaunay",
    interpolation_power: float = 5,
    nearest_neighbors: int = 7,
) -> tuple[np.ndarray | tri.Triangulation, np.ndarray, np.ndarray | None]:
    """
    Interpolate MT data to regular map grid.

    Dispatcher function that selects appropriate interpolation method.
    Supports both griddata-based (nearest, linear, cubic) and
    triangulation-based (delaunay, fancy) methods.

    Parameters
    ----------
    plot_array : np.ndarray
        Structured array with 'longitude', 'latitude', and component fields
    component : str
        Name of component to interpolate (e.g., 'res_xy', 'phase_xy')
    cell_size : float, optional
        Size of grid cells in decimal degrees, by default 0.002
    n_padding_cells : int, optional
        Number of padding cells around data extent, by default 10
    interpolation_method : str, optional
        Interpolation method:
        - 'nearest', 'linear', 'cubic': scipy.interpolate.griddata methods
        - 'delaunay', 'fancy', 'triangulate': triangulation with IDW
        by default 'delaunay'
    interpolation_power : float, optional
        Power parameter for IDW in triangulation methods, by default 5
    nearest_neighbors : int, optional
        Number of nearest neighbors for IDW, by default 7

    Returns
    -------
    tuple[np.ndarray | tri.Triangulation, np.ndarray, np.ndarray | None]
        grid_x or triangulation : Grid x coordinates (2D) or triangulation object
        grid_y or image : Grid y coordinates (2D) or interpolated values
        image or inside_indices : Interpolated values or boolean hull mask

    Notes
    -----
    Return types differ by method:
    - griddata methods: (grid_x, grid_y, image)
    - triangulation methods: (triangulation, image, inside_indices)

    """

    if interpolation_method in ["nearest", "linear", "cubic"]:
        return interpolate_to_map_griddata(
            plot_array,
            component,
            cell_size=cell_size,
            n_padding_cells=n_padding_cells,
            interpolation_method=interpolation_method,
        )

    elif interpolation_method in [
        "fancy",
        "delaunay",
        "triangulate",
    ]:
        return interpolate_to_map_triangulate(
            plot_array,
            component,
            cell_size=cell_size,
            n_padding_cells=n_padding_cells,
            nearest_neighbors=nearest_neighbors,
            interp_pow=interpolation_power,
        )
