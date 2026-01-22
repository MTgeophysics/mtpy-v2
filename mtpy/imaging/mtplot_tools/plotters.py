# -*- coding: utf-8 -*-
"""
Simple plotters elements that can be assembled in various plotting classes

Created on Sun Sep 25 15:27:28 2022

:author: jpeacock
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colorbar as mcb
import matplotlib.colors as colors
import matplotlib.patches as patches

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from mtpy.imaging.mtcolors import get_plot_color

from .utils import (
    add_colorbar_axis,
    get_period_limits,
    make_color_list,
    period_label_dict,
)


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure


# =============================================================================


def plot_errorbar(
    ax: Axes,
    x_array: np.ndarray,
    y_array: np.ndarray,
    y_error: np.ndarray | None = None,
    x_error: np.ndarray | None = None,
    **kwargs,
) -> ErrorbarContainer:
    """
    Create error bar plot with customizable properties.

    Convenience function to generate matplotlib errorbar plots with
    sensible defaults that can be overridden via kwargs.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    x_array : np.ndarray
        Array of x values
    y_array : np.ndarray
        Array of y values
    y_error : np.ndarray | None, optional
        Array of y-direction error values, by default None
    x_error : np.ndarray | None, optional
        Array of x-direction error values, by default None
    **kwargs : dict, optional
        Additional errorbar properties:
        - color : marker, line, and error bar color
        - marker : marker style
        - mew : marker edge width
        - mec : marker edge color
        - ms : marker size
        - ls : line style
        - lw : line width
        - capsize : error bar cap size
        - capthick : error bar cap thickness
        - ecolor : error bar color
        - elinewidth : error bar line width
        - picker : pick radius in points

    Returns
    -------
    ErrorbarContainer
        Matplotlib errorbar container with line data and error bars

    """
    # this is to make sure error bars plot in full and not just a dashed line
    if x_error is not None:
        x_err = abs(x_error)
    else:
        x_err = None
    if y_error is not None:
        y_err = abs(y_error)
    else:
        y_err = None
    plt_settings = {
        "color": "k",
        "marker": "x",
        "mew": 1,
        "mec": "k",
        "ms": 2,
        "ls": ":",
        "lw": 1,
        "capsize": 2,
        "capthick": 0.5,
        "ecolor": "k",
        "elinewidth": 1,
        "picker": None,
    }

    for key, value in kwargs.items():
        plt_settings[key] = value
    errorbar_object = ax.errorbar(
        x_array, y_array, xerr=x_err, yerr=y_err, **plt_settings
    )
    return errorbar_object


# =============================================================================
#  plotting functions
# =============================================================================
def plot_resistivity(
    ax: Axes,
    period: np.ndarray,
    resistivity: np.ndarray | None,
    error: np.ndarray | None,
    **properties,
) -> list[ErrorbarContainer | None]:
    """
    Plot apparent resistivity with error bars.

    Plots only non-zero resistivity values on the given axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    period : np.ndarray
        Array of period values
    resistivity : np.ndarray | None
        Array of apparent resistivity values (ohm-m)
    error : np.ndarray | None
        Array of resistivity error values
    **properties : dict, optional
        Additional errorbar properties passed to plot_errorbar

    Returns
    -------
    list[ErrorbarContainer | None]
        List containing errorbar container or [None] if no data

    """
    if resistivity is None:
        return [None]
    nz = np.nonzero(resistivity)

    if error is not None:
        error = error[nz]

    return plot_errorbar(
        ax,
        period[nz],
        resistivity[nz],
        y_error=error,
        **properties,
    )


def plot_phase(
    ax: Axes,
    period: np.ndarray,
    phase: np.ndarray | None,
    error: np.ndarray | None,
    yx: bool = False,
    **properties,
) -> list[ErrorbarContainer | None]:
    """
    Plot phase with error bars.

    Plots only non-zero phase values. Optionally adds 180 degrees
    to yx component for proper quadrant representation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    period : np.ndarray
        Array of period values
    phase : np.ndarray | None
        Array of phase values (degrees)
    error : np.ndarray | None
        Array of phase error values
    yx : bool, optional
        If True, adds 180 degrees to phase for yx component,
        by default False
    **properties : dict, optional
        Additional errorbar properties passed to plot_errorbar

    Returns
    -------
    list[ErrorbarContainer | None]
        List containing errorbar container or [None] if no data

    """
    if phase is None:
        return [None]
    # need this for the yx component
    nz = np.nonzero(phase)
    if error is not None:
        error = error[nz]
    if yx:
        return plot_errorbar(
            ax,
            period[nz],
            phase[nz] + 180,
            y_error=error,
            **properties,
        )
    return plot_errorbar(
        ax,
        period[nz],
        phase[nz],
        y_error=error,
        **properties,
    )


def plot_pt_lateral(
    ax: Axes,
    pt_obj,
    color_array: np.ndarray,
    ellipse_properties: dict,
    y_shift: float = 0,
    fig: Figure | None = None,
    edge_color: str | tuple | None = None,
    n_index: int = 0,
) -> tuple[Axes | None, mcb.Colorbar | None]:
    """
    Plot phase tensor ellipses on lateral (period) axis.

    Creates phase tensor ellipse plot with ellipses scaled by phimin/phimax
    and oriented by azimuth. Includes optional colorbar.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    pt_obj : PhaseTensor
        Phase tensor object with frequency, phimin, phimax, and azimuth
    color_array : np.ndarray
        Array of values for coloring ellipses
    ellipse_properties : dict
        Dictionary with keys:
        - 'size': ellipse size scaling factor
        - 'spacing': spacing between ellipses on period axis
        - 'colorby': property to color by
        - 'cmap': colormap name
        - 'range': [min, max, step] for color mapping
    y_shift : float, optional
        Vertical offset for ellipse centers, by default 0
    fig : Figure | None, optional
        Figure for adding colorbar, by default None
    edge_color : str | tuple | None, optional
        Color for ellipse edges, by default None
    n_index : int, optional
        Index for controlling colorbar creation (only at 0), by default 0

    Returns
    -------
    tuple[Axes | None, mcb.Colorbar | None]
        Colorbar axes and colorbar object (both None if n_index != 0)

    """
    bounds = None
    try:
        ellipse_properties["range"][2]
    except IndexError:
        ellipse_properties["range"][2] = 3
    if ellipse_properties["cmap"] == "mt_seg_bl2wh2rd":
        bounds = np.arange(
            ellipse_properties["range"][0],
            ellipse_properties["range"][1] + ellipse_properties["range"][2],
            ellipse_properties["range"][2],
        )
        nseg = float(
            (ellipse_properties["range"][1] - ellipse_properties["range"][0])
            / (2 * ellipse_properties["range"][2])
        )
    # -------------plot ellipses-----------------------------------
    for ii, ff in enumerate(1.0 / pt_obj.frequency):
        # make sure the ellipses will be visable
        if pt_obj.phimax[ii] == 0:
            continue
        eheight = pt_obj.phimin[ii] / pt_obj.phimax[ii] * ellipse_properties["size"]
        ewidth = pt_obj.phimax[ii] / pt_obj.phimax[ii] * ellipse_properties["size"]

        # create an ellipse scaled by phimin and phimax and oriented
        # along the azimuth which is calculated as clockwise but needs
        # to be plotted counter-clockwise hence the negative sign.
        ellipd = patches.Ellipse(
            (np.log10(ff) * ellipse_properties["spacing"], y_shift),
            width=ewidth,
            height=eheight,
            angle=90 - pt_obj.azimuth[ii],
        )

        ax.add_patch(ellipd)

        # get ellipse color
        ellipd.set_facecolor(
            get_plot_color(
                color_array[ii],
                ellipse_properties["colorby"],
                ellipse_properties["cmap"],
                ellipse_properties["range"][0],
                ellipse_properties["range"][1],
                bounds=bounds,
            )
        )
        if edge_color is not None:
            ellipd.set_edgecolor(edge_color)
    # set axis properties
    ax.set_ylim(
        ymin=-1.5 * ellipse_properties["size"],
        ymax=y_shift + 1.5 * ellipse_properties["size"],
    )
    cbax = None
    cbpt = None
    if n_index == 0:
        if fig is not None:
            cbax = add_colorbar_axis(ax, fig)
        if ellipse_properties["cmap"] == "mt_seg_bl2wh2rd":
            # make the colorbar
            nseg = float(
                (ellipse_properties["range"][1] - ellipse_properties["range"][0])
                / (2 * ellipse_properties["range"][2])
            )
            cbpt = make_color_list(
                cbax,
                nseg,
                ellipse_properties["range"][0],
                ellipse_properties["range"][1],
                ellipse_properties["range"][2],
            )
        else:
            cbpt = mcb.ColorbarBase(
                cbax,
                cmap=plt.get_cmap(ellipse_properties["cmap"]),
                norm=colors.Normalize(
                    vmin=ellipse_properties["range"][0],
                    vmax=ellipse_properties["range"][1],
                ),
                orientation="vertical",
            )
        cbpt.set_ticks(
            [
                ellipse_properties["range"][0],
                (ellipse_properties["range"][1] - ellipse_properties["range"][0]) / 2,
                ellipse_properties["range"][1],
            ]
        )
        cbpt.set_ticklabels(
            [
                f"{ellipse_properties['range'][0]:.0f}",
                f"{(ellipse_properties['range'][1] - ellipse_properties['range'][0]) / 2:.0f}",
                f"{ellipse_properties['range'][1]:.0f}",
            ]
        )

        cbpt.ax.yaxis.set_label_position("left")
        cbpt.ax.yaxis.set_label_coords(-1.05, 0.5)
        cbpt.ax.yaxis.tick_right()
        cbpt.ax.tick_params(axis="y", direction="in")
    return cbax, cbpt


def plot_tipper_lateral(
    axt: Axes | None,
    t_obj,
    plot_tipper: str | bool,
    real_properties: dict,
    imag_properties: dict,
    font_size: int = 6,
    legend: bool = True,
    zero_reference: bool = False,
    arrow_direction: int = 1,
) -> tuple[Axes | None, list[Line2D] | None, list[str] | None]:
    """
    Plot tipper arrows on lateral (period) axis.

    Creates tipper arrow plot showing real and/or imaginary components
    as arrows with magnitude and direction.

    Parameters
    ----------
    axt : Axes | None
        Matplotlib axes to plot on (returns None if None)
    t_obj : Tipper
        Tipper object with frequency, mag_real, angle_real, mag_imag,
        and angle_imag attributes
    plot_tipper : str | bool
        Tipper plotting mode:
        - 'yri': plot both real and imaginary
        - 'yr': plot real only
        - 'yi': plot imaginary only
        - 'y': plot both
        - False/None: no plot
    real_properties : dict
        Arrow properties for real component (matplotlib arrow kwargs)
    imag_properties : dict
        Arrow properties for imaginary component (matplotlib arrow kwargs)
    font_size : int, optional
        Font size for axis labels and legend, by default 6
    legend : bool, optional
        Whether to show legend, by default True
    zero_reference : bool, optional
        Whether to plot zero reference line, by default False
    arrow_direction : int, optional
        Arrow direction multiplier (1 or -1), by default 1

    Returns
    -------
    tuple[Axes | None, list[Line2D] | None, list[str] | None]
        Updated axes, legend handles list, and legend labels list
        All None if axt is None or t_obj is None

    """
    if t_obj is None:
        return None, None, None

    if axt is None:
        return None, None, None

    if plot_tipper.find("y") == 0 or plot_tipper:
        txr = t_obj.mag_real * np.cos(
            np.deg2rad(-t_obj.angle_real) + arrow_direction * np.pi
        )
        tyr = t_obj.mag_real * np.sin(
            np.deg2rad(-t_obj.angle_real) + arrow_direction * np.pi
        )

        txi = t_obj.mag_imag * np.cos(
            np.deg2rad(-t_obj.angle_imag) + arrow_direction * np.pi
        )
        tyi = t_obj.mag_imag * np.sin(
            np.deg2rad(-t_obj.angle_imag) + arrow_direction * np.pi
        )

        nt = len(txr)
        period = 1.0 / t_obj.frequency
        x_limits = get_period_limits(period)

        tiplist = []
        tiplabel = []

        if plot_tipper.find("r") > 0:
            line = Line2D([0], [0], color=real_properties["facecolor"], lw=1)
            tiplist.append(line)
            tiplabel.append("real")
        if plot_tipper.find("i") > 0:
            line = Line2D([0], [0], color=imag_properties["facecolor"], lw=1)
            tiplist.append(line)
            tiplabel.append("imag")
        for aa in range(nt):
            xlenr = txr[aa] * np.log10(period[aa])
            xleni = txi[aa] * np.log10(period[aa])

            if xlenr == 0 and xleni == 0:
                continue
            # --> plot real arrows
            if plot_tipper.find("r") > 0:
                axt.arrow(
                    np.log10(period[aa]),
                    0,
                    xlenr,
                    tyr[aa],
                    **real_properties,
                )
            # --> plot imaginary arrows
            if plot_tipper.find("i") > 0:
                axt.arrow(
                    np.log10(period[aa]),
                    0,
                    xleni,
                    tyi[aa],
                    **imag_properties,
                )
        # make a line at 0 for reference
        if zero_reference:
            axt.plot(np.log10(period), [0] * nt, "k", lw=0.5)
        if legend:
            axt.legend(
                tiplist,
                tiplabel,
                loc="upper left",
                markerscale=1,
                borderaxespad=0.01,
                labelspacing=0.07,
                handletextpad=0.2,
                borderpad=0.1,
                prop={"size": 6},
            )
        # set axis properties

        axt.set_xlim(np.log10(x_limits[0]), np.log10(x_limits[1]))

        tklabels = []
        xticks = []

        for tk in axt.get_xticks():
            try:
                tklabels.append(period_label_dict[tk])
                xticks.append(tk)
            except KeyError:
                pass
        axt.set_xticks(xticks)
        axt.set_xticklabels(tklabels, fontdict={"size": font_size})
        # need to reset the x_limits caouse they get reset when calling
        # set_ticks for some reason
        axt.set_xlim(np.log10(x_limits[0]), np.log10(x_limits[1]))

        # axt.set_xscale('log', nonpositive='clip')
        tmax = max([np.nanmax(tyr), np.nanmax(tyi)])
        if tmax > 1:
            tmax = 0.899
        tmin = min([np.nanmin(tyr), np.nanmin(tyi)])
        if tmin < -1:
            tmin = -0.899
        tipper_limits = (tmin - 0.1, tmax + 0.1)
        axt.set_ylim(tipper_limits)
        axt.grid(True, alpha=0.25, which="both", color=(0.25, 0.25, 0.25), lw=0.25)
    return axt, tiplist, tiplabel


def add_raster(
    ax: Axes, raster_fn: str, add_colorbar: bool = True, **kwargs
) -> tuple[Axes, mcb.Colorbar | None]:
    """
    Add a raster image to axes using rasterio.

    Overlays a georeferenced raster (e.g., GeoTIFF) on matplotlib axes
    with optional colorbar.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add raster to
    raster_fn : str
        Path to raster file (must be readable by rasterio)
    add_colorbar : bool, optional
        Whether to add colorbar, by default True
    **kwargs : dict, optional
        Additional keyword arguments passed to rasterio.plot.show

    Returns
    -------
    tuple[Axes, mcb.Colorbar | None]
        Updated axes and colorbar (None if add_colorbar=False)

    """

    import rasterio
    from rasterio.plot import show

    tif = rasterio.open(raster_fn)
    ax2 = show(tif, ax=ax, **kwargs)
    cb = None
    if add_colorbar:
        im = ax2.get_images()[0]
        fig = ax2.get_figure()
        cb = fig.colorbar(im, ax=ax)

    return ax2, cb
