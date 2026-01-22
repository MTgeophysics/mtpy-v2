# -*- coding: utf-8 -*-
"""
Utility functions for plotting

Created on Sun Sep 25 15:49:01 2022

:author: jpeacock
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colorbar as mcb
import matplotlib.colors as colors

# =============================================================================
# Imports
# =============================================================================
import numpy as np


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# =============================================================================

period_label_dict = dict([(ii, "$10^{" + str(ii) + "}$") for ii in range(-20, 21)])


def get_period_limits(period: np.ndarray) -> tuple[float, float]:
    """
    Calculate period axis limits as powers of 10.

    Parameters
    ----------
    period : np.ndarray
        Array of period values

    Returns
    -------
    tuple[float, float]
        Minimum and maximum limits as powers of 10 (floor and ceil)

    """
    return (
        10 ** (np.floor(np.log10(period.min()))),
        10 ** (np.ceil(np.log10(period.max()))),
    )


def add_colorbar_axis(ax: Axes, fig: Figure) -> Axes:
    """
    Add colorbar axes positioned to the left of given axes.

    Creates a new axes for colorbar placement with dimensions calculated
    relative to the input axes position.

    Parameters
    ----------
    ax : Axes
        Reference axes for positioning colorbar
    fig : Figure
        Figure to add colorbar axes to

    Returns
    -------
    Axes
        New axes object for colorbar

    """
    # add colorbar for PT
    axpos = ax.get_position()
    cb_position = (
        axpos.bounds[0] - 0.0575,
        axpos.bounds[1] + 0.02,
        0.01,
        axpos.bounds[3] * 0.75,
    )

    cbax = fig.add_axes(cb_position)
    return cbax


def get_log_tick_labels(ax: Axes, spacing: float = 1) -> tuple[list[str], list[float]]:
    """
    Get LaTeX-formatted tick labels for logarithmic period axis.

    Generates tick labels in format $10^{exponent}$ for valid tick positions.

    Parameters
    ----------
    ax : Axes
        Axes to extract tick positions from
    spacing : float, optional
        Spacing factor for tick positions, by default 1

    Returns
    -------
    tuple[list[str], list[float]]
        Tick labels (LaTeX strings) and corresponding tick positions

    """

    tklabels = []
    xticks = []
    for tk in ax.get_xticks():
        try:
            tklabels.append(period_label_dict[tk / spacing])
            xticks.append(tk)
        except KeyError:
            pass
    return tklabels, xticks


def make_color_list(
    cbax: Axes, nseg: float, ckmin: float, ckmax: float, ckstep: float
) -> mcb.ColorbarBase:
    """
    Create segmented blue-white-red colorbar.

    Generates a colorbar with blue-to-white-to-red color transition
    using discrete segments with specified bounds.

    Parameters
    ----------
    cbax : Axes
        Axes to place colorbar in
    nseg : float
        Number of color segments
    ckmin : float
        Minimum value for colorbar range
    ckmax : float
        Maximum value for colorbar range
    ckstep : float
        Step size between color boundaries

    Returns
    -------
    mcb.ColorbarBase
        Colorbar object with segmented colormap

    """

    # make a color list
    clist = [(cc, cc, 1) for cc in np.arange(0, 1 + 1.0 / (nseg), 1.0 / (nseg))] + [
        (1, cc, cc) for cc in np.arange(1, -1.0 / (nseg), -1.0 / (nseg))
    ]

    # make segmented colormap
    mt_seg_bl2wh2rd = colors.ListedColormap(clist)

    # make bounds so that the middle is white
    bounds = np.arange(ckmin - ckstep, ckmax + 2 * ckstep, ckstep)

    # normalize the colors
    norms = colors.BoundaryNorm(bounds, mt_seg_bl2wh2rd.N)

    # make the colorbar
    return mcb.ColorbarBase(
        cbax,
        cmap=mt_seg_bl2wh2rd,
        norm=norms,
        orientation="vertical",
        ticks=bounds[1:-1],
    )


def round_to_step(num: float, base: float = 5) -> float:
    """
    Round number to nearest multiple of base.

    Parameters
    ----------
    num : float
        Number to round
    base : float, optional
        Base value to round to multiples of, by default 5

    Returns
    -------
    float
        Rounded value

    """
    return base * round(num / base)


# ==============================================================================
# function for writing values to file
# ==============================================================================
def make_value_str(
    value: float,
    value_list: list[str] | str | None = None,
    spacing: str = "{0:^8}",
    value_format: str = "{0: .2f}",
    append: bool = False,
    add: bool = False,
) -> list[str] | str:
    """
    Format value as string for file output.

    Helper function for writing values to file. Converts value to formatted
    string and either appends to list, concatenates to string, or returns
    standalone string.

    Parameters
    ----------
    value : float
        Numeric value to format
    value_list : list[str] | str | None, optional
        Existing list or string to modify, by default None
    spacing : str, optional
        Format string for spacing (e.g., '{0:^8}'), by default '{0:^8}'
    value_format : str, optional
        Format string for value (e.g., '{0: .2f}'), by default '{0: .2f}'
    append : bool, optional
        If True, append to value_list (must be list), by default False
    add : bool, optional
        If True, concatenate to value_list (must be str), by default False

    Returns
    -------
    list[str] | str
        Modified list/string or standalone formatted string

    Notes
    -----
    If both append and add are False, returns formatted string.
    If append is True, appends to list and returns list.
    If add is True, concatenates to string and returns string.

    """

    value_str = spacing.format(value_format.format(value))

    if append is True:
        value_list.append(value_str)
        return value_list
    if add is True:
        value_list += value_str
        return value_list
    if append == False and add == False:
        return value_str
    return value_list
