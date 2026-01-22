# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:19:16 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import numpy as np


# =============================================================================


class MTEllipse:
    """
    Helper class for managing phase tensor ellipse properties.

    Configures ellipse visualization parameters including size, color mapping,
    and color ranges for plotting phase tensor ellipses.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments to set ellipse properties. Valid keys correspond
        to the class attributes listed below.

    Attributes
    ----------
    ellipse_size : float
        Size of ellipse in points, by default 2
    ellipse_colorby : str
        Property to color ellipses by:
        - 'phimin' or 'phiminang': minimum phase (default)
        - 'phimax' or 'phimaxang': maximum phase
        - 'skew': skew angle (beta)
        - 'skew_seg': skew in discrete segments
        - 'normalized_skew': normalized skew (Booker, 2014)
        - 'normalized_skew_seg': normalized skew in discrete segments
        - 'phidet': determinant of phase tensor
        - 'ellipticity': ellipticity of phase tensor
        - 'strike' or 'azimuth': strike angle
    ellipse_range : tuple[float, float, float]
        Range for color mapping as (min, max, step), by default (0, 90, 10)
    ellipse_cmap : str
        Colormap name:
        - 'mt_yl2rd': yellow to red
        - 'mt_bl2yl2rd': blue to yellow to red
        - 'mt_wh2bl': white to blue
        - 'mt_rd2bl': red to blue
        - 'mt_bl2wh2rd': blue to white to red
        - 'mt_bl2gr2rd': blue to green to red (default)
        - 'mt_rd2gr2bl': red to green to blue
        - 'mt_seg_bl2wh2rd': discrete blue to white to red
    ellipse_spacing : float
        Spacing between ellipses, by default 1

    """

    def __init__(self, **kwargs) -> None:
        self.ellipse_size = 2
        self.ellipse_colorby = "phimin"
        self.ellipse_range = (0, 90, 10)
        self.ellipse_cmap = "mt_bl2gr2rd"
        self.ellipse_spacing = 1

        # Set class property values from kwargs and pop them
        for v in vars(self):
            if v in list(kwargs.keys()):
                setattr(self, v, kwargs.pop(v, None))
        self.get_range()
        self.get_color_map()

    def get_color_map(self) -> None:
        """
        Set appropriate colormap based on colorby property.

        Automatically sets colormap to 'mt_seg_bl2wh2rd' for segmented
        skew coloring modes.

        """
        if self.ellipse_colorby in ["skew_seg", "normalized_skew_seg"]:
            self.ellipse_cmap = "mt_seg_bl2wh2rd"

    def get_range(self) -> None:
        """
        Set appropriate color range based on colorby property.

        Automatically determines color range if not explicitly set:
        - Skew-based: (-9, 9, 3)
        - Ellipticity: (0, 1, 0.1)
        - Phase-based: (0, 90, 5)

        """
        # set color ranges
        if self.ellipse_range[0] == self.ellipse_range[1]:
            if self.ellipse_colorby in [
                "skew",
                "skew_seg",
                "normalized_skew",
                "normalized_skew_seg",
            ]:
                self.ellipse_range = (-9, 9, 3)
            elif self.ellipse_colorby == "ellipticity":
                self.ellipse_range = (0, 1, 0.1)
            else:
                self.ellipse_range = (0, 90, 5)

    @property
    def ellipse_cmap_n_segments(self) -> float:
        """
        Calculate number of segments for colormap.

        Returns
        -------
        float
            Number of colormap segments based on ellipse_range

        """
        return float(
            (self.ellipse_range[1] - self.ellipse_range[1])
            / (2 * self.ellipse_range[2])
        )

    @property
    def ellipse_cmap_bounds(self) -> np.ndarray | None:
        """
        Get colormap boundaries for discrete coloring.

        Returns
        -------
        np.ndarray | None
            Array of boundary values for colormap bins, or None if
            ellipse_range is not properly defined

        """
        try:
            return np.arange(
                self.ellipse_range[0],
                self.ellipse_range[1] + self.ellipse_range[2],
                self.ellipse_range[2],
            )
        except IndexError:
            return None

    def get_pt_color_array(self, pt_object) -> np.ndarray:
        """
        Extract appropriate array from phase tensor for coloring ellipses.

        Parameters
        ----------
        pt_object : PhaseTensor
            Phase tensor object containing phase tensor properties

        Returns
        -------
        np.ndarray
            Array of values corresponding to the selected colorby property

        Raises
        ------
        NameError
            If ellipse_colorby is not a supported option

        """

        # get the properties to color the ellipses by
        if self.ellipse_colorby == "phiminang" or self.ellipse_colorby == "phimin":
            color_array = pt_object.phimin
        elif self.ellipse_colorby == "phimaxang" or self.ellipse_colorby == "phimax":
            color_array = pt_object.phimax
        elif self.ellipse_colorby == "phidet":
            color_array = np.sqrt(abs(pt_object.det)) * (180 / np.pi)
        elif self.ellipse_colorby == "skew" or self.ellipse_colorby == "skew_seg":
            color_array = pt_object.beta
        elif self.ellipse_colorby == "ellipticity":
            color_array = pt_object.ellipticity
        elif self.ellipse_colorby in ["strike", "azimuth"]:
            color_array = pt_object.azimuth % 180
            color_array[np.where(color_array > 90)] -= 180
        else:
            raise NameError(self.ellipse_colorby + " is not supported")
        return color_array

    @property
    def ellipse_properties(self) -> dict[str, float | tuple | str]:
        """
        Get dictionary of ellipse properties.

        Returns
        -------
        dict[str, float | tuple | str]
            Dictionary containing ellipse visualization parameters:
            - size: ellipse size
            - range: color range tuple
            - cmap: colormap name
            - colorby: property to color by
            - spacing: ellipse spacing

        """
        return {
            "size": self.ellipse_size,
            "range": self.ellipse_range,
            "cmap": self.ellipse_cmap,
            "colorby": self.ellipse_colorby,
            "spacing": self.ellipse_spacing,
        }
