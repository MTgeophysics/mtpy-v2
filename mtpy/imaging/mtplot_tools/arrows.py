# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:16:31 2022

@author: jpeacock
"""

from __future__ import annotations


# ==============================================================================
# Arrows properties for induction vectors
# ==============================================================================
class MTArrows:
    """
    Helper class to configure arrow properties for plotting induction vectors.

    This class manages arrow styling parameters for real and imaginary
    components of induction vectors in magnetotelluric plots.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments to set arrow properties. Valid keys correspond
        to the class attributes listed below.

    Attributes
    ----------
    arrow_size : float
        Multiplier to scale the arrow size, by default 2.5
    arrow_head_length : float
        Length of the arrow head, by default 0.15 * arrow_size
    arrow_head_width : float
        Width of the arrow head, by default 0.1 * arrow_size
    arrow_lw : float
        Line width of the arrow, by default 0.5 * arrow_size
    arrow_threshold : float
        Threshold value above which arrows will not be plotted. This helps
        filter out poor quality data. Applied before scaling by arrow_size,
        by default 2
    arrow_color_imag : str
        Color for imaginary component arrows, by default 'c' (cyan)
    arrow_color_real : str
        Color for real component arrows, by default 'k' (black)
    arrow_direction : int
        Arrow direction convention:
        - 0: arrows point away from conductor
        - 1: arrows point toward conductor (Parkinson convention)
        By default 1

    Notes
    -----
    The data inherently points away from conductors, so arrow_direction=1
    flips the arrows to follow the Parkinson convention.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.arrow_size = 2.5
        self.arrow_head_length = 0.15 * self.arrow_size
        self.arrow_head_width = 0.1 * self.arrow_size
        self.arrow_lw = 0.5 * self.arrow_size
        self.arrow_threshold = 2
        self.arrow_color_imag = "c"
        self.arrow_color_real = "k"
        # the data is inherently pointing away from conductor, need to flip
        # to be in parkinson convention
        self.arrow_direction = 1

        # Set class property values from kwargs and pop them
        for v in vars(self):
            if v in list(kwargs.keys()):
                setattr(self, v, kwargs.pop(v, None))
