# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:34:53 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

from mtpy.imaging.mtplot_tools import PlotBase
from .model import Occam1DModel

# =============================================================================
class PlotOccam1DL2(PlotBase):
    """
    plot L2 curve of iteration vs rms and roughness

    Arguments:
    ----------
        **rms_arr** : structured array with keys:
                      * 'iteration' --> for iteration number (int)
                      * 'rms' --> for rms (float)
                      * 'roughness' --> for roughness (float)

    ======================= ===================================================
    Keywords/attributes     Description
    ======================= ===================================================
    ax1                     matplotlib.axes instance for rms vs iteration
    ax2                     matplotlib.axes instance for roughness vs rms
    fig                     matplotlib.figure instance
    fig_dpi                 resolution of figure in dots-per-inch
    fig_num                 number of figure instance
    fig_size                size of figure in inches (width, height)
    font_size               size of axes tick labels, axes labels is +2
    plot_yn                 [ 'y' | 'n']
                            'y' --> to plot on instantiation
                            'n' --> to not plot on instantiation
    rms_arr                 structure np.array as described above
    rms_color               color of rms marker and line
    rms_lw                  line width of rms line
    rms_marker              marker for rms values
    rms_marker_size         size of marker for rms values
    rms_mean_color          color of mean line
    rms_median_color        color of median line
    rough_color             color of roughness line and marker
    rough_font_size         font size for iteration number inside roughness
                            marker
    rough_lw                line width for roughness line
    rough_marker            marker for roughness
    rough_marker_size       size of marker for roughness
    subplot_bottom          subplot spacing from bottom
    subplot_left            subplot spacing from left
    subplot_right           subplot spacing from right
    subplot_top             subplot spacing from top
    ======================= ===================================================

    =================== =======================================================
    Methods             Description
    =================== =======================================================
    plot                plots L2 curve.
    redraw_plot         call redraw_plot to redraw the figures,
                        if one of the attributes has been changed
    save_figure         saves the matplotlib.figure instance to desired
                        location and format
    =================== ======================================================

    """

    def __init__(self, dir_path, model_fn, **kwargs):
        self.dir_path = Path(dir_path)
        self.model_fn = Path(model_fn)
        self._get_iter_list()

        self.subplot_right = 0.98
        self.subplot_left = 0.085
        self.subplot_top = 0.91
        self.subplot_bottom = 0.1

        self.fig_num = kwargs.pop("fig_num", 1)
        self.fig_size = kwargs.pop("fig_size", [6, 6])
        self.fig_dpi = kwargs.pop("dpi", 300)
        self.font_size = kwargs.pop("font_size", 8)

        self.rms_lw = kwargs.pop("rms_lw", 1)
        self.rms_marker = kwargs.pop("rms_marker", "d")
        self.rms_color = kwargs.pop("rms_color", "k")
        self.rms_marker_size = kwargs.pop("rms_marker_size", 5)
        self.rms_median_color = kwargs.pop("rms_median_color", "red")
        self.rms_mean_color = kwargs.pop("rms_mean_color", "orange")

        self.rough_lw = kwargs.pop("rough_lw", 0.75)
        self.rough_marker = kwargs.pop("rough_marker", "o")
        self.rough_color = kwargs.pop("rough_color", "b")
        self.rough_marker_size = kwargs.pop("rough_marker_size", 7)
        self.rough_font_size = kwargs.pop("rough_font_size", 6)

        self.plot_yn = kwargs.pop("plot_yn", "y")
        if self.plot_yn == "y":
            self.plot()

    def _get_iter_list(self):
        """
        get all iteration files in dir_path
        """

        if not self.dir_path.exists():
            raise IOError(f"Could not find {self.dir_path}")

        iter_list = list(self.dir_path.glob("*.iter"))

        self.rms_arr = np.zeros(
            len(iter_list),
            dtype=np.dtype(
                [
                    ("iteration", np.int),
                    ("rms", np.float),
                    ("roughness", np.float),
                ]
            ),
        )
        for ii, fn in enumerate(iter_list):
            m1 = Occam1DModel()
            m1.read_iter_file(fn, self.model_fn)
            self.rms_arr[ii]["iteration"] = int(m1.itdict["Iteration"])
            self.rms_arr[ii]["rms"] = float(m1.itdict["Misfit Value"])
            self.rms_arr[ii]["roughness"] = float(m1.itdict["Roughness Value"])

        self.rms_arr.sort(order="iteration")

    def plot(self):
        """
        plot L2 curve
        """

        nr = self.rms_arr.shape[0]
        med_rms = np.median(self.rms_arr["rms"])
        mean_rms = np.mean(self.rms_arr["rms"])

        # set the dimesions of the figure
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["figure.subplot.left"] = self.subplot_left
        plt.rcParams["figure.subplot.right"] = self.subplot_right
        plt.rcParams["figure.subplot.bottom"] = self.subplot_bottom
        plt.rcParams["figure.subplot.top"] = self.subplot_top

        # make figure instance
        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()

        # make a subplot for RMS vs Iteration
        self.ax1 = self.fig.add_subplot(1, 1, 1)

        # plot the rms vs iteration
        (l1,) = self.ax1.plot(
            self.rms_arr["iteration"],
            self.rms_arr["rms"],
            "-k",
            lw=1,
            marker="d",
            ms=5,
        )

        # plot the median of the RMS
        (m1,) = self.ax1.plot(
            self.rms_arr["iteration"],
            np.repeat(med_rms, nr),
            ls="--",
            color=self.rms_median_color,
            lw=self.rms_lw * 0.75,
        )

        # plot the mean of the RMS
        (m2,) = self.ax1.plot(
            self.rms_arr["iteration"],
            np.repeat(mean_rms, nr),
            ls="--",
            color=self.rms_mean_color,
            lw=self.rms_lw * 0.75,
        )

        # make subplot for RMS vs Roughness Plot
        self.ax2 = self.ax1.twiny()

        self.ax2.set_xlim(
            self.rms_arr["roughness"][1:].min(),
            self.rms_arr["roughness"][1:].max(),
        )

        self.ax1.set_ylim(0, self.rms_arr["rms"][1])

        # plot the rms vs roughness
        (l2,) = self.ax2.plot(
            self.rms_arr["roughness"],
            self.rms_arr["rms"],
            ls="--",
            color=self.rough_color,
            lw=self.rough_lw,
            marker=self.rough_marker,
            ms=self.rough_marker_size,
            mfc="white",
        )

        # plot the iteration number inside the roughness marker
        for rms, ii, rough in zip(
            self.rms_arr["rms"],
            self.rms_arr["iteration"],
            self.rms_arr["roughness"],
        ):
            # need this because if the roughness is larger than this number
            # matplotlib puts the text out of bounds and a draw_text_image
            # error is raised and file cannot be saved, also the other
            # numbers are not put in.
            if rough > 1e8:
                pass
            else:
                self.ax2.text(
                    rough,
                    rms,
                    f"{ii}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontdict={
                        "size": self.rough_font_size,
                        "weight": "bold",
                        "color": self.rough_color,
                    },
                )

        # make a legend
        self.ax1.legend(
            [l1, l2, m1, m2],
            [
                "RMS",
                "Roughness",
                f"Median_RMS={med_rms:.2f}",
                f"Mean_RMS={mean_rms:.2f}",
            ],
            ncol=1,
            loc="upper right",
            columnspacing=0.25,
            markerscale=0.75,
            handletextpad=0.15,
        )

        # set the axis properties for RMS vs iteration
        self.ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        self.ax1.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax1.set_ylabel(
            "RMS", fontdict={"size": self.font_size + 2, "weight": "bold"}
        )
        self.ax1.set_xlabel(
            "Iteration", fontdict={"size": self.font_size + 2, "weight": "bold"}
        )
        self.ax1.grid(alpha=0.25, which="both", lw=self.rough_lw)
        self.ax2.set_xlabel(
            "Roughness",
            fontdict={
                "size": self.font_size + 2,
                "weight": "bold",
                "color": self.rough_color,
            },
        )

        for t2 in self.ax2.get_xticklabels():
            t2.set_color(self.rough_color)

        plt.show()
