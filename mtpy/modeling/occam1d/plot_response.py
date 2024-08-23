# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:33:37 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
class Plot1DResponse(object):
    """Plot the 1D response and model.

    Plots apparent resisitivity and phase
in different subplots with the model on the far right.  You can plot both
    TE and TM modes together along with different iterations of the model.
    These will be plotted in different colors or shades of gray depneng on
    color_scale.

    :Example: ::

        >>> import mtpy.modeling.occam1d as occam1d
        >>> p1 = occam1d.Plot1DResponse(plot_yn='n')
        >>> p1.data_te_fn = r"/home/occam1d/mt01/TE/Occam_DataFile_TE.dat"
        >>> p1.data_tm_fn = r"/home/occam1d/mt01/TM/Occam_DataFile_TM.dat"
        >>> p1.model_fn = r"/home/occam1d/mt01/TE/Model1D"
        >>> p1.iter_te_fn = [r"/home/occam1d/mt01/TE/TE_{0}.iter".format(ii)
        >>> ...              for ii in range(5,10)]
        >>> p1.iter_tm_fn = [r"/home/occam1d/mt01/TM/TM_{0}.iter".format(ii)
        >>> ...              for ii in range(5,10)]
        >>> p1.resp_te_fn = [r"/home/occam1d/mt01/TE/TE_{0}.resp".format(ii)
        >>> ...              for ii in range(5,10)]
        >>> p1.resp_tm_fn = [r"/home/occam1d/mt01/TM/TM_{0}.resp".format(ii)
        >>> ...              for ii in range(5,10)]
        >>> p1.plot()

    ==================== ======================================================
    Attributes           Description
    ==================== ======================================================
    axm                  matplotlib.axes instance for model subplot
    axp                  matplotlib.axes instance for phase subplot
    axr                  matplotlib.axes instance for app. res subplot
    color_mode           [ 'color' | 'bw' ]
    cted                 color of TE data markers
    ctem                 color of TM data markers
    ctmd                 color of TE model markers
    ctmm                 color of TM model markers
    data_te_fn           full path to data file for TE mode
    data_tm_fn           full path to data file for TM mode
    depth_limits         (min, max) limits for depth plot in depth_units
    depth_scale          [ 'log' | 'linear' ] *default* is linear
    depth_units          [ 'm' | 'km' ] *default is 'km'
    e_capsize            capsize of error bars
    e_capthick           cap thickness of error bars
    fig                  matplotlib.figure instance for plot
    fig_dpi              resolution in dots-per-inch for figure
    fig_num              number of figure instance
    fig_size             size of figure in inches [width, height]
    font_size            size of axes tick labels, axes labels are +2
    grid_alpha           transparency of grid
    grid_color           color of grid
    iter_te_fn           full path or list of .iter files for TE mode
    iter_tm_fn           full path or list of .iter files for TM mode
    lw                   width of lines for model
    model_fn             full path to model file
    ms                   marker size
    mted                 marker for TE data
    mtem                 marker for TM data
    mtmd                 marker for TE model
    mtmm                 marker for TM model
    phase_limits         (min, max) limits on phase in degrees
    phase_major_ticks    spacing for major ticks in phase
    phase_minor_ticks    spacing for minor ticks in phase
    plot_yn              [ 'y' | 'n' ] plot on instantiation
    res_limits           limits of resistivity in linear scale
    resp_te_fn           full path or list of .resp files for TE mode
    resp_tm_fn           full path or list of .iter files for TM mode
    subplot_bottom       spacing of subplots from bottom of figure
    subplot_hspace       height spacing between subplots
    subplot_left         spacing of subplots from left of figure
    subplot_right        spacing of subplots from right of figure
    subplot_top          spacing of subplots from top of figure
    subplot_wspace       width spacing between subplots
    title_str            title of plot
    ==================== ======================================================
    """

    def __init__(
        self,
        data_te_fn=None,
        data_tm_fn=None,
        model_fn=None,
        resp_te_fn=None,
        resp_tm_fn=None,
        iter_te_fn=None,
        iter_tm_fn=None,
        **kwargs
    ):
        self.data_te_fn = data_te_fn
        self.data_tm_fn = data_tm_fn

        self.model_fn = model_fn

        self.override_legend_subscript = kwargs.pop(
            "override_legend_subscript", None
        )
        self.resp_te_fn = resp_te_fn
        if type(self.resp_te_fn) is not list:
            self.resp_te_fn = [self.resp_te_fn]

        self.resp_tm_fn = resp_tm_fn
        if type(self.resp_tm_fn) is not list:
            self.resp_tm_fn = [self.resp_tm_fn]

        self.iter_te_fn = iter_te_fn
        if type(self.iter_te_fn) is not list:
            self.iter_te_fn = [self.iter_te_fn]

        self.iter_tm_fn = iter_tm_fn
        if type(self.iter_tm_fn) is not list:
            self.iter_tm_fn = [self.iter_tm_fn]

        self.color_mode = kwargs.pop("color_mode", "color")

        self.ms = kwargs.pop("ms", 1.5)
        self.lw = kwargs.pop("lw", 0.5)
        self.ls = kwargs.pop("ls", ":")
        self.e_capthick = kwargs.pop("e_capthick", 0.5)
        self.e_capsize = kwargs.pop("e_capsize", 2)

        self.phase_major_ticks = kwargs.pop("phase_major_ticks", 10)
        self.phase_minor_ticks = kwargs.pop("phase_minor_ticks", 5)

        self.grid_color = kwargs.pop("grid_color", (0.25, 0.25, 0.25))
        self.grid_alpha = kwargs.pop("grid_alpha", 0.3)

        # color mode
        if self.color_mode == "color":
            # color for data
            self.cted = kwargs.pop("cted", (0, 0, 1))
            self.ctmd = kwargs.pop("ctmd", (1, 0, 0))
            self.mted = kwargs.pop("mted", "s")
            self.mtmd = kwargs.pop("mtmd", "o")

            # color for occam2d model
            self.ctem = kwargs.pop("ctem", (0, 0.6, 0.3))
            self.ctmm = kwargs.pop("ctmm", (0.9, 0, 0.8))
            self.mtem = kwargs.pop("mtem", "+")
            self.mtmm = kwargs.pop("mtmm", "+")

        # black and white mode
        elif self.color_mode == "bw":
            # color for data
            self.cted = kwargs.pop("cted", (0, 0, 0))
            self.ctmd = kwargs.pop("ctmd", (0, 0, 0))
            self.mted = kwargs.pop("mted", "*")
            self.mtmd = kwargs.pop("mtmd", "v")

            # color for occam2d model
            self.ctem = kwargs.pop("ctem", (0.6, 0.6, 0.6))
            self.ctmm = kwargs.pop("ctmm", (0.6, 0.6, 0.6))
            self.mtem = kwargs.pop("mtem", "+")
            self.mtmm = kwargs.pop("mtmm", "x")

        self.phase_limits = kwargs.pop("phase_limits", (-5, 95))
        self.res_limits = kwargs.pop("res_limits", None)
        self.depth_limits = kwargs.pop("depth_limits", None)
        self.depth_scale = kwargs.pop("depth_scale", "linear")
        self.depth_units = kwargs.pop("depth_units", "km")

        self.fig_num = kwargs.pop("fig_num", 1)
        self.fig_size = kwargs.pop("fig_size", [6, 6])
        self.fig_dpi = kwargs.pop("dpi", 300)
        self.fig = None
        self.axr = None
        self.axp = None
        self.axm = None

        self.subplot_wspace = 0.25
        self.subplot_hspace = 0.15
        self.subplot_right = 0.92
        self.subplot_left = 0.085
        self.subplot_top = 0.93
        self.subplot_bottom = 0.1

        self.font_size = kwargs.pop("font_size", 6)

        self.title_str = kwargs.pop("title_str", "")
        self.plot_yn = kwargs.pop("plot_yn", "y")

        if self.plot_yn == "y":
            self.plot()

    def plot(self):
        """Plot data, response and model."""
        if type(self.resp_te_fn) is not list:
            self.resp_te_fn = [self.resp_te_fn]

        if type(self.resp_tm_fn) is not list:
            self.resp_tm_fn = [self.resp_tm_fn]

        if type(self.iter_te_fn) is not list:
            self.iter_te_fn = [self.iter_te_fn]

        if type(self.iter_tm_fn) is not list:
            self.iter_tm_fn = [self.iter_tm_fn]

        # make a grid of subplots
        gs = gridspec.GridSpec(
            6, 5, hspace=self.subplot_hspace, wspace=self.subplot_wspace
        )

        # make a figure
        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()

        # set some plot parameters
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["figure.subplot.left"] = self.subplot_left
        plt.rcParams["figure.subplot.right"] = self.subplot_right
        plt.rcParams["figure.subplot.bottom"] = self.subplot_bottom
        plt.rcParams["figure.subplot.top"] = self.subplot_top

        # subplot resistivity
        self.axr = self.fig.add_subplot(gs[:4, :4])

        # subplot for phase
        self.axp = self.fig.add_subplot(gs[4:, :4], sharex=self.axr)

        # subplot for model
        self.axm = self.fig.add_subplot(gs[:, 4])

        legend_marker_list_te = []
        legend_label_list_te = []
        legend_marker_list_tm = []
        legend_label_list_tm = []
        # --> plot data apparent resistivity and phase-------------------------
        if self.data_te_fn is not None:
            d1 = Data()
            d1.read_data_file(self.data_te_fn)

            # --> cut out missing data
            rxy = np.where(d1.res_te[0] != 0)[0]

            # --> TE mode Data
            if len(rxy) > 0:
                rte = self.axr.errorbar(
                    1.0 / d1.freq[rxy],
                    d1.res_te[0][rxy],
                    ls=self.ls,
                    marker=self.mted,
                    ms=self.ms,
                    mfc=self.cted,
                    mec=self.cted,
                    color=self.cted,
                    yerr=d1.res_te[1][rxy],
                    ecolor=self.cted,
                    picker=2,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
                legend_marker_list_te.append(rte[0])
                if self.override_legend_subscript is not None:
                    legend_label_list_tm.append(
                        "$Obs_{"
                        + str.upper(self.override_legend_subscript)
                        + "}$"
                    )
                else:
                    legend_label_list_te.append("$Obs_{TM}$")
            else:
                pass
            # --------------------plot phase--------------------------------
            # cut out missing data points first
            pxy = np.where(d1.phase_te[0] != 0)[0]

            # --> TE mode data
            if len(pxy) > 0:
                self.axp.errorbar(
                    1.0 / d1.freq[pxy],
                    d1.phase_te[0][pxy],
                    ls=self.ls,
                    marker=self.mted,
                    ms=self.ms,
                    mfc=self.cted,
                    mec=self.cted,
                    color=self.cted,
                    yerr=d1.phase_te[1][pxy],
                    ecolor=self.cted,
                    picker=1,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
            else:
                pass
        # --> plot tm data------------------------------------------------------
        if self.data_tm_fn is not None:
            d1 = Data()
            d1.read_data_file(self.data_tm_fn)

            ryx = np.where(d1.res_tm[0] != 0)[0]

            # --> TM mode data
            if len(ryx) > 0:
                rtm = self.axr.errorbar(
                    1.0 / d1.freq[ryx],
                    d1.res_tm[0][ryx],
                    ls=self.ls,
                    marker=self.mtmd,
                    ms=self.ms,
                    mfc=self.ctmd,
                    mec=self.ctmd,
                    color=self.ctmd,
                    yerr=d1.res_tm[1][ryx],
                    ecolor=self.ctmd,
                    picker=2,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
                legend_marker_list_tm.append(rtm[0])
                if self.override_legend_subscript is not None:
                    legend_label_list_tm.append(
                        "$Obs_{"
                        + str.upper(self.override_legend_subscript)
                        + "}$"
                    )
                else:
                    legend_label_list_te.append("$Obs_{TM}$")
            else:
                pass

                # --------------------plot phase--------------------------------
            # cut out missing data points first
            pyx = np.where(d1.phase_tm[0] != 0)[0]

            # --> TM mode data
            if len(pyx) > 0:
                self.axp.errorbar(
                    1.0 / d1.freq[pyx],
                    d1.phase_tm[0][pyx],
                    ls=self.ls,
                    marker=self.mtmd,
                    ms=self.ms,
                    mfc=self.ctmd,
                    mec=self.ctmd,
                    color=self.ctmd,
                    yerr=d1.phase_tm[1][pyx],
                    ecolor=self.ctmd,
                    picker=1,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
            else:
                pass

        # --> plot model apparent resistivity and phase-------------------------
        nr = len(self.resp_te_fn)
        for rr, rfn in enumerate(self.resp_te_fn):
            if rfn is None:
                break
            # accommodate larger number of iterations that might have > 2 digits
            itnum = rfn[-8:-5]
            while not str.isdigit(itnum[0]):
                itnum = itnum[1:]
                if itnum == "":
                    break
            if self.color_mode == "color":
                cxy = (0, 0.4 + float(rr) / (3 * nr), 0)
            elif self.color_mode == "bw":
                cxy = (
                    1 - 1.25 / (rr + 2.0),
                    1 - 1.25 / (rr + 2.0),
                    1 - 1.25 / (rr + 2.0),
                )

            d1 = Data()

            d1.read_resp_file(rfn, data_fn=self.data_te_fn)

            # get non zero data
            rxy = np.where(d1.res_te[2] != 0)[0]

            # --> TE mode Data
            if len(rxy) > 0:
                rte = self.axr.errorbar(
                    1.0 / d1.freq[rxy],
                    d1.res_te[2][rxy],
                    ls=self.ls,
                    marker=self.mtem,
                    ms=self.ms,
                    mfc=cxy,
                    mec=cxy,
                    color=cxy,
                    yerr=None,
                    ecolor=cxy,
                    picker=2,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
                legend_marker_list_te.append(rte[0])
                if self.override_legend_subscript is not None:
                    legend_label_list_tm.append(
                        "$Mod_{"
                        + str.upper(self.override_legend_subscript)
                        + "}$"
                        + itnum
                    )
                else:
                    legend_label_list_te.append("$Mod_{TE}$" + itnum)
            else:
                pass

            # --------------------plot phase--------------------------------
            # cut out missing data points first
            # --> data
            pxy = np.where(d1.phase_te[2] != 0)[0]

            # --> TE mode phase
            if len(pxy) > 0:
                self.axp.errorbar(
                    1.0 / d1.freq[pxy],
                    d1.phase_te[2][pxy],
                    ls=self.ls,
                    marker=self.mtem,
                    ms=self.ms,
                    mfc=cxy,
                    mec=cxy,
                    color=cxy,
                    yerr=None,
                    ecolor=cxy,
                    picker=1,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
            else:
                pass
        # ---------------plot TM model response---------------------------------
        nr = len(self.resp_tm_fn)
        for rr, rfn in enumerate(self.resp_tm_fn):
            if rfn is None:
                break
            # accommodate larger number of iterations that might have > 2 digits
            itnum = rfn[-8:-5]
            while not str.isdigit(itnum[0]):
                itnum = itnum[1:]
                if itnum == "":
                    break
            if self.color_mode == "color":
                cyx = (
                    0.7 + float(rr) / (4 * nr),
                    0.13,
                    0.63 - float(rr) / (4 * nr),
                )
            elif self.color_mode == "bw":
                cyx = (
                    1 - 1.25 / (rr + 2.0),
                    1 - 1.25 / (rr + 2.0),
                    1 - 1.25 / (rr + 2.0),
                )
            d1 = Data()

            d1.read_resp_file(rfn, data_fn=self.data_tm_fn)
            ryx = np.where(d1.res_tm[2] != 0)[0]
            # --> TM mode model
            if len(ryx) > 0:
                rtm = self.axr.errorbar(
                    1.0 / d1.freq[ryx],
                    d1.res_tm[2][ryx],
                    ls=self.ls,
                    marker=self.mtmm,
                    ms=self.ms,
                    mfc=cyx,
                    mec=cyx,
                    color=cyx,
                    yerr=None,
                    ecolor=cyx,
                    picker=2,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
                legend_marker_list_tm.append(rtm[0])
                if self.override_legend_subscript is not None:
                    legend_label_list_tm.append(
                        "$Mod_{"
                        + str.upper(self.override_legend_subscript)
                        + "}$"
                        + itnum
                    )
                else:
                    legend_label_list_te.append("$Mod_{TM}$" + itnum)
            else:
                pass

            pyx = np.where(d1.phase_tm[2] != 0)[0]

            # --> TM mode model
            if len(pyx) > 0:
                self.axp.errorbar(
                    1.0 / d1.freq[pyx],
                    d1.phase_tm[0][pyx],
                    ls=self.ls,
                    marker=self.mtmm,
                    ms=self.ms,
                    mfc=cyx,
                    mec=cyx,
                    color=cyx,
                    yerr=None,
                    ecolor=cyx,
                    picker=1,
                    lw=self.lw,
                    elinewidth=self.lw,
                    capsize=self.e_capsize,
                    capthick=self.e_capthick,
                )
            else:
                pass

        # --> set axis properties-----------------------------------------------
        self.axr.set_xscale("log", nonposx="clip")
        self.axp.set_xscale("log", nonposx="clip")
        self.axr.set_yscale("log", nonposy="clip")
        self.axr.grid(
            True, alpha=self.grid_alpha, which="both", color=self.grid_color
        )
        plt.setp(self.axr.xaxis.get_ticklabels(), visible=False)
        self.axp.grid(
            True, alpha=self.grid_alpha, which="both", color=self.grid_color
        )
        self.axp.yaxis.set_major_locator(
            MultipleLocator(self.phase_major_ticks)
        )
        self.axp.yaxis.set_minor_locator(
            MultipleLocator(self.phase_minor_ticks)
        )

        if self.res_limits is not None:
            self.axr.set_ylim(self.res_limits)

        self.axp.set_ylim(self.phase_limits)
        self.axr.set_ylabel(
            "App. Res. ($\Omega \cdot m$)",
            fontdict={"size": self.font_size, "weight": "bold"},
        )
        self.axp.set_ylabel(
            "Phase (deg)", fontdict={"size": self.font_size, "weight": "bold"}
        )
        self.axp.set_xlabel(
            "Period (s)", fontdict={"size": self.font_size, "weight": "bold"}
        )
        plt.suptitle(
            self.title_str, fontsize=self.font_size + 2, fontweight="bold"
        )
        if legend_marker_list_te == [] or legend_marker_list_tm == []:
            num_col = 1
        else:
            num_col = 2
        self.axr.legend(
            legend_marker_list_te + legend_marker_list_tm,
            legend_label_list_te + legend_label_list_tm,
            loc=2,
            markerscale=1,
            borderaxespad=0.05,
            labelspacing=0.08,
            handletextpad=0.15,
            borderpad=0.05,
            ncol=num_col,
            prop={"size": self.font_size + 1},
        )

        # --> plot depth model--------------------------------------------------
        if self.model_fn is not None:
            # put axis labels on the right side for clarity
            self.axm.yaxis.set_label_position("right")
            self.axm.yaxis.set_tick_params(
                left="off", right="on", labelright="on"
            )
            self.axm.yaxis.tick_right()

            if self.depth_units == "km":
                dscale = 1000.0
            else:
                dscale = 1.0

            # --> plot te models
            nr = len(self.iter_te_fn)
            for ii, ifn in enumerate(self.iter_te_fn):
                if ifn is None:
                    break
                if self.color_mode == "color":
                    cxy = (0, 0.4 + float(ii) / (3 * nr), 0)
                elif self.color_mode == "bw":
                    cxy = (
                        1 - 1.25 / (ii + 2.0),
                        1 - 1.25 / (ii + 2.0),
                        1 - 1.25 / (ii + 2.0),
                    )
                m1 = Model()
                m1.read_iter_file(ifn, self.model_fn)
                plot_depth = m1.model_depth[1:] / dscale
                plot_model = abs(10 ** m1.model_res[1:, 1])
                self.axm.semilogx(
                    plot_model[::-1],
                    plot_depth[::-1],
                    ls="-",
                    color=cxy,
                    lw=self.lw,
                )

            # --> plot TM models
            nr = len(self.iter_tm_fn)
            for ii, ifn in enumerate(self.iter_tm_fn):
                if ifn is None:
                    break
                if self.color_mode == "color":
                    cyx = (
                        0.7 + float(ii) / (4 * nr),
                        0.13,
                        0.63 - float(ii) / (4 * nr),
                    )
                elif self.color_mode == "bw":
                    cyx = (
                        1 - 1.25 / (ii + 2.0),
                        1 - 1.25 / (ii + 2.0),
                        1 - 1.25 / (ii + 2.0),
                    )
                m1 = Model()
                m1.read_iter_file(ifn, self.model_fn)
                plot_depth = m1.model_depth[1:] / dscale
                plot_model = abs(10 ** m1.model_res[1:, 1])
                self.axm.semilogx(
                    plot_model[::-1],
                    plot_depth[::-1],
                    ls="steps-",
                    color=cyx,
                    lw=self.lw,
                )

            m1 = Model()
            m1.read_model_file(self.model_fn)
            if self.depth_limits is None:
                dmin = min(plot_depth)
                if dmin == 0:
                    dmin = 1
                dmax = max(plot_depth)
                self.depth_limits = (dmin, dmax)

            self.axm.set_ylim(
                ymin=max(self.depth_limits), ymax=min(self.depth_limits)
            )
            if self.depth_scale == "log":
                self.axm.set_yscale("log", nonposy="clip")
            self.axm.set_ylabel(
                f"Depth ({self.depth_units})",
                fontdict={"size": self.font_size, "weight": "bold"},
            )
            self.axm.set_xlabel(
                "Resistivity ($\Omega \cdot m$)",
                fontdict={"size": self.font_size, "weight": "bold"},
            )
            self.axm.grid(True, which="both", alpha=0.25)

        plt.show()

    def redraw_plot(self):
        """Redraw plot if parameters were changed

        use this function if you updated some attributes and want to re-plot.

        :Example: ::

            >>> # change the color and marker of the xy components
            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Occam2DData(r"/home/occam2d/Data.dat")
            >>> p1 = ocd.plotAllResponses()
            >>> #change line width
            >>> p1.lw = 2
            >>> p1.redraw_plot().
        """
        plt.close(self.fig)
        self.plot()

    def update_plot(self, fig):
        """Update any parameters that where changed using the built-in draw from
        canvas.

        Use this if you change an of the .fig or axes properties

        :Example: ::

            >>> # to change the grid lines to only be on the major ticks
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotAllResponses()
            >>> [ax.grid(True, which='major') for ax in [ps1.axrte,ps1.axtep]]
            >>> ps1.update_plot()
        """

        fig.canvas.draw()

    def save_figure(
        self,
        save_fn,
        file_format="pdf",
        orientation="portrait",
        fig_dpi=None,
        close_plot="y",
    ):
        """Save_plot will save the figure to save_fn.

        Arguments::

                **save_fn** : string
                              full path to save figure to, can be input as
                              * directory path -> the directory path to save to
                                in which the file will be saved as
                                save_fn/station_name_PhaseTensor.file_format

                              * full path -> file will be save to the given
                                path.  If you use this option then the format
                                will be assumed to be provided by the path

                **file_format** : [ pdf | eps | jpg | png | svg ]
                                  file type of saved figure pdf,svg,eps...

                **orientation** : [ landscape | portrait ]
                                  orientation in which the file will be saved
                                  *default* is portrait

                **fig_dpi** : int
                              The resolution in dots-per-inch the file will be
                              saved.  If None then the dpi will be that at
                              which the figure was made.  I don't think that
                              it can be larger than dpi of the figure.

                **close_plot** : [ y | n ]
                                 * 'y' will close the plot after saving.
                                 * 'n' will leave plot open

            :Example: ::

                >>> # to save plot as jpg
                >>> import mtpy.modeling.occam2d as occam2d
                >>> dfn = r"/home/occam2d/Inv1/data.dat"
                >>> ocd = occam2d.Occam2DData(dfn)
                >>> ps1 = ocd.plotPseudoSection()
                >>> ps1.save_plot(r'/home/MT/figures', file_format='jpg')
        """

        if fig_dpi is None:
            fig_dpi = self.fig_dpi

        if not os.path.isdir(save_fn):
            file_format = save_fn[-3:]
            self.fig.savefig(
                save_fn,
                dpi=fig_dpi,
                format=file_format,
                orientation=orientation,
                bbox_inches="tight",
            )

        else:
            save_fn = os.path.join(save_fn, "Occam1d." + file_format)
            self.fig.savefig(
                save_fn,
                dpi=fig_dpi,
                format=file_format,
                orientation=orientation,
                bbox_inches="tight",
            )

        if close_plot == "y":
            plt.clf()
            plt.close(self.fig)

        else:
            pass

        self.fig_fn = save_fn
        print("Saved figure to: " + self.fig_fn)

    def __str__(self):
        """Rewrite the string builtin to give a useful message."""

        return "Plots model responses and model for 1D occam inversion"
