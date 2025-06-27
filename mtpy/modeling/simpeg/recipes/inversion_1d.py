# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:58:59 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from loguru import logger

from mtpy.core import MTDataFrame
from mtpy.imaging.mtplot_tools.plotters import plot_errorbar

try:
    from discretize import TensorMesh
    from simpeg.electromagnetics import natural_source as nsem
    from simpeg import (
        maps,
        data,
        data_misfit,
        regularization,
        optimization,
        inverse_problem,
        inversion,
        directives,
    )
except ImportError:
    logger.warning("Could not import Simpeg.")

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator

# =============================================================================


class Simpeg1D:
    """Run a 1D simpeg inversion."""

    def __init__(self, mt_dataframe=None, **kwargs):
        self._acceptable_modes = ["te", "tm", "det"]
        self.mt_dataframe = MTDataFrame(data=mt_dataframe)

        self.mode = "det"
        self.dz = 5
        self.n_layers = 50
        self.z_factor = 1.2
        self.rho_initial = 100
        self.rho_reference = 100
        self.output_dict = None
        self.max_iterations = 40

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._sub_df = self._get_sub_dataframe()

    @property
    def mode(self):
        """Mode function."""
        return self._mode

    @mode.setter
    def mode(self, mode):
        """Mode function."""
        if mode not in self._acceptable_modes:
            raise ValueError(
                f"Mode {mode} not in accetable modes {self._acceptable_modes}"
            )
        self._mode = mode
        self._get_sub_dataframe()

    @property
    def thicknesses(self):
        """Thicknesses function."""
        return self.dz * self.z_factor ** np.arange(self.n_layers)[::-1]

    @property
    def mesh(self):
        """Mesh function."""
        return TensorMesh([np.r_[self.thicknesses, self.thicknesses[-1]]], "N")

    @property
    def frequencies(self):
        """Frequencies function."""
        return self._sub_df.frequency

    @property
    def periods(self):
        """Periods function."""
        return 1.0 / self.frequencies

    def _get_sub_dataframe(self):
        """Get sub dataframe."""
        if self._mode == "te":
            sub_df = pd.DataFrame(
                {
                    "frequency": self.mt_dataframe.frequency,
                    "res": self.mt_dataframe.dataframe.res_xy,
                    "res_error": self.mt_dataframe.dataframe.res_xy_model_error,
                    "phase": self.mt_dataframe.dataframe.phase_xy,
                    "phase_error": self.mt_dataframe.dataframe.phase_xy_model_error,
                }
            )

        elif self._mode == "tm":
            sub_df = pd.DataFrame(
                {
                    "frequency": self.mt_dataframe.frequency,
                    "res": self.mt_dataframe.dataframe.res_yx,
                    "res_error": self.mt_dataframe.dataframe.res_yx_model_error,
                    "phase": self.mt_dataframe.dataframe.phase_yx % 180,
                    "phase_error": self.mt_dataframe.dataframe.phase_yx_model_error,
                }
            )

        elif self._mode == "det":
            z_obj = self.mt_dataframe.to_z_object()

            sub_df = pd.DataFrame(
                {
                    "frequency": self.mt_dataframe.frequency,
                    "res": z_obj.res_det,
                    "res_error": z_obj.res_model_error_det,
                    "phase": z_obj.phase_det,
                    "phase_error": z_obj.phase_model_error_det,
                }
            )

        sub_df = sub_df.sort_values("frequency", ascending=False).reindex()
        sub_df = self._remove_outofquadrant_phase(sub_df)
        sub_df = self._remove_nan_in_errors(sub_df)
        sub_df = self._remove_zeros(sub_df)

        return sub_df

    def _remove_outofquadrant_phase(self, sub_df):
        """Remove out of quadrant phase from data."""

        sub_df.loc[(sub_df.phase % 180 < 0), "phase"] = 0

        return sub_df

    def _remove_nan_in_errors(self, sub_df, large_error=1e2):
        """Remove nans in error."""
        sub_df["res_error"] = sub_df.res_error.fillna(large_error)
        sub_df["phase_error"] = sub_df.phase_error.fillna(large_error)
        return sub_df

    def _remove_zeros(self, sub_df):
        """Remove zeros from the data frame.
        :param sub_df: DESCRIPTION.
        :type sub_df: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        return sub_df.loc[(sub_df != 0).all(axis=1)]

    def cull_from_difference(self, sub_df, max_diff_res=1.0, max_diff_phase=10):
        """
        Remove points based on a simple difference between neighboring points

        uses np.diff

        res difference is in log space.
        :param sub_df:
        :param max_diff_res: DESCRIPTION, defaults to 1.0.
        :type max_diff_res: TYPE, optional
        :param max_diff_phase: DESCRIPTION, defaults to 10.
        :type max_diff_phase: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        sub_df.phase[
            np.where(abs(np.diff(sub_df.phase)) > max_diff_phase)[0] + 1
        ] = 0
        sub_df.phase[
            np.where(abs(np.diff(sub_df.phase)) > max_diff_phase)[0] + 1
        ] = 0
        sub_df.res[
            np.where(np.log10(abs(np.diff(sub_df.res))) > max_diff_res)[0] + 1
        ] = 0
        sub_df.res[
            np.where(np.log10(abs(np.diff(sub_df.res))) > max_diff_res)[0] + 1
        ] = 0

        return sub_df.loc[(sub_df != 0).all(axis=1)]

    def cull_from_interpolated(self, sub_df, tolerance=0.1, s_factor=2):
        """
        create a cubic spline as a smooth version of the data and then
        find points a certain distance away to remove.

        :param : DESCRIPTION
        :type : TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        from scipy import interpolate

        spline_res = interpolate.splrep(
            sub_df.period,
            sub_df.resistivity,
            s=s_factor * len(sub_df.period),
        )

        bad_res = np.where(
            abs(
                interpolate.splev(sub_df.period, spline_res)
                - sub_df.resistivity
            )
            > tolerance
        )

    def cull_from_model(self, iteration):
        """Remove bad point based on initial run.
        :param iteration: DESCRIPTION.
        :type iteration: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.output_dict is None:
            raise ValueError("Must run an inversion first")
        pass

    @property
    def data(self):
        """Data function."""
        return np.c_[self._sub_df.res, self._sub_df.phase].flatten()

    @property
    def data_error(self):
        """Data error."""
        return np.c_[self._sub_df.res_error, self._sub_df.phase_error].flatten()

    def run_fixed_layer_inversion(
        self,
        cull_from_difference=False,
        maxIter=40,
        maxIterCG=30,
        alpha_s=1e-10,
        alpha_z=1,
        beta0_ratio=1,
        coolingFactor=2,
        coolingRate=1,
        chi_factor=1,
        use_irls=False,
        p_s=2,
        p_z=2,
        **kwargs,
    ):
        """Run fixed layer inversion."""
        receivers_list = [
            nsem.receivers.PointNaturalSource(component="app_res"),
            nsem.receivers.PointNaturalSource(component="phase"),
        ]

        # Cull the data
        if cull_from_difference:
            self._sub_df = self.cull_from_difference(self._sub_df)

        source_list = []
        for freq in self.frequencies:
            source_list.append(nsem.sources.Planewave(receivers_list, freq))

        survey = nsem.survey.Survey(source_list)

        sigma_map = maps.ExpMap(nP=self.n_layers + 1)
        simulation = nsem.simulation_1d.Simulation1DRecursive(
            survey=survey,
            sigmaMap=sigma_map,
            thicknesses=self.thicknesses,
        )

        data_object = data.Data(
            survey, dobs=self.data, standard_deviation=self.data_error
        )

        # Initial model
        m0 = np.ones(self.n_layers + 1) * np.log(1.0 / self.rho_initial)

        # Reference model
        mref = np.ones(self.n_layers + 1) * np.log(1.0 / self.rho_reference)

        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

        # Define the regularization (model objective function)
        reg = regularization.Sparse(
            self.mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_z,
            reference_model=mref,
            mapping=maps.IdentityMap(self.mesh),
            norms=np.array([p_s, p_z]),
        )

        # Define how the optimization problem is solved. Here we will use an inexact
        # Gauss-Newton approach that employs the conjugate gradient solver.
        opt = optimization.InexactGaussNewton(
            maxIter=maxIter, maxIterCG=maxIterCG
        )

        # Define the inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #######################################################################
        # Define Inversion Directives
        # ---------------------------
        #
        # Here we define any directives that are carried out during the inversion. This
        # includes the cooling schedule for the trade-off parameter (beta), stopping
        # criteria for the inversion and saving inversion results at each iteration.
        #

        # Defining a starting value for the trade-off parameter (beta) between the data
        # misfit and the regularization.
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)

        # Set the rate of reduction in trade-off parameter (beta) each time the
        # the inverse problem is solved. And set the number of Gauss-Newton iterations
        # for each trade-off paramter value.
        beta_schedule = directives.BetaSchedule(
            coolingFactor=coolingFactor, coolingRate=coolingRate
        )
        save_dictionary = directives.SaveOutputDictEveryIteration()
        save_dictionary.outDict = {}
        # Setting a stopping criteria for the inversion.
        target_misfit = directives.TargetMisfit(chifact=chi_factor)
        if use_irls:
            reg.norms = [p_s, p_z]
            # Reach target misfit for L2 solution, then use IRLS until model stops changing.
            IRLS = directives.Update_IRLS(
                max_irls_iterations=maxIter, minGNiter=1, f_min_change=1e-5
            )

            # The directives are defined as a list.
            directives_list = [
                IRLS,
                starting_beta,
                save_dictionary,
            ]
        else:
            # The directives are defined as a list.
            directives_list = [
                starting_beta,
                beta_schedule,
                target_misfit,
                save_dictionary,
            ]

        #####################################################################
        # Running the Inversion
        # ---------------------
        #
        # To define the inversion object, we need to define the inversion problem and
        # the set of directives. We can then run the inversion.
        #

        # Here we combine the inverse problem and the set of directives
        inv = inversion.BaseInversion(inv_prob, directives_list)

        # Run the inversion
        recovered_model = inv.run(m0)

        self.output_dict = save_dictionary.outDict

    def plot_model_fitting(self, scale="log", fig_num=1):
        """Plot predicted vs model.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        target_misfit = self.data.size
        iterations = list(self.output_dict.keys())
        n_iteration = len(iterations)
        phi_ds = np.zeros(n_iteration)
        phi_ms = np.zeros(n_iteration)
        betas = np.zeros(n_iteration)
        for ii, iteration in enumerate(iterations):
            phi_ds[ii] = self.output_dict[iteration]["phi_d"]
            phi_ms[ii] = self.output_dict[iteration]["phi_m"]
            betas[ii] = self.output_dict[iteration]["beta"]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=fig_num, dpi=200)

        ax.plot(phi_ms, phi_ds, marker="o", mfc="w", color="r", ls=":", ms=10)
        for x, y, num in zip(phi_ms, phi_ds, range(len(phi_ms))):
            ax.text(x, y, num, ha="center", va="center")
        ax.set_xlabel("$\phi_m$")
        ax.set_ylabel("$\phi_d$")
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        xlim = ax.get_xlim()
        ax.grid(which="both", alpha=0.5)
        ax.plot(xlim, np.ones(2) * target_misfit, "--")
        ax.set_title(
            "Iteration={:d}, Beta = {:.1e}".format(
                iteration, betas[iteration - 1]
            )
        )
        ax.set_xlim(xlim)
        plt.show()

        return fig

    @property
    def _plot_z(self):
        """Plot z."""
        z_grid = np.r_[0.0, np.cumsum(self.thicknesses[::-1])]
        return z_grid / 1000

    def plot_response(self, iteration=None, fig_num=2):
        """Plot response.
        :param fig_num:
            Defaults to 2.
        :param iteration: DESCRIPTION, defaults to None.
        :type iteration: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if iteration is None:
            iteration = sorted(list(self.output_dict.keys()))[-1]

        dpred = self.output_dict[iteration]["dpred"]
        m = self.output_dict[iteration]["m"]

        fig = plt.figure(fig_num, figsize=(10, 6), dpi=200)
        gs = gridspec.GridSpec(
            3,
            5,
            figure=fig,
            wspace=0.6,
            hspace=0.25,
            left=0.08,
            right=0.98,
            top=0.98,
        )

        ax_model = fig.add_subplot(gs[:, 0])
        ax_model.step(
            (1.0 / (np.exp(m))),
            self._plot_z,
            color="k",
            **{"linestyle": "-"},
        )

        # ax_model.legend()
        ax_model.set_xlabel("Resistivity ($\Omega$m)")

        ax_model.set_ylim((self._plot_z.max(), 0.01))
        ax_model.set_ylabel("Depth (km)")
        ax_model.set_xscale("log")
        ax_model.set_yscale("symlog")
        ax_model.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax_model.yaxis.set_minor_locator(
            LogLocator(base=10.0, numticks=10, subs="auto")
        )
        ax_model.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax_model.xaxis.set_minor_locator(
            LogLocator(base=10.0, numticks=10, subs="auto")
        )
        ax_model.grid(which="both", alpha=0.5)

        nf = len(self.frequencies)

        ax_res = fig.add_subplot(gs[0:2, 1:])
        ax_res.set_xscale("log")
        ax_res.set_yscale("log")
        eb_res = plot_errorbar(
            ax_res,
            self.periods,
            self.data.reshape((nf, 2))[:, 0],
            self.data_error.reshape((nf, 2))[:, 0],
            **{
                "marker": "s",
                "ms": 5,
                "mew": 1,
                "mec": (0.25, 0.35, 0.75),
                "color": (0.25, 0.35, 0.75),
                "ecolor": (0.25, 0.35, 0.75),
                "ls": ":",
                "lw": 1,
                "capsize": 2.5,
                "capthick": 1,
            },
        )
        eb_res_m = plot_errorbar(
            ax_res,
            self.periods,
            dpred.reshape((nf, 2))[:, 0],
            **{
                "marker": "v",
                "ms": 4,
                "mew": 1,
                "mec": (0.25, 0.5, 0.5),
                "color": (0.25, 0.5, 0.5),
                "ecolor": (0.25, 0.5, 0.5),
                "ls": ":",
                "lw": 1,
                "capsize": 2.5,
                "capthick": 1,
            },
        )

        ax_res.grid(True, which="both", alpha=0.5)
        ax_res.set_ylabel("Apparent resistivity ($\Omega$m)")

        ax_phase = fig.add_subplot(gs[-1, 1:], sharex=ax_res)
        eb_phase = plot_errorbar(
            ax_phase,
            self.periods,
            self.data.reshape((nf, 2))[:, 1],
            self.data_error.reshape((nf, 2))[:, 1],
            **{
                "marker": "o",
                "ms": 5,
                "mew": 1,
                "mec": (0.75, 0.25, 0.25),
                "color": (0.75, 0.25, 0.25),
                "ecolor": (0.75, 0.25, 0.25),
                "ls": ":",
                "lw": 1,
                "capsize": 2.5,
                "capthick": 1,
            },
        )
        eb_phase_m = plot_errorbar(
            ax_phase,
            self.periods,
            dpred.reshape((nf, 2))[:, 1],
            **{
                "marker": "d",
                "ms": 4,
                "mew": 1,
                "mec": (0.9, 0.5, 0.05),
                "color": (0.9, 0.5, 0.05),
                "ecolor": (0.9, 0.5, 0.05),
                "ls": ":",
                "lw": 1,
                "capsize": 2.5,
                "capthick": 1,
            },
        )

        ax_res.legend(
            [eb_res, eb_phase, eb_res_m, eb_phase_m],
            ["Res_data", "Phase_data", "Res_m", "Phase_m"],
            ncol=2,
        )

        ax_phase.set_ylabel("Phase ($\degree$)")
        ax_phase.set_xlabel("Period(s)")
        ax_phase.grid(True, which="both", alpha=0.5)
        plt.show()

        return fig
