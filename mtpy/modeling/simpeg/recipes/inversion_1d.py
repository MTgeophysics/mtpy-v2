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

from mtpy.core import MTDataFrame
from mtpy.imaging.mtplot_tools.plotters import plot_errorbar

from discretize import TensorMesh
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================


class Simpeg1D:
    """
    Run a 1D simpeg inversion

    """

    def __init__(self, mt_dataframe=None, **kwargs):

        self._acceptable_modes = ["te" "tm", "det"]
        self.mt_dataframe = MTDataFrame(data=mt_dataframe)

        self.mode = "det"
        self.dz = 5
        self.n_layers = 50
        self.z_factor = 1.2
        self.rho_initial = 100
        self.rho_reference = 100

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._sub_df = self._get_sub_dataframe()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):

        if mode not in self._acceptable_modes:
            raise ValueError(
                f"Mode {mode} not in accetable modes {self._acceptable_modes}"
            )
        self._mode = mode
        self._get_sub_dataframe()

    @property
    def thicknesses(self):
        return self.dz * self.z_factor ** np.arange(self.n_layers)[::-1]

    @property
    def mesh(self):
        return TensorMesh(
            [(np.r_[self.thicknesses, self.thicknesses[-1]])], "N"
        )

    @property
    def frequencies(self):
        return self._sub_df.frequency

    @property
    def periods(self):
        return 1.0 / self.frequencies

    def _get_sub_dataframe(self):
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
                    "phase": self.mt_dataframe.dataframe.phase_yx,
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
        sub_df = self._remove_zeros(sub_df)

        return sub_df

    def _remove_outofquadrant_phase(self, sub_df):
        """
        remove out of quadrant phase from data
        """

        sub_df.loc[(sub_df.phase % 180 < 0), "phase"] = 0

        return sub_df

    def _remove_zeros(self, sub_df):
        """
        remove zeros from the data frame

        :param sub_df: DESCRIPTION
        :type sub_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        sub_df.loc[(sub_df != 0).any(axis=1)]
        return sub_df

    @property
    def data(self):
        return np.c_[self._sub_df.res, self._sub_df.phase].flatten()

    @property
    def data_error(self):
        return np.c_[
            self._sub_df.res_error, self._sub_df.phase_error
        ].flatten()

    def run_fixed_layer_inversion(
        self,
        maxIter=10,
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
    ):

        receivers_list = [
            nsem.receivers.PointNaturalSource(component="app_res"),
            nsem.receivers.PointNaturalSource(component="phase"),
        ]

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
        # Define the data
        data_object = data.Data(
            survey, dobs=self.data, standard_deviation=self.data_error
        )

        # Initial model
        m0 = np.ones(self.n_layers + 1) * np.log(1.0 / self.rho_initial)

        # Reference model
        mref = np.ones(self.n_layers + 1) * np.log(1.0 / self.rho_reference)

        dmis = data_misfit.L2DataMisfit(
            simulation=simulation, data=data_object
        )

        # Define the regularization (model objective function)
        reg = regularization.Sparse(
            self.mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_z,
            mref=mref,
            mapping=maps.IdentityMap(self.mesh),
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
            reg.norms = np.c_[p_s, p_z]
            # Reach target misfit for L2 solution, then use IRLS until model stops changing.
            IRLS = directives.Update_IRLS(
                max_irls_iterations=40, minGNiter=1, f_min_change=1e-5
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

    def plot_model_fitting(self, scale="log"):
        """
        plot predicted vs model

        :return: DESCRIPTION
        :rtype: TYPE

        """

        target_misfit = self.data.size / 2.0
        iterations = list(self.output_dict.keys())
        n_iteration = len(iterations)
        phi_ds = np.zeros(n_iteration)
        phi_ms = np.zeros(n_iteration)
        betas = np.zeros(n_iteration)
        for ii, iteration in enumerate(iterations):
            phi_ds[ii] = self.output_dict[iteration]["phi_d"]
            phi_ms[ii] = self.output_dict[iteration]["phi_m"]
            betas[ii] = self.output_dict[iteration]["beta"]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.plot(phi_ms, phi_ds)
        ax.plot(phi_ms[iteration - 1], phi_ds[iteration - 1], "ro")
        ax.set_xlabel("$\phi_m$")
        ax.set_ylabel("$\phi_d$")
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        xlim = ax.get_xlim()
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
        z_grid = np.r_[0.0, np.cumsum(self.thicknesses[::-1])]
        return z_grid / 1000

    def plot_response(self, iteration=None):
        """
        plot response

        :param iteration: DESCRIPTION, defaults to None
        :type iteration: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if iteration is None:
            iteration = sorted(list(self.output_dict.keys()))[-1]

        dpred = self.output_dict[iteration]["dpred"]
        m = self.output_dict[iteration]["m"]

        fig = plt.figure(figsize=(10, 6), dpi=200)
        gs = gridspec.GridSpec(
            1, 5, figure=fig, wspace=0.4, hspace=0.4, left=0.08, right=0.91
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.loglog(
            (1.0 / (np.exp(m))),
            self._plot_z,
            color="k",
            **{"linestyle": "-"},
        )

        # ax0.legend()
        ax0.set_xlabel("Resistivity ($\Omega$m)")
        ax0.grid(which="both", alpha=0.5)
        ax0.set_ylim((self.thicknesses.sum() / 1000, 0.001))
        ax0.set_ylabel("Depth (km)")

        nf = len(self.frequencies)

        ax = fig.add_subplot(gs[0, 1:])
        ax.set_xscale("log")
        ax.set_yscale("log")
        eb_res = plot_errorbar(
            ax,
            self.periods,
            self.data.reshape((nf, 2))[:, 0],
            self.data_error.reshape((nf, 2))[:, 0],
            **{
                "marker": "s",
                "ms": 2.5,
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
            ax,
            self.periods,
            dpred.reshape((nf, 2))[:, 0],
            **{
                "marker": "s",
                "ms": 2.5,
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

        ax_1 = ax.twinx()
        eb_phase = plot_errorbar(
            ax_1,
            self.periods,
            self.data.reshape((nf, 2))[:, 1],
            self.data_error.reshape((nf, 2))[:, 1],
            **{
                "marker": "o",
                "ms": 2.5,
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
            ax_1,
            self.periods,
            dpred.reshape((nf, 2))[:, 1],
            **{
                "marker": "o",
                "ms": 2.5,
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

        ax.legend(
            [eb_res, eb_phase, eb_res_m, eb_phase_m],
            ["Res_data", "Phase_data", "Res_m", "Phase_m"],
            ncol=2,
        )

        # ax.set_xlabel("Period (s)")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_ylabel("Apparent resistivity ($\Omega$m)")
        ax_1.set_ylabel("Phase ($\degree$)")
        ax.set_xlabel("Period(s)")
        # ax.legend(loc=2)
        # ax_1.legend(loc=1)
        # ax.set_ylim(1, 10000)
        # ax_1.set_ylim(0, 90)
        # ax0.set_xlim(1, 10000)
        plt.show()

        return fig
