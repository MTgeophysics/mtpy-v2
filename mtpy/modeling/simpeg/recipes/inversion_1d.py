# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:58:59 2023

@author: jpeacock
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from mtpy.core import MTDataFrame
from mtpy.imaging.mtplot_tools.plotters import plot_errorbar

try:
    from discretize import TensorMesh
    from simpeg import (
        data,
        data_misfit,
        directives,
        inverse_problem,
        inversion,
        maps,
        optimization,
        regularization,
    )
    from simpeg.electromagnetics import natural_source as nsem
except ImportError:
    logger.warning("Could not import Simpeg.")

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator

# =============================================================================


class Simpeg1D:
    """Run a 1D SimPEG inversion for MT apparent resistivity and phase.

    Parameters
    ----------
    mt_dataframe : pandas.DataFrame or dict or None, optional
        Input tabular data that can be consumed by :class:`mtpy.core.MTDataFrame`.
    **kwargs : Any
        Optional overrides for instance attributes such as ``mode``,
        ``n_layers``, ``rho_initial``, etc.

    Notes
    -----
    For inversion, phase values are internally mapped to the SimPEG recursive
    1D convention ``[-180, -90]``. During plotting, phase values are converted
    to display convention ``[0, 90]``.

    Examples
    --------
    >>> inv = Simpeg1D(mt_dataframe=df, mode="tm", n_layers=60)
    >>> inv.run_fixed_layer_inversion(maxIter=30)
    >>> _ = inv.plot_response()
    """

    def __init__(
        self,
        mt_dataframe: pd.DataFrame | dict[str, Any] | None = None,
        resistivity_error: float = 10,
        phase_error: float = 2.5,
        **kwargs: Any,
    ) -> None:
        self._acceptable_modes: list[str] = ["te", "tm", "det"]
        self.mt_dataframe = MTDataFrame(data=mt_dataframe)
        self.resistivity_error: float = resistivity_error
        self.phase_error: float = phase_error
        self.mode: str = "det"
        self.dz: float = 5
        self.n_layers: int = 50
        self.z_factor: float = 1.2
        self.rho_initial: float = 100
        self.rho_reference: float = 100
        self.output_dict: dict[int, dict[str, Any]] | None = None
        self.max_iterations: int = 40

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._sub_df: pd.DataFrame = self._get_sub_dataframe()

    @property
    def mode(self) -> str:
        """Inversion mode.

        Returns
        -------
        str
            One of ``"te"``, ``"tm"``, or ``"det"``.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Set inversion mode.

        Parameters
        ----------
        mode : str
            Inversion mode. Must be one of ``"te"``, ``"tm"``, or ``"det"``.
        """
        if mode not in self._acceptable_modes:
            raise ValueError(
                f"Mode {mode} not in accetable modes {self._acceptable_modes}"
            )
        self._mode = mode
        self._get_sub_dataframe()

    @property
    def thicknesses(self) -> np.ndarray:
        """Layer thicknesses used in the recursive 1D simulation.

        Returns
        -------
        numpy.ndarray
            Layer thicknesses in meters ordered from top to bottom.
        """
        return self.dz * self.z_factor ** np.arange(self.n_layers)[::-1]

    @property
    def mesh(self) -> TensorMesh:
        """Regularization mesh.

        Returns
        -------
        discretize.TensorMesh
            1D tensor mesh used by SimPEG regularization.
        """
        return TensorMesh([np.r_[self.thicknesses, self.thicknesses[-1]]], "N")

    @property
    def frequencies(self) -> pd.Series:
        """Frequency series used by inversion.

        Returns
        -------
        pandas.Series
            Frequencies in Hz sorted high-to-low.
        """
        return self._sub_df.frequency

    @property
    def periods(self) -> pd.Series:
        """Period series used by inversion and plotting.

        Returns
        -------
        pandas.Series
            Periods in seconds.
        """
        return 1.0 / self.frequencies

    def _get_sub_dataframe(self) -> pd.DataFrame:
        """Build and clean the mode-specific inversion DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``frequency``, ``res``, ``res_error``,
            ``phase``, and ``phase_error``.
        """
        if self._mode == "te":
            self.mt_dataframe.dataframe["res_xy_model_error"] = (
                self.mt_dataframe.dataframe.res_xy * self.resistivity_error / 100
            )
            self.mt_dataframe.dataframe.loc[
                self.mt_dataframe.dataframe.res_xy_error
                > self.mt_dataframe.dataframe.res_xy_model_error,
                "res_xy_model_error",
            ] = self.mt_dataframe.dataframe.res_xy_error

            self.mt_dataframe.dataframe["phase_xy_model_error"] = self.phase_error
            self.mt_dataframe.dataframe.loc[
                self.mt_dataframe.dataframe.phase_xy_error
                > self.mt_dataframe.dataframe.phase_xy_model_error,
                "phase_xy_model_error",
            ] = self.mt_dataframe.dataframe.phase_xy_error

            sub_df = pd.DataFrame(
                {
                    "frequency": self.mt_dataframe.frequency,
                    "res": self.mt_dataframe.dataframe.res_xy,
                    "res_error": self.mt_dataframe.dataframe.res_xy_model_error,
                    "phase": self.mt_dataframe.dataframe.phase_xy - 180,
                    "phase_error": self.mt_dataframe.dataframe.phase_xy_model_error,
                }
            )

        elif self._mode == "tm":
            self.mt_dataframe.dataframe["res_yx_model_error"] = (
                self.mt_dataframe.dataframe.res_yx * self.resistivity_error / 100
            )
            self.mt_dataframe.dataframe.loc[
                self.mt_dataframe.dataframe.res_yx_error
                > self.mt_dataframe.dataframe.res_yx_model_error,
                "res_yx_model_error",
            ] = self.mt_dataframe.dataframe.res_yx_error

            self.mt_dataframe.dataframe["phase_yx_model_error"] = self.phase_error
            self.mt_dataframe.dataframe.loc[
                self.mt_dataframe.dataframe.phase_yx_error
                > self.mt_dataframe.dataframe.phase_yx_model_error,
                "phase_yx_model_error",
            ] = self.mt_dataframe.dataframe.phase_yx_error

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

            res_model_error = z_obj.res_det * self.resistivity_error / 100
            res_model_error[
                np.where(z_obj.res_model_error_det > res_model_error)[0]
            ] = z_obj.res_model_error_det[
                np.where(z_obj.res_model_error_det > res_model_error)[0]
            ]

            phase_model_error = np.ones_like(z_obj.phase_det) * self.phase_error
            # phase_model_error[
            #     np.where(z_obj.phase_model_error_det > phase_model_error)[0]
            # ] = z_obj.phase_model_error_det[
            #     np.where(z_obj.phase_model_error_det > phase_model_error)[0]
            # ]

            sub_df = pd.DataFrame(
                {
                    "frequency": self.mt_dataframe.frequency,
                    "res": z_obj.res_det,
                    "res_error": res_model_error,
                    "phase": z_obj.phase_det - 180,
                    "phase_error": phase_model_error,
                }
            )

        sub_df = sub_df.sort_values("frequency", ascending=False).reindex()
        sub_df = self._ensure_inversion_phase_convention(sub_df)
        sub_df = self._remove_nan_in_errors(sub_df)
        sub_df = self._remove_zeros(sub_df)

        return sub_df

    def _ensure_inversion_phase_convention(self, sub_df: pd.DataFrame) -> pd.DataFrame:
        """Map phase to the branch expected by recursive 1D SimPEG.

        Parameters
        ----------
        sub_df : pandas.DataFrame
            Mode-specific data table.

        Returns
        -------
        pandas.DataFrame
            Updated table with phase constrained to ``[-180, -90]``. Values
            outside the target branch are set to 0 and removed downstream.
        """

        phase = sub_df.phase.to_numpy(dtype=float)

        # Wrap to [-180, 180) first, then move first-quadrant values to their
        # equivalent third-quadrant representation by subtracting 180.
        phase = ((phase + 180.0) % 360.0) - 180.0
        phase = np.where(phase > 0.0, phase - 180.0, phase)

        # Keep only the target inversion phase branch. Out-of-band values are
        # marked as 0 and removed by _remove_zeros.
        in_band = (phase >= -180.0) & (phase <= -90.0)
        phase = np.where(in_band, phase, 0.0)

        sub_df["phase"] = phase

        return sub_df

    def _phase_for_plotting(self, phase_values: np.ndarray | pd.Series) -> np.ndarray:
        """Convert inversion phase convention ``[-180, -90]`` to display ``[0, 90]``.

        Parameters
        ----------
        phase_values : numpy.ndarray or pandas.Series
            Input phase values in degrees.

        Returns
        -------
        numpy.ndarray
            Phase values in the plotting convention ``[0, 90]``.
        """

        phase_values = np.asarray(phase_values, dtype=float)
        return np.where(phase_values < 0.0, phase_values + 180.0, phase_values)

    def _remove_nan_in_errors(
        self, sub_df: pd.DataFrame, large_error: float = 1e2
    ) -> pd.DataFrame:
        """Replace missing error values with a large fallback error.

        Parameters
        ----------
        sub_df : pandas.DataFrame
            Input data table.
        large_error : float, default=1e2
            Replacement value for missing ``res_error`` and ``phase_error``.

        Returns
        -------
        pandas.DataFrame
            Data table with missing error values filled.
        """
        sub_df["res_error"] = sub_df.res_error.fillna(large_error)
        sub_df["phase_error"] = sub_df.phase_error.fillna(large_error)
        return sub_df

    def _remove_zeros(self, sub_df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows containing zero values.

        Parameters
        ----------
        sub_df : pandas.DataFrame
            Input data table.

        Returns
        -------
        pandas.DataFrame
            Filtered table where all columns are non-zero.
        """
        return sub_df.loc[(sub_df != 0).all(axis=1)]

    def cull_from_difference(
        self,
        sub_df: pd.DataFrame,
        max_diff_res: float = 1.0,
        max_diff_phase: float = 10.0,
    ) -> pd.DataFrame:
        """Cull outliers using nearest-neighbor differences.

        Resistivity culling is done in log10-difference space.

        Parameters
        ----------
        sub_df : pandas.DataFrame
            Input data table.
        max_diff_res : float, default=1.0
            Maximum allowed neighboring difference in log10 resistivity.
        max_diff_phase : float, default=10.0
            Maximum allowed neighboring difference in phase (degrees).

        Returns
        -------
        pandas.DataFrame
            Filtered table after removing rows flagged as outliers.
        """

        sub_df.phase[np.where(abs(np.diff(sub_df.phase)) > max_diff_phase)[0] + 1] = 0
        sub_df.phase[np.where(abs(np.diff(sub_df.phase)) > max_diff_phase)[0] + 1] = 0
        sub_df.res[
            np.where(np.log10(abs(np.diff(sub_df.res))) > max_diff_res)[0] + 1
        ] = 0
        sub_df.res[
            np.where(np.log10(abs(np.diff(sub_df.res))) > max_diff_res)[0] + 1
        ] = 0

        return sub_df.loc[(sub_df != 0).all(axis=1)]

    def cull_from_interpolated(
        self, sub_df: pd.DataFrame, tolerance: float = 0.1, s_factor: float = 2
    ) -> None:
        """Prototype spline-based data culling method.

        Parameters
        ----------
        sub_df : pandas.DataFrame
            Input data table.
        tolerance : float, default=0.1
            Allowed absolute residual from spline prediction.
        s_factor : float, default=2
            Smoothing multiplier applied to the spline fit.

        Notes
        -----
        This method is currently incomplete and does not return filtered data.
        """

        from scipy import interpolate

        spline_res = interpolate.splrep(
            sub_df.period,
            sub_df.resistivity,
            s=s_factor * len(sub_df.period),
        )

        bad_res = np.where(
            abs(interpolate.splev(sub_df.period, spline_res) - sub_df.resistivity)
            > tolerance
        )

    def cull_from_model(self, iteration: int) -> None:
        """Placeholder for model-based culling logic.

        Parameters
        ----------
        iteration : int
            Inversion iteration number.
        """

        if self.output_dict is None:
            raise ValueError("Must run an inversion first")

    @property
    def data(self) -> np.ndarray:
        """Flattened inversion data vector.

        Returns
        -------
        numpy.ndarray
            Data vector ordered by ``[rho, phase]`` for each frequency.
        """
        return np.c_[self._sub_df.res, self._sub_df.phase].flatten()

    @property
    def data_error(self) -> np.ndarray:
        """Flattened data standard deviation vector.

        Returns
        -------
        numpy.ndarray
            Error vector ordered by ``[rho_error, phase_error]``.
        """
        return np.c_[self._sub_df.res_error, self._sub_df.phase_error].flatten()

    def run_fixed_layer_inversion(
        self,
        cull_from_difference: bool = False,
        maxIter: int = 40,
        maxIterCG: int = 30,
        alpha_s: float = 1e-10,
        alpha_z: float = 1,
        beta0_ratio: float = 1,
        coolingFactor: float = 2,
        coolingRate: int = 1,
        chi_factor: float = 1,
        use_irls: bool = False,
        p_s: float = 2,
        p_z: float = 2,
        **kwargs: Any,
    ) -> None:
        """Run a fixed-layer 1D inversion.

        Parameters
        ----------
        cull_from_difference : bool, default=False
            If ``True``, apply simple nearest-neighbor culling before inversion.
        maxIter : int, default=40
            Maximum Gauss-Newton iterations.
        maxIterCG : int, default=30
            Maximum conjugate-gradient iterations per GN step.
        alpha_s : float, default=1e-10
            Smallness regularization weight.
        alpha_z : float, default=1
            Vertical smoothness regularization weight.
        beta0_ratio : float, default=1
            Initial beta estimate scaling.
        coolingFactor : float, default=2
            Beta reduction factor.
        coolingRate : int, default=1
            Number of iterations between beta updates.
        chi_factor : float, default=1
            Target misfit factor.
        use_irls : bool, default=False
            If ``True``, enable IRLS regularization updates.
        p_s : float, default=2
            Smallness norm exponent.
        p_z : float, default=2
            Smoothness norm exponent.
        **kwargs : Any
            Reserved for future optional controls.

        Examples
        --------
        >>> inv = Simpeg1D(mt_dataframe=df, mode="tm")
        >>> inv.run_fixed_layer_inversion(maxIter=25, alpha_z=2.0)
        """

        receivers_list = [
            nsem.receivers.Impedance([[]], component="rho", orientation="yx"),
            nsem.receivers.Impedance([[]], component="phase", orientation="yx"),
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
        opt = optimization.InexactGaussNewton(maxIter=maxIter, maxIterCG=maxIterCG)

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
        _ = recovered_model

        self.output_dict = save_dictionary.outDict

    def plot_model_fitting(self, scale: str = "log", fig_num: int = 1) -> Figure:
        """Plot trade-off curve for inversion iterations.

        Parameters
        ----------
        scale : str, default="log"
            Axis scale. Common choice is ``"log"``.
        fig_num : int, default=1
            Matplotlib figure number.

        Returns
        -------
        matplotlib.figure.Figure
            Created matplotlib figure.

        Examples
        --------
        >>> inv.run_fixed_layer_inversion()
        >>> fig = inv.plot_model_fitting(scale="log")
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
        ax.set_xlabel(r"$\phi_m$")
        ax.set_ylabel(r"$\phi_d$")
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        xlim = ax.get_xlim()
        ax.grid(which="both", alpha=0.5)
        ax.plot(xlim, np.ones(2) * target_misfit, "--")
        ax.set_title(
            "Iteration={:d}, Beta = {:.1e}".format(iteration, betas[iteration - 1])
        )
        ax.set_xlim(xlim)
        plt.show()

        return fig

    @property
    def _plot_z(self) -> np.ndarray:
        """Depth axis used for model step plots.

        Returns
        -------
        numpy.ndarray
            Depths in kilometers.
        """
        z_grid = np.r_[0.0, np.cumsum(self.thicknesses[::-1])]
        return z_grid / 1000

    def plot_response(
        self, iteration: int | None = None, fig_num: int = 1, **kwargs: Any
    ) -> Figure:
        """Plot recovered 1D model and data fit.

        Parameters
        ----------
        iteration : int or None, default=None
            Iteration index to plot. If ``None``, uses the final iteration.
        fig_num : int, default=1
            Matplotlib figure number.
        **kwargs : Any
            Optional plotting controls such as ``y_limits`` and ``y_scale``.

        Returns
        -------
        matplotlib.figure.Figure
            Created matplotlib figure.

        Examples
        --------
        >>> inv.run_fixed_layer_inversion()
        >>> fig = inv.plot_response(y_scale="log")
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
        ax_model.set_xlabel(r"Resistivity ($\Omega$m)")
        y_limits = kwargs.get("y_limits", (self._plot_z.max(), 0.5))
        ax_model.set_ylim(y_limits)
        ax_model.set_ylabel("Depth (km)")
        ax_model.set_xscale("log")
        yscale = kwargs.get("y_scale", "symlog")
        ax_model.set_yscale(yscale)
        if "log" in yscale:
            ax_model.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax_model.yaxis.set_minor_locator(
                LogLocator(base=10.0, numticks=10, subs="auto")
            )
            ax_model.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax_model.xaxis.set_minor_locator(
                LogLocator(base=10.0, numticks=10, subs="auto")
            )
        else:
            pass
            # ax_model.yaxis.set_major_locator(
            #     plt.(integer=True, nbins=10)
            # )
            # ax_model.xaxis.set_major_locator(
            #     plt.MaxNLocator(integer=True, nbins=10)
            # )

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
        ax_res.set_ylabel(r"Apparent resistivity ($\Omega$m)")

        ax_phase = fig.add_subplot(gs[-1, 1:], sharex=ax_res)
        eb_phase = plot_errorbar(
            ax_phase,
            self.periods,
            self._phase_for_plotting(self.data.reshape((nf, 2))[:, 1]),
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
            self._phase_for_plotting(dpred.reshape((nf, 2))[:, 1]),
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

        ax_phase.set_ylabel(r"Phase ($\degree$)")
        ax_phase.set_xlabel("Period(s)")
        ax_phase.grid(True, which="both", alpha=0.5)
        plt.show()

        return fig
