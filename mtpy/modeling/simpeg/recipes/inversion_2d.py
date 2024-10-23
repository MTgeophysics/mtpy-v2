# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:17:41 2024

@author: jpeacock

A vanilla recipe to invert 2D MT data.  

- For now the default is a quad tree mesh
- Optimization: Inexact Gauss Newton 
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from simpeg.electromagnetics import natural_source as nsem
from simpeg import (
    maps,
    optimization,
    inversion,
    inverse_problem,
    directives,
    data_misfit,
    regularization,
)

try:
    from pymatsolver import Pardiso

    pardiso_imported = True
except ImportError:
    warnings.warn(
        "Pardiso not installed see https://github.com/simpeg/pydiso/blob/main/README.md."
    )
    pardiso_imported = False

# from dask.distributed import Client, LocalCluster
from mtpy.modeling.simpeg.data_2d import Simpeg2DData
from mtpy.modeling.simpeg.make_2d_mesh import QuadTreeMesh, StructuredMesh

warnings.filterwarnings("ignore")


# =============================================================================


class Simpeg2D:
    """
    A vanilla recipe to invert 2D MT data.

    - For now the default is a quad tree mesh
    - Optimization: Inexact Gauss Newton
    - Regularization: Sparse

    # change mesh to tensor mesh.
    """

    def __init__(
        self,
        dataframe,
        data_kwargs={},
        mesh_kwargs={},
        mesh_type="tensor",
        **kwargs,
    ):
        self.data = Simpeg2DData(dataframe, **data_kwargs)
        if mesh_type in ["tensor"]:
            self.mesh = StructuredMesh(
                self.data.station_locations,
                self.data.frequencies,
                **mesh_kwargs,
            )
        elif mesh_type in ["quad", "tree", "quadtree"]:
            self.mesh = QuadTreeMesh(
                self.data.station_locations,
                self.data.frequencies,
                **mesh_kwargs,
            )
        else:
            raise ValueError(f"mesh {mesh_type} is unsupported.")

        self.mesh_type = mesh_type

        self.ax = self.make_mesh()
        self.air_conductivity = 1e-8
        self.initial_conductivity = 1e-2
        if pardiso_imported:
            self.solver = "pardiso"
            self._solvers_dict = {"pardiso": Pardiso}

        else:
            self.solver = None
            self._solvers_dict = {}

        # regularization parameters
        self.alpha_s = 1e-15
        self.alpha_y = 1
        self.alpha_z = 0.5

        # optimization parameters
        self.max_iterations = 30
        self.max_iterations_cg = 30
        self.max_iterations_irls = 40
        self.minimum_gauss_newton_iterations = 1
        self.f_min_change = 1e-5
        self.optimization_tolerance = 1e-30

        # inversion parameters
        self.use_irls = False
        self.p_s = 0
        self.p_y = 0
        self.p_z = 0
        self.beta_starting_ratio = 1
        self.beta_cooling_factor = 2
        self.beta_cooling_rate = 1

        self.target_misfit_chi_factor = 1

        self.saved_model_outputs = directives.SaveOutputDictEveryIteration()
        self.saved_model_outputs.outDict = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        # get attributes
        attr_list = []
        property_list = []
        method_list = []
        for key in dir(self):
            if key.startswith("_"):
                continue
            value = getattr(self, key)
            if isinstance(getattr(type(self), key, None), property):
                property_list.append(f"\t{key}: {value}")
            else:
                if isinstance(
                    value, (float, int, str, bool, list, tuple, dict)
                ):
                    attr_list.append(f"\t{key}: {value}")
                else:
                    method_list.append(f"\t{key}")

        return "\n".join(
            ["Attributes:"]
            + sorted(attr_list)
            + ["Properties:"]
            + sorted(property_list)
            + ["Methods:"]
            + sorted(method_list)
        )

        return "\n".join(
            ["Attributes"]
            + sorted(attr_list)
            + ["Properties"]
            + sorted(property_list)
            + ["Methods:"]
            + sorted(method_list)
        )

    def __repr__(self):
        return self.__str__()

    def make_mesh(self, **kwargs):
        """
        make QuadTree Mesh
        """
        ax = self.mesh.make_mesh(**kwargs)
        return ax

    @property
    def active_map(self):
        """
        Active cells mapping

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return maps.InjectActiveCells(
            self.mesh.mesh,
            self.mesh.active_cell_index,
            np.log(self.air_conductivity),
        )

    @property
    def exponent_map(self):
        """
        compute fields on an exponential mapping
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return maps.ExpMap(mesh=self.mesh.mesh)

    @property
    def conductivity_map(self):
        """
        conductivity mapping

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return self.exponent_map * self.active_map

    def _get_solver(self):
        """
        get solver
        """
        try:
            return self._solvers_dict[self.solver]
        except KeyError:
            return None

    @property
    def tm_simulation(self):
        """
        Simulation for TE Mode
        """
        solver = self._get_solver()
        if solver is not None:
            return nsem.simulation.Simulation2DElectricField(
                self.mesh.mesh,
                survey=self.data.tm_survey,
                sigmaMap=self.conductivity_map,
                solver=solver,
            )
        else:
            return nsem.simulation.Simulation2DElectricField(
                self.mesh.mesh,
                survey=self.data.tm_survey,
                sigmaMap=self.conductivity_map,
            )

    @property
    def te_simulation(self):
        """
        Simulation for TE Mode
        """
        solver = self._get_solver()
        if solver is not None:
            return nsem.simulation.Simulation2DMagneticField(
                self.mesh.mesh,
                survey=self.data.te_survey,
                sigmaMap=self.conductivity_map,
                solver=solver,
            )
        else:
            nsem.simulation.Simulation2DMagneticField(
                self.mesh.mesh,
                survey=self.data.te_survey,
                sigmaMap=self.conductivity_map,
            )

    @property
    def te_data_misfit(self):
        """
        data misfit of TE mode
        """

        return data_misfit.L2DataMisfit(
            data=self.data.te_data, simulation=self.te_simulation
        )

    @property
    def tm_data_misfit(self):
        """
        data misfit of TM mode
        """

        return data_misfit.L2DataMisfit(
            data=self.data.tm_data, simulation=self.tm_simulation
        )

    @property
    def data_misfit(self):
        """
        data misfit of all components TE + TM
        """
        return self.te_data_misfit + self.tm_data_misfit

    @property
    def reference_model(self):
        """
        reference model

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return np.ones(self.mesh.number_of_active_cells) * np.log(
            self.initial_conductivity
        )

    @property
    def regularization(self):
        """
        Create sparse regularization using paramaters

         - alpha_s = smallness parameter
         - alpha_y = smoothing in y direction
         - alpha_z = smoothing in z direction

        :return: DESCRIPTION
        :rtype: TYPE

        """

        reg = regularization.Sparse(
            self.mesh.mesh,
            active_cells=self.mesh.active_cell_index,
            reference_model=self.reference_model,
            alpha_s=self.alpha_s,
            alpha_x=self.alpha_y,
            alpha_y=self.alpha_z,
            mapping=maps.IdentityMap(nP=self.mesh.number_of_active_cells),
        )

        if self.use_irls:
            reg.norms = np.c_[self.p_s, self.p_y, self.p_z]

        return reg

    @property
    def optimization(self):
        """
        optimization algorithm

        default is InexactGaussNewton
        """

        return optimization.InexactGaussNewton(
            maxIter=self.max_iterations,
            maxIterCG=self.max_iterations_cg,
            tolX=self.optimization_tolerance,
        )

    @property
    def inverse_problem(self):
        """
        setup the inverse problem

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return inverse_problem.BaseInvProblem(
            self.data_misfit, self.regularization, self.optimization
        )

    @property
    def starting_beta(self):
        """
        set up the starting beta value

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return directives.BetaEstimate_ByEig(
            beta0_ratio=self.beta_starting_ratio
        )

    @property
    def beta_schedule(self):
        """
        how quickly beta is reduced

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return directives.BetaSchedule(
            coolingFactor=self.beta_cooling_factor,
            coolingRate=self.beta_cooling_rate,
        )

    @property
    def target_misfit(self):
        """
        target misfit

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return directives.TargetMisfit(chifact=self.target_misfit_chi_factor)

    @property
    def directives(self):
        """
        list of directives to supply to the inversion

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.use_irls:
            IRLS = directives.Update_IRLS(
                max_irls_iterations=self.max_iteration_irls,
                minGNiter=self.minimum_gauss_newton_iterations,
                f_min_change=self.f_min_change,
            )
            return [
                IRLS,
                self.starting_beta,
                self.saved_model_outputs,
            ]
        else:
            return [
                self.starting_beta,
                self.beta_schedule,
                self.saved_model_outputs,
                # self.target_misfit,
            ]

    def run_inversion(self):
        """
        run the inversion using the attributes as input.
        """

        mt_inversion = inversion.BaseInversion(
            self.inverse_problem, directiveList=self.directives
        )

        return mt_inversion.run(self.reference_model)

    @property
    def iterations(self):
        """
        return dictionary of model outputs
        """

        return self.saved_model_outputs.outDict

    def plot_iteration(self, iteration_number, resistivity=True, **kwargs):
        """ """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        m = self.iterations[iteration_number]["m"]

        sigma = np.ones(self.mesh.mesh.nC) * self.air_conductivity
        sigma[self.mesh.active_cell_index] = np.exp(m)
        if resistivity:
            sigma = 1.0 / sigma
            vmin = kwargs.get("vmin", 0.3)
            vmax = kwargs.get("vmax", 3000)
            cmap = kwargs.get("cmap", "turbo_r")
        else:
            vmin = kwargs.get("vmin", 1e-3)
            vmax = kwargs.get("vmax", 1)
            cmap = kwargs.get("cmap", "turbo")
        out = self.mesh.mesh.plot_image(
            sigma,
            grid=False,
            ax=ax,
            pcolor_opts={
                "norm": LogNorm(vmin=vmin, vmax=vmax),
                "cmap": cmap,
            },
            range_x=(
                self.data.station_locations[:, 0].min() - 5 * self.mesh.dx,
                self.data.station_locations[:, 0].max() + 5 * self.mesh.dx,
            ),
            range_y=kwargs.get(
                "z_limits",
                (-self.mesh.mesh.h[1].sum() / 2, 500),
            ),
        )
        cb = plt.colorbar(out[0], fraction=0.01, ax=ax)
        if resistivity:
            cb.set_label("Resistivity ($\Omega \cdot m$)")
        else:
            cb.set_label("Conductivity (S/m)")
        ax.set_aspect(1)
        ax.set_xlabel("Offset (m)")
        ax.set_ylabel("Elevation (m)")
        if self.mesh.topography:
            ax.scatter(
                self.data.station_locations[:, 0],
                self.data.station_locations[:, 1],
                marker="v",
                s=30,
                color="k",
            )
        else:
            ax.scatter(
                self.data.station_locations[:, 0],
                np.zeros_like(self.data.station_locations[:, 1]),
                marker="v",
                s=30,
                color="k",
            )

    def plot_tikhonov_curve(self):
        """
        plot L-like curve

        :return: DESCRIPTION
        :rtype: TYPE

        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            [
                self.iterations[iteration]["phi_m"]
                for iteration in self.iterations.keys()
            ],
            [
                self.iterations[iteration]["phi_d"]
                for iteration in self.iterations.keys()
            ],
            marker="o",
            ms=15,
            mec="k",
            mfc="r",
        )

        ax.set_xlabel("$\phi_m$ [model smallness]")
        ax.set_ylabel("$\phi_d$ [data fit]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        xlim = ax.get_xlim()
        ax.plot(
            xlim,
            np.ones(2)
            * (
                self.data.te_observations.size + self.data.tm_observations.size
            ),
            "--",
        )
        ax.set_xlim(xlim)
        plt.tight_layout()
        plt.show()

    def plot_responses(self, iteration_number, **kwargs):
        """
        Plot responses all together

        :param iteration: DESCRIPTION
        :type iteration: TYPE
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        shape = (self.data.n_frequencies, 2, self.data.n_stations)

        dpred = self.iterations[iteration_number]["dpred"]
        te_pred = dpred.reshape(
            (2, self.data.n_frequencies, 2, self.data.n_stations)
        )[0, :, :, :]
        tm_pred = dpred.reshape(
            (2, self.data.n_frequencies, 2, self.data.n_stations)
        )[1, :, :, :]

        te_obs = self.data.te_data.dobs.copy().reshape(shape)
        tm_obs = self.data.tm_data.dobs.copy().reshape(shape)

        ## With these plot frequency goes from high on the left to low on the right.
        ## Moving shallow to deep from left to right.

        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2, sharex=ax1)
        ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
        ax4 = fig.add_subplot(2, 2, 4, sharex=ax1)

        # plot TE Resistivity
        ax1.semilogy(te_obs[:, 0, :].flatten(), "b.", label="observed")
        ax1.semilogy(te_pred[:, 0, :].flatten(), "r.", label="predicted")
        ax1.set_title("TE")
        ax1.set_ylabel("Apparent Resistivity")
        ax1.set_xlim((self.data.n_stations * self.data.n_frequencies, 0))
        ax1.legend()

        # plot TM Resistivity
        ax2.semilogy(tm_obs[:, 0, :].flatten(), "b.", label="observed")
        ax2.semilogy(tm_pred[:, 0, :].flatten(), "r.", label="predicted")
        ax2.set_title("TM")
        ax2.legend()

        # plot TE Phase
        ax3.plot(te_obs[:, 1, :].flatten(), "b.", label="observed")
        ax3.plot(te_pred[:, 1, :].flatten(), "r.", label="predicted")
        ax3.set_xlabel("data point")
        ax3.legend()

        # plot TM Phase
        ax4.plot(tm_obs[:, 1, :].flatten(), "b.", label="observed")
        ax4.plot(tm_pred[:, 1, :].flatten(), "r.", label="predicted")
        ax3.legend()

        # plt.ylim(10, 1000)
