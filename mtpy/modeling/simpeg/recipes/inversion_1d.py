# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:58:59 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np

from SimPEG.electromagnetics import natural_source as nsem

from discretize import TensorMesh
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

# =============================================================================


class Simpeg1D:
    """
    Run a 1D simpeg inversion

    """

    def __init__(self, **kwargs):

        self.dz = 50
        self.n_layers = 30
        self.z_factor = 1.2

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def depth_model(self):
        return self.dz * self.z_factor ** np.arange(self.n_layer)[::-1]

    @property
    def mesh(self):
        return TensorMesh(
            [(np.r_[self.depth_model, self.depth_model[-1]])], "N"
        )

    def run_fixed_layer_inversion(
        self,
        dobs,
        standard_deviation,
        rho_0,
        rho_ref,
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
        for freq in frequencies:
            source_list.append(nsem.sources.Planewave(receivers_list, freq))

        survey = nsem.survey.Survey(source_list)

        sigma_map = maps.ExpMap(nP=self.n_layers + 1)
        simulation = nsem.simulation_1d.Simulation1DRecursive(
            survey=survey,
            sigmaMap=sigma_map,
            thicknesses=layer_thicknesses_inv,
        )
        # Define the data
        data_object = data.Data(
            survey, dobs=dobs, standard_deviation=standard_deviation
        )

        # Initial model
        m0 = np.ones(len(layer_thicknesses_inv) + 1) * np.log(1.0 / rho_0)

        # Reference model
        mref = np.ones(len(layer_thicknesses_inv) + 1) * np.log(1.0 / rho_ref)

        dmis = data_misfit.L2DataMisfit(
            simulation=simulation, data=data_object
        )

        # Define the regularization (model objective function)
        reg = regularization.Sparse(
            mesh_inv,
            alpha_s=alpha_s,
            alpha_x=alpha_z,
            mref=mref,
            mapping=maps.IdentityMap(mesh_inv),
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

        return recovered_model, save_dictionary.outDict
