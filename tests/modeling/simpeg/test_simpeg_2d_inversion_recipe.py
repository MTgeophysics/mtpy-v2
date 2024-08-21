# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:09:02 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np

from mtpy import MTData
from mtpy_data import PROFILE_LIST
from mtpy.modeling.simpeg.data import Simpeg2DData
from mtpy.modeling.simpeg.recipes.inversion_2d import Simpeg2D

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
from pymatsolver import Pardiso

# =============================================================================


class TestSimpeg2DRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.md = MTData()
        self.md.add_station(
            [fn for fn in PROFILE_LIST if fn.name.startswith("16")]
        )
        # australian epsg
        self.md.utm_epsg = 4462

        # extract profile
        self.profile = self.md.get_profile(
            149.15, -22.3257, 149.20, -22.3257, 1000
        )
        # interpolate onto a common period range
        self.new_periods = np.logspace(-3, 0, 4)
        self.profile.interpolate(
            self.new_periods, inplace=True, bounds_error=False
        )
        self.profile.compute_model_errors()

        self.mt_df = self.profile.to_dataframe()

        self.simpeg_inversion = Simpeg2D(self.mt_df)

    def test_active_map(self):
        self.assertEqual(
            self.simpeg_inversion.active_map.nP,
            self.simpeg_inversion.quad_tree.number_of_active_cells,
        )

    def test_exponent_map(self):
        self.assertEqual(
            self.simpeg_inversion.quad_tree.active_cell_index.size,
            self.simpeg_inversion.exponent_map.nP,
        )

    def test_conductivity_map(self):
        self.assertEqual(
            self.simpeg_inversion.conductivity_map.nP,
            self.simpeg_inversion.quad_tree.number_of_active_cells,
        )

    def test_tm_simulation(self):
        self.assertIsInstance(
            self.simpeg_inversion.tm_simulation,
            nsem.simulation.Simulation2DElectricField,
        )

    def test_te_simulation(self):
        self.assertIsInstance(
            self.simpeg_inversion.te_simulation,
            nsem.simulation.Simulation2DMagneticField,
        )

    def test_te_data_misfit(self):
        self.assertIsInstance(
            self.simpeg_inversion.te_data_misfit, data_misfit.L2DataMisfit
        )

    def test_tm_data_misfit(self):
        self.assertIsInstance(
            self.simpeg_inversion.tm_data_misfit, data_misfit.L2DataMisfit
        )

    def test_reference_model(self):
        self.assertTrue(
            np.allclose(
                self.simpeg_inversion.reference_model,
                np.log(self.simpeg_inversion.initial_conductivity),
            )
        )

    def test_regularization(self):
        self.assertIsInstance(
            self.simpeg_inversion.regularization, regularization.Sparse
        )

    def test_optimization(self):
        self.assertIsInstance(
            self.simpeg_inversion.optimization, optimization.InexactGaussNewton
        )

    def test_inverse_problem(self):
        self.assertIsInstance(
            self.simpeg_inversion.inverse_problem,
            inverse_problem.BaseInvProblem,
        )

    def test_starting_beta(self):
        with self.subTest("Is Instance"):
            self.assertIsInstance(
                self.simpeg_inversion.inverse_problem,
                inverse_problem.BaseInvProblem,
            )
        with self.subTest("beta value"):
            self.assertEqual(
                self.simpeg_inversion.starting_beta.beta0_ratio,
                self.simpeg_inversion.beta_starting_ratio,
            )

    def test_beta_schedule(self):
        self.assertIsInstance(
            self.simpeg_inversion.beta_schedule, directives.BetaSchedule
        )

    def target_misfit(self):
        with self.subTest("Is Instance"):
            self.assertIsInstance(
                self.simpeg_inversion.target_misfit, directives.TargetMisfit
            )
        with self.subTest("chi factor"):
            self.assertEqual(
                self.simpeg_inversion.target_misfit._chifact,
                self.simpeg_inversion.target_misfit_chi_factor,
            )

    def test_directives(self):
        self.assertEqual(4, len(self.simpeg_inversion.directives))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
