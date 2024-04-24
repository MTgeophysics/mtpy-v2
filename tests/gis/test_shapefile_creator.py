# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:06:57 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mtpy import MTData
from mtpy.gis.shapefile_creator import ShapefileCreator

from mtpy_data import FWD_CONDUCTIVE_CUBE_GRID_LIST

# =============================================================================

has_data = True
if len(FWD_CONDUCTIVE_CUBE_GRID_LIST) == 0:
    has_data = False


@unittest.skipIf(has_data == False, "mtpy_data is not installed properly.")
class TestShapefileCreator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.md = MTData()
        self.md.add_station(FWD_CONDUCTIVE_CUBE_GRID_LIST, survey="test")
        self.md.utm_epsg = 32754

        self.mt_df = self.md.to_mt_dataframe()

        self.save_dir = Path().cwd().joinpath("shp_test")

        self.sc = ShapefileCreator(
            self.mt_df, self.md.utm_epsg, save_dir=self.save_dir
        )

    def test_mt_dataframe(self):
        self.assertEqual(self.mt_df, self.sc.mt_dataframe)

    def test_x_key(self):
        self.assertEqual(self.sc.x_key, "longitude")

    def test_y_key(self):
        self.assertEqual(self.sc.y_key, "latitude")

    def test_estimate_ellipse_size(self):
        self.assertAlmostEqual(0.00526422, self.sc.estimate_ellipse_size())

    def test_estimate_arrow_size(self):
        self.assertAlmostEqual(0.00692118, self.sc.estimate_arrow_size())

    def test_create_pt_shp(self):
        self.sc.ellipse_size = self.sc.estimate_ellipse_size()
        shp_fn = self.sc._create_phase_tensor_shp(1)
        self.assertEqual(
            shp_fn,
            self.save_dir.joinpath(
                f"Phase_Tensor_EPSG_{self.md.utm_epsg}_Period_1s.shp"
            ),
        )

    def test_create_tip_real_shp(self):
        self.sc.arrow_size = self.sc.estimate_arrow_size()
        shp_fn = self.sc._create_tipper_real_shp(1)
        self.assertEqual(
            shp_fn,
            self.save_dir.joinpath(
                f"Tipper_Real_EPSG_{self.md.utm_epsg}_Period_1s.shp"
            ),
        )

    def test_create_tip_imag_shp(self):
        self.sc.arrow_size = self.sc.estimate_arrow_size()
        shp_fn = self.sc._create_tipper_imag_shp(1)
        self.assertEqual(
            shp_fn,
            self.save_dir.joinpath(
                f"Tipper_Imag_EPSG_{self.md.utm_epsg}_Period_1s.shp"
            ),
        )

    @classmethod
    def tearDownClass(self):
        for fn in self.save_dir.glob("*"):
            fn.unlink()
        self.save_dir.rmdir()


# =============================================================================
#
# =============================================================================
if __name__ in "__main__":
    unittest.main()
